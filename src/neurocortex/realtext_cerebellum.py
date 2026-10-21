"""小脳の合成勾配で、皮質の遠いフィードバックがまれにしか届かない下の層を学ばせる（12.6.257〜258節）。

大脳-小脳ループの計算理論（Boven et al. 2023 Nat Commun「cerebro-cerebellar networks facilitate learning through
feedback decoupling」; Jaderberg et al. 2017 の decoupled neural interface）では、小脳は皮質の活動から、皮質に
遅れて・まれにしか届かない誤差のフィードバックを予測し、それを前倒しで与えて皮質の学習を速める。小脳はその予測を、
本物のフィードバックが届いたときの食い違い（登上線維の誤差）で学ぶ。

ここでは 12.6.213節の基盤（Europarl の実テキスト、文脈ごとのヘッドを持つ因果的な文字 Transformer）を2つに分ける。
- 下の皮質: 文字と位置の埋め込み＋自己注意の1層目。出力 h1 [B, T, d]。SGD で学ぶ（更新の大きさが教える信号の
  大きさに比例する。Adam は弱い予測も全幅の更新に正規化してしまい、12.6.257節の予備実行で発散した）。
- 上の皮質: 2層目以降・正規化・中間層・文脈ごとの読み出し。出力層に近く、毎ステップ自分の誤差の勾配で学ぶ（Adam）。
下の皮質に届く本物の誤差の勾配 g = ∂L/∂h1 は、feedback_every ステップに1度だけ届く（遠いフィードバックがまれ）。

小脳（Marr 1969; Albus 1971）: 苔状線維の入力は、h1 の各位置（パラメータのない LayerNorm を通す）と、その位置の
結果（次の文字。感覚の入力として苔状線維で届く。Jaderberg et al. の条件つき DNI と同じ）。顆粒細胞は固定の乱数の
拡大（granule 個、ReLU。閾値で約 25% が発火）、プルキンエ細胞は線形の読み出し（0 で初期化）で、位置ごとの
トークンあたりの勾配 ĝ（g × B·T）を出す。本物の g が届いたステップで、登上線維の誤差 g·B·T − ĝ で読み出しを
正規化 LMS（デルタ則。Marr–Albus–Ito）で学ぶ。皮質へ送る前に、ĝ から h1 の方向の成分を除く（tangent。本物の g は
ほぼ h1 に直交し、除かないと活動の大きさが暴走した）。

条件（cond）:
- bp: 下の皮質にも毎ステップ本物の勾配が届く（上限）。
- frozen: 下の皮質は一度も学ばない（フィードバックなし）。
- sparse: 本物の勾配が届いたステップでだけ下の皮質が学ぶ（小脳なし）。
- sparse10: sparse の学習率を feedback_every 倍にする（届く回数の少なさを歩幅で補う参照）。
- cb: sparse に加え、届かないステップでは小脳の予測 ĝ / (B·T) で下の皮質が学ぶ。
- cb_shuf: cb と同じだが、小脳の皮質の入力 h1 をバッチの中で別の断片のものに入れ替える（学習・予測とも。結果の
  入力は本来のまま）。小脳は皮質の状態に固有の予測ができない（状態に固有の予測の対照）。
- cb_state: cb と同じだが、結果（次の文字）の入力を与えない（皮質の状態だけからの予測。Boven et al. の小脳の入力）。

学習は言語の組の全言語を毎ステップ混ぜる（言語の番号は与える）。評価はテスト文の固定の断片の次の文字の CE（nats）。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .context_realtext import LangData, TransformerContextLM, _passages
from .realtext_forgetting import eval_segments, measure

CONDS = ("bp", "frozen", "sparse", "sparse10", "cb", "cb_shuf", "cb_state")


@dataclass
class CBConfig:
    cond: str = "cb"
    steps: int = 3000
    batch: int = 64
    seq_len: int = 32
    lr: float = 3e-3  # 上の皮質（Adam）
    lower_lr: float = 1.0  # 下の皮質（SGD。更新の大きさが教える信号の大きさに比例する）
    d: int = 64
    layers: int = 2
    heads: int = 4
    hidden: int = 128
    split: int = 1  # 下の皮質に入る自己注意の層の数
    feedback_every: int = 10  # 本物の勾配が下の皮質に届く間隔（ステップ）
    granule: int = 1024  # 顆粒細胞の数
    granule_active: float = 0.25  # 顆粒細胞の発火の割合（閾値の目安）
    cb_mu: float = 0.5  # プルキンエ細胞の正規化 LMS の歩幅
    eval_every: int = 300
    eval_segments: int = 400


class Cerebellum(nn.Module):
    """苔状線維 → 顆粒細胞（固定の乱数の拡大、ReLU）→ プルキンエ細胞（線形、学習）。入力 [..., d] → 出力 [..., d]。"""

    def __init__(self, d: int, granule: int, active: float, generator: torch.Generator, vocab: int = 0):
        super().__init__()
        self.register_buffer("mf", torch.randn(d, granule, generator=generator) / d ** 0.5)
        # 結果（次の文字）の苔状線維の入力: 文字ごとの固定の乱数の符号（vocab = 0 なら使わない）
        self.register_buffer("mf_y", torch.randn(max(vocab, 1), granule, generator=generator))
        # 正規化した入力の射影はおよそ標準正規分布なので、上位 active の分位点を閾値にする
        thr = float(torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(1.0 - active)))
        self.register_buffer("thr", torch.tensor(thr))
        self.pc = nn.Linear(granule, d)
        nn.init.zeros_(self.pc.weight)
        nn.init.zeros_(self.pc.bias)

    def granule_code(self, h: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        u = nn.functional.layer_norm(h, h.shape[-1:]) @ self.mf
        if y is not None:  # 皮質の状態と結果を半分ずつ（和の分散が 1 になるように）
            u = (u + self.mf_y[y]) / 2 ** 0.5
        return torch.relu(u - self.thr)

    def forward(self, h: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        return self.pc(self.granule_code(h, y))

    @torch.no_grad()
    def lms(self, h: torch.Tensor, y: torch.Tensor | None, target: torch.Tensor, mu: float) -> torch.Tensor:
        """登上線維の誤差 e = target − 予測 で、プルキンエ細胞の読み出しを正規化 LMS（デルタ則）で学ぶ。
        W ← W + μ·mean(e xᵀ) / mean(‖x‖² + 1)（バイアスは入力 1 の重み）。0 < μ < 2 で安定。学ぶ前の予測を返す。"""
        x = self.granule_code(h, y).reshape(-1, self.mf.shape[1])
        pred = self.pc(x)
        e = target.reshape(-1, target.shape[-1]) - pred
        step = mu / float((x.pow(2).sum(-1) + 1).mean())
        self.pc.weight += step * e.T @ x / x.shape[0]
        self.pc.bias += step * e.mean(0)
        return pred.reshape(target.shape)


def lower(model: TransformerContextLM, x: torch.Tensor, split: int) -> torch.Tensor:
    """下の皮質: 埋め込み＋最初の split 層 → h1 [B, T, d]。"""
    t = x.shape[1]
    h = model.tok(x) + model.pos(torch.arange(t))[None]
    for layer in model.body.layers[:split]:
        h = layer(h, src_mask=model.mask[:t, :t], is_causal=True)
    return h


def upper_logp(model: TransformerContextLM, h: torch.Tensor, c: torch.Tensor, split: int) -> torch.Tensor:
    """上の皮質: 残りの層・正規化・中間層・文脈の読み出し → 対数確率 [B, T, V]。"""
    t = h.shape[1]
    for layer in model.body.layers[split:]:
        h = layer(h, src_mask=model.mask[:t, :t], is_causal=True)
    z = torch.tanh(model.mid(model.norm(h)) + model.ctx(c)[:, None, :])
    return torch.log_softmax(torch.einsum("bth,bhv->btv", z, model.head_w[c]) + model.head_b[c][:, None], -1)


def lower_params(model: TransformerContextLM, split: int) -> list[nn.Parameter]:
    ps = [model.tok.weight, model.pos.weight]
    for layer in model.body.layers[:split]:
        ps += list(layer.parameters())
    return ps


def tangent(g: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """g から h の方向（各位置の活動の大きさを変える成分）を除く。本物の g はほぼ h に直交する（12.6.257節の予備実行で
    位置ごとの |cos(g, h)| 0.05）ので、小脳の予測にも同じ制約をかける（活動の大きさの暴走を防ぐ）。"""
    u = nn.functional.normalize(h, dim=-1)
    return g - (g * u).sum(-1, keepdim=True) * u


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(nn.functional.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).squeeze())


def train_cerebellum(data: list[LangData], vocab: int, cfg: CBConfig, generator: torch.Generator,
                     eval_generator: torch.Generator) -> dict:
    """全言語を混ぜて cfg.steps 学び、eval_every ごとにテストの CE・正確度を測る。

    返り値: curve（[ステップ, 言語ごとの CE の平均, 正確度の平均]）、final_ce（言語ごと）、auc（曲線の CE の平均）、
    小脳の予測の質（本物の勾配が届いたステップで、学ぶ前の ĝ と本物の g の余弦。全体と、学習の後半の平均）。"""
    if cfg.cond not in CONDS:
        raise ValueError(f"cond は {CONDS}")
    n_l, t_len = len(data), cfg.seq_len
    flat = [_passages(d.train, 400).reshape(-1) for d in data]
    segs = eval_segments(data, cfg.eval_segments, t_len, eval_generator)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(torch.randint(0, 2 ** 31 - 1, (1,), generator=generator)))
        model = TransformerContextLM(vocab, n_l, t_len, cfg.d, cfg.layers, cfg.heads, hidden=cfg.hidden)
    cb = Cerebellum(cfg.d, cfg.granule, cfg.granule_active,
                    torch.Generator().manual_seed(int(torch.randint(0, 2 ** 31 - 1, (1,), generator=generator))), vocab)
    low = lower_params(model, cfg.split)
    low_ids = {id(p) for p in low}
    opt = torch.optim.Adam([p for p in model.parameters() if id(p) not in low_ids], lr=cfg.lr)
    low_lr = cfg.lower_lr * (cfg.feedback_every if cfg.cond == "sparse10" else 1)
    low_opt = torch.optim.SGD(low, lr=low_lr)
    use_cb = cfg.cond in ("cb", "cb_shuf", "cb_state")

    def batch() -> tuple[torch.Tensor, torch.Tensor]:
        c = torch.randint(0, n_l, (cfg.batch,), generator=generator)
        seg = torch.empty(cfg.batch, t_len + 1, dtype=torch.long)
        for j in range(n_l):
            sel = c == j
            k = int(sel.sum())
            if k:
                s = torch.randint(0, flat[j].shape[0] - t_len - 1, (k,), generator=generator)
                seg[sel] = flat[j][s[:, None] + torch.arange(t_len + 1)[None]]
        return seg, c

    def evaluate(step: int) -> None:
        model.eval()
        a, e = measure(model, segs)
        curve.append([step, sum(e) / n_l, sum(a) / n_l])
        model.train()

    curve: list = []
    cos_all: list[tuple[int, float]] = []
    losses: list[float] = []
    lower_updates = 0
    evaluate(0)
    for it in range(1, cfg.steps + 1):
        seg, c = batch()
        n_tok = seg.shape[0] * t_len
        feedback = it % cfg.feedback_every == 0
        perm = torch.randperm(cfg.batch, generator=generator) if cfg.cond == "cb_shuf" else None
        opt.zero_grad()
        low_opt.zero_grad()
        h1 = lower(model, seg[:, :-1], cfg.split)
        h1d = h1.detach().requires_grad_(True)
        lp = upper_logp(model, h1d, c, cfg.split)
        loss = nn.functional.nll_loss(lp.reshape(-1, lp.shape[-1]), seg[:, 1:].reshape(-1))
        loss.backward()  # 上の皮質の勾配と、本物の g = ∂L/∂h1（h1d.grad）
        g = h1d.grad
        cb_in = h1.detach() if perm is None else h1.detach()[perm]
        cb_y = None if cfg.cond == "cb_state" else seg[:, 1:]
        if cfg.cond == "bp" or (feedback and cfg.cond != "frozen"):
            h1.backward(g)
            lower_updates += 1
        elif use_cb:
            with torch.no_grad():
                g_hat = tangent(cb(cb_in, cb_y), h1.detach()) / n_tok
            h1.backward(g_hat)
            lower_updates += 1
        opt.step()
        low_opt.step()
        if use_cb and feedback:  # 登上線維の誤差で小脳の読み出しを学ぶ
            pred = cb.lms(cb_in, cb_y, g * n_tok, cfg.cb_mu)
            cos_all.append((it, _cos(pred, g)))
        losses.append(float(loss.detach()))
        if it % cfg.eval_every == 0 or it == cfg.steps:
            evaluate(it)
    model.eval()
    _, final_ce = measure(model, segs)
    half = [v for i, v in cos_all if i > cfg.steps // 2]
    return {"curve": curve, "final_ce": final_ce, "final_ce_mean": sum(final_ce) / n_l,
            "auc": sum(r[1] for r in curve[1:]) / max(1, len(curve) - 1),
            "final_loss": sum(losses[-100:]) / min(100, len(losses)), "lower_updates": lower_updates,
            "cb_cos": sum(v for _, v in cos_all) / len(cos_all) if cos_all else None,
            "cb_cos_late": sum(half) / len(half) if half else None}

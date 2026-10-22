"""小脳の合成勾配で、時間方向の逆伝播を短く切った再帰型の皮質の学習を助ける（12.6.259〜260節）。

12.6.257〜258節（層の間のフィードバックの切り離し）の否定的結果の残る課題 (1)(3) を、Boven et al. 2023（Nat Commun
「cerebro-cerebellar networks facilitate learning through feedback decoupling」）の主設定に近い形で検証する:
再帰型の皮質（GRU）を、時間方向の逆伝播を K 文字ごとに切って学ぶ（truncated BPTT）。窓の終わりの状態 h_K には、
窓より先の誤差の勾配が届かない。小脳は窓の終わりの皮質の状態から、その先の誤差の勾配 ĝ(h_K) を予測し、それを窓の
終わりに注入する（Jaderberg et al. 2017 の再帰型の DNI）。

小脳の学習（登上線維の誤差）: 次の窓を学ぶとき、窓の始まりの状態 s（= 前の窓の h_K）への勾配
    target = ∂(次の窓の誤差の和)/∂s + γ (∂h'_K/∂s)ᵀ ĝ(h'_K)
（次の窓の本物の勾配と、その先の小脳の予測のブートストラップ。TD 学習と同じ形）が届く。小脳はこれと前の予測の
食い違いで、プルキンエ細胞の読み出しを正規化 LMS で学ぶ（12.6.257節の Cerebellum を再利用。入力は皮質の状態だけ）。

信頼度の利得（12.6.258節の残る課題 (3)。偏りのない推定）: target と予測 ĝ の最小二乗の係数 a·b/‖b‖² を、1回ごとに
0〜1 に切ってから平均すると上に偏る（12.6.258節で cb_state でも 0.22）。ここでは a·b と ‖b‖² をそれぞれ指数移動平均
してから比をとり、0〜1 に切る（gain_unbiased）。比較のため、切ってから平均する推定（gain_biased）も記録する。

条件（cond）:
- long: 窓の誤差の勾配を過去 back 文字まで逆伝播する（TBPTT(K, back)。切断の害のない参照・上限）。
- tbptt: 窓の中だけで逆伝播する（切断。小脳なし）。
- cb: tbptt に加え、窓の終わりに小脳の予測 ĝ(h_K) を注入する。
- cb_shuf: cb と同じだが、小脳の入力（学習・予測とも）をバッチの中で別の流れの状態に入れ替える（状態に固有の予測の
  対照）。
- cb_gain: cb の注入を偏りのない信頼度の利得 gain_unbiased で重みづける。

言語の組の全言語を混ぜる（各流れは1つの言語の学習文を連続して読み、言語の番号を入力に加える）。評価はテスト文の
連続した流れの次の文字の CE（nats。最初の warm 文字は状態を温めるだけで数えない）。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .context_realtext import LangData
from .realtext_cerebellum import Cerebellum

CONDS = ("long", "tbptt", "cb", "cb_shuf", "cb_gain")


@dataclass
class TBPTTConfig:
    cond: str = "cb"
    steps: int = 5000  # 更新の回数（1回に各流れが K 文字進む）
    batch: int = 64  # 並行して読む流れの数
    k: int = 4  # 逆伝播の窓（切断の長さ）
    back: int = 64  # long の逆伝播の長さ（文字）。k の倍数
    d: int = 64  # 文字の埋め込み
    hidden: int = 128  # 皮質の状態の次元
    lr: float = 3e-3
    clip: float = 1.0  # 勾配のノルムの上限（全条件で同じ）
    granule: int = 1024
    granule_active: float = 0.25
    cb_mu: float = 0.5
    gamma: float = 1.0  # ブートストラップの割引
    gain_ema: float = 0.9
    eval_every: int = 500
    eval_streams: int = 16  # 言語ごとのテストの流れの数
    eval_len: int = 512
    eval_warm: int = 64


class RNNCortex(nn.Module):
    """文字の埋め込み＋言語の埋め込み → GRU（1層）→ 線形の読み出し。"""

    def __init__(self, vocab: int, n_lang: int, d: int, hidden: int):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.lang = nn.Embedding(n_lang, d)
        self.rnn = nn.GRU(d, hidden, batch_first=True)
        self.out = nn.Linear(hidden, vocab)

    def run(self, x: torch.Tensor, c: torch.Tensor, h0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x [B, T]、c [B]、h0 [B, H] → 対数確率 [B, T, V] と各位置の状態 [B, T, H]（最後が窓の終わりの状態）。"""
        hs, _ = self.rnn(self.tok(x) + self.lang(c)[:, None], h0[None])
        return torch.log_softmax(self.out(hs), -1), hs


def project_gain(a: torch.Tensor, b: torch.Tensor) -> float:
    """1回の最小二乗の係数 a·b/‖b‖²（‖b‖ = 0 なら 0）。"""
    bb = float((b * b).sum())
    return float((a * b).sum()) / bb if bb > 0 else 0.0


class GainEstimator:
    """小脳の予測 b を本物の target a に合わせる係数の推定。biased は1回ごとに 0〜1 に切ってから平均（12.6.258節の
    修正1）、unbiased は a·b と ‖b‖² を別々に平均してから比をとり 0〜1 に切る。"""

    def __init__(self, ema: float):
        self.ema = ema
        self.biased = 0.0
        self.ab = 0.0
        self.bb = 0.0
        self.n = 0

    def update(self, a: torch.Tensor, b: torch.Tensor) -> None:
        ab, bb = float((a * b).sum()), float((b * b).sum())
        if bb <= 0:
            return
        e = self.ema
        self.biased = e * self.biased + (1 - e) * min(1.0, max(0.0, ab / bb))
        self.ab = e * self.ab + (1 - e) * ab
        self.bb = e * self.bb + (1 - e) * bb
        self.n += 1

    @property
    def unbiased(self) -> float:
        return min(1.0, max(0.0, self.ab / self.bb)) if self.bb > 0 else 0.0


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(nn.functional.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).squeeze())


def eval_streams(data: list[LangData], cfg: TBPTTConfig, generator: torch.Generator) -> list[torch.Tensor]:
    """言語ごとのテスト文の連続した流れ [eval_streams, eval_len + 1]。"""
    out = []
    for d in data:
        s = torch.randint(0, d.test.shape[0] - cfg.eval_len - 1, (cfg.eval_streams,), generator=generator)
        out.append(d.test[s[:, None] + torch.arange(cfg.eval_len + 1)[None]])
    return out


@torch.no_grad()
def measure_rnn(model: RNNCortex, streams: list[torch.Tensor], warm: int) -> list[float]:
    """各言語の流れを 0 の状態から読み、warm 文字以降の次の文字の CE（nats）。"""
    ce = []
    h = model.rnn.hidden_size
    for j, s in enumerate(streams):
        lp, _ = model.run(s[:, :-1], torch.full((s.shape[0],), j), torch.zeros(s.shape[0], h))
        y = s[:, 1:]
        ce.append(float(nn.functional.nll_loss(lp[:, warm:].reshape(-1, lp.shape[-1]), y[:, warm:].reshape(-1))))
    return ce


def train_tbptt(data: list[LangData], vocab: int, cfg: TBPTTConfig, generator: torch.Generator,
                eval_generator: torch.Generator) -> dict:
    """全言語の流れを並行して cfg.steps 回（各 K 文字）学び、eval_every ごとにテストの CE を測る。"""
    if cfg.cond not in CONDS:
        raise ValueError(f"cond は {CONDS}")
    if cfg.back % cfg.k:
        raise ValueError("back は k の倍数")
    n_l, k, bsz = len(data), cfg.k, cfg.batch
    streams = eval_streams(data, cfg, eval_generator)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(torch.randint(0, 2 ** 31 - 1, (1,), generator=generator)))
        model = RNNCortex(vocab, n_l, cfg.d, cfg.hidden)
    cb = Cerebellum(cfg.hidden, cfg.granule, cfg.granule_active,
                    torch.Generator().manual_seed(int(torch.randint(0, 2 ** 31 - 1, (1,), generator=generator))))
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    params = list(model.parameters())
    use_cb = cfg.cond.startswith("cb")
    # 流れ i は言語 i mod n_l の学習文を、無作為な位置から連続して読む（終わりに来たら先頭へ）
    lang = torch.arange(bsz) % n_l
    flat = [d.train for d in data]
    pos = torch.tensor([int(torch.randint(0, flat[int(j)].shape[0], (1,), generator=generator)) for j in lang])
    lens = torch.tensor([flat[int(j)].shape[0] for j in lang])

    def window() -> torch.Tensor:
        """各流れの次の k + 1 文字 [B, k + 1]（最後の文字は次の窓の最初の入力と同じ）。"""
        idx = (pos[:, None] + torch.arange(k + 1)[None]) % lens[:, None]
        return torch.stack([flat[int(lang[i])][idx[i]] for i in range(bsz)])

    n_tok = bsz * k
    h = torch.zeros(bsz, cfg.hidden)
    hist_x: list[torch.Tensor] = []  # long: 過去の窓の入力
    hist_h: list[torch.Tensor] = []  # long: 過去の窓の始まりの状態
    prev: dict | None = None  # 前の窓の終わりの状態と、そこでの小脳の入力・予測
    perm = None
    gain = GainEstimator(cfg.gain_ema)
    curve: list = []
    cos_true: list[tuple[int, float]] = []
    cos_boot: list[tuple[int, float]] = []
    gains: list[tuple[float, float]] = []
    losses: list[float] = []
    sg_norm: list[float] = []

    def evaluate(step: int) -> None:
        model.eval()
        e = measure_rnn(model, streams, cfg.eval_warm)
        curve.append([step, sum(e) / n_l])
        model.train()

    evaluate(0)
    for it in range(1, cfg.steps + 1):
        seg = window()
        pos = (pos + k) % lens
        x, y = seg[:, :-1], seg[:, 1:]
        opt.zero_grad()
        if cfg.cond == "long":
            hist_x.append(x)
            hist_h.append(h)
            if len(hist_x) > cfg.back // k:
                hist_x.pop(0)
                hist_h.pop(0)
            lp, hs = model.run(torch.cat(hist_x, 1), lang, hist_h[0])
            lp = lp[:, -k:]
            loss = nn.functional.nll_loss(lp.reshape(-1, lp.shape[-1]), y.reshape(-1), reduction="sum")
            (loss / n_tok).backward()
            # 窓の始まりの状態を今のパラメータで計算し直したもので置き換える（次の逆伝播の始まり）
            for i in range(1, len(hist_h)):
                hist_h[i] = hs[:, i * k - 1].detach()
            h_end = hs[:, -1].detach()
        else:
            s = h.requires_grad_(use_cb and prev is not None)
            lp, hs = model.run(x, lang, s)
            loss = nn.functional.nll_loss(lp.reshape(-1, lp.shape[-1]), y.reshape(-1), reduction="sum")
            h_end = hs[:, -1]
            if use_cb:
                perm = torch.randperm(bsz, generator=generator) if cfg.cond == "cb_shuf" else None
                cb_in = h_end.detach() if perm is None else h_end.detach()[perm]
                with torch.no_grad():
                    g_hat = cb(cb_in)  # cb_shuf では別の流れの状態からの予測を、流れ i の状態に注入する
                sg_norm.append(float(g_hat.norm(dim=-1).mean()))
                inject = (h_end * g_hat).sum()
                wrt = params + ([s] if prev is not None else [])
                g_loss = torch.autograd.grad(loss, wrt, retain_graph=True, allow_unused=True)
                g_sg = torch.autograd.grad(inject, wrt, allow_unused=True)
                w = gain.unbiased if cfg.cond == "cb_gain" else 1.0
                for p, a, b in zip(params, g_loss, g_sg):
                    gp = torch.zeros_like(p) if a is None else a.clone()
                    if b is not None:
                        gp += w * b
                    p.grad = gp / n_tok
                if prev is not None:  # 登上線維: 前の窓の終わりの状態への本物＋ブートストラップの勾配
                    true = g_loss[-1]
                    boot = true + cfg.gamma * g_sg[-1]
                    gain.update(boot, prev["g_hat"])
                    gains.append((gain.biased, gain.unbiased))
                    cos_true.append((it, _cos(prev["g_hat"], true)))
                    cos_boot.append((it, _cos(prev["g_hat"], boot)))
                    cb.lms(prev["cb_in"], None, boot, cfg.cb_mu)
                prev = {"cb_in": cb_in, "g_hat": g_hat}
            else:
                (loss / n_tok).backward()
            h_end = h_end.detach()
        nn.utils.clip_grad_norm_(params, cfg.clip)
        opt.step()
        h = h_end
        losses.append(float(loss.detach()) / n_tok)
        if it % cfg.eval_every == 0 or it == cfg.steps:
            evaluate(it)
    model.eval()
    final_ce = measure_rnn(model, streams, cfg.eval_warm)
    half = cfg.steps // 2

    def avg(v: list[tuple[int, float]], late: bool = False) -> float | None:
        sel = [c for i, c in v if not late or i > half]
        return sum(sel) / len(sel) if sel else None

    return {"curve": curve, "final_ce": final_ce, "final_ce_mean": sum(final_ce) / n_l,
            "auc": sum(r[1] for r in curve[1:]) / max(1, len(curve) - 1),
            "final_loss": sum(losses[-100:]) / min(100, len(losses)),
            "cb_cos_true": avg(cos_true), "cb_cos_true_late": avg(cos_true, True),
            "cb_cos_boot": avg(cos_boot), "cb_cos_boot_late": avg(cos_boot, True),
            "gain_biased": sum(g[0] for g in gains) / len(gains) if gains else None,
            "gain_unbiased": sum(g[1] for g in gains) / len(gains) if gains else None,
            "sg_norm": sum(sg_norm) / len(sg_norm) if sg_norm else None}

"""文字 LM の文脈推定への、能動的な手がかりの選択と海馬のバッファからの候補の移植（12.6.173〜174節）。

パリティ課題（12.6.143〜152節）で得た2つの機構を、規則の異なる複数の合成言語の文字 LM に移す。
- 合成言語: 文字 V 種の2次マルコフ連鎖。言語0（基本）の遷移をディリクレ分布から引き、他の言語は状態（直前2文字）
  の一部 diff_frac だけ遷移を引き直す。言語どうしは大半の状態で同じ次の文字の分布を持ち、一部の状態だけで食い違う
  （パリティ課題の反転領域にあたる）。
- モデル: 直前 k 文字の窓から次の文字を予測する小さな神経言語モデル（Bengio et al. 2003 型）。埋め込みと隠れ層は
  共有、文脈ごとに隠れ層へのバイアス（文脈の埋め込み）と出力ヘッドを持つ。文脈なしの対照は文脈1個の同じモデル。
- 学習: joint（全言語を混ぜて学ぶ。上書きなし）と seq（言語を順に学び、過去の言語は海馬のバッファの窓だけを
  リプレイする。上書きあり）。
- テスト時の文脈推定: 未知の言語の話し手（情報源）に、窓（プロンプト）を見せて次の文字を1つ答えてもらうことを
  手がかりの観測とする。n 個の観測から文脈を選び、選んだ文脈のヘッドで同じ言語の新しい文の次の文字を予測する。
  規則は passive（ひと続きの文を先頭から n 文字読む。LM が文脈として受け取る形）、uniform（言語の文から独立に取った
  n 個の窓。12.6.143節の batch）、active（uniform の窓を先頭に含む P·n 個のラベルなしの窓から期待情報量が最大の
  ものを1個ずつ選ぶ）、uniform_replay（バッファの窓から無作為に n 個）、active_replay（バッファの
  窓から期待情報量で選ぶ。12.6.148・152節の gen_replay）、active_holdout（保存したがリプレイしなかった窓から。統制）。

12.6.143節と違い、能動的な選択も打ち切りなしで n 個すべてを観測する（観測の数を規則の間で揃えるため）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

EPS = 0.02  # 予測の平滑化 p̃ = (1 − ε)·p + ε/V（12.6.135節の確定値）
RULES = ("passive", "uniform", "active", "uniform_replay", "active_replay", "active_holdout")
# 修正1（12.6.173節）: 事後分布を手がかりにした、文脈ごとのバッファからの想起。active_replay_own は本来の文脈の
# バッファだけから取る診断用（本来の文脈を知っているので規則ではない）
FIX1_RULES = ("active_replay_cued", "active_holdout_cued", "active_replay_own")


@dataclass(frozen=True)
class LangFamily:
    """合成言語の族。n_langs 個の言語、文字 vocab 種、order 次のマルコフ連鎖。"""
    n_langs: int = 4
    vocab: int = 10
    order: int = 2
    diff_frac: float = 0.3  # 基本言語から遷移を引き直す状態の割合
    alpha: float = 0.3  # 遷移のディリクレ分布の集中度


def make_languages(fam: LangFamily, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """遷移確率 [言語, 状態, V] と、各言語が基本言語と食い違う状態のマスク [言語, 状態] を返す。"""
    n_states = fam.vocab ** fam.order
    def draw() -> torch.Tensor:
        # ガンマ分布の乱数は Generator を取れないので、generator から引いた種で大域の乱数を一時的に固定する
        seed = int(torch.randint(0, 2 ** 31 - 1, (1,), generator=generator))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            g = torch._standard_gamma(torch.full((n_states, fam.vocab), fam.alpha))
        return g / g.sum(-1, keepdim=True)

    base = draw()
    trans = [base]
    masks = [torch.zeros(n_states, dtype=torch.bool)]
    n_diff = round(fam.diff_frac * n_states)
    for _ in range(1, fam.n_langs):
        idx = torch.randperm(n_states, generator=generator)[:n_diff]
        t = base.clone()
        t[idx] = draw()[idx]
        m = torch.zeros(n_states, dtype=torch.bool)
        m[idx] = True
        trans.append(t)
        masks.append(m)
    return torch.stack(trans), torch.stack(masks)


def state_index(x: torch.Tensor, vocab: int, order: int) -> torch.Tensor:
    """窓 x [..., k] の最後の order 文字から状態の番号を作る。"""
    idx = torch.zeros(x.shape[:-1], dtype=torch.long)
    for i in range(order):
        idx = idx * vocab + x[..., x.shape[-1] - order + i]
    return idx


def sample_text(trans: torch.Tensor, fam: LangFamily, n_chains: int, length: int,
                generator: torch.Generator) -> torch.Tensor:
    """1つの言語（trans [状態, V]）から n_chains 本の文 [n_chains, length] を生成する（最初の 8 文字は捨てる）。"""
    burn = 8
    seq = torch.randint(0, fam.vocab, (n_chains, fam.order), generator=generator)
    out = [seq[:, i] for i in range(fam.order)]
    for _ in range(length + burn - fam.order):
        s = torch.zeros(n_chains, dtype=torch.long)
        for i in range(fam.order):
            s = s * fam.vocab + out[-fam.order + i]
        out.append(torch.multinomial(trans[s], 1, generator=generator).squeeze(1))
    return torch.stack(out[burn:], dim=1)


def windows(text: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """文 [n_chains, length] から全ての窓（直前 k 文字）と次の文字を取り出す。"""
    xs = text.unfold(1, k + 1, 1).reshape(-1, k + 1)
    return xs[:, :k].contiguous(), xs[:, k].contiguous()


class ContextCharLM(nn.Module):
    """直前 k 文字の窓から次の文字を予測する。埋め込み・隠れ層は共有、文脈ごとにバイアスと出力ヘッド。"""

    def __init__(self, vocab: int, k: int, n_ctx: int, emb: int = 16, hidden: int = 128):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.inp = nn.Linear(k * emb, hidden)
        self.ctx = nn.Embedding(n_ctx, hidden)
        nn.init.zeros_(self.ctx.weight)
        self.head_w = nn.Parameter(torch.randn(n_ctx, hidden, vocab) / math.sqrt(hidden))
        self.head_b = nn.Parameter(torch.zeros(n_ctx, vocab))

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        return self.inp(self.emb(x).flatten(1))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """窓 x [N, k] と文脈 c [N] の対数確率 [N, V]。"""
        h = torch.tanh(self.hidden(x) + self.ctx(c))
        return torch.log_softmax(torch.einsum("nh,nhv->nv", h, self.head_w[c]) + self.head_b[c], -1)

    def logp_all(self, x: torch.Tensor) -> torch.Tensor:
        """全文脈の対数確率 [N, 文脈, V]。"""
        h = torch.tanh(self.hidden(x)[:, None, :] + self.ctx.weight[None])
        return torch.log_softmax(torch.einsum("nch,chv->ncv", h, self.head_w) + self.head_b[None], -1)


class GainCharLM(nn.Module):
    """共有の表現に、文脈信号で乗算的なゲイン変調をかけるモデル（12.6.175節）。

    窓の埋め込み → 隠れ層1（tanh, 共有）→ 文脈ごとのゲイン g_c ⊙ h1 → 隠れ層2（tanh, 共有）→ 出力（共有）。
    g_c = 2·sigmoid(a_c) ∈ (0, 2) で、a_c = 0（ゲイン 1）から学習する。文脈ごとの独立したヘッドを持たず、文脈は
    共有の特徴のどれを強め・弱めるかだけを決める（前頭前野のトップダウンのバイアス信号による選択的なゲーティング:
    Miller & Cohen 2001、文脈による選択的な統合: Mante et al. 2013）。文脈なしの対照は文脈1個の同じモデル。"""

    def __init__(self, vocab: int, k: int, n_ctx: int, emb: int = 16, hidden: int = 128):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.inp = nn.Linear(k * emb, hidden)
        self.gain_logit = nn.Parameter(torch.zeros(n_ctx, hidden))
        self.mid = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, vocab)

    def gains(self) -> torch.Tensor:
        """文脈ごとのゲイン [文脈, hidden]。"""
        return 2 * torch.sigmoid(self.gain_logit)

    def _logp(self, h1: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(self.out(torch.tanh(self.mid(h1 * g))), -1)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """窓 x [N, k] と文脈 c [N] の対数確率 [N, V]。"""
        return self._logp(torch.tanh(self.inp(self.emb(x).flatten(1))), self.gains()[c])

    def logp_all(self, x: torch.Tensor) -> torch.Tensor:
        """全文脈の対数確率 [N, 文脈, V]。"""
        h1 = torch.tanh(self.inp(self.emb(x).flatten(1)))
        return self._logp(h1[:, None, :], self.gains()[None])


def batched_information(probs: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
    """各エピソードの事前分布のもとでの各候補の期待情報量（文脈と答えの相互情報量 / log V）。

    probs [E, 候補, 文脈, V]、prior [E, 文脈] → [E, 候補]。context_inference.cue_information を、エピソードごとに
    異なる事前分布で一度に計算する版（値は同じ）。"""
    ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum(-1)  # [E, M, C]
    mix = (prior[:, None, :, None] * probs).sum(2)  # [E, M, V]
    h_mix = -(mix * torch.log(mix.clamp_min(1e-12))).sum(-1)
    return ((h_mix - (prior[:, None, :] * ent).sum(-1)) / math.log(probs.shape[-1])).clamp_min(0.0)


def select_contexts(logp: torch.Tensor, answers: torch.Tensor, n_obs: int, active: bool,
                    source: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """手がかりから文脈を選ぶ。

    logp [E, M, 文脈, V] は各候補の窓での各文脈の対数確率、answers [E, M] は情報源の答え（観測するまで使わない）。
    active=False は先頭 n_obs 個をそのまま観測する（batch）。active=True は1個ずつ、まだ観測していない候補のうち
    現在の事後分布のもとで期待情報量が最大のものを選んで観測する。source [E, M]（候補の窓を保存した文脈）を与えると、
    期待情報量にその文脈の現在の事後確率を掛けて選ぶ（修正1。信じている文脈の記憶ほど想起されやすい）。返り値は選んだ文脈 [E] と観測した手がかりの
    一様事前のもとでの情報量の平均 [E]。"""
    e_n, m_n, c_n, v_n = logp.shape
    probs = (1 - EPS) * logp.exp() + EPS / v_n
    loglik = torch.log(probs.gather(-1, answers[:, :, None, None].expand(e_n, m_n, c_n, 1)).squeeze(-1))  # [E, M, C]
    info_uniform = batched_information(probs, torch.full((e_n, c_n), 1.0 / c_n))  # [E, M]
    if not active:
        post = loglik[:, :n_obs].sum(1)
        return post.argmax(-1), info_uniform[:, :n_obs].mean(-1)
    log_post = torch.zeros(e_n, c_n)
    seen = torch.zeros(e_n, m_n, dtype=torch.bool)
    rows = torch.arange(e_n)
    for _ in range(n_obs):
        prior = torch.softmax(log_post, -1)
        info = batched_information(probs, prior)
        if source is not None:
            info = info * prior.gather(1, source)
        info = info.masked_fill(seen, -1.0)
        pick = info.argmax(-1)
        seen[rows, pick] = True
        log_post = log_post + loglik[rows, pick]
    return log_post.argmax(-1), (info_uniform * seen).sum(-1) / n_obs


@dataclass
class TrainConfig:
    regime: str = "joint"  # joint / seq
    steps_per_lang: int = 1500
    batch: int = 64
    replay_batch: int = 32
    buffer: int = 200  # 文脈ごとのバッファの窓の数（12.6.156節の痕跡200個にあたる）
    lr: float = 3e-3
    k: int = 3
    hidden: int = 128
    arch: str = "head"  # head（文脈ごとのヘッド。12.6.173節）/ gain（共有の表現へのゲイン変調。12.6.175節）


def train_models(fam: LangFamily, trans: torch.Tensor, cfg: TrainConfig, generator: torch.Generator,
                 text_len: int = 400) -> dict:
    """文脈つきのモデルと文脈なしのモデルを同じバッチで学習する。

    seq では言語 l の学習の終わりに、その学習文から buffer 個の窓をバッファに、別の buffer 個を保留サンプルに保存し、
    以後の各ステップで過去の言語のバッファから replay_batch 個をリプレイする。joint では学習文から同じ数の窓を
    保存するが、特別にリプレイはしない（全言語を毎ステップ混ぜて学ぶ）。"""
    if cfg.regime not in ("joint", "seq"):
        raise ValueError("regime は joint・seq")
    n_l = fam.n_langs
    n_chains = max(1, cfg.steps_per_lang * cfg.batch * (n_l if cfg.regime == "joint" else 1) // (n_l * text_len))
    train_x, train_y = [], []
    for j in range(n_l):
        x, y = windows(sample_text(trans[j], fam, n_chains, text_len, generator), cfg.k)
        train_x.append(x)
        train_y.append(y)
    with torch.random.fork_rng(devices=[]):  # 初期化の種も generator から決める
        torch.manual_seed(int(torch.randint(0, 2 ** 31 - 1, (1,), generator=generator)))
        arch = {"head": ContextCharLM, "gain": GainCharLM}[cfg.arch]
        model = arch(fam.vocab, cfg.k, n_l, hidden=cfg.hidden)
        plain = arch(fam.vocab, cfg.k, 1, hidden=cfg.hidden)
    opt = torch.optim.Adam(list(model.parameters()) + list(plain.parameters()), lr=cfg.lr)
    buf_x, buf_y, hold_x, hold_y = [None] * n_l, [None] * n_l, [None] * n_l, [None] * n_l

    def store(j: int) -> None:
        perm = torch.randperm(train_x[j].shape[0], generator=generator)[:2 * cfg.buffer]
        buf_x[j], buf_y[j] = train_x[j][perm[:cfg.buffer]], train_y[j][perm[:cfg.buffer]]
        hold_x[j], hold_y[j] = train_x[j][perm[cfg.buffer:]], train_y[j][perm[cfg.buffer:]]

    def step(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> float:
        opt.zero_grad()
        loss = nn.functional.nll_loss(model(x, c), y)
        loss_plain = nn.functional.nll_loss(plain(x, torch.zeros_like(c)), y)
        (loss + loss_plain).backward()
        opt.step()
        return float(loss.detach())

    losses = []
    if cfg.regime == "joint":
        for j in range(n_l):
            store(j)
        for _ in range(cfg.steps_per_lang * n_l):
            c = torch.randint(0, n_l, (cfg.batch,), generator=generator)
            x = torch.empty(cfg.batch, cfg.k, dtype=torch.long)
            y = torch.empty(cfg.batch, dtype=torch.long)
            for j in range(n_l):
                sel = c == j
                idx = torch.randint(0, train_x[j].shape[0], (int(sel.sum()),), generator=generator)
                x[sel], y[sel] = train_x[j][idx], train_y[j][idx]
            losses.append(step(x, y, c))
    else:
        for j in range(n_l):
            for _ in range(cfg.steps_per_lang):
                idx = torch.randint(0, train_x[j].shape[0], (cfg.batch,), generator=generator)
                x, y, c = train_x[j][idx], train_y[j][idx], torch.full((cfg.batch,), j)
                if j > 0:
                    rc = torch.randint(0, j, (cfg.replay_batch,), generator=generator)
                    ri = torch.randint(0, cfg.buffer, (cfg.replay_batch,), generator=generator)
                    rx = torch.stack([buf_x[a][b] for a, b in zip(rc.tolist(), ri.tolist())])
                    ry = torch.stack([buf_y[a][b] for a, b in zip(rc.tolist(), ri.tolist())])
                    x, y, c = torch.cat([x, rx]), torch.cat([y, ry]), torch.cat([c, rc])
                losses.append(step(x, y, c))
            store(j)
    return {"model": model, "plain": plain, "buf_x": buf_x, "buf_y": buf_y, "hold_x": hold_x, "hold_y": hold_y,
            "final_loss": sum(losses[-100:]) / 100}


def kl_to_truth(model: nn.Module, trans: torch.Tensor, fam: LangFamily, x: torch.Tensor, c: int, j: int) -> float:
    """文脈 c のヘッドの、言語 j の真の次の文字の分布からの KL（窓 x の平均, nats）。保存された答えではなく真の分布で
    測る（12.6.151節の「そのタスクの正解で測る」にあたる）。"""
    with torch.no_grad():
        logq = model(x, torch.full((x.shape[0],), c))
    p = trans[j][state_index(x, fam.vocab, fam.order)]
    return float((p * (torch.log(p.clamp_min(1e-12)) - logq)).sum(-1).mean())


def evaluate(fam: LangFamily, trans: torch.Tensor, trained: dict, cfg: TrainConfig, cue_ns: tuple[int, ...],
             generator: torch.Generator, episodes: int = 100, pool: int = 4, test_chains: int = 20,
             rules: tuple[str, ...] = RULES + FIX1_RULES) -> dict:
    """各言語・各手がかりの数・各規則で文脈を選び、選択精度と次の文字の CE（選んだヘッドの新しい文での CE）を測る。

    CE は言語 j・文脈 c ごとに新しい文の全窓で一度計算した表 CE[c, j] を、選んだ文脈で引く（選択の分布についての
    期待値。12.6.136節の、1回の選択で測るばらつきを避ける）。"""
    model, plain = trained["model"], trained["plain"]
    n_l, k, v = fam.n_langs, cfg.k, fam.vocab
    out: dict = {"ce": {}, "native": {}}
    with torch.no_grad():
        ce = torch.zeros(n_l, n_l)
        ce_plain = torch.zeros(n_l)
        entropy = torch.zeros(n_l)
        for j in range(n_l):
            x, y = windows(sample_text(trans[j], fam, test_chains, 200, generator), k)
            lp = model.logp_all(x)  # [N, C, V]
            ce[:, j] = -lp.gather(-1, y[:, None, None].expand(-1, n_l, 1)).squeeze(-1).mean(0)
            ce_plain[j] = -plain(x, torch.zeros_like(y)).gather(-1, y[:, None]).mean()
            p = trans[j][state_index(x, v, fam.order)]
            entropy[j] = -(p * torch.log(p.clamp_min(1e-12))).sum(-1).mean()
        # 本来の文脈の真の分布からの KL を入力の出どころごとに（上書きの切り分け。12.6.151節の own_buf・fresh など）
        native = {"own_buf": [], "other_buf": [], "holdout": [], "fresh": []}
        for j in range(n_l):
            fx, _ = windows(sample_text(trans[j], fam, test_chains, 200, generator), k)
            others = torch.cat([trained["buf_x"][a] for a in range(n_l) if a != j])
            native["own_buf"].append(kl_to_truth(model, trans, fam, trained["buf_x"][j], j, j))
            native["other_buf"].append(kl_to_truth(model, trans, fam, others, j, j))
            native["holdout"].append(kl_to_truth(model, trans, fam, trained["hold_x"][j], j, j))
            native["fresh"].append(kl_to_truth(model, trans, fam, fx, j, j))
        out["native"] = {key: sum(val) / n_l for key, val in native.items()}
        out["native_per_lang"] = native
        out["ce_matrix"] = ce.tolist()
        out["ce_oracle"] = float(ce.diagonal().mean())
        out["ce_nocontext"] = float(ce_plain.mean())
        out["entropy"] = float(entropy.mean())
        fix_gen = torch.Generator().manual_seed(generator.initial_seed() + 777)  # 修正1の候補専用
        all_buf = torch.cat(trained["buf_x"])
        all_hold = torch.cat(trained["hold_x"])
        res: dict = {}
        for n in cue_ns:
            m = pool * n
            acc = {r: 0.0 for r in rules}
            ce_sel = {r: 0.0 for r in rules}
            info = {r: 0.0 for r in rules}
            per_lang = {r: [] for r in rules}
            for j in range(n_l):
                # 情報源の答えは真の分布から引く（窓の出どころによらず同じ規則）
                def answers(xw: torch.Tensor, gen: torch.Generator = generator) -> torch.Tensor:
                    p = trans[j][state_index(xw, v, fam.order)]
                    return torch.multinomial(p.reshape(-1, v), 1, generator=gen).reshape(xw.shape[:-1])

                # 与えられる手がかりの窓は、言語 j の文から独立に取る（12.6.143節の batch の手がかりにあたる）
                task_x = sample_text(trans[j], fam, episodes * m, k, generator).reshape(episodes, m, k)
                # passive: ひと続きの文（n + k 文字）の n 個の窓。答えは文の実際の次の文字
                px_, py_ = windows(sample_text(trans[j], fam, episodes, n + k, generator), k)
                pools = {"passage": px_.reshape(episodes, n, k), "task": task_x,
                         "replay": all_buf[torch.randint(0, all_buf.shape[0], (episodes, m), generator=generator)],
                         "holdout": all_hold[torch.randint(0, all_hold.shape[0], (episodes, m), generator=generator)]}
                ans = {key: answers(px) for key, px in pools.items() if key != "passage"}
                # 修正1: 文脈ごとに max(1, round(P·n / 文脈の数)) 個ずつ、その文脈のバッファ（保留サンプル）から。
                # 従来の規則の値を変えないよう、専用の乱数を使う
                m_c = max(1, round(m / n_l))
                src = torch.arange(n_l).repeat_interleave(m_c)[None].expand(episodes, -1)
                fix = {}
                for key, bank in (("replay_cued", trained["buf_x"]), ("holdout_cued", trained["hold_x"])):
                    fix[key] = torch.cat([b[torch.randint(0, b.shape[0], (episodes, m_c), generator=fix_gen)]
                                          for b in bank], dim=1)
                own = trained["buf_x"][j]
                fix["replay_own"] = own[torch.randint(0, own.shape[0], (episodes, m), generator=fix_gen)]
                for key, px in fix.items():
                    ans[key] = answers(px, fix_gen)
                pools.update(fix)
                ans["passage"] = py_.reshape(episodes, n)
                logp = {key: model.logp_all(px.reshape(-1, k)).reshape(episodes, px.shape[1], n_l, v)
                        for key, px in pools.items()}
                spec = {"passive": ("passage", False), "uniform": ("task", False), "active": ("task", True),
                        "uniform_replay": ("replay", False),
                        "active_replay": ("replay", True), "active_holdout": ("holdout", True),
                        "active_replay_cued": ("replay_cued", True), "active_holdout_cued": ("holdout_cued", True),
                        "active_replay_own": ("replay_own", True)}
                for r in rules:
                    key, act = spec[r]
                    chosen, inf = select_contexts(logp[key], ans[key], n, act, src if key.endswith("_cued") else None)
                    a = float((chosen == j).float().mean())
                    acc[r] += a / n_l
                    ce_sel[r] += float(ce[chosen, j].mean()) / n_l
                    info[r] += float(inf.mean()) / n_l
                    per_lang[r].append(a)
            res[str(n)] = {"acc": acc, "ce": ce_sel, "info": info, "acc_per_lang": per_lang}
        out["cues"] = res
    return out

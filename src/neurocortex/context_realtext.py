"""比較器ゲートの CA3 補完を、実テキストの多言語の混合と自己回帰の文字 Transformer に移す（12.6.213〜214節）。

12.6.191〜192節（context_lm の active_replay_cmp）の枠組みをそのまま保ち、合成言語と窓のモデルだけを差し替える。
- 言語: Europarl（NLTK の europarl_raw の標本。11言語の議事録の対訳、各言語10日分）の実テキスト。小文字にし、
  全言語で共有する文字の集合（出現の割合が min_frac 以上の文字。他は未知の文字1つ）で符号化する。
- 情報源（話し手）: 言語ごとに、LM の学習とは別の日の議事録で数えた Witten–Bell の補間つき文字 n-gram。
  窓（プロンプト）を見せると、その言語の統計で次の文字を答える（合成言語の真の遷移の代わり）。
- モデル: 文脈ごとのヘッドを持つ因果的な（自己回帰の）文字 Transformer。胴体（埋め込み・位置・自己注意）は共有し、
  文脈ごとに隠れ層へのバイアスと出力ヘッドを持つ（context_lm.ContextCharLM の形を自己回帰の胴体に載せた）。
  長さ T の文の全位置の次の文字を学習する。手がかりの窓（k 文字）は、窓の最後の位置の予測で読む。
- 痕跡: 学習文の段落（text_len 文字）の中の長さ T + 1 の断片。リプレイは断片の全位置を学び、手がかりとしての窓は
  断片の入力の最後の k 文字（答えの位置は断片の最後の文字）。時間文脈のタグは、段落の先頭から窓の最後の文字までの
  3文字組の項目を、固定の乱数の表で dim 個の単位へ写して漂流させたもの（3文字組の one-hot は次元が大きすぎるため）。
- テスト時の文脈推定と規則（active・active_replay_*）は context_lm.evaluate と同じ。補完と比較器は
  context_lm.ca3_complete・ca1_mismatch と同じ値を、選んだ痕跡の平均を行列の積で計算する速い版（complete・mismatch。
  テストで一致を確認）で求める。recall_uncertainty・select_contexts は context_lm のものをそのまま使う。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

from .context_lm import recall_uncertainty, select_contexts

LANG_CODES = {"danish": "da", "dutch": "nl", "english": "en", "finnish": "fi", "french": "fr", "german": "de",
              "greek": "el", "italian": "it", "portuguese": "pt", "spanish": "es", "swedish": "sv"}
# 12.6.213節の規則。own は本来の文脈のバッファだけから取る診断（規則ではない）
REAL_RULES = ("uniform", "active", "active_replay", "active_replay_own", "active_replay_tcm", "active_replay_ca3",
              "active_replay_ach", "active_replay_ach_shuf", "active_replay_cmp", "active_replay_cmp_shuf",
              "active_holdout_cmp")


# ---------------------------------------------------------------- コーパス

def normalize_text(s: str) -> str:
    """小文字にし、空白をまとめる。"""
    return re.sub(r"\s+", " ", s.lower()).strip()


def read_europarl(root: Path, lang: str) -> list[str]:
    """europarl_raw/<lang>/ の各ファイル（日付の順）を、タグの行を除いて1つの文字列にしたもの。"""
    docs = []
    for p in sorted((Path(root) / lang).iterdir()):
        lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("<")]
        docs.append(normalize_text(" ".join(lines)))
    return docs


def build_vocab(texts: list[str], min_frac: float = 1e-4) -> str:
    """全言語で共有する文字の集合（出現の割合が min_frac 以上の文字を出現の多い順に）。番号 0 は未知の文字。"""
    counts: dict[str, int] = {}
    for t in texts:
        for ch in t:
            counts[ch] = counts.get(ch, 0) + 1
    total = sum(counts.values())
    return "".join(sorted((c for c, n in counts.items() if n / total >= min_frac), key=lambda c: (-counts[c], c)))


def encode(text: str, vocab: str) -> torch.Tensor:
    """文字列 → 番号 [L]（vocab の文字は 1 から、未知の文字は 0）。"""
    table = {c: i + 1 for i, c in enumerate(vocab)}
    return torch.tensor([table.get(c, 0) for c in text], dtype=torch.long)


@dataclass
class LangData:
    """1つの言語の符号化した文字列。train は LM の学習、speaker は話し手の n-gram、test は手がかりと評価。"""
    name: str
    train: torch.Tensor
    speaker: torch.Tensor
    test: torch.Tensor


def split_docs(docs: list[str], vocab: str, n_train: int, n_speaker: int, name: str = "") -> LangData:
    """日ごとの文書を、学習（先頭 n_train 日）・話し手（次の n_speaker 日）・テスト（残り）に分ける。"""
    enc = [encode(d, vocab) for d in docs]
    return LangData(name, torch.cat(enc[:n_train]), torch.cat(enc[n_train:n_train + n_speaker]),
                    torch.cat(enc[n_train + n_speaker:]))


# ---------------------------------------------------------------- 話し手（文字 n-gram）

class NgramSpeaker:
    """Witten–Bell の補間つき文字 n-gram（order 文字の組。直前 order − 1 文字で次の文字を予測する）。

    p_o(w|h) = (c(h, w) + T(h)·p_{o−1}(w|h')) / (c(h) + T(h))、T(h) は h の後に現れた文字の種類の数、h' は h の最後の
    o − 2 文字。最低次は加算1の一様分布との補間。見たことのない h は1つ下の次数の分布をそのまま使う。"""

    def __init__(self, ids: torch.Tensor, vocab: int, order: int = 5):
        self.v, self.order = vocab, order
        self.uni = (torch.bincount(ids, minlength=vocab).double() + 1.0)
        self.uni = (self.uni / self.uni.sum()).float()
        self.tables = []  # 次数 o = 2..order: (文脈の番号の昇順の表 [K], 回数 [K, V])
        for o in range(2, order + 1):
            h = o - 1
            n = ids.shape[0] - h
            ctx = torch.zeros(n, dtype=torch.long)
            for i in range(h):
                ctx = ctx * vocab + ids[i:i + n]
            keys, inv = torch.unique(ctx, return_inverse=True)
            cnt = torch.zeros(keys.shape[0] * vocab)
            cnt.index_add_(0, inv * vocab + ids[h:h + n], torch.ones(n))
            self.tables.append((keys, cnt.reshape(-1, vocab)))

    def probs(self, x: torch.Tensor) -> torch.Tensor:
        """窓 x [..., k]（k ≧ order − 1）の次の文字の分布 [..., V]。"""
        shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        p = self.uni.expand(x.shape[0], -1).clone()
        for o, (keys, cnt) in zip(range(2, self.order + 1), self.tables):
            h = o - 1
            ctx = torch.zeros(x.shape[0], dtype=torch.long)
            for i in range(h):
                ctx = ctx * self.v + x[:, x.shape[1] - h + i]
            pos = torch.searchsorted(keys, ctx).clamp_max(keys.shape[0] - 1)
            hit = keys[pos] == ctx
            c = cnt[pos]
            tot = c.sum(-1, keepdim=True)
            types = (c > 0).float().sum(-1, keepdim=True)
            p_new = (c + types * p) / (tot + types).clamp_min(1e-12)
            p = torch.where(hit[:, None], p_new, p)
        return p.reshape(*shape, self.v)

    def sample(self, x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """窓 x [..., k] に話し手が答える次の文字 [...]。"""
        p = self.probs(x)
        return torch.multinomial(p.reshape(-1, self.v), 1, generator=generator).reshape(x.shape[:-1])


# ---------------------------------------------------------------- モデル

class TransformerContextLM(nn.Module):
    """因果的な文字 Transformer の胴体（共有）＋文脈ごとのバイアスと出力ヘッド。

    z = tanh(W·h_t + b_c)、log p(·|x_≤t, c) = log_softmax(z·W_c + a_c)。h_t は位置 t までの因果的な自己注意の出力。"""

    def __init__(self, vocab: int, n_ctx: int, max_len: int, d: int = 64, layers: int = 2, heads: int = 4,
                 ff: int = 256, hidden: int = 128):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout=0.0, batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.mid = nn.Linear(d, hidden)
        self.ctx = nn.Embedding(n_ctx, hidden)
        nn.init.zeros_(self.ctx.weight)
        self.head_w = nn.Parameter(torch.randn(n_ctx, hidden, vocab) / math.sqrt(hidden))
        self.head_b = nn.Parameter(torch.zeros(n_ctx, vocab))
        self.register_buffer("mask", torch.triu(torch.full((max_len, max_len), float("-inf")), 1), persistent=False)

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, T] → 共有の隠れ層 [B, T, hidden]（文脈を足す前）。"""
        t = x.shape[1]
        h = self.tok(x) + self.pos(torch.arange(t))[None]
        return self.mid(self.norm(self.body(h, mask=self.mask[:t, :t], is_causal=True)))

    def seq_logp(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """全位置の対数確率 [B, T, V]（学習用）。"""
        z = torch.tanh(self.trunk(x) + self.ctx(c)[:, None, :])
        return torch.log_softmax(torch.einsum("bth,bhv->btv", z, self.head_w[c]) + self.head_b[c][:, None], -1)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """窓 x [N, k] の最後の位置の次の文字の対数確率 [N, V]。"""
        z = torch.tanh(self.trunk(x)[:, -1] + self.ctx(c))
        return torch.log_softmax(torch.einsum("nh,nhv->nv", z, self.head_w[c]) + self.head_b[c], -1)

    def logp_all(self, x: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
        """全文脈の対数確率 [N, 文脈, V]。"""
        out = []
        for i in range(0, x.shape[0], chunk):
            z = torch.tanh(self.trunk(x[i:i + chunk])[:, -1][:, None, :] + self.ctx.weight[None])
            out.append(torch.log_softmax(torch.einsum("nch,chv->ncv", z, self.head_w) + self.head_b[None], -1))
        return torch.cat(out) if out else torch.empty(0, self.head_b.shape[0], self.head_b.shape[1])


# ---------------------------------------------------------------- 時間文脈のタグ（3文字組の項目を dim 個の単位へ）

def item_table(vocab: int, dim: int, length: int = 3, seed: int = 12345) -> torch.Tensor:
    """length 文字の組の番号 → dim 個の単位の番号（固定の乱数の表 [vocab^length]）。"""
    return torch.randint(0, dim, (vocab ** length,), generator=torch.Generator().manual_seed(seed))


def items_of(x: torch.Tensor, vocab: int, table: torch.Tensor, length: int = 3) -> torch.Tensor:
    """文字列 x [..., L] の各位置の項目（そこで終わる length 文字の組を表で写した単位の番号）[..., L − length + 1]。"""
    n = x.shape[-1] - length + 1
    idx = torch.zeros(x.shape[:-1] + (n,), dtype=torch.long)
    for i in range(length):
        idx = idx * vocab + x[..., i:i + n]
    return table[idx]


def drift_last(items: torch.Tensor, dim: int, beta: float) -> torch.Tensor:
    """時間文脈の漂流 t_i = (1 − β)·t_{i−1} + β·e(item_i)（t_0 = 0）の最後の時点 [B, dim]（items [B, T]）。
    t_T = Σ_i β·(1 − β)^{T − i}·e(item_i) を直接足す（context_lm.tcm_drift の最後の時点と同じ値）。"""
    b, t_n = items.shape
    w = beta * (1 - beta) ** torch.arange(t_n - 1, -1, -1, dtype=torch.float64)
    out = torch.zeros(b, dim, dtype=torch.float64)
    out.scatter_add_(1, items, w[None].expand(b, -1).contiguous())
    return out.float()


def trace_tags(passages: torch.Tensor, rows: torch.Tensor, ends: torch.Tensor, vocab: int, table: torch.Tensor,
               dim: int, beta: float, length: int = 3) -> torch.Tensor:
    """段落 passages [P, L] の rows 番目の段落の、位置 ends（窓の最後の文字）までの時間文脈のタグ [N, dim]。
    段落の先頭から漂流させ、ends より後の文字（答え）は含まない。"""
    items = items_of(passages, vocab, table, length)  # 位置 i の項目は items[:, i − length + 1]
    n_it = items.shape[1]
    j = torch.arange(n_it)
    last = ends - (length - 1)  # 含める最後の項目の番号
    w = torch.where(j[None] <= last[:, None], beta * (1 - beta) ** (last[:, None] - j[None]).clamp_min(0).double(),
                    torch.zeros((), dtype=torch.float64))
    out = torch.zeros(rows.shape[0], dim, dtype=torch.float64)
    out.scatter_add_(1, items[rows], w)
    return out.float()


def now_context(cue_windows: torch.Tensor, vocab: int, table: torch.Tensor, dim: int, beta: float,
                length: int = 3) -> torch.Tensor:
    """情報源が示したラベルなしの窓 cue_windows [E, M0, k] を順に聞いて漂流させた今の文脈 [E, dim]。"""
    items = items_of(cue_windows, vocab, table, length).reshape(cue_windows.shape[0], -1)
    return drift_last(items, dim, beta)


# ---------------------------------------------------------------- CA3 の補完と CA1 の比較器（速い版）

def complete(t: torch.Tensor, x_hat: torch.Tensor, iters: int, n_active: int, lam: torch.Tensor | None = None,
             return_first: bool = False, x_sp: torch.Tensor | None = None):
    """context_lm.ca3_complete と同じ補完。x_hat は正規化したタグ [N, D]（x_sp はその疎な CSR。タグは疎なので、
    点数 s·x̂_i を疎な積で求める）。上位 n_active 個の平均を、疎な選択の行列 [E, N]（選んだ痕跡に 1/n_active）と
    x_hat の積で求める。return_first=True では、最初の1回の反復の状態も返す（λ を与えないときの ca1_mismatch 用）。"""
    s = nn.functional.normalize(t, dim=-1)
    t_in, first = s, s
    n_active = min(n_active, x_hat.shape[0])
    e_n, n_n = s.shape[0], x_hat.shape[0]
    for it in range(iters):
        score = (x_sp @ s.T).T if x_sp is not None else s @ x_hat.T
        idx = score.topk(n_active, dim=-1).indices
        w = torch.sparse_coo_tensor(torch.stack([torch.arange(e_n).repeat_interleave(n_active), idx.reshape(-1)]),
                                    torch.full((e_n * n_active,), 1.0 / n_active), (e_n, n_n))
        s = nn.functional.normalize(torch.sparse.mm(w, x_hat), dim=-1)
        if lam is not None:
            s = nn.functional.normalize((1 - lam[:, None]) * t_in + lam[:, None] * s, dim=-1)
        if it == 0:
            first = s
    return (s, first) if return_first else s


def mismatch(t: torch.Tensor, x_hat: torch.Tensor, iters: int, n_active: int,
             x_sp: torch.Tensor | None = None) -> torch.Tensor:
    """context_lm.ca1_mismatch と同じ食い違い δ = 1 − cos(s_S, t̂)/cos(s_1, t̂)（0〜1 に切る）。"""
    t_hat = nn.functional.normalize(t, dim=-1)
    s, first = complete(t, x_hat, iters, n_active, return_first=True, x_sp=x_sp)
    c1, c_s = (first * t_hat).sum(-1), (s * t_hat).sum(-1)
    return (1 - c_s / c1.clamp_min(1e-12)).clamp(0.0, 1.0)


# ---------------------------------------------------------------- 学習

@dataclass
class RealConfig:
    regime: str = "joint"  # joint / seq
    steps_per_lang: int = 1500
    batch: int = 64
    replay_batch: int = 32
    seq_len: int = 32  # 学習の文・痕跡の断片の長さ T（断片は T + 1 文字）
    k: int = 4  # 手がかりの窓の長さ
    text_len: int = 400  # 時間文脈を漂流させる段落の長さ
    buffer: int = 2000  # 文脈ごとのバッファの痕跡の数（保留サンプルも同じ数）
    lr: float = 3e-3
    d: int = 64
    layers: int = 2
    heads: int = 4
    hidden: int = 128
    tcm_beta: float = 0.005
    tcm_dim: int = 4096
    ca3_iters: int = 3
    ca3_sparsity: float = 0.025
    order: list[int] = field(default_factory=list)  # seq で学ぶ言語の順（空なら 0, 1, …）


def _passages(ids: torch.Tensor, text_len: int) -> torch.Tensor:
    n = ids.shape[0] // text_len
    return ids[:n * text_len].reshape(n, text_len)


def train_realtext(data: list[LangData], vocab: int, cfg: RealConfig, generator: torch.Generator) -> dict:
    """文脈つきの Transformer を学習し、各言語のバッファと保留サンプル（断片・窓・答え・タグ）を返す。

    joint は全言語を毎ステップ混ぜて学ぶ。seq は言語を順に学び、学習の終わりにその言語の痕跡を保存し、以後の各
    ステップで過去の言語のバッファから replay_batch 個の断片をリプレイする。"""
    if cfg.regime not in ("joint", "seq"):
        raise ValueError("regime は joint・seq")
    n_l, t_len = len(data), cfg.seq_len
    pas = [_passages(d.train, cfg.text_len) for d in data]
    flat = [p.reshape(-1) for p in pas]
    table = item_table(vocab, cfg.tcm_dim)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(torch.randint(0, 2 ** 31 - 1, (1,), generator=generator)))
        model = TransformerContextLM(vocab, n_l, t_len, cfg.d, cfg.layers, cfg.heads, hidden=cfg.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    bank: dict = {"buf": [None] * n_l, "hold": [None] * n_l}

    def store(j: int) -> None:
        p = pas[j]
        n = 2 * cfg.buffer
        rows = torch.randint(0, p.shape[0], (n,), generator=generator)
        starts = torch.randint(0, cfg.text_len - t_len, (n,), generator=generator)
        seg = p[rows[:, None], starts[:, None] + torch.arange(t_len + 1)[None]]  # [n, T + 1]
        ends = starts + t_len - 1  # 窓の最後の文字の段落の中の位置
        tags = trace_tags(p, rows, ends, vocab, table, cfg.tcm_dim, cfg.tcm_beta)
        for key, sl in (("buf", slice(0, cfg.buffer)), ("hold", slice(cfg.buffer, n))):
            bank[key][j] = {"seg": seg[sl], "x": seg[sl, t_len - cfg.k:t_len], "y": seg[sl, t_len],
                            "tag": tags[sl]}

    def batch_from(j: int, n: int) -> torch.Tensor:
        s = torch.randint(0, flat[j].shape[0] - t_len - 1, (n,), generator=generator)
        return flat[j][s[:, None] + torch.arange(t_len + 1)[None]]

    def step(seg: torch.Tensor, c: torch.Tensor) -> float:
        opt.zero_grad()
        lp = model.seq_logp(seg[:, :-1], c)
        loss = nn.functional.nll_loss(lp.reshape(-1, lp.shape[-1]), seg[:, 1:].reshape(-1))
        loss.backward()
        opt.step()
        return float(loss.detach())

    losses = []
    order = cfg.order or list(range(n_l))
    if cfg.regime == "joint":
        for j in range(n_l):
            store(j)
        for _ in range(cfg.steps_per_lang * n_l):
            c = torch.randint(0, n_l, (cfg.batch,), generator=generator)
            seg = torch.empty(cfg.batch, t_len + 1, dtype=torch.long)
            for j in range(n_l):
                sel = c == j
                if int(sel.sum()):
                    seg[sel] = batch_from(j, int(sel.sum()))
            losses.append(step(seg, c))
    else:
        for pos, j in enumerate(order):
            past = order[:pos]
            for _ in range(cfg.steps_per_lang):
                seg, c = batch_from(j, cfg.batch), torch.full((cfg.batch,), j)
                if past:
                    rc = torch.tensor(past)[torch.randint(0, len(past), (cfg.replay_batch,), generator=generator)]
                    ri = torch.randint(0, cfg.buffer, (cfg.replay_batch,), generator=generator)
                    rseg = torch.stack([bank["buf"][a]["seg"][b] for a, b in zip(rc.tolist(), ri.tolist())])
                    seg, c = torch.cat([seg, rseg]), torch.cat([c, rc])
                losses.append(step(seg, c))
            store(j)
    model.eval()
    return {"model": model, "bank": bank, "table": table, "final_loss": sum(losses[-100:]) / 100}


# ---------------------------------------------------------------- 評価

def _ce_windows(model: TransformerContextLM, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """窓 x [N, k] の実際の次の文字 y の CE を文脈ごとに [文脈]。"""
    lp = model.logp_all(x)
    return -lp.gather(-1, y[:, None, None].expand(-1, lp.shape[1], 1)).squeeze(-1).mean(0)


def random_windows(ids: torch.Tensor, shape: tuple[int, ...], k: int, generator: torch.Generator):
    """文字列 ids から無作為な位置の窓 [*shape, k] と実際の次の文字 [*shape]。"""
    n = math.prod(shape)
    s = torch.randint(0, ids.shape[0] - k - 1, (n,), generator=generator)
    w = ids[s[:, None] + torch.arange(k + 1)[None]]
    return w[:, :k].reshape(*shape, k), w[:, k].reshape(shape)


def evaluate_realtext(data: list[LangData], speakers: list[NgramSpeaker], trained: dict, cfg: RealConfig,
                      vocab: int, cue_ns: tuple[int, ...], generator: torch.Generator, episodes: int = 100,
                      pool: int = 4, rules: tuple[str, ...] = REAL_RULES, n_fresh: int = 4000) -> dict:
    """各言語・各手がかりの数・各規則で文脈を選び、選択精度と選んだヘッドの CE（テスト文の実際の次の文字）を測る。
    規則の定義・乱数の分け方は context_lm.evaluate（12.6.191節）と同じ。"""
    model, bank, table = trained["model"], trained["bank"], trained["table"]
    n_l, k = len(data), cfg.k
    out: dict = {}
    with torch.no_grad():
        ce = torch.zeros(n_l, n_l)
        native = {"own_buf": [], "holdout": [], "fresh": []}
        for j in range(n_l):
            fx, fy = random_windows(data[j].test, (n_fresh,), k, generator)
            ce[:, j] = _ce_windows(model, fx, fy)
            native["fresh"].append(float(ce[j, j]))
            native["own_buf"].append(float(_ce_windows(model, bank["buf"][j]["x"], bank["buf"][j]["y"])[j]))
            native["holdout"].append(float(_ce_windows(model, bank["hold"][j]["x"], bank["hold"][j]["y"])[j]))
        out["native"] = {key: sum(v) / n_l for key, v in native.items()}
        out["native_per_lang"] = native
        out["ce_matrix"] = ce.tolist()
        out["ce_oracle"] = float(ce.diagonal().mean())
        # 話し手の n-gram の、テスト文での CE（話し手の質の確認）
        out["speaker_ce"] = []
        for j in range(n_l):
            fx, fy = random_windows(data[j].test, (n_fresh,), k, generator)
            p = speakers[j].probs(fx)
            out["speaker_ce"].append(float(-torch.log(p.gather(-1, fy[:, None]).clamp_min(1e-12)).mean()))
        seed0 = generator.initial_seed()
        gens = {name: torch.Generator().manual_seed(seed0 + off) for name, off in
                (("fix", 777), ("tcm", 888), ("ca3", 999), ("ach", 1111), ("ashuf", 1212), ("cmp", 1313),
                 ("cshuf", 1414))}
        buf_x = torch.cat([b["x"] for b in bank["buf"]])
        hold_x = torch.cat([b["x"] for b in bank["hold"]])
        buf_tag = torch.cat([b["tag"] for b in bank["buf"]])
        hold_tag = torch.cat([b["tag"] for b in bank["hold"]])
        buf_src = torch.cat([torch.full((b["x"].shape[0],), a) for a, b in enumerate(bank["buf"])])
        hold_src = torch.cat([torch.full((b["x"].shape[0],), a) for a, b in enumerate(bank["hold"])])
        n_act = max(1, round(cfg.ca3_sparsity * buf_tag.shape[0]))
        buf_hat, hold_hat = nn.functional.normalize(buf_tag, dim=-1), nn.functional.normalize(hold_tag, dim=-1)
        buf_mem, hold_mem = (buf_hat, buf_hat.to_sparse_csr()), (hold_hat, hold_hat.to_sparse_csr())
        res: dict = {}
        for n in cue_ns:
            m = pool * n
            acc = {r: 0.0 for r in rules}
            ce_sel = {r: 0.0 for r in rules}
            purity = {r: 0.0 for r in rules if (r.startswith("active_replay_") or r.startswith("active_holdout_"))
                      and r != "active_replay_own"}
            lam_mean = {r: 0.0 for r in rules if "ach" in r or "cmp" in r}
            for j in range(n_l):
                spk = speakers[j]
                task_x, _ = random_windows(data[j].test, (episodes, m), k, generator)
                pools = {"task": task_x,
                         "replay": buf_x[torch.randint(0, buf_x.shape[0], (episodes, m), generator=generator)]}
                ans = {key: spk.sample(px, generator) for key, px in pools.items()}
                own = bank["buf"][j]["x"]
                pools["replay_own"] = own[torch.randint(0, own.shape[0], (episodes, m), generator=gens["fix"])]
                ans["replay_own"] = spk.sample(pools["replay_own"], gens["fix"])
                t_now = now_context(task_x, vocab, table, cfg.tcm_dim, cfg.tcm_beta)

                def recall(key, bank_x, mem, srcs, lam, iters, gen):
                    x_hat, x_sp = mem
                    s = complete(t_now, x_hat, iters, n_act, lam, x_sp=x_sp)
                    idx = (x_sp @ s.T).T.topk(min(m, x_hat.shape[0]), dim=-1).indices
                    pools[key] = bank_x[idx]
                    ans[key] = spk.sample(pools[key], gen)
                    if "active_" + key in purity:
                        purity["active_" + key] += float((srcs[idx] == j).float().mean()) / n_l
                    if lam is not None and "active_" + key in lam_mean:
                        lam_mean["active_" + key] += float(lam.mean()) / n_l

                if "active_replay_tcm" in rules:
                    recall("replay_tcm", buf_x, buf_mem, buf_src, None, 0, gens["tcm"])
                if "active_replay_ca3" in rules:
                    recall("replay_ca3", buf_x, buf_mem, buf_src, None, cfg.ca3_iters, gens["ca3"])
                if any(r.startswith("active_replay_ach") for r in rules):
                    s0 = (buf_mem[1] @ nn.functional.normalize(t_now, dim=-1).T).T
                    u = recall_uncertainty(s0.topk(n_act, dim=-1).indices, buf_src, n_l)
                    recall("replay_ach", buf_x, buf_mem, buf_src, u, cfg.ca3_iters, gens["ach"])
                    if "active_replay_ach_shuf" in rules:
                        recall("replay_ach_shuf", buf_x, buf_mem, buf_src,
                               u[torch.randperm(u.shape[0], generator=gens["ashuf"])], cfg.ca3_iters, gens["ach"])
                if any("cmp" in r for r in rules):
                    lam = 1 - mismatch(t_now, buf_mem[0], cfg.ca3_iters, n_act, buf_mem[1])
                    recall("replay_cmp", buf_x, buf_mem, buf_src, lam, cfg.ca3_iters, gens["cmp"])
                    if "active_replay_cmp_shuf" in rules:
                        recall("replay_cmp_shuf", buf_x, buf_mem, buf_src,
                               lam[torch.randperm(lam.shape[0], generator=gens["cshuf"])], cfg.ca3_iters, gens["cmp"])
                    if "active_holdout_cmp" in rules:
                        lam_h = 1 - mismatch(t_now, hold_mem[0], cfg.ca3_iters, n_act, hold_mem[1])
                        recall("holdout_cmp", hold_x, hold_mem, hold_src, lam_h, cfg.ca3_iters, gens["cmp"])
                spec = {"uniform": ("task", False), "active": ("task", True), "active_replay": ("replay", True),
                        **{r: (r.removeprefix("active_"), True) for r in rules if r.startswith("active_")
                           and r not in ("active", "active_replay")}}
                logps: dict = {}
                for r in rules:
                    key, act = spec[r]
                    if key not in logps:
                        px = pools[key]
                        logps[key] = model.logp_all(px.reshape(-1, k)).reshape(episodes, px.shape[1], n_l, -1)
                    chosen, _ = select_contexts(logps[key], ans[key], n, act)
                    acc[r] += float((chosen == j).float().mean()) / n_l
                    ce_sel[r] += float(ce[chosen, j].mean()) / n_l
            res[str(n)] = {"acc": acc, "ce": ce_sel, "purity": purity, "lambda": lam_mean}
        out["cues"] = res
    return out

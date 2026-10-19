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

12.6.179〜180節: tcm_* は痕跡に保存時の時間文脈のタグ（TCM）をつけ、今の文脈に近い痕跡を想起する（TrainConfig.tcm_beta > 0 のときだけ）。
12.6.183〜184節: ca3_complete は、手がかりから作った部分的な今の文脈を、保存したタグへの CA3 の再帰結合のアトラクタの
反復（k-WTA の抑制つき）で補完してから想起する（active_*_ca3 の規則のときだけ）。
12.6.187〜188節: 補完の強さ λ を想起の不確かさ（補完なしで想起した痕跡の保存時の文脈の分布のエントロピー）で
ゲートする（アセチルコリンによる再帰結合の抑制: Hasselmo & Schnell 1994; Hasselmo 2006。active_*_ach の規則のときだけ）。
12.6.191〜192節: 補完の強さ λ を CA1 の比較器（CA3 が想起した文脈と嗅内皮質からの入力の食い違い δ）で決める
（Hasselmo, Wyble & Wallenstein 1996; Lisman & Grace 2005。active_*_cmp の規則のときだけ）。
12.6.197〜198節: 痕跡を歯状回のパターン分離（疎な拡張と k-WTA。dg_codes）で符号化して CA3 に保存し、補完を
その符号の空間で行う（手がかりは貫通路から生のタグで入る。ca3_recall_codes。active_*_dg・rp の規則のときだけ）。

12.6.143節と違い、能動的な選択も打ち切りなしで n 個すべてを観測する（観測の数を規則の間で揃えるため）。
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import torch
from torch import nn

EPS = 0.02  # 予測の平滑化 p̃ = (1 − ε)·p + ε/V（12.6.135節の確定値）
RULES = ("passive", "uniform", "active", "uniform_replay", "active_replay", "active_holdout")
# 修正1（12.6.173節）: 事後分布を手がかりにした、文脈ごとのバッファからの想起。active_replay_own は本来の文脈の
# バッファだけから取る診断用（本来の文脈を知っているので規則ではない）
FIX1_RULES = ("active_replay_cued", "active_holdout_cued", "active_replay_own")
# 12.6.175節の修正1: active と同じ手がかりの後の事後分布で、文脈を1つに決めずに予測する。active_soft は事後確率で
# 重みづけたゲイン Σ_c p(c)·g_c（GainCharLM のみ）、active_mix は予測の混合 Σ_c p(c)·p_c(·|窓)（参考）
SOFT_RULES = ("active_soft", "active_mix")
# 12.6.179節: 痕跡に保存時の時間文脈のタグ（TCM: Howard & Kahana 2002）をつけ、今の文脈に近い痕跡を想起して候補に
# する。active_replay_tcm はバッファ、active_holdout_tcm は保留サンプル（リプレイしていない。統制）から想起する
TCM_RULES = ("active_replay_tcm", "active_holdout_tcm")
# 12.6.183節: 今の文脈を CA3 のパターン補完（TrainConfig.ca3_iters 回の反復）で補完してから想起する。_s{S} は反復の
# 回数を S に変えた対照
CA3_RULES = ("active_replay_ca3", "active_holdout_ca3", "active_replay_ca3_s1", "active_replay_ca3_s10")
# 12.6.187節: 補完の強さ λ を想起の不確かさ u でゲートする。ach は λ = u、ach_hard は λ = 1[u > ACH_THRESHOLD]、
# ach_shuf は同じ言語・手がかりの数のエピソードの間で λ を並べ替えた統制（強さの分布は同じで、エピソードとの対応だけ壊す）、
# ach_inv は逆向き λ = 1 − u（不確かなときほど再帰結合を抑える。Yu & Dayan 2005 を字義通りに当てはめた向き。参考）
ACH_RULES = ("active_replay_ach", "active_holdout_ach", "active_replay_ach_hard", "active_replay_ach_shuf",
             "active_replay_ach_inv")
ACH_THRESHOLD = 0.5
# 12.6.191節: CA1 の比較器のゲート。cmp は λ = 1 − δ（δ = ca1_mismatch）、cmp_shuf は同じ言語・手がかりの数のエピソードの
# 間で λ を並べ替えた統制、cmp_u は不確かさと組み合わせた λ = u·(1 − δ)（参考）
CMP_RULES = ("active_replay_cmp", "active_holdout_cmp", "active_replay_cmp_shuf", "active_replay_cmp_u")
# 12.6.197節: 歯状回のパターン分離。dg は歯状回の符号で補完（λ = 1）、dgcmp はさらに比較器のゲート（λ = 1 − δ）、
# rp・rpcmp は疎性を合わせた乱数射影（競合なし、入力の幾何を保つ）の対照
DG_RULES = ("active_replay_dg", "active_replay_dgcmp", "active_holdout_dgcmp", "active_replay_rp",
            "active_replay_rpcmp")


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


def tcm_items(x: torch.Tensor, vocab: int, length: int = 2) -> torch.Tensor:
    """文字列 x [..., L] の各位置 i ≥ length − 1 の項目（そこで終わる length 文字の組の番号）[..., L − length + 1]。
    TCM の項目の表現は互いに直交する（length 文字の組の one-hot, vocab^length 次元）。"""
    n = x.shape[-1] - length + 1
    idx = torch.zeros(x.shape[:-1] + (n,), dtype=torch.long)
    for i in range(length):
        idx = idx * vocab + x[..., i:i + n]
    return idx


def tcm_drift(items: torch.Tensor, dim: int, beta: float) -> torch.Tensor:
    """時間文脈の漂流 t_i = (1 − β)·t_{i−1} + β·e(item_i)（t_0 = 0）。items [B, T] → 各時点の文脈 [B, T, dim]。

    Howard & Kahana 2002 の t_i = ρ_i·t_{i−1} + β·t^IN_i で、ノルムの正規化を比較時の余弦に回した形。"""
    b, t_n = items.shape
    out = torch.empty(b, t_n, dim)
    t = torch.zeros(b, dim)
    rows = torch.arange(b)
    for i in range(t_n):
        t = t * (1 - beta)
        t[rows, items[:, i]] += beta
        out[:, i] = t
    return out


def tcm_tags(text: torch.Tensor, k: int, vocab: int, beta: float, length: int = 2) -> torch.Tensor:
    """文 [n_chains, L] の各窓（windows() と同じ順）の時間文脈のタグ [窓, vocab^length]。窓の最後の文字までの
    項目を取り込んだ文脈で、次の文字（答え）は含まない。length ≦ k。"""
    dim = vocab ** length
    ctx = tcm_drift(tcm_items(text, vocab, length), dim, beta)  # 位置 i の文脈は ctx[:, i − length + 1]
    return ctx[:, k - length:text.shape[1] - length].reshape(-1, dim)


def ca3_complete(t: torch.Tensor, tags: torch.Tensor, iters: int, n_active: int,
                 lam: torch.Tensor | None = None) -> torch.Tensor:
    """CA3 の再帰結合によるパターン補完（12.6.183節）。部分的な文脈 t [E, D] から始め、保存したタグ tags [N, D] を
    記憶したアトラクタの網を iters 回反復する: s ← normalize(mean{x̂_i : s との余弦が上位 n_active 個の痕跡})。
    痕跡ごとの記憶の単位が k-WTA の抑制で上位 n_active 個だけ発火し、その結合が CA3 の状態を書き戻す形
    （単位ごとに1つの痕跡を持つホップフィールド網）。反復は今の文脈の近くの痕跡の重心へ向かう（平均移動）。
    文脈のラベルは使わない。iters = 0 なら t を正規化して返す（補完なし）。

    lam [E]（12.6.187節）を与えると、嗅内皮質からの入力 t̂ を保ったまま再帰結合の寄与を λ 倍にする:
    s ← normalize((1 − λ)·t̂ + λ·normalize(mean{...}))。λ = 1 は上の補完と同じ、λ = 0 は補完なし。"""
    x = nn.functional.normalize(tags, dim=-1)
    s = nn.functional.normalize(t, dim=-1)
    t_in = s
    n_active = min(n_active, x.shape[0])
    for _ in range(iters):
        idx = (s @ x.T).topk(n_active, dim=-1).indices
        s = nn.functional.normalize(x[idx].mean(1), dim=-1)
        if lam is not None:
            s = nn.functional.normalize((1 - lam[:, None]) * t_in + lam[:, None] * s, dim=-1)
    return s


def ca1_mismatch(t: torch.Tensor, tags: torch.Tensor, iters: int, n_active: int) -> torch.Tensor:
    """CA1 の比較器の食い違い δ [E]（12.6.191節）。嗅内皮質から直接入る入力 t̂（今の文脈）と、CA3 が再帰結合で
    想起した文脈 s_S（iters 回の補完、λ = 1）の一致 cos(s_S, t̂) を、最初の想起 s_1（1回の反復）の一致で割った比の
    不足 δ = 1 − cos(s_S, t̂)/cos(s_1, t̂)（0 未満は 0 に切る）。再帰結合が入力から離れた文脈へ落ち着くほど大きい。
    一致そのものは手がかりの量（今の文脈の疎さ）で大きく変わるので、最初の想起との比で量の影響を除く。"""
    t_hat = nn.functional.normalize(t, dim=-1)
    c1 = (ca3_complete(t, tags, 1, n_active) * t_hat).sum(-1)
    c_s = (ca3_complete(t, tags, iters, n_active) * t_hat).sum(-1)
    return (1 - c_s / c1.clamp_min(1e-12)).clamp(0.0, 1.0)


def dg_weights(dim: int, n_units: int, conn: float, generator: torch.Generator) -> torch.Tensor:
    """嗅内皮質 → 歯状回の結合 [n_units, dim]。各顆粒細胞は入力の割合 conn に無作為に結合する（0/1）。"""
    return (torch.rand(n_units, dim, generator=generator) < conn).float()


def dg_codes(tags: torch.Tensor, w: torch.Tensor, n_active: int, chunk: int = 2000) -> tuple:
    """歯状回のパターン分離（12.6.197節）。タグ tags [N, D] を結合 w [M, D] で M 個の顆粒細胞へ拡張し、抑制性の
    介在ニューロンによる競合（k-WTA）で上位 n_active 個だけを残す: c = relu(h − θ)、θ は (n_active + 1) 番目の
    入力（減算の抑制）。共通の成分は閾値で消え、重なった入力の符号の重なりが下がる（Marr 1969・1971;
    Treves & Rolls 1994）。疎な符号 (番号 [N, a], 値 [N, a]（行ごとに正規化）, M) を返す。"""
    x = nn.functional.normalize(tags, dim=-1)
    ind, val = [], []
    for i in range(0, x.shape[0], chunk):
        top = (x[i:i + chunk] @ w.T).topk(n_active + 1, dim=-1)
        order = top.indices[:, :-1].sort(dim=-1)  # CSR の列の番号は行ごとに昇順
        v = (top.values[:, :-1] - top.values[:, -1:]).clamp_min(0.0).gather(1, order.indices)
        ind.append(order.values)
        val.append(nn.functional.normalize(v, dim=-1))
    return torch.cat(ind), torch.cat(val), w.shape[0]


def rp_codes(tags: torch.Tensor, w: torch.Tensor, n_active: int, conn: float) -> torch.Tensor:
    """疎性を合わせた乱数射影の対照（12.6.197節）。dg_codes と同じ結合から平均（conn）を引いて0平均にし、入力に
    よらない固定の n_active 個の単位で読む線形の射影（競合なし。余弦の幾何をほぼ保つ: Johnson–Lindenstrauss）。
    発火する単位の数は歯状回と同じ。符号 [N, n_active]（行ごとに正規化）。"""
    x = nn.functional.normalize(tags, dim=-1)
    return nn.functional.normalize(x @ (w[:n_active] - conn).T, dim=-1)


def _code_ops(codes):
    """符号の空間の演算（余弦の点数 [E, N] と、選んだ痕跡の符号の平均 [E, dim]）。codes は密な [N, dim] か、
    dg_codes の疎な (番号, 値, dim)。"""
    if isinstance(codes, tuple):
        ind, val, dim = codes
        n, a = ind.shape
        with warnings.catch_warnings():  # CSR は beta の警告を出す
            warnings.simplefilter("ignore", UserWarning)
            csr = torch.sparse_csr_tensor(torch.arange(0, n * a + 1, a), ind.reshape(-1), val.reshape(-1), (n, dim),
                                          check_invariants=False)

        def score(s: torch.Tensor) -> torch.Tensor:
            return (csr @ s.T).T

        def mean(idx: torch.Tensor) -> torch.Tensor:
            e = idx.shape[0]
            out = torch.zeros(e, dim).scatter_add_(1, ind[idx].reshape(e, -1), val[idx].reshape(e, -1))
            return out / idx.shape[1]
        return score, mean
    return (lambda s: s @ codes.T), (lambda idx: codes[idx].mean(1))


def ca3_recall_codes(t: torch.Tensor, tags: torch.Tensor, codes, iters: int, n_active: int, m: int,
                     lam: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """CA3 の痕跡を符号 codes（歯状回の苔状線維が決めた CA3 の表現）で保存したときのパターン補完と想起（12.6.197節）。

    手がかりは貫通路から入る（Treves & Rolls 1992: 苔状線維は保存、貫通路は想起）: 記憶の単位 i への入力は
    p_i = cos(t̂, x̂_i)（x̂ は嗅内皮質の空間のタグ）。k-WTA で上位 n_active 個が発火し、CA3 の状態を
    s ← normalize(mean{c_i}) と書き戻す。次の反復の入力は u_i = (1 − λ)·p_i + λ·cos(s, ĉ_i)（アセチルコリンは
    再帰結合だけを抑える）。最後に u の上位 m 個を想起する。codes = 正規化したタグなら ca3_complete + tcm_recall と
    同じ順位になる（テストで確認）。

    返り値: 想起した番号 [E, m] と、比較器の食い違い δ [E]（λ = 1 の反復のときの定義どおり。CA3 の状態を
    嗅内皮質の空間へ読み出した r = normalize(mean{x̂_i}) で δ = 1 − cos(r_S, t̂)/cos(r_1, t̂)）。"""
    x = nn.functional.normalize(tags, dim=-1)
    t_hat = nn.functional.normalize(t, dim=-1)
    score, mean = _code_ops(codes)
    p = t_hat @ x.T
    u = p
    n_active = min(n_active, x.shape[0])
    c1 = c_s = torch.ones(t.shape[0])
    for it in range(iters):
        idx = u.topk(n_active, dim=-1).indices
        s = nn.functional.normalize(mean(idx), dim=-1)
        c_s = (nn.functional.normalize(x[idx].mean(1), dim=-1) * t_hat).sum(-1)
        if it == 0:
            c1 = c_s
        rec = score(s)
        u = rec if lam is None else (1 - lam[:, None]) * p + lam[:, None] * rec
    delta = (1 - c_s / c1.clamp_min(1e-12)).clamp(0.0, 1.0)
    return u.topk(min(m, x.shape[0]), dim=-1).indices, delta


def code_overlap(codes, srcs: torch.Tensor, n: int, generator: torch.Generator) -> dict:
    """符号の重なり（操作の確認、12.6.197節）。無作為な n 個の痕跡の対の余弦の平均を、同じ言語の対と違う言語の対で
    分けて返す。比 between/within が小さいほど言語どうしが分かれている。"""
    sel = torch.randperm(srcs.shape[0], generator=generator)[:n]
    score, mean = _code_ops(codes)
    c = nn.functional.normalize(mean(sel[:, None]), dim=-1)
    sim = c @ c.T
    same = srcs[sel][:, None] == srcs[sel][None]
    off = ~torch.eye(len(sel), dtype=torch.bool)
    w, b = float(sim[same & off].mean()), float(sim[~same].mean())
    return {"within": w, "between": b, "ratio": b / w if w else float("nan")}


def recall_uncertainty(idx: torch.Tensor, srcs: torch.Tensor, n_ctx: int) -> torch.Tensor:
    """想起の不確かさ u [E]（12.6.187節）。痕跡 idx [E, m]（補完の前の k-WTA で発火する集団）の保存時の文脈 srcs[idx] の分布の正規化エントロピー
    H(q)/log C（0〜1）。q は記憶が示す文脈の事後分布の推定、max q は想起の純度の推定にあたる。保存時の文脈は痕跡に
    結びついた内部の索引で、今の情報源の言語のラベルではない。"""
    q = nn.functional.one_hot(srcs[idx], n_ctx).float().mean(1)
    return -(q * torch.log(q.clamp_min(1e-12))).sum(-1) / math.log(n_ctx)


def tcm_now(cue_windows: torch.Tensor, vocab: int, beta: float, length: int = 2) -> torch.Tensor:
    """情報源が示したラベルなしの窓 cue_windows [E, M0, k] を順に聞いて漂流させた今の文脈 t_now [E, vocab^length]。"""
    items = tcm_items(cue_windows, vocab, length).reshape(cue_windows.shape[0], -1)
    return tcm_drift(items, vocab ** length, beta)[:, -1]


def tcm_recall(cue_windows: torch.Tensor, tags: torch.Tensor, vocab: int, beta: float, m: int,
               length: int = 2, ca3_iters: int = 0, ca3_active: int = 1,
               ca3_lam: torch.Tensor | None = None) -> torch.Tensor:
    """今の文脈に近い痕跡の想起。cue_windows [E, M0, k]（情報源が示したラベルなしの窓。答えは使わない）を順に聞いて
    漂流させた文脈 t_now と、各痕跡のタグ tags [N, D] の余弦が大きい順に m 個の痕跡の番号 [E, m] を返す。
    ca3_iters > 0 なら、t_now を ca3_complete で補完してから比べる（12.6.183節）。"""
    t_now = tcm_now(cue_windows, vocab, beta, length)
    s = ca3_complete(t_now, tags, ca3_iters, ca3_active, ca3_lam)
    sim = s @ nn.functional.normalize(tags, dim=-1).T  # [E, N]
    return sim.topk(min(m, tags.shape[0]), dim=-1).indices


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
                    source: torch.Tensor | None = None, return_post: bool = False) -> tuple[torch.Tensor, ...]:
    """手がかりから文脈を選ぶ。

    logp [E, M, 文脈, V] は各候補の窓での各文脈の対数確率、answers [E, M] は情報源の答え（観測するまで使わない）。
    active=False は先頭 n_obs 個をそのまま観測する（batch）。active=True は1個ずつ、まだ観測していない候補のうち
    現在の事後分布のもとで期待情報量が最大のものを選んで観測する。source [E, M]（候補の窓を保存した文脈）を与えると、
    期待情報量にその文脈の現在の事後確率を掛けて選ぶ（修正1。信じている文脈の記憶ほど想起されやすい）。返り値は選んだ文脈 [E] と観測した手がかりの
    一様事前のもとでの情報量の平均 [E]。return_post=True では、観測後の対数事後分布（正規化前）[E, 文脈] も返す。"""
    e_n, m_n, c_n, v_n = logp.shape
    probs = (1 - EPS) * logp.exp() + EPS / v_n
    loglik = torch.log(probs.gather(-1, answers[:, :, None, None].expand(e_n, m_n, c_n, 1)).squeeze(-1))  # [E, M, C]
    info_uniform = batched_information(probs, torch.full((e_n, c_n), 1.0 / c_n))  # [E, M]
    if not active:
        post = loglik[:, :n_obs].sum(1)
        res = (post.argmax(-1), info_uniform[:, :n_obs].mean(-1))
        return res + (post,) if return_post else res
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
    res = (log_post.argmax(-1), (info_uniform * seen).sum(-1) / n_obs)
    return res + (log_post,) if return_post else res


def soft_ce(model: nn.Module, x: torch.Tensor, y: torch.Tensor, post: torch.Tensor, mode: str,
            chunk: int = 20) -> torch.Tensor:
    """事後分布 post [E, 文脈]（正規化済み）のもとで、文脈を1つに決めずに予測したときの、窓 x [N, k] の次の文字 y の
    CE [E]。mode="gain" は事後確率で重みづけたゲインで1回予測する（GainCharLM のみ）。mode="mix" は全文脈の予測を
    事後確率で混ぜる。"""
    out = []
    if mode == "gain":
        h1 = torch.tanh(model.inp(model.emb(x).flatten(1)))  # [N, H]
        g = post @ model.gains()  # [E, H]
        for i in range(0, post.shape[0], chunk):
            lp = model._logp(h1[None], g[i:i + chunk, None, :])  # [e, N, V]
            out.append(-lp.gather(-1, y[None, :, None].expand(lp.shape[0], -1, 1)).squeeze(-1).mean(1))
    elif mode == "mix":
        lp_y = model.logp_all(x).gather(-1, y[:, None, None].expand(-1, post.shape[1], 1)).squeeze(-1)  # [N, C]
        out.append(-torch.logsumexp(torch.log(post.clamp_min(1e-30))[:, None, :] + lp_y[None], -1).mean(1))
    else:
        raise ValueError("mode は gain・mix")
    return torch.cat(out)


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
    tcm_beta: float = 0.0  # > 0 で、保存する痕跡に時間文脈のタグをつける（12.6.179節）。乱数は使わない
    tcm_item: int = 2  # TCM の項目の文字の組の長さ（初期設定 2、12.6.179節の修正1 で 3）
    ca3_iters: int = 3  # CA3 のパターン補完の反復の回数（active_*_ca3 の規則。12.6.183節）
    ca3_sparsity: float = 0.025  # 補完で発火する記憶の単位の割合（k-WTA。痕跡の総数に対する割合）
    dg_expand: int = 4  # 歯状回の顆粒細胞の数（タグの次元の倍数。12.6.197節）
    dg_conn: float = 0.05  # 各顆粒細胞が結合する嗅内皮質の入力の割合
    dg_sparsity: float = 0.05  # 歯状回で発火する顆粒細胞の割合（k-WTA）。乱数射影の対照も同じ数の単位


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
    train_x, train_y, texts = [], [], []
    for j in range(n_l):
        text = sample_text(trans[j], fam, n_chains, text_len, generator)
        x, y = windows(text, cfg.k)
        train_x.append(x)
        train_y.append(y)
        texts.append(text)
    with torch.random.fork_rng(devices=[]):  # 初期化の種も generator から決める
        torch.manual_seed(int(torch.randint(0, 2 ** 31 - 1, (1,), generator=generator)))
        arch = {"head": ContextCharLM, "gain": GainCharLM}[cfg.arch]
        model = arch(fam.vocab, cfg.k, n_l, hidden=cfg.hidden)
        plain = arch(fam.vocab, cfg.k, 1, hidden=cfg.hidden)
    opt = torch.optim.Adam(list(model.parameters()) + list(plain.parameters()), lr=cfg.lr)
    buf_x, buf_y, hold_x, hold_y = [None] * n_l, [None] * n_l, [None] * n_l, [None] * n_l
    buf_tag, hold_tag = [None] * n_l, [None] * n_l

    def store(j: int) -> None:
        perm = torch.randperm(train_x[j].shape[0], generator=generator)[:2 * cfg.buffer]
        buf_x[j], buf_y[j] = train_x[j][perm[:cfg.buffer]], train_y[j][perm[:cfg.buffer]]
        hold_x[j], hold_y[j] = train_x[j][perm[cfg.buffer:]], train_y[j][perm[cfg.buffer:]]
        if cfg.tcm_beta > 0:  # 痕跡に、それを経験したときの時間文脈をタグとして結びつける
            tags = tcm_tags(texts[j], cfg.k, fam.vocab, cfg.tcm_beta, cfg.tcm_item)
            buf_tag[j], hold_tag[j] = tags[perm[:cfg.buffer]], tags[perm[cfg.buffer:]]

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
            "buf_tag": buf_tag, "hold_tag": hold_tag, "final_loss": sum(losses[-100:]) / 100}


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
        test_xy = []
        for j in range(n_l):
            x, y = windows(sample_text(trans[j], fam, test_chains, 200, generator), k)
            test_xy.append((x, y))
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
        tcm_gen = torch.Generator().manual_seed(generator.initial_seed() + 888)  # TCM の想起の答え専用
        ca3_gen = torch.Generator().manual_seed(generator.initial_seed() + 999)  # CA3 の補完の答え専用
        ach_gen = torch.Generator().manual_seed(generator.initial_seed() + 1111)  # ゲートつき補完の答え専用
        shuf_gen = torch.Generator().manual_seed(generator.initial_seed() + 1212)  # ach_shuf の並べ替え専用
        cmp_gen = torch.Generator().manual_seed(generator.initial_seed() + 1313)  # 比較器のゲートの答え専用
        cshuf_gen = torch.Generator().manual_seed(generator.initial_seed() + 1414)  # cmp_shuf の並べ替え専用
        dg_gen = torch.Generator().manual_seed(generator.initial_seed() + 1515)  # 歯状回の結合・答え専用
        tcm = [r for r in rules if r in TCM_RULES or r in CA3_RULES or r in ACH_RULES or r in CMP_RULES
               or r in DG_RULES]
        dg = [r for r in rules if r in DG_RULES]
        ca3 = [r for r in rules if r in CA3_RULES]
        ach = [r for r in rules if r in ACH_RULES]
        cmp = [r for r in rules if r in CMP_RULES]
        if tcm:
            if cfg.tcm_beta <= 0:
                raise ValueError("TCM の規則には TrainConfig.tcm_beta > 0 が要る")
            buf_src = torch.cat([torch.full((b.shape[0],), a) for a, b in enumerate(trained["buf_x"])])
            hold_src = torch.cat([torch.full((b.shape[0],), a) for a, b in enumerate(trained["hold_x"])])
            all_buf_tag, all_hold_tag = torch.cat(trained["buf_tag"]), torch.cat(trained["hold_tag"])
        if dg:  # 痕跡を歯状回の符号（と乱数射影の対照）で CA3 に保存する。結合は痕跡によらず1つ
            d_in = all_buf_tag.shape[1]
            w_dg = dg_weights(d_in, cfg.dg_expand * d_in, cfg.dg_conn, dg_gen)
            n_dg = max(1, round(cfg.dg_sparsity * w_dg.shape[0]))
            dg_bank = {"dg": {"replay": dg_codes(all_buf_tag, w_dg, n_dg),
                              "holdout": dg_codes(all_hold_tag, w_dg, n_dg)},
                       "rp": {"replay": rp_codes(all_buf_tag, w_dg, n_dg, cfg.dg_conn),
                              "holdout": rp_codes(all_hold_tag, w_dg, n_dg, cfg.dg_conn)}}
            # 操作の確認: 言語どうしの符号の重なり（同じ痕跡の標本で比べる）
            out["dg_overlap"] = {name: code_overlap(c, buf_src, 600, torch.Generator().manual_seed(
                generator.initial_seed() + 1616)) for name, c in
                (("raw", nn.functional.normalize(all_buf_tag, dim=-1)), ("dg", dg_bank["dg"]["replay"]),
                 ("rp", dg_bank["rp"]["replay"]))}
        all_buf = torch.cat(trained["buf_x"])
        all_hold = torch.cat(trained["hold_x"])
        res: dict = {}
        for n in cue_ns:
            m = pool * n
            acc = {r: 0.0 for r in rules}
            ce_sel = {r: 0.0 for r in rules}
            info = {r: 0.0 for r in rules}
            per_lang = {r: [] for r in rules}
            purity = {"replay_tcm": 0.0, "holdout_tcm": 0.0}
            purity.update({r.removeprefix("active_"): 0.0 for r in ca3 + ach + cmp + dg})
            lam_mean = {r.removeprefix("active_"): 0.0 for r in ach + cmp + [d for d in dg if d.endswith("cmp")]}
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
                if tcm:  # 情報源が示したラベルなしの窓（active の候補と同じ）で今の文脈を作り、近い痕跡を想起する
                    for key, bank, tags, srcs in (("replay_tcm", all_buf, all_buf_tag, buf_src),
                                                  ("holdout_tcm", all_hold, all_hold_tag, hold_src)):
                        idx = tcm_recall(task_x, tags, v, cfg.tcm_beta, m, cfg.tcm_item)
                        pools[key] = bank[idx]
                        ans[key] = answers(pools[key], tcm_gen)
                        purity[key] += float((srcs[idx] == j).float().mean()) / n_l
                for r in ca3:  # 今の文脈を CA3 のパターン補完で補完してから想起する（12.6.183節）
                    key = r.removeprefix("active_")
                    bank, tags, srcs = ((all_buf, all_buf_tag, buf_src) if key.startswith("replay")
                                        else (all_hold, all_hold_tag, hold_src))
                    iters = int(key.rsplit("_s", 1)[1]) if "_s" in key else cfg.ca3_iters
                    n_act = max(1, round(cfg.ca3_sparsity * tags.shape[0]))
                    idx = tcm_recall(task_x, tags, v, cfg.tcm_beta, m, cfg.tcm_item, iters, n_act)
                    pools[key] = bank[idx]
                    ans[key] = answers(pools[key], ca3_gen)
                    purity[key] += float((srcs[idx] == j).float().mean()) / n_l
                for r in ach:  # 補完の強さを想起の不確かさでゲートしてから想起する（12.6.187節）
                    key = r.removeprefix("active_")
                    bank, tags, srcs = ((all_buf, all_buf_tag, buf_src) if key.startswith("replay")
                                        else (all_hold, all_hold_tag, hold_src))
                    n_act = max(1, round(cfg.ca3_sparsity * tags.shape[0]))
                    # 不確かさは、補完の前に今の文脈で最も強く発火する n_act 個（k-WTA の集団）の保存時の文脈の分布から
                    u = recall_uncertainty(tcm_recall(task_x, tags, v, cfg.tcm_beta, n_act, cfg.tcm_item), srcs, n_l)
                    lam = (u > ACH_THRESHOLD).float() if key.endswith("_hard") else 1 - u if key.endswith("_inv") else u
                    if key.endswith("_shuf"):
                        lam = lam[torch.randperm(lam.shape[0], generator=shuf_gen)]
                    idx = tcm_recall(task_x, tags, v, cfg.tcm_beta, m, cfg.tcm_item, cfg.ca3_iters, n_act, lam)
                    pools[key] = bank[idx]
                    ans[key] = answers(pools[key], ach_gen)
                    purity[key] += float((srcs[idx] == j).float().mean()) / n_l
                    lam_mean[key] += float(lam.mean()) / n_l
                for r in cmp:  # 補完の強さを CA1 の比較器の食い違いで決めてから想起する（12.6.191節）
                    key = r.removeprefix("active_")
                    bank, tags, srcs = ((all_buf, all_buf_tag, buf_src) if key.startswith("replay")
                                        else (all_hold, all_hold_tag, hold_src))
                    n_act = max(1, round(cfg.ca3_sparsity * tags.shape[0]))
                    t_now = tcm_now(task_x, v, cfg.tcm_beta, cfg.tcm_item)
                    lam = 1 - ca1_mismatch(t_now, tags, cfg.ca3_iters, n_act)
                    if key.endswith("_u"):
                        lam = lam * recall_uncertainty(
                            tcm_recall(task_x, tags, v, cfg.tcm_beta, n_act, cfg.tcm_item), srcs, n_l)
                    if key.endswith("_shuf"):
                        lam = lam[torch.randperm(lam.shape[0], generator=cshuf_gen)]
                    idx = tcm_recall(task_x, tags, v, cfg.tcm_beta, m, cfg.tcm_item, cfg.ca3_iters, n_act, lam)
                    pools[key] = bank[idx]
                    ans[key] = answers(pools[key], cmp_gen)
                    purity[key] += float((srcs[idx] == j).float().mean()) / n_l
                    lam_mean[key] += float(lam.mean()) / n_l
                for r in dg:  # 歯状回の符号（rp は乱数射影の対照）で補完してから想起する（12.6.197節）
                    key = r.removeprefix("active_")
                    bank_name, code = key.split("_")
                    gated = code.endswith("cmp")
                    code = code.removesuffix("cmp")
                    bank, tags, srcs = ((all_buf, all_buf_tag, buf_src) if bank_name == "replay"
                                        else (all_hold, all_hold_tag, hold_src))
                    n_act = max(1, round(cfg.ca3_sparsity * tags.shape[0]))
                    t_now = tcm_now(task_x, v, cfg.tcm_beta, cfg.tcm_item)
                    codes = dg_bank[code][bank_name]
                    idx, delta = ca3_recall_codes(t_now, tags, codes, cfg.ca3_iters, n_act, m)
                    if gated:  # 比較器のゲート λ = 1 − δ（δ は同じ符号での λ = 1 の補完から）
                        lam = 1 - delta
                        idx, _ = ca3_recall_codes(t_now, tags, codes, cfg.ca3_iters, n_act, m, lam)
                        lam_mean[key] += float(lam.mean()) / n_l
                    pools[key] = bank[idx]
                    ans[key] = answers(pools[key], dg_gen)
                    purity[key] += float((srcs[idx] == j).float().mean()) / n_l
                ans["passage"] = py_.reshape(episodes, n)
                logp = {key: model.logp_all(px.reshape(-1, k)).reshape(episodes, px.shape[1], n_l, v)
                        for key, px in pools.items()}
                spec = {"passive": ("passage", False), "uniform": ("task", False), "active": ("task", True),
                        "uniform_replay": ("replay", False),
                        "active_replay": ("replay", True), "active_holdout": ("holdout", True),
                        "active_replay_cued": ("replay_cued", True), "active_holdout_cued": ("holdout_cued", True),
                        "active_replay_own": ("replay_own", True),
                        "active_replay_tcm": ("replay_tcm", True), "active_holdout_tcm": ("holdout_tcm", True),
                        **{r: (r.removeprefix("active_"), True) for r in CA3_RULES + ACH_RULES + CMP_RULES + DG_RULES}}
                soft = [r for r in rules if r in SOFT_RULES]
                for r in rules:
                    if r in SOFT_RULES:
                        continue
                    key, act = spec[r]
                    sel = select_contexts(logp[key], ans[key], n, act, src if key.endswith("_cued") else None,
                                          return_post=bool(soft) and r == "active")
                    chosen, inf = sel[0], sel[1]
                    if len(sel) == 3:
                        post_active, inf_active = torch.softmax(sel[2], -1), inf
                    a = float((chosen == j).float().mean())
                    acc[r] += a / n_l
                    ce_sel[r] += float(ce[chosen, j].mean()) / n_l
                    info[r] += float(inf.mean()) / n_l
                    per_lang[r].append(a)
                for r in soft:  # 選ぶ文脈（選択精度）と情報量は active と同じ。CE だけ事後分布で予測する
                    if "active" not in rules:
                        raise ValueError("active_soft・active_mix には active が要る")
                    cs = soft_ce(model, *test_xy[j], post_active, "gain" if r == "active_soft" else "mix")
                    acc[r] += per_lang["active"][-1] / n_l
                    ce_sel[r] += float(cs.mean()) / n_l
                    info[r] += float(inf_active.mean()) / n_l
                    per_lang[r].append(per_lang["active"][-1])
            res[str(n)] = {"acc": acc, "ce": ce_sel, "info": info, "acc_per_lang": per_lang}
            if tcm:  # 想起した痕跡のうち本来の文脈のものの割合（操作の確認）
                res[str(n)]["tcm_purity"] = purity
            if ach or cmp or dg:  # ゲートの強さ λ の平均（操作の確認）
                res[str(n)]["ach_lambda"] = lam_mean
        out["cues"] = res
    return out

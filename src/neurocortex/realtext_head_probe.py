"""胴体とヘッドのずれの診断（12.6.253〜254節）: 実言語の継続学習の忘却を「胴体の情報の喪失」と「読み出しのずれ」に分ける。

12.6.240節で、naive では古い言語の文が新しい言語のヘッドに割り当てられ、忘却の一部が胴体とヘッドのずれに見えた。
表象のドリフト（Driscoll et al. 2017; Rule et al. 2020）の見方では、上流の表現が動いても情報は残り、下流の読み出しが
追従すれば取り戻せる。ここでは学習の各時点のモデルのスナップショットで、胴体（共有の Transformer と中間層）を凍結し、
言語 j の文脈の読み出し（文脈のバイアス b_j・ヘッド W_j・a_j）だけを言語 j の学習データで学び直す（読み出しの
プローブ）。言語 j の学習の前（start）・直後（diag）・全言語の後（final）の3時点で、

- F = CE_final − CE_diag（忘却。正が忘れた量）
- L = CE*_final − CE*_diag（プローブ後に残る忘却 = 胴体の情報の喪失。* はプローブ後）
- M = F − L = (CE_final − CE*_final) − (CE_diag − CE*_diag)（読み出しのずれ。学び直しで取り戻せる分）
- I = (CE*_start − CE*_final) / (CE*_start − CE*_diag)（保持の指標。言語 j を学んで胴体に入った分のうち最後に残る割合。
  CE*_start は言語 j を学ぶ前の胴体の上のプローブ = 他の言語からの転移だけで読める量）

と、胴体の特徴の動き（diag と final の線形 CKA）を測る。
"""

from __future__ import annotations

import copy

import torch
from torch import nn

from .context_realtext import LangData, TransformerContextLM, _passages
from .realtext_forgetting import ForgetConfig, eval_segments, train_forgetting

PROBE_STEPS = (500, 1000, 2000)  # プローブの CE を記録するステップ（最後が主の値。12.6.253節で計測前に固定）
PROBE_BATCH = 64
PROBE_LR = 1e-2
PROBE_POOL = 4000  # プローブの学習データの断片の数（胴体は凍結なので特徴を一度だけ求める）


def readout_ce(model: TransformerContextLM, seg: torch.Tensor, c: int) -> float:
    """断片 seg [N, T + 1] を文脈 c の読み出しで読んだ次の文字の CE（nats、全位置の平均）。"""
    with torch.no_grad():
        lp = model.seq_logp(seg[:, :-1], torch.full((seg.shape[0],), c))
        return float(nn.functional.nll_loss(lp.reshape(-1, lp.shape[-1]), seg[:, 1:].reshape(-1)))


def probe_readout(model: TransformerContextLM, flat: torch.Tensor, test: torch.Tensor, c: int, t_len: int,
                  generator: torch.Generator, steps: tuple[int, ...] = PROBE_STEPS, batch: int = PROBE_BATCH,
                  lr: float = PROBE_LR, pool: int = PROBE_POOL) -> list[float]:
    """胴体を凍結し、文脈 c の読み出し（ctx[c]・head_w[c]・head_b[c]）だけを flat（学習データの文字列）で学び直す。

    flat から無作為な断片を pool 個取り、その胴体の特徴を一度だけ求めて、そこからバッチを引く。
    model は変えない（複製を学ぶ）。steps の各時点の test [N, T + 1] の CE を返す。"""
    m = copy.deepcopy(model).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    ctx = m.ctx.weight[c].detach().clone().requires_grad_(True)
    w = m.head_w[c].detach().clone().requires_grad_(True)
    b = m.head_b[c].detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([ctx, w, b], lr=lr)

    def feats(seg: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.cat([m.trunk(seg[i:i + 500, :-1]) for i in range(0, seg.shape[0], 500)])

    def ce(h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        lp = torch.log_softmax(torch.tanh(h + ctx) @ w + b, -1)
        return nn.functional.nll_loss(lp.reshape(-1, lp.shape[-1]), y.reshape(-1))

    s = torch.randint(0, flat.shape[0] - t_len - 1, (pool,), generator=generator)
    tr = flat[s[:, None] + torch.arange(t_len + 1)[None]]
    h_tr, y_tr = feats(tr), tr[:, 1:]
    h_te, y_te = feats(test), test[:, 1:]
    out = []
    for k in range(1, max(steps) + 1):
        i = torch.randint(0, pool, (batch,), generator=generator)
        opt.zero_grad()
        ce(h_tr[i], y_tr[i]).backward()
        opt.step()
        if k in steps:
            with torch.no_grad():
                out.append(float(ce(h_te, y_te)))
    return out


def trunk_features(model: TransformerContextLM, seg: torch.Tensor) -> torch.Tensor:
    """胴体の特徴（文脈を足す前の中間層）[N·T, hidden]。"""
    with torch.no_grad():
        return model.trunk(seg[:, :-1]).reshape(-1, model.mid.out_features)


def linear_cka(a: torch.Tensor, b: torch.Tensor) -> float:
    """線形 CKA（Kornblith et al. 2019）。a・b は [n, d]（同じ n）。1 なら表現が線形変換と等方な拡大の範囲で同じ。"""
    a = (a - a.mean(0)).double()
    b = (b - b.mean(0)).double()
    num = (a.T @ b).pow(2).sum()
    den = (a.T @ a).pow(2).sum().sqrt() * (b.T @ b).pow(2).sum().sqrt()
    return float(num / den) if den > 0 else 0.0


def components(ce: dict, probe: dict) -> dict:
    """1つの言語の CE（読み出しのまま: diag・final）とプローブ後の CE（start・diag・final）から F・L・M・I。"""
    f = ce["final"] - ce["diag"]
    loss = probe["final"] - probe["diag"]
    gain = probe["start"] - probe["diag"]
    return {"F": f, "L": loss, "M": f - loss, "I": (probe["start"] - probe["final"]) / gain if gain > 0 else None,
            "gain_diag": ce["diag"] - probe["diag"], "gain_final": ce["final"] - probe["final"], "learned": gain}


def train_and_probe(data: list[LangData], vocab: int, cfg: ForgetConfig, order: list[int],
                    generator: torch.Generator, eval_generator: torch.Generator,
                    probe_generator: torch.Generator, probe_segments: int = 200,
                    probe_kw: dict | None = None) -> dict:
    """train_forgetting で言語を順に学び、各言語の学習の前後のスナップショットを取り、最後に読み出しのプローブで分解する。

    プローブの評価は各言語のテスト文の断片 probe_segments 個（評価の乱数で別に取る）。学習そのものは train_forgetting と
    同じ（同じシードなら R 行列も同じ）。"""
    snaps: dict[tuple[str, int], TransformerContextLM] = {}

    def hook(event: str, pos: int, model: TransformerContextLM) -> None:
        snaps[(event, pos)] = copy.deepcopy(model).eval()

    out = train_forgetting(data, vocab, cfg, order, generator, eval_generator, hook=hook)
    n = len(order)
    final = snaps[("end", n - 1)]
    flat = [_passages(d.train, 400).reshape(-1) for d in data]
    pg = torch.Generator().manual_seed(int(torch.randint(0, 2 ** 31 - 1, (1,), generator=eval_generator)))
    tests = eval_segments(data, probe_segments, cfg.seq_len, pg)
    langs = []
    for pos, j in enumerate(order):
        c = 0 if cfg.shared_head else j
        rec: dict = {"pos": pos, "lang": j, "ce": {}, "probe_curve": {}}
        for name, m in (("start", snaps[("start", pos)]), ("diag", snaps[("end", pos)]), ("final", final)):
            if name != "start":
                rec["ce"][name] = readout_ce(m, tests[j], c)
            rec["probe_curve"][name] = probe_readout(m, flat[j], tests[j], c, cfg.seq_len, probe_generator,
                                                       **(probe_kw or {}))
        rec["probe"] = {k: v[-1] for k, v in rec["probe_curve"].items()}
        rec["cka_diag_final"] = linear_cka(trunk_features(snaps[("end", pos)], tests[j]),
                                           trunk_features(final, tests[j]))
        rec.update(components(rec["ce"], rec["probe"]))
        langs.append(rec)
    out["probe"] = {"langs": langs, "steps": list((probe_kw or {}).get("steps", PROBE_STEPS)),
                    "segments": probe_segments}
    return out

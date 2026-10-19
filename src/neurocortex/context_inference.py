"""文脈推定機構（12.6.105節）。タスク ID を与えずに、予測誤差の驚きから文脈（潜在原因）を推定し、
推定した文脈ごとにリプレイする。

  oracle   … 真のタスク ID を文脈として入力する（上限）
  none     … 文脈を入力しない（下限）
  inferred … 驚き（確立済みの文脈での損失が閾値 τ を超えたこと）を合図に、既存の文脈から最も合うものに
             切り替えるか、合うものがなければ新しい文脈を割り当てる（Gershman, Blei & Niv 2010 の潜在原因推論）
  random   … 文脈スロットを毎回ランダムに選ぶ（統制。文脈が情報を持たない場合）

テスト時、inferred は各タスクの少数のラベル付き手がかりを最もよく説明する文脈を選ぶ。
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F

from dataclasses import dataclass

from .basal_ganglia_core import ParitySpec
from .basal_ganglia_core import make_batch as _parity_batch
from .hippocampus_replay import make_model

CONDITIONS = ("oracle", "none", "inferred", "random", "inferred_gated")


@dataclass(frozen=True)
class FlipSpec:
    """似たタスク（12.6.108節）。正解は b[a] XOR b[b] を基本とし、flip の全ビットが +1 の領域でだけ反転する。
    反転領域の割合は 2^-len(flip)。flip が空なら基本タスクそのもの。"""

    n_inputs: int = 8
    relevant: tuple[int, int] = (0, 1)
    flip: tuple[int, ...] = ()


@dataclass(frozen=True)
class CueSpec:
    """入力分布がタスクで異なる課題（12.6.111節）。正解は b[a] XOR b[b]（最初の n_parity ビット）。
    残りの手がかりビットのうち cue に含まれるビットは確率 q で +1、他は一様。"""

    n_inputs: int = 16
    relevant: tuple[int, int] = (0, 1)
    cue: tuple[int, ...] = ()
    q: float = 0.9


def make_batch(spec, batch_size: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """ParitySpec はステップ10と同じ乱数消費で、FlipSpec は同じ入力に反転領域を適用して生成する。"""
    if isinstance(spec, ParitySpec):
        return _parity_batch(spec, batch_size, generator)
    bits = torch.randint(0, 2, (batch_size, spec.n_inputs), generator=generator).float() * 2 - 1
    if isinstance(spec, CueSpec) and spec.cue:
        u = torch.rand(batch_size, len(spec.cue), generator=generator)
        bits[:, list(spec.cue)] = torch.where(u < spec.q, 1.0, -1.0)
    return bits, task_labels(spec, bits)


def task_labels(spec, bits: torch.Tensor) -> torch.Tensor:
    """入力 bits に対する spec の正解（make_batch と同じ規則。乱数は消費しない）。12.6.147節の自己生成の候補に
    環境が答える（問い合わせの正解を返す）ために使う。"""
    a, b = spec.relevant
    labels = (bits[:, a] > 0) ^ (bits[:, b] > 0)
    if isinstance(spec, FlipSpec) and spec.flip:
        labels = labels ^ (bits[:, list(spec.flip)] > 0).all(dim=1)
    return labels.long()


def cue_information(probs: torch.Tensor, prior: torch.Tensor | None = None) -> torch.Tensor:
    """各候補の手がかりについて、文脈と正解の相互情報量 H(Σπ_c p_c) − Σπ_c H(p_c) を log K で正規化して返す
    （12.6.143節）。probs は [候補, 文脈, K]、prior は文脈の事前分布 π（既定は一様）。"""
    pi = torch.full((probs.shape[1],), 1.0 / probs.shape[1]) if prior is None else prior
    ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum(-1)  # [候補, 文脈]
    mix = (pi[None, :, None] * probs).sum(1)
    h_mix = -(mix * torch.log(mix.clamp_min(1e-12))).sum(-1)
    return ((h_mix - (pi[None] * ent).sum(1)) / math.log(probs.shape[-1])).clamp_min(0.0)


EPISODIC_READOUTS = ("param", "mix_all", "switch_all", "mix_buf", "mix_hold", "mix_nogate", "mix_all200",
                     "switch_all200")
EPISODIC_EXTRA_READOUTS = ("mix_rep", "switch_rep", "arb_bits", "arb_rep")  # 12.6.159節
RECON_READOUTS = ("recon_buf", "recon_all", "recon_all200", "recon_inter", "recon_hold")  # 12.6.159節
RECON_EXTRA_READOUTS = ("recon_buf_inter", "recon_buf_err")  # 12.6.161節（バッファでの学び直しが効く理由の切り分け）
WRITEBACKS = ("none", "all", "inter")  # 12.6.161節（学び直しの共有の重みへの書き戻し）
REPLAY_RULES = ("gain", "need", "sleep", "ach", "ach_need", "ach_need2")  # 12.6.163節（海馬の痕跡のリプレイの配分）


def episodic_readout(p_param: torch.Tensor, x: torch.Tensor, store_bits: torch.Tensor, store_labels: torch.Tensor,
                     lam: float = 0.5, kappa: float = 1.0, rule: str = "mix") -> torch.Tensor:
    """記憶に基づく予測と、選んだ文脈のパラメトリックな予測の組み合わせ（12.6.155節）。

    p_param は [N, K] の予測分布、x は [N, n] の ±1 入力、store_* は検索する痕跡（入力とその保存時のラベル）。
    痕跡 i との類似度 s_i = exp(−d_i / λ)（d_i はハミング距離）、記憶の予測 p_epi = Σ s_i onehot(y_i) / Σ s_i、
    記憶の重み w = Σ s_i / (Σ s_i + κ)（κ はパラメトリックな予測への擬似的な数）。
      "mix"    … (1 − w)·p_param + w·p_epi（痕跡との近さで重み付けした混合）
      "switch" … w ≥ 0.5 なら p_epi、そうでなければ p_param（記憶の信頼度による切り替え）
    痕跡がなければ p_param をそのまま返す。"""
    if rule not in ("mix", "switch"):
        raise ValueError(rule)
    if lam <= 0 or kappa <= 0:
        raise ValueError("lam と kappa は正")
    if store_bits.shape[0] == 0:
        return p_param
    d = (x.shape[1] - x @ store_bits.T) / 2  # [N, 痕跡]
    return episodic_readout_dist(p_param, d, store_labels, lam, kappa, rule)


def episodic_readout_dist(p_param: torch.Tensor, d: torch.Tensor, store_labels: torch.Tensor, lam: float = 0.5,
                          kappa: float = 1.0, rule: str = "mix") -> torch.Tensor:
    """episodic_readout の距離を与える版（12.6.159節）。d は [N, 痕跡] の距離（ハミング距離と同じ尺度）。"""
    if rule not in ("mix", "switch"):
        raise ValueError(rule)
    if lam <= 0 or kappa <= 0:
        raise ValueError("lam と kappa は正")
    if d.shape[1] == 0:
        return p_param
    s = torch.exp(-d / lam)
    onehot = F.one_hot(store_labels, p_param.shape[1]).float()
    mass = s.sum(1, keepdim=True)
    p_epi = (s @ onehot) / mass.clamp_min(1e-30)
    w = mass / (mass + kappa)
    if rule == "mix":
        return (1 - w) * p_param + w * p_epi
    return torch.where(w >= 0.5, p_epi, p_param)


def run_context_sequence(condition: str, specs: list, hidden_dim: int, generator: torch.Generator,
                         eval_generator: torch.Generator, n_slots: int | None = None, tau: float = 0.4,
                         ema: float = 0.9, pretrain_steps: int = 500, pretrain_weight_decay: float = 0.01,
                         steps: int = 500, batch: int = 64, lr: float = 0.05, buffer_size: int = 50,
                         cue_n: int = 16, eval_n: int = 2000,
                         selection_trials: int = 0, transition_steps: int = 0,
                         novel_patience: int = 20, store_consistent: bool = False, train_consistent: bool = False,
                         diagnostics: bool = False, trace_every: int = 0, diag_oracle_filter: str = "",
                         reset_opt_on_new: bool = False, plasticity_cap: float = 0.0, plasticity_ema: float = 0.99,
                         plasticity_ref: str = "capped", optimizer: str = "adam", split_window: int = 0,
                         split_ctx_gain: float = 1.0, split_shared_gain: float = 1.0,
                         split_shared_lr_floor: float = 0.0, split_ctx_lr: float = 0.0,
                         cue_select: str = "batch", evidence_lapse: float = 0.1, evidence_alpha: float = 0.01,
                         evidence_reliability: bool = False, evidence_com: bool = False, com_alpha: float = 0.1,
                         com_window: int = 0, reliability_lapse: bool = True,
                         reliability_source: str = "gated", reliability_holdout: int = 0,
                         cue_weight: str = "", active_pool: int = 0, active_source: str = "task",
                         active_verify: int = 0, verify_mode: str = "top2", episodic_eval: int = 0,
                         episodic_hold: int = 50, episodic_lambda: float = 0.5, episodic_kappa: float = 1.0,
                         episodic_extra: bool = False, recon_steps: int = 0, recon_lr: float = 0.01,
                         recon_batch: int = 32, recon_extra: bool = False, writeback: str = "",
                         writeback_at: int = 0, writeback_steps: int = 100, writeback_eval: int = 2000,
                         writeback_keep: bool = False, replay_rule: str = "", replay_beta: float = 0.5,
                         replay_need: float = 0.5, sleep_every: int = 50, sleep_steps: int = 25) -> dict:
    """戻り値: {"R": 正確度行列, "assign": 学習中の (タスク, 文脈) の列, "n_contexts": 使った文脈数,
    "selection_acc": 文脈の選択精度（inferred かつ selection_trials > 0 のとき）}

    selection_acc は最終モデルで、cue_n 個の手がかりから選んだ文脈が、そのタスクに学習中最も多く割り当てられた
    文脈と一致する割合（タスクごとに selection_trials 回、評価用の乱数で試行）。12.6.107節。

    transition_steps = L > 0 では、タスク k（k ≥ 1）の最初の L ステップを移行期間とし、各サンプルを確率
    p = (s+1)/(L+1) でタスク k から、残りをタスク k−1 から取る（少しずつ変わる文脈, 12.6.109節）。oracle は
    サンプルごとに本当の出どころのタスクを文脈として受け取る。L = 0 では以前と同じ乱数消費になる。

    store_consistent = True（inferred_gated のみ）では、保存する文脈で正しく予測できるサンプルだけをバッファに
    入れる（12.6.116節）。train_consistent = True（inferred_gated のみ）では、確立した文脈で驚きのなかったバッチ
    のうち、その文脈で正しく予測できるサンプルだけで学習する。どちらも既定の False では 12.6.115節と同じ挙動。

    diagnostics = True では原因の切り分け用（12.6.116節）に "diag" を加える。学習・評価の乱数は消費しない
    （診断専用の乱数を使う）ので、R と assign は diagnostics = False と同じになる。
      ctx_acc[i][j][c] … タスク i の学習後、タスク j を文脈 c で評価した正確度（used の順）
      chosen[i][j]     … 同じ時点で R の計算が手がかりから選んだ文脈
      buffer_src[c]    … 文脈 c のバッファのサンプルの本当の出どころのタスクごとの数
      buffer           … バッファの中身（bits, labels, ctx, src のテンソル）
      trained_src[i]   … タスク i の期間に、新しい入力として学習に使ったサンプルの (文脈, 出どころ) ごとの数
      trace            … trace_every > 0 のとき、trace_every ステップごとの (タスク, ステップ, 現在の文脈, 学習したか,
                         保存したか, 各タスク j を各文脈で評価した正確度) の列（12.6.121節）

    reset_opt_on_new = True では、新しい文脈を作った時点で最適化器（Adam）の状態を作り直す（12.6.121節）。
    作成の直前はリプレイだけの学習で勾配が極小になり、2次モーメントの推定が縮んでいるため、作成直後の大きな
    勾配で更新が過大になり、前の文脈を壊す。既定の False では 12.6.116節と同じ挙動。

    diag_oracle_filter は原因の切り分け専用の反実仮想（12.6.121節。機構ではない）。本当の出どころを使い、
    "new" ではタスク k の期間に作った文脈の学習から前のタスクのサンプルを、"old" ではそれ以外の文脈の学習から
    タスク k のサンプルを除く（"both" は両方）。
    diagnostics = True では "upd"（各最適化ステップの (タスク, ステップ, 文脈, 勾配ノルム, 更新ノルム)）も加える。
    更新ノルムは plasticity_cap による縮小の後の値。

    plasticity_cap = ρ > 0 では、驚きに応じた可塑性の調節（12.6.125節。reset_opt_on_new の代わり）を行う。
    新しい入力を学習したステップの更新ノルムの指数移動平均 ū（減衰 plasticity_ema、平常時の可塑性の大きさ）を
    保持し、更新ノルムが ρ·ū を超える（平常時から予期せぬほど大きい）ときは、更新を ρ·ū に縮める。ū の更新には
    縮めた後の値を使うので、持続的な変化には1ステップあたり最大 (ρ − 1)(1 − plasticity_ema) の割合で追従する。
    最適化器の内部状態は変えない。既定の 0 では従来と同じ挙動。plasticity_ref = "raw" では、ū の更新に縮める前の
    値を使う（修正2。縮めた値を使うと ū が自分の縮小で下がり続け、平常時の可塑性まで抑えるため）。

    optimizer は "adam"（既定）か "sgd"（モーメントなし。Adam 以外でも同じ問題が起きるかの確認用, 12.6.125節）。

    split_window = W > 0（inferred / inferred_gated のみ）では、文脈固有と共有で可塑性を分ける（12.6.129節）。
    新しい文脈を作った時点から W ステップの間、各最適化ステップの更新（パラメータの変化）に利得をかける。
    新しい文脈の one-hot 入力からの重み（fc1.weight の該当列）には split_ctx_gain、それ以外（共有の重みと
    既存の文脈の列）には split_shared_gain。最適化器の内部状態は変えない。既定の 0 では従来と同じ挙動。

    学習率への依存の改善（12.6.132節。どちらも既定の 0 では無効）:
      split_shared_lr_floor = f > 0 … 共有の重みの実効学習率 lr·g_shared に下限 min(f, lr) を設ける。利得は
          max(split_shared_gain, min(1, f / lr)) になる（土台の学習率より大きくはしない）。
      split_ctx_lr = η > 0 … 新しい文脈の列の利得を、土台の学習率に対する相対値ではなく実効学習率 η で決める
          （利得 η / lr。split_ctx_gain の代わり）。

    cue_select はテスト時の文脈の選択の規則（inferred / inferred_gated のみ。12.6.135節）。
      "batch"（既定）… cue_n 個の手がかりの平均損失が最小の文脈を1回で選ぶ（従来と同じ）。
      "evidence" … 手がかりを1個ずつ観測し、文脈ごとの対数尤度 log((1 − ε)·p + ε/K)（ε = evidence_lapse,
          K はクラス数）を積み上げる（一様な事前分布）。最大の事後確率が 1 − evidence_alpha に達した時点で決め、
          達しなければ cue_n 個を使い切った時点で事後確率が最大の文脈に決める（逐次確率比検定, Gold & Shadlen 2007）。
      どちらも手がかりの生成（乱数の消費）は同じ。diagnostics = True では "cues_used"（chosen と同じ形の、決定まで
      に使った手がかりの数）を加える。selection_trials > 0 では "selection_choices"（タスクごとの試行ごとの選択）と
      "selection_cues"（同じく手がかりの数）も返す。

    証拠の蓄積の変種（cue_select="evidence" のみ。12.6.139節。どちらも既定の False では無効）:
      evidence_reliability = True … 各文脈 c の信頼性 a_c（学習中の正確度の移動平均と、評価時点のバッファの整合度の
          小さい方）から、lapse を ε_c = max(ε, K(1 − a_c)/(K − 1))、先頭の文脈 c* に決める閾値を事後対数オッズ
          log((1 − α)/α) / max(2a_{c*} − 1, 0.1) にする（信頼性の低い文脈ほど多くの証拠を要する）。
      evidence_com = True … 最初の決定の後も手がかりを最大 com_window 個（0 は cue_n 個まで）観測し続け、他の文脈の
          最大が現在の選択を対数オッズ log((1 − com_alpha)/com_alpha) 上回ったら乗り換える（change of mind,
          Resulaj et al. 2009）。最終の選択を使う。
      reliability_lapse = False … 信頼性を閾値だけに使い、lapse は全文脈で共通の ε にする（12.6.139節の修正1。
          文脈ごとに ε_c が違うと、平らにされた文脈が誤っても罰されにくく、選択が信頼性の低い文脈に偏るため）。
      reliability_source = "pre" … 学習中の正確度を、ゲート・更新の前に、その文脈を選んだ全バッチで測る（ただし
          新しい原因の判定と同じく、正答率 0.75 未満のバッチは除く）。既定の "gated" は、学習・保存したバッチの
          更新後の正確度（損失 ≦ τ のバッチに偏るため高めに出る）。
      reliability_holdout = m > 0 … バッファの整合度の代わりに、リプレイに使わない保留サンプル（文脈ごとに m 個まで。
          保存するバッチのうちバッファに入れなかった残りから、同じ整合フィルタで取る）の整合度を使う（12.6.139節の
          修正2。リプレイはバッファのサンプルそのものを学習し続けるので、バッファの整合度には忘却が現れない）。
          学習・乱数の消費には影響しない。
      cues_used / selection_cues は最初の決定までの手がかりの数。diagnostics = True では "cues_observed"（観測した
      総数）・"switches"（乗り換えの回数）・"first_choice"（最初の選択）を、evidence_reliability では "reliability"
      （評価時点ごとの {文脈: a_c}）も加える。selection_trials > 0 では "selection_observed"・"selection_switches"・
      "selection_first" も返す。

    手がかりの情報量（cue_select="evidence" のみ。12.6.143節。どちらも既定では無効）:
      cue_weight = "info" … 手がかり i の証拠（各文脈の対数尤度）に、一様な事前分布のもとでの文脈と正解の相互情報量
          I_i = H(平均の予測) − 平均の H(予測)（log K で正規化。文脈の間の予測の不一致）を重みとして掛ける
          （Yang & Shadlen 2007 の、手がかりの診断性で重み付けた対数尤度比の蓄積）。全文脈が同じ予測をする手がかりは
          確信度が違っても重みがほぼ 0 になる。
      cue_weight = "info_post" … 重みを、その手がかりを観測する直前の事後分布のもとでの相互情報量にする（12.6.143節の
          修正1。一様な事前分布では、忘れた文脈の雑音の予測でどの手がかりも不一致が大きく見え、重みが区別しないため）。
      active_pool = P > 0 … 手がかりを P·cue_n 個のラベルなしの候補（先頭 cue_n 個は従来の手がかり、残りは評価用の
          乱数を消費しない専用の乱数で生成）から、現在の事後分布のもとでの期待情報量 H(Σπ_c p_c) − Σπ_c H(p_c) が最大の
          ものを1個ずつ選んで正解を観測する（能動的知覚、Yang, Lengyel & Wolpert 2016）。観測するのは最大 cue_n 個で、
          打ち切りの規則は evidence と同じ。
      diagnostics = True では "cue_info"（観測した手がかりの一様事前のもとでの情報量の平均）を、selection_trials > 0
      では "selection_info" も返す。

    候補の出どころと打ち切り（active_pool > 0 のみ。12.6.147節。既定では 12.6.143節と同じ挙動）:
      active_source = "task"（既定）… 候補は与えられた手がかりと、同じタスクの分布からの追加の入力（12.6.143節）。
          "uniform" … 候補 P·cue_n 個を入力空間から一様に自分で生成し、選んだ入力の正解を環境に問い合わせる
          （task_labels）。タスクの入力分布は使わない。
          "replay" … 候補 P·cue_n 個を海馬のバッファ（学習中に保存した過去の入力）から取り出す（足りなければ全部）。
          "search" … "uniform" の候補から情報量が最大のものを選び、1ビットずつ反転して情報量が増える限り登る
          （山登り、最大 n_inputs 回）。既に観測した入力には移らず、観測した入力と同じ候補は選ばない。
          どの出どころでも、評価用の乱数の消費（従来の cue_n 個の手がかり）は同じにする。
      active_verify = n > 0 … 打ち切りの閾値に達したら、先頭の文脈と2番目の文脈の2つだけを等しい重みで区別する
          手がかり（先頭の仮説を最も反証しうる手がかり）を n 個観測してから決める。その間に閾値を割るか先頭が
          入れ替われば、通常の能動的な蓄積に戻り、再び閾値に達したら検証をやり直す。観測は最大 cue_n 個。

    海馬のバッファの候補の切り分けと、対立仮説が複数残る場面の検証（12.6.151節。既定では 12.6.147節と同じ挙動）:
      active_source = "holdout" … 候補を、リプレイに使わない保留サンプル（reliability_holdout > 0 で文脈ごとに保存。
          バッファと同じ時点・同じ整合フィルタで取る）から取り出す。"replay" との違いは、リプレイで学習し続けたか
          どうかだけ。reliability_holdout > 0 では、evidence_reliability によらず保留サンプルを保存する（学習・乱数の
          消費には影響しない）。
      verify_mode … active_verify の検証の手がかりの作り方。"top2"（既定。12.6.147節）は先頭と2番目、"all" は先頭に
          0.5、残りの全文脈に 0.5 を等分した事前分布で区別する手がかり（全対立仮説をまとめて反証しうる観測）、"each" は
          先頭以外の文脈を事後確率の順に1つずつ、先頭とその文脈だけを区別する手がかりで検証する（対立仮説ごとに反証を
          試みる。検証の手がかりは active_verify·(文脈の数 − 1) 個）。
      能動的な選択（active_pool > 0）では、最初に閾値に達した時点の事後分布について、先頭以外の質量のうち2番目の
      文脈が占める割合（"rival_share"）と、文脈の事後確率の順（"order"）を記録する（selection_trials > 0 では
      "selection_rival_share"・"selection_order"）。diagnostics = True では "origin_acc"（最終モデルで、各タスク j を
      各文脈 c で、入力の出どころ（own_buf: c のバッファ、other_buf: 他の文脈のバッファ、holdout: c の保留サンプル）
      ごとに、タスク j の正解で測った正確度）を加える。

    記憶に基づく予測との組み合わせ（12.6.155節。diagnostics = True かつ episodic_eval = N > 0 のとき。学習・評価の
    乱数は消費しない）: 最終モデルで、各タスク j を専用の乱数による新しい入力 N 個で、各文脈 c について
    episodic_readout（λ = episodic_lambda、κ = episodic_kappa）の正確度を測り、diag["episodic_acc"][j][c の順][読み出し]
    に入れる。読み出しは EPISODIC_READOUTS で、検索する痕跡は c の痕跡に限る（文脈に依存した検索）:
      param … パラメトリックな予測だけ。mix_all / switch_all … c のバッファ（リプレイした痕跡）と保留サンプルの先頭
      episodic_hold 個（保存だけでリプレイしなかった痕跡）。mix_buf … バッファだけ。mix_hold … 保留サンプルの先頭
      episodic_hold 個だけ。mix_nogate … 全文脈のバッファと保留サンプルの先頭 episodic_hold 個（文脈に依存しない検索）。
      mix_all200 / switch_all200 … c のバッファと保留サンプルのすべて（reliability_holdout 個まで）。
    diag["episodic_cover"][j][c の順] は新しい入力のうち c の痕跡（mix_all と同じ）に完全に一致するものの割合、
    diag["store_purity"][j][c の順] は同じ痕跡のラベルがタスク j の規則と一致する割合。

    隠れ層の表現での近さ・保留サンプルの正確さによる調停・テスト時の再固定化（12.6.159節。episodic_eval > 0 のとき。
    学習・評価の乱数は消費しない）:
      episodic_extra = True … diag["episodic_acc"] に EPISODIC_EXTRA_READOUTS を加える。
          mix_rep / switch_rep … mix_all / switch_all と同じ痕跡を、ネットワークの隠れ層の表現 h(x, c)
              （嗅内皮質の表現にあたる）の二乗距離で検索する。尺度は c の痕跡の対について、二乗距離の平均がハミング
              距離の平均に等しくなるように合わせる（h が入力そのものならハミング距離による検索と一致する）。
          arb_bits / arb_rep … c の保留サンプルのうち検索に使わない残り（較正用）で param・mix・switch の正確度を
              測り、最も正確なものを使う（同点は param、mix の順に優先）。diag["episodic_arb"][j][c の順] に選んだ
              読み出しを入れる。
      recon_steps = S > 0 … 各文脈 c について、最終モデルの複製を c の痕跡で S ステップ学び直し（Adam、学習率
          recon_lr、バッチ recon_batch、痕跡から復元抽出）、その複製の正確度を RECON_READOUTS として加える。
          recon_buf … c のバッファだけ。recon_all … c のバッファと保留サンプルの先頭 episodic_hold 個。
          recon_all200 … c のバッファと保留サンプルのすべて。recon_inter … recon_all の痕跡に、他の文脈の
          バッファ（保存時の文脈つき）を同じ数だけ交互に混ぜる。recon_hold … 保留サンプルの先頭 episodic_hold 個
          だけ（12.6.159節の修正1。recon_buf と痕跡の数を揃えた対照）。diag["recon_side"][変種][c の順][j][c' の順] は、
          recon_all・recon_inter の複製で各タスク j を各文脈 c' で測った正確度から最終モデルの値を引いた差
          （学び直しが他の文脈に及ぼす影響）。

    バッファでの学び直しが効く理由の切り分け（12.6.161節。recon_extra = True かつ recon_steps > 0 のとき。他の変種の
    値は変えない）: RECON_EXTRA_READOUTS を加える。recon_buf_inter … c のバッファに他の文脈のバッファを同じ数だけ
    交互に混ぜる（一つの文脈への集中を除く）。recon_buf_err … c のバッファのうち、複製が誤って予測するサンプルだけの
    損失で学ぶ（正しく予測できているサンプルの確信度のマージンを広げる学習を除く。誤りのないバッチでは更新しない）。
    diag["recon_margin"][c の順] は最終モデルでの c のバッファの損失・正確度・正解の確率の平均。

    学び直しの書き戻し（系統的な再固定化。12.6.161節。writeback が空でないとき。inferred / inferred_gated のみ）:
    タスク writeback_at の学習を始める前に、各文脈 c（used の順）を一度ずつ再活性化し、共有の重み（モデルそのもの）を
    c の痕跡（バッファと保留サンプルの先頭 episodic_hold 個、recon_all と同じ）で writeback_steps ステップ学び直す
    （文脈ごとに新しい Adam、学習率 recon_lr、バッチ recon_batch、専用の乱数）。
      "none" … 書き戻さない（測定だけ）。"all" … c の痕跡だけで学ぶ（一つの文脈だけの書き戻し）。"inter" … c の痕跡に
      他の文脈のバッファ（保存時の文脈つき）を同じ数だけ交互に混ぜる（recon_inter と同じ）。
    バッファ・保留サンプルは変えず、その後の学習（新しいタスクとリプレイ）は従来どおり続ける。学習・評価・診断の乱数は
    消費しないので、"none" の R と assign は writeback = "" と同じになる。diagnostics = True では
    diag["writeback"] = {"at", "used", "pre", "post"} を加える。pre / post は書き戻しの前後で、タスク j < writeback_at を
    専用の乱数による新しい入力 writeback_eval 個で各文脈 c で測った正確度（"fresh"[j][c の順]）と、c のバッファの入力を
    タスク j の正解で測った正確度（"own_buf"[j][c の順]）。
    writeback_keep = True（12.6.161節の修正1）では、書き戻しの後（"none" でも）、各文脈の保留サンプルの先頭
    episodic_hold 個をその文脈のバッファに加え、以後のリプレイに使う（再活性化した痕跡が再び海馬のリプレイの対象に
    なる。出どころは −1 として記録する）。post の測定はバッファに加える前に行う。

    リプレイの配分（12.6.163節。replay_rule が空でないとき。writeback があればタスク writeback_at から、なければタスク 1
    から適用する。既定の "" では従来どおり、バッファ全体から一様に batch 個を取る）:
      "gain" … 優先度つきのリプレイ（Mattar & Daw 2018 の gain の単純化）。バッファの各痕跡 i の現在のモデルでの損失
          ℓ_i（保存時の文脈、予測誤差）から、確率 p_i = β/N + (1 − β)·ℓ_i/Σℓ（β = replay_beta）で復元抽出する。
      "need" … 新旧の比率（Mattar & Daw 2018 の need の単純化）。リプレイのうち round(replay_need·batch) 個を、その
          ステップで選んだ文脈のバッファから、残りを他の文脈のバッファから一様に取る。選んだ文脈の痕跡がまだなければ
          他の文脈の分だけを取り（古い文脈のリプレイの量は変えない）、他の文脈の痕跡がなければ全体から一様に取る。
      "sleep" … 覚醒と睡眠に似た段階の交互。覚醒中（通常のステップ）のリプレイは、writeback_keep で加えた痕跡
          （出どころ −1）を除いたバッファから一様に取る。sleep_every ステップごと（step > 0）に、新しい入力なしで
          sleep_steps ステップ、加えた痕跡を含むバッファ全体（新しい文脈の痕跡も含む）から一様に batch 個を取って、
          同じ最適化器で学習する（オフラインの交互の固定化）。
      "ach" … 新しさによる符号化と固定化の切り替え（12.6.163節の修正1。Hasselmo 1999 のアセチルコリン）。そのステップの
          バッチが確立した文脈で驚きなく予測できる（gate の store、損失 ≦ τ）ときだけ、加えた痕跡を含むバッファ全体から
          一様に取る。驚きのある（新しい）ステップでは加えた痕跡を除いたバッファから一様に取る。
      "ach_need" … "ach" と同じ除外をした上で、"need" と同じ新旧の比率で取る。
      "ach_need2" … 驚きのあるステップでは従来と同じ（加えた痕跡を除いたバッファから一様に batch 個）、驚きのない
          ステップでは加えた痕跡を含めて "need" と同じ新旧の比率で取る（12.6.163節の修正2）。
      diagnostics = True では diag["replay_mix"][k]（規則を適用したタスク k の、リプレイの痕跡のうちそのステップで
      選んだ文脈のものの割合 "own"、加えた痕跡の割合 "kept"、1ステップのリプレイの数の平均 "size"、睡眠のステップ数
      "sleep"）を加える。
    """
    if condition not in CONDITIONS:
        raise ValueError(condition)
    if optimizer not in ("adam", "sgd"):
        raise ValueError(optimizer)
    if plasticity_ref not in ("capped", "raw"):
        raise ValueError(plasticity_ref)
    if min(split_window, split_ctx_gain, split_shared_gain, split_shared_lr_floor, split_ctx_lr) < 0:
        raise ValueError("split_window / split_ctx_gain / split_shared_gain / split_shared_lr_floor / split_ctx_lr は 0 以上")
    if cue_select not in ("batch", "evidence"):
        raise ValueError(cue_select)
    if not (0.0 <= evidence_lapse <= 1.0 and 0.0 <= evidence_alpha < 1.0):
        raise ValueError("evidence_lapse は [0, 1]、evidence_alpha は [0, 1)")
    if reliability_holdout < 0:
        raise ValueError("reliability_holdout は 0 以上")
    if reliability_source not in ("gated", "pre"):
        raise ValueError(reliability_source)
    if not (0.0 < com_alpha < 1.0) or com_window < 0:
        raise ValueError("com_alpha は (0, 1)、com_window は 0 以上")
    if cue_weight not in ("", "info", "info_post") or active_pool < 0:
        raise ValueError("cue_weight は ''・'info'・'info_post'、active_pool は 0 以上")
    if (cue_weight or active_pool) and (cue_select != "evidence" or evidence_reliability or evidence_com):
        raise ValueError("cue_weight / active_pool は cue_select='evidence'（信頼性・乗り換えなし）でのみ使う")
    if active_source not in ("task", "uniform", "replay", "search", "holdout") or active_verify < 0:
        raise ValueError("active_source は task・uniform・replay・search・holdout、active_verify は 0 以上")
    if (active_source != "task" or active_verify) and active_pool <= 0:
        raise ValueError("active_source / active_verify は active_pool > 0 でのみ使う")
    if verify_mode not in ("top2", "all", "each"):
        raise ValueError("verify_mode は top2・all・each")
    if active_source == "holdout" and reliability_holdout <= 0:
        raise ValueError("active_source='holdout' は reliability_holdout > 0 でのみ使う")
    if episodic_eval < 0 or episodic_hold < 0 or episodic_lambda <= 0 or episodic_kappa <= 0:
        raise ValueError("episodic_eval・episodic_hold は 0 以上、episodic_lambda・episodic_kappa は正")
    if recon_steps < 0 or recon_lr <= 0 or recon_batch <= 0:
        raise ValueError("recon_steps は 0 以上、recon_lr・recon_batch は正")
    if writeback and (writeback not in WRITEBACKS or condition not in ("inferred", "inferred_gated")
                      or not 0 < writeback_at < len(specs) or writeback_steps < 0 or writeback_eval <= 0):
        raise ValueError("writeback は none・all・inter（inferred / inferred_gated のみ）、0 < writeback_at < タスク数")
    if replay_rule and (replay_rule not in REPLAY_RULES or not 0.0 <= replay_beta <= 1.0
                        or not 0.0 < replay_need < 1.0 or sleep_every <= 0 or sleep_steps < 0):
        raise ValueError("replay_rule は gain・need・sleep・ach・ach_need・ach_need2、replay_beta は [0, 1]、replay_need は (0, 1)、"
                         "sleep_every は正、sleep_steps は 0 以上")
    rule_from = (writeback_at if writeback else 1) if replay_rule else len(specs)
    if split_shared_lr_floor > 0:
        split_shared_gain = max(split_shared_gain, min(1.0, split_shared_lr_floor / lr))
    if split_ctx_lr > 0:
        split_ctx_gain = split_ctx_lr / lr
    T = len(specs)
    C = 0 if condition == "none" else (T if condition == "oracle" else (n_slots or T + 2))
    model = make_model(specs[0].n_inputs + C, hidden_dim)

    def with_ctx(bits: torch.Tensor, ctx: torch.Tensor | int) -> torch.Tensor:
        if C == 0:
            return bits
        oh = torch.zeros(bits.shape[0], C)
        if isinstance(ctx, int):
            oh[:, ctx] = 1.0
        else:
            oh[torch.arange(bits.shape[0]), ctx] = 1.0
        return torch.cat([bits, oh], dim=1)

    @torch.no_grad()
    def loss_under(bits: torch.Tensor, labels: torch.Tensor, ctx: torch.Tensor | int) -> float:
        return float(F.cross_entropy(model(with_ctx(bits, ctx)), labels))

    @torch.no_grad()
    def select_ctx(bits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
        """手がかりから文脈を選ぶ。戻り値は (選んだ文脈, 決定までに使った手がかりの数)。"""
        if cue_select == "batch":
            return min(used, key=lambda c: loss_under(bits, labels, c)), bits.shape[0]
        if revised:
            return select_ctx_revised(bits, labels)
        if informed:
            return select_ctx_info(bits, labels)
        ll = []
        for c in used:
            p = F.softmax(model(with_ctx(bits, c)), dim=-1)
            p = (1 - evidence_lapse) * p + evidence_lapse / p.shape[-1]
            ll.append(torch.log(p.gather(1, labels[:, None])[:, 0]))
        cum = torch.stack(ll, dim=1).cumsum(dim=0)  # [手がかり, 文脈]
        post = F.softmax(cum, dim=1)
        hit = (post.max(dim=1).values >= 1 - evidence_alpha).nonzero()
        n = int(hit[0, 0]) + 1 if hit.numel() else bits.shape[0]
        return used[int(cum[n - 1].argmax())], n

    @torch.no_grad()
    def context_reliability() -> dict[int, float]:
        """12.6.139節: 各文脈の信頼性 a_c = min(学習中の正確度の移動平均, バッファの整合度)。"""
        src_b, src_l, src_c = (hold_bits, hold_labels, hold_ctx) if reliability_holdout > 0 else (buf_bits, buf_labels, buf_ctx)
        bc = torch.cat(src_c) if src_c else torch.zeros(0, dtype=torch.long)
        rel: dict[int, float] = {}
        for c in used:
            vals = [ema_acc[c]] if c in ema_acc else []
            m = bc == c
            if bool(m.any()):
                bb, bl = torch.cat(src_b)[m], torch.cat(src_l)[m]
                vals.append(float((model(with_ctx(bb, c)).argmax(-1) == bl).float().mean()))
            rel[c] = min(vals) if vals else 1.0
        return rel

    @torch.no_grad()
    def select_ctx_revised(bits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
        """12.6.139節の証拠の蓄積（信頼性による lapse・閾値、決定後の乗り換え）。sel_info に詳細を残す。"""
        N = bits.shape[0]
        ll = []
        for c in used:
            p = F.softmax(model(with_ctx(bits, c)), dim=-1)
            K = p.shape[-1]
            eps = (max(evidence_lapse, K * (1 - rel_now[c]) / (K - 1)) if evidence_reliability and reliability_lapse
                   else evidence_lapse)
            eps = min(eps, 1.0)
            p = (1 - eps) * p + eps / K
            ll.append(torch.log(p.gather(1, labels[:, None])[:, 0]))
        cum = torch.stack(ll, dim=1).cumsum(dim=0)  # [手がかり, 文脈]
        lead = cum.argmax(dim=1)
        if len(used) == 1:
            n = 1
        else:
            others = cum.clone()
            others[torch.arange(N), lead] = -math.inf
            log_odds = cum.max(dim=1).values - torch.logsumexp(others, dim=1)
            bound = math.log((1 - evidence_alpha) / evidence_alpha) if evidence_alpha > 0 else math.inf
            if evidence_reliability:
                r = torch.tensor([max(2 * rel_now[c] - 1, 0.1) for c in used])
                bound = bound / r[lead]
            hit = (log_odds >= bound).nonzero()
            n = int(hit[0, 0]) + 1 if hit.numel() else N
        cur_i = int(lead[n - 1])
        first, observed, switches = used[cur_i], n, 0
        if evidence_com and n < N and len(used) > 1:
            end = N if com_window == 0 else min(N, n + com_window)
            b_com = math.log((1 - com_alpha) / com_alpha)
            for t in range(n, end):
                row = cum[t].clone()
                row[cur_i] = -math.inf
                alt = int(row.argmax())
                if float(cum[t, alt] - cum[t, cur_i]) >= b_com:
                    cur_i, switches = alt, switches + 1
            observed = end
        sel_info.update(observed=observed, switches=switches, first=first)
        return used[cur_i], n

    @torch.no_grad()
    def ctx_probs(bits: torch.Tensor) -> torch.Tensor:
        """各文脈の lapse 付きの予測 [入力, 文脈, K]。"""
        probs = torch.stack([F.softmax(model(with_ctx(bits, c)), dim=-1) for c in used], dim=1)
        return (1 - evidence_lapse) * probs + evidence_lapse / probs.shape[-1]

    @torch.no_grad()
    def climb(x: torch.Tensor, g: float, prior: torch.Tensor, seen_pts: torch.Tensor) -> torch.Tensor:
        """12.6.147節の "search": 1ビットの反転で情報量が増える限り登る（既に観測した入力には移らない）。"""
        n = x.shape[0]
        for _ in range(n):
            nb = x.repeat(n, 1)
            nb[torch.arange(n), torch.arange(n)] *= -1
            gain = cue_information(ctx_probs(nb), prior)
            if seen_pts.numel():
                gain[(nb[:, None, :] == seen_pts[None]).all(-1).any(-1)] = -math.inf
            b = int(gain.argmax())
            if float(gain[b]) <= g + 1e-9:
                break
            x, g = nb[b], float(gain[b])
        return x

    @torch.no_grad()
    def select_ctx_info(bits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
        """12.6.143節: 手がかりの情報量で重み付けた証拠の蓄積と、能動的な手がかりの選択。bits の行数は
        active_pool·cue_n（能動）か cue_n。観測するのは最大 cue_n 個。12.6.147節の候補の出どころ（"search" は
        候補を登って作り直し、正解を環境に問い合わせる）と、検証つきの打ち切り（active_verify）にも対応する。"""
        probs = ctx_probs(bits)  # [候補, 文脈, K]
        ll = torch.log(probs.gather(2, labels[:, None, None].expand(-1, len(used), 1))[..., 0])  # [候補, 文脈]
        info_u = cue_information(probs)  # 一様な事前分布のもとでの情報量（重みと記録に使う）
        w = info_u if cue_weight == "info" else torch.ones(bits.shape[0])
        cum = torch.zeros(len(used))
        avail = torch.ones(bits.shape[0], dtype=torch.bool)
        seen: list[int] = []
        seen_pts = torch.zeros(0, bits.shape[1])
        verify_left, leader0, verify_total = 0, -1, 0
        rivals: list[int] = []
        first_hit: dict = {}
        for t in range(cue_n):
            if active_pool > 0:
                post = F.softmax(cum, dim=0)
                prior = post
                if verify_left > 0:  # 先頭と対立仮説を区別する手がかり（12.6.147・151節）
                    prior = torch.zeros(len(used))
                    if verify_mode == "all":
                        prior[:] = 0.5 / (len(used) - 1)
                        prior[leader0] = 0.5
                    elif verify_mode == "each":
                        prior[leader0] = 0.5
                        prior[rivals[(verify_total - verify_left) % len(rivals)]] = 0.5
                    else:
                        prior[post.topk(2).indices] = 0.5
                gain = cue_information(probs, prior)
                gain[~avail] = -math.inf
                if active_source == "search" and seen_pts.numel():
                    gain[(bits[:, None, :] == seen_pts[None]).all(-1).any(-1)] = -math.inf
                i = int(gain.argmax())
                if active_source == "search":
                    x = climb(bits[i], float(gain[i]), prior, seen_pts)
                    if not bool((x == bits[i]).all()):  # 作った入力を候補に足し、正解を環境に問い合わせる
                        bits, labels = torch.cat([bits, x[None]]), torch.cat([labels, task_labels(oracle[0], x[None])])
                        p_new = ctx_probs(x[None])
                        probs = torch.cat([probs, p_new])
                        ll = torch.cat([ll, torch.log(p_new.gather(2, labels[-1:, None, None].expand(-1, len(used), 1))[..., 0])])
                        info_u = torch.cat([info_u, cue_information(p_new)])
                        w = torch.cat([w, info_u[-1:] if cue_weight == "info" else torch.ones(1)])
                        avail = torch.cat([avail, torch.ones(1, dtype=torch.bool)])
                        i = bits.shape[0] - 1
            else:
                i = t
            avail[i] = False
            seen.append(i)
            seen_pts = torch.cat([seen_pts, bits[i:i + 1]])
            if cue_weight == "info_post":
                w_i = cue_information(probs[i:i + 1], F.softmax(cum, dim=0))[0]
            else:
                w_i = w[i]
            cum = cum + w_i * ll[i]
            if len(used) == 1:
                break
            conf = float(F.softmax(cum, dim=0).max()) >= 1 - evidence_alpha
            lead = int(cum.argmax())
            if verify_left > 0:
                verify_left -= 1
                if not conf or lead != leader0:  # 検証で先頭が揺らいだら通常の蓄積に戻る
                    verify_left = 0
                elif verify_left == 0:
                    break
            elif conf:
                if not first_hit:
                    post = F.softmax(cum, dim=0)
                    srt = post.argsort(descending=True)
                    first_hit.update(rival_share=float(post[srt[1]] / post[srt[1:]].sum().clamp_min(1e-300)),
                                     order=[used[int(i_)] for i_ in srt])
                if active_verify == 0:
                    break
                leader0 = lead
                rivals = [int(i_) for i_ in F.softmax(cum, dim=0).argsort(descending=True)[1:]]
                verify_total = active_verify * (len(used) - 1) if verify_mode == "each" else active_verify
                verify_left = verify_total
        sel_info.update(info=float(info_u[seen].mean()), rival_share=first_hit.get("rival_share"),
                        order=first_hit.get("order"))
        return used[int(cum.argmax())], len(seen)

    revised = cue_select == "evidence" and (evidence_reliability or evidence_com)
    informed = cue_select == "evidence" and (bool(cue_weight) or active_pool > 0)
    active_gen = torch.Generator().manual_seed(eval_generator.initial_seed() + 777)  # active_pool の候補専用
    oracle: list = [None]  # 12.6.147節: 自己生成の候補の正解を返す、評価中のタスクの spec
    sel_info: dict = {}
    rel_now: dict[int, float] = {}
    ema_acc: dict[int, float] = {}  # 12.6.139節: 文脈ごとの学習中の正確度の移動平均（evidence_reliability のみ）
    hold_bits: list[torch.Tensor] = []  # 12.6.139節の修正2: リプレイに使わない保留サンプル
    hold_labels: list[torch.Tensor] = []
    hold_ctx: list[torch.Tensor] = []
    hold_count: dict[int, int] = {}
    buf_bits: list[torch.Tensor] = []
    buf_labels: list[torch.Tensor] = []
    buf_ctx: list[torch.Tensor] = []
    buf_src: list[torch.Tensor] = []
    diag_gen = torch.Generator().manual_seed(1234)
    trace_gen = torch.Generator().manual_seed(4321)  # trace 専用（diag_gen の消費を変えない）
    diag: dict = {"ctx_acc": [], "chosen": [], "cues_used": [], "trained_src": [], "trace": [], "upd": []}
    if revised:
        diag.update(cues_observed=[], switches=[], first_choice=[], reliability=[])
    buf_count: dict[int, int] = {}
    ema_loss: dict[int, float] = {}
    cur = 0
    used = [0]
    assign: list[tuple[int, int]] = []

    gate = {"train": True, "store": True}
    novel_streak = [0]

    def pick_ctx(k: int, bits: torch.Tensor, labels: torch.Tensor) -> int:
        nonlocal cur
        gate["train"], gate["store"] = True, True
        if condition == "oracle":
            return k
        if condition == "random":
            return int(torch.randint(C, (1,), generator=generator))
        if condition == "none":
            return 0
        established = ema_loss.get(cur, math.inf) < tau
        if condition == "inferred_gated":
            # 12.6.115節: 驚きのバッチ（混合の疑い）は学習・保存に使わない。新しい文脈は、現在の文脈での正答率が
            # 完全とチャンスの中間（0.75）を下回ったとき、つまり新しい原因が多数派になってから作る。たまたまの1バッチで作らないよう、それが novel_patience 回続くことを条件にする
            gate["train"], gate["store"] = True, False
            if not established:
                return cur
            if loss_under(bits, labels, cur) <= tau:
                gate["store"] = True
                return cur
            losses = {c: loss_under(bits, labels, c) for c in used}
            best = min(losses, key=losses.get)
            if best != cur and losses[best] < tau:
                cur = best
                return cur
            with torch.no_grad():
                accs = {c: float((model(with_ctx(bits, c)).argmax(-1) == labels).float().mean()) for c in used}
            novel_streak[0] = novel_streak[0] + 1 if max(accs.values()) < 0.75 else 0
            if novel_streak[0] >= novel_patience and len(used) < C:
                novel_streak[0] = 0
                cur = len(used)
                used.append(cur)
                return cur
            gate["train"] = False
            return cur
        if established and loss_under(bits, labels, cur) > tau:
            losses = {c: loss_under(bits, labels, c) for c in used}
            best = min(losses, key=losses.get)
            if losses[best] < tau:
                cur = best
            elif len(used) < C:
                cur = len(used)
                used.append(cur)
            else:
                cur = best
        return cur

    def draw_cues(j: int) -> tuple[torch.Tensor, torch.Tensor]:
        """テスト時の手がかり。active_pool > 0 では、従来の cue_n 個の後に専用の乱数で候補を足す（評価用の乱数の消費は同じ）。"""
        cb, cl = make_batch(specs[j], cue_n, eval_generator)
        oracle[0] = specs[j]
        if informed and active_source in ("uniform", "search"):
            cb = torch.randint(0, 2, (active_pool * cue_n, specs[j].n_inputs), generator=active_gen).float() * 2 - 1
            return cb, task_labels(specs[j], cb)
        if informed and active_source in ("replay", "holdout"):
            src_bits = buf_bits if active_source == "replay" else hold_bits
            pool = torch.cat(src_bits) if src_bits else cb
            cb = pool[torch.randperm(pool.shape[0], generator=active_gen)[:active_pool * cue_n]]
            return cb, task_labels(specs[j], cb)
        if informed and active_pool > 1:
            xb, xl = make_batch(specs[j], (active_pool - 1) * cue_n, active_gen)
            cb, cl = torch.cat([cb, xb]), torch.cat([cl, xl])
        return cb, cl

    @torch.no_grad()
    def accuracy(j: int, chosen: list[int] | None = None, cues: list[int] | None = None,
                 extra: dict | None = None) -> float:
        if condition in ("inferred", "inferred_gated"):
            cb, cl = draw_cues(j)
            ctx, n_cues = select_ctx(cb, cl)
            if informed and extra is not None:
                extra.setdefault("info", []).append(sel_info["info"])
            if chosen is not None:
                chosen.append(ctx)
            if cues is not None:
                cues.append(n_cues)
            if extra is not None and revised:
                for k_ in ("observed", "switches", "first"):
                    extra.setdefault(k_, []).append(sel_info[k_])
        elif condition == "random":
            ctx = int(torch.randint(C, (1,), generator=eval_generator))
        else:
            ctx = j if condition == "oracle" else 0
        bits, labels = make_batch(specs[j], eval_n, eval_generator)
        return float((model(with_ctx(bits, ctx)).argmax(-1) == labels).float().mean())

    def make_opt(k: int) -> torch.optim.Optimizer:
        wd = pretrain_weight_decay if k == 0 else 0.0
        if optimizer == "sgd":
            return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    @torch.no_grad()
    def wb_measure(fresh: list, n_tasks: int) -> dict:
        bb = torch.cat(buf_bits) if buf_bits else torch.zeros(0, n_bits)
        bc = torch.cat(buf_ctx) if buf_ctx else torch.zeros(0, dtype=torch.long)
        fr = [[float((model(with_ctx(x, c)).argmax(-1) == y).float().mean()) for c in used] for x, y in fresh]
        ob = [[float((model(with_ctx(bb[bc == c], c)).argmax(-1) == task_labels(specs[j], bb[bc == c])).float().mean())
               if bool((bc == c).any()) else None for c in used] for j in range(n_tasks)]
        return {"fresh": fr, "own_buf": ob}

    def writeback_phase(n_tasks: int) -> None:
        """12.6.161節: 各文脈を一度ずつ再活性化し、その痕跡での学び直しを共有の重みに書き戻す（系統的な再固定化）。"""
        egen = torch.Generator().manual_seed(5679)  # 書き戻しの前後の測定専用
        fresh = [make_batch(specs[j], writeback_eval, egen) for j in range(n_tasks)]
        pre = wb_measure(fresh, n_tasks) if diagnostics else None
        if writeback != "none":
            wgen = torch.Generator().manual_seed(6791)
            bb = torch.cat(buf_bits) if buf_bits else torch.zeros(0, n_bits)
            bl = torch.cat(buf_labels) if buf_labels else torch.zeros(0, dtype=torch.long)
            bc = torch.cat(buf_ctx) if buf_ctx else torch.zeros(0, dtype=torch.long)
            hb = torch.cat(hold_bits) if hold_bits else torch.zeros(0, n_bits)
            hl = torch.cat(hold_labels) if hold_labels else torch.zeros(0, dtype=torch.long)
            hc = torch.cat(hold_ctx) if hold_ctx else torch.zeros(0, dtype=torch.long)
            h_rank = torch.zeros(hc.shape[0], dtype=torch.long)  # 保留サンプルの文脈ごとの保存の順
            for c in hc.unique():
                idx = (hc == c).nonzero()[:, 0]
                h_rank[idx] = torch.arange(idx.shape[0])
            for c in list(used):
                mc, mh = bc == c, (hc == c) & (h_rank < episodic_hold)
                sb, sl = torch.cat([bb[mc], hb[mh]]), torch.cat([bl[mc], hl[mh]])
                if sb.shape[0] == 0:
                    continue
                ob, ol, oc = bb[~mc], bl[~mc], bc[~mc]
                wopt = torch.optim.Adam(model.parameters(), lr=recon_lr)
                for _ in range(writeback_steps):
                    idx = torch.randint(sb.shape[0], (recon_batch,), generator=wgen)
                    xr, yr = with_ctx(sb[idx], c), sl[idx]
                    if writeback == "inter" and ob.shape[0] > 0:
                        io = torch.randint(ob.shape[0], (recon_batch,), generator=wgen)
                        xr, yr = torch.cat([xr, with_ctx(ob[io], oc[io])]), torch.cat([yr, ol[io]])
                    loss = F.cross_entropy(model(xr), yr)
                    wopt.zero_grad(set_to_none=True)
                    loss.backward()
                    wopt.step()
        if diagnostics:
            diag["writeback"] = {"at": n_tasks, "used": list(used), "pre": pre, "post": wb_measure(fresh, n_tasks)}
        if writeback_keep and hold_ctx:
            hc = torch.cat(hold_ctx)
            keep = torch.zeros(hc.shape[0], dtype=torch.bool)
            for c in hc.unique():
                keep[(hc == c).nonzero()[:episodic_hold, 0]] = True
            buf_bits.append(torch.cat(hold_bits)[keep])
            buf_labels.append(torch.cat(hold_labels)[keep])
            buf_ctx.append(hc[keep])
            buf_src.append(torch.full((int(keep.sum()),), -1, dtype=torch.long))

    def replay_draw(rule: str, ctx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """12.6.163節: リプレイする痕跡（入力に文脈を付けたもの、ラベル、出どころ）を取る。rule = "" は従来と同じ。"""
        bb, bl, bc = torch.cat(buf_bits), torch.cat(buf_labels), torch.cat(buf_ctx)
        bs = torch.cat(buf_src)
        novel = rule in ("ach", "ach_need", "ach_need2") and not gate["store"]
        if rule == "sleep" or novel:
            wake = bs != -1
            bb, bl, bc, bs = bb[wake], bl[wake], bc[wake], bs[wake]
        n = bb.shape[0]
        idx = None
        if rule == "gain" and n > 0:
            with torch.no_grad():
                li = F.cross_entropy(model(with_ctx(bb, bc)), bl, reduction="none")
            p = replay_beta / n + (1 - replay_beta) * li / li.sum().clamp_min(1e-12)
            idx = torch.multinomial(p, batch, replacement=True, generator=generator)
        elif rule in ("need", "ach_need") or (rule == "ach_need2" and not novel):
            own, oth = (bc == ctx).nonzero()[:, 0], (bc != ctx).nonzero()[:, 0]
            n_own = round(replay_need * batch)
            if oth.numel() > 0:
                # 古い文脈のリプレイは常に batch − n_own 個まで。今の文脈の痕跡がまだなければ、その分は取らない
                idx = oth[torch.randint(oth.numel(), (batch - n_own,), generator=generator)]
                if own.numel() > 0:
                    idx = torch.cat([own[torch.randint(own.numel(), (n_own,), generator=generator)], idx])
        if idx is None:
            idx = torch.randint(n, (batch,), generator=generator)
        return with_ctx(bb[idx], bc[idx]), bl[idx], torch.stack([bc[idx] == ctx, bs[idx] == -1])

    def sleep_phase(o: torch.optim.Optimizer) -> None:
        """12.6.163節: 新しい入力なしで、加えた痕跡を含むバッファ全体を交互にリプレイする（睡眠に似た段階）。"""
        bb, bl, bc = torch.cat(buf_bits), torch.cat(buf_labels), torch.cat(buf_ctx)
        for _ in range(sleep_steps):
            idx = torch.randint(bb.shape[0], (batch,), generator=generator)
            loss = F.cross_entropy(model(with_ctx(bb[idx], bc[idx])), bl[idx])
            o.zero_grad(set_to_none=True)
            loss.backward()
            o.step()

    upd_ref: list[float] = []  # plasticity_cap の ū（平常時の更新ノルムの移動平均）
    split = {"left": 0, "ctx": -1}  # split_window の残りステップと、対象の新しい文脈
    split_on = split_window > 0 and condition in ("inferred", "inferred_gated")
    fc1_w = model.backbone.fc1.weight
    n_bits = specs[0].n_inputs
    R: list[list[float]] = []
    for k, spec in enumerate(specs):
        if writeback and k == writeback_at:
            writeback_phase(k)
        trained: dict[tuple[int, int], int] = {}
        n_used_at_start = len(used)
        opt = make_opt(k)
        rule = replay_rule if k >= rule_from else ""
        mix = {"own": 0.0, "kept": 0.0, "size": 0, "n": 0, "sleep": 0}
        for step in range(pretrain_steps if k == 0 else steps):
            if rule == "sleep" and step > 0 and step % sleep_every == 0 and buf_bits:
                sleep_phase(opt)
                mix["sleep"] += sleep_steps
            bits, labels = make_batch(spec, batch, generator)
            src = None
            if k > 0 and step < transition_steps:
                old_bits, old_labels = make_batch(specs[k - 1], batch, generator)
                take_new = torch.rand(batch, generator=generator) < (step + 1) / (transition_steps + 1)
                bits = torch.where(take_new[:, None], bits, old_bits)
                labels = torch.where(take_new, labels, old_labels)
                src = torch.where(take_new, k, k - 1)
            n_used_before = len(used)
            ctx = pick_ctx(k, bits, labels)
            assign.append((k, ctx))
            if evidence_reliability and revised and reliability_source == "pre":
                with torch.no_grad():
                    pre_acc = float((model(with_ctx(bits, ctx)).argmax(-1) == labels).float().mean())
                if pre_acc >= 0.75:
                    ema_acc[ctx] = pre_acc if ctx not in ema_acc else ema * ema_acc[ctx] + (1 - ema) * pre_acc
            if reset_opt_on_new and len(used) > n_used_before:
                opt = make_opt(k)
            if split_on and len(used) > n_used_before:
                split["left"], split["ctx"] = split_window, ctx
            split_now = split["left"] > 0
            split["left"] = max(split["left"] - 1, 0)
            ctx_in = src if (condition == "oracle" and src is not None) else ctx
            x = with_ctx(bits, ctx_in)
            y = labels
            s_ = src if src is not None else torch.full((batch,), k)
            if not gate["train"]:
                x, y = x[:0], y[:0]
            elif train_consistent and condition == "inferred_gated" and gate["store"]:
                # 12.6.116節: 確立した文脈で驚きのなかったバッチでも、文脈の規則と食い違うサンプルでは学習しない
                # （移行の初期に混ざる次のタスクのサンプルで、確立した文脈が上書きされるのを防ぐ）
                with torch.no_grad():
                    keep = model(x).argmax(-1) == y
                x, y, s_ = x[keep], y[keep], s_[keep]
            if diag_oracle_filter and gate["train"] and src is not None and condition == "inferred_gated":
                new_ctx = ctx >= n_used_at_start
                if (new_ctx and diag_oracle_filter in ("new", "both")) or (not new_ctx and diag_oracle_filter in ("old", "both")):
                    keep = s_ == (k if new_ctx else k - 1)
                    x, y, s_ = x[keep], y[keep], s_[keep]
            if gate["train"] and diagnostics:
                for t in s_.unique().tolist():
                    trained[(ctx, t)] = trained.get((ctx, t), 0) + int((s_ == t).sum())
            n_new = y.shape[0]
            if buf_bits and k > 0:
                if rule:
                    xr, yr, flags = replay_draw(rule, ctx)
                    mix["own"] += float(flags[0].float().mean())
                    mix["kept"] += float(flags[1].float().mean())
                    mix["size"] += yr.shape[0]
                    mix["n"] += 1
                else:
                    bb, bl, bc = torch.cat(buf_bits), torch.cat(buf_labels), torch.cat(buf_ctx)
                    idx = torch.randint(bb.shape[0], (batch,), generator=generator)
                    xr, yr = with_ctx(bb[idx], bc[idx]), bl[idx]
                x = torch.cat([x, xr])
                y = torch.cat([y, yr])
            if y.numel() > 0:
                loss = F.cross_entropy(model(x), y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if diagnostics or plasticity_cap > 0 or split_now:
                    before = [q.detach().clone() for q in model.parameters()]
                if diagnostics:
                    gnorm = float(torch.sqrt(sum((q.grad ** 2).sum() for q in model.parameters() if q.grad is not None)))
                opt.step()
                if split_now:
                    # 12.6.129節: 新しい原因を立てた直後は、その文脈の入力重みの可塑性を上げ（新しい情報からの学習を
                    # 速める）、共有の重みの可塑性を下げる（前の文脈への干渉を抑える）
                    with torch.no_grad():
                        for q, b in zip(model.parameters(), before):
                            d = (q - b) * split_shared_gain
                            if q is fc1_w:
                                col = n_bits + split["ctx"]
                                d[:, col] = (q[:, col] - b[:, col]) * split_ctx_gain
                            q.copy_(b + d)
                if diagnostics or plasticity_cap > 0:
                    unorm = float(torch.sqrt(sum(((q.detach() - b) ** 2).sum() for q, b in zip(model.parameters(), before))))
                if plasticity_cap > 0:
                    # 12.6.125節: 平常時の可塑性 ū から予期せぬほど大きい更新を縮める（ノルアドレナリンによる利得調節の
                    # 単純化）。リプレイだけの静かな期間の後、最適化器の古い推定のまま大きな誤差に反応するのを抑える
                    raw_unorm = unorm
                    if upd_ref and unorm > plasticity_cap * upd_ref[0]:
                        scale = plasticity_cap * upd_ref[0] / unorm
                        with torch.no_grad():
                            for q, b in zip(model.parameters(), before):
                                q.copy_(b + scale * (q - b))
                        unorm *= scale
                    if n_new > 0:
                        u_ = raw_unorm if plasticity_ref == "raw" else unorm
                        upd_ref[:] = [u_ if not upd_ref else plasticity_ema * upd_ref[0] + (1 - plasticity_ema) * u_]
                if diagnostics:
                    diag["upd"].append((k, step, ctx, gnorm, unorm))
            if gate["train"]:
                with torch.no_grad():
                    cur_loss = loss_under(bits, labels, ctx_in)
                ema_loss[ctx] = cur_loss if ctx not in ema_loss else ema * ema_loss[ctx] + (1 - ema) * cur_loss
                if evidence_reliability and revised and reliability_source == "gated":
                    with torch.no_grad():
                        cur_acc = float((model(with_ctx(bits, ctx_in)).argmax(-1) == labels).float().mean())
                    ema_acc[ctx] = cur_acc if ctx not in ema_acc else ema * ema_acc[ctx] + (1 - ema) * cur_acc
            if diagnostics and trace_every > 0 and step % trace_every == 0 and condition in ("inferred", "inferred_gated"):
                with torch.no_grad():
                    accs_t = []
                    for j in range(T):
                        tb, tl = make_batch(specs[j], 500, trace_gen)
                        accs_t.append([float((model(with_ctx(tb, c)).argmax(-1) == tl).float().mean()) for c in used])
                diag["trace"].append({"task": k, "step": step, "ctx": ctx, "train": gate["train"],
                                      "store": gate["store"], "acc": accs_t})
            if not gate["store"]:
                continue
            # 観測したサンプルを文脈ラベル付きで保存する。保存枠は inferred では推定した文脈ごと
            # （タスクの区切りを知らない）、他の条件では真のタスクごとに buffer_size 個まで
            key = ctx if condition in ("inferred", "inferred_gated") else k
            if store_consistent and condition == "inferred_gated":
                # 12.6.116節: 文脈の今の規則と整合する（正しく予測できる）サンプルだけを保存する。移行の終盤に
                # 混ざる前のタスクのサンプルのうち、規則と食い違うものがバッファに入るのを防ぐ（スキーマ整合的な固定化）
                with torch.no_grad():
                    keep = model(with_ctx(bits, ctx)).argmax(-1) == labels
                bits, labels = bits[keep], labels[keep]
                src = src[keep] if src is not None else None
            n = min(buffer_size - buf_count.get(key, 0), labels.shape[0])
            if n > 0:
                buf_count[key] = buf_count.get(key, 0) + n
                buf_bits.append(bits[:n])
                buf_labels.append(labels[:n])
                buf_ctx.append(ctx_in[:n].long() if isinstance(ctx_in, torch.Tensor)
                               else torch.full((n,), ctx, dtype=torch.long))
                buf_src.append(src[:n].long() if src is not None else torch.full((n,), k, dtype=torch.long))
            if reliability_holdout > 0:
                n0 = max(n, 0)
                m = min(reliability_holdout - hold_count.get(key, 0), labels.shape[0] - n0)
                if m > 0:
                    hold_count[key] = hold_count.get(key, 0) + m
                    hold_bits.append(bits[n0:n0 + m])
                    hold_labels.append(labels[n0:n0 + m])
                    hold_ctx.append(torch.full((m,), ctx, dtype=torch.long))
        chosen: list[int] = []
        cues_used: list[int] = []
        extra: dict = {}
        if evidence_reliability and revised:
            rel_now.clear()
            rel_now.update(context_reliability())
        R.append([accuracy(j, chosen, cues_used, extra) for j in range(T)])
        if diagnostics:
            with torch.no_grad():
                acc_row = []
                for j in range(T):
                    bits, labels = make_batch(specs[j], eval_n, diag_gen)
                    acc_row.append([float((model(with_ctx(bits, c)).argmax(-1) == labels).float().mean())
                                    for c in (used if condition in ("inferred", "inferred_gated") else [0])])
            diag["ctx_acc"].append(acc_row)
            diag["chosen"].append(chosen)
            diag["cues_used"].append(cues_used)
            if revised:
                diag["cues_observed"].append(extra.get("observed", []))
                diag["switches"].append(extra.get("switches", []))
                diag["first_choice"].append(extra.get("first", []))
                diag["reliability"].append(dict(rel_now))
            if informed:
                diag.setdefault("cue_info", []).append(extra.get("info", []))
            diag["trained_src"].append({f"{c}<-{t}": n for (c, t), n in sorted(trained.items())})
            if rule:
                m_ = max(mix["n"], 1)
                diag.setdefault("replay_mix", {})[k] = {"own": mix["own"] / m_, "kept": mix["kept"] / m_,
                                                        "size": mix["size"] / m_, "sleep": mix["sleep"]}
    selection_acc = None
    sel_choices: list[list[int]] = []
    sel_cues: list[list[int]] = []
    sel_extra: dict[str, list[list[int]]] = {"observed": [], "switches": [], "first": []}
    if condition in ("inferred", "inferred_gated") and selection_trials > 0:
        majority = assignment_majority(assign, T)
        if evidence_reliability and revised:
            rel_now.clear()
            rel_now.update(context_reliability())
        hits = 0
        sel_inf: list[list[float]] = [[] for _ in range(T)]
        sel_share: list[list] = [[] for _ in range(T)]
        sel_order: list[list] = [[] for _ in range(T)]
        for j in range(T):
            sel_choices.append([])
            sel_cues.append([])
            for v in sel_extra.values():
                v.append([])
            for _ in range(selection_trials):
                cb, cl = draw_cues(j)
                ctx, n_cues = select_ctx(cb, cl)
                if informed:
                    sel_inf[j].append(sel_info["info"])
                    sel_share[j].append(sel_info["rival_share"])
                    sel_order[j].append(sel_info["order"])
                hits += int(ctx == majority[j])
                sel_choices[j].append(ctx)
                sel_cues[j].append(n_cues)
                if revised:
                    for k_, v in sel_extra.items():
                        v[j].append(sel_info[k_])
        selection_acc = hits / (T * selection_trials)
    out = {"R": R, "assign": assign, "n_contexts": len(used) if condition in ("inferred", "inferred_gated") else C,
           "selection_acc": selection_acc}
    if sel_choices:
        out.update(selection_choices=sel_choices, selection_cues=sel_cues)
        if informed:
            out.update(selection_info=sel_inf)
            if active_pool > 0:
                out.update(selection_rival_share=sel_share, selection_order=sel_order)
        if revised:
            out.update(selection_observed=sel_extra["observed"], selection_switches=sel_extra["switches"],
                       selection_first=sel_extra["first"])
    if diagnostics:
        bc = torch.cat(buf_ctx) if buf_ctx else torch.zeros(0, dtype=torch.long)
        bs = torch.cat(buf_src) if buf_src else torch.zeros(0, dtype=torch.long)
        bb = torch.cat(buf_bits) if buf_bits else torch.zeros(0)
        bl = torch.cat(buf_labels) if buf_labels else torch.zeros(0, dtype=torch.long)
        diag["buffer"] = {"bits": bb, "labels": bl, "ctx": bc, "src": bs}
        if condition in ("inferred", "inferred_gated"):
            # 12.6.151節: 入力の出どころごとの正確度（正解は評価するタスク j の規則）
            hc = torch.cat(hold_ctx) if hold_ctx else torch.zeros(0, dtype=torch.long)
            hb = torch.cat(hold_bits) if hold_bits else torch.zeros(0, n_bits)
            bb_ = bb if bb.numel() else torch.zeros(0, n_bits)

            @torch.no_grad()
            def acc_on(x: torch.Tensor, j: int, c: int) -> float | None:
                if x.shape[0] == 0:
                    return None
                return float((model(with_ctx(x, c)).argmax(-1) == task_labels(specs[j], x)).float().mean())

            diag["origin_acc"] = [[{"own_buf": acc_on(bb_[bc == c], j, c), "other_buf": acc_on(bb_[bc != c], j, c),
                                    "holdout": acc_on(hb[hc == c], j, c)} for c in used] for j in range(T)]
            if episodic_eval > 0:
                epi_gen = torch.Generator().manual_seed(5678)  # 12.6.155節の読み出し専用
                blab = bl if bl.numel() else torch.zeros(0, dtype=torch.long)
                hl = torch.cat(hold_labels) if hold_labels else torch.zeros(0, dtype=torch.long)
                # 保留サンプルの先頭 episodic_hold 個（文脈ごと、保存の順）。reliability_holdout = episodic_hold の
                # 場合の保留サンプルと同じになる
                h_rank = torch.zeros(hc.shape[0], dtype=torch.long)
                for c in hc.unique():
                    idx = (hc == c).nonzero()[:, 0]
                    h_rank[idx] = torch.arange(idx.shape[0])
                h_small = h_rank < episodic_hold

                def store(c: int | None, buf: bool, hold: str) -> tuple[torch.Tensor, torch.Tensor]:
                    mb = (bc == c) if c is not None else torch.ones_like(bc, dtype=torch.bool)
                    mh = (hc == c) if c is not None else torch.ones_like(hc, dtype=torch.bool)
                    mh = mh & h_small if hold == "small" else (mh if hold == "all" else torch.zeros_like(mh))
                    if not buf:
                        mb = torch.zeros_like(mb)
                    return torch.cat([bb_[mb], hb[mh]]), torch.cat([blab[mb], hl[mh]])

                plans = {"mix_all": (True, "small", "mix", False), "switch_all": (True, "small", "switch", False),
                         "mix_buf": (True, "none", "mix", False), "mix_hold": (False, "small", "mix", False),
                         "mix_nogate": (True, "small", "mix", True), "mix_all200": (True, "all", "mix", False),
                         "switch_all200": (True, "all", "switch", False)}
                fresh = [make_batch(specs[j], episodic_eval, epi_gen) for j in range(T)]
                ext: dict[int, dict] = {}  # 12.6.159節: 文脈ごとの尺度・調停の選択
                if episodic_extra:
                    for c in used:
                        sb, sl = store(c, True, "small")
                        mh = (hc == c) & ~h_small
                        cal_x, cal_y = hb[mh], hl[mh]
                        with torch.no_grad():
                            hs = model.backbone(with_ctx(sb, c))
                            ham = (n_bits - sb @ sb.T) / 2
                            sq = torch.cdist(hs, hs) ** 2
                        scale = float(ham.sum() / sq.sum()) if sb.shape[0] > 1 and float(sq.sum()) > 0 else 1.0
                        ext[c] = {"sb": sb, "sl": sl, "hs": hs, "scale": scale, "arb": {}}
                        if cal_x.shape[0] == 0:
                            ext[c]["arb"] = {"bits": "param", "rep": "param"}
                            continue
                        with torch.no_grad():
                            pc = F.softmax(model(with_ctx(cal_x, c)), dim=-1)
                            dc = {"bits": (n_bits - cal_x @ sb.T) / 2,
                                  "rep": scale * torch.cdist(model.backbone(with_ctx(cal_x, c)), hs) ** 2}
                        for kind, dd in dc.items():
                            cand = {"param": pc, **{r: episodic_readout_dist(pc, dd, sl, episodic_lambda,
                                                                             episodic_kappa, r)
                                                    for r in ("mix", "switch")}}
                            ca = {r: float((v.argmax(-1) == cal_y).float().mean()) for r, v in cand.items()}
                            ext[c]["arb"][kind] = max(("param", "mix", "switch"), key=lambda r: ca[r])  # 同点は先のもの
                models = {}  # 12.6.159節: テスト時の再固定化（c の痕跡で学び直した複製）
                if recon_steps > 0:
                    rgen = torch.Generator().manual_seed(6789)
                    hgen = torch.Generator().manual_seed(6790)  # recon_hold 専用（他の変種の乱数の消費を変えない）
                    for c in used:
                        for name, (ub, hold) in {"recon_buf": (True, "none"), "recon_all": (True, "small"),
                                                 "recon_all200": (True, "all"), "recon_inter": (True, "small"),
                                                 "recon_hold": (False, "small")}.items():
                            sb, sl = store(c, ub, hold)
                            m_ = copy.deepcopy(model)
                            if sb.shape[0] > 0:
                                ob, ol, oc = bb_[bc != c], blab[bc != c], bc[bc != c]
                                ropt = torch.optim.Adam(m_.parameters(), lr=recon_lr)
                                for _ in range(recon_steps):
                                    g_ = hgen if name == "recon_hold" else rgen
                                    idx = torch.randint(sb.shape[0], (recon_batch,), generator=g_)
                                    xr, yr = with_ctx(sb[idx], c), sl[idx]
                                    if name == "recon_inter" and ob.shape[0] > 0:
                                        io = torch.randint(ob.shape[0], (recon_batch,), generator=rgen)
                                        xr = torch.cat([xr, with_ctx(ob[io], oc[io])])
                                        yr = torch.cat([yr, ol[io]])
                                    loss = F.cross_entropy(m_(xr), yr)
                                    ropt.zero_grad(set_to_none=True)
                                    loss.backward()
                                    ropt.step()
                            models[(name, c)] = m_
                    if recon_extra:
                        xgen = torch.Generator().manual_seed(6792)  # 12.6.161節の切り分け専用
                        margin = []
                        for c in used:
                            sb, sl = store(c, True, "none")
                            ob, ol, oc = bb_[bc != c], blab[bc != c], bc[bc != c]
                            with torch.no_grad():
                                lg = model(with_ctx(sb, c))
                                margin.append({"loss": float(F.cross_entropy(lg, sl)) if sb.shape[0] else None,
                                               "acc": float((lg.argmax(-1) == sl).float().mean()) if sb.shape[0] else None,
                                               "p_correct": float(F.softmax(lg, -1).gather(1, sl[:, None]).mean())
                                               if sb.shape[0] else None})
                            for name in RECON_EXTRA_READOUTS:
                                m_ = copy.deepcopy(model)
                                if sb.shape[0] > 0:
                                    ropt = torch.optim.Adam(m_.parameters(), lr=recon_lr)
                                    for _ in range(recon_steps):
                                        idx = torch.randint(sb.shape[0], (recon_batch,), generator=xgen)
                                        xr, yr = with_ctx(sb[idx], c), sl[idx]
                                        if name == "recon_buf_inter" and ob.shape[0] > 0:
                                            io = torch.randint(ob.shape[0], (recon_batch,), generator=xgen)
                                            xr = torch.cat([xr, with_ctx(ob[io], oc[io])])
                                            yr = torch.cat([yr, ol[io]])
                                        lg = m_(xr)
                                        if name == "recon_buf_err":
                                            wrong = lg.argmax(-1) != yr
                                            if not bool(wrong.any()):
                                                continue  # 誤りのないバッチでは学ばない（誤り駆動）
                                            loss = (F.cross_entropy(lg, yr, reduction="none") * wrong).sum() / yr.shape[0]
                                        else:
                                            loss = F.cross_entropy(lg, yr)
                                        ropt.zero_grad(set_to_none=True)
                                        loss.backward()
                                        ropt.step()
                                models[(name, c)] = m_
                        diag["recon_margin"] = margin
                    with torch.no_grad():
                        base = [[float((model(with_ctx(x, c2)).argmax(-1) == y).float().mean()) for c2 in used]
                                for x, y in fresh]
                        diag["recon_side"] = {
                            name: [[[float((models[(name, c)](with_ctx(x, c2)).argmax(-1) == y).float().mean())
                                     - base[j][i2] for i2, c2 in enumerate(used)] for j, (x, y) in enumerate(fresh)]
                                   for c in used] for name in ("recon_all", "recon_inter")}
                epi_acc, epi_cover, purity, arb_rec = [], [], [], []
                for j in range(T):
                    x, y = fresh[j]
                    row_a, row_c, row_p, row_r = [], [], [], []
                    for c in used:
                        with torch.no_grad():
                            pp = F.softmax(model(with_ctx(x, c)), dim=-1)
                        accs = {"param": float((pp.argmax(-1) == y).float().mean())}
                        for name, (ub, hold, rule, nogate) in plans.items():
                            sb, sl = store(None if nogate else c, ub, hold)
                            pe = episodic_readout(pp, x, sb, sl, episodic_lambda, episodic_kappa, rule)
                            accs[name] = float((pe.argmax(-1) == y).float().mean())
                        if episodic_extra:
                            e = ext[c]
                            with torch.no_grad():
                                d_rep = e["scale"] * torch.cdist(model.backbone(with_ctx(x, c)), e["hs"]) ** 2
                            d_bits = (n_bits - x @ e["sb"].T) / 2
                            outs = {}
                            for kind, dd in (("bits", d_bits), ("rep", d_rep)):
                                outs[kind] = {"param": pp, **{r: episodic_readout_dist(pp, dd, e["sl"], episodic_lambda,
                                                                                      episodic_kappa, r)
                                                              for r in ("mix", "switch")}}
                            acc_of = lambda p_: float((p_.argmax(-1) == y).float().mean())  # noqa: E731
                            accs.update(mix_rep=acc_of(outs["rep"]["mix"]), switch_rep=acc_of(outs["rep"]["switch"]),
                                        arb_bits=acc_of(outs["bits"][e["arb"]["bits"]]),
                                        arb_rep=acc_of(outs["rep"][e["arb"]["rep"]]))
                            row_r.append(dict(e["arb"]))
                        if recon_steps > 0:
                            with torch.no_grad():
                                for name in RECON_READOUTS + (RECON_EXTRA_READOUTS if recon_extra else ()):
                                    accs[name] = float((models[(name, c)](with_ctx(x, c)).argmax(-1) == y)
                                                       .float().mean())
                        sb, sl = store(c, True, "small")
                        row_a.append(accs)
                        row_c.append(float(((x @ sb.T) == x.shape[1]).any(1).float().mean()) if sb.shape[0] else 0.0)
                        row_p.append(float((task_labels(specs[j], sb) == sl).float().mean()) if sb.shape[0] else None)
                    epi_acc.append(row_a)
                    epi_cover.append(row_c)
                    purity.append(row_p)
                    arb_rec.append(row_r)
                diag.update(episodic_acc=epi_acc, episodic_cover=epi_cover, store_purity=purity)
                if episodic_extra:
                    diag["episodic_arb"] = arb_rec
        diag["buffer_src"] = {int(c): {int(t): int(((bc == c) & (bs == t)).sum()) for t in bs[bc == c].unique()}
                              for c in bc.unique()}
        out["diag"] = diag
    return out


def assignment_majority(assign: list[tuple[int, int]], n_tasks: int) -> list[int]:
    """各タスクで学習中に最も多く割り当てられた文脈。"""
    out = []
    for k in range(n_tasks):
        counts: dict[int, int] = {}
        for t, c in assign:
            if t == k:
                counts[c] = counts.get(c, 0) + 1
        out.append(max(counts, key=counts.get))
    return out


def assignment_purity(assign: list[tuple[int, int]], n_tasks: int) -> dict[str, float]:
    """各タスクのバッチのうち、そのタスクで最も多く使われた文脈に割り当てられた割合（純度）と、
    タスクごとの最頻文脈が互いに異なるかどうか。"""
    majority: list[int] = []
    pure = []
    for k in range(n_tasks):
        ctxs = [c for t, c in assign if t == k]
        counts: dict[int, int] = {}
        for c in ctxs:
            counts[c] = counts.get(c, 0) + 1
        m = max(counts, key=counts.get)
        majority.append(m)
        pure.append(counts[m] / len(ctxs))
    return {"purity": sum(pure) / n_tasks, "distinct": float(len(set(majority)) == n_tasks)}

"""ステップ2（基底核モジュール, 12.6.10節）の主実験。

フェーズ1で腕0（チャンピオン）を良腕の中で選好させ、フェーズ2で無関係腕への
負報酬を集中投入したときの「チャンピオンの相対的選好の低下（破局的抑制）」を
条件間で比較する。

条件:
    single       … 単一経路・対称学習率。パラメータ数は dual と揃える（統制2）
    single-asym  … 単一経路・非対称学習率（統制5）
    dual         … 直接路（Go）／間接路（NoGo）を分離（本設計）

主指標は `champion_share = pi[0] / sum(pi[良腕])`（良腕どうしの相対的な選好）で
あり、良腕全体の選択確率（`good_prob`）ではない。後者は無関係腕のロジットが
下がるだけで softmax の正規化を通じて機械的に上昇してしまい、経路の設計差とは
無関係な天井効果を生むことが実測で分かった（`bandit.py` のdocstring参照）。

フェーズ1は固定ステップ数ではなく `--phase1-target`（champion_share）に到達した
時点で打ち切る（統制1）。天井まで学習させると、フェーズ2で守るべき余地が残らず
条件間の差が測れなくなる（12.6.5〜12.6.9節と同型の天井効果）。

使い方:
    PYTHONPATH=src python -m neurocortex.experiments.run_basal_ganglia --seeds 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ..bandit import BanditSpec, dummy_context, sample_phase1, sample_phase2_actions, sample_phase2_rewards
from ..basal_ganglia import Critic, DualPathwayPolicy, SinglePathwayPolicy

CONDITIONS = ("single", "single-asym", "dual")


def build_policy(condition: str, d_model: int, n_actions: int):
    if condition == "dual":
        return DualPathwayPolicy(d_model, n_actions)
    return SinglePathwayPolicy(d_model, n_actions, asymmetric_lr=(condition == "single-asym"))


def policy_update(policy, condition: str, x: torch.Tensor, actions: torch.Tensor,
                  advantages: torch.Tensor, args: argparse.Namespace) -> None:
    if condition == "dual":
        policy.update(x, actions, advantages, lr_go=args.lr_go, lr_nogo=args.lr_nogo)
    elif condition == "single-asym":
        policy.update(x, actions, advantages, lr=args.lr, lr_neg=args.lr_neg)
    else:
        policy.update(x, actions, advantages, lr=args.lr)


@torch.no_grad()
def metrics(policy, spec: BanditSpec) -> tuple[float, float]:
    """(champion_share, good_prob) を返す。champion_share が主指標（本文参照）。"""
    logits = policy(dummy_context(1))
    pi = torch.softmax(logits, dim=-1)[0]
    good = pi[: spec.n_good]
    return float(pi[0] / good.sum().clamp_min(1e-12)), float(good.sum())


def run_phase1(policy, critic, condition, spec, args, generator) -> int:
    """フェーズ1を学習させる。

    固定ステップ数まで回すと条件によって収束速度が異なり、速い条件ほど早く
    good_prob=1.0000（softmaxの浮動小数点上の天井）に張り付いてしまう。これは
    12.6.5〜12.6.9節が繰り返し遭遇した「天井効果」と同型で、天井に達した後は
    フェーズ2でどれだけ良腕を守れたかが測れなくなる（守る余地がそもそも無い）。

    そこで **phase1_target に達した時点で打ち切る**（統制1）。全条件を同じ
    到達点で揃えることで、フェーズ1の学習速度の差ではなく、フェーズ2での
    崩れ方の差だけを比較できるようにする。`phase1_max_steps` は目標に到達
    しない場合の打ち切り上限（学習が破綻している条件を無限ループさせない）。
    """
    for step in range(1, args.phase1_max_steps + 1):
        x = dummy_context(args.batch_size)
        with torch.no_grad():
            probs = torch.softmax(policy(x), dim=-1)
            actions = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
        rewards = sample_phase1(spec, actions, generator)
        baseline = critic(x).detach()
        advantages = rewards - baseline
        critic.update(x, rewards, lr=args.critic_lr)
        policy_update(policy, condition, x, actions, advantages, args)
        if step % args.check_every == 0 and metrics(policy, spec)[0] >= args.phase1_target:
            return step
    return args.phase1_max_steps


def run_phase2(policy, critic, condition, spec, args, generator) -> int:
    """フェーズ2を学習させる。

    固定ステップ数まで回すと、無関係腕をすでに十分抑制した後も負の優位度を
    与え続けることになる。REINFORCEはsoftmaxの正規化項 `-advantage*pi(j)` を
    通じて、生存している腕のロジットをその時点の確率に比例して押し上げるため
    （確率が高い腕ほど強く押し上げられる、いわゆる rich-get-richer 的な不安定性）、
    無関係腕がほぼ0まで落ちた後も学習を続けると、経路の設計とは無関係にチャンピオン
    の取り分が暴走的に1.0000へ張り付く。これは統制対象の現象ではなく、REINFORCEに
    エントロピー正則化を入れていない場合に一般に起きる既知の不安定性である。

    そこで **無関係腕の合計確率が phase2_target 以下に落ちた時点で打ち切る**
    （「十分抑制できた」時点で投与を止める）。`phase2_max_steps` は打ち切り上限。
    """
    for step in range(1, args.phase2_max_steps + 1):
        x = dummy_context(args.batch_size)
        actions = sample_phase2_actions(spec, args.batch_size, generator)
        rewards = sample_phase2_rewards(spec, actions, generator)
        baseline = critic(x).detach()
        advantages = rewards - baseline
        critic.update(x, rewards, lr=args.critic_lr)
        policy_update(policy, condition, x, actions, advantages, args)
        if step % args.check_every == 0:
            bad_prob = 1.0 - metrics(policy, spec)[1]
            if bad_prob <= args.phase2_target:
                return step
    return args.phase2_max_steps


def run_one(condition: str, spec: BanditSpec, args: argparse.Namespace, seed: int) -> dict:
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed + 10_000)
    policy = build_policy(condition, d_model=1, n_actions=spec.n_arms)
    critic = Critic(d_model=1)

    phase1_steps_used = run_phase1(policy, critic, condition, spec, args, g)
    champion_before, good_before = metrics(policy, spec)

    phase2_steps_used = run_phase2(policy, critic, condition, spec, args, g)
    champion_after, good_after = metrics(policy, spec)

    return {
        "condition": condition,
        "seed": seed,
        "phase1_steps_used": phase1_steps_used,
        "phase2_steps_used": phase2_steps_used,
        "champion_share_before": champion_before,
        "champion_share_after": champion_after,
        "champion_drop": champion_before - champion_after,
        "good_prob_before": good_before,
        "good_prob_after": good_after,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--n-good", type=int, default=4)
    ap.add_argument("--n-bad", type=int, default=4)
    ap.add_argument("--good-reward-prob", type=float, default=0.1)
    ap.add_argument("--champion-reward-prob", type=float, default=0.3)
    ap.add_argument("--bad-penalty-prob", type=float, default=1.0)
    ap.add_argument("--phase1-target", type=float, default=0.6,
                    help="この champion_share に達した時点でフェーズ1を打ち切る（天井効果対策）")
    ap.add_argument("--phase1-max-steps", type=int, default=20000,
                    help="目標に達しない場合の打ち切り上限")
    ap.add_argument("--check-every", type=int, default=5,
                    help="phase1-target への到達を何ステップおきに確認するか")
    ap.add_argument("--phase2-target", type=float, default=0.05,
                    help="無関係腕の合計確率がここまで落ちた時点でフェーズ2を打ち切る"
                         "（過剰投与によるrich-get-richer暴走の対策）")
    ap.add_argument("--phase2-max-steps", type=int, default=2000,
                    help="目標に達しない場合の打ち切り上限")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.1, help="single の対称学習率")
    ap.add_argument("--lr-neg", type=float, default=0.1,
                    help="single-asym の負優位度側の学習率（統制5）")
    ap.add_argument("--lr-go", type=float, default=0.1)
    ap.add_argument("--lr-nogo", type=float, default=0.1)
    ap.add_argument("--critic-lr", type=float, default=0.1)
    ap.add_argument("--out", type=Path,
                    default=Path("results/basal_ganglia/catastrophic_suppression.json"))
    args = ap.parse_args()

    spec = BanditSpec(n_good=args.n_good, n_bad=args.n_bad,
                      good_reward_prob=args.good_reward_prob,
                      champion_reward_prob=args.champion_reward_prob,
                      bad_penalty_prob=args.bad_penalty_prob)

    t0 = time.time()
    rows: list[dict] = []
    for condition in args.conditions:
        for seed in range(args.seeds):
            row = run_one(condition, spec, args, seed)
            rows.append(row)
            print(f"{condition:>12} seed={seed}: champion before={row['champion_share_before']:.4f}"
                  f" after={row['champion_share_after']:.4f} drop={row['champion_drop']:.4f}"
                  f"  (phase1={row['phase1_steps_used']} phase2={row['phase2_steps_used']})",
                  flush=True)

    summary: dict[str, dict] = {}
    for condition in args.conditions:
        drops = torch.tensor([r["champion_drop"] for r in rows if r["condition"] == condition])
        before = torch.tensor([r["champion_share_before"] for r in rows if r["condition"] == condition])
        steps = torch.tensor([r["phase1_steps_used"] for r in rows if r["condition"] == condition],
                             dtype=torch.float32)
        summary[condition] = {
            "champion_drop_mean": float(drops.mean()),
            "champion_drop_std": float(drops.std(unbiased=False)),
            "champion_drop_min": float(drops.min()),
            "champion_drop_max": float(drops.max()),
            "champion_before_mean": float(before.mean()),
            "champion_before_std": float(before.std(unbiased=False)),
            "phase1_steps_mean": float(steps.mean()),
            "phase1_hit_max_steps": int((steps >= args.phase1_max_steps).sum()),
        }

    # 統制1: フェーズ2直前の到達点が条件間で揃っていることを機械的に確認する。
    # 揃っていなければ、フェーズ2後の差は「学習速度の違い」を測ってしまっている
    # 可能性があり、破局的抑制の比較として無意味になる。
    before_means = [summary[c]["champion_before_mean"] for c in args.conditions]
    control1_ok = (max(before_means) - min(before_means)) < 0.1
    if not control1_ok:
        print("警告: 統制1（フェーズ1到達点の統制）が崩れています。"
              f" champion_before_mean の条件間差: {max(before_means) - min(before_means):.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "説明": "ステップ2 基底核モジュール: フェーズ2（無関係腕への集中的な失敗信号）"
                "によるチャンピオン（良腕内の相対的選好）の破局的抑制の比較（12.6.10節）",
        "条件": vars(args) | {"out": str(args.out)},
        "control1_ok": control1_ok,
        "summary": summary,
        "rows": rows,
        "sec": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n書き出し: {args.out}")
    for condition in args.conditions:
        s = summary[condition]
        print(f"{condition:>12}  champion_drop={s['champion_drop_mean']:.4f}"
              f"±{s['champion_drop_std']:.4f}  (before={s['champion_before_mean']:.4f})")


if __name__ == "__main__":
    main()

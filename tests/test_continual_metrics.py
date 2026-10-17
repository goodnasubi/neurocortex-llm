import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.basal_ganglia_core import ParitySpec
from neurocortex.continual_metrics import forgetting_metrics, normalize
from neurocortex.continual_sequence import run_sequence


def test_no_forgetting_gives_zero_bwt_and_f():
    R = [[1.0, 0.5, 0.5], [1.0, 1.0, 0.5], [1.0, 1.0, 1.0]]
    m = forgetting_metrics(R, chance=0.5)
    assert m["BWT"] == 0 and m["F"] == 0 and m["ACC"] == 1 and m["LA"] == 1


def test_full_forgetting_to_chance():
    R = [[1.0, 0.5], [0.5, 1.0]]
    m = forgetting_metrics(R, chance=0.5)
    assert m["BWT"] == -1 and m["F"] == 1 and m["ACC"] == 0.5


def test_forgetting_uses_best_past_value():
    # タスク0: 学習直後 0.8 → 次で 1.0（後方転移で改善）→ 最後 0.6。F は最高値 1.0 からの低下
    R = [[0.8, 0.5, 0.5], [1.0, 0.9, 0.5], [0.6, 0.9, 1.0]]
    m = forgetting_metrics(R, chance=0.5)
    assert m["F"] == pytest.approx(((1.0 - 0.2) + 0.0) / 2)
    assert m["BWT"] == pytest.approx(((0.2 - 0.6) + 0.0) / 2)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        forgetting_metrics([[1.0]])
    with pytest.raises(ValueError):
        normalize(0.5, 1.0)


def test_sequence_matrix_shape_and_reproducibility():
    specs = [ParitySpec(n_inputs=8, relevant=p) for p in [(0, 1), (2, 3), (4, 5)]]
    kw = dict(pretrain_steps=5, steps=5, eval_n=50, fisher_batches=2)
    for cond in ("naive-finetune", "replay", "ewc"):
        Rs = []
        for _ in range(2):
            torch.manual_seed(0)
            Rs.append(run_sequence(cond, specs, 8, torch.Generator().manual_seed(1),
                                   torch.Generator().manual_seed(2), **kw))
        assert len(Rs[0]) == 3 and all(len(r) == 3 for r in Rs[0])
        assert Rs[0] == Rs[1]


def test_t2_training_matches_step10():
    # T = 2 では学習過程がステップ10の run_condition と一致する（評価乱数だけが異なる）
    from neurocortex.hippocampus_replay import run_condition
    a, b = ParitySpec(8, (0, 1)), ParitySpec(8, (2, 3))
    torch.manual_seed(0)
    r10 = run_condition("replay", 8, 8, a, b, 20, 64, 0.05, 50, 2, 64, 1000.0, 20, 64, 0.05, 4000,
                        torch.Generator().manual_seed(3))
    torch.manual_seed(0)
    R = run_sequence("replay", [a, b], 8, torch.Generator().manual_seed(3), torch.Generator().manual_seed(9),
                     pretrain_steps=20, steps=20, fisher_batches=2, eval_n=4000)
    assert abs(R[1][0] - r10["acc_a"]) < 0.04 and abs(R[1][1] - r10["acc_b"]) < 0.04


def test_capacity_eval_generator_does_not_change_training():
    from neurocortex.hippocampus_capacity import make_task_specs, run_condition
    specs = make_task_specs(3, 6)
    args = (specs, 8, 40, 20, 10, 16, 0.05, 2, 8, 100)
    torch.manual_seed(0)
    a = run_condition("uniform-compress", *args, torch.Generator().manual_seed(1))
    torch.manual_seed(0)
    b = run_condition("uniform-compress", *args, torch.Generator().manual_seed(1),
                      eval_generator=torch.Generator().manual_seed(2))
    assert a.per_task_acc == b.per_task_acc and a.acc_matrix is None
    assert len(b.acc_matrix) == 3 and all(len(r) == 3 for r in b.acc_matrix)


def test_task_id_input_widens_model_input():
    specs = [ParitySpec(n_inputs=8, relevant=p) for p in [(0, 1), (2, 3), (4, 5)]]
    torch.manual_seed(0)
    R = run_sequence("replay", specs, 8, torch.Generator().manual_seed(1), torch.Generator().manual_seed(2),
                     pretrain_steps=3, steps=3, eval_n=20, task_id_input=True)
    assert len(R) == 3 and all(len(r) == 3 for r in R)

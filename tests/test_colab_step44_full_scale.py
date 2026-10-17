import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from colab_step44_full_scale_experiment import BrainInspiredLLMFullScale, FullScaleTrainer, Step444Config

V, H = 50, 24


def _model(phase, **kw):
    torch.manual_seed(0)
    cfg = Step444Config(phase=phase, device="cpu", fp16=False, max_seq_length=16, **kw)
    return cfg, BrainInspiredLLMFullScale(cfg, vocab_size=V, hidden_size=H, num_layers=1).eval()


def _batches(n=4, T=12):
    g = torch.Generator().manual_seed(1)
    x = torch.randint(0, V, (n, T), generator=g)
    mask = torch.ones_like(x)
    mask[0, 8:] = 0
    labels = x.masked_fill(mask == 0, -100)
    return [{"input_ids": x, "labels": labels, "attention_mask": mask}]


def test_backbone_is_causal():
    _, m = _model("A")
    x = torch.randint(0, V, (1, 12))
    x2 = x.clone()
    x2[0, 7:] = (x2[0, 7:] + 1) % V
    with torch.no_grad():
        l1, l2 = m(x)["logits"], m(x2)["logits"]
    assert torch.allclose(l1[0, :7], l2[0, :7], atol=1e-5)


def test_eval_metric_independent_of_phase_and_weights():
    vals = []
    for phase, w in [("A", 0.05), ("C", 1.0)]:
        cfg, m = _model(phase, hippocampus_loss_weight=w)
        vals.append(FullScaleTrainer(m, cfg).evaluate(DataLoader(_batches(), batch_size=None))["val_ce"])
    assert abs(vals[0] - vals[1]) < 1e-5


def test_padding_excluded_from_loss():
    _, m = _model("A")
    b = _batches()[0]
    with torch.no_grad():
        ce = m(b["input_ids"], b["labels"], b["attention_mask"])["backbone_ce"].item()
        b2 = dict(b, input_ids=b["input_ids"].clone())
        b2["input_ids"][0, 8:] = (b2["input_ids"][0, 8:] + 1) % V
        ce2 = m(b2["input_ids"], b["labels"], b["attention_mask"])["backbone_ce"].item()
    assert abs(ce - ce2) < 1e-5


def test_gradient_accumulation_keeps_gradients():
    cfg, m = _model("A", gradient_accumulation_steps=4)
    m.train()
    for mod in m.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.0
        if isinstance(mod, torch.nn.MultiheadAttention):
            mod.dropout = 0.0
    tr = FullScaleTrainer(m, cfg)
    b = _batches()[0]
    tr.train_step(b)
    g1 = m.lm_head.weight.grad.clone()
    tr.train_step(b)
    assert torch.allclose(m.lm_head.weight.grad, 2 * g1, atol=1e-6)


def test_checkpoint_resume_matches_uninterrupted(tmp_path):
    from colab_step44_full_scale_experiment import load_checkpoint, save_checkpoint

    def deterministic(m):
        for mod in m.modules():
            if isinstance(mod, torch.nn.Dropout):
                mod.p = 0.0
            if isinstance(mod, torch.nn.MultiheadAttention):
                mod.dropout = 0.0
        return m

    b = _batches()[0]
    cfg, m = _model("B")
    ref = FullScaleTrainer(deterministic(m), cfg)
    for _ in range(4):
        ref.train_step(b)

    cfg, m = _model("B")
    tr = FullScaleTrainer(deterministic(m), cfg)
    for _ in range(2):
        tr.train_step(b)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, tr, {"val_ppls": [1.0]}, {"epoch": 0, "batch_idx": 2, "epoch_losses": [0.5]})

    cfg, m = _model("B")
    m.lm_head.weight.data.zero_()
    resumed = FullScaleTrainer(deterministic(m), cfg)
    results = {}
    state = load_checkpoint(path, resumed, results)
    assert state["batch_idx"] == 2 and results["val_ppls"] == [1.0]
    for _ in range(2):
        resumed.train_step(b)
    for p1, p2 in zip(ref.model.parameters(), resumed.model.parameters()):
        assert torch.allclose(p1, p2, atol=1e-6)


def test_bg_critic_learns_and_stops_gradient():
    from colab_step44_full_scale_experiment import BasalGangliaHead
    torch.manual_seed(0)
    head = BasalGangliaHead(H, V, mode="td_sg")
    h = torch.randn(2, 10, H, requires_grad=True)
    labels = torch.randint(0, V, (2, 10))
    labels[0, 6:] = -100
    out = head(h, labels)
    # critic の損失だけでは皮質（入力）へ勾配が流れない
    out["critic_loss"].backward()
    assert h.grad is None or h.grad.abs().sum() == 0
    assert head.critic[0].weight.grad.abs().sum() > 0

    opt = torch.optim.Adam(head.critic.parameters(), lr=1e-2)
    first = None
    for _ in range(100):
        opt.zero_grad()
        c = head(h.detach(), labels)["critic_loss"]
        c.backward()
        opt.step()
        first = first if first is not None else c.item()
    assert c.item() < first * 0.5


def test_bg_critic_ignores_padding():
    from colab_step44_full_scale_experiment import BasalGangliaHead
    torch.manual_seed(0)
    head = BasalGangliaHead(H, V, mode="td_sg")
    h = torch.randn(1, 10, H)
    labels = torch.randint(0, V, (1, 10))
    labels[0, 6:] = -100
    a = head(h, labels)["critic_loss"]
    h2 = h.clone()
    h2[0, 7:] = torch.randn(3, H)  # パディング区間の隠れ状態を変えても損失は不変
    b = head(h2, labels)["critic_loss"]
    assert torch.allclose(a, b)


def test_phase_b_trains_bg_critic():
    cfg, m = _model("B")
    out = m(**_batches()[0])
    assert "bg_critic_loss" in out["losses"]


def test_bg_value_init_sets_bias():
    import math
    from colab_step44_full_scale_experiment import BasalGangliaHead
    head = BasalGangliaHead(H, V, value_init=True, gamma=0.9)
    assert abs(head.critic[-1].bias.item() + math.log(V) / 0.1) < 1e-4


def test_bg_ema_baseline_centers_reward():
    from colab_step44_full_scale_experiment import BasalGangliaHead
    torch.manual_seed(0)
    head = BasalGangliaHead(H, V, reward_baseline="ema")
    h = torch.randn(2, 10, H)
    labels = torch.randint(0, V, (2, 10))
    out = head(h, labels)
    assert torch.isfinite(head.ce_baseline)
    # 初回は基準 = バッチ平均 CE なので、報酬（−CE+b）はほぼ平均0になり TD ターゲットの分散は有限
    assert out["td_target_var"] is not None and torch.isfinite(out["td_target_var"])
    b0 = head.ce_baseline.clone()
    head.eval()
    head(h, labels)
    assert torch.equal(b0, head.ce_baseline)  # 評価時は基準を更新しない


def test_bg_default_unchanged():
    cfg, m = _model("B")
    assert cfg.bg_reward_baseline == "none" and not cfg.bg_value_init


def test_td_sg_critic_does_not_change_backbone_updates():
    # 全体一括の勾配クリッピングでは critic の勾配が backbone の更新を縮めていた（2026-10-17）
    g = torch.Generator().manual_seed(1)
    xs = torch.randint(0, V, (20, 4, 12), generator=g)
    emb = {}
    for mode in ("none", "td_sg"):
        torch.manual_seed(0)
        cfg = Step444Config(phase="B", device="cpu", fp16=False, max_seq_length=16,
                            gradient_accumulation_steps=1, bg_critic_mode=mode, learning_rate=1e-3)
        m = BrainInspiredLLMFullScale(cfg, vocab_size=V, hidden_size=H, num_layers=1).eval()
        t = FullScaleTrainer(m, cfg)
        for x in xs:
            t.train_step({"input_ids": x, "labels": x, "attention_mask": torch.ones_like(x)})
        emb[mode] = m.embedding.weight.detach()
    assert torch.allclose(emb["none"], emb["td_sg"], atol=1e-6)

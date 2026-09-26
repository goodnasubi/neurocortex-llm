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


def test_local_wikitext_word_level(tmp_path):
    from colab_step44_full_scale_experiment import WordChunkDataset, load_local_wikitext
    (tmp_path / "train.txt").write_text(" a b a\n c a <unk>\n")
    (tmp_path / "valid.txt").write_text(" a z\n")
    (tmp_path / "test.txt").write_text(" b\n")
    ids, v = load_local_wikitext(str(tmp_path), max_vocab=3)
    assert v == 3  # <unk>, a, <eos>（b・c は語彙外）
    assert ids["train"].tolist() == [1, 0, 1, 2, 0, 1, 0, 2]
    assert ids["valid"].tolist() == [1, 0, 2]
    ds = WordChunkDataset(ids["train"], 3)
    assert len(ds) == 2 and ds[1]["input_ids"].tolist() == [2, 0, 1]

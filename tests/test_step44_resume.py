"""tools/colab/step44_resume.py（Colab 手動再開スクリプト）を CPU の小さな設定で実際に動かすテスト。

仮 Drive・仮 WORK を tmp_path に作り、STEP44_DRIVE / STEP44_WORK で差し替える。
"""
import json
import os
import random
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "tools" / "colab" / "step44_resume.py"


def _write_wikitext(d: Path) -> None:
    rng = random.Random(0)
    words = [f"w{i}" for i in range(40)]
    d.mkdir()
    for name, n_lines in [("train", 40), ("valid", 8), ("test", 8)]:
        lines = [" ".join(rng.choice(words) for _ in range(15)) for _ in range(n_lines)]
        (d / f"{name}.txt").write_text("\n".join(lines) + "\n")


def _run(drive: Path, work: Path, data: Path, *resume_args: str) -> str:
    env = dict(os.environ, STEP44_DRIVE=str(drive), STEP44_WORK=str(work), CUDA_VISIBLE_DEVICES="")
    cmd = [sys.executable, str(SCRIPT), *resume_args, "--sync-seconds", "0.5", "--",
           "--data-dir", str(data), "--max-vocab", "50", "--hidden-size", "16", "--num-layers", "1",
           "--nhead", "2", "--max-seq-length", "16", "--epochs", "3", "--batch-size", "4",
           "--grad-accum", "1", "--phases", "A", "B", "--seeds", "0"]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    out = p.stdout + p.stderr
    assert p.returncode == 0, out
    return out


def test_resume_script_stops_and_resumes(tmp_path):
    drive, work, data = tmp_path / "drive", tmp_path / "work", tmp_path / "data"
    _write_wikitext(data)
    # (a) Phase A seed 0 は Drive に完了済みの結果がある → 飛ばされる
    drive.mkdir()
    done = {"phase": "A", "seed": 0, "val_ppls": [123.0]}
    (drive / "phaseA_seed0.json").write_text(json.dumps(done))

    # (b) --hours 0: 最初の同期で停止ファイル epoch が書かれ、Phase B の1エポック目の末で止まる
    out = _run(drive, work, data, "--hours", "0")
    assert "Skip Phase A seed 0" in out
    assert (work / "STOP").read_text() == "epoch"
    assert "STOPPED_AT_BREAKPOINT" in out and "epoch 1 終了時" in out
    assert (drive / "checkpoints" / "phaseB_seed0.pt").exists()
    assert not (drive / "phaseB_seed0.json").exists()
    assert json.loads((drive / "phaseA_seed0.json").read_text()) == done
    assert (drive / "summary.json").exists()

    # (c) 作業領域を消して（新しい VM を想定）再実行 → Drive から復元し、2エポック目から最後まで走る
    import shutil
    shutil.rmtree(work)
    out = _run(drive, work, data)
    assert "Resumed from" in out and "epoch 2, batch 0" in out
    assert "STOPPED_AT_BREAKPOINT" not in out
    rb = json.loads((drive / "phaseB_seed0.json").read_text())
    assert len(rb["val_ppls"]) == 3 and "test_ppl" in rb
    assert json.loads((drive / "phaseA_seed0.json").read_text()) == done
    summary = json.loads((drive / "summary.json").read_text())
    assert set(summary) == {"A", "B"}
    # 完了した実行のチェックポイントは Drive からも消える
    assert not list((drive / "checkpoints").glob("*.pt"))


def test_copy_dir_keeps_mtime(tmp_path):
    """Drive から戻したファイルが、次の定期同期で Drive へ写し直されない（更新時刻を保つ）。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("step44_resume", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    src, dst = tmp_path / "drive", tmp_path / "local"
    src.mkdir()
    f = src / "phaseA_seed0.pt"
    f.write_bytes(b"x")
    os.utime(f, (1_000_000, 1_000_000))
    mod.copy_dir(str(src), str(dst), force=True)
    assert os.path.getmtime(dst / f.name) == os.path.getmtime(f)
    f.write_bytes(b"changed")
    os.utime(f, (1_000_000, 1_000_000))
    mod.copy_dir(str(dst), str(src))  # 同期（force なし）: 手元は新しくないので写さない
    assert f.read_bytes() == b"changed"
    assert not list(src.glob("*.part")) and not list(dst.glob("*.part"))

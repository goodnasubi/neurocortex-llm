import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurocortex.run_env import env_info  # noqa: E402


def test_env_info_records_versions_and_commit():
    info = env_info()
    for key in ("python", "platform", "torch", "numpy", "git_commit", "utc"):
        assert key in info
    assert info["torch"] and info["python"].count(".") >= 1


def test_colab_scripts_import_env_info():
    import colab_step44_parameter_sweep as sweep  # noqa: F401  （src のパス追加と import が通ること）
    assert callable(sweep.env_info)

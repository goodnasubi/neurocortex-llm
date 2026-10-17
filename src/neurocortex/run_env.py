"""結果ファイルに実行環境を記録するための情報（12.6.114節）。

9月の結果が torch のバージョン差で数値として完全には再現しなかった（12.6.103節）ため、
実験スクリプトは結果の JSON にこの情報を書き込む。
"""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _version(module: str) -> str | None:
    try:
        return __import__(module).__version__
    except Exception:  # 未インストールなど
        return None


def _git_commit() -> str | None:
    try:
        root = Path(__file__).resolve().parents[2]
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        commit = out.stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True,
                               timeout=5).stdout.strip()
        return (commit + ("+dirty" if dirty else "")) or None
    except Exception:
        return None


def env_info() -> dict:
    """Python・OS・主要ライブラリのバージョン、git のコミット、実行日時（UTC）。"""
    info = {
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.machine()}",
        "torch": _version("torch"),
        "numpy": _version("numpy"),
        "git_commit": _git_commit(),
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        import torch
        info["cuda"] = torch.version.cuda if torch.cuda.is_available() else None
    except Exception:
        pass
    return info

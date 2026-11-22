"""Load the packaged JianYing runtime."""

import sys
from pathlib import Path


def setup_env():
    root = Path(__file__).resolve().parents[1]
    vendor = root / "vendor"
    if not (vendor / "pyJianYingDraft" / "__init__.py").is_file():
        raise ImportError(f"缺少内置剪映工程库: {vendor}")
    for directory in (root, vendor):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))

"""Repository path constants.

Default: ``data/`` at the project root (parent of ``src/``). Override with env
``DOCUMENT_IR_DATA_DIR`` for non-editable installs or custom locations.
"""

import os
from pathlib import Path

_data_override = (os.environ.get("DOCUMENT_IR_DATA_DIR") or "").strip()
if _data_override:
    DATA_DIR = Path(_data_override).expanduser().resolve()
    REPO_ROOT = DATA_DIR.parent
else:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = REPO_ROOT / "data"

KB_DIR = DATA_DIR / "kbs"
SQLITE_PATH = DATA_DIR / "kb_registry.db"
CHROMA_DIR = DATA_DIR / "chroma"

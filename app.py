"""Gradio entrypoint: ``python app.py`` from the repo root.

Prefer ``pip install -e .`` so ``document_ir`` is on the path everywhere; if the
package is not installed, ``src/`` is prepended so this script still runs.
"""

import sys
from pathlib import Path

try:
    from document_ir.ui.app import main
except ModuleNotFoundError:
    _src = Path(__file__).resolve().parent / "src"
    sys.path.insert(0, str(_src))
    from document_ir.ui.app import main

if __name__ == "__main__":
    main()

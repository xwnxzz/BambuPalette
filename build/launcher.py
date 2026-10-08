"""Launcher used by the PyInstaller build.

The package is imported as ``app`` (the spec puts the project root on
``pathex``), so relative imports inside the package keep working when frozen.
"""

from __future__ import annotations

import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from app.main import main

    sys.exit(main())

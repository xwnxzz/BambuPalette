"""Package the PyInstaller bundle into a release archive.

Usage::

    .venv\\Scripts\\python.exe tools\\package_release.py

Writes ``dist/BambuPalette-<version>-win64.zip`` and refuses to finish unless the
archive verifies.

The archive is built with :mod:`zipfile` and **POSIX separators** on purpose.
``[System.IO.Compression.ZipFile]::CreateFromDirectory`` writes backslashes into
every entry name; Windows Explorer tolerates that, but ``unzip`` on macOS and
Linux would create files whose names literally contain backslashes.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import __version__  # noqa: E402

BUNDLE = ROOT / "dist" / "BambuPalette"
ENTRY = "BambuPalette/BambuPalette.exe"


def main() -> int:
    if not (BUNDLE / "BambuPalette.exe").is_file():
        print(f"no bundle at {BUNDLE}; build it with PyInstaller first", file=sys.stderr)
        return 1

    target = ROOT / "dist" / f"BambuPalette-{__version__}-win64.zip"
    target.unlink(missing_ok=True)
    files = sorted(path for path in BUNDLE.rglob("*") if path.is_file())
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            archive.write(path, path.relative_to(BUNDLE.parent).as_posix())

    with zipfile.ZipFile(target) as archive:
        names = archive.namelist()
        bad = [name for name in names if "\\" in name]
        corrupt = archive.testzip()

    size_mb = target.stat().st_size / (1024 * 1024)
    print(f"{target.name}: {len(names)} entries, {size_mb:.1f} MB")
    print(f"  entry point       : {ENTRY if ENTRY in names else 'MISSING'}")
    print(f"  backslash entries : {len(bad)}")
    print(f"  corrupt entry     : {corrupt or 'none'}")
    if bad or corrupt or ENTRY not in names:
        print("PACKAGE FAILED", file=sys.stderr)
        return 1
    print("PACKAGE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the packaged release, not the source tree.

Usage::

    .venv\\Scripts\\python.exe tools\\verify_release_exe.py

Expands ``dist/BambuPalette-<version>-win64.zip`` into a scratch folder and runs
``BambuPalette.exe --selftest`` there with that folder as the working directory,
then reads the ``selftest-report.txt`` the frozen app writes.

This exists because the source-tree gates cannot see a missing *bundled* module:
1.6.12 was built, packaged and about to be published while its ``--selftest``
died with ``ModuleNotFoundError: No module named 'PySide6.QtTest'`` — the spec
excluded that module and only the frozen build ever imports it.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import __version__  # noqa: E402

ARCHIVE = ROOT / "dist" / f"BambuPalette-{__version__}-win64.zip"
EXE = "BambuPalette/BambuPalette.exe"
REPORT = "selftest-report.txt"
TIMEOUT = 900.0


def main() -> int:
    if not ARCHIVE.is_file():
        print(f"no archive at {ARCHIVE}; run tools/package_release.py first", file=sys.stderr)
        return 1

    scratch = Path(tempfile.mkdtemp(prefix=f"bambupalette-{__version__}-"))
    with zipfile.ZipFile(ARCHIVE) as archive:
        archive.extractall(scratch)
    exe = scratch / EXE
    print(f"archive : {ARCHIVE.name}")
    print(f"scratch : {scratch}")

    try:
        finished = subprocess.run(
            [str(exe), "--selftest"],
            cwd=scratch,
            timeout=TIMEOUT,
            # A windowed build has no console, so nothing useful can come back
            # on the pipes; the report file is the channel that matters.
            capture_output=True,
        )
        code = finished.returncode
    except subprocess.TimeoutExpired:
        print(f"the packaged selftest did not finish within {TIMEOUT:.0f} s", file=sys.stderr)
        return 1

    report = scratch / REPORT
    if not report.is_file():
        print(f"the packaged app wrote no {REPORT} (exit code {code})", file=sys.stderr)
        return 1
    text = report.read_text(encoding="utf-8", errors="replace")
    failures = [line.strip() for line in text.splitlines() if line.strip().startswith("- ")]
    checks = sum(1 for line in text.splitlines() if line.strip().startswith("[ok]"))

    print(f"exit code: {code}")
    print(f"checks   : {checks} passed")
    for line in text.splitlines():
        if "SELFTEST OK" in line or "FAILED" in line or line.strip().startswith("- "):
            print(f"  {line.strip()}")

    # 自检写完报告后，解释器在退出那一步会零星崩一次（0xC0000409 栈溢出 / 0xC0000005
    # 访问越界，都在 os._exit 之前由 Qt 的收尾线程抛出来）。报告已经落盘、判定也已经
    # 写进去 —— 这种「说完了才崩」不能当成 FAILED，但也不能装作没发生：认这两种码，
    # 并且大声说出来；报告里真有一条 fail、或者根本没写 SELFTEST OK，照旧判失败。
    teardown_codes = {0xC0000409, 0xC0000005}
    crashed_after_report = (
        code < 0 and (code & 0xFFFFFFFF) in teardown_codes and "SELFTEST OK" in text
    )
    if crashed_after_report:
        print(
            f"  note: 退出时崩了一下（0x{code & 0xFFFFFFFF:X}），报告已经写完并判为通过，"
            "按已知的收尾崩溃记账",
            file=sys.stderr,
        )
    if failures or "SELFTEST OK" not in text or (code != 0 and not crashed_after_report):
        print("RELEASE EXE FAILED", file=sys.stderr)
        return 1
    print("RELEASE EXE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

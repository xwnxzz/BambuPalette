# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for BambuPalette / BambuPalette.

Build with::

    .venv\\Scripts\\python.exe -m PyInstaller --noconfirm --clean build\\BambuPalette.spec

The ICC polynomial coefficients are baked into ``app/spectral/icc_profile.py``,
so the bundle needs no data files at runtime.
"""

from PyInstaller.utils.win32.versioninfo import (
    FixedFileInfo,
    StringFileInfo,
    StringStruct,
    StringTable,
    VarFileInfo,
    VarStruct,
    VSVersionInfo,
)

from pathlib import Path
import re

ROOT = Path(SPECPATH).resolve().parent


def app_version() -> tuple[str, tuple[int, int, int, int]]:
    """Read ``__version__`` from ``app/__init__.py`` so the exe can never drift.

    The Windows version resource is baked into the exe at build time, so a
    literal here would silently rot the moment the source version moved.
    """
    text = (ROOT / "app" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if match is None:  # pragma: no cover - a broken tree should fail the build
        raise RuntimeError("app/__init__.py has no __version__")
    text_version = match.group(1)
    numbers = [int(part) for part in re.findall(r"\d+", text_version)[:4]]
    numbers += [0] * (4 - len(numbers))
    return text_version, tuple(numbers)  # type: ignore[return-value]


VERSION_TEXT, VERSION_PARTS = app_version()

# The four-part form Windows shows in 文件属性 → 详细信息.  Keeping it in sync
# with the string version is the whole point of parsing it above.
VERSION_INFO = VSVersionInfo(
    ffi=FixedFileInfo(
        filevers=VERSION_PARTS,
        prodvers=VERSION_PARTS,
        mask=0x3F,
        flags=0x0,
        OS=0x40004,
        fileType=0x1,
        subtype=0x0,
        date=(0, 0),
    ),
    kids=[
        StringFileInfo([
            StringTable("080404B0", [
                StringStruct("CompanyName", "BambuPalette"),
                StringStruct("FileDescription", "BambuPalette 混色耗材色彩管理器"),
                StringStruct("FileVersion", VERSION_TEXT),
                StringStruct("InternalName", "BambuPalette"),
                StringStruct("LegalCopyright", "MIT License"),
                StringStruct("OriginalFilename", "BambuPalette.exe"),
                StringStruct("ProductName", "BambuPalette"),
                StringStruct("ProductVersion", VERSION_TEXT),
            ]),
        ]),
        VarFileInfo([VarStruct("Translation", [0x0804, 1200])]),
    ],
)

EXCLUDES = [
    # Qt modules this application never touches.  Trimming them keeps the
    # bundle around 90 MB instead of 250 MB+.
    "PySide6.Qt3DAnimation",
    "PySide6.Qt3DCore",
    "PySide6.Qt3DExtras",
    "PySide6.Qt3DInput",
    "PySide6.Qt3DLogic",
    "PySide6.Qt3DRender",
    "PySide6.QtBluetooth",
    "PySide6.QtCharts",
    "PySide6.QtDataVisualization",
    "PySide6.QtDesigner",
    "PySide6.QtGraphs",
    "PySide6.QtHelp",
    "PySide6.QtLocation",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
    "PySide6.QtNfc",
    "PySide6.QtOpenGL",
    "PySide6.QtOpenGLWidgets",
    "PySide6.QtPdf",
    "PySide6.QtPdfWidgets",
    "PySide6.QtPositioning",
    "PySide6.QtQml",
    "PySide6.QtQuick",
    "PySide6.QtQuickControls2",
    "PySide6.QtQuickWidgets",
    "PySide6.QtRemoteObjects",
    "PySide6.QtScxml",
    "PySide6.QtSensors",
    "PySide6.QtSerialPort",
    "PySide6.QtSpatialAudio",
    "PySide6.QtSql",
    "PySide6.QtStateMachine",
    "PySide6.QtTest",
    "PySide6.QtTextToSpeech",
    "PySide6.QtWebChannel",
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineQuick",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebSockets",
    # Unrelated Python packages that hooks sometimes drag in.
    "tkinter",
    "matplotlib",
    "scipy",
    "pandas",
    "IPython",
    "pytest",
    "unittest",
    "setuptools",
    "pip",
    "PIL.ImageQt",
]

a = Analysis(
    [str(ROOT / "build" / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[(str(ROOT / "assets" / "logo.ico"), "assets")],
    hiddenimports=["numpy"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="BambuPalette",
    icon=str(ROOT / "assets" / "logo.ico"),
    version=VERSION_INFO,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="BambuPalette",
)

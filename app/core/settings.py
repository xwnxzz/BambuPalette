"""Persisted user preferences.

Three things are remembered: which mixing engine the user picked, how a click
in 「预览」 chooses a colour, and which 耗材种类 the library is currently filtered
to.  The Bambu 2.8 pigment engine is the shipped default because it reproduces
what the user's installed Bambu Studio paints; the legacy sRGB engine and the
Kubelka-Munk spectral engine are first-class choices, so the selection is
remembered instead of resetting on every launch.

``version`` exists because the meaning of the default moved once already: before
build 02.08 the id ``"bambu"`` named the sRGB average and was also the shipped
default, so a stored ``"bambu"`` cannot be told apart from "the user never
touched it".  A file without the current version is therefore migrated forward,
while a file that carries a version is obeyed exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import paths
from .atomicio import atomic_write_text
from .engines import DEFAULT_ENGINE, resolve_engine_id

__all__ = [
    "settings_path",
    "load_settings",
    "save_settings",
    "DEFAULT_SETTINGS",
    "DEFAULT_MATERIAL",
    "SETTINGS_VERSION",
    "PICK_COLOUR",
    "PICK_EDGE",
    "PICK_MODES",
    "DEFAULT_PICK_MODE",
    "resolve_pick_mode",
]

#: Bumped whenever a stored value needs reinterpreting.
SETTINGS_VERSION = 2

#: 「点图片时」 how a click in 预览 chooses a colour: the block the pixel belongs
#: to (「按颜色选择」), or the block whose edge is nearest to the cursor
#: (「按色块边缘选择」, which is how a one-pixel brow can be hit at all).  The user
#: picks between them and gets whichever they used last (m01411 item 4).
PICK_COLOUR = "colour"
PICK_EDGE = "edge"
PICK_MODES = (PICK_COLOUR, PICK_EDGE)
DEFAULT_PICK_MODE = PICK_COLOUR

DEFAULT_SETTINGS: dict = {
    "engine": DEFAULT_ENGINE,
    "pick_mode": DEFAULT_PICK_MODE,
    "material": "",
    "version": SETTINGS_VERSION,
}

#: 上次按哪种耗材配色 (m05557 item 1). 空串＝不按材质过滤，和库层的
#: ``library.ANY_MATERIAL`` 是同一个哨兵；settings 这层只认字符串，不去 import
#: 库（那会把 numpy/光谱那一摞拖进来），所以这个字面量在这儿再写一遍。
DEFAULT_MATERIAL = ""

#: Engine ids that used to be the shipped default.  A pre-version file holding
#: one of these is migrated to the current default rather than kept.
_PRE_VERSION_DEFAULT_IDS = frozenset({"bambu"})


def resolve_pick_mode(value: object) -> str:
    """A stored pick mode, or the default when it is missing or unknown."""
    return value if value in PICK_MODES else DEFAULT_PICK_MODE


def settings_path() -> Path:
    """JSON file holding the persisted preferences."""
    return paths.data_dir() / "settings.json"


def _migrate_engine(engine: object, version: object) -> str:
    """Resolve a stored engine id, migrating pre-version files forward."""
    if not (isinstance(version, int) and version >= SETTINGS_VERSION):
        if engine in _PRE_VERSION_DEFAULT_IDS:
            return DEFAULT_ENGINE
    return resolve_engine_id(engine) or DEFAULT_ENGINE


def load_settings() -> dict:
    """Preferences on disk, falling back to the defaults for anything missing.

    Never raises: a corrupt or unreadable file must not stop the application
    from starting, so it is treated as "no preferences yet".
    """
    values = dict(DEFAULT_SETTINGS)
    try:
        raw = json.loads(settings_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return values
    if not isinstance(raw, dict):
        return values
    values["engine"] = _migrate_engine(raw.get("engine"), raw.get("version"))
    values["pick_mode"] = resolve_pick_mode(raw.get("pick_mode"))
    values["material"] = str(raw.get("material") or "").strip()
    return values


def save_settings(values: dict) -> Path:
    """Merge ``values`` into the stored preferences and write them back."""
    merged = load_settings()
    merged.update(values)
    merged["version"] = SETTINGS_VERSION
    target = settings_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    # 先写临时文件再改名（1.6.50）：偏好是一份小 JSON，但被截断也等于「设置全丢」，
    # 而这条路径在窗口构造与每次切页时都会走到。
    atomic_write_text(
        target, json.dumps(merged, ensure_ascii=False, indent=2) + "\n"
    )
    return target

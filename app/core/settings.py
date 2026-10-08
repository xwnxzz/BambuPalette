"""Persisted user preferences.

Currently one thing is remembered: which mixing engine the user picked.  The
Bambu 2.8 pigment engine is the shipped default because it reproduces what the
user's installed Bambu Studio paints; the legacy sRGB engine and the
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
from .engines import DEFAULT_ENGINE, resolve_engine_id

__all__ = [
    "settings_path",
    "load_settings",
    "save_settings",
    "DEFAULT_SETTINGS",
    "SETTINGS_VERSION",
]

#: Bumped whenever a stored value needs reinterpreting.
SETTINGS_VERSION = 2

DEFAULT_SETTINGS: dict = {"engine": DEFAULT_ENGINE, "version": SETTINGS_VERSION}

#: Engine ids that used to be the shipped default.  A pre-version file holding
#: one of these is migrated to the current default rather than kept.
_PRE_VERSION_DEFAULT_IDS = frozenset({"bambu"})


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
    return values


def save_settings(values: dict) -> Path:
    """Merge ``values`` into the stored preferences and write them back."""
    merged = load_settings()
    merged.update(values)
    merged["version"] = SETTINGS_VERSION
    target = settings_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return target

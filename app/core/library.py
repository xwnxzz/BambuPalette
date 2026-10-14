"""Filament library: the user's own spools, plus JSON persistence.

A filament record carries everything the user types for a spool:

* ``brand``          — 品牌, e.g. ``大简``
* ``material_type``  — 耗材种类, e.g. ``PETG HF``
* ``color_hex``      — the measured / declared colour of the printed material
* ``note``           — free text 备注
* ``name``           — optional display name, derived when left empty
* ``hidden``         — 已用完、还没买回来的料：留在表里，但不参与任何混色

``hidden`` is the "the spool is empty and the replacement has not arrived"
switch.  A hidden spool keeps its row (the user still has to see it to unhide
it) but contributes no colour anywhere: it is not offered as an ingredient of a
two- or three-filament mix, and its own colour is not one of 「全部颜色」.  The
filtering happens where the catalogues are built — see
:attr:`FilamentLibrary.visible` — so every consumer agrees by construction.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

from ..spectral import color as _color
from . import paths

LIBRARY_FORMAT_VERSION = 2

#: ``FilamentLibrary.material`` 取这个值 = 不过滤材质（库里还没有耗材、或者调用方
#: 根本不关心材质时都是它）。
ANY_MATERIAL = ""

#: Suggested 耗材种类 values offered by the UI combo box.
COMMON_MATERIAL_TYPES = (
    "PLA",
    "PLA Basic",
    "PLA Matte",
    "PLA Silk",
    "PLA-CF",
    "PETG",
    "PETG HF",
    "PETG-CF",
    "ABS",
    "ASA",
    "TPU",
    "PA",
    "PA-CF",
    "PC",
    "PVA",
    "HIPS",
)

#: Suggested 品牌 values; the combo box stays editable.
COMMON_BRANDS = (
    "Bambu Lab",
    "大简",
    "Sunlu",
    "eSun",
    "Polymaker",
    "JAYO",
    "Creality",
    "Overture",
    "Elegoo",
    "Anycubic",
    "Kexcelled",
    "R3D",
)


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _write_temp_file(target: Path, payload: str, attempts: int = 8) -> Path:
    """Write ``payload`` next to *target* and return the temporary path.

    :func:`tempfile.mkstemp` is deliberately not used.  On Windows its worker
    treats *any* persistent refusal that arrives while the chosen directory is a
    directory we think we can write to as a name collision:

        except PermissionError:
            if _os.name == 'nt' and _os.path.isdir(dir) and _os.access(dir, _os.W_OK):
                continue

    ``os.access`` only consults the DACL, so a directory that refuses new files
    because of a mandatory integrity label, a "delete pending" state left by an
    interrupted delete, or an anti-virus lock looks writable, and the loop runs
    up to ``TMP_MAX`` = 2147483647 times.  A denied save then presents as the
    application failing to start at all, spinning one core forever, because the
    window saves the library while it is still being constructed.  A bounded
    ``open(..., "x")`` loop raises :class:`LibraryError` immediately instead.
    """
    for attempt in range(attempts):
        candidate = target.with_name(f"{target.name}.{os.getpid()}.{attempt}.tmp")
        try:
            with open(candidate, "x", encoding="utf-8", newline="\n") as stream:
                stream.write(payload)
        except FileExistsError:
            continue
        except OSError as exc:
            raise LibraryError(
                f"cannot write library to {target}: {exc.strerror or exc}"
            ) from exc
        return candidate
    raise LibraryError(f"cannot write library to {target}: 临时文件名重复")


def _discard(path: Path) -> None:
    """Best-effort removal of a temporary file we are giving up on."""
    try:
        path.unlink()
    except OSError:  # pragma: no cover - defensive
        pass


@dataclass
class Filament:
    """One spool of filament with a known printed colour."""

    id: str = field(default_factory=_new_id)
    brand: str = ""
    material_type: str = ""
    #: The colour this spool actually prints as.  The default exists only so the
    #: dataclass can hold a value; every path that creates a spool for the user
    #: supplies one, because a made-up white spool is a lie the user only finds
    #: out about after a print.
    color_hex: str = "#FFFFFF"
    note: str = ""
    name: str = ""
    #: 料用完了还没买回来，就先藏起来。 行还在，颜色不在：它不进任何混色，自己
    #: 的颜色也不出现在「全部颜色」。 见 FilamentLibrary.visible。
    hidden: bool = False
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        self.color_hex = _color.normalize_hex(self.color_hex)
        self.brand = (self.brand or "").strip()
        self.material_type = (self.material_type or "").strip()
        self.note = (self.note or "").strip()
        self.name = (self.name or "").strip()
        self.hidden = bool(self.hidden)

    # -- derived ------------------------------------------------------------------
    @property
    def display_name(self) -> str:
        """Human label: explicit name, else ``brand type``, else ``brand``, else hex."""
        if self.name:
            return self.name
        parts = [p for p in (self.brand, self.material_type) if p]
        if parts:
            return " ".join(parts)
        return self.color_hex

    @property
    def subtitle(self) -> str:
        """Secondary label used under the display name in list rows."""
        parts = [self.color_hex]
        if self.brand:
            parts.append(self.brand)
        if self.material_type:
            parts.append(self.material_type)
        return " · ".join(parts)

    @property
    def rgb(self) -> tuple[int, int, int]:
        return _color.hex_to_rgb(self.color_hex)

    def to_dict(self) -> dict:
        data = {
            "id": self.id,
            "name": self.name,
            "brand": self.brand,
            "materialType": self.material_type,
            "colorHex": self.color_hex,
            "note": self.note,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
        }
        # 只在真的藏起来时才写这个键：老档案、和别人共用的档案里，没隐藏过的料
        # 一个字节都不变。
        if self.hidden:
            data["hidden"] = True
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Filament":
        return cls(
            id=str(data.get("id") or _new_id()),
            name=str(data.get("name") or ""),
            brand=str(data.get("brand") or ""),
            material_type=str(data.get("materialType") or data.get("material_type") or ""),
            color_hex=str(data.get("colorHex") or data.get("color_hex") or ""),
            note=str(data.get("note") or ""),
            hidden=bool(data.get("hidden")),
            created_at=str(data.get("createdAt") or _now()),
            updated_at=str(data.get("updatedAt") or _now()),
        )

    def copy(self, **changes) -> "Filament":
        updated = replace(self, **changes)
        updated.updated_at = _now()
        return updated

    def touch(self) -> None:
        self.updated_at = _now()


class LibraryError(RuntimeError):
    """Raised when the library file cannot be read or written."""


class FilamentLibrary:
    """Ordered collection of :class:`Filament` records with JSON persistence.

    Besides the spools themselves the library carries the one piece of state
    the whole app shares: :attr:`material`, the 耗材种类 the colours are
    currently being drawn from.  Two different plastics cannot be blended into
    one extruder — the printer would be asked to print a mixture that does not
    exist — so the list every catalogue is built from (:attr:`visible`) is
    filtered by it, and switching it in the 「我的耗材」 panel is what makes every
    colour on every page follow that material (m05557 item 1).
    """

    def __init__(
        self,
        filaments: Iterable[Filament] | None = None,
        *,
        material: str = ANY_MATERIAL,
    ) -> None:
        self._filaments: list[Filament] = list(filaments or [])
        self._index: dict[str, Filament] = {f.id: f for f in self._filaments}
        self._material = str(material or "").strip()

    # -- container protocol -------------------------------------------------------
    def __len__(self) -> int:
        return len(self._filaments)

    def __iter__(self) -> Iterator[Filament]:
        return iter(self._filaments)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._filaments[item]
        return self._index[item]

    def __contains__(self, item) -> bool:
        key = item.id if isinstance(item, Filament) else item
        return key in self._index

    @property
    def filaments(self) -> list[Filament]:
        return list(self._filaments)

    @property
    def material(self) -> str:
        """The 耗材种类 the colours come from (``""`` = every material)."""
        return self._material

    @material.setter
    def material(self, value: str) -> None:
        self._material = str(value or "").strip()

    def accepts(self, filament: Filament) -> bool:
        """Whether this spool is part of the current material's list.

        ``ANY_MATERIAL`` accepts everything, which is what every library built
        without a material gets — so nothing that does not care about the
        switch (tests, the exporter, a project being opened) changes behaviour.
        """
        if not self._material:
            return True
        return filament.material_type.casefold() == self._material.casefold()

    @property
    def visible(self) -> list[Filament]:
        """The spools that may take part in mixing — not hidden, right material.

        Every catalogue is built from this list rather than from
        :attr:`filaments`, which is what makes 「隐藏」 mean the same thing on
        all three pages at once: a hidden spool is never an ingredient, so no
        two-filament mix, no three-filament mix and no spool colour of its own
        can be produced.  Filtering once here beats filtering in the matcher,
        the grid, 「全部颜色」 and the exporter separately, where the four could
        drift apart.

        :attr:`material` rides the same list, for the same reason: 「PETG 不能
        和 PLA 混色」 is a rule about ingredients, and the one place that decides
        what an ingredient is answers it for the matcher, the mixes, the triples
        and the 3MF exporter at once (m05557 item 1).
        """
        return [
            filament
            for filament in self._filaments
            if not filament.hidden and self.accepts(filament)
        ]

    @property
    def hidden_count(self) -> int:
        return sum(1 for filament in self._filaments if filament.hidden)

    @property
    def other_material_count(self) -> int:
        """Spools that are usable but are another 耗材种类 (they come back on switch)."""
        return sum(
            1 for filament in self._filaments if not filament.hidden and not self.accepts(filament)
        )

    def materials(self) -> list[str]:
        """Every 耗材种类 in the library, most spools first, ties alphabetically.

        Hidden spools do not count — a material whose every spool is hidden is
        not a material the user can switch to and get colours out of.
        """
        counts: dict[str, int] = {}
        for filament in self._filaments:
            if filament.hidden:
                continue
            name = (filament.material_type or "").strip()
            if name:
                counts[name] = counts.get(name, 0) + 1
        return sorted(counts, key=lambda name: (-counts[name], name.casefold()))

    def get(self, filament_id: str) -> Filament | None:
        return self._index.get(filament_id)

    def require(self, filament_id: str) -> Filament:
        filament = self._index.get(filament_id)
        if filament is None:
            raise KeyError(f"unknown filament id {filament_id!r}")
        return filament

    # -- mutation -----------------------------------------------------------------
    def add(self, filament: Filament) -> Filament:
        if filament.id in self._index:
            filament.id = _new_id()
        self._filaments.append(filament)
        self._index[filament.id] = filament
        return filament

    def create(self, **fields) -> Filament:
        return self.add(Filament(**fields))

    def update(self, filament_id: str, **changes) -> Filament:
        existing = self.require(filament_id)
        updated = existing.copy(**changes)
        position = self._filaments.index(existing)
        self._filaments[position] = updated
        self._index[filament_id] = updated
        return updated

    def remove(self, filament_id: str) -> Filament:
        filament = self.require(filament_id)
        self._filaments.remove(filament)
        del self._index[filament_id]
        return filament

    def move(self, filament_id: str, new_index: int) -> None:
        filament = self.require(filament_id)
        self._filaments.remove(filament)
        self._filaments.insert(max(0, min(new_index, len(self._filaments))), filament)

    def clear(self) -> None:
        self._filaments.clear()
        self._index.clear()

    def replace_all(self, filaments: Iterable[Filament]) -> None:
        self._filaments = list(filaments)
        self._index = {f.id: f for f in self._filaments}

    # -- queries ------------------------------------------------------------------
    def find_duplicate(self, color_hex: str, material_type: str = "", brand: str = "") -> Filament | None:
        """Return an existing spool with the same colour/type/brand, if any."""
        target = _color.normalize_hex(color_hex)
        for filament in self._filaments:
            if filament.color_hex != target:
                continue
            if material_type and filament.material_type.casefold() != material_type.strip().casefold():
                continue
            if brand and filament.brand.casefold() != brand.strip().casefold():
                continue
            return filament
        return None

    def sort_by_color(self) -> list[Filament]:
        return sorted(self._filaments, key=lambda f: (f.rgb[0] << 16) | (f.rgb[1] << 8) | f.rgb[2])

    def to_dict(self) -> dict:
        return {
            "formatVersion": LIBRARY_FORMAT_VERSION,
            "savedAt": _now(),
            "filaments": [f.to_dict() for f in self._filaments],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FilamentLibrary":
        raw = data.get("filaments")
        if not isinstance(raw, list):
            raise LibraryError("library file has no 'filaments' list")
        filaments: list[Filament] = []
        seen: set[str] = set()
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            try:
                filament = Filament.from_dict(entry)
            except (ValueError, TypeError):
                # A row with no usable colour is not a spool. Skipping it keeps one
                # damaged entry from taking the whole library down with it.
                continue
            if filament.id in seen:
                filament.id = _new_id()
            seen.add(filament.id)
            filaments.append(filament)
        return cls(filaments)

    # -- persistence --------------------------------------------------------------
    def save(self, path: Path | str | None = None) -> Path:
        target = Path(path) if path is not None else paths.library_path()
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise LibraryError(
                f"cannot create {target.parent}: {exc.strerror or exc}"
            ) from exc

        # Keep the previous version beside the new one.  Saving is atomic, so the
        # old file is never *corrupt*; but "atomic" says nothing about whether
        # what just got written was what the user meant.  One generation of undo
        # costs a few kilobytes and turns an accident into an inconvenience.
        if target.exists():
            try:
                shutil.copy2(target, target.with_name(target.name + ".bak"))
            except OSError:  # pragma: no cover - defensive
                pass

        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        temp_name = _write_temp_file(target, payload)
        try:
            # Atomic replace so a crash cannot truncate the user's library.
            os.replace(temp_name, target)
        except OSError as exc:
            _discard(temp_name)
            raise LibraryError(f"cannot write library to {target}: {exc}") from exc
        return target

    @classmethod
    def load(cls, path: Path | str | None = None) -> "FilamentLibrary":
        source = Path(path) if path is not None else paths.library_path()
        if not source.exists():
            return cls()
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LibraryError(f"cannot read library from {source}: {exc}") from exc
        if not isinstance(data, dict):
            raise LibraryError(f"library file {source} is not a JSON object")
        return cls.from_dict(data)

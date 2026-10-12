"""保存 / 打开一个「修图工程」——一张图加它被改过的一切。

用户的原话是「导出当前已经被修改过的图片的文件……方便用户下次打开继续修图，而不是从头
开始修」。要做到这一点，文件里必须带三样东西：

* **原图**——不然没法重新匹配；
* **基线匹配**（每像素归属哪个颜色，加上它用到的调色板）——用户的删除 / 合并 / 更换
  都是记在基线下标上的，没有基线就没有这些编辑；
* **编辑本身**（删了哪些、哪些合并到哪、哪些换成了什么）。

反过来说，文件**不能**依赖这台机器的耗材库：换台机器打开、或者用户清空了耗材库，工程
里那几种颜色和那几卷耗材照样得显示出来。所以颜色和耗材都以可读的清单写进 JSON，缺哪卷
就照实标出来，而不是假装它还在。

一个工程 = 一个 zip：

* ``工程.json`` —— 设置、基线调色板、编辑，以及两张给人看的清单
  （这个模型需要的颜色 / 用到的耗材）；
* ``原图.png`` —— 导入的那张图，原样存；
* ``预览.png`` —— 保存时的预览图；
* ``基线.npy`` —— 基线的每像素颜色下标（H×W int32）。
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from ..mesh.plate import PlateSettings
from ..spectral import color as _color
from .image_matching import (
    MatchResult,
    MatchSettings,
    PaletteEntry,
    spool_usage,
)
from .library import FilamentLibrary
from .mixes import DEFAULT_ENGINE, MixRecipe

#: 工程文件的扩展名与文件对话框过滤器。扩展名不跟着版本走：一个工程文件就该在
#: 1.6.x 的每一个小版本之间都能打开，格式号（``PROJECT_FORMAT``）才是把关的地方。
PROJECT_SUFFIX = ".bpproj"
PROJECT_FILTER = "BambuPalette 工程 (*.bpproj);;所有文件 (*)"
#: 格式版本。读的时候严格相等：不认识的文件宁可报错，也不要猜着读一半。
PROJECT_FORMAT = 1

_DOCUMENT = "工程.json"
_SOURCE = "原图.png"
_PREVIEW = "预览.png"
_INDICES = "基线.npy"

#: 工程里记的初始值，和 :class:`MatchSettings` 的字段一一对应。只写这几个：其余
#: （alpha_threshold / merge_delta_e / smooth_radius）用户改不到，写进文件只会变成
#: 一份需要维护的副本。
_SETTING_KEYS = ("max_colours", "max_dimension", "dither", "max_spools")
#: :class:`PlateSettings` 的全部字段。
_PLATE_KEYS = (
    "target_width_mm",
    "base_thickness_mm",
    "colour_thickness_mm",
    "max_footprint_mm",
    "min_pixel_mm",
)


class ProjectError(Exception):
    """A project file that cannot be read back."""


@dataclass
class ProjectState:
    """What :func:`load_project` hands back to the picture page."""

    image: Image.Image
    image_name: str
    base: MatchResult
    deleted: set[int] = field(default_factory=set)
    merged: dict[int, int] = field(default_factory=dict)
    replaced: dict[int, PaletteEntry] = field(default_factory=dict)
    plate: PlateSettings = field(default_factory=PlateSettings)
    engine: str = DEFAULT_ENGINE
    #: 匹配时选的是哪一套耗材搭配（``BudgetOption.key``）。文件里的颜色表本来就是那
    #: 一套的结果，这个 key 只是让「打开工程」能说出当初选的是哪一套；没有这个键的
    #: 旧文件读成空字符串，页面照旧按文件里的颜色表显示。
    plan: str = ""
    #: 保存时那张「用到的耗材」清单，读回来只用于显示与核对（真正的耗材归属跟着
    #: 调色板走，耗材库里没有的那一卷会照实标出来）。
    spools: list[dict] = field(default_factory=list)
    colours: list[dict] = field(default_factory=list)
    saved_at: str = ""
    version: str = ""


# ---------------------------------------------------------------------------
# 颜色条目 <-> JSON
# ---------------------------------------------------------------------------
def _rgb_to_hex(rgb) -> str:
    return "#{:02X}{:02X}{:02X}".format(*(int(value) & 0xFF for value in rgb))


def _hex_to_rgb(text: str) -> tuple[int, int, int]:
    value = str(text).lstrip("#")
    if len(value) != 6:
        return (0, 0, 0)
    try:
        return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))
    except ValueError:
        return (0, 0, 0)


def _recipe_to_json(recipe: MixRecipe) -> list:
    """一个配方：``[a_id, b_id, percent_a, pair_index]``。

    ``percent_b`` 不存——二色配方必然是 100 减出来的，存两份只会有一天对不上。
    """
    return [str(recipe.a_id), str(recipe.b_id), int(recipe.percent_a), int(recipe.pair_index)]


def _recipe_from_json(
    item, hex_text: str, rgb: tuple[int, int, int], lab, engine: str
) -> MixRecipe:
    a_id, b_id, percent_a, pair_index = item
    return MixRecipe(
        pair_index=int(pair_index),
        a_id=str(a_id),
        b_id=str(b_id),
        percent_a=int(percent_a),
        percent_b=100 - int(percent_a),
        color_hex=hex_text,
        lab=(float(lab[0]), float(lab[1]), float(lab[2])),
        rgb=rgb,
        engine=engine,
    )


def entry_to_json(entry: PaletteEntry) -> dict:
    """A :class:`PaletteEntry` as plain JSON-compatible data."""
    data: dict = {
        "key": entry.key,
        "hex": entry.color_hex,
        "rgb": [int(value) for value in entry.rgb],
        "label": entry.label,
        "kind": entry.kind,
        "ids": [str(value) for value in entry.filament_ids],
        "parts": [[str(parent), int(percent)] for parent, percent in entry.parts],
        "recipes": [_recipe_to_json(recipe) for recipe in entry.recipes],
    }
    if entry.recipe is not None:
        data["recipe"] = _recipe_to_json(entry.recipe)
    return data


def entry_from_json(
    data: dict, library: FilamentLibrary | None = None, engine: str = DEFAULT_ENGINE
) -> PaletteEntry:
    """Rebuild one colour from its JSON form.

    ``lab`` is recomputed from ``rgb`` and ``key`` from kind + hex, so a file
    written by another version cannot smuggle in an inconsistent colour table.
    ``filament`` is filled in only when this machine's library still has the
    spool; the entry stays usable either way.
    """
    rgb = tuple(int(value) for value in (data.get("rgb") or (0, 0, 0)))
    hex_text = str(data.get("hex") or _rgb_to_hex(rgb))
    lab = _color.lab_from_rgb(rgb)
    ids = tuple(str(value) for value in (data.get("ids") or ()))
    kind = str(data.get("kind") or ("filament" if len(ids) == 1 else "mix"))
    recipes = tuple(
        _recipe_from_json(item, hex_text, rgb, lab, engine)
        for item in (data.get("recipes") or ())
    )
    recipe = (
        _recipe_from_json(data["recipe"], hex_text, rgb, lab, engine)
        if data.get("recipe")
        else (recipes[0] if recipes else None)
    )
    if kind == "filament" and ids:
        key = f"filament:{ids[0]}"
    else:
        key = f"colour:{hex_text}"
    filament = None
    if kind == "filament" and ids and library is not None:
        filament = library.get(ids[0])
    return PaletteEntry(
        key=str(data.get("key") or key),
        color_hex=hex_text,
        rgb=rgb,
        lab=lab,
        label=str(data.get("label") or hex_text),
        kind=kind,
        filament_ids=ids,
        recipe=recipe,
        filament=filament,
        recipes=recipes,
        parts=tuple((str(parent), int(percent)) for parent, percent in (data.get("parts") or ())),
    )


# ---------------------------------------------------------------------------
# 预览 / 清单
# ---------------------------------------------------------------------------
def render_preview(result: MatchResult | None) -> Image.Image:
    """The matched picture as an image: each pixel in its own colour, RGBA.

    This is the same picture the preview panel draws, built from the result
    rather than grabbed from a widget, so the file carries what the user sees
    and not whatever the window happened to be sized to.
    """
    if result is None:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    height, width = (int(result.indices.shape[0]), int(result.indices.shape[1]))
    canvas = np.zeros((height, width, 4), dtype=np.uint8)
    for index, entry in enumerate(result.palette):
        mask = result.indices == index
        if mask.any():
            canvas[mask] = (int(entry.rgb[0]), int(entry.rgb[1]), int(entry.rgb[2]), 255)
    return Image.fromarray(canvas, "RGBA")


def colour_list(result: MatchResult | None) -> list[dict]:
    """「这个模型需要的颜色」：每种颜色占的像素比例和它的配料。"""
    if result is None or not result.palette:
        return []
    shares = result.shares()
    rows = []
    for index, entry in enumerate(result.palette):
        rows.append(
            {
                "hex": entry.color_hex,
                "share_of_pixels": round(float(shares[index]), 6),
                "pixels": int(result.counts[index]),
                "label": entry.label,
                "kind": entry.kind,
                "parts": [[str(parent), int(percent)] for parent, percent in entry.parts],
            }
        )
    return rows


def spool_list(
    result: MatchResult | None, library: FilamentLibrary | None
) -> list[dict]:
    """「用到的耗材颜色」：每卷的 id、名称、颜色、占比，缺的一卷也照实写。"""
    rows = []
    for use in spool_usage(result, library):
        filament = use.filament
        rows.append(
            {
                "id": use.filament_id,
                "name": filament.name if filament is not None else "",
                "brand": filament.brand if filament is not None else "",
                "material_type": filament.material_type if filament is not None else "",
                "color_hex": use.color_hex,
                "note": filament.note if filament is not None else "",
                "share": round(float(use.share), 6),
                "in_library": filament is not None,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# 存 / 读
# ---------------------------------------------------------------------------
def _indices_bytes(indices: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(indices, dtype=np.int32), allow_pickle=False)
    return buffer.getvalue()


def _indices_from_bytes(payload: bytes) -> np.ndarray:
    array = np.load(io.BytesIO(payload), allow_pickle=False)
    return np.asarray(array, dtype=np.int32)


def save_project(
    path: str | Path,
    *,
    image: Image.Image,
    image_name: str,
    base: MatchResult,
    live: MatchResult | None = None,
    deleted=(),
    merged=None,
    replaced=None,
    engine: str = DEFAULT_ENGINE,
    plate: PlateSettings | None = None,
    library: FilamentLibrary | None = None,
    version: str = "",
    preview: Image.Image | None = None,
    plan: str = "",
) -> Path:
    """Write one ``.bpproj`` file.  Returns the path actually written."""
    target = Path(path)
    if target.suffix.lower() != PROJECT_SUFFIX:
        target = target.with_suffix(PROJECT_SUFFIX)
    shown = live if live is not None else base
    document = {
        "app": "BambuPalette",
        "format": PROJECT_FORMAT,
        "version": version,
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "engine": engine,
        "plan": str(plan),
        "image": {
            "name": image_name,
            "file": _SOURCE,
            "width": int(image.width),
            "height": int(image.height),
        },
        "preview": {"file": _PREVIEW},
        "settings": {key: getattr(base.settings, key) for key in _SETTING_KEYS},
        "base": {
            "palette": [entry_to_json(entry) for entry in base.palette],
            "means": [
                _rgb_to_hex(means) for means in (base.means if base.means is not None else ())
            ],
            "source": base.source,
        },
        "edits": {
            "deleted": sorted(int(value) for value in deleted),
            "merged": {str(int(key)): int(value) for key, value in dict(merged or {}).items()},
            "replaced": {
                str(int(key)): entry_to_json(entry)
                for key, entry in dict(replaced or {}).items()
            },
        },
        "colours": colour_list(shown),
        "spools": spool_list(shown, library),
    }
    settings = plate or PlateSettings()
    document["plate"] = {key: getattr(settings, key) for key in _PLATE_KEYS}
    document["plate"]["base_hex"] = (
        shown.palette[int(settings.base_index)].color_hex
        if 0 <= int(settings.base_index) < len(shown.palette)
        else ""
    )

    preview_image = preview if preview is not None else render_preview(shown)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(_DOCUMENT, json.dumps(document, ensure_ascii=False, indent=2))
        archive.writestr(_INDICES, _indices_bytes(base.indices))
        source_buffer = io.BytesIO()
        image.convert("RGBA").save(source_buffer, format="PNG")
        archive.writestr(_SOURCE, source_buffer.getvalue())
        preview_buffer = io.BytesIO()
        preview_image.convert("RGBA").save(preview_buffer, format="PNG")
        archive.writestr(_PREVIEW, preview_buffer.getvalue())
    return target


def load_project(
    path: str | Path, *, library: FilamentLibrary | None = None
) -> ProjectState:
    """Read one ``.bpproj`` file back into a :class:`ProjectState`."""
    target = Path(path)
    try:
        with zipfile.ZipFile(target) as archive:
            names = set(archive.namelist())
            missing = [name for name in (_DOCUMENT, _SOURCE, _INDICES) if name not in names]
            if missing:
                raise ProjectError(f"工程文件缺少 {('、'.join(missing))}，可能不是 BambuPalette 的工程。")
            document = json.loads(archive.read(_DOCUMENT).decode("utf-8"))
            # 先认文件、再读图：别人的 zip 第一步就该被说清楚，而不是先报一句
            # 「图片读不出来」——那会让人以为是图坏了，而不是拿错了文件。
            if not isinstance(document, dict) or document.get("app") != "BambuPalette":
                raise ProjectError("这个文件不是 BambuPalette 工程。")
            if int(document.get("format") or 0) != PROJECT_FORMAT:
                raise ProjectError(
                    f"工程格式是 {document.get('format')}，这个版本只认 {PROJECT_FORMAT}。"
                )
            source_bytes = archive.read(_SOURCE)
            indices = _indices_from_bytes(archive.read(_INDICES))
    except zipfile.BadZipFile as error:
        raise ProjectError(f"{target.name} 不是有效的工程文件（{error}）。") from error
    except (OSError, ValueError, KeyError) as error:
        raise ProjectError(f"读取工程失败：{error}") from error

    try:
        image = Image.open(io.BytesIO(source_bytes))
        image.load()
    except (OSError, ValueError) as error:
        raise ProjectError(f"工程里的原图读不出来：{error}") from error

    engine = str(document.get("engine") or DEFAULT_ENGINE)
    base_data = document.get("base") or {}
    palette = [
        entry_from_json(item, library, engine) for item in (base_data.get("palette") or ())
    ]
    if not palette:
        raise ProjectError("工程里没有颜色表，读不出这一版用的是什么颜色。")
    if indices.ndim != 2:
        raise ProjectError("工程里的基线像素表坏了。")
    counts = np.array(
        [int(np.count_nonzero(indices == index)) for index in range(len(palette))],
        dtype=np.int64,
    )
    means = None
    if base_data.get("means"):
        means = np.array(
            [_hex_to_rgb(value) for value in base_data["means"]], dtype=np.uint8
        )
        if len(means) != len(palette):
            means = None
    settings = MatchSettings(
        **{
            key: (document.get("settings") or {}).get(key, getattr(MatchSettings(), key))
            for key in _SETTING_KEYS
        }
    ).clamped()
    base = MatchResult(
        width=int(indices.shape[1]),
        height=int(indices.shape[0]),
        palette=palette,
        indices=indices,
        counts=counts,
        settings=settings,
        source=str(base_data.get("source") or ""),
        means=means,
    )

    edits = document.get("edits") or {}
    replaced = {
        int(key): entry_from_json(value, library, engine)
        for key, value in dict(edits.get("replaced") or {}).items()
    }
    plate_data = document.get("plate") or {}
    plate = PlateSettings(
        **{
            key: plate_data.get(key, getattr(PlateSettings(), key))
            for key in _PLATE_KEYS
        }
    )
    base_hex = str(plate_data.get("base_hex") or "")
    base_index = -1
    live_palette = _live_palette(base, edits, replaced)
    for index, entry in enumerate(live_palette):
        if base_hex and entry.color_hex == base_hex:
            base_index = index
            break
    plate = plate.__class__(**{**plate.__dict__, "base_index": base_index}).clamped()

    return ProjectState(
        image=image,
        image_name=str((document.get("image") or {}).get("name") or "混色底板"),
        base=base,
        deleted={int(value) for value in (edits.get("deleted") or ())},
        merged={int(key): int(value) for key, value in dict(edits.get("merged") or {}).items()},
        replaced=replaced,
        plate=plate,
        engine=engine,
        plan=str(document.get("plan") or ""),
        spools=list(document.get("spools") or []),
        colours=list(document.get("colours") or []),
        saved_at=str(document.get("saved_at") or ""),
        version=str(document.get("version") or ""),
    )


def _live_palette(base: MatchResult, edits: dict, replaced: dict) -> list[PaletteEntry]:
    """The colour list the page will show for this project, in its order.

    Mirrors ``PicturePage._resolve``/``_compose`` exactly — merge chains are
    followed, a colour deleted takes everything merged into it with it, a merge
    *source* stops being a colour of its own, and the survivors keep the
    original palette order.  It lives here (not only in the page) so the file's
    ``base_hex`` can be turned back into a palette index while loading.
    """
    merged = {int(key): int(value) for key, value in dict(edits.get("merged") or {}).items()}
    for source in list(merged):
        target = merged[source]
        seen = {source}
        while target in merged and target not in seen:
            seen.add(target)
            target = merged[target]
        merged[source] = target
    gone = {int(value) for value in (edits.get("deleted") or ())}
    for source, target in merged.items():
        if target in gone:
            gone.add(source)
    alive = [
        index
        for index in range(len(base.palette))
        if index not in gone and index not in merged
    ]
    return [replaced.get(index, base.palette[index]) for index in alive]

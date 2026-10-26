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
import time
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
    # 1.6.53（m21616 第 8 条）：「设置」里能改的默认参数与快捷键。把它们放进这份
    # 偏好文件，是因为它们的意思和 engine / pick_mode 一样 —— 「下次打开还是这样」。
    "picture_colours": 32,
    "picture_spools": 4,
    "picture_dither": False,
    "picture_only_spools": False,
    "picture_base_mm": 1.4,
    "model_colours": 32,
    "model_spools": 4,
    "model_only_spools": False,
    "model_flip_rotate": True,
    #: 三维视图底色（1.6.54，m22211 第 6 条）：默认是 SolidWorks 视口那种浅蓝灰。
    "model_background": "#B4C6D2",
    # 1.6.57（m26604 第 9 条）：导出 .blend 要让本机的 Blender 动手，路径可以手填。
    "model_blender_path": "",
    # 1.6.60（m27359 第 3 条）：「实验性：没有颜色上限」。勾上以后两页匹配都不再按
    # 「颜色数量」停手（默认 32），改用节目里能容纳的最大值 —— 板子颜色太多、
    # 32 种明显不够用时才值得开，颜色一多，每个色块对应的像素就少。
    # 1.6.60（m27359 第 3 条）/ 1.6.61（m27752 第 1、2、6 条）：实验性开关按页分开记 ——
    # 「图像转换」与「模型配色」各自决定要不要设限，也各自分颜色上限与耗材上限。
    # ``experimental_no_colour_limit`` 是旧版的单个开关，只留作迁移来源。
    "experimental_no_colour_limit": False,
    "picture_no_colour_limit": False,
    "picture_no_spool_limit": False,
    "model_no_colour_limit": False,
    "model_no_spool_limit": False,
    "shortcut_export_glb": "Ctrl+S",
    # 1.6.66（m29882 第 4 条）：每种导出的实测用时（秒），进度框的倒计时拿它当起点。
    # 键是 "3mf" / "obj" / "glb" / "blend" / "svg"，值是最近几次的指数滑动平均。
    "export_seconds": {},
    # 1.6.54（m22211 第 7 条）：每个页自己的快捷键。允许跨页重复（同一时刻只有一页
    # 在屏幕上），但同一个页里不许撞车 —— 撞了 settings_dialog 会拒绝保存。
    "shortcut_picture_import": "Ctrl+O",
    "shortcut_picture_rematch": "Ctrl+R",
    "shortcut_picture_save_project": "Ctrl+S",
    "shortcut_picture_export_png": "Ctrl+E",
    "shortcut_model_open": "Ctrl+O",
    "shortcut_model_rematch": "Ctrl+R",
    "shortcut_model_export_obj": "Ctrl+Shift+E",
    "shortcut_model_export_3mf": "Ctrl+Shift+M",
    "shortcut_recipe_find": "Ctrl+F",
    "shortcut_recipe_add": "Ctrl+N",
    "sort": "rgb",
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
    # 1.6.53：「设置」里那些默认值。只认 DEFAULT_SETTINGS 里有的键，值类型跟着默认值走，
    # 于是手改坏一个数字（写成字符串、负数、null）不会把页面初始化弄炸。
    for key, fallback in DEFAULT_SETTINGS.items():
        if key in ("engine", "pick_mode", "material", "version"):
            continue
        if key not in raw:
            continue
        stored = raw[key]
        if isinstance(fallback, bool):
            values[key] = bool(stored)
        elif isinstance(fallback, int):
            try:
                values[key] = int(stored)
            except (TypeError, ValueError):
                continue
        elif isinstance(fallback, float):
            try:
                values[key] = float(stored)
            except (TypeError, ValueError):
                continue
        elif isinstance(fallback, dict):
            # 1.6.66（m29882 第 4 条）：`export_seconds` 是一张「导出种类 → 秒」的表，
            # 给进度框的倒计时当起点。只收数字，别的当没写过。
            if isinstance(stored, dict):
                values[key] = {
                    str(name): float(seconds)
                    for name, seconds in stored.items()
                    if isinstance(seconds, (int, float)) and float(seconds) > 0
                }
        else:
            values[key] = str(stored)
    # 1.6.61（m27752 第 1、2 条）：旧版只有一个「实验性：没有颜色上限」的开关，它对两页
    # 一起生效。旧文件里勾过的人，两页都算勾过；新键一旦写过就以新键为准。
    legacy = bool(raw.get("legacy_no_colour_limit", raw.get("experimental_no_colour_limit", False)))
    for key in ("picture_no_colour_limit", "model_no_colour_limit"):
        if key not in raw and legacy:
            values[key] = True
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
    # 1.6.62（m28297 第 2 条）：Windows 上杀软/索引器会短时间攥住刚写出来的文件，
    # 一次失败不代表存不下。重试三次、每次隔 120 ms，把这种抖动挡掉；三次都败才
    # 往上报，并在消息里带上路径 —— 以前的报错只有一句「写不进偏好文件」，用户
    # 完全不知道是哪个盘、哪个文件夹。
    payload = json.dumps(merged, ensure_ascii=False, indent=2) + "\n"
    last: OSError | None = None
    for attempt in range(3):
        try:
            atomic_write_text(target, payload)
            return target
        except OSError as error:  # pragma: no cover - 需要真的锁住文件才走得到
            last = error
            time.sleep(0.12 * (attempt + 1))
    raise OSError(f"写不进 {target}（{last}）") from last

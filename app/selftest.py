"""Self-check: prove the whole pipeline works in this build.

Run from the packaged application with ``--selftest``.  It is deliberately
independent of the test suite so a user (or support) can confirm that numpy,
Pillow, Qt and the mesh writers all function in the frozen bundle:

* build a five-spool library and the full mix catalogue,
* render a small picture and match it against that library,
* build the flat multi-colour plate,
* write both a 3MF and an OBJ + MTL,
* re-open the 3MF and check its structure.

The report is written to ``selftest-report.txt`` and, when a console exists,
also printed.  Exit code 0 means everything passed.
"""

from __future__ import annotations

import json
import os
import sys
import zipfile
from pathlib import Path

SPOOLS = (
    ("大简 PETG HF 白", "#F2F0EB"),
    ("大简 PETG HF 黑", "#17181C"),
    ("大简 PETG HF 金", "#D9A441"),
    ("大简 PETG HF 青", "#2FA8B8"),
    ("大简 PETG HF 红", "#C8342E"),
)


def run_selftest(*, report_path: Path | None = None) -> int:
    lines: list[str] = []
    failures: list[str] = []

    def check(condition: bool, label: str) -> None:
        lines.append(f"[{'ok' if condition else 'FAIL'}] {label}")
        if not condition:
            failures.append(label)

    from .core import paths

    try:
        # Bounded, unlike tempfile.mkdtemp: a refused temp directory must not
        # turn the self-check into an unkillable retry loop.
        scratch = paths.scratch_dir("fcs-selftest-")
    except OSError:
        scratch = paths.data_dir() / "selftest"
        scratch.mkdir(parents=True, exist_ok=True)
    paths.library_path = lambda: scratch / "filament-library.json"  # type: ignore[assignment]
    paths.data_dir = lambda: scratch  # type: ignore[assignment]

    from . import APP_NAME, __version__

    lines.append(f"app         {APP_NAME} {__version__}")
    lines.append(f"python      {sys.version.split()[0]}")
    lines.append(f"frozen      {bool(getattr(sys, 'frozen', False))}")
    lines.append(f"scratch     {scratch}")
    lines.append("")

    try:
        import numpy

        lines.append(f"numpy       {numpy.__version__}")
        from PIL import __version__ as pillow_version

        lines.append(f"Pillow      {pillow_version}")
        from PySide6 import __version__ as qt_version

        lines.append(f"PySide6     {qt_version}")
    except Exception as error:  # pragma: no cover - only on a broken bundle
        check(False, f"importing the dependencies failed: {error}")

    from .core.library import Filament, FilamentLibrary
    from .main import build_application

    app = build_application([])

    from PySide6.QtCore import QSize as _QSize
    from PySide6.QtGui import QIcon as _QIcon

    from .core.paths import resource_path as _resource_path

    _logo = _resource_path("logo.ico")
    check(_logo.is_file(), f"the logo ships with the program ({_logo})")
    _icon = app.windowIcon()
    check(
        not _icon.isNull() and not _icon.pixmap(_QSize(32, 32)).isNull(),
        "the application has a window icon",
    )
    del _QIcon, _QSize

    # Objective "默认保存地址为下载": the frozen build must offer the user's own
    # downloads folder, not the hidden private one, even though a write probe
    # inside the bundle used to answer "no" and send every export to <app>\data.
    _exports = paths.exports_dir()
    lines.append(f"exports     {_exports}")
    check(_exports.is_dir(), f"the export dialog opens in an existing folder ({_exports})")
    _downloads = [candidate for candidate in paths._downloads_candidates() if candidate.is_dir()]
    if _downloads:
        check(
            _exports == _downloads[0],
            "the export dialog opens in the downloads folder, not a hidden data folder",
        )

    # Objective "Bambu 报错": the exported 3MF carries a complete project config,
    # and the template it is built from has to be found from inside the bundle —
    # a data file read through ``__file__`` is exactly the thing a frozen build
    # gets wrong.
    from .mesh.threemf import (
        THREEMF_APPLICATION,
        _PROJECT_TEMPLATE_PATH,
        _project_settings,
    )

    lines.append(f"template    {_PROJECT_TEMPLATE_PATH}")
    check(_PROJECT_TEMPLATE_PATH.is_file(), "the project-config template ships with the app")
    check(
        THREEMF_APPLICATION.startswith("BambuStudio-")
        and THREEMF_APPLICATION.split("-", 1)[1][:1].isdigit(),
        f"Application names a client Bambu Studio reads the config for ({THREEMF_APPLICATION})",
    )
    _config = json.loads(_project_settings(["#112233", "#445566"]))
    check(
        _config.get("filament_colour") == ["#112233", "#445566"],
        "the project config takes the palette it is given",
    )
    _sized = {
        key: len(value)
        for key, value in _config.items()
        if isinstance(value, list) and value and key.startswith("filament_")
    }
    check(
        bool(_sized) and set(_sized.values()) == {2},
        "every per-filament array is resized to the palette length",
    )
    check(
        _config.get("ensure_vertical_shell_thickness") == "enabled",
        "the one value Bambu Studio spells differently is written its way",
    )

    from .core.engines import ENGINES, ENGINE_BAMBU, ENGINE_MIXER
    from .core.image_matching import MatchSettings, build_palette, match_image
    from .core.mixes import expected_recipe_count
    from .mesh.objfile import write_obj
    from .mesh.plate import PlateSettings, build_plate
    from .mesh.threemf import slots_from_palette, write_3mf
    from .ui.main_window import MainWindow

    lines.append(f"engines     {', '.join(sorted(ENGINES))}")
    lines.append("")

    library = FilamentLibrary([
        Filament(name=name, brand="大简", material_type="PETG HF", color_hex=value)
        for name, value in SPOOLS
    ])
    # auto_triples=False keeps this window deterministic: the three-filament
    # table is built on its own timer and would change every count below.
    window = MainWindow(library=library, auto_triples=False)
    window.resize(1200, 760)
    window.show()
    app.processEvents()

    # The app must open maximised: a windowed default made people maximise by
    # hand on every start.  Exercised through the same helper main() calls.
    from .main import show_main_window as _show_main_window

    starter = MainWindow(library=FilamentLibrary(list(library.filaments)), auto_triples=False)
    _show_main_window(starter)
    app.processEvents()
    check(starter.isMaximized(), "the main window opens maximised")
    starter.close()

    check(expected_recipe_count(5) == 810, "5 spools produce 810 mixes")
    merged = window._catalog.colour_count
    everything = window._combined.colour_count
    check(merged <= 810, f"merging identical colours never adds any ({merged} <= 810)")
    check(
        len(window._recipes) == everything,
        f"the grid holds one card per distinct colour, got {len(window._recipes)} for {everything}",
    )
    check(window._catalog.pair_count == 10, "5 spools produce 10 pairs")

    # 「颜色详情里有合成这个颜色的所有配方」: a colour reached by two recipes must
    # list both, not just the first.
    shared = [colour for colour in window._catalog.colours if colour.recipe_count > 1]
    check(
        len(shared) == 810 - merged,
        f"{810 - merged} colours are reachable by more than one recipe, got {len(shared)}",
    )
    if shared:
        window._grid.selectRecipe(shared[0])
        window._on_grid_selected(shared[0])
        app.processEvents()
        check(window._detail.colour() is shared[0], "the detail panel keeps the merged colour")
        listed = [row for row in window._detail._formula_rows if not row["frame"].isHidden()]
        check(
            len(listed) == shared[0].recipe_count,
            f"the panel lists all {shared[0].recipe_count} recipes of that colour, got {len(listed)}",
        )
        window._detail.clear()
    check(window._catalog.engine.id == ENGINE_MIXER, "the Bambu 2.8 mixer engine is the default")
    check(
        ENGINES[ENGINE_MIXER].mix_rgb_pair((0, 33, 133), (252, 211, 0), 50) == (47, 141, 56),
        "the bundled pigment polynomial reproduces its reference sample",
    )
    check(
        ENGINES[ENGINE_BAMBU].mix_hex("#FFFFFF", "#000000", 90) == "#E5E5E5",
        "the legacy sRGB engine is still selectable and exact",
    )
    check(
        all(
            len({*engine.mix_rgb_pair((255, 255, 255), (0, 0, 0), percent)}) == 1
            for engine in ENGINES.values()
            for percent in (10, 50, 90)
        ),
        "every engine keeps a white + black mix neutral (no blue cast)",
    )
    check(
        window._target_color.isSet()
        and window._target_color.hex() == "#000000"
        and window._target_color.rgb() == (0, 0, 0),
        "目标颜色 starts pre-set to 000 (m02366 item 1)",
    )
    # Clicking a mix marks it with a ring in its OWN colour, and blank space clears
    # the selection again — the old build dimmed every other mix and boxed the
    # selection in blue.
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    from .ui import mix_grid

    single = window._recipes[0]
    window._grid.setRecipes([single])
    app.processEvents()
    window._grid.mousePressEvent(
        QMouseEvent(
            QEvent.Type.MouseButtonPress,
            QPointF(mix_grid.CELL / 2, mix_grid.CELL / 2),
            QPointF(mix_grid.CELL / 2, mix_grid.CELL / 2),
            Qt.MouseButton.LeftButton,
            Qt.MouseButtons.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
    )
    app.processEvents()
    check(window._grid.selectedRecipe() is not None, "clicking a mix swatch selects it")
    window._grid.mousePressEvent(
        QMouseEvent(
            QEvent.Type.MouseButtonPress,
            QPointF(mix_grid.CELL + 6, mix_grid.CELL / 2),
            QPointF(mix_grid.CELL + 6, mix_grid.CELL / 2),
            Qt.MouseButton.LeftButton,
            Qt.MouseButtons.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
    )
    app.processEvents()
    check(window._grid.selectedRecipe() is None, "clicking blank space clears the selection")
    check(window._detail.recipe() is None, "clearing the selection empties the recipe panel")
    check(
        not hasattr(window._grid, "setFocusPair"),
        "the grid no longer dims the mixes that were not selected",
    )
    window._refresh_grid()
    app.processEvents()
    check(len(window._recipes) == everything, "the full catalogue comes back after clearing")

    # 「全部颜色」 — 两色混色、三色混色和耗材本色合成一张表，没有勾选框。
    from .core.mixes import expected_recipe_count

    check(expected_recipe_count(2) + 2 == 83, "81 mixes + 2 raw spools = 83 colours")
    check(
        not hasattr(window, "_all_colours") and not hasattr(window, "_triples_box"),
        "the 全部颜色 / 三色混色 checkboxes are gone",
    )
    spool_count = len(window.library.filaments)
    check(
        len(window._recipes) >= merged,
        f"the grid carries at least the merged mixes, got {len(window._recipes)}",
    )
    check(
        window._combined is not None and len(window._combined) >= merged,
        "「全部颜色」 holds the mixes",
    )
    spool_rows = [
        window._combined[i]
        for i in range(len(window._combined))
        if window._combined[i].filaments
    ]
    check(len(spool_rows) >= 1, "a spool's own colour is in the same list")
    check(
        all(row.color_hex == row.filaments[0].color_hex for row in spool_rows),
        "a spool colour keeps the spool's own hex",
    )
    check(
        [cell.rgb for cell in window._recipes]
        == sorted(cell.rgb for cell in window._recipes),
        "the list follows the 排序 control",
    )
    first_spool = spool_rows[0]
    window._on_grid_selected(first_spool)
    app.processEvents()
    check(window._detail.colour() is first_spool, "the detail panel shows that colour")
    window._detail.clear()
    app.processEvents()
    check(len(window._recipes) >= merged, "the full list is still there")

    # A new spool has no colour until the user picks one, and the dialog refuses
    # to be confirmed until then — a made-up white default is a lie the user only
    # discovers in Bambu Studio.
    from PySide6.QtGui import QKeySequence

    from .ui.filament_dialog import FilamentDialog

    add_dialog = FilamentDialog(
        None, [f.brand for f in window.library.filaments], [], window
    )
    check(not add_dialog._color.isSet(), "添加耗材 starts with NO preset colour")
    # m01647 item 1 (「不要预设颜色」): nothing may stand in for the colour the user
    # has not chosen.  The three spins used to read 0,0,0 and 确认 accepted that as
    # black, so a spool could be created in a colour nobody picked.
    check(
        add_dialog.values()["color_hex"] == "",
        f"an untouched dialog carries no colour, got {add_dialog.values()['color_hex']!r}",
    )
    check(
        not add_dialog._ok_button.isEnabled(),
        "确认 is grey until a colour is given",
    )
    check(
        not add_dialog._next_button.isEnabled(),
        "下一个 is grey until a colour is given",
    )
    check(
        add_dialog._name.placeholderText() == "",
        "名称 carries no example placeholder (m01647 item 7)",
    )
    check(
        add_dialog._next_button.toolTip() == "",
        "the dialog has no hover pop-up on 下一个 (m01647 item 4)",
    )
    add_dialog._color.setValue("#123456")
    check(add_dialog._ok_button.isEnabled(), "确认 comes alive once a colour is entered")
    check(add_dialog._next_button.isEnabled(), "下一个 comes alive with it")
    check(add_dialog.values()["color_hex"] == "#123456", "the entered colour is reported")
    # m10238 item 3: 下一个 sits immediately left of 确认 and means "save this one,
    # then open a fresh empty form".
    check(
        add_dialog._next_button.text() == "下一个",
        f"the add dialog offers 下一个, got {add_dialog._next_button.text()!r}",
    )
    check(not add_dialog.wants_another(), "a plain 确认 does not ask for another spool")
    # 下一个 must sit immediately left of 确认: the row is read off the dialog, not
    # assumed, because a QDialogButtonBox would park it at the far left edge.
    from PySide6.QtWidgets import QHBoxLayout as _QHBoxLayout

    button_row = next(
        layout
        for layout in add_dialog.findChildren(_QHBoxLayout)
        if layout.indexOf(add_dialog._ok_button) >= 0
    )
    left, right = button_row.indexOf(add_dialog._next_button), button_row.indexOf(
        add_dialog._ok_button
    )
    check(
        0 <= left == right - 1,
        f"下一个 sits immediately before 确认, got next={left} ok={right}",
    )
    add_dialog._next_button.click()
    check(add_dialog.wants_another(), "下一个 asks for another spool")
    check(
        add_dialog.result() == FilamentDialog.DialogCode.Accepted,
        "下一个 accepts the dialog so the spool is kept",
    )
    check(
        add_dialog.build_filament().color_hex == "#123456",
        "a spool built from the dialog carries the entered colour",
    )
    add_dialog.close()

    # The colour is typed on the panel itself; nothing opens a colour dialog.
    from .ui import swatch as swatch_module

    check(not hasattr(swatch_module, "ColorButton"), "the pop-up colour picker is gone")
    field = swatch_module.ColorField(None)
    check(not field.isSet() and field.hex() == "", "a colour field starts empty")
    field._spins[0].setValue(200)
    field._on_spin()
    check(field.hex() == "#C80000", f"the R spin box drives the colour, got {field.hex()!r}")
    field._hex.setText("nonsense")
    field._on_hex_edited()
    check(field.hex() == "#C80000", "gibberish leaves the last good colour alone")

    # Deleting a spool asks nothing: it just goes.
    check(
        window._delete_shortcut.key() == QKeySequence(QKeySequence.StandardKey.Delete),
        "Del is bound to the spool list",
    )
    doomed = window.library.filaments[-1]
    window._select_filament(doomed.id)
    before = len(window.library.filaments)
    snapshot = list(window.library.filaments)
    window._delete_shortcut.activated.emit()
    app.processEvents()
    check(
        len(window.library.filaments) == before - 1,
        "pressing Del deletes the selected spool with no confirmation",
    )
    check(
        window.library.get(doomed.id) is None,
        "the spool Del removed is really gone",
    )
    # Put the library back: the rest of this self-test counts the 810 mixes of a
    # five-spool library, and the deletion above was only here to prove the key
    # is wired up.
    window.library.replace_all(snapshot)
    window._reload_library()
    app.processEvents()
    check(len(window.library.filaments) == before, "the library can be restored after the Del test")

    # One spool makes no mixes, but its own colour still has to reach the grid.
    from .core.library import Filament as _SelftestFilament

    lone_window = MainWindow(
        library=FilamentLibrary([_SelftestFilament(material_type="PLA", color_hex="#0000FF")])
    )
    lone_window.resize(1100, 760)
    lone_window.show()
    app.processEvents()
    check(lone_window._catalog.recipe_count == 0, "one spool makes zero mixes")
    app.processEvents()
    check(
        len(lone_window._recipes) == 1,
        f"a lone spool's colour still reaches the grid, got {len(lone_window._recipes)}",
    )
    if lone_window._recipes:
        check(lone_window._recipes[0].color_hex == "#0000FF", "the lone cell is the spool's own colour")
    lone_window.close()

    # An unnamed spool's display name IS its hex; repeating it reads like a bug.
    unnamed_window = MainWindow(library=FilamentLibrary([_SelftestFilament(color_hex="#000000")]))
    app.processEvents()
    row = unnamed_window._list.item(0).text()
    check("#000000   #000000" not in row, f"an unnamed spool does not print its hex twice ({row})")
    check("RGB 0, 0, 0" in row, "the duplicate hex is replaced by the RGB numbers")
    unnamed_window.close()

    check(
        window._picture_page is not None and window._tabs.count() == 2,
        "both pages are present",
    )

    # m00940 #1: 刚打开软件时，耗材方案面板里那两句解释文本要同时出现、而且是面板开头
    # 挨着的两行。1.6.17 里第二句要等第一次匹配/编辑才写上去，两个 QLabel 又各自被面板
    # 的多余高度摊到 116 px 高的槽里（文字在自己槽里垂直居中），看上去像分散在不同盒子
    # 里、只出现了一句。
    from .ui.picture_page import PLAN_HINT_TEXT as _PLAN_HINT_TEXT  # noqa: PLC0415
    from .ui.picture_page import USAGE_HINT_TEXT as _USAGE_HINT_TEXT  # noqa: PLC0415

    window._tabs.setCurrentIndex(1)
    app.processEvents()
    panel = window._picture_page._plan_box
    fresh = window._picture_page
    check(
        fresh._plan_hint.text() == _PLAN_HINT_TEXT and fresh._plan_hint.isVisibleTo(panel),
        f"刚打开时面板就写着「匹配一张图片后…」，got {fresh._plan_hint.text()!r}",
    )
    check(
        fresh._usage.text() == _USAGE_HINT_TEXT and fresh._usage.isVisibleTo(panel),
        f"… 而「匹配完成后…」同时就在旁边，got {fresh._usage.text()!r}",
    )
    check(
        fresh._plan_hint.height() <= 40 and fresh._usage.height() <= 40,
        f"两句各占自己那一行，没有被面板的多余高度摊开"
        f"（{fresh._plan_hint.height()} px / {fresh._usage.height()} px）",
    )
    check(
        0 <= fresh._usage.geometry().top() - fresh._plan_hint.geometry().bottom() <= 24,
        f"两句是挨着的上下两行"
        f"（{fresh._plan_hint.geometry().bottom()} → {fresh._usage.geometry().top()}）",
    )
    window._tabs.setCurrentIndex(0)
    app.processEvents()

    # A picture with four flat fields and a gradient.
    from PIL import Image, ImageDraw

    picture = scratch / "selftest.png"
    image = Image.new("RGB", (160, 120), "#F2F0EB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 75, 65), fill="#C8342E")
    draw.ellipse((85, 10, 150, 60), fill="#D9A441")
    draw.polygon([(20, 110), (80, 75), (140, 110)], fill="#2FA8B8")
    for x in range(8, 150):
        shade = int(32 + (x - 8) / 142 * 200)
        draw.line([(x, 66), (x, 72)], fill=(shade, shade, shade))
    image.save(picture)

    palette = build_palette(library, window._catalog)
    check(
        len(palette) == 5 + merged,
        f"the candidate palette has every spool and every distinct mix ({len(palette)})",
    )

    result = match_image(picture, palette, MatchSettings(max_colours=12))
    check(result.printed_pixels > 0, f"the picture matched ({result.printed_pixels} solid pixels)")
    check(len(result.palette) >= 3, f"{len(result.palette)} colours were needed")

    page = window._picture_page
    page.loadImage(picture)
    app.processEvents()
    check(page.result is not None, "the picture page matched the picture")
    # m01885 #7: 刚匹配完不选中任何颜色。以前这里会退回第一行，于是高亮落在用户没点过
    # 的颜色上，「取消选中」也就到不了。
    check(
        page._list.current() == -1 and not page._list.selected_indices(),
        f"匹配完不默认选中第一个颜色，got 第 {page._list.current()} 行",
    )
    check(
        not page._delete_button.isEnabled() and not page._replace_button.isEnabled(),
        "没有选中颜色时 删除 / 更换颜色 是灰的",
    )

    # m00519 #2: 「原图 / 预览」用滚轮缩放、按住左键拖动平移、双击还原成整张。
    from PySide6.QtCore import QEvent as _QEvent
    from PySide6.QtCore import QPointF as _QPointF
    from PySide6.QtCore import Qt as _Qt
    from PySide6.QtGui import QMouseEvent as _QMouse

    from .core.image_matching import plan_help as _plan_help
    from .core.settings import load_settings as _load_settings
    from .ui import picture_page as _pp

    def _mouse(kind, pos, *, buttons=_Qt.MouseButtons.NoButton):
        return _QMouse(
            kind, pos, pos, _Qt.MouseButton.LeftButton, buttons, _Qt.KeyboardModifier.NoModifier
        )

    probe = _pp.PictureView()
    probe.resize(400, 300)
    probe.setResult(page.result, page._view._source)
    app.processEvents()
    check(probe.zoom() == _pp.ZOOM_MIN, "预览 starts at 「整张铺满面板」")
    fit = probe._target_rect()
    fit_scale = probe._scale
    focus = _QPointF(probe.width() / 2.0, probe.height() / 2.0)
    probe.zoom_by(2.0, focus)
    app.processEvents()
    zoomed = probe._target_rect()
    check(
        abs(probe.zoom() - _pp.ZOOM_STEP**2) < 1e-9,
        f"two wheel notches zoom to {probe.zoom():.3f}×",
    )
    check(
        zoomed.width() > fit.width() and probe._scale > fit_scale,
        "zooming in draws the picture bigger than the panel",
    )
    check(
        abs((zoomed.x() + (focus.x() - fit.x()) / fit_scale * probe._scale) - focus.x()) < 0.5,
        "the pixel under the cursor stays under the cursor while zooming",
    )
    probe.zoom_by(50.0, focus)
    check(abs(probe.zoom() - _pp.ZOOM_MAX) < 1e-9, f"zoom stops at {_pp.ZOOM_MAX:g}×")

    # 拖动平移：按下、移动超过阈值、松手 —— 只平移，不挑颜色。
    picked: list[int] = []
    probe.regionClicked.connect(picked.append)
    start = _QPointF(probe.width() / 2.0, probe.height() / 2.0)
    dragged_to = _QPointF(start.x() + 60.0, start.y() + 45.0)
    probe.mousePressEvent(
        _mouse(_QEvent.Type.MouseButtonPress, start, buttons=_Qt.MouseButtons.LeftButton)
    )
    probe.mouseMoveEvent(
        _mouse(_QEvent.Type.MouseMove, dragged_to, buttons=_Qt.MouseButtons.LeftButton)
    )
    probe.mouseReleaseEvent(_mouse(_QEvent.Type.MouseButtonRelease, dragged_to))
    app.processEvents()
    check(not picked, "dragging to pan never picks a colour on the way")
    check(probe._pan != _QPointF(0.0, 0.0), "dragging moves the picture inside the panel")
    limit_x = max(0.0, (probe._draw_size()[0] - probe.width()) / 2.0)
    check(
        abs(probe._pan.x()) <= limit_x + 0.5,
        "panning is clamped so the picture never leaves the panel",
    )
    # The 滚轮缩放 / 拖动平移 instructions used to be a tooltip on both panels, then
    # an always-visible line at the top of the page.  m01647 item 4 deleted the hover
    # pop-ups; 1.6.14 #6 moved the whole paragraph into the 帮助 window, so the page
    # now carries no explanation text at all.
    check(
        _pp.ZOOM_HINT in page.help_text(),
        f"the help explains 滚轮缩放 / 拖动平移, got {page.help_text()!r}",
    )
    check(
        not hasattr(page, "_page_hint"),
        "图像转换页面上不再挂那段常驻说明（1.6.14 #6）",
    )
    check(
        page._original_view.toolTip() == "" and probe.toolTip() == "",
        "neither panel carries a hover pop-up any more",
    )

    probe.reset_zoom()
    app.processEvents()
    check(
        probe.zoom() == _pp.ZOOM_MIN and probe._pan == _QPointF(0.0, 0.0),
        "double-clicking (reset) goes back to the whole picture",
    )
    # A plain click must still pick the colour under the cursor.
    import numpy as np  # noqa: PLC0415

    rect = probe._target_rect()
    printed = np.argwhere(page.result.indices >= 0)
    row, column = (int(value) for value in printed[len(printed) // 2])
    spot = _QPointF(rect.x() + (column + 0.5) * probe._scale, rect.y() + (row + 0.5) * probe._scale)
    probe.mousePressEvent(
        _mouse(_QEvent.Type.MouseButtonPress, spot, buttons=_Qt.MouseButtons.LeftButton)
    )
    probe.mouseReleaseEvent(_mouse(_QEvent.Type.MouseButtonRelease, spot))
    app.processEvents()
    check(
        picked == [int(page.result.indices[row, column])],
        f"a plain click still picks the colour under the cursor ({picked})",
    )
    probe.deleteLater()

    # 抖动以前在带透明区域的图上被灰掉：有颜色的像素会被挤成一行再量化，那一行上没有
    # 邻居关系。1.6.12 改成「先按老办法定出调色板，再用这块调色板在二维原图上做误差
    # 扩散」（见 image_matching._dither_masked），所以那种图上它也该能勾、勾了要有区别。
    check(page._dither.isEnabled(), "an opaque picture keeps 抖动 selectable")
    transparent = scratch / "selftest-alpha.png"
    alpha_picture = Image.new("RGBA", (48, 32))
    for x in range(48):
        for y in range(32):
            alpha_picture.putpixel(
                (x, y), (180 + x * 3, 90 + y * 4, 200 - x * 2, 0 if x < 8 else 255)
            )
    alpha_picture.save(transparent)
    page._dither.setChecked(False)
    page.loadImage(transparent)
    app.processEvents()
    check(page._dither.isEnabled(), "a transparent picture keeps 抖动 selectable too")
    check(
        "不能用抖动" not in page.help_text(),
        "and the help no longer claims it cannot",
    )
    plain_indices = np.array(page.result.indices, dtype=np.int64, copy=True)
    page._dither.setChecked(True)
    page._on_rematch()
    app.processEvents()
    check(page.result is not None, "a transparent picture still matches with 抖动 on")
    indices = page.result.indices
    check(bool((indices[:, :8] < 0).all()), "the transparent strip stays unprinted with 抖动 on")
    check(bool((indices[:, 8:] >= 0).all()), "and the artwork is printed in full")
    changed = int((indices != plain_indices).sum())
    check(changed > 0, f"turning 抖动 on really changes the match ({changed} 像素)")
    page._dither.setChecked(False)
    page.loadImage(picture)
    app.processEvents()
    check(page._dither.isEnabled(), "抖动 stays selectable on an opaque picture")

    # 1.6.12 的 3/4/5：点预览里的色块要像点列表一样选中；已经显示色块时，重绘、放大、
    # 切到别的窗口再回来都不能把它弄丢（老代码把色块缓存在 _dim_pixmap 里只交出一次，
    # 第二次重绘就变成 None）；再点同一行 = 取消选中，并退回上一个预览。
    from PySide6.QtCore import QPoint as _QPoint  # noqa: PLC0415
    from PySide6.QtTest import QTest as _QTest  # noqa: PLC0415

    page.resize(1200, 800)
    page.show()
    app.processEvents()

    block = int(page.result.indices[page.result.indices >= 0][0])
    page._view.regionClicked.emit(block)
    app.processEvents()
    check(page._view._selected == block, "clicking a colour block in 预览 selects that colour")
    check(page._list.current() == block, "and the colour list highlights the same row")

    first_shot = page._view.grab().toImage()
    app.processEvents()
    check(
        page._view.grab().toImage() == first_shot,
        "the block survives a second repaint (it used to be handed out only once)",
    )
    page._view.zoom_by(1.25)
    app.processEvents()
    check(page._view._dim_pixmap() is not None, "and it is still painted after zooming in")
    page._view.reset_zoom()
    app.processEvents()

    rows = page._list._list

    def _click(index: int) -> None:
        row = page._list._row_of(index)
        item = rows.item(row)
        rect = rows.visualItemRect(item)
        check(not rect.isEmpty(), f"the row for colour {index} has a clickable rect")
        _QTest.mouseClick(rows.viewport(), _Qt.MouseButton.LeftButton, pos=rect.center())
        app.processEvents()

    other = 1 if block == 0 else 0
    page._select(block)
    app.processEvents()
    _click(other)
    check(page._view._selected == other, "clicking another row moves the block to it")
    check(page._list.current() == other, "and the list follows the click")
    _click(other)
    check(
        page._view._selected == block,
        "clicking the same colour again steps back to the preview before it",
    )
    _click(block)
    check(
        page._view._selected == -1 and page._list.current() == -1,
        "with nothing to step back to it simply puts the colour away",
    )
    page._select(block)
    app.processEvents()

    # 颜色数量: the 32 ceiling belongs to Bambu Studio, not to us, so the page has
    # to say so — the user asked for a bigger number and got "拓竹还没放开".
    from .mesh.threemf import MAX_PAINTED_FILAMENTS as _MAX_PAINTED
    from .ui.picture_page import DEFAULT_SPOOL_CAP as _DEFAULT_SPOOL_CAP

    check(
        page._colour_count.maximum() == _MAX_PAINTED,
        f"颜色数量 stops at Bambu Studio's own ceiling ({_MAX_PAINTED})",
    )
    check(
        page._colour_count.value() == _MAX_PAINTED,
        f"颜色数量 starts at the ceiling ({_MAX_PAINTED})",
    )
    page_help = page.help_text()
    check(
        "Bambu Studio" in page_help and str(_MAX_PAINTED) in page_help,
        "「帮助」里写明 32 这个上限是 Bambu Studio 自己的",
    )
    check("工单" in page_help, "and says the ceiling is reported to Bambu, not invented by us")
    check(
        page_help in window._help_text(),
        "那段说明整段搬进了「帮助」窗口，页面上不再单独显示（1.6.14 #6）",
    )

    # 耗材上限 (m00780/m00840 #4): 0 = 不限制, otherwise the plan may spend only
    # that many spools, and colours that do not fit are replaced.
    check(
        page._spool_cap.value() == _DEFAULT_SPOOL_CAP,
        f"耗材上限 starts at {_DEFAULT_SPOOL_CAP}",
    )
    check(
        page._spool_cap.maximum() == _MAX_PAINTED,
        "耗材上限 cannot promise more spools than a project can hold",
    )
    check(
        page._settings().max_spools == _DEFAULT_SPOOL_CAP,
        f"{_DEFAULT_SPOOL_CAP} carries through to the matcher",
    )
    page._spool_cap.setValue(0)
    check(page._settings().max_spools == 0, "0 still means 不限制")
    page._spool_cap.setValue(2)
    check(page._settings().max_spools == 2, "the spin box reaches the matcher")
    page._spool_cap.setValue(_DEFAULT_SPOOL_CAP)

    capped = match_image(picture, palette, MatchSettings(max_colours=12, max_spools=1))
    spent = sorted({parent for entry in capped.palette for parent in entry.filament_ids})
    check(len(spent) <= 1, f"耗材上限 1 really spends one spool ({spent})")
    check(capped.printed_pixels > 0, "and the picture is still printed, not blanked")

    # 耗材方案 (m00480「多给用户选项，让用户自己选择用那一组耗材」): 匹配不再只给一个
    # 答案，右边「耗材方案」把几套搭配并排列出来，每套自带平均/最差 ΔE00、差块个数与
    # 卷数，点一下就换过去。每一套都必须落在「耗材上限」之内 —— 那是用户对这台机器的
    # 交代，不是程序可以越过的建议。
    from PySide6.QtWidgets import QLabel as _QLabel  # noqa: PLC0415

    from .core.image_matching import spool_usage as _spool_usage  # noqa: PLC0415
    from .ui.picture_page import BudgetOption as _PlanOption  # noqa: PLC0415

    check(
        bool(page._plans) and all(isinstance(option, _PlanOption) for option in page._plans),
        f"匹配给出 {len(page._plans)} 套耗材搭配",
    )
    check(
        page._plan_key == page._plans[0].key and page._base is page._plans[0].result,
        "第一套就是单答案匹配会给的那一套，也是页面正在显示的",
    )

    def _plan_rows() -> list[tuple[str, str]]:
        """Read (名称, 数字) straight off the panel instead of trusting the state."""
        found: list[tuple[str, str]] = []
        for index in range(len(page._plans)):
            row = page._plan_rows_layout.itemAt(index).widget()
            labels = [child.text() for child in row.findChildren(_QLabel)]
            button = page._plan_group.button(index)
            found.append((button.text() if button is not None else "", " ".join(labels)))
        return found

    def _panel_titles() -> list[str]:
        """The titles of the right-hand column, top to bottom."""
        column = page._right_splitter
        titles = []
        for position in range(column.count()):
            title = column.widget(position).findChild(_QLabel)
            titles.append(title.text() if title is not None else "")
        return titles

    def _overspent() -> list[str]:
        """名字 + 卷数 of every 方案 (or 用到的耗材 row) that breaks 耗材上限."""
        ceiling = page._settings().max_spools
        blamed = (
            [
                f"{option.title} {len(option.spools)} 卷"
                for option in page._plans
                if len(option.spools) > ceiling
            ]
            if ceiling
            else []
        )
        used = _spool_usage(page._base, page._library)
        if ceiling and len(used) > ceiling:
            blamed.append(f"用到的耗材 {len(used)} 卷")
        return blamed

    seen = _plan_rows()
    check(
        len(seen) == len(page._plans) and all(name for name, _numbers in seen),
        f"每一套都有一个单选行 ({[name for name, _numbers in seen]})",
    )
    check(
        all("平均 ΔE00" in numbers and "差块" in numbers and "卷" in numbers
            for _name, numbers in seen),
        "每行的平均/最差色差、差块个数和卷数都印在屏幕上，不用悬停",
    )
    check(
        len([button for button in page._plan_group.buttons() if button.isChecked()]) == 1,
        "同时只有一套被选中",
    )
    check(
        all(option.title and option.note and option.spools for option in page._plans),
        "每套都写清楚它是什么、代价是什么、用哪几卷",
    )
    check(not _overspent(), f"没有一套超过耗材上限 {page._settings().max_spools} ({_overspent()})")

    check(len(page._plans) >= 2, f"图片装不下时不只一套可选（{len(page._plans)} 套）")
    if len(page._plans) >= 2:
        chosen = page._plans[1]

        def _click_plan(index: int) -> None:
            """Click the real radio, so Qt moves the tick before the handler runs."""
            button = page._plan_group.button(index)
            check(button is not None, f"第 {index + 1} 套方案有一个可点的单选行")
            _QTest.mouseClick(button, _Qt.MouseButton.LeftButton)
            app.processEvents()

        _click_plan(1)
        check(page._plan_key == chosen.key, "点第二行就换成第二套")
        check(page._base is chosen.result, "而且页面真换了那张图（_base 换成它的结果）")
        check(page._dirty, "换方案之后工程文件需要重新保存")
        checked = [button.text() for button in page._plan_group.buttons() if button.isChecked()]
        check(checked == [chosen.title], f"单选状态跟着走 ({checked})")
        _click_plan(0)
        check(page._plan_key == page._plans[0].key, "点回第一行就回到第一套")

        # m00830 item 2: 换方案等于换一张图，上一份结果里的高亮块/描边不能留下来。
        page._select(0)
        app.processEvents()
        check(page._view._selected == 0, "点一个颜色，预览才画那一块的高亮")
        _click_plan(1)
        check(
            page._view._selected == -1
            and page._list.current() == -1
            and not page._list.selected_indices()
            and page._view._dim_pixmap() is None,
            "换一套方案之后，预览不再留着上一张图的高亮块和描边 "
            f"(view={page._view._selected}, row={page._list.current()})",
        )
        _click_plan(0)
        check(page._plan_key == page._plans[0].key, "再点回第一套")

    # 「用到的耗材」 and 「选中的颜色」 are not panels any more (1.6.13 #1/#2): the rolls
    # belong to the 方案 that spends them, and one colour's recipe is asked for by
    # double-clicking its row instead of taking a third of the column.  The lower
    # panel is called 「打印图片需要的颜色」 since 1.6.14 #2.
    check(
        _panel_titles() == ["耗材方案", "打印图片需要的颜色"],
        f"右栏两块：耗材方案 / 打印图片需要的颜色，got {_panel_titles()}",
    )
    check(
        page._plan_box.isAncestorOf(page._usage),
        "「用到的耗材」那一行在「耗材方案」面板里面",
    )
    check(
        page._usage.text().startswith("共") and "卷" in page._usage.text(),
        f"它按当前这一套列出各卷占比，got {page._usage.text()[:30]!r}",
    )
    check(
        all(
            use.display_name in page._usage.text()
            for use in _spool_usage(page._base, page._library)
        ),
        "「用到的耗材」点名列出了这套的每一卷",
    )
    # m01885 #1: 每一行自己带着这一套要装的那几卷的颜色，所以行与下面「共 N 卷」的列表
    # 说的是同一批卷，也不再需要给搭法起「（备选 N）」这种名字。
    rows = [
        page._plan_rows_layout.itemAt(index).widget()
        for index in range(page._plan_rows_layout.count())
    ]
    check(len(rows) == len(page._plans), f"每套方案占一行（{len(rows)} 行 / {len(page._plans)} 套）")
    chips = [
        [label for label in row.findChildren(_QLabel) if label.width() == 14]
        for row in rows
    ]
    check(
        [len(row) for row in chips] == [len(option.spools) for option in page._plans],
        f"行里的颜色方块数 = 这一套用几卷，got {[len(row) for row in chips]}",
    )
    colours = [
        [
            chip.styleSheet().split("background-color: ", 1)[1].split(";", 1)[0]
            for chip in row
        ]
        for row in chips
    ]
    wanted_colours = [
        [
            filament.color_hex if (filament := page._library.get(fid)) is not None else "#000000"
            for fid in option.spools
        ]
        for option in page._plans
    ]
    check(colours == wanted_colours, f"方块画的是这一套每一卷自己的颜色，got {colours}")
    check(
        all("备选" not in option.title for option in page._plans),
        f"方案名里不再有「（备选 N）」，got {[option.title for option in page._plans]}",
    )
    # m01885 #8: 面板上不再打印任何一句解释，解释在「帮助」里。
    check(
        page._plan_hint.text() == "" and not page._plan_hint.isVisibleTo(page._plan_box),
        f"有方案时面板上不写解释文本，got {page._plan_hint.text()!r}",
    )
    check(
        _plan_help() in window._help_text(),
        "三种口径的说明都在「帮助」窗口里",
    )
    # m01885 #3: 分隔条能一路往上拖，把「耗材方案」压到几乎没有。以前用量表的
    # QScrollArea 撑住了面板的最小高度，拖到一半就被顶回来。
    right = page._right_splitter
    room = max(200, right.height())
    right.setSizes([40, room])
    app.processEvents()
    check(
        right.sizes()[0] <= 100,
        f"分隔条能把「耗材方案」压到 {right.sizes()[0]} px（只剩标题 + 一小条），不再被内容顶住",
    )
    check(right.isCollapsible(0), "而且它还能被整块收起来")
    check(not right.isCollapsible(1), "下面的颜色列表不会被收掉")
    check(
        page._plan_scroll.minimumSizeHint().height() <= 80,
        f"面板的最小高度很小（{page._plan_scroll.minimumSizeHint().height()} px），所以拖得动",
    )
    right.setSizes([430, 350])
    app.processEvents()

    # 筛选 (1.6.13 #3): 「必须包含」/「不要用」, per spool, and every 方案 obeys it.
    from .ui.plan_filter import SpoolFilterDialog as _FilterDialog  # noqa: PLC0415

    wanted = sorted(page._plans[0].spools)[0]
    spare = [
        filament.id
        for filament in page._library.visible
        if all(filament.id not in option.spools for option in page._plans)
    ]
    banned = spare[0] if spare else ""
    dialog = _FilterDialog(
        page._library.visible, [wanted], [banned], cap=page._settings().max_spools
    )
    check(
        dialog.required() == (wanted,) and dialog.forbidden() == ((banned,) if banned else ()),
        "筛选窗口按传进来的答案先摆好每一行的状态",
    )
    row = dialog._list.item(0)
    dialog._on_row_clicked(row)
    check(
        dialog._list.item(0).text().startswith(("必须有 · ", "不要用 · ")),
        f"点一行就换一档状态，并写在行上，got {dialog._list.item(0).text()[:16]!r}",
    )
    dialog._on_clear()
    check(
        dialog.required() == () and dialog.forbidden() == (),
        "「全部不要求」把筛选清空",
    )
    dialog.close()

    page._require = frozenset({wanted})
    page._avoid = frozenset({banned}) if banned else frozenset()
    page._update_filter_hint()
    check(
        "必须有" in page._filter_hint.text() and "每一卷都可以用" not in page._filter_hint.text(),
        f"筛选条件写在「耗材方案」面板上，got {page._filter_hint.text()!r}",
    )
    page._on_rematch()
    app.processEvents()
    check(
        bool(page._plans) and all(wanted in option.spools for option in page._plans),
        f"「必须有」那一卷每一套都用上了（{[sorted(option.spools) for option in page._plans]}）",
    )
    check(
        all(not ({banned} & option.spools) for option in page._plans) if banned else True,
        "「不要用」那一卷一套都没碰",
    )
    check(not _overspent(), f"带筛选的搭配也没超过耗材上限 ({_overspent()})")
    page._require = frozenset()
    page._avoid = frozenset()
    page._update_filter_hint()
    page._on_rematch()
    app.processEvents()

    # 上限越小越容易越界:1 卷和 2 卷各重匹配一次,方案和「用到的耗材」都不许超。
    for small in (1, 2):
        page._spool_cap.setValue(small)
        page._on_rematch()
        app.processEvents()
        check(
            bool(page._plans) and not _overspent(),
            f"耗材上限 {small} 时每一套都在上限之内 ({_overspent()})",
        )
    page._spool_cap.setValue(_DEFAULT_SPOOL_CAP)
    page._on_rematch()
    app.processEvents()

    # 三色配方 (m00840 #1): always part of the pool, and the two checkboxes that
    # used to gate 混色 / 三色 are gone (m01647 item 5).
    from .core.triples import TripleCatalog as _TripleCatalog
    from .ui.picture_page import TriplePalette as _TriplePalette

    check(
        not hasattr(page, "_include_triples") and not hasattr(page, "_include_mixes"),
        "the 混色 / 三色 checkboxes are gone: every colour takes part",
    )
    head_pool = page._candidate_palette()
    check(
        not isinstance(page._match_palette(), _TriplePalette),
        "without a finished 3-colour table the pool is spools + pairs",
    )
    triples = _TripleCatalog(library.filaments, engine=ENGINE_MIXER).build()
    page.setTriples(triples, library)
    app.processEvents()
    pool = page._match_palette()
    check(
        isinstance(pool, _TriplePalette),
        "the 3-colour table joins the pool on its own, no checkbox to tick",
    )
    check(len(pool) > len(head_pool), f"the pool grows ({len(head_pool)} → {len(pool)})")
    check(
        all(entry.key == other.key for entry, other in zip(pool[: len(head_pool)], head_pool)),
        "the spools and pairs still come first, untouched",
    )
    tail = pool[len(head_pool)]
    check(
        len(tail.parts) == 3 and len(tail.filament_ids) == 3,
        f"the first new colour names three spools ({tail.label})",
    )
    check(
        sum(percent for _parent, percent in tail.parts) == 100,
        f"its recipe adds up to 100% ({tail.ratio_text})",
    )
    check(
        page._match_palette() is pool,
        "the pool is built once, not once per match",
    )
    page.setTriples(None, library)
    app.processEvents()
    check(
        not isinstance(page._match_palette(), _TriplePalette),
        "dropping the table (a spool changed) puts the pool back to spools + pairs",
    )

    page.buildPlate()
    plate = page.plate
    check(plate is not None, "a plate was built")
    if plate is not None:
        check(plate.width_mm > 0 and plate.depth_mm > 0, f"the plate is {plate.width_mm} × {plate.depth_mm} mm")
        check(plate.triangle_count > 0, f"the plate has {plate.triangle_count} triangles")

        # 「选中的颜色」: both colours, the recipe, and manual replacement with undo.
        from .core.mixes import SORT_SIMILARITY
        from .spectral import color as _selftest_color
        from .ui.colour_picker import ColourDetail, ColourPickerDialog

        check(isinstance(page._detail, ColourDetail), "the picture page shows the selected colour")
        check(
            page._base is not None
            and page._base.palette == page.result.palette
            and not page._replaced
            and not page._merged
            and not page._deleted
            and not page._redo_stack,
            # 1.6.16 (m01411 item 1) makes 重新匹配/换方案 undoable, so one step stays
            # on the undo stack on purpose — it holds the match this one replaced.
            "a fresh match keeps the automatic palette and no edits",
        )
        check(MatchSettings().merge_delta_e == 2.0, "near-identical matches merge by default")

        page._select(0)
        auto = page.result.palette[0]
        check(
            page._detail.image_hex() == _selftest_color.rgb_to_hex(page.result.region_colour(0)),
            "the panel shows the picture region's own colour",
        )
        check(
            page._detail.match_hex() == auto.color_hex.upper(),
            f"the panel shows the matched colour ({page._detail.match_hex()})",
        )
        check(
            "配方" in page._detail.recipe_text() or "耗材本色" in page._detail.recipe_text(),
            "the matched colour carries its recipe",
        )
        check(page._detail.can_replace(), "the colour can be replaced by hand")
        check(not page._detail.can_restore(), "nothing is hand-picked before a replacement")

        # 1.6.13 #2: 详情 is a dialog now, and double-clicking a row is how it is asked
        # for.  The dialog is not modal, so the next row can be asked about straight
        # away, and closing it must not change which colour is selected.
        rows = page._list._list
        row = page._list._row_of(0)
        check(row >= 0, "第 1 种颜色在列表里有自己的一行")
        target = rows.item(row)
        rows.scrollToItem(target)
        # A synthetic double click only reaches the view when the window is the
        # active one and the list holds the focus (measured: without this,
        # QTest.mouseDClick delivers nothing at all).
        window.raise_()
        window.activateWindow()
        rows.setFocus()
        app.processEvents()
        _QTest.qWait(30)
        spot = rows.visualItemRect(target).center()
        check(
            rows.viewport().rect().contains(spot),
            f"那一行在列表的可视区里（点 {spot.x()},{spot.y()}）",
        )
        fired: list[int] = []
        page._list.colourActivated.connect(fired.append)
        # A synthetic double click does NOT reach a QListWidget inside this window
        # (measured: QTest.mouseDClick delivers nothing, with or without the window
        # active and the list focused), so hand Qt's own double-click signal to the
        # view instead — that is the link the handler hangs off, and 合并 has relied
        # on the same signal since 1.6.x.
        rows.itemDoubleClicked.emit(target)
        app.processEvents()
        check(
            fired == [0],
            f"双击第 1 行让列表说出被双击的是哪种颜色，got {fired}",
        )
        check(
            page._detail_dialog is not None and page._detail_dialog.isVisible(),
            "双击颜色列表里的一行，弹出这个颜色的详情",
        )
        if page._detail_dialog is not None:
            check(
                "颜色详情" in page._detail_dialog.windowTitle(),
                f"弹窗标题写明是哪个颜色，got {page._detail_dialog.windowTitle()!r}",
            )
            page._detail_dialog.close()
            app.processEvents()
            check(
                not page._detail_dialog.isVisible() and page._list.current() >= 0,
                "关掉弹窗之后选中的颜色还在",
            )
        page._select(0)
        app.processEvents()
        check(page._delete_button.isEnabled(), "a selected colour can be deleted")
        check(not page._merge_button.isEnabled(), "合并 needs two colours, not one")

        menu = build_palette(page.library, page._catalog, include_mixes=True)
        picker = ColourPickerDialog(
            menu,
            page.result.region_colour(0),
            library=page.library,
            current_key=auto.key,
        )
        check(
            picker._sort.currentData() == SORT_SIMILARITY,
            "the replacement menu opens sorted by similarity to the picture colour",
        )
        ordered = picker.ordered()
        check(len(ordered) == len(menu), f"the replacement menu lists all {len(menu)} colours")
        check(
            ordered[0].color_hex.upper() == auto.color_hex.upper(),
            "the automatic match leads the similarity sort",
        )
        picker.close()

        replacement = next(entry for entry in ordered if entry.key != auto.key)
        # Exactly what _on_replace does: snapshot FIRST, then record the change,
        # which is why 撤销 covers a manual replacement with no extra button.
        page._push()
        page._replaced[0] = replacement
        page._apply_edits(keep=0)
        app.processEvents()
        check(
            page.result.palette[0].key == replacement.key,
            "the replacement is the colour the plate will print",
        )
        check(page._detail.can_restore(), "the panel knows the colour was hand-picked")
        check(page._undo_button.isEnabled(), "a replacement can be undone with 撤销")
        check(page._replace_button.isEnabled(), "a selected colour can be swapped")
        check(
            page._replace_button.property("accent") in (None, "false"),
            "更换颜色 is a plain button like 删除 / 合并, not a blue one",
        )
        page._undo()
        app.processEvents()
        check(
            page.result.palette[0].key == auto.key and page._replaced == {},
            "undoing the replacement puts the automatic match back",
        )
        check(not page._detail.can_restore(), "there is nothing left to undo")

        # 「删除」 makes the region transparent: no base plate, no prism. The mesh
        # builder already treats -1 that way, so the check is that the index
        # really becomes -1 and the plate loses exactly those pixels.
        import numpy as np  # noqa: PLC0415

        before_pixels = page.result.printed_pixels
        before_colours = len(page.result.palette)
        doomed = int(np.bincount(page.result.indices[page.result.indices >= 0].ravel()).argmax())
        doomed_count = int((page.result.indices == doomed).sum())
        page._list.select([doomed])
        check(page._multi_button.isCheckable(), "多选 is a toggle")
        check(not page._multi_button.isChecked(), "多选 starts off")
        page._multi_button.setChecked(True)
        app.processEvents()
        rows = [page._list._list.item(i) for i in range(page._list._list.count())]
        rows[0].setSelected(True)
        rows[1].setSelected(True)
        app.processEvents()
        check(
            len(page._list.selected_indices()) == 2,
            f"多选 lets two plain clicks keep two colours: {page._list.selected_indices()}",
        )
        check(page._merge_button.isEnabled(), "合并 lights up from a plain multi-selection")
        # m10238 item 4a: while 多选 is on, clicking a block in 「预览」 has to join
        # the selection too — the picture and the list are the same regions.
        before_toggle = set(page._list.selected_indices())
        page._on_region_clicked(2)
        app.processEvents()
        check(
            len(page._list.selected_indices()) == len(before_toggle) + 1,
            f"a click in 预览 adds a colour while 多选 is on: {page._list.selected_indices()}",
        )
        page._on_region_clicked(2)
        app.processEvents()
        check(
            set(page._list.selected_indices()) == before_toggle,
            "clicking the same block again takes it back out",
        )
        page._multi_button.setChecked(False)
        app.processEvents()
        check(
            len(page._list.selected_indices()) == 2,
            "the selection survives turning 多选 back off",
        )
        # m10238 item 4b: a click must never reshuffle the list, or the row under
        # the cursor becomes a different colour than the one just clicked.
        order_before = [
            page._list._list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(page._list._list.count())
        ]
        page._list.pin(1)
        app.processEvents()
        check(
            [
                page._list._list.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(page._list._list.count())
            ]
            == order_before,
            "clicking a colour never reorders the list",
        )
        check(page._list._row_of(1) == 1, "the clicked colour stays in its own row")
        page._list.clear_selection()
        page._list.select([doomed])
        page._on_delete_colours()
        app.processEvents()
        check(doomed in page._deleted, "the deleted colour is remembered as an edit")
        check(
            len(page.result.palette) == before_colours - 1,
            f"deleting one colour leaves {len(page.result.palette)}",
        )
        check(
            page.result.printed_pixels == before_pixels - doomed_count,
            "the deleted pixels are the ones that stop being printed",
        )
        check(bool((page.result.indices < 0).any()), "the deleted region is transparent")
        check(
            page._list.current() == -1 and not page._list.selected_indices(),
            f"删完之后列表不自动跳到第一行（m01885 第 7 条），got {page._list.current()}",
        )
        # m01411 item 3: 预览画着高亮，列表里就必须有一行是选中的——「多选」开着时
        # 露出高亮那一行的老写法会把选中又抹掉，于是列表空着、预览还罩着一层。
        check(
            page._view._selected == -1 and page._view._dim_pixmap() is None,
            f"删掉选中的那一块之后不再有高亮（选中 {page._view._selected}）",
        )
        check(page._undo_button.isEnabled(), "删除 can be undone")
        page._undo()
        app.processEvents()
        check(
            not page._deleted
            and len(page.result.palette) == before_colours
            and page.result.printed_pixels == before_pixels,
            "Ctrl+Z brings the deleted colour back",
        )
        page._redo()
        app.processEvents()
        check(
            len(page.result.palette) == before_colours - 1,
            "Ctrl+Y deletes it again",
        )
        page._undo()
        app.processEvents()

        # m01411 item 4: 点图片时怎么算选中了哪一块。原来是「点在哪一块就是哪一块」，
        # 现在还能选「按色块边缘选择」——取离点最近的那条边，细到一像素的眉毛才点得中。
        check(
            [page._pick_combo.itemText(i) for i in range(page._pick_combo.count())]
            == [_pp.PICK_MODE_LABELS[_pp.PICK_COLOUR], _pp.PICK_MODE_LABELS[_pp.PICK_EDGE]],
            "工具栏上能选「按颜色选择」和「按色块边缘选择」",
        )
        page._set_pick_mode(_pp.PICK_COLOUR, save=False)
        check(
            page._view.pickMode() == _pp.PICK_COLOUR,
            "默认还是老做法：按颜色选择",
        )
        page._pick_combo.setCurrentIndex(1)
        app.processEvents()
        check(
            page._view.pickMode() == _pp.PICK_EDGE,
            "在下拉里换一种，预览立刻跟着换",
        )
        check(
            _load_settings().get("pick_mode") == _pp.PICK_EDGE,
            "选过的那一种会记下来，下次开程序还是它",
        )
        # A synthetic 5x5 match: one painted pixel at (2, 2), the rest unpainted.
        # A real picture has no 「空白」 for the model to draw, so the two modes are
        # compared on a shape where the difference is unambiguous: the cursor sits
        # on a blank pixel whose nearest painted pixel is the one to the left of it.
        from dataclasses import replace as _dataclass_replace  # noqa: PLC0415

        lonely = _dataclass_replace(
            page.result,
            width=5,
            height=5,
            palette=[page.result.palette[0]],
            counts=np.asarray([1], dtype=np.int64),
            indices=np.full((5, 5), -1, dtype=np.int32),
        )
        lonely.indices[2, 2] = 0
        page._set_pick_mode(_pp.PICK_COLOUR, save=False)
        check(
            page._view._pick_index(lonely, 2, 2) == 0,
            "点在色块本身上，按颜色选择选中它",
        )
        check(
            page._view._pick_index(lonely, 3, 2) == -1,
            "按颜色选择：点在旁边的空白上什么都不选",
        )
        page._set_pick_mode(_pp.PICK_EDGE, save=False)
        check(
            page._view._pick_index(lonely, 2, 2) == 0,
            "点在色块本身上，按色块边缘选择也选中它",
        )
        check(
            page._view._pick_index(lonely, 3, 2) == 0,
            "按色块边缘选择：点在旁边的空白上选中最近的那一块",
        )
        page._set_pick_mode(_pp.PICK_COLOUR, save=False)

        # 1.6.17 (m02366 item 2/3)：切到「按色块边缘选择」之后，列表一行是一块，
        # 删除 / 合并 / 更换颜色都按块来，而且每一样都能撤销——包括切换选择方式本身。
        class _StubDialog:  # noqa: N801 自检里不弹窗：直接说「用户按了确定、挑了哪个」
            def __init__(self, choice, chosen):
                self._choice = choice
                self._chosen = chosen

            def exec(self):
                return _pp.QDialog.DialogCode.Accepted

            def chosen(self):
                return self._chosen if self._chosen is not None else self._choice

        page._set_pick_mode(_pp.PICK_EDGE, save=False)
        app.processEvents()
        blocks = page._blocks
        block_count = 0 if blocks is None else blocks.count
        check(blocks is not None and block_count > 0, f"按色块边缘选择数出 {block_count} 个色块")
        check(
            page._list.row_count() == block_count,
            f"列表一行一块（{page._list.row_count()} 行 / {block_count} 块）",
        )
        check(
            page._list.row_ids() == blocks.order(),
            f"列表从上到下就是块从大到小的顺序（{page._list.row_ids()[:4]}… / {blocks.order()[:4]}…）",
        )
        biggest = blocks.order()[0]
        page._list.clear_selection()
        page._on_region_clicked(biggest)
        app.processEvents()
        check(
            page._view._selected == -1
            and page._view._selected_mask is not None
            and np.array_equal(page._view._selected_mask, blocks.mask(biggest)),
            "点图片选中的是一整块：预览高亮这块的全部像素，不再按「一种颜色」高亮",
        )

        # 删除：只让选中的那几块留空，别的块照打。
        block_pixels = page.result.printed_pixels
        steps = len(page._undo_stack)
        page._on_delete_colours()
        app.processEvents()
        check(
            page.result.printed_pixels == block_pixels - blocks.size(biggest),
            f"删除一块只让这一块留空（少印 {block_pixels - page.result.printed_pixels} 像素）",
        )
        check(len(page._undo_stack) == steps + 1, "块级删除记了一步历史")
        page._undo()
        app.processEvents()
        check(page.result.printed_pixels == block_pixels, "Ctrl+Z 把这一块印回来")

        # 合并：在选中的块里挑一种颜色，所有选中的块都变成它。
        page._set_pick_mode(_pp.PICK_EDGE, save=False)
        app.processEvents()
        blocks = page._blocks
        pair = list(blocks.order()[:2]) if blocks is not None and blocks.count >= 2 else []
        merged = False
        if len(pair) == 2 and blocks.value(pair[0]) != blocks.value(pair[1]):
            merged = True
            page._list.select(pair)
            keep_entry = page.result.palette[blocks.value(pair[1])]
            real_merge = _pp.MergeDialog
            _pp.MergeDialog = lambda *args, **kwargs: _StubDialog(1, None)
            try:
                page._on_merge_colours()
                app.processEvents()
            finally:
                _pp.MergeDialog = real_merge
            picked = page._picked_blocks()[1]
            keys = {
                page.result.palette[value].key
                for value in page.result.indices[picked].tolist()
            }
            check(keys == {keep_entry.key}, "合并把选中的块都变成挑中的那种颜色")
            check(len(page._block_edits) == 2, "合并给每一块各记了一条色块级改动")
            page._undo()
            app.processEvents()
            check(not page._block_edits, "Ctrl+Z 撤销合并，色块级改动回到空")

        # 更换颜色：选中的块一起换。
        if merged:
            page._list.select(pair)
            current = {page.result.palette[blocks.value(label)].key for label in pair}
            wanted = next(
                (entry for entry in page._candidate_palette() if entry.key not in current),
                None,
            )
            if wanted is not None:
                real_picker = _pp.ColourPickerDialog
                _pp.ColourPickerDialog = lambda *args, **kwargs: _StubDialog(0, wanted)
                try:
                    page._on_replace()
                    app.processEvents()
                finally:
                    _pp.ColourPickerDialog = real_picker
                picked = page._picked_blocks()[1]
                hexes = {
                    page.result.palette[value].color_hex
                    for value in page.result.indices[picked].tolist()
                }
                check(hexes == {wanted.color_hex}, "更换颜色把选中的块一起换成新颜色")
                check(
                    wanted.key in [entry.key for entry in page.result.palette],
                    "新颜色进了色表",
                )
                page._undo()
                app.processEvents()
                check(not page._block_edits, "Ctrl+Z 撤销更换颜色")

        # 切换选择方式本身也算一步：Ctrl+Z 能把用户带回上一种模式。
        page._set_pick_mode(_pp.PICK_COLOUR, save=False)
        app.processEvents()
        steps = len(page._undo_stack)
        page._pick_combo.setCurrentIndex(1)
        app.processEvents()
        check(page._mode == _pp.PICK_EDGE, "从下拉里切到按色块边缘选择")
        check(len(page._undo_stack) == steps + 1, "切换选择方式算一步历史")
        page._undo()
        app.processEvents()
        check(page._mode == _pp.PICK_COLOUR, "Ctrl+Z 把选择方式也退回去")
        page._redo()
        app.processEvents()
        check(page._mode == _pp.PICK_EDGE, "Ctrl+Y 再切回来")
        page._undo()
        app.processEvents()
        check(
            page._list.row_count() == len(page.result.palette),
            "回到颜色模式：列表又是一行一种颜色",
        )

        # 1.6.19 (m01566 item 1)：Del 就是点「删除」。事件直接派给页面／列表，焦点在
        # 输入框里时两边都必须放手——那个 Del 是在删字符。
        import numpy as _np  # noqa: PLC0415
        from PySide6.QtCore import QEvent as _QEvent2  # noqa: PLC0415
        from PySide6.QtGui import QKeyEvent as _QKeyEvent  # noqa: PLC0415

        def _del() -> None:
            app.sendEvent(
                page._list._list,
                _QKeyEvent(
                    _QEvent2.Type.KeyPress,
                    _Qt.Key.Key_Delete,
                    _Qt.KeyboardModifier.NoModifier,
                ),
            )
            app.processEvents()

        page._set_pick_mode(_pp.PICK_EDGE, save=False)
        app.processEvents()
        blocks = page._blocks
        if blocks is not None and blocks.count:
            whole = blocks.mask(blocks.order()[0])
            grid = _pp._to_image_grid(_np.asarray(whole, dtype=bool), page._image.size)
            page._list.clear_selection()
            page._on_region_clicked(blocks.order()[0])
            app.processEvents()
            before_del = page.result.printed_pixels
            page._list.setFocus()
            app.processEvents()
            check(not page._typing_in_field(), "焦点不在输入框里：Del 归「删除」")
            # 焦点在数字框里时，Del 属于那个框，页面不能抢。
            page._colour_count.setFocus()
            app.processEvents()
            check(page._typing_in_field(), "焦点在数字框里：Del 归输入框")
            app.sendEvent(
                page,
                _QKeyEvent(
                    _QEvent2.Type.KeyPress,
                    _Qt.Key.Key_Delete,
                    _Qt.KeyboardModifier.NoModifier,
                ),
            )
            app.processEvents()
            check(
                page.result.printed_pixels == before_del and not page._block_edits,
                "在输入框里按 Del 不会删掉色块",
            )
            page._list.setFocus()
            app.processEvents()
            _del()
            check(
                page.result.printed_pixels == before_del - int(whole.sum()),
                f"按 Del 和点「删除」一样（少了 {int(whole.sum())} 像素）",
            )
            # 1.6.19 (m01566 item 2)：删除同时改「原图」本身，并挂出「原图已修改」。
            check(page._original_edited, "删除把原图一起改了")
            check(
                page._original_badge is not None and page._original_badge.isVisible(),
                "原图标题边上出现「原图已修改」",
            )
            del_alpha = _np.asarray(page._image.convert("RGBA"))[..., 3][grid]
            check(
                int(del_alpha.max()) == 0,
                f"删掉的那些像素在原图里真的透明了（最大不透明度 {int(del_alpha.max())}）",
            )
            check(
                page._original_means(page.result) is not None,
                "改过的原图能量出每个色块自己的颜色",
            )
            page._undo()
            app.processEvents()
            check(
                not page._original_edited
                and not page._original_badge.isVisible()
                and int(
                    _np.asarray(page._image.convert("RGBA"))[..., 3][grid].min()
                )
                == 255,
                "Ctrl+Z 把原图也退回去：像素重新不透明、标记消失",
            )
        # m02316 item 3：「删除」两个字就是全部，键位由 Esc 那条快捷方式负责（Del 仍然
        # 好使，见上）——按钮上挂着「(Del)」反而像按钮名字的一部分。
        check(
            page._delete_button.text() == "删除",
            f"「删除」按钮不再写着 (Del)（现在是 {page._delete_button.text()!r}）",
        )
        # m02316 item 5：这一栏装什么，标题就写什么。
        check(
            page._list_panel.title_label.text() == _pp.LIST_TITLE_BLOCK,
            f"块模式下这一栏叫「{_pp.LIST_TITLE_BLOCK}」（现在是 "
            f"{page._list_panel.title_label.text()!r}）",
        )
        # m02316 item 4：只改一块就只选中那一块。
        if page._blocks is not None and page._blocks.count:
            solo = page._blocks.order()[0]
            page._list.clear_selection()
            page._on_region_clicked(solo)
            app.processEvents()
            wanted_entry = next(
                (
                    entry
                    for entry in page._candidate_palette()
                    if entry.key != page.result.palette[page._blocks.value(solo)].key
                ),
                None,
            )
            if wanted_entry is not None:
                real_picker = _pp.ColourPickerDialog
                _pp.ColourPickerDialog = lambda *args, **kwargs: _StubDialog(0, wanted_entry)
                try:
                    page._on_replace()
                    app.processEvents()
                finally:
                    _pp.ColourPickerDialog = real_picker
                picked_rows = page._list.selected_indices()
                current = page._list.current()
                check(
                    len(picked_rows) == 1 and current == picked_rows[0],
                    f"只改一块就只选中那一块（选中 {picked_rows}，当前 {current}）",
                )
                # 块模式下预览画的是「选中的这几块」的并集（掩码），不是调色板下标：
                # 画的必须是这一块自己的像素，而不是相邻那块、也不是整片同色区域。
                highlight = page._view._selected_mask
                check(
                    highlight is not None
                    and int((highlight & page._blocks.mask(current)).sum())
                    == int(highlight.sum()),
                    "预览画的正是那一块的像素",
                )
                page._undo()
                app.processEvents()
            # m02316 item 6：单选时再点一下已经选中的那一块 = 取消选中，不是跳到别的一块。
            page._list.clear_selection()
            page._on_region_clicked(solo)
            app.processEvents()
            shown = page._shown
            rows = page._list._list
            row = page._list._row_of(shown)
            item = rows.item(row)
            rect = rows.visualItemRect(item)
            if not rect.isEmpty():
                _QTest.mouseClick(rows.viewport(), _Qt.MouseButton.LeftButton, pos=rect.center())
                app.processEvents()
                check(
                    page._shown == -1
                    and page._list.current() == -1
                    and not page._list.selected_indices()
                    and page._view._selected_mask is None,
                    "块模式下再点一次取消选中，而不是选中另一块"
                    f"（_shown={page._shown}，当前 {page._list.current()}）",
                )
        page._set_pick_mode(_pp.PICK_COLOUR, save=False)
        app.processEvents()
        check(
            page._list_panel.title_label.text() == _pp.LIST_TITLE_COLOUR,
            f"颜色模式下这一栏叫「{_pp.LIST_TITLE_COLOUR}」（现在是 "
            f"{page._list_panel.title_label.text()!r}）",
        )

        # m02316 item 2：按颜色删除也改「原图」——用户在图三里报的就是「原图没跟着预览改」。
        # 一层一层都改图：删掉一种颜色 = 这些像素在源图里变透明，和按块删除是同一件事
        # （m02039 的「原图只改色块」从此作废，用户后来选了「颜色也改图」）。
        if len(page.result.palette) >= 1:
            page._list.select([0])
            before_colour_image = page._image.tobytes()
            page._on_delete_colours()
            app.processEvents()
            check(
                page._image.tobytes() != before_colour_image
                and page._original_edited
                and page._original_badge.isVisible(),
                "按颜色删除也改原图，并挂出「原图已修改」",
            )
            check(
                page._original_means(page.result) is not None,
                "改过的原图照样能量出每块自己的颜色",
            )
            page._undo()
            app.processEvents()
            check(
                page._image.tobytes() == before_colour_image
                and not page._original_edited,
                "Ctrl+Z 把按颜色删除改过的原图也退回去",
            )

        # 1.6.19：「更换颜色」只决定这块地方打印成什么颜色，原图一个像素都不动。
        if len(page.result.palette) >= 1:
            page._list.select([0])
            wanted = next(
                (
                    entry
                    for entry in page._candidate_palette()
                    if entry.key != page.result.palette[0].key
                ),
                None,
            )
            if wanted is not None:
                before_image = page._image.tobytes()
                real_picker = _pp.ColourPickerDialog
                _pp.ColourPickerDialog = lambda *args, **kwargs: _StubDialog(0, wanted)
                try:
                    page._on_replace()
                    app.processEvents()
                finally:
                    _pp.ColourPickerDialog = real_picker
                check(
                    page._image.tobytes() == before_image,
                    "「更换颜色」一个像素都不改原图",
                )
                page._undo()
                app.processEvents()

        # 合并: only reachable through the dialog, so drive the non-UI half.
        if len(page.result.palette) >= 2:
            keep = page.result.palette[1].key
            page._list.select([0, 1])
            page._push()
            page._merged[page._live_map[0]] = page._live_map[1]
            page._apply_edits(keep=page._live_map[1])
            app.processEvents()
            check(
                len(page.result.palette) == before_colours - 1
                and page.result.palette[0].key == keep,
                "merging two colours leaves the chosen one",
            )
            check(
                page.result.printed_pixels == before_pixels,
                "merging prints every pixel the two colours covered",
            )
            # m01411 item 3: 合并之后活下来的那一块会被选中，而预览的高亮必须跟列表一
            # 致——列表说选中的是它，预览画的也是它。
            check(
                page._list.current() >= 0
                and page._view._selected == page._list.current()
                and page._view._dim_pixmap() is not None,
                "合并后列表和预览指的是同一块"
                f"（列表 {page._list.current()}，预览 {page._view._selected}）",
            )
            page._undo()
            app.processEvents()
            check(
                not page._merged and len(page.result.palette) == before_colours,
                "undoing the merge restores both colours",
            )

        # 1.6.21 (m01191 item 2)：合并写回原图的是「留下来的那一块**本来的**颜色」，不是
        # 预览里那卷耗材印出来的颜色。用户的规矩写得很清楚：原图色块 a 对应预览色块 b、
        # 原图色块 c 对应预览色块 d，b 和 d 合并之后原图里 c 也并进 a，用的是 a 自己的
        # 颜色——所以原图的颜色集合一点都不变，变的只有色块大小。
        if len(page.result.palette) >= 2:
            picked = page._list.row_ids()[:2]
            page._list.select(picked)
            app.processEvents()
            keeper = int(picked[1])
            indices = _np.asarray(page.result.indices)
            touched = _np.isin(indices, _np.asarray(picked, dtype=indices.dtype))
            colours_before = {
                tuple(int(value) for value in pixel)
                for pixel in _np.unique(
                    _np.asarray(page._image.convert("RGBA"))[..., :3].reshape(-1, 3), axis=0
                )
            }
            keeper_colour = page._picture_colour(indices == keeper)
            check(keeper_colour is not None, "能量出留下来的那一块在原图里本来的颜色")
            before_merge_image = page._image.tobytes()
            real_merge = _pp.MergeDialog
            _pp.MergeDialog = lambda *args, **kwargs: _StubDialog(1, None)
            try:
                page._on_merge_colours()
                app.processEvents()
            finally:
                _pp.MergeDialog = real_merge
            check(page._original_edited, "合并也改原图")
            painted = _pp._to_image_grid(touched, page._image.size)
            pixels = _np.asarray(page._image.convert("RGBA"))
            merged = pixels[painted][pixels[painted][:, 3] > 0][:, :3]
            check(
                merged.size > 0
                and bool((merged == _np.asarray(keeper_colour, dtype=merged.dtype)).all()),
                "原图里并掉的那些像素用的是留下来的那一块本来的颜色"
                f"（期望 {keeper_colour}）",
            )
            colours_after = {
                tuple(int(value) for value in pixel)
                for pixel in _np.unique(
                    _np.asarray(page._image.convert("RGBA"))[..., :3].reshape(-1, 3), axis=0
                )
            }
            check(
                colours_after <= colours_before,
                f"合并不会让原图多出颜色（多了 {sorted(colours_after - colours_before)}）",
            )
            check(
                len(page.result.palette) == before_colours - 1,
                "合并在预览里仍然是少了一种颜色",
            )
            page._undo()
            app.processEvents()
            check(
                page._image.tobytes() == before_merge_image,
                "Ctrl+Z 把合并改过的原图也退回去",
            )

        # 1.6.21 (m01191 item 1)：列表里一个都没选中，预览就不画高亮、不画描边。
        # 用户报的正是这条（未选中颜色时预览还显示色块），而他说不清自己是怎么把选中取消掉
        # 的，所以这里按「选中集是空的」本身来验证，而不是按某一条路来验证。
        one = page._list.row_ids()
        if one:
            page._list.select([one[0]])
            app.processEvents()
            check(page._view._dim_pixmap() is not None, "选中一种颜色时预览画着它")
            rows = page._list._list
            rows.clearSelection()
            rows.setCurrentRow(-1)
            app.processEvents()
            check(
                page._view._selected == -1
                and page._view._dim_pixmap() is None
                and page._list.current() == -1
                and not page._list.selected_indices(),
                "选中集空掉之后预览什么都不画（点列表下面空白处那条路）",
            )
        page._set_pick_mode(_pp.PICK_EDGE, save=False)
        app.processEvents()
        if page._blocks is not None and page._blocks.count:
            page._list.select([page._list.row_ids()[0]])
            app.processEvents()
            check(page._view._selected_mask is not None, "块模式下选中的那一块画了出来")
            page._list._list.clearSelection()
            page._list._list.setCurrentRow(-1)
            app.processEvents()
            check(
                page._view._selected_mask is None
                and page._list.current() == -1
                and not page._list.selected_indices(),
                "块模式下选中集空掉，预览的描边也收掉",
            )
            page._set_pick_mode(_pp.PICK_COLOUR, save=False)
            app.processEvents()

        # 1.6.13 #4: 「按当前色块重新匹配」 takes the colours on screen as settled — a
        # deleted colour stays deleted — and only chooses the rolls behind them again.
        keep_count = len(page.result.palette)
        page._list.select([0])
        page._on_delete_colours()
        app.processEvents()
        check(bool(page._deleted), "先真的删掉一种颜色")
        unpainted = page.result.total_pixels - page.result.printed_pixels
        # m01411 item 1: 这一次把整张匹配都换掉（新的 _base、新的方案表），所以它必须
        # 进历史——用户按了它、不喜欢结果，Ctrl+Z 得能回到按之前。
        before_plans = [option.key for option in page._plans]
        before_base = page._base
        before_key = page._plan_key
        before_undo = len(page._undo_stack)
        page._on_rematch_blocks()
        app.processEvents()
        check(
            len(page._undo_stack) == before_undo + 1,
            f"按当前色块重新匹配会记一步历史（{len(page._undo_stack)} 步）",
        )
        check(
            bool(page._plans) and page._base is page._plans[0].result,
            f"按当前色块重新匹配给出 {len(page._plans)} 套搭配，并显示第一套",
        )
        check(
            0 < len(page._base.palette) <= keep_count - 1,
            f"色块不会变多（{len(page._base.palette)} 块，删除前 {keep_count} 块）",
        )
        check(
            len({entry.key for entry in page._base.palette}) == len(page._base.palette),
            "同一卷不会在两块里重复出现（两块落到同一卷上会合成一块）",
        )
        check(
            page._base.total_pixels - page._base.printed_pixels == unpainted,
            "被删掉的色块没有回来（不打印的像素数没变）",
        )
        check(not _overspent(), f"它给出的搭配也在耗材上限之内 ({_overspent()})")
        check(
            all(option.key != "extra" for option in page._plans),
            "搭配里没有越界的「多用 1 卷」",
        )
        page._undo()
        app.processEvents()
        check(
            page._base is before_base
            and [option.key for option in page._plans] == before_plans
            and page._plan_key == before_key,
            "撤销把被换掉的那张匹配、方案表和勾选的那一套一起放回去",
        )
        check(
            bool(page._deleted) and len(page._undo_stack) == before_undo,
            "撤销之后之前那一步（删掉的色块）还在，历史长度也退回去了",
        )
        page._redo()
        app.processEvents()
        page._on_rematch()
        app.processEvents()
        # m02316 item 2：按颜色删除现在连原图一起改，所以「重新匹配」从改过的图出发——
        # 删掉的那些像素已经透明了，不会再被匹配回来（这正是用户要的：图里没有它，方案里
        # 也不该有它）。m02039 的「按颜色不动原图、删了还会回来」到此结束。
        opaque = float((_np.asarray(page._image.convert("RGBA"))[..., 3] > 0).mean())
        printed = page.result.printed_pixels / max(1, page.result.total_pixels)
        check(
            not page._deleted
            and page.result.printed_pixels < before_pixels
            and abs(printed - opaque) < 0.01,
            f"「重新匹配」从改过的原图出发，删掉的颜色没有回来"
            f"（不透明 {opaque:.1%}、打印 {printed:.1%}）",
        )

        # 1.6.22 (m01647)：导出时的「圆滑色块」。默认勾上；勾上之后模型是在细一档的
        # 栅格上重新分类的，斜着走的边界不再是台阶，但整块仍然是一块闭合实体——这是
        # 轮廓平滑法做不到的那条（1.6.5 报过 14,650 条开口边）。
        from .mesh.plate import edge_health as _edge_health
        from .mesh.plate import plate_mesh as _plate_mesh

        check(
            page._smooth_box.text() == _pp.SMOOTH_LABEL,
            f"底栏那个勾叫「{_pp.SMOOTH_LABEL}」（现在是 {page._smooth_box.text()!r}）",
        )
        check(page._smooth_box.isChecked(), "它默认是勾上的")
        check(
            page._plate_settings().smooth_edges,
            "勾上的状态进了 PlateSettings",
        )
        smooth_plate = page.buildPlate()
        smooth_size = smooth_plate.pixel_size_mm
        smooth_tris = smooth_plate.triangle_count
        smooth_points, smooth_faces, _smooth_parts = _plate_mesh(smooth_plate)
        _smooth_open, _smooth_manifold = _edge_health(smooth_points, smooth_faces)
        check(
            _smooth_open == 0 and _smooth_manifold == 0,
            f"圆滑之后的整块还是闭合实体（开口 {_smooth_open}、非流形 {_smooth_manifold}）",
        )
        check(
            any("圆滑" in note for note in smooth_plate.notes),
            f"状态里说明了这次是圆滑过的（{smooth_plate.notes}）",
        )
        page._smooth_box.setChecked(False)
        app.processEvents()
        check(not page._plate_settings().smooth_edges, "取消勾选之后 PlateSettings 里就是 False")
        plain_plate = page.buildPlate()
        check(
            abs(plain_plate.pixel_size_mm - plain_plate.width_mm / page.result.width) < 1e-9,
            f"不勾时网格就是图片自己那一格（{plain_plate.pixel_size_mm:.4f} mm）",
        )
        check(
            smooth_size < plain_plate.pixel_size_mm,
            f"勾上时网格更细（{smooth_size:.4f} mm vs {plain_plate.pixel_size_mm:.4f} mm）",
        )
        check(
            abs(plain_plate.width_mm - smooth_plate.width_mm) < 1e-9
            and abs(plain_plate.depth_mm - smooth_plate.depth_mm) < 1e-9
            and abs(plain_plate.volume_mm3 - smooth_plate.volume_mm3)
            / smooth_plate.volume_mm3 < 0.02,
            f"圆滑不改板子的宽、深和体积（{smooth_plate.volume_mm3:.1f} → {plain_plate.volume_mm3:.1f} mm³）",
        )
        check(
            plain_plate.triangle_count > 0 and smooth_tris > 0,
            f"两条路的三角面数都是正的（圆滑 {smooth_tris}）",
        )
        page._smooth_box.setChecked(True)
        app.processEvents()
        plate = page.buildPlate()

        three_mf = write_3mf(
            plate,
            scratch / "selftest.3mf",
            object_name="自检底板",
            filaments=slots_from_palette(page.result.palette, page.library),
        )
        obj = write_obj(plate, scratch / "selftest.obj")
        check(three_mf.is_file() and three_mf.stat().st_size > 0, f"3MF written ({three_mf.stat().st_size} B)")
        check(obj.is_file() and obj.stat().st_size > 0, f"OBJ written ({obj.stat().st_size} B)")
        check((scratch / "selftest.mtl").is_file(), "MTL written")

        if zipfile.is_zipfile(three_mf):
            with zipfile.ZipFile(three_mf) as archive:
                names = set(archive.namelist())
            needed = {
                "[Content_Types].xml",
                "_rels/.rels",
                "3D/3dmodel.model",
                "Metadata/model_settings.config",
            }
            check(needed <= names, "the 3MF holds every required entry")
        else:
            check(False, "the 3MF is a zip package")

    # 「颜色配方」页面：长说明搬到状态栏，右边面板不再被那条文字钉死宽度。
    from PySide6.QtWidgets import QLabel as _QLabel
    from PySide6.QtWidgets import QSizePolicy as _QSizePolicy
    from PySide6.QtWidgets import QSplitter as _QSplitter
    from PySide6.QtWidgets import QStatusBar as _QStatusBar

    from .spectral import color as _selftest_colour

    check(not hasattr(window, "_grid_info"), "the overflowing one-line note is gone")
    check(
        "每两种耗材" in window._status.text() and "10%" in window._status.text(),
        f"the bottom line carries the rules, got {window._status.text()[:60]!r}",
    )
    check(
        window._status.sizePolicy().horizontalPolicy() == _QSizePolicy.Policy.Ignored,
        "the bottom line can never pin the window's width",
    )
    check(window._status.wordWrap(), "the bottom line wraps instead of clipping its tail")
    check(
        window.findChild(_QStatusBar) is None,
        "the sentence is not in a one-line status bar that would cut it off",
    )
    tab_note = window._status.text()
    window._tabs.setCurrentIndex(1)
    app.processEvents()
    check(
        window._status.text() == "",
        f"the colour-table sentence is hidden on 图像转换, got {window._status.text()[:40]!r}",
    )
    window._tabs.setCurrentIndex(0)
    app.processEvents()
    check(
        window._status.text() == tab_note,
        "switching back to 颜色配方 restores the colour-table sentence",
    )

    # The first page is 「颜色配方」 everywhere: tab, panel title and the tooltip
    # on the target colour must not disagree about what the page is called.
    check(
        window._tabs.tabText(0) == "颜色配方",
        f"the first tab is 颜色配方, got {window._tabs.tabText(0)!r}",
    )
    check(
        window._detail._title.text() == "颜色配方",
        f"the right-hand panel is 颜色配方, got {window._detail._title.text()!r}",
    )
    # No hover pop-ups anywhere on this window any more (m01647 item 4): the page
    # name lives in the tab and the panel title, not in a black box on hover.
    check(
        window._target_color.toolTip() == "" and window._find_button.toolTip() == "",
        "the target colour and 找最接近的颜色 carry no tooltip",
    )

    # The version must be readable without opening 帮助, and it is the same string
    # build/BambuPalette.spec parses out of app/__init__.py for the exe resource.
    from . import __version__ as _version

    about = window._about_text()
    check(_version in about, f"帮助 names the running version {_version!r}")
    check(about.startswith("BambuPalette"), "帮助 still opens with the product name")
    check(
        "BambuPalette.exe 的文件属性里也写着同一个版本号。" in about,
        "帮助 points at the exe's file properties",
    )
    # m01885 #5/#6/#8: 右上角那颗按钮叫「帮助」，点开是一个窗口，里面装着图像转换那
    # 一整段说明、三种方案口径的说明，最后才是版本与算法出处。
    check(
        window._help_button.text() == "帮助",
        f"右上角是「帮助」按钮，got {window._help_button.text()!r}",
    )
    check(
        not hasattr(window, "_about_button"),
        "旧的「关于」按钮已经不在了（1.6.14 #5）",
    )
    check(
        all(
            part in window._help_text()
            for part in (window._picture_page.help_text(), _plan_help(), about)
        ),
        "帮助窗口 = 图像转换说明 + 方案说明 + 版本与算法出处",
    )
    window._on_help()
    app.processEvents()
    manual = window._help_window
    check(manual is not None and manual.isVisible(), "点「帮助」会弹出窗口")
    if manual is not None:
        check(
            manual.windowTitle() == "帮助" and manual.text() == window._help_text(),
            f"窗口标题与正文对得上，got {manual.windowTitle()!r}",
        )
        check(manual.parent() is window, "帮助窗口挂在主窗口上，不是独立的一份")
        check(
            manual._scroll.widget() is not None and manual._scroll.widget() is manual._body,
            "正文放在可滚动的区域里，长文本能翻",
        )
        manual.close()
    app.processEvents()
    check(window._find_button.isEnabled(), "找最接近的颜色 is clickable before a target is picked")
    check(
        _version in window.windowTitle(),
        f"the title bar carries the version, got {window.windowTitle()!r}",
    )

    # m10238 item 2: with an empty 目标颜色 the button must SEARCH, not just print
    # a hint — it borrows the colour the user is looking at and says so.
    window._target_color.clear()
    window._clear_filters()
    app.processEvents()
    borrowed_cell = window._recipes[len(window._recipes) // 2]
    window._grid.selectRecipe(borrowed_cell)
    window._find_button.click()
    app.processEvents()
    check(
        "目标颜色取自" in window._status.text(),
        f"an empty target borrows one instead of refusing, got {window._status.text()[:70]!r}",
    )
    check(
        borrowed_cell.color_hex.upper() in window._status.text().upper(),
        "the borrowed target is the colour the user was looking at",
    )
    check(
        window._target_color.hex().upper() == borrowed_cell.color_hex.upper(),
        "the borrowed colour is shown in the 目标颜色 field",
    )
    window._target_color.clear()
    window._grid.clearSelection()
    app.processEvents()
    window._find_button.click()
    app.processEvents()
    check(
        "数字框里的 RGB" in window._status.text(),
        f"with no table selection the R/G/B numbers are used, got {window._status.text()[:70]!r}",
    )

    # A spool's own colour has exactly ONE parent — itself.  Row 3 used to keep
    # showing the previous three-colour recipe's 「耗材丝3 … 12%」, so a raw spool
    # looked like it was mixed from two spools that had nothing to do with it.
    spool_cell = next(
        (cell for cell in window._recipes if cell.filaments and not cell.recipes), None
    )
    if spool_cell is not None:
        window._on_grid_selected(spool_cell)
        app.processEvents()
        rows = window._detail._rows
        check(
            [row["frame"].isHidden() for row in rows] == [False] + [True] * (len(rows) - 1),
            f"a spool colour shows exactly one parent row, hidden={[r['frame'].isHidden() for r in rows]}",
        )
        check(
            rows[0]["percent"].text().startswith("100%"),
            f"the one visible row is the 100% spool, got {rows[0]['percent'].text()!r}",
        )
    recipe_splitter = window._tabs.widget(0).findChild(_QSplitter)
    if recipe_splitter is not None:
        pinned = [
            recipe_splitter.widget(index).minimumSizeHint().width()
            for index in range(recipe_splitter.count())
        ]
        check(
            max(pinned) < 720,
            f"every 颜色配方 pane can still be dragged, pinned widths {pinned}",
        )

    # 「找最接近的颜色」把耗材本色也算进去：问一个自己就有的颜色，答案就是它。
    if spool_rows:
        own = spool_rows[0]
        answer = window.nearest_recipe(own.color_hex)
        check(answer is not None, "找最接近的颜色 answers for a spool's own colour")
        if answer is not None:
            gap = _selftest_colour.delta_e_2000(
                answer.lab,
                tuple(_selftest_colour.lab_from_rgb(_selftest_colour.hex_to_rgb(own.color_hex))),
            )
            check(gap <= 1e-6, f"the spool's own colour answers itself, ΔE00 {gap:.4f}")
        window._target_color.setValue(own.color_hex)
        window._find_button.click()
        app.processEvents()
        check(
            "耗材本色" in window._status.text(),
            f"the answer says so when it is a raw spool, got {window._status.text()[:60]!r}",
        )
        window._clear_filters()
        app.processEvents()

    # 提示、配方、状态栏全是要粘进 Bambu Studio 的文本，都该能用鼠标选中。
    selectable_flag = Qt.TextInteractionFlag.TextSelectableByMouse
    blind = [
        label.text()[:32]
        for label in window.findChildren(_QLabel)
        if label.text().strip() and not (label.textInteractionFlags() & selectable_flag)
    ]
    check(not blind, f"every label in the window can be selected, blind: {blind[:4]}")

    # A three-filament recipe must NOT be narrowed to its first two spools: that
    # showed a different colour's 81 two-colour ratios.  The window was built
    # with auto_triples=False to keep every count above deterministic, so the
    # table is asked for explicitly here.
    window._auto_triples = True
    window._ensure_triples()
    for _ in range(20000):
        if window._triples is not None and window._triples.built:
            break
        app.processEvents()
    check(
        window._triples is not None and window._triples.built,
        "the three-filament table can be built on demand",
    )
    combined = window._combined
    triple_only = None
    if combined is not None and window._triples is not None:
        entries = window._triples.colours()
        for position in range(min(len(entries), 800)):
            found = combined.index_of_key(f"colour|{entries[position].color_hex}")
            if found >= 0 and combined.pair_colour_at(found) is None:
                triple_only = combined[found]
                break
    check(triple_only is not None, "the triple table holds a colour no two-colour mix reaches")
    if triple_only is not None:
        first_recipe = triple_only.recipes[0]
        check(hasattr(first_recipe, "percent_c"), "that colour is reached by a three-filament recipe")
        window._on_grid_selected(triple_only)
        app.processEvents()
        check(window._pair_filter is None, "selecting a triple does not filter to two spools")
        check(
            window._detail.colour() is triple_only,
            "selecting a triple keeps showing that same colour",
        )
    # The double-click jump is gone: activating a two-colour mix must not narrow
    # the grid either.  Only the detail panel's own button sets ``_pair_filter``.
    pair_cell = None
    if combined is not None:
        for position in range(len(combined)):
            if combined[position].recipes and combined.pair_colour_at(position) is not None:
                pair_cell = combined[position]
                break
    check(pair_cell is not None, "the combined table holds a two-colour mix to test with")
    if pair_cell is not None:
        window._pair_filter = None
        window._on_grid_selected(pair_cell)
        app.processEvents()
        check(
            window._pair_filter is None,
            "activating a two-colour mix no longer filters the grid to its pair",
        )

    # Render the real window once, so a broken Qt platform plugin or a missing
    # font shows up here rather than in front of the user.
    shot = None
    try:
        shot = Path.cwd() / "selftest-ui.png"
        check(window.grab().save(str(shot)), "the user interface rendered to a PNG")
    except Exception as error:  # pragma: no cover - only on a broken bundle
        check(False, f"rendering the user interface failed: {error}")

    window.close()

    lines.append("")
    if shot is not None and shot.is_file():
        lines.append(f"ui          {shot} ({shot.stat().st_size} B)")
    if failures:
        lines.append(f"FAILED ({len(failures)})")
        lines.extend(f"  - {item}" for item in failures)
    else:
        lines.append("SELFTEST OK")

    text = "\n".join(lines) + "\n"
    target = report_path or (Path.cwd() / "selftest-report.txt")
    try:
        target.write_text(text, encoding="utf-8")
    except OSError:  # pragma: no cover - read-only working directory
        target = scratch / "selftest-report.txt"
        target.write_text(text, encoding="utf-8")

    if sys.stdout is not None:  # a windowed build has no console
        try:
            sys.stdout.write(text)
            sys.stdout.write(f"report: {target}\n")
        except Exception:  # pragma: no cover
            pass

    code = 1 if failures else 0
    # The report is written and the window is closed, so nothing is left to
    # clean up — and letting the interpreter tear Qt down afterwards exits with
    # 0xC0000409 (STATUS_STACK_BUFFER_OVERRUN) on roughly half of all runs.
    # That crash happens *after* the report, but a diagnostic that reports
    # "SELFTEST OK" and then fails is worse than no diagnostic at all, so this
    # path leaves through the front door instead.
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:  # pragma: no cover - no streams in a windowed build
        pass
    os._exit(code)

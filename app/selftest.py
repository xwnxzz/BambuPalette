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
        not window._target_color.isSet(),
        "the 目标颜色 picker starts with no preset colour",
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
    # m08102 item 1: the R/G/B spins start at 0, so 确认 must work right away and
    # mean black — it used to sit disabled until a colour was picked.
    check(
        add_dialog.values()["color_hex"] == "#000000",
        f"an untouched dialog means RGB 0,0,0, got {add_dialog.values()['color_hex']!r}",
    )
    check(add_dialog._ok_button.isEnabled(), "确认 is enabled on the default RGB 0,0,0")
    add_dialog._color.setValue("#123456")
    check(add_dialog._ok_button.isEnabled(), "确认 stays enabled once a colour is entered")
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

    # m00519 #2: 「原图 / 预览」用滚轮缩放、按住左键拖动平移、双击还原成整张。
    from PySide6.QtCore import QEvent as _QEvent
    from PySide6.QtCore import QPointF as _QPointF
    from PySide6.QtCore import Qt as _Qt
    from PySide6.QtGui import QMouseEvent as _QMouse

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
    check(
        page._original_view.toolTip() == _pp.ZOOM_HINT and probe.toolTip() == _pp.ZOOM_HINT,
        "both panels explain 滚轮缩放 / 拖动平移",
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

    # 抖动对有透明区域的图没有意义（有颜色的像素会被挤成一行再量化，那一行上没有
    # 邻居关系），所以那种图上直接灰掉并且说明原因，而不是让用户勾了没反应。
    check(page._dither.isEnabled(), "an opaque picture keeps 抖动 selectable")
    check("误差扩散" in page._dither.toolTip(), "抖动 explains itself while it can still act")
    transparent = scratch / "selftest-alpha.png"
    alpha_image = Image.new("RGBA", (32, 32), "#F2F0EB")
    ImageDraw.Draw(alpha_image).rectangle((0, 0, 7, 31), fill=(242, 240, 235, 0))
    alpha_image.save(transparent)
    page.loadImage(transparent)
    app.processEvents()
    check(not page._dither.isEnabled(), "no meaningful dithering on a transparent picture")
    check("透明" in page._dither.toolTip(), "and the checkbox says why it went grey")
    page.loadImage(picture)
    app.processEvents()
    check(page._dither.isEnabled(), "抖动 comes back on an opaque picture")

    # 颜色数量: the 32 ceiling belongs to Bambu Studio, not to us, so the box has
    # to say so — the user asked for a bigger number and got "拓竹还没放开".
    from .mesh.threemf import MAX_PAINTED_FILAMENTS as _MAX_PAINTED

    check(
        page._colour_count.maximum() == _MAX_PAINTED,
        f"颜色数量 stops at Bambu Studio's own ceiling ({_MAX_PAINTED})",
    )
    ceiling_tip = page._colour_count.toolTip()
    check(
        "Bambu Studio" in ceiling_tip and str(_MAX_PAINTED) in ceiling_tip,
        "颜色数量 names Bambu Studio as the reason for the ceiling",
    )
    check("工单" in ceiling_tip, "and says the ceiling is reported to Bambu, not invented by us")

    # 耗材上限 (m00780/m00840 #4): 0 = 不限制, otherwise the plan may spend only
    # that many spools, and colours that do not fit are replaced.
    check(page._spool_cap.value() == 0, "耗材上限 starts at 0 = 不限制")
    check(
        page._spool_cap.maximum() == _MAX_PAINTED,
        "耗材上限 cannot promise more spools than a project can hold",
    )
    check(page._settings().max_spools == 0, "0 carries through to the matcher as 不限制")
    page._spool_cap.setValue(2)
    check(page._settings().max_spools == 2, "the spin box reaches the matcher")
    page._spool_cap.setValue(0)

    capped = match_image(picture, palette, MatchSettings(max_colours=12, max_spools=1))
    spent = sorted({parent for entry in capped.palette for parent in entry.filament_ids})
    check(len(spent) <= 1, f"耗材上限 1 really spends one spool ({spent})")
    check(capped.printed_pixels > 0, "and the picture is still printed, not blanked")

    # 三色配方 (m00840 #1): the candidate pool cannot be built up front, so the
    # checkbox stays grey until the 颜色配方 page has finished the triples.
    from .core.triples import TripleCatalog as _TripleCatalog
    from .ui.picture_page import TriplePalette as _TriplePalette

    check(
        not page._include_triples.isEnabled(),
        "也算三色配方 waits, greyed, until the triples are finished",
    )
    head_pool = page._candidate_palette()
    check(
        not isinstance(page._match_palette(), _TriplePalette),
        "an unticked 也算三色配方 keeps the two-filament pool",
    )
    triples = _TripleCatalog(library.filaments, engine=ENGINE_MIXER).build()
    page.setTriples(triples, library)
    app.processEvents()
    check(page._include_triples.isEnabled(), "也算三色配方 comes alive once the triples are built")
    page._include_triples.setChecked(True)
    pool = page._match_palette()
    check(isinstance(pool, _TriplePalette), "ticking it swaps in the three-filament pool")
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
    page._include_triples.setChecked(False)
    page.setTriples(None, library)
    app.processEvents()
    check(
        not page._include_triples.isEnabled() and not page._include_triples.isChecked(),
        "without triples the box goes back to grey and unticked",
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
            and not page._undo_stack
            and not page._redo_stack,
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
            page._undo()
            app.processEvents()
            check(
                not page._merged and len(page.result.palette) == before_colours,
                "undoing the merge restores both colours",
            )

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
    check(
        "颜色配方" in window._target_color.toolTip(),
        "the target colour tooltip says 颜色配方 too",
    )

    # The version must be readable without opening 关于, and it is the same string
    # build/BambuPalette.spec parses out of app/__init__.py for the exe resource.
    from . import __version__ as _version

    about = window._about_text()
    check(_version in about, f"关于 names the running version {_version!r}")
    check(about.startswith("BambuPalette"), "关于 still opens with the product name")
    check(
        "BambuPalette.exe 的文件属性里也写着同一个版本号。" in about,
        "关于 points at the exe's file properties",
    )
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

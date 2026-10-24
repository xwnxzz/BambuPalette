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

#: 自检里造的一个最小 3MF：两块、两种颜色、各两个三角形，15 mm 高（1.6.55）。
_SELFTEST_3MF = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<model unit="millimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">'
    '<resources><basematerials id="1">'
    '<base name="a" displaycolor="#C8342E"/><base name="b" displaycolor="#2FA84F"/>'
    "</basematerials>"
    '<object id="1" type="model" name="a" pid="1" pindex="0"><mesh><vertices>'
    '<vertex x="-7.5" y="-7.5" z="-7.5"/><vertex x="7.5" y="-7.5" z="-7.5"/>'
    '<vertex x="-7.5" y="7.5" z="-7.5"/><vertex x="7.5" y="7.5" z="7.5"/>'
    "</vertices><triangles>"
    '<triangle v1="0" v2="1" v3="2"/><triangle v1="1" v2="3" v3="2"/>'
    "</triangles></mesh></object>"
    '<object id="2" type="model" name="b" pid="1" pindex="1"><mesh><vertices>'
    '<vertex x="12" y="-7.5" z="-7.5"/><vertex x="27" y="-7.5" z="-7.5"/>'
    '<vertex x="12" y="7.5" z="-7.5"/><vertex x="27" y="7.5" z="7.5"/>'
    "</vertices><triangles>"
    '<triangle v1="0" v2="1" v3="2"/><triangle v1="1" v2="3" v3="2"/>'
    "</triangles></mesh></object>"
    '</resources><build><item objectid="1"/><item objectid="2"/></build></model>'
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
    check(add_dialog._color.isSet(), "添加耗材 opens on a colour (m09456 第 1 条)")
    # m09456 第 1 条 (1.6.28)：用户要的就是反过来 —— 打开就写 0,0,0，直接能确认。
    # （m01647 当年要的是「不要预设颜色」；1.6.28 按用户的新要求改回来。）
    check(
        add_dialog.values()["color_hex"] == "#000000",
        f"添加耗材 opens on 0, 0, 0 (m09456 第 1 条), got {add_dialog.values()['color_hex']!r}",
    )
    check(
        add_dialog._ok_button.isEnabled(),
        "确认 is usable before anything is touched",
    )
    check(
        add_dialog._next_button.isEnabled(),
        "下一个 is usable before anything is touched",
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

    # m09456 第 1 条 (1.6.28): 数字框里直接敲 0 —— 值本来就在最小值 0 上，valueChanged
    # 不响，只有 lineEdit 的 textEdited 接得住。
    zeroes = swatch_module.ColorField(None)
    zeroes._spins[0].lineEdit().textEdited.emit("0")
    check(
        zeroes.isSet() and zeroes.hex() == "#000000",
        f"typing 0 into a channel commits black (m09456 第 1 条), got {zeroes.hex()!r}",
    )
    # 新耗材默认就是 0,0,0（同一条），打开即可确认。
    fresh_dialog = FilamentDialog(None, [], [], window)
    check(
        fresh_dialog._color.isSet() and fresh_dialog.values()["color_hex"] == "#000000",
        f"添加耗材 starts on 0, 0, 0 (m09456 第 1 条), got {fresh_dialog.values()['color_hex']!r}",
    )
    check(fresh_dialog._ok_button.isEnabled(), "a new spool can be confirmed straight away")
    fresh_dialog.close()

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

    # 材质开关 (m05557 item 1)：两种塑料不能混进同一根喷嘴，所以配色只认一个耗
    # 材种类。切一下，每页的颜色池都得跟着换——库里别的材质的卷还列着，只是灰掉
    # 了；切回来它们也该原样回来。
    from .core.settings import load_settings as _material_settings

    material_window = MainWindow(
        library=FilamentLibrary(
            [
                _SelftestFilament(
                    name="A-PETG", brand="大简", material_type="PETG HF", color_hex="#F5D2B8"
                ),
                _SelftestFilament(
                    name="B-PETG", brand="大简", material_type="PETG HF", color_hex="#C52A36"
                ),
                _SelftestFilament(name="C-PLA", brand="b", material_type="PLA", color_hex="#2C2A2C"),
                _SelftestFilament(name="D-PLA", brand="b", material_type="PLA", color_hex="#FF6C2B"),
                # 只有一卷的材料（m10318 item 1）：切过来时两种料的混色必须一个不剩。
                _SelftestFilament(name="E-ABS", brand="b", material_type="ABS", color_hex="#101010"),
            ]
        ),
        auto_triples=False,
    )
    material_window.resize(1100, 760)
    material_window.show()
    app.processEvents()
    material_combo = material_window._material_combo
    labels = [material_combo.itemText(index) for index in range(material_combo.count())]
    check(
        labels == ["PETG HF（2）", "PLA（2）", "ABS（1）"],
        f"材质下拉按种类列出库里的耗材和卷数（{labels}）",
    )
    check(
        material_window.library.material == "PETG HF",
        f"没有记忆时先落在卷数最多的那一种（{material_window.library.material!r}）",
    )
    check(
        [f.name for f in material_window.library.visible] == ["A-PETG", "B-PETG"],
        "默认只有这一种材质的耗材参与配色",
    )

    def _grid_hexes() -> set[str]:
        return {recipe.color_hex.upper() for recipe in material_window._recipes}

    check(
        {"#F5D2B8", "#C52A36"} <= _grid_hexes(),
        "切换前格子里是 PETG 那两卷的颜色",
    )

    material_combo.setCurrentIndex(material_combo.findData("PLA"))
    app.processEvents()
    check(
        material_window.library.material == "PLA",
        f"动一下下拉就换了配色材质（{material_window.library.material!r}）",
    )
    check(
        sorted(f.name for f in material_window.library.visible) == ["C-PLA", "D-PLA"],
        "切换后只剩 PLA 那两卷参与配色",
    )
    check(
        not ({"#F5D2B8", "#C52A36"} & _grid_hexes()),
        "切换后 PETG 的颜色一个都不在格子里",
    )
    check({"#2C2A2C", "#FF6C2B"} <= _grid_hexes(), "PLA 的两卷本色都在格子里")
    listed = [material_window._list.item(index).text() for index in range(material_window._list.count())]
    wanted_rows = sum(1 for filament in material_window.library if material_window.library.accepts(filament))
    check(
        material_window._list.count() == wanted_rows and not any("#F5D2B8" in row for row in listed),
        f"我的耗材只列现在这种材质（m09456 第 2 条）：{material_window._list.count()} 行 / 应有 {wanted_rows} 行",
    )
    spacey = Filament(name="E-空格", brand="b", material_type=" PLA ", color_hex="#FFFFFF")
    check(
        material_window.library.accepts(spacey),
        "耗材种类末尾多一个空格也算同一种材质",
    )
    material_window._list.setCurrentRow(0)
    app.processEvents()
    check(
        not hasattr(material_window, "_spell_out") and "#F5D2B8 · " not in material_window._status.text(),
        f"点耗材不再把它一行写在底栏（m09456 第 3 条）：{material_window._status.text()[:40]!r}",
    )
    check(
        material_window.library.hidden_count == 0,
        "切换材质不是隐藏：没有一卷被标成已隐藏",
    )
    check(
        material_window._status.text().find("现在按 PLA 配色") >= 0,
        f"状态栏说明现在按什么配色（{material_window._status.text()[:60]!r}）",
    )

    material_combo.setCurrentIndex(material_combo.findData("PETG HF"))
    app.processEvents()
    check(
        "#F5D2B8" in _grid_hexes() and "#2C2A2C" not in _grid_hexes(),
        "切回去 PETG 的颜色回来、PLA 的走开",
    )

    # 只有一卷的材质：两色表归零，「全部颜色」只剩这一卷的本色（m10318 item 1）。
    material_combo.setCurrentIndex(material_combo.findData("ABS"))
    app.processEvents()
    pair_count = material_window._catalog.pair_count if material_window._catalog else -1
    recipe_count = material_window._catalog.recipe_count if material_window._catalog else -1
    check(
        pair_count == 0 and recipe_count == 0,
        f"切到只有一卷的材质时没有留下任何配方对（{pair_count} 对 / {recipe_count} 条）",
    )
    combined = len(material_window._combined) if material_window._combined is not None else -1
    check(
        combined == 1 and len(material_window._recipes) == 1,
        f"「全部颜色」只剩这一卷的黑色（{combined} 个颜色 / {len(material_window._recipes)} 格）",
    )
    check(
        "去重后共 1 个颜色" in material_window._status.text(),
        f"底栏的去重颜色数跟着归一（{material_window._status.text()[:70]!r}）",
    )
    material_combo.setCurrentIndex(material_combo.findData("PLA"))
    app.processEvents()
    check(
        sorted(f.name for f in material_window.library.visible) == ["C-PLA", "D-PLA"]
        and material_window._catalog.pair_count == 1
        and material_window._catalog.recipe_count == 81,
        "切回两种料的材质时，它自己的 81 条配方原样回来，不多不少",
    )

    remembered = _material_settings().get("material")
    check(
        remembered == "PLA",
        f"选的材质记住了，下次开窗就用它（{remembered!r}）",
    )

    # 一种材质下的卷被删光：这种材质要从下拉里消失、材质回落到第一种、并且说出来
    # （m10789 item 1）。以前下拉悄悄落回第一种，状态栏却还印着被删掉的那种名字，
    # 设置里也还记着它 —— 用户看到的就是「材质无法切换了」。
    material_combo.setCurrentIndex(material_combo.findData("PLA"))
    app.processEvents()
    check(
        material_window.library.material == "PLA",
        f"下拉还能切到别的材质（{material_window.library.material!r}）",
    )
    for _round in range(2):
        material_window._list.setCurrentRow(0)
        app.processEvents()
        material_window._on_delete()
        app.processEvents()
    check(
        material_window.library.material == "PETG HF",
        f"删光一种材质的耗材后回落到第一种材质（{material_window.library.material!r}）",
    )
    left = [material_combo.itemText(index) for index in range(material_combo.count())]
    check(
        left == ["PETG HF（2）", "ABS（1）"],
        f"删空的材质从下拉里消失（{left}）",
    )
    check(
        material_combo.isEnabled() and material_combo.currentData() == "PETG HF",
        f"材质下拉还能继续用（{material_combo.currentData()!r}）",
    )
    check(
        "「PLA」下的耗材已经全部删除" in material_window._status.text()
        and "已切回「PETG HF」" in material_window._status.text(),
        f"底栏说清为什么换了材质（{material_window._status.text()[:70]!r}）",
    )
    check(
        "PLA" not in material_window._library_hint.text(),
        f"空库提示不再提被删掉的那种（{material_window._library_hint.text()[:50]!r}）",
    )
    check(
        _material_settings().get("material") == "PETG HF",
        "回落的材质写进设置，下次开窗不会再捡回被删掉的那种",
    )
    combined = len(material_window._combined) if material_window._combined is not None else -1
    check(
        combined == 83 and material_window._catalog.pair_count == 1,
        f"留下的那种材质接手了颜色（{combined} 个 / {material_window._catalog.pair_count} 对）",
    )
    material_window.close()
    app.processEvents()

    check(
        window._picture_page is not None
        and window._tabs.count() == 3
        and window._tabs.tabText(2) == "模型配色",
        f"the three pages are present ({window._tabs.count()} tabs)",
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
    # m16941 第 1 条：界面上不再有「最大边长」，导入时就按图片自己的像素数建栅格。
    # m14567 的「原图 → 匹配栅格」那句话仍然要有：内部上限或老工程把栅格压小时，
    # 状态行里得写着缩了多少。
    check(
        page._original_panel.title_label.text() == "原图 160×120",
        f"「原图」面板的标题带原图尺寸，got {page._original_panel.title_label.text()!r}",
    )
    check(
        not hasattr(page, "_max_dimension"),
        "界面上不再有「最大边长」这个数字框（m16941 第 1 条）",
    )
    check(
        page._span == max(page._image.size) and "一格就是一个原像素" in page._status.text(),
        f"导入后栅格 = 图片自己的像素数，状态行也这么说，"
        f"got {page._span} / {page._image.size} / {page._status.text()!r}",
    )
    _span = page._span
    page._span = 60
    page._on_rematch_blocks()
    app.processEvents()
    check(
        "原图 160×120 → 匹配 60×45（缩小 2.7 倍）" in page._status.text(),
        f"栅格被压小时，状态行写清缩了多少，got {page._status.text()!r}",
    )
    page._span = _span
    page._on_rematch_blocks()
    app.processEvents()

    # m00519 #2: 「原图 / 预览」用滚轮缩放、按住左键拖动平移、双击还原成整张。
    from PySide6.QtCore import QEvent as _QEvent
    from PySide6.QtCore import QPointF as _QPointF
    from PySide6.QtCore import Qt as _Qt
    from PySide6.QtGui import QMouseEvent as _QMouse

    from .core.image_matching import plan_help as _plan_help
    from .core.settings import load_settings as _load_settings
    from .ui import picture_page as _pp

    def _mouse(
        kind,
        pos,
        *,
        button=_Qt.MouseButton.LeftButton,
        buttons=_Qt.MouseButtons.NoButton,
    ):
        return _QMouse(kind, pos, pos, button, buttons, _Qt.KeyboardModifier.NoModifier)

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

    # 平移改成中键（1.6.51，m20572 item 1）：左键只挑颜色。中键按下、移动、松手 —— 只平移。
    picked: list[int] = []
    probe.regionClicked.connect(picked.append)
    start = _QPointF(probe.width() / 2.0, probe.height() / 2.0)
    dragged_to = _QPointF(start.x() + 60.0, start.y() + 45.0)
    probe.mousePressEvent(
        _mouse(
            _QEvent.Type.MouseButtonPress,
            start,
            button=_Qt.MouseButton.MiddleButton,
            buttons=_Qt.MouseButtons.MiddleButton,
        )
    )
    check(probe._panning, "middle-button press starts a pan")
    probe.mouseMoveEvent(
        _mouse(_QEvent.Type.MouseMove, dragged_to, buttons=_Qt.MouseButtons.MiddleButton)
    )
    probe.mouseReleaseEvent(
        _mouse(
            _QEvent.Type.MouseButtonRelease,
            dragged_to,
            button=_Qt.MouseButton.MiddleButton,
        )
    )
    app.processEvents()
    check(not probe._panning, "releasing the middle button ends the pan")
    check(not picked, "panning never picks a colour on the way")
    check(probe._pan != _QPointF(0.0, 0.0), "the middle button moves the picture inside the panel")
    middle_pan = _QPointF(probe._pan)
    probe.mousePressEvent(_mouse(_QEvent.Type.MouseButtonPress, start))
    probe.mouseMoveEvent(_mouse(_QEvent.Type.MouseMove, dragged_to, buttons=_Qt.MouseButtons.LeftButton))
    probe.mouseReleaseEvent(_mouse(_QEvent.Type.MouseButtonRelease, dragged_to))
    app.processEvents()
    check(probe._pan == middle_pan, "a left-button drag no longer pans")
    check(len(picked) == 1, "a left-button press-and-drag counts as a plain click now")
    picked.clear()
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
        page._view._selected == -1 and page._list.current() == -1,
        "再点一下已经选中的颜色：什么都不选"
        "（m12426 item 2：以前会退回「上一次那个」，现在不退了）",
    )
    _click(other)
    check(page._view._selected == other, "一个都没选中时点一下还是正常选中它")
    _click(other)
    check(
        page._view._selected == -1 and page._list.current() == -1,
        "再点一下又收起来：取消选中就是什么都不选，没有「上一次」可退",
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
    # m06704 第 3 条 (1.6.27)：默认口径换成「大色差色块优先」——先把差得最明显的色块
    # 认出来，平均差得少退到第二行；「重新匹配」和打开不带方案的工程都落在第一行。
    from .core import image_matching as _matching  # noqa: PLC0415

    check(
        _matching._budget_objective is _matching._objective_priority,
        "没点名的时候用的是「大色差色块优先」这把尺子",
    )
    check(
        _matching._PLAN_SPECS[0][0] == "priority"
        and _matching._PLAN_SPECS[0][1] == page._plans[0].title
        and page._plans[0].key == "priority",
        f"第一行是「{_matching._PLAN_SPECS[0][1]}」（现在是 {page._plans[0].key!r}）",
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
    # m09456 第 4 条 (1.6.28)：上限改小以后，手上那几行可能已经装不下了 —— 用户截图里
    # 出现过「耗材上限 4」而方案行写着「用 9 卷」。当场把超限的行标出来，不必等重新匹配。
    widest = max(len(option.spools) for option in page._plans)
    if widest >= 2:
        page._spool_cap.setValue(widest - 1)
        app.processEvents()
        marked = []
        for index, option in enumerate(page._plans):
            if len(option.spools) <= widest - 1:
                continue
            row = page._plan_rows_layout.itemAt(index).widget()
            numbers = " ".join(child.text() for child in row.findChildren(_QLabel))
            marked.append((numbers, page._plan_group.button(index)))
        check(
            bool(marked)
            and all(
                button is not None and not button.isEnabled() and "超过现在的上限" in numbers
                for numbers, button in marked
            ),
            f"耗材上限改成 {widest - 1} 卷后，用 {widest} 卷的那一行当场被标成超限",
        )
        page._spool_cap.setValue(_DEFAULT_SPOOL_CAP)
        app.processEvents()
    check(not _overspent(), f"没有一套超过耗材上限 {page._settings().max_spools} ({_overspent()})")

    # m11180 第 2 条：填完颜色以后改色块（删除 / 合并 / 更换颜色），方案行里的数字是上一次
    # 匹配算出来的，不会自己跟着走 —— 用户截图里「共 2 种颜色」配「差块 9」就是这么来的
    # （差块按图片的颜色数算，两种颜色最多两个差块）。行不改数字（不替用户重挑卷），但
    # 必须标出来，并把「按当前色块重新匹配」指出来。


    def _row_numbers(index: int) -> str:
        row = page._plan_rows_layout.itemAt(index).widget()
        return " ".join(
            child.text() for child in row.findChildren(_QLabel) if "平均" in child.text()
        )

    stale_colours = len(page._result.palette)
    stale_rows = len(page._plans)
    check(not page._plans_stale, "刚匹配完的方案表是新鲜的")
    if stale_rows and stale_colours > 1:
        page._list.select([0])
        page._on_delete_colours()
        app.processEvents()
        check(len(page._result.palette) < stale_colours, "删掉一种颜色之后调色板少了")
        check(page._plans_stale, "删掉一种颜色之后方案表被标成「改过了」")
        check(
            len(page._plans) == stale_rows and "色块改过了" in _row_numbers(0),
            f"行没被偷偷换掉，但写明了数字是改之前的：{_row_numbers(0)!r}",
        )
        check(
            "按当前色块重新匹配" in page._plan_group.button(0).toolTip(),
            "行上的提示里说清按哪个按钮重算",
        )
        page._on_rematch_blocks()
        app.processEvents()
        check(not page._plans_stale, "按下「按当前色块重新匹配」之后标记清掉了")
        check(
            all(option.defects <= len(page._result.palette) for option in page._plans),
            "重算之后每一行的差块都在图片颜色数之内："
            f"{[(option.key, option.defects) for option in page._plans]}",
        )
        check(
            "色块改过了" not in _row_numbers(0),
            f"重算之后行上不再挂标记：{_row_numbers(0)!r}",
        )
        page._undo()
        app.processEvents()
        check(len(page._plans) == stale_rows, f"撤销把方案表放回去了（{len(page._plans)} 行）")
        # 再退一步，把「删除一种颜色」也退掉，后面的检查还在原来那张图上跑。
        page._undo()
        app.processEvents()
        check(
            len(page._result.palette) == stale_colours and not page._plans_stale,
            "撤销回到改色块之前，调色板和方案表都回到原样",
        )

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
        # 1.6.61（m27752 第 5 条）：详情改成右键打开。三击要每一下都落在大约
        # 0.5 s 内，手一快就变成「选中 / 取消选中」，用户直接说「太难按了」。
        from PySide6.QtCore import Qt as _QtEnum  # noqa: PLC0415

        check(
            page._list._list.contextMenuPolicy()
            == _QtEnum.ContextMenuPolicy.CustomContextMenu,
            "颜色列表把右键留给上下文菜单（详情走右键）",
        )
        page._list._on_row_context_menu(spot)
        app.processEvents()
        check(
            fired == [0],
            f"右键那一行就让列表说出选中的是哪种颜色，got {fired}",
        )
        check(
            page._detail_dialog is not None and page._detail_dialog.isVisible(),
            "右键颜色列表里的一行，弹出这个颜色的详情",
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
        # m12261: 用户按不动 合并，是因为清单默认是单选的 —— 这句提醒挂在按钮上。
        check(
            "多选" in page._merge_button.toolTip(),
            "合并 的提示告诉他先开多选",
        )
        # m12426 item 1：清单一行一种**打印色** —— 同一个配方的几处合成一行（数量对不上
        # 时行里写「等 N 处」），小结里的「共 N 种颜色」也按打印色数。
        keys = [entry.key for entry in page._result.palette]
        check(
            rows.count() == len(set(keys)),
            f"「按颜色选择」一行一种打印色（{rows.count()} 行 / {len(set(keys))} 种）",
        )
        check(
            f"共 {len(set(keys))} 种颜色" in page._list._summary.text(),
            f"小结也按打印色计数：{page._list._summary.text()[-14:]}",
        )
        # m12426 item 2：单选时再点一下已经选中的颜色 = 什么都不选（以前会退回上一次那个）。
        page._list.select([0])
        app.processEvents()
        page._select(page._list.selected_indices()[0])
        app.processEvents()
        first_row = rows.item(0)
        page._list._press_index = int(first_row.data(0x0100))
        page._list._press_shown = page._list._pinned
        page._list._on_row_clicked(first_row)
        app.processEvents()
        check(
            not page._list.selected_indices() and page._shown < 0,
            "再点一下已经选中的颜色：什么都不选"
            f"（选中 {page._list.selected_indices()}、_shown {page._shown}）",
        )
        check(
            not page._delete_button.isEnabled() and not page._merge_button.isEnabled(),
            "一个都没选中时 删除 / 合并 都灰着",
        )

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

        # m16941 第 2 条 / m17910 第 1 条：底板颜色走同一个挑选框，而且按下去真的要能建出来。
        # 早先这里把 `color_hex` 当 RGB 传下去，框在 `int('#')` 上抛 ValueError，界面上就是
        # 「更换…」点不动 —— 所以这里既查参数类型，也真的把框建一个出来。
        check(
            not hasattr(page, "_base_combo") and hasattr(page, "_base_button"),
            "底板颜色不再是只列图上几行的下拉框（m16941 第 2 条）",
        )
        check(page._base_entry is None, "没挑过底板颜色时是「自动」")
        base_menu = build_palette(page.library, page._catalog, include_mixes=True)
        base_rgb = page._base_pick_rgb()
        check(
            len(base_rgb) == 3 and all(isinstance(value, int) for value in base_rgb),
            f"挑选框要的「现在这一种」是 RGB 三元组（m17910 第 1 条），got {base_rgb!r}",
        )
        base_picker = ColourPickerDialog(
            base_menu,
            base_rgb,
            library=page.library,
            current_key=page._shown_base_entry().key,
            title="更换底板颜色",
            target_label="现在的底板颜色",
        )
        check(
            tuple(base_picker._target_rgb) == tuple(base_rgb)
            and base_picker.windowTitle() == "更换底板颜色",
            f"底板颜色的框收得下三元组、标题也写对了，got {base_picker._target_rgb} / "
            f"{base_picker.windowTitle()}",
        )
        base_picker.close()

        real_base_picker = _pp.ColourPickerDialog
        seen_base: list[tuple] = []

        class _RecordingBasePicker:  # noqa: N801
            def __init__(self, entries, target_rgb, **kwargs):
                seen_base.append((tuple(target_rgb), kwargs.get("title")))
                self._entries = list(entries)

            def exec(self):
                return _pp.QDialog.DialogCode.Accepted

            def chosen(self):
                printed = {str(entry.key) for entry in page.result.palette}
                return next(
                    (entry for entry in self._entries if str(entry.key) not in printed),
                    self._entries[0],
                )

        _pp.ColourPickerDialog = _RecordingBasePicker
        try:
            page._base_button.click()
        finally:
            _pp.ColourPickerDialog = real_base_picker
        app.processEvents()
        check(
            len(seen_base) == 1
            and len(seen_base[0][0]) == 3
            and all(isinstance(value, int) for value in seen_base[0][0])
            and seen_base[0][1] == "更换底板颜色",
            f"「底板颜色 → 更换…」按一下就把框建出来了（m17910 第 1 条），got {seen_base}",
        )
        check(
            page._base_entry is not None and "底板颜色换成" in page._status.text(),
            f"挑完底板颜色记在 _base_entry 上并写了状态行，got {page._base_entry}",
        )
        page._base_entry = None
        page._plate = None
        page._refresh_base_view()
        # m18149 第 2 条：按钮上只写它干什么（「更换底板颜色」），「自动」不再塞进按钮文字里。
        check(
            page._base_button.text() == "更换底板颜色"
            and "自动挑的" in page._base_button.toolTip(),
            f"按钮写「更换底板颜色」、自动与否在 tooltip 里，got "
            f"{page._base_button.text()!r} / {page._base_button.toolTip()!r}",
        )
        app.processEvents()

        # m17910 第 2 条：栅格与 SVG 渲染都不再有上限。
        check(
            MatchSettings(max_dimension=9999).clamped().max_dimension == 9999,
            f"匹配栅格不封顶，got {MatchSettings(max_dimension=9999).clamped().max_dimension}",
        )
        check(
            not hasattr(_pp, "SVG_RENDER_LIMIT"),
            "SVG 渲染也不封 4096（m17910 第 2 条）",
        )

        # m19523：1.6.46 加的「宽/长」两个框与「按当前px重新导入图片」撤掉了；
        # 导入仍然按图片自己的像素数分色。
        _picture_size = (int(page._image.width), int(page._image.height))
        check(
            not hasattr(page, "_px_width_spin")
            and not hasattr(page, "_px_height_spin")
            and not hasattr(page, "_rematch_px_button")
            and not hasattr(page, "_px_lock"),
            "「宽/长」两个框与「按当前px重新导入图片」都撤了（m19523）",
        )
        check(
            int(page._span) == max(_picture_size),
            f"导入按图片自己的像素数分色，got span {page._span} / 图 "
            f"{_picture_size[0]}×{_picture_size[1]}",
        )
        check(
            (int(page.result.width), int(page.result.height)) == _picture_size,
            f"栅格就是原图的像素数（一格一个像素），got "
            f"{page.result.width}×{page.result.height}",
        )
        # m18338 第 2 条：耗材方案多给几行，其中一档是「少用一个卷」。
        _cap_now = int(page._settings().max_spools)
        _spare_rows = [row for row in page._plans if "少用一个卷" in row.title]
        check(
            len(page._plans) >= 6,
            f"「耗材方案」的行数比从前多（m18338 第 2 条），got {len(page._plans)} 行",
        )
        if _cap_now > 0:
            check(
                bool(_spare_rows)
                and all(len(row.spools) == _cap_now - 1 for row in _spare_rows),
                f"「少用一个卷」那几行真的比上限 {_cap_now} 少一卷，got "
                f"{[(row.title, len(row.spools)) for row in page._plans]}",
            )
        else:
            check(
                not _spare_rows,
                "「耗材上限 0」不限制时没有「少用一个卷」可说",
            )

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
            len(page._list.selected_indices()) == 1,
            "关掉「多选」之后列表只剩一行 (m02289 item 1): "
            f"{page._list.selected_indices()}",
        )
        page._list.select([0, 1])
        app.processEvents()
        check(
            len(page._list.selected_indices()) == 1,
            "「多选」关着的时候，就算代码想选两行也只留下一行: "
            f"{page._list.selected_indices()}",
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
        # m12426 item 1：一行是一种**打印色**，删一行就把这一行背后的几处一起删掉。
        doomed_members = page._list.members_of(doomed)
        doomed_pixels = sum(
            int((page.result.indices == position).sum()) for position in doomed_members
        )
        page._list.clear_selection()
        page._list.select([doomed])
        page._on_delete_colours()
        app.processEvents()
        check(doomed in page._deleted, "the deleted colour is remembered as an edit")
        check(
            set(doomed_members) <= set(page._deleted),
            f"删掉这一行 = 删掉它的 {len(doomed_members)} 处（m12426 item 1）",
        )
        check(
            len(page.result.palette) == before_colours - len(doomed_members),
            f"deleting one colour leaves {len(page.result.palette)}",
        )
        check(
            page.result.printed_pixels == before_pixels - doomed_pixels,
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
            len(page.result.palette) == before_colours - len(doomed_members),
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
        # m13246 item 1：圆滑色块把小斑点画得比它的像素大，点在画出来的斑点上时
        # 脚下那一块往往是大笔画。3 像素内有比脚下更小的块，就听小块的。
        speck = np.full((5, 5), -1, dtype=np.int32)
        speck[0:5, 0:4] = 0
        speck[2, 2] = 1
        speck_sizes = np.asarray([24, 1], dtype=np.int64)
        page._view.setBlockLabels(speck, speck_sizes)
        check(
            page._view._pick_index(lonely, 3, 2) == 1,
            "脚下是大笔画、1 像素外有小块：听小块的（m13246 item 1）",
        )
        check(
            page._view._pick_index(lonely, 2, 2) == 1,
            "点在斑点本身上也选中小块（m13246 item 1）",
        )
        page._view.setBlockLabels(None)
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

        # m14748 第 2 条：颜色表列出来的颜色，色块表里也要有一个位子；而且去噪
        # 只许并掉位置，不许把小碎片的颜色盖到大色区上（以前并查集取「编号小的
        # 根」的颜色，碎片排在光栅序前面就会把整块大色区改色）。
        from .core.blocks import min_block_pixels as _min_block_pixels  # noqa: PLC0415

        threshold = _min_block_pixels(page.result.printed_pixels)
        printed_values = {
            int(value) for value in np.unique(page.result.indices).tolist() if int(value) >= 0
        }
        check(
            printed_values <= {int(value) for value in blocks.values.tolist()},
            "色块表覆盖颜色表列出的每一种颜色（m14748 第 2 条）："
            f"印了 {sorted(printed_values)}，色块表只有 {sorted({int(v) for v in blocks.values.tolist()})}",
        )
        repainted: list[str] = []
        for label in blocks.order():
            mine = blocks.mask(label)
            if not mine.any():
                continue
            inside, counts = np.unique(page.result.indices[mine].tolist(), return_counts=True)
            own = blocks.value(label)
            for value, count in zip(inside.tolist(), counts.tolist()):
                if int(value) != own and int(count) >= threshold:
                    repainted.append(f"{count} px #{int(value)} → #{own}")
        check(
            not repainted,
            "去噪没有把小碎片的颜色盖到大色区上（m14748 第 2 条）：" + "、".join(repainted[:3]),
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
            # 一次动两块：只有「多选」开着才做得到 (m02289 item 1)。
            page._multi_button.setChecked(True)
            page._list.select(pair)
            app.processEvents()
            keep_entry = page.result.palette[blocks.value(pair[1])]
            real_merge = _pp.MergeDialog
            _pp.MergeDialog = lambda *args, **kwargs: _StubDialog(1, None)
            try:
                page._on_merge_colours()
                app.processEvents()
            finally:
                _pp.MergeDialog = real_merge
            keys = {
                page.result.palette[value].key
                for item in pair
                for value in page.result.indices[blocks.mask(item)].tolist()
            }
            check(
                keys == {keep_entry.key},
                f"合并把选中的块都变成挑中的那种颜色 {sorted(keys)} != {keep_entry.key}",
            )
            check(len(page._block_edits) == 2, "合并给每一块各记了一条色块级改动")
            page._undo()
            app.processEvents()
            check(not page._block_edits, "Ctrl+Z 撤销合并，色块级改动回到空")

        # 更换颜色：选中的块一起换。
        if merged:
            page._multi_button.setChecked(True)
            page._list.select(pair)
            app.processEvents()
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
                hexes = {
                    page.result.palette[value].color_hex
                    for item in pair
                    for value in page.result.indices[blocks.mask(item)].tolist()
                }
                check(
                    hexes == {wanted.color_hex},
                    f"更换颜色把选中的块一起换成新颜色 {sorted(hexes)} != {wanted.color_hex}",
                )
                check(
                    wanted.key in [entry.key for entry in page.result.palette],
                    "新颜色进了色表",
                )
                page._undo()
                app.processEvents()
                check(not page._block_edits, "Ctrl+Z 撤销更换颜色")
            page._multi_button.setChecked(False)
            app.processEvents()

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
            page._list.row_count() == len({entry.key for entry in page.result.palette}),
            "回到颜色模式：列表又是一行一种打印色（m12426 item 1），"
            f"got {page._list.row_count()} 行 / "
            f"{len({entry.key for entry in page.result.palette})} 种",
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
            # 焦点这件事在无头/被别的窗口抢走时会拿不到：QApplication.focusWidget()
            # 看的是「活动窗口」的焦点部件，窗口没激活时它返回 None。所以这里先激活
            # 窗口再要焦点，拿不到就重试几次，断言的仍然是「焦点在框里时页面不抢 Del」。
            window.activateWindow()
            window.raise_()
            app.processEvents()
            for _ in range(6):
                page._colour_count.setFocus()
                app.processEvents()
                if page._typing_in_field():
                    break
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
        # 1.6.26 (m05557 item 2)：「打开工程 → 再保存」这一条路以前是断的。工程文件里
        # 每一步色块改动是一个字典 ``{"mask": …, "entry": …}``，1.6.25 的
        # ``_adopt_project`` 却把它当成一对解包，于是 ``entry`` 成了字符串 "entry"、
        # ``mask`` 成了 "mask"，一路活到保存时才炸成「'str' object has no attribute
        # 'key'」——用户看到的正是「打开好好的，一保存就失败」。这里整条路走一遍。
        from .core.project import load_project as _load_project  # noqa: PLC0415

        reopened = scratch / "selftest-reopen.bpproj"
        page._set_pick_mode(_pp.PICK_EDGE, save=False)
        app.processEvents()
        # 这一遍会把「删掉一块」和「打开工程」都做一次，做完必须把页面摆回原样，
        # 否则后面那些检查看到的就不是它们上一次留下的状态了。
        before_round_trip = page._snapshot()
        path_before = page._project_path
        dirty_before = page._dirty
        blocks = page._blocks
        if blocks is not None and blocks.count:
            page._list.clear_selection()
            page._on_region_clicked(blocks.order()[0])
            app.processEvents()
            page._delete_button.click()
            app.processEvents()
            # 删除存的是 entry=None，抓不到「entry 成了字符串」这种坏法；再换一块
            # 颜色，让文件里真的带着一个颜色对象走一圈。删除会把块重排，块对象和
            # 编号都换新的，所以要再读一次，否则下面拿到的是旧编号。
            blocks = page._blocks
            target = blocks.order()[1] if blocks.count > 1 else blocks.order()[0]
            current = page.result.palette[blocks.value(target)].key
            wanted = next(
                (
                    entry
                    for entry in page._candidate_palette()
                    if entry.key != current
                ),
                None,
            )
            if wanted is not None:
                real_picker = _pp.ColourPickerDialog
                _pp.ColourPickerDialog = lambda *args, **kwargs: _StubDialog(0, wanted)
                try:
                    page._list.clear_selection()
                    page._on_region_clicked(target)
                    app.processEvents()
                    page._on_replace()
                    app.processEvents()
                finally:
                    _pp.ColourPickerDialog = real_picker
        check(bool(page._block_edits), "先造一处色块改动，好跟着走一遍打开→保存")
        page._project_path = reopened
        page._on_save_project()
        check(reopened.is_file() and not page._dirty, "带色块改动的工程能存下来")
        if reopened.is_file():
            state = _load_project(reopened, library=page._library)
            page._adopt_project(state, reopened)
            app.processEvents()
            entries = [edit.entry for edit in page._block_edits]
            check(
                len(entries) == len(state.block_paints),
                f"重开工程把 {len(state.block_paints)} 处色块改动都摆回来"
                f"（现在是 {len(entries)} 处）",
            )
            check(
                all(entry is None or hasattr(entry, "key") for entry in entries),
                "重开后色块改动里的「颜色」还是颜色对象，不是字符串 "
                f"（类型：{[type(entry).__name__ for entry in entries]}）",
            )
            check(
                any(entry is not None for entry in entries),
                "这一遍真的带着一个颜色对象走了一圈（不只是删除留下的 None）",
            )
            check(
                all(
                    _np.asarray(edit.mask).shape == state.base.indices.shape
                    for edit in page._block_edits
                ),
                "重开后每处色块改动的掩码和图片一样大（不是一格 True）",
            )
            page._on_save_project()
            check(not page._dirty, "重开工程再保存一次不会失败")
            # m10318 第 2 条：文件里那一套方案的数字是存档那一刻的，界面上要写明它
            # 不是照现在的规则重算的，否则升级了看着像「匹配没改」。
            from PySide6.QtWidgets import QLabel as _RowLabel  # noqa: PLC0415

            marked = [
                child.text()
                for index in range(page._plan_rows_layout.count())
                for row in (page._plan_rows_layout.itemAt(index).widget(),)
                for child in row.findChildren(_RowLabel)
                if "从工程里读的那套" in child.text()
            ]
            check(
                bool(page._plans) and page._plans[0].from_file,
                "打开工程时文件里那一套被标成「从工程里读的」",
            )
            check(
                len(marked) == 1,
                f"方案表里正好一行写着「从工程里读的那套」（{len(marked)} 行）",
            )
            check(
                all(not option.from_file for option in page._plans[1:]),
                "同一批色块现算出来的那几行没有这个标记",
            )
        page._project_path = path_before
        page._restore(before_round_trip)
        page._set_pick_mode(page._mode, save=False)
        page._apply_edits()
        page._refresh_plans()
        page._dirty = dirty_before
        app.processEvents()

        # 保存失败时日志要落在数据目录里：只弹一个框的话，用户报「保存失败」时谁也
        # 不知道栈在哪（1.6.26）。这里真的 raise 一次再转手给日志，走的就是
        # ``except Exception as error`` 那条路。
        try:
            raise RuntimeError("自检：故意失败，检查日志")
        except RuntimeError as planned:
            log_path = page._log_save_failure(planned)
        check(
            log_path is not None and log_path.is_file(),
            f"保存失败会写日志（{log_path}）",
        )
        if log_path is not None and log_path.is_file():
            logged = log_path.read_text(encoding="utf-8")
            check(
                "自检：故意失败，检查日志" in logged and "Traceback" in logged,
                "日志里既有出错的话，也有调用栈",
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
            page._multi_button.setChecked(True)
            page._list.select(picked)
            app.processEvents()
            keeper = int(picked[1])
            # m12426 item 1：一行可能顶着好几个位子，合并的时候它们一起走。
            expanded = page._list.expanded(picked)
            folded = [
                index
                for index in expanded
                if index not in page._list.members_of(keeper)
            ]
            indices = _np.asarray(page.result.indices)
            touched = _np.isin(indices, _np.asarray(expanded, dtype=indices.dtype))
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
                len(page.result.palette) == before_colours - len(folded),
                f"合并在预览里少掉 {len(folded)} 种颜色"
                f"（{before_colours} → {len(page.result.palette)}，m12426 item 1）",
            )
            page._undo()
            app.processEvents()
            check(
                page._image.tobytes() == before_merge_image,
                "Ctrl+Z 把合并改过的原图也退回去",
            )
            page._multi_button.setChecked(False)
            app.processEvents()

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
        # m13246 item 2：「颜色数量」是用户给的硬上限，「按当前色块重新匹配」也得守。
        # 收口只换「这些像素打印成什么卷」：打印的像素一个不少，色块的像素也都在，
        # 变的是顶着同一种打印色的几个位子被并成了一种。
        live_keys = {entry.key for entry in page._base.palette}
        if len(live_keys) > 3:
            count_box = page._colour_count
            was_count = count_box.value()
            # 拿「用户看到的那张结果」的打印像素数做基准：`_base` 是没算删改的原始匹配，
            # 而收口以后新的 `_base` 已经把删掉的那一块算进去了，两边不可比。
            painted_before = int((page._result.indices >= 0).sum())
            count_box.setValue(3)
            page._on_rematch_blocks()
            app.processEvents()
            capped = {entry.key for entry in page._base.palette}
            check(
                len(capped) <= 3,
                f"颜色数量收到 3 以后再重匹配，打印色不超过 3 种，got {len(capped)}",
            )
            # 收口会把顶着同一种打印色的几个位子并成一个（palette 条数因此会少），
            # 真正不能变的是「打印出来的像素一个不少」（m13246 item 2）。
            after_painted = int((page._result.indices >= 0).sum())
            check(
                after_painted == painted_before,
                f"收口以后打印的像素一个不少（{painted_before} → {after_painted} 个）",
            )
            count_box.setValue(was_count)
            page._undo()
            app.processEvents()
        else:
            check(
                True,
                f"样张的打印色本来就在 3 种以内（{len(live_keys)} 种），跳过收口检查",
            )
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
            0 < len(page._base.palette) <= page._settings().max_colours,
            f"颜色重找之后行数不超过「颜色数量」上限（{len(page._base.palette)} 种 / "
            f"上限 {page._settings().max_colours}，按下之前 {keep_count} 种）",
        )
        # m11985: 色块是用户自己的决定，一次按下不许把两块悄悄并成一块 —— 再按一次，
        # 颜色和行数都必须原样。用户的猪第二次按下把眼睛印成皮肤色，就是「合成一块 +
        # 目标色取并集均值」造成的；这条守着它不再回来。
        after_first = [entry.key for entry in page._base.palette]
        page._on_rematch_blocks()
        app.processEvents()
        check(
            [entry.key for entry in page._base.palette] == after_first,
            f"再按一次手里的色块不变（{len(after_first)} 块原样）",
        )
        # 多按的这一下只为验证上面那条，退回去，后面几步的撤销节奏（按一次 → 撤销 →
        # 重做）才和原来一样。
        page._undo()
        app.processEvents()
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
        # 1.6.27 (m06704 第 2 条)：圆滑不再是把细栅格重新分类一遍（那样台阶只是变小），
        # 而是直接把那条边界线描出来、顶点落在格线之间，所以斜边是直的。
        check(
            any("描出边界线" in note for note in smooth_plate.notes),
            f"状态里写的是描出边界线（{smooth_plate.notes}）",
        )
        _offsets = [
            max(
                abs(coordinate - round(coordinate))
                for row in part.vertices[:, :2] / smooth_plate.pixel_size_mm
                for coordinate in row
            )
            for part in smooth_plate.parts[1:]
        ]
        check(
            bool(_offsets) and max(_offsets) > 1e-6,
            f"颜色件的顶点落在格线之间（最大偏离 {max(_offsets, default=0.0):.4f} 格）",
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

        # 1.6.26 (m05557 item 3)：这个勾现在也管预览。从前它只进 PlateSettings，
        # 在界面上勾来勾去看到的还是原始格子，唯一能看出差别的地方是 Bambu Studio。
        page._smooth_box.setChecked(False)
        app.processEvents()
        blocky_matched = page._view._matched
        check(
            page._view._smooth is False
            and blocky_matched is not None
            and blocky_matched.width() == page.result.width,
            f"取消勾选后预览回到原始格子（{None if blocky_matched is None else blocky_matched.width()} px）",
        )
        page._smooth_box.setChecked(True)
        app.processEvents()
        round_matched = page._view._matched
        check(
            page._view._smooth is True
            and round_matched is not None
            and blocky_matched is not None
            and round_matched.width() > blocky_matched.width(),
            f"勾上之后预览画的是导出用的细格（{None if round_matched is None else round_matched.width()} px）",
        )
        page._smooth_box.setChecked(False)
        app.processEvents()
        check(
            page._view._matched is blocky_matched,
            "来回勾不会重算细格（第二遍拿的是缓存里那张）",
        )
        page._smooth_box.setChecked(True)
        app.processEvents()
        check(
            page._view._matched is round_matched,
            "勾回来也是缓存那张，图案两次一模一样",
        )

        # 1.6.23 (m02289 item 2)：图片印在哪一面。默认底面：色块层贴着打印板，顶面是
        # 一整片底板色。换一面就是把整块实体在 Z 上照一次镜子，绕序再翻回来，所以
        # 仍旧是一块闭合实体、体积不变。
        from .mesh.plate import signed_volume as _signed_volume

        check(
            page._side_combo.currentText() == _pp.IMAGE_SIDE_BOTTOM,
            f"「{_pp.IMAGE_SIDE_LABEL}」默认是{_pp.IMAGE_SIDE_BOTTOM}"
            f"（现在是 {page._side_combo.currentText()!r}）",
        )
        check(
            page._side_combo.count() == 2
            and page._side_combo.itemText(0) == _pp.IMAGE_SIDE_BOTTOM
            and page._side_combo.itemText(1) == _pp.IMAGE_SIDE_TOP,
            "只有底面和顶面两个选项",
        )
        check(page._plate_settings().image_on_bottom, "默认的底面进了 PlateSettings")
        face_down = page.buildPlate()
        check(
            abs(face_down.parts[0].z_bottom_mm - face_down.colour_thickness_mm) < 1e-9
            and all(abs(part.z_bottom_mm) < 1e-9 for part in face_down.parts[1:]),
            f"底面：色块层在下（{face_down.parts[0].z_bottom_mm:.2f} mm 起），底板在上",
        )
        page._side_combo.setCurrentIndex(1)
        app.processEvents()
        check(not page._plate_settings().image_on_bottom, "选顶面之后 PlateSettings 里就是 False")
        face_up = page.buildPlate()
        check(
            abs(face_up.parts[0].z_bottom_mm) < 1e-9
            and all(
                abs(part.z_bottom_mm - (face_up.base_thickness_mm - face_up.colour_thickness_mm))
                < 1e-9
                for part in face_up.parts[1:]
            ),
            f"顶面：底板在下，色块层压在上面（{face_up.parts[1].z_bottom_mm:.2f} mm 起）",
        )
        check(
            abs(face_up.width_mm - face_down.width_mm) < 1e-9
            and abs(face_up.depth_mm - face_down.depth_mm) < 1e-9
            and abs(face_up.volume_mm3 - face_down.volume_mm3) / face_down.volume_mm3 < 1e-9,
            f"换一面不改宽、深和体积（{face_down.volume_mm3:.1f} vs {face_up.volume_mm3:.1f} mm³）",
        )
        _thickness = face_up.total_thickness_mm
        check(
            len(face_down.parts) == len(face_up.parts)
            and all(
                abs(float(a.vertices[:, 2].min()) - (_thickness - float(b.vertices[:, 2].max())))
                < 1e-9
                and abs(
                    float(a.vertices[:, 2].max()) - (_thickness - float(b.vertices[:, 2].min()))
                )
                < 1e-9
                for a, b in zip(face_down.parts, face_up.parts)
            ),
            "两面的每个部件逐一对得上（底面就是顶面照 Z 的镜子）",
        )
        _down_points, _down_faces, _down_tags = _plate_mesh(face_down)
        _down_open, _down_manifold = _edge_health(_down_points, _down_faces)
        check(
            _down_open == 0
            and _down_manifold == 0
            and _signed_volume(_down_points, _down_faces) > 0,
            f"底面的整块照样闭合、法线朝外（开口 {_down_open}、非流形 {_down_manifold}）",
        )
        page._side_combo.setCurrentIndex(0)
        app.processEvents()
        check(
            page._side_combo.currentText() == _pp.IMAGE_SIDE_BOTTOM
            and page._plate_settings().image_on_bottom,
            "换回底面之后一切都回去了",
        )
        plate = page.buildPlate()

        # 1.6.42（m15721 第 3 条）：一格的尺寸固定 0.4 mm（就是 0.4 喷嘴的宽度），板的长宽
        # 由图片的像素数推出来，不再有「打印宽度」和「颜色层厚度」两个框；「底板厚度」是
        # 整块的厚度，里面含图上那 0.4 mm 颜色。
        from .mesh.plate import CELL_MM as _cell_mm

        check(
            not hasattr(page, "_width_spin") and not hasattr(page, "_colour_spin"),
            "「打印宽度」和「颜色层厚度」两个框都撤了（m15721 第 3 条）",
        )
        check(
            abs(page._plate_settings().cell_mm - _cell_mm) < 1e-12
            and abs(plate.width_mm - page.result.width * _cell_mm) < 1e-9
            and abs(plate.depth_mm - page.result.height * _cell_mm) < 1e-9,
            f"一格 {_cell_mm} mm，板长宽 = 图片像素数 × {_cell_mm}"
            f"（量到 {page._plate_settings().cell_mm:.3f} mm、"
            f"{plate.width_mm:.1f} × {plate.depth_mm:.1f} mm）",
        )
        check(
            abs(plate.total_thickness_mm - page._base_spin.value()) < 1e-9
            and abs(page._plate_settings().base_thickness_mm - page._base_spin.value()) < 1e-9,
            f"「底板厚度」就是整块的厚度（{plate.total_thickness_mm:.2f} mm）",
        )
        check(
            all("颜色层" not in note for note in plate.notes),
            f"厚度提醒说的是底板色那一段，不再提颜色层（{plate.notes}）",
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

        # m12426 item 3：导出修改后的原图 —— 存下来的就是「原图」面板画的那一张，
        # 删除留下的透明也一起存（PNG）。
        saved_dialog = _pp.QFileDialog
        _pp.QFileDialog = type(
            "StubSaveDialog",
            (),
            {
                "getSaveFileName": staticmethod(
                    lambda *args, **kwargs: (str(scratch / "edited-original.png"), "")
                )
            },
        )
        try:
            page._on_export_original()
        finally:
            _pp.QFileDialog = saved_dialog
        app.processEvents()
        exported_original = scratch / "edited-original.png"
        check(
            exported_original.is_file() and exported_original.stat().st_size > 0,
            f"导出原图 written ({exported_original.stat().st_size if exported_original.exists() else 0} B)",
        )
        from PIL import Image as _ExportImage  # noqa: PLC0415

        check(
            _ExportImage.open(exported_original).size == page._image.size,
            f"导出的原图尺寸和图一致 {_ExportImage.open(exported_original).size} == {page._image.size}",
        )
        check(
            "导出原图" in page._status.text(),
            f"状态栏说明导出了原图：{page._status.text()[:28]}",
        )

        # m13907 item 2：矢量图（SVG）先渲染成位图再走全流程；1.6.49（m20069）起渲成
        # 多少像素由导入弹窗里的用户定，`_load_vector(..., size=…)` 明着给尺寸。
        vector_target = scratch / "vector.svg"
        vector_target.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="60" height="30" viewBox="0 0 60 30">'
            '<rect width="60" height="30" fill="#E8C7BD"/>'
            "</svg>",
            encoding="utf-8",
        )
        rendered_vector = _pp._load_vector(vector_target, 120)
        check(
            rendered_vector.size == (120, 60),
            f"SVG 按最大边长渲染成位图：{rendered_vector.size}",
        )
        check(
            rendered_vector.convert("RGB").getpixel((60, 30)) == (232, 199, 189),
            f"SVG 的颜色按 Qt 的渲染来：{rendered_vector.convert('RGB').getpixel((60, 30))}",
        )
        check(
            _pp._vector_declared_size(vector_target) == (60, 30),
            f"弹窗的默认值读的是 SVG 自己声明的尺寸：{_pp._vector_declared_size(vector_target)}",
        )
        check(
            _pp._load_vector(vector_target, 1, size=(90, 45)).size == (90, 45),
            f"弹窗写了 90×45 就渲成 90×45（宽长各自独立）："
            f"{_pp._load_vector(vector_target, 1, size=(90, 45)).size}",
        )

        # 弹窗自己：默认值、范围、锁定长宽比、实时写出来的板尺寸。
        size_dialog = _pp.SvgSizeDialog((60, 30), name="vector.svg")
        check(
            size_dialog.chosen() == (60, 30),
            f"弹窗默认填声明尺寸，直接回车就是 1.6.48 的行为：{size_dialog.chosen()}",
        )
        check(
            (size_dialog._width.minimum(), size_dialog._width.maximum()) == (16, 4096)
            and (size_dialog._height.minimum(), size_dialog._height.maximum()) == (16, 4096),
            f"宽长两个框都能写 16～4096："
            f"{size_dialog._width.minimum()}～{size_dialog._width.maximum()} / "
            f"{size_dialog._height.minimum()}～{size_dialog._height.maximum()}",
        )
        check(size_dialog._lock.isChecked(), "「锁定长宽比」默认勾着")
        size_dialog._width.setValue(120)
        check(
            size_dialog.chosen() == (120, 60),
            f"锁着的时候改宽，长按原比例跟着走：{size_dialog.chosen()}",
        )
        size_dialog._height.setValue(90)
        check(
            size_dialog.chosen() == (180, 90),
            f"锁着的时候改长，宽按原比例跟着走：{size_dialog.chosen()}",
        )
        size_dialog._lock.setChecked(False)
        size_dialog._width.setValue(200)
        check(
            size_dialog.chosen() == (200, 90),
            f"取消锁定之后两边各自独立（能拉成不等比）：{size_dialog.chosen()}",
        )
        check(
            "80.0 × 36.0 mm" in size_dialog._note.text(),
            f"框里实时写着这么渲出来是多大的板（一格 0.4 mm）：{size_dialog._note.text()}",
        )
        size_dialog.deleteLater()

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

        # 1.6.25（m04136 第 2 条）：导出的 3MF 自带混色配方。
        # Bambu Studio 从 `filament_is_mixed` 加 `filament_mixed_components` /
        # `_sublayer_ratios` 读混色耗材，所以导出的底板得把每个成分耗材也列成槽
        # ——包括只被配方用到的那些。
        import json as _json

        from .mesh.threemf import (
            FilamentComponent as _Component,
            FilamentSlot as _Slot,
            _project_settings as _settings_from_slots,
            project_filaments as _project_slots,
        )

        _slots = slots_from_palette(page.result.palette, page.library)
        _project = _project_slots(_slots)
        with zipfile.ZipFile(three_mf) as _archive:
            _written = _json.loads(
                _archive.read("Metadata/project_settings.config").decode("utf-8")
            )
        check(
            _written["filament_colour"] == [slot.color_hex for slot in _project.slots],
            f"3MF 列出配方需要的全部 {_project.count} 个耗材槽",
        )
        check(
            _written["filament_is_mixed"]
            == ["1" if index in _project.mixed else "0" for index in range(_project.count)],
            f"逐槽写明是不是混色（这一页 {_project.mixed_count} 个混色）",
        )
        check(
            all(
                _written["filament_mixed_components"][index]
                == ",".join(str(parent) for parent in parents)
                and _written["filament_mixed_sublayer_ratios"][index]
                == ",".join(f"{ratio:.4f}" for ratio in ratios)
                for index, (parents, ratios) in _project.mixed.items()
            ),
            "每个混色槽都写明成分下标与四位小数的比例",
        )
        _planted = (
            _Slot(color_hex="#F5D2B8", material_type="PETG HF", vendor="大简", name="肤色"),
            _Slot(
                color_hex="#EA818A",
                material_type="PETG HF",
                vendor="大简",
                name="红色 38% + 肤色 62%",
                components=(
                    _Component("#C52A36", material_type="PETG HF", vendor="大简", percent=38.0),
                    _Component("#F5D2B8", material_type="PETG HF", vendor="大简", percent=62.0),
                ),
            ),
        )
        _planted_project = _project_slots(_planted)
        _planted_settings = _json.loads(
            _settings_from_slots(_planted_project.slots, _planted_project.mixed)
        )
        check(
            _planted_project.count == 3
            and _planted_project.added == 1
            and _planted_project.mixed[2][0] == (2, 1)
            and _planted_project.order == {1: 1, 2: 3},
            f"混色引用的料没被涂时补成槽（{_planted_project.count} 槽，补 {_planted_project.added}），混色自己排最后",
        )
        check(
            _planted_settings["filament_colour"] == ["#F5D2B8", "#C52A36", "#EA818A"]
            and _planted_settings["filament_is_mixed"] == ["0", "0", "1"]
            and _planted_settings["filament_mixed_components"] == ["", "", "2,1"]
            and _planted_settings["filament_mixed_sublayer_ratios"]
            == ["", "", "0.3800,0.6200"],
            "末位混色指向 (2,1)、比例 0.3800/0.6200，与 Bambu Studio 写的一致",
        )
        check(
            _written["filament_is_mixed"] == sorted(_written["filament_is_mixed"])
            and all(
                index + 1 > max(parents)
                for index, (parents, _ratios) in _project.mixed.items()
            ),
            "混色槽一律排在最后、成分都在它前面（Bambu Studio 2.8.2.61 只认这个顺序）",
        )
        check(
            _planted_settings["enable_mixed_color_sublayer"] == "1",
            "混色子层打开：1.4 mm 只有 7 层，逐层交替会出条纹",
        )
        _plain_settings = _json.loads(_settings_from_slots(_planted[:1], {}))
        check(
            "enable_mixed_color_sublayer" not in _plain_settings
            and _plain_settings["filament_is_mixed"] == ["0"],
            "没有混色时就不写这个键",
        )

        # 1.6.25（m04136 第 1 条）：颜色缩减换成「唯一色上的加权 k-means」，直接
        # 按预算产色。以前是超采 6 倍再折叠，折叠会把渐变上等距的簇吃掉——要 32
        # 色只给 20 色；现在要多少给多少，而且更快。
        import numpy as _np  # noqa: PLC0415
        from PIL import Image as _ReduceImage  # noqa: PLC0415

        from .core.image_matching import (
            _KMEANS_ITERATIONS as _kmeans_rounds,
            _median_cut_seeds as _cut_seeds,
            _merge_similar_colours as _merge_colours,
            _pack as _pack_colours,
            reduce_colours as _reduce,
        )

        _gradient = _np.zeros((96, 128, 3), dtype=_np.uint8)
        _gradient[..., 0] = _np.linspace(0, 255, 128, dtype=_np.uint8).reshape(1, 128)
        _gradient[..., 1] = _np.linspace(0, 255, 96, dtype=_np.uint8).reshape(96, 1)
        _gradient[..., 2] = _np.linspace(255, 0, 96, dtype=_np.uint8).reshape(96, 1)
        _ramp_image = _ReduceImage.fromarray(_gradient, "RGB").convert("RGBA")
        _delivered = []
        for _budget in (8, 16, 32):
            _colours, _classes = _reduce(_ramp_image, MatchSettings(max_colours=_budget))
            _delivered.append(len(_colours))
        check(
            _delivered == [8, 16, 32] and int(_classes.max()) + 1 == 32,
            f"渐变图要多少色就产多少色（8/16/32 → {_delivered}，全用上了）",
        )
        _flat = _np.zeros((16, 16, 3), dtype=_np.uint8)
        _flat[:, :8] = (200, 40, 60)
        _flat[:, 8:] = (30, 90, 200)
        _few, _ = _reduce(
            _ReduceImage.fromarray(_flat, "RGB").convert("RGBA"), MatchSettings(max_colours=48)
        )
        check(
            sorted(tuple(int(value) for value in row) for row in _few)
            == [(30, 90, 200), (200, 40, 60)],
            "本来只有两种颜色的图，原样给回这两种颜色",
        )
        _keys, _histogram = _np.unique(_pack_colours(_gradient.reshape(-1, 3)), return_counts=True)
        check(
            len(_cut_seeds(_keys, _histogram, 16)) == 16 and _kmeans_rounds >= 1,
            f"种子由中位切分给出、数量与预算一致；聚类最多 {_kmeans_rounds} 轮（标签不再变就提前收）",
        )
        _greys = _np.array([(120 + 3 * index,) * 3 for index in range(6)], dtype=_np.uint8)
        _folded, _ = _merge_colours(
            _greys, _np.tile(_np.arange(6, dtype=_np.int64), (4, 6)), 2.0
        )
        check(
            len(_folded) == 3,
            f"六个相差 1.1 ΔE00 的灰阶不会连成一片（并查集会合成 1 色，领队法留下 {len(_folded)} 色）",
        )

        # 1.6.24 (m03703)：三条外部评审。
        # ① 最近颜色搜索：短名单是近似，池子小的时候直接逐行量到底（精确通道）。
        from .core.image_matching import (
            _EXACT_PAIRS as _exact_pairs,
            _SHORTLIST as _shortlist,
            _nearest_entries as _nearest,
            _palette_table as _palette_rows,
        )
        from .spectral import color as _match_colour

        check(
            1024 <= _shortlist and _exact_pairs > 0,
            f"短名单放宽到 {_shortlist} 行，另留一条全表精确通道（{_exact_pairs} 对以内走它）",
        )
        live = page.result.palette
        rows = _palette_rows(live)
        targets = rows[:: max(1, len(rows) // 7)][:7]
        picked = [int(index) for index in _nearest(targets, live, rows)]
        brute = []
        for lab in targets:
            scored = [_match_colour.delta_e_2000(lab, entry.lab) for entry in live]
            brute.append(int(min(range(len(live)), key=lambda index: scored[index])))
        check(
            picked == brute,
            f"这个池子只有 {len(live)} 个颜色，走的是精确通道：逐个量到底，和暴力 argmin 完全一致",
        )

        # ② 全透明图片：以前会被当成整张黑图匹配，现在直接回绝并说清原因。
        from PIL import Image as _SelftestImage

        try:
            match_image(
                _SelftestImage.new("RGBA", (24, 24), (0, 0, 0, 0)),
                live,
                MatchSettings(max_colours=8),
            )
        except ValueError as error:
            check("透明" in str(error), f"全透明的图片被回绝了，理由说得清（{error}）")
        else:
            check(False, "全透明图片该被回绝，而不是匹配出一整片黑")
        faint = match_image(
            _SelftestImage.new("RGBA", (8, 8), (200, 30, 30, 1)),
            live,
            MatchSettings(max_colours=4),
        )
        check(
            len(faint.palette) >= 1 and faint.printed_pixels > 0,
            f"不透明度全在阈值以下、但 RGB 还是画本身的图照旧匹配（{len(faint.palette)} 色）",
        )

        # ③ 不到三卷料时三色目录自己也得站得住（主界面会拦，目录类不能炸）。
        from .core.library import Filament as _SelftestFilament
        from .core.triples import TripleCatalog as _SelftestTriples

        couple = [
            _SelftestFilament(name="自检卷 1", color_hex="#E8E4DE", material_type="PLA"),
            _SelftestFilament(name="自检卷 2", color_hex="#2A2A2E", material_type="PLA"),
        ]
        empty_triples = _SelftestTriples(couple, engine=ENGINE_MIXER).build()
        check(
            empty_triples.built
            and empty_triples.triple_count == 0
            and empty_triples.recipe_count == 0,
            "两卷料：三色目录照样建得出来，只是空的",
        )
        check(
            empty_triples.packed.shape == (0,)
            and list(empty_triples.offsets) == [0]
            and empty_triples.first_triple_index().shape == (0,),
            "空目录的形状是对的（没有配方，偏移表只有一个起点）",
        )

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

    # --- 1.6.52：模型配色的三个出口；两页的「只匹配耗材」 ---------------------
    # 前半截不进界面也能跑：GLB 写出 → 读回来，确认几何和颜色都在。
    import tempfile as _tempfile

    import numpy as _np

    from app.core import model_export as _model_export
    from app.core.glb import GlbModel as _GlbModel
    from app.core.glb import GlbPrimitive as _GlbPrimitive
    from app.core.glb import load_model as _load_glb
    from app.core.model_export import glb_bytes as _glb_bytes
    from app.core.model_export import plate_from_model as _plate_from_model

    def _box(low, high, colour):
        x0, y0, z0 = low
        x1, y1, z1 = high
        corners = _np.array(
            [
                [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
            ],
            dtype=_np.float64,
        )
        quads = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        triangles = []
        for a, b, c, d in quads:
            triangles.extend([[a, b, c], [a, c, d]])
        return _GlbPrimitive(
            name="box",
            material="m",
            positions=corners,
            normals=_np.zeros_like(corners),
            triangles=_np.array(triangles, dtype=_np.int64),
            colour=colour,
        )

    boxes = _GlbModel(
        name="两只盒子",
        primitives=[
            _box((0, 0, 0), (0.01, 0.02, 0.03), (255, 0, 0)),
            _box((0.02, 0, 0), (0.03, 0.02, 0.03), (0, 0, 255)),
        ],
        low=_np.zeros(3),
        high=_np.array([0.03, 0.02, 0.03]),
    )
    payload = _glb_bytes(boxes, [(200, 40, 40), (40, 40, 200)])
    check(payload[:4] == b"glTF", "导出的 GLB 带 glTF 头")
    with _tempfile.TemporaryDirectory() as folder:
        target = Path(folder) / "boxes.glb"
        target.write_bytes(payload)
        again = _load_glb(target)
        check(again.triangle_count == 24, f"GLB 往返后还是 24 个三角形，got {again.triangle_count}")
        check(
            sorted(item.colour for item in again.primitives) == [(40, 40, 200), (200, 40, 40)],
            "GLB 往返后颜色换成了给它的那两个",
        )
    plate = _plate_from_model(boxes, [(255, 0, 0), (0, 0, 255)])
    check(len(plate.parts) == 2, f"两个图元摊成两个零件，got {len(plate.parts)}")
    check(abs(plate.width_mm - 30.0) < 1e-6, f"板宽 30 mm，got {plate.width_mm}")
    check(
        abs(float(_np.vstack([part.vertices for part in plate.parts])[:, 2].min())) < 1e-6,
        "摊出来的件站在 z=0 上",
    )
    check(hasattr(_model_export, "MM_PER_UNIT"), "模型导出认 1 单位 = 1 米")
    with _tempfile.TemporaryDirectory() as folder:
        written = _model_export.write_glb(boxes, [(1, 2, 3), (4, 5, 6)], Path(folder) / "m")
        check(written.exists() and written.suffix == ".glb", "write_glb 写出了 .glb")
        check(not list(Path(folder).rglob("*.tmp")), "write_glb 没留下临时文件")

    model_page = window._model_page
    check(model_page._colour_count.value() == 32, "模型配色页颜色数量默认 32")
    check(not model_page._only_spools.isChecked(), "模型配色页「只匹配耗材」默认不勾")
    check(model_page._filter_button.isEnabled() is False, "没有模型时「筛选耗材」是灰的")
    check(model_page._save_shortcut is not None, "模型配色页绑了 Ctrl+S")
    picture_page = window._picture_page
    picture_page._only_spools.setChecked(True)
    check(
        {entry.kind for entry in picture_page._candidate_palette()} <= {"filament"},
        "图像转换页勾上只匹配耗材后没有混色",
    )
    picture_page._only_spools.setChecked(False)
    check(
        any(entry.kind != "filament" for entry in picture_page._candidate_palette()),
        "图像转换页取消勾选后混色回来了",
    )

    # --- 1.6.53（m21616）：设置对话框、默认值不触发重算、翻转 -------------------
    from .core import settings as _settings_api
    from .ui import settings_dialog as _settings_dialog
    from .ui import tips as _tips

    check(_tips.tooltips_disabled(), "悬停提示框全程序关掉了（ToolTip 事件被吃掉）")

    settings_dialog = _settings_dialog.SettingsDialog(dict(_settings_api.DEFAULT_SETTINGS))
    values = settings_dialog.values()
    check(values["model_colours"] == 32, "设置里模型配色颜色数量的默认值是 32")
    check(values["model_spools"] == 4, "设置里模型配色耗材上限的默认值是 4")
    check(values["model_flip_rotate"] is True, "设置里「左右翻转」默认勾着")
    check(values["shortcut_export_glb"] == "Ctrl+S", "设置里模型配色的保存快捷键默认 Ctrl+S")
    settings_dialog._model_colours.setValue(24)
    check(settings_dialog.values()["model_colours"] == 24, "改设置里的颜色数量会写进 values()")
    settings_dialog._on_reset()
    check(settings_dialog.values()["model_colours"] == 32, "「恢复默认」把颜色数量改回 32")
    settings_dialog.deleteLater()

    # 把默认参数下发到页面上：不该因此触发一次重算（1.6.53 里 apply_settings 用
    # blockSignals 掐掉 valueChanged），所以颜色数量会变、方案数不会变。
    before = len(model_page._plans)
    model_page.apply_settings(dict(_settings_api.DEFAULT_SETTINGS))
    check(model_page._colour_count.value() == 32, "设置里的颜色数量下发到了模型配色页")
    check(len(model_page._plans) == before, "下发默认参数不会触一次重算")
    check(model_page._source_view.flipOrbit() is True, "模型配色页默认按「左右翻转」转模型")
    model_page._source_view.setFlipOrbit(False)
    check(model_page._source_view.flipOrbit() is False, "左右翻转可以关掉")
    model_page._source_view.setFlipOrbit(True)

    # --- 1.6.55（m25568）：导入前可改参数、3MF 能读、GLB 带走方案与换色 ---------
    from .core.threemf_import import load_3mf_model as _load_3mf

    model_page._set_enabled(False)
    check(model_page._colour_count.isEnabled(), "没打开模型时「颜色数量」也能改")
    check(model_page._spool_cap.isEnabled(), "没打开模型时「耗材上限」也能改")
    model_page._set_enabled(True)
    check(model_page._source_view.parent().objectName() == "panel", "三维视图套在跟图像转换页一样的卡片里")
    check(model_page._preview_view.parent().objectName() == "panel", "预览也套在同一款卡片里")
    check(model_page._open_button.text() == "打开模型", "按钮叫「打开模型」，3MF 也认")

    three_mf = scratch / "selftest-two.3mf"
    with zipfile.ZipFile(three_mf, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("3D/3dmodel.model", _SELFTEST_3MF)
    three_model = _load_3mf(three_mf)
    check(three_model.triangle_count == 4, "3MF 里的 4 个三角形读到了")
    check(len(three_model.primitives) == 2, "3MF 的两种颜色分成两块")
    check(abs(float(three_model.size[1]) - 0.015) < 1e-6, "3MF 的毫米换成了 1 单位 = 1 米的坐标")

    round_trip = scratch / "selftest-round.glb"
    # 往返在核心层验证（页面级的往返由 tools\smoke_app.py 覆盖）：自检里往这个窗口的
    # 模型页塞模型会让冻结版在退出时崩一次（0xC0000005，2026-10 实测），不值得。
    from .core.glb import load_model as _load_glb
    from .core.model_export import glb_bytes as _glb_bytes

    extras = {
        "bambupalette": {
            "version": 1,
            "plan": "colour:#C8342E",
            "settings": {"colours": 16, "spools": 3},
            "primitives": [{"index": 0, "original": "#C8342E", "painted": "#B34F50"}],
        }
    }
    round_trip.write_bytes(
        _glb_bytes(three_model, [(0xB3, 0x4F, 0x50), (0x2F, 0xA8, 0x4F)], extras=extras)
    )
    back = _load_glb(round_trip)
    check(back.triangle_count == three_model.triangle_count, "GLB 往返后三角形数不变")
    check(back.saved is not None and back.saved.get("plan") == "colour:#C8342E",
          "GLB 往返后 extras 里的方案还在")
    check(back.saved is not None and (back.saved.get("settings") or {}).get("colours") == 16,
          "GLB 往返后 extras 里的设置还在")
    check({tuple(item.colour) for item in back.primitives} == {(0xB3, 0x4F, 0x50), (0x2F, 0xA8, 0x4F)},
          "GLB 往返后每块材质是配到的颜色")

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

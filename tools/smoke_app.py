"""Drive the whole window programmatically and assert the observable behaviour.

Usage::

    .venv\\Scripts\\python.exe tools\\smoke_app.py

This is the end-to-end check for feature 1: five spools in, 810 mixes out, every
sort key and both engines usable, search / pair filter / nearest-match working,
and the selection round-tripping into the detail panel.  The user's real library
is never touched — storage is redirected to a temporary folder.

Exit code 0 means every assertion held.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app.core import paths  # noqa: E402

TEMP = Path(tempfile.mkdtemp(prefix="fcs-smoke-"))
paths.library_path = lambda: TEMP / "filament-library.json"  # type: ignore[assignment]
paths.data_dir = lambda: TEMP  # type: ignore[assignment]

from app.core.engines import ENGINES, ENGINE_BAMBU, ENGINE_MIXER, ENGINE_SPECTRAL  # noqa: E402
from app.core.library import Filament, FilamentLibrary  # noqa: E402
from app.core.mixes import SORT_CHOICES, expected_recipe_count  # noqa: E402
from app.main import build_application  # noqa: E402
from app.ui.main_window import MainWindow  # noqa: E402

SAMPLES = (
    ("大简 PETG HF 白", "大简", "PETG HF", "#F2F0EB"),
    ("大简 PETG HF 黑", "大简", "PETG HF", "#17181C"),
    ("大简 PETG HF 金", "大简", "PETG HF", "#D9A441"),
    ("大简 PETG HF 青", "大简", "PETG HF", "#2FA8B8"),
    ("大简 PETG HF 红", "大简", "PETG HF", "#C8342E"),
)

FAILURES: list[str] = []
CHECKS = 0


def check(condition: bool, label: str) -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(label)


class _StubChooser:
    """Stands in for ``QFileDialog`` so the export path can be driven headlessly."""

    def __init__(self, target: Path) -> None:
        self._target = target

    def getSaveFileName(self, *args, **kwargs):  # noqa: D102
        return (str(self._target), "")

    def getOpenFileName(self, *args, **kwargs):  # noqa: D102
        return (str(self._target), "")


class _StubMessageBox:
    """Stands in for ``QMessageBox`` — ``exec()`` would block with no user."""

    class Icon:  # noqa: D106
        Information = 0

    def __init__(self, *args, **kwargs) -> None:
        pass

    def setWindowTitle(self, *args) -> None:
        pass

    def setIcon(self, *args) -> None:
        pass

    def setText(self, *args) -> None:
        pass

    def exec(self) -> int:
        return 0

    @staticmethod
    def warning(*args, **kwargs) -> int:  # noqa: D102
        return 0


def _make_sample(target: Path) -> None:
    """A small picture with flat colour fields and one soft gradient."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (160, 120), "#F2F0EB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 75, 65), fill="#C8342E")
    draw.ellipse((85, 10, 150, 60), fill="#D9A441")
    draw.polygon([(20, 110), (80, 75), (140, 110)], fill="#2FA8B8")
    image.save(target)


def main() -> int:
    app = build_application([])
    library = FilamentLibrary([
        Filament(name=name, brand=brand, material_type=material, color_hex=value)
        for name, brand, material, value in SAMPLES
    ])
    window = MainWindow(library=library)
    window.resize(1460, 900)
    window.show()
    app.processEvents()

    total = expected_recipe_count(5)
    check(total == 810, f"expected_recipe_count(5) == 810, got {total}")

    # --- catalogue -------------------------------------------------------------
    check(window._catalog.pair_count == 10, "10 pairs for 5 spools")
    check(len(window._recipes) == 810, f"810 mixes in the grid, got {len(window._recipes)}")
    check(window._catalog.engine.id == ENGINE_MIXER, "the Bambu 2.8 pigment engine is the default")
    check(all(r.engine == ENGINE_MIXER for r in window._recipes), "every recipe records its engine")
    check(
        ENGINES[ENGINE_MIXER].mix_rgb_pair((0, 33, 133), (252, 211, 0), 50) == (47, 141, 56),
        "the pigment polynomial matches its upstream reference sample",
    )
    check(
        ENGINES[ENGINE_BAMBU].mix_hex("#FFFFFF", "#000000", 90) == "#E5E5E5",
        "the legacy sRGB engine still reproduces its reference value",
    )
    # The two Bambu models must actually disagree, or the choice is cosmetic.
    # A CHROMATIC pair on purpose: for two neutral spools every engine is forced
    # to the same grey by the neutral guard (see the grey checks below), so a
    # white/black pair would compare two identical values.
    legacy_hex = ENGINES[ENGINE_BAMBU].mix_hex("#D9A441", "#2FA8B8", 50)
    modern_hex = ENGINES[ENGINE_MIXER].mix_hex("#D9A441", "#2FA8B8", 50)
    check(
        legacy_hex != modern_hex,
        f"the 2.8 and 2.5 models differ ({modern_hex} vs {legacy_hex})",
    )
    # Two neutral spools must never mix to a tinted colour, whatever the engine.
    for engine in ENGINES.values():
        for percent in (10, 50, 90):
            r, g, b = engine.mix_rgb_pair((255, 255, 255), (0, 0, 0), percent)
            check(
                r == g == b,
                f"{engine.id} keeps white + black neutral at {percent}% (#{r:02X}{g:02X}{b:02X})",
            )

    # --- every sort key renders the whole catalogue ----------------------------
    check(window._sort_combo.count() == len(SORT_CHOICES), "sort combo lists every sort key")
    for index, (label, key) in enumerate(SORT_CHOICES):
        window._sort_combo.setCurrentIndex(index)
        app.processEvents()
        check(len(window._recipes) == 810, f"sort {key!r} keeps all 810 mixes")
        check(len({id(r) for r in window._recipes}) == 810, f"sort {key!r} has no duplicate cards")
    window._sort_combo.setCurrentIndex(0)
    app.processEvents()

    # --- ascending RGB is really ascending -------------------------------------
    window._sort_combo.setCurrentIndex(0)
    app.processEvents()
    keyed = [r.rgb for r in window._recipes]
    check(keyed == sorted(keyed), "RGB sort is ascending")

    # --- engine switching rebuilds the catalogue -------------------------------
    engine_labels = [window._engine_combo.itemText(i) for i in range(window._engine_combo.count())]
    check(len(engine_labels) == 3, f"three engines offered, got {engine_labels}")

    def select_engine(engine_id: str) -> None:
        index = window._engine_combo.findData(engine_id)
        check(index >= 0, f"the combo offers the {engine_id} engine")
        window._engine_combo.setCurrentIndex(index)
        app.processEvents()

    select_engine(ENGINE_SPECTRAL)
    check(window._catalog.engine.id == ENGINE_SPECTRAL, "switching the combo swaps the engine")
    check(len(window._recipes) == 810, "spectral engine still yields 810 mixes")
    check(
        all(r.engine == ENGINE_SPECTRAL for r in window._recipes),
        "spectral recipes are tagged with the spectral engine",
    )
    # The choice must survive a restart, because the user has to be able to keep it.
    from app.core import settings

    check(
        settings.load_settings().get("engine") == ENGINE_SPECTRAL,
        "the chosen engine is persisted to disk",
    )
    reopened = MainWindow(library=library)
    check(
        reopened._engine_combo.currentData() == ENGINE_SPECTRAL,
        "a fresh window remembers the chosen engine",
    )
    check(reopened._catalog.engine.id == ENGINE_SPECTRAL, "a fresh window rebuilds with it")
    reopened.close()

    select_engine(ENGINE_BAMBU)
    check(window._catalog.engine.id == ENGINE_BAMBU, "the legacy 2.5 sRGB engine is selectable")
    check(len(window._recipes) == 810, "legacy engine still yields 810 mixes")
    check(
        all(r.engine == ENGINE_BAMBU for r in window._recipes),
        "legacy recipes are tagged with the legacy engine",
    )

    select_engine(ENGINE_MIXER)
    check(window._catalog.engine.id == ENGINE_MIXER, "switching back restores the default engine")
    check(
        settings.load_settings().get("engine") == ENGINE_MIXER,
        "switching back is persisted too",
    )
    check(
        all(r.engine == ENGINE_MIXER for r in window._recipes),
        "every recipe is retagged when the engine changes back",
    )

    # --- search ---------------------------------------------------------------
    probe = window._recipes[len(window._recipes) // 3].color_hex[1:5]
    window._search.setText(probe)
    app.processEvents()
    check(0 < len(window._recipes) < 810, f"hex search {probe!r} narrows the grid")
    check(all(probe in r.color_hex for r in window._recipes), "hex search only keeps matches")
    window._search.setText("大简")
    app.processEvents()
    check(len(window._recipes) == 810, "a brand search keeps every mix")
    window._search.setText("PETG HF 青")
    app.processEvents()
    check(0 < len(window._recipes) < 810, "a spool-name search narrows the grid")
    window._search.setText("zzzz-not-a-colour")
    app.processEvents()
    check(window._recipes == [], "a miss empties the grid")
    window._search.clear()
    app.processEvents()
    check(len(window._recipes) == 810, "clearing the search restores all 810")

    # --- pair filter ----------------------------------------------------------
    a, b = library[0], library[2]
    window._show_pair(a.id, b.id)
    app.processEvents()
    check(len(window._recipes) == 81, f"pair filter shows 81 mixes, got {len(window._recipes)}")
    check(
        all({r.a_id, r.b_id} == {a.id, b.id} for r in window._recipes),
        "pair filter shows only that pair",
    )
    check(
        sorted(r.percent_a for r in window._recipes) == list(range(10, 91)),
        "pair filter spans 10%..90% exactly once each",
    )
    window._clear_filters()
    app.processEvents()
    check(len(window._recipes) == 810, "clear filters restores the full catalogue")

    # --- selection drives the detail panel ------------------------------------
    target = window._recipes[123]
    window._grid.selectRecipe(target)
    window._on_grid_selected(target)
    app.processEvents()
    shown = window._detail.recipe()
    check(shown is not None and shown.key == target.key, "selection reaches the detail panel")
    check(shown.percent_a + shown.percent_b == 100, "a pair ratio always sums to 100")
    check(
        shown.a_id in {f.id for f in library} and shown.b_id in {f.id for f in library},
        "the detail panel names two real spools",
    )
    check(
        tuple(shown.pair_key) == (shown.a_id, shown.b_id),
        "pair_key is the parent spool id pair",
    )

    # --- blank space clears the selection; nothing is ever dimmed --------------
    from PySide6.QtCore import QEvent, QPointF, Qt  # noqa: E402
    from PySide6.QtGui import QMouseEvent  # noqa: E402

    from app.ui import mix_grid  # noqa: E402

    def click_grid(x: float, y: float) -> None:
        window._grid.mousePressEvent(
            QMouseEvent(
                QEvent.Type.MouseButtonPress,
                QPointF(x, y),
                QPointF(x, y),
                Qt.MouseButton.LeftButton,
                Qt.MouseButtons.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
        )

    # One lonely cell in the top-left leaves the rest of its row blank.
    single = window._recipes[0]
    window._grid.setRecipes([single])
    app.processEvents()
    click_grid(mix_grid.CELL / 2, mix_grid.CELL / 2)
    app.processEvents()
    check(window._grid.selectedRecipe() is not None, "clicking a swatch selects it")
    click_grid(mix_grid.CELL + 6, mix_grid.CELL / 2)
    app.processEvents()
    check(window._grid.selectedRecipe() is None, "clicking blank space deselects the mix")
    check(window._detail.recipe() is None, "deselecting clears the detail panel")
    # The old build dimmed every other mix to 22% and boxed the selection in blue.
    check(not hasattr(window._grid, "setFocusPair"), "the grid has no dimming hook any more")
    ring = mix_grid._selection_colour((255, 255, 255))
    check(ring.alpha() < 255, "the selection ring is drawn at reduced opacity")
    check(
        max(ring.red(), ring.green(), ring.blue()) < 160,
        "a white selection gets a darkened ring so it stays visible on white",
    )
    red_ring = mix_grid._selection_colour((216, 52, 44))
    check(
        red_ring.red() == 216 and red_ring.green() == 52 and red_ring.blue() == 44,
        "a saturated selection keeps its own colour for the ring",
    )
    window._refresh_grid()
    app.processEvents()
    check(len(window._recipes) == 810, "restoring the full catalogue after the blank click")

    # --- 全部颜色: raw spools beside the mixes --------------------------------
    # The request was "输入两个颜色，全部颜色就是83个颜色" -> 81 mixes + 2 raw spools.
    from PySide6.QtWidgets import QCheckBox, QLabel  # noqa: E402

    check(hasattr(window, "_all_colours"), "the 全部混色 panel has a 全部颜色 control")
    check(isinstance(window._all_colours, QCheckBox), "it is a checkbox, as requested")
    check(window._all_colours.text() == "全部颜色", "the checkbox is labelled 全部颜色")
    check(not window._all_colours.isChecked(), "the checkbox starts unchecked (mixes only)")
    panel = window._all_colours.parentWidget()
    check(panel is not None and panel.objectName() == "panel",
          "the checkbox lives on the 全部混色 panel")
    headings = [w for w in panel.findChildren(QLabel) if w.text() == "全部混色"]
    check(len(headings) == 1, "the panel still carries its 全部混色 heading")
    if headings:
        app.processEvents()
        check(
            window._all_colours.x() > headings[0].x() + headings[0].width(),
            "the checkbox sits to the RIGHT of the 全部混色 heading (top right, as asked)",
        )
    check(expected_recipe_count(2) == 81, "two spools make 81 mixes")
    check(expected_recipe_count(2) + 2 == 83, "81 mixes + the 2 raw spools = the 83 colours asked for")

    check(len(window._recipes) == 810, "with the box off the grid is mixes only")
    window._all_colours.setChecked(True)
    app.processEvents()
    check(
        len(window._recipes) == 810 + len(library),
        f"ticking it adds one row per spool, got {len(window._recipes)} for {len(library)} spools",
    )
    spools = [cell for cell in window._recipes if getattr(cell, "pair_index", 0) < 0]
    mixes = [cell for cell in window._recipes if getattr(cell, "pair_index", 0) >= 0]
    check(len(spools) == len(library), f"one spool row per spool, got {len(spools)}")
    check(len(mixes) == 810, f"the 810 mixes are untouched, got {len(mixes)}")
    check(all(cell.is_spool for cell in spools), "every spool row reports itself as a spool")
    check(
        all(cell.color_hex == cell.filament.color_hex for cell in spools),
        "a spool row shows the spool's own colour, not a mix",
    )
    check(
        [cell.rgb for cell in window._recipes]
        == sorted(cell.rgb for cell in window._recipes),
        "「全部颜色」 keeps the WHOLE list in RGB order instead of pinning the spools on top",
    )
    check(
        any(cell.key not in {s.key for s in spools} for cell in window._recipes[: len(spools)]),
        "a spool takes its place in the colour order rather than being prepended",
    )
    check(
        [cell.rgb for cell in spools] == sorted(cell.rgb for cell in spools),
        "spool rows follow the 排序 control (RGB ascending here)",
    )
    # The same control must order both halves, so an explicit colour sort has to
    # move the spools too rather than leaving them frozen at the top.
    from app.spectral import color as spectral_color  # noqa: E402

    window._sort_combo.setCurrentIndex(
        [key for key, _ in SORT_CHOICES].index("lightness")
    )
    app.processEvents()
    light_spools = [c for c in window._recipes if getattr(c, "pair_index", 0) < 0]
    levels = [spectral_color.lab_from_rgb(c.rgb)[0] for c in light_spools]
    check(levels == sorted(levels, reverse=True), "排序 by lightness reorders the spool rows too")
    window._sort_combo.setCurrentIndex([key for key, _ in SORT_CHOICES].index("rgb"))
    app.processEvents()

    # Selecting a raw spool must explain itself rather than pretend to be a mix.
    window._grid.selectRecipe(spools[0])
    window._on_grid_selected(spools[0])
    app.processEvents()
    check(window._detail.recipe() is None, "a raw spool is not shown as a mix recipe")
    shown_spool = window._detail.filament()
    check(
        shown_spool is not None and shown_spool.id == spools[0].filament.id,
        "the detail panel shows the spool itself",
    )
    check(
        window._detail.selectionKey() == f"spool|{spools[0].filament.id}",
        "the detail panel keys a spool selection distinctly from a mix",
    )
    check(window.nearest_recipe("#982C29") is not None, "找最接近的 still works with spools listed")

    window._search.setText("大简")
    app.processEvents()
    check(
        len(window._recipes) == 810 + len(library),
        "a brand search keeps every spool row too",
    )
    window._search.clear()
    app.processEvents()
    window._all_colours.setChecked(False)
    app.processEvents()
    check(len(window._recipes) == 810, "un-ticking 全部颜色 restores the mixes-only grid")
    window._detail.clear()

    # --- no preset colour, no delete prompt, Del deletes -----------------------
    from PySide6.QtGui import QKeySequence  # noqa: E402
    from PySide6.QtWidgets import QMessageBox  # noqa: E402

    from app.ui.filament_dialog import FilamentDialog  # noqa: E402

    add_dialog = FilamentDialog(None, [f.brand for f in library.filaments], [], window)
    check(not add_dialog._color.isSet(), "添加耗材 starts with NO preset colour")
    check(add_dialog.values()["color_hex"] == "", "an untouched dialog reports no colour")
    check(not add_dialog._ok_button.isEnabled(), "确认 is disabled while no colour is chosen")
    add_dialog._hex.setText("#C8342E")
    add_dialog._on_hex_edited()
    check(add_dialog._ok_button.isEnabled(), "确认 wakes up once the user types a colour")
    check(add_dialog.values()["color_hex"] == "#C8342E", "the typed colour is what the spool gets")
    add_dialog._hex.setText("nonsense")
    add_dialog._on_hex_edited()
    check(add_dialog.values()["color_hex"] == "#C8342E", "gibberish leaves the last good colour alone")
    add_dialog.close()

    check(
        window._delete_shortcut.key() == QKeySequence(QKeySequence.StandardKey.Delete),
        "Del is bound to the spool list",
    )
    # Deleting must not stop to ask. Make any question box an outright failure.
    original_question = QMessageBox.question
    QMessageBox.question = staticmethod(
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("删除耗材 asked for confirmation"))
    )
    try:
        doomed = window.library.filaments[-1]
        window._select_filament(doomed.id)
        before = len(window.library.filaments)
        snapshot = list(window.library.filaments)
        window._delete_shortcut.activated.emit()
        app.processEvents()
        check(
            len(window.library.filaments) == before - 1,
            "pressing Del deletes the selected spool without asking",
        )
        check(window.library.get(doomed.id) is None, "the spool Del removed is really gone")
    finally:
        QMessageBox.question = original_question
        window.library.replace_all(snapshot)
        window._reload_library()
        app.processEvents()
    check(len(window.library.filaments) == before, "the library is restored after the Del test")

    # --- nearest match --------------------------------------------------------
    nearest = window.nearest_recipe("#982C29")
    check(nearest is not None, "nearest_recipe returns something for #982C29")
    if nearest is not None:
        check(len(nearest.color_hex) == 7, f"nearest recipe has a hex, got {nearest.color_hex!r}")
    # The target colour must start UNSET: a pre-filled colour would claim the user
    # already aimed at something they never chose.
    check(window._target_color.hex() == "", "the 目标颜色 picker starts with no preset colour")
    check(not window._target_color.isSet(), "the picker reports itself unset")
    check(not window._find_button.isEnabled(), "找最接近的混色 is disabled until a colour is chosen")
    window._target_color.setValue("#808080")
    check(window._find_button.isEnabled(), "choosing a colour enables 找最接近的混色")
    window._on_find_nearest()
    app.processEvents()
    check(window._detail.recipe() is not None, "找最接近的混色 selects a recipe")

    # --- the add/edit dialog builds a valid filament ---------------------------
    from app.ui.filament_dialog import FilamentDialog  # noqa: E402

    dialog = FilamentDialog(brands=("大简",), types=("PETG HF",))
    dialog.setWindowTitle("smoke")
    dialog._name.setText("测试耗材")
    dialog._brand.setCurrentText("大简")
    dialog._type.setCurrentText("PETG HF")
    dialog._hex.setText("#1234AB")
    dialog._on_hex_edited()
    app.processEvents()
    built = dialog.build_filament()
    check(built.name == "测试耗材", "the dialog carries the name through")
    check(built.color_hex == "#1234AB", f"the dialog carries the hex through, got {built.color_hex}")
    check(built.brand == "大简" and built.material_type == "PETG HF", "the dialog carries brand/type through")
    check(dialog._color.hex() == "#1234AB", "the colour button follows the hex field")
    dialog.close()

    edited = FilamentDialog(filament=library[1], brands=("大简",), types=("PETG HF",))
    check(edited.build_filament().id == library[1].id, "editing keeps the filament id")
    check(edited.build_filament().color_hex == library[1].color_hex, "editing preloads the colour")
    edited.close()

    # --- picture page -----------------------------------------------------------
    from PySide6.QtCore import Qt

    from app.ui import picture_page as _picture_page

    sample = TEMP / "sample.png"
    _make_sample(sample)
    page = window._picture_page
    page.loadImage(sample)
    app.processEvents()
    result = page.result
    check(result is not None, "the sample picture matched a palette")
    if result is not None:
        from app.core.image_matching import build_palette

        full = build_palette(library, window._catalog)
        check(
            len(full) == len(library) + len(window._recipes),
            f"the candidate palette is every spool plus every mix, got {len(full)}",
        )
        check(
            all(int(count) > 0 for count in result.counts),
            "an unused colour is dropped from the matched palette",
        )
        check(len(result.palette) >= 3, f"the picture matched several colours, got {len(result.palette)}")
        check(result.printed_pixels > 0, "the picture has solid pixels")
        rows = page._list._list
        check(rows.count() == len(result.palette), f"the colour list lists every entry, got {rows.count()}")

        last = len(result.palette) - 1
        page._on_colour_selected(last)
        app.processEvents()
        check(page._view._selected == last, "choosing a colour highlights it in the picture")
        check(
            rows.item(0).data(Qt.ItemDataRole.UserRole) == last,
            "the chosen colour is pinned to the top of the list",
        )
        order = [rows.item(i).data(Qt.ItemDataRole.UserRole) for i in range(rows.count())]
        expected = [last] + [i for i in range(len(result.palette)) if i != last]
        check(order == expected, "every other colour keeps its original order")

        page._on_region_clicked(0)
        app.processEvents()
        check(
            rows.item(0).data(Qt.ItemDataRole.UserRole) == 0,
            "clicking the picture pins the colour that owns that pixel",
        )

        page.buildPlate()
        plate = page.plate
        check(plate is not None, "a plate was built from the picture")
        if plate is not None:
            check(
                len(plate.parts) == len(result.palette) + 1,
                f"the plate is the base plus one part per colour, got {len(plate.parts)} parts",
            )
            check(
                plate.extruder_count == len(result.palette),
                f"one extruder per colour, got {plate.extruder_count}",
            )
            check(
                plate.parts[0].region_pixels == result.printed_pixels,
                "the base plate covers the whole silhouette",
            )
            check(plate.parts[0].z_bottom_mm == 0.0, "the base plate starts on the bed")
            check(
                all(part.z_bottom_mm == plate.base_thickness_mm for part in plate.parts[1:]),
                "every colour prism sits on top of the base plate",
            )

            # --- 选中的颜色: both RGBs, the recipe, and manual replacement -------
            from app.core.image_matching import MatchSettings, sort_palette
            from app.core.mixes import (
                SORT_CHOICES_WITH_SIMILARITY,
                SORT_SIMILARITY,
            )
            from app.spectral import color as _spectral_color
            from app.ui.colour_picker import ColourDetail, ColourPickerDialog

            check(isinstance(page._detail, ColourDetail), "the picture page has a 选中的颜色 panel")
            check(
                page._auto_palette == list(result.palette) and page._overrides == {},
                "a fresh match starts with the automatic palette and no replacements",
            )
            check(MatchSettings().merge_delta_e == 2.0, "near-identical matches merge by default")

            page._select(0)
            app.processEvents()
            entry = page._result.palette[0]
            check(
                page._detail.image_hex() == _spectral_color.rgb_to_hex(page._result.region_colour(0)),
                "the panel shows the PICTURE's colour, not only the match",
            )
            check(
                page._detail.match_hex() == entry.color_hex.upper(),
                f"the panel shows the matched colour, got {page._detail.match_hex()}",
            )
            check(len(page._detail.image_hex()) == 7, "the picture colour is a real hex")
            check(
                page._detail.image_hex() == page._detail.image_hex().upper(),
                "the picture colour is shown as an uppercase hex",
            )
            recipe = page._detail.recipe_text()
            check(
                "配方" in recipe or "耗材本色" in recipe,
                f"the match is labelled with its recipe, got {recipe!r}",
            )
            if entry.is_mix:
                check("%" in recipe, "a mix recipe carries its percentages")
            check(page._detail.can_replace(), "the selected colour can be replaced")
            check(not page._detail.can_restore(), "there is nothing to restore before a change")

            # The replacement menu is 「全部颜色」, ordered like the 混色配方 grid.
            entries = build_palette(library, window._catalog, include_mixes=True)
            picker = ColourPickerDialog(
                entries,
                page._result.region_colour(0),
                library=library,
                current_key=entry.key,
            )
            app.processEvents()
            check(
                picker._sort.currentData() == SORT_SIMILARITY,
                "the replacement menu opens sorted by similarity to the picture colour",
            )
            ordered = picker.ordered()
            check(len(ordered) == len(entries), "the replacement menu lists 全部颜色 in full")
            differences = [
                float(
                    _spectral_color.delta_e_2000(
                        candidate.lab,
                        _spectral_color.lab_from_rgb(page._result.region_colour(0)),
                    )
                )
                for candidate in ordered
            ]
            check(differences == sorted(differences), "similarity sort really is nearest first")
            check(
                ordered[0].color_hex.upper() == entry.color_hex.upper(),
                "the automatic match leads the similarity sort",
            )
            for key, _ in SORT_CHOICES_WITH_SIMILARITY:
                with_key = sort_palette(entries, key, target_rgb=page._result.region_colour(0))
                check(
                    {e.key for e in with_key} == {e.key for e in entries},
                    f"replacement sort {key!r} keeps every candidate",
                )
            picker.close()

            replacement = next(
                candidate for candidate in ordered if candidate.key != entry.key
            )
            page._overrides[0] = replacement
            page._apply_overrides()
            app.processEvents()
            check(
                page._result.palette[0].key == replacement.key,
                "the replacement lands in the palette the print uses",
            )
            check(page._detail.can_restore(), "a replaced colour can be restored")
            check(
                replacement.color_hex.upper() in page._status.text()
                or "已手动更换" in page._status.text(),
                "the status line reports the manual replacement",
            )
            check(
                replacement.color_hex.upper()
                in {part.color_hex.upper() for part in page.buildPlate().parts},
                "the plate really uses the replacement colour",
            )

            page._on_restore()
            app.processEvents()
            check(
                page._result.palette[0].key == entry.key,
                "restoring puts the automatic match back",
            )
            check(page._overrides == {}, "restoring forgets the replacement")
            check(not page._detail.can_restore(), "nothing left to restore")
            check(
                entry.color_hex.upper()
                in {part.color_hex.upper() for part in page.buildPlate().parts},
                "the plate goes back to the automatic colour",
            )

            # Exercise the two export buttons, with the file chooser stubbed out.
            original_dialog = _picture_page.QFileDialog
            original_box = _picture_page.QMessageBox
            _picture_page.QMessageBox = _StubMessageBox
            try:
                for kind, suffix in (("3mf", ".3mf"), ("obj", ".obj")):
                    target = TEMP / f"plate{suffix}"
                    _picture_page.QFileDialog = _StubChooser(target)
                    page._on_export(kind)
                    app.processEvents()
                    check(target.is_file(), f"exporting {kind} wrote {target.name}")
                    check(target.stat().st_size > 0, f"the {kind} export is not empty")
                import zipfile

                check(zipfile.is_zipfile(TEMP / "plate.3mf"), "the 3MF is a real zip package")
                check((TEMP / "plate.mtl").is_file(), "the OBJ export wrote its matching .mtl")
            finally:
                _picture_page.QFileDialog = original_dialog
                _picture_page.QMessageBox = original_box

    # --- persistence round trip ------------------------------------------------
    window._save_library()
    check(paths.library_path().is_file(), "the library was written to disk")
    reloaded = FilamentLibrary.load()
    check(len(reloaded) == 5, f"reload sees 5 spools, got {len(reloaded)}")
    check(reloaded.filaments[0].color_hex == "#F2F0EB", "colours survive the round trip")

    # --- empty state ----------------------------------------------------------
    empty = FilamentLibrary()
    bare = MainWindow(library=empty)
    bare.resize(1200, 760)
    bare.show()
    app.processEvents()
    check(bare._recipes == [], "an empty library produces no mixes")
    check(bare._detail.recipe() is None, "an empty library shows no recipe")
    bare.close()

    window.close()

    print(f"checks: {CHECKS}")
    if FAILURES:
        print(f"FAILED ({len(FAILURES)})")
        for item in FAILURES:
            print("  -", item)
        return 1
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

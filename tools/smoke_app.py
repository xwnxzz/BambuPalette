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

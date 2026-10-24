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
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _key_parts(key: str) -> tuple[str, str, int]:
    """Split a MixRecipe.key back into find_recipe() arguments."""
    a_id, b_id, percent = key.split("|")
    return a_id, b_id, int(percent)

# Redirect storage before anything imports it, both ways: the environment
# variable is the documented hook, the patch keeps this tool correct even if a
# module cached the paths at import time.
TEMP = Path(tempfile.mkdtemp(prefix="fcs-smoke-"))
os.environ["BAMBU_PALETTE_DATA_DIR"] = str(TEMP)

from app.core import paths  # noqa: E402

paths.library_path = lambda: TEMP / "filament-library.json"  # type: ignore[assignment]
paths.data_dir = lambda: TEMP  # type: ignore[assignment]

from app.core.engines import ENGINES, ENGINE_BAMBU, ENGINE_MIXER, ENGINE_SPECTRAL  # noqa: E402
from app.core.library import Filament, FilamentLibrary  # noqa: E402
from app.core.mixes import SORT_CHOICES, expected_recipe_count  # noqa: E402
from app.main import build_application, show_main_window as _show_main_window  # noqa: E402
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


class _StubDialog:
    """Stands in for the block dialogs (``MergeDialog`` / ``ColourPickerDialog``).

    Both are *called* to build a dialog and then ``exec()``-ed, so the picture page
    asks for the chosen row (``int``) or the chosen entry (``PaletteEntry``).
    """

    def __init__(self, choice: int, chosen=None) -> None:
        self._choice = choice
        self._chosen = chosen

    def exec(self):  # noqa: D102
        from PySide6.QtWidgets import QDialog  # noqa: PLC0415

        return QDialog.DialogCode.Accepted

    def chosen(self):  # noqa: D102
        return self._chosen if self._chosen is not None else self._choice


class _StubButton:
    """A stand-in for a QMessageBox button: identity is what the code compares."""

    def __init__(self, text: str) -> None:
        self.text = text


class _StubMessageBox:
    """Stands in for ``QMessageBox`` — ``exec()`` would block with no user.

    ``answer`` is the text of the button the fake user presses.  The destructive
    one is the default because that is what someone who pressed 清空耗材库 and
    then confirmed meant; a check that wants the safe path sets it to 取消.
    """

    class Icon:  # noqa: D106
        Information = 0
        Warning = 1

    class ButtonRole:  # noqa: D106
        AcceptRole = 0
        DestructiveRole = 2
        RejectRole = 3

    answer = "清空"

    def __init__(self, *args, **kwargs) -> None:
        self.buttons: dict[str, _StubButton] = {}
        self.informative = ""
        self.default: _StubButton | None = None

    def setWindowTitle(self, *args) -> None:
        pass

    def setIcon(self, *args) -> None:
        pass

    def setText(self, *args) -> None:
        pass

    def setInformativeText(self, text: str, *args) -> None:
        self.informative = text

    def addButton(self, text: str, *args) -> _StubButton:
        button = _StubButton(text)
        self.buttons[text] = button
        return button

    def setDefaultButton(self, button: _StubButton) -> None:
        self.default = button

    def exec(self) -> int:
        return 0

    def clickedButton(self) -> _StubButton | None:
        return self.buttons.get(self.answer)

    @staticmethod
    def warning(*args, **kwargs) -> int:  # noqa: D102
        return 0

    @staticmethod
    def information(*args, **kwargs) -> int:  # noqa: D102
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
    # auto_triples=False keeps this window deterministic: the three-filament
    # table costs a real build and would change every grid-size number below.
    # The dedicated 三色混色 section further down builds one on purpose.
    window = MainWindow(library=library, auto_triples=False)
    window.resize(1460, 900)
    window.show()
    app.processEvents()

    # The app must open maximised: the colour table wants every pixel, and a
    # windowed default made people maximise by hand on every single start.
    starter = MainWindow(library=FilamentLibrary(list(library.filaments)), auto_triples=False)
    _show_main_window(starter)
    app.processEvents()
    check(starter.isMaximized(), "the main window opens maximised")
    starter.close()

    total = expected_recipe_count(5)
    check(total == 810, f"expected_recipe_count(5) == 810, got {total}")

    # --- catalogue -------------------------------------------------------------
    check(window._catalog.pair_count == 10, "10 pairs for 5 spools")
    # The grid shows DISTINCT colours since m06962, so its size is the merged
    # count — never larger than the raw recipe count.
    merged = window._catalog.colour_count
    # 「全部颜色」 is ONE list: the mixes and the spools' own colours, deduplicated
    # across both. So the grid is at least the merged mixes, usually a little more.
    everything = window._combined.colour_count
    check(merged <= 810, f"merging never invents colours ({merged} <= 810)")
    check(
        everything >= merged,
        f"「全部颜色」 holds the merged mixes plus the raw spools ({everything} >= {merged})",
    )
    check(
        len(window._recipes) == everything,
        f"{everything} distinct colours in the grid, got {len(window._recipes)}",
    )
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
        check(len(window._recipes) == everything, f"sort {key!r} keeps all {everything} colours")
        check(
            len({r.key for r in window._recipes}) == everything,
            f"sort {key!r} has no duplicate cards",
        )
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
    check(
        len(window._recipes) == window._combined.colour_count,
        "spectral engine still covers every merged colour",
    )
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
    reopened = MainWindow(library=library, auto_triples=False)
    check(
        reopened._engine_combo.currentData() == ENGINE_SPECTRAL,
        "a fresh window remembers the chosen engine",
    )
    check(reopened._catalog.engine.id == ENGINE_SPECTRAL, "a fresh window rebuilds with it")
    reopened.close()

    select_engine(ENGINE_BAMBU)
    check(window._catalog.engine.id == ENGINE_BAMBU, "the legacy 2.5 sRGB engine is selectable")
    check(
        len(window._recipes) == window._combined.colour_count,
        "legacy engine still covers every merged colour",
    )
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
    check(0 < len(window._recipes) < everything, f"hex search {probe!r} narrows the grid")
    check(all(probe in r.color_hex for r in window._recipes), "hex search only keeps matches")
    window._search.setText("大简")
    app.processEvents()
    check(len(window._recipes) == everything, "a brand search keeps every colour")
    window._search.setText("PETG HF 青")
    app.processEvents()
    check(0 < len(window._recipes) < everything, "a spool-name search narrows the grid")
    window._search.setText("zzzz-not-a-colour")
    app.processEvents()
    check(window._recipes == [], "a miss empties the grid")
    window._search.clear()
    app.processEvents()
    check(len(window._recipes) == everything, f"clearing the search restores all {everything} colours")

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
    # --- a filter must announce itself and offer the way back -----------------
    check(
        not window._filter_bar.isHidden(),
        "the pair filter raises a banner above the grid",
    )
    check(
        a.display_name in window._filter_label.text() or b.display_name in window._filter_label.text(),
        f"the banner names the filtered spools, got {window._filter_label.text()!r}",
    )
    hidden = max(window._combined.colour_count - 81, 0)
    check(
        "现在显示 81 个" in window._filter_label.text()
        and f"另外 {hidden} 个没有显示" in window._filter_label.text(),
        "the banner says how many are shown and how many are hidden, "
        f"got {window._filter_label.text()!r}",
    )
    check(
        window._filter_back.text() == "显示全部颜色",
        "the banner carries a 显示全部颜色 button",
    )
    check(
        window._show_all_button.property("accent") == "true",
        "the toolbar's 显示全部颜色 button is highlighted while a filter is on",
    )
    check(
        window._detail._pair_button.text() == "显示全部颜色",
        "the detail panel's pair button becomes a way out",
    )

    window._clear_filters()
    app.processEvents()
    check(len(window._recipes) == everything, "clear filters restores the full catalogue")
    check(window._filter_bar.isHidden(), "clearing the filter hides the banner")
    check(
        window._show_all_button.property("accent") == "false",
        "clearing the filter drops the toolbar highlight",
    )
    check(
        window._detail._pair_button.text() == "只看这一对耗材的全部混色",
        "the pair button goes back to narrowing",
    )

    # --- selection drives the detail panel ------------------------------------
    target = window._recipes[123]
    window._grid.selectRecipe(target)
    window._on_grid_selected(target)
    app.processEvents()
    shown = window._detail.recipe()
    check(
        window._detail.colour() is target and shown is not None,
        "selection reaches the detail panel",
    )
    check(
        shown.key == target.first.key,
        "the panel explains the merged colour's first recipe",
    )
    check(
        window._detail.selectionKey() == target.key,
        "the detail panel remembers the merged colour, not just one recipe",
    )
    check(target.recipe_count >= 1, "a merged colour always keeps at least one recipe")
    check(shown.percent_a + shown.percent_b == 100, "a pair ratio always sums to 100")
    check(
        shown.a_id in {f.id for f in library} and shown.b_id in {f.id for f in library},
        "the detail panel names two real spools",
    )
    check(
        tuple(shown.pair_key) == (shown.a_id, shown.b_id),
        "pair_key is the parent spool id pair",
    )

    # --- a merged colour lists EVERY recipe that reaches it --------------------
    shared = [colour for colour in window._catalog.colours if colour.recipe_count > 1]
    check(bool(shared), f"some colours are reachable more than once ({len(shared)})")
    if shared:
        merged_colour = max(shared, key=lambda colour: colour.recipe_count)
        window._grid.selectRecipe(merged_colour)
        window._on_grid_selected(merged_colour)
        app.processEvents()
        check(
            window._detail.colour() is merged_colour,
            "the detail panel keeps the clicked merged colour",
        )
        listed = [row for row in window._detail._formula_rows if not row["frame"].isHidden()]
        check(
            len(listed) == merged_colour.recipe_count,
            f"all {merged_colour.recipe_count} recipes are listed, got {len(listed)}",
        )
        check(
            window._detail._formula_title.isVisible()
            and str(merged_colour.recipe_count) in window._detail._formula_title.text(),
            f"the list is titled with the count, got {window._detail._formula_title.text()!r}",
        )
        keys = [row["key"].text() for row in listed]
        check(
            set(keys) == {recipe.key for recipe in merged_colour.recipes},
            "the listed rows are exactly that colour's recipes",
        )
        check(
            all(row["swatch"].hex() == merged_colour.color_hex for row in listed),
            "every listed recipe really produces the selected colour",
        )
        check(
            all(row["text"].text().count("%") == 2 for row in listed),
            "each listed row spells out both percentages",
        )
        window._detail.clear()
    window._refresh_grid()
    app.processEvents()

    # --- 三色混色: every triple, merged, cached --------------------------------
    from app.core.triples import TripleCatalog, triple_recipe_count  # noqa: E402

    spools = len(library)
    check(spools >= 3, f"the library has enough spools for triples ({spools})")
    expected_triples = spools * (spools - 1) * (spools - 2) // 6
    reference = TripleCatalog(library.filaments, engine=window._catalog.engine.id).build()
    check(
        reference.triple_count == expected_triples,
        f"{spools} spools give {expected_triples} triples",
    )
    check(
        reference.recipe_count == triple_recipe_count(spools),
        f"every triple carries 2556 ratios ({reference.recipe_count:,} recipes)",
    )
    first_recipes = reference.recipes_at(0)
    check(
        bool(first_recipes) and all(r.color_hex == reference.color_hex_at(0) for r in first_recipes),
        "every recipe behind one triple colour really makes that colour",
    )
    check(
        all(sum(r.percents) == 100 for r in first_recipes)
        and all(min(r.percents) >= 10 for r in first_recipes),
        "every triple ratio sums to 100 with each colour at least 10%",
    )

    # 三色混色 is not optional any more: _ensure_triples starts it on its own,
    # because 「全部颜色」 is meant to hold the two- and three-filament mixes and
    # the spool colours in one list with no checkbox to tick.
    triple_window = MainWindow(library=FilamentLibrary(list(library.filaments)))
    triple_window.resize(1460, 900)
    app.processEvents()
    deadline = time.time() + 120
    while triple_window._triple_timer is not None and time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    check(
        triple_window._triples is not None and triple_window._triples.built,
        "三色混色 starts on its own, with no checkbox",
    )
    check(
        not hasattr(triple_window, "_triples_box")
        and not hasattr(triple_window, "_all_colours"),
        "neither checkbox exists any more",
    )
    check(
        len(triple_window._combined) == len(triple_window._recipes),
        "the grid holds exactly the combined 「全部颜色」 table",
    )
    check(
        len(triple_window._combined) >= reference.colour_count,
        "「全部颜色」 contains every distinct triple colour",
    )
    triple_keys = set()
    step = max(1, reference.colour_count // 400)
    for index in range(0, reference.colour_count, step):
        triple_keys.add(reference.color_hex_at(index))
    missing = [
        hex_value
        for hex_value in triple_keys
        if triple_window._combined.index_of_key(f"colour|{hex_value}") < 0
    ]
    check(
        not missing,
        f"every sampled triple colour is inside 「全部颜色」 ({len(triple_keys)} sampled, "
        f"{len(missing)} missing)",
    )
    check(
        "三色混色" in triple_window._status.text(),
        f"the bottom line explains the triple table, got {triple_window._status.text()!r}",
    )
    cache_file = triple_window._triples_path()
    check(cache_file.is_file(), f"the triple table is cached at {cache_file.name}")
    # The search box stays usable: a text query is resolved against the spools,
    # and a hex fragment is matched vectorised over millions of colours.
    check(triple_window._search.isEnabled(), "search stays on with the triple table up")
    triple_window._search.setText("PETG HF 青")
    app.processEvents()
    check(
        0 < len(triple_window._recipes) < len(triple_window._combined),
        "a spool-name search still narrows the combined list",
    )
    triple_window._search.clear()
    app.processEvents()
    check(
        len(triple_window._recipes) == len(triple_window._combined),
        "clearing the search brings the whole combined list back",
    )

    # Adding a spool must throw the stale triple table away and start over —
    # otherwise the old colours would be cached under the new library's file name
    # and every later launch would show the wrong mixes.
    before_colours = len(triple_window._combined)
    spare = Filament(brand="大简", material_type="PETG HF", color_hex="#123456")
    triple_window.library.add(spare)
    triple_window._reload_library()
    app.processEvents()
    deadline = time.time() + 120
    while triple_window._triple_timer is not None and time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    check(
        triple_window._triples is not None
        and triple_window._triples.built
        and len(triple_window._combined) != before_colours,
        "editing the library rebuilds the triple table instead of keeping stale colours",
    )
    check(
        any(c.color_hex.upper() == spare.color_hex.upper() and c.filaments for c in triple_window._combined),
        "the new spool's own colour is in 「全部颜色」",
    )
    # Issue 1: double-clicking a THREE-colour mix used to filter the grid to its
    # first two spools, i.e. it swapped the colour under the cursor for a
    # different one's 81 two-colour ratios.  A triple-only colour must stay put.
    combined_store = triple_window._combined
    triple_only = None
    triple_catalogue = triple_window._triples.colours()
    for position in range(min(len(triple_catalogue), 800)):
        found = combined_store.index_of_key(f"colour|{triple_catalogue[position].color_hex}")
        if found >= 0 and combined_store.pair_colour_at(found) is None:
            triple_only = combined_store[found]
            break
    check(triple_only is not None, "the triple table holds a colour no two-colour mix reaches")
    if triple_only is not None:
        first_recipe = triple_only.recipes[0]
        check(
            hasattr(first_recipe, "percent_c"),
            f"that colour's first recipe is a three-spool one, got {first_recipe!r}",
        )
        check(
            first_recipe.ratio_text.count("+") == 2,
            f"a three-colour recipe names all three spools, got {first_recipe.ratio_text!r}",
        )
        triple_window._pair_filter = None
        triple_window._on_grid_selected(triple_only)
        app.processEvents()
        check(
            triple_window._pair_filter is None,
            "selecting a three-colour mix does not fall back to two spools",
        )
        check(
            triple_window._detail.colour() is triple_only,
            "selecting a three-colour mix keeps it in the detail panel",
        )
        # The bottom line carries the table's own explanation, never a one-line
        # recipe that could only ever name the first two spools.
        check(
            "配比" in triple_window._status.text(),
            f"the bottom line is the table note, got {triple_window._status.text()!r}",
        )
    # The double-click jump was removed entirely: activating a TWO-colour mix must
    # not narrow the grid to that pair either.  The pair filter now has exactly one
    # entry point, the 「只看这一对耗材的全部混色」 button in the detail panel.
    pair_cell = None
    for position in range(len(combined_store)):
        found = combined_store[position]
        if getattr(found, "recipes", None) and combined_store.pair_colour_at(position) is not None:
            pair_cell = found
            break
    check(pair_cell is not None, "the combined table holds a two-colour mix to test with")
    if pair_cell is not None:
        triple_window._pair_filter = None
        triple_window._on_grid_selected(pair_cell)
        app.processEvents()
        check(
            triple_window._pair_filter is None,
            "activating a two-colour mix no longer filters the grid to its pair",
        )
        check(
            triple_window._detail.colour() is pair_cell,
            "activating a two-colour mix keeps it in the detail panel",
        )
    # Issue 4: the right-hand pane could not be dragged — it snapped.  A QLabel
    # holding the ~1400 px one-line note had no word wrap, and a non-wrapping
    # QLabel's minimumSizeHint IS that full width, so the grid pane pinned the
    # splitter.  The note now lives at the bottom of the page body with an
    # Ignored width policy, so no pane can be wider than the window by accident.
    from PySide6.QtWidgets import QSizePolicy, QSplitter, QStatusBar  # noqa: E402

    splitter = triple_window._tabs.widget(0).findChild(QSplitter)
    check(splitter is not None, "the 颜色配方 page still has its splitter")
    if splitter is not None:
        pinned = [splitter.widget(i).minimumSizeHint().width() for i in range(splitter.count())]
        check(
            max(pinned) < 720,
            f"no pane is pinned wider than the window by a label, got {pinned}",
        )
        check(
            triple_window._status.sizePolicy().horizontalPolicy()
            == QSizePolicy.Policy.Ignored,
            "the bottom line lets the window shrink instead of pinning its width",
        )
    # And it must WRAP: at the minimum window size a one-line status bar clipped
    # the tail of the sentence mid-glyph.  So it is a wrapping label in the body
    # layout (which honours height-for-width), not an item in a QStatusBar.
    check(triple_window._status.wordWrap(), "the bottom line wraps instead of clipping")
    check(
        triple_window._status.sizePolicy().hasHeightForWidth(),
        "the layout is told the wrapped height, so the second line is not cut off",
    )
    check(
        triple_window.findChild(QStatusBar) is None,
        "the window has no one-line status bar to clip the sentence",
    )

    # The sentence is about the colour table, so 图像转换 must not show it.
    tab_note = triple_window._status.text()
    triple_window._tabs.setCurrentIndex(1)
    app.processEvents()
    check(
        triple_window._status.text() == "",
        f"the colour-table sentence is hidden on 图像转换, got {triple_window._status.text()[:40]!r}",
    )
    # An empty bottom line must not keep a row of height open either (m14748 第 3 条):
    # 空着的那一行高度以前全变成窗口底部的白边，收起来之后省下的高度归上面的分栏。
    check(
        not triple_window._status.isVisibleTo(triple_window),
        "an empty bottom line hides itself instead of holding an empty row open",
    )
    check(
        triple_window._status.sizeHint().height() > 0,
        "hiding it is a visibility decision, not a zero-height trick",
    )
    triple_window._tabs.setCurrentIndex(0)
    app.processEvents()
    check(
        triple_window._status.text() == tab_note,
        "switching back to 颜色配方 restores the colour-table sentence",
    )
    check(
        triple_window._status.isVisibleTo(triple_window),
        "and the line comes back into the layout with it",
    )

    # The first page is 「颜色配方」 now, all the way through: tab, panel title and
    # the tooltip on the target colour all name the same thing.
    check(
        triple_window._tabs.tabText(0) == "颜色配方",
        f"the first tab is 颜色配方, got {triple_window._tabs.tabText(0)!r}",
    )
    check(
        triple_window._detail._title.text() == "颜色配方",
        f"the right-hand panel is 颜色配方, got {triple_window._detail._title.text()!r}",
    )
    # The hover pop-ups are gone on purpose (the black box the user asked to
    # delete), so nothing on this window may carry a tooltip any more.
    check(
        triple_window._target_color.toolTip() == "",
        "the target colour has no tooltip any more",
    )
    check(
        triple_window._find_button.toolTip() == "",
        "找最接近的颜色 has no tooltip any more",
    )

    # The version has to be readable without opening 帮助: it is the same string
    # the exe's file properties carry (build/BambuPalette.spec parses app/__init__.py).
    from app import __version__ as _version

    about = triple_window._about_text()
    check(
        _version in about,
        f"帮助 names the running version {_version!r}",
    )
    check(
        about.startswith("BambuPalette"),
        "帮助 still opens with the product name",
    )
    check(
        "BambuPalette.exe 的文件属性里也写着同一个版本号。" in about,
        "帮助 points at the exe's file properties",
    )
    # m01885 #5/#6/#8: 「关于」改叫「帮助」，点开是一个装着整本说明的窗口。
    from app.core.image_matching import plan_help as _plan_help

    check(
        triple_window._help_button.text() == "帮助",
        f"右上角是「帮助」按钮，got {triple_window._help_button.text()!r}",
    )
    manual = triple_window._help_text()
    check(
        triple_window._picture_page.help_text() in manual and _plan_help() in manual and about in manual,
        "帮助窗口 = 图像转换说明 + 方案说明 + 版本与算法出处",
    )
    check(
        not hasattr(triple_window._picture_page, "_page_hint"),
        "图像转换页面上不再压着那段说明（1.6.14 #6）",
    )
    # m10238 item 1: and it must be visible without opening any dialog at all.
    check(
        _version in triple_window.windowTitle(),
        f"the title bar carries the version, got {triple_window.windowTitle()!r}",
    )

    # A spool's own colour must hide EVERY parent row, not just the second one:
    # row 3 kept the 「耗材丝3 … 12%」 text from the previous three-colour recipe,
    # so a raw spool looked like it was mixed from two unrelated spools.
    spool_cell = next(
        (cell for cell in triple_window._recipes if cell.filaments and not cell.recipes), None
    )
    if spool_cell is not None:
        triple_window._on_grid_selected(spool_cell)
        app.processEvents()
        rows = triple_window._detail._rows
        # isHidden(), not isVisible(): this window is never shown, so isVisible()
        # is False for every row and would prove nothing.
        check(
            [row["frame"].isHidden() for row in rows] == [False] + [True] * (len(rows) - 1),
            "a spool colour shows exactly one parent row",
        )
        check(
            rows[0]["percent"].text().startswith("100%"),
            f"the one visible row is the 100% spool, got {rows[0]['percent'].text()!r}",
        )
    triple_window.close()

    # --- the grid's flat layout, exercised with a stub -------------------------
    # A real 200,000-colour table would cost seconds to build, so the switch to
    # arithmetic rows is proved against a stand-in lazy sequence instead.
    from PySide6.QtCore import QPoint  # noqa: E402

    from app.ui import mix_grid as _grid_module  # noqa: E402

    class _StubColour:
        __slots__ = (
            "index",
            "rgb",
            "color_hex",
            "key",
            "pair_index",
            "ratio_text",
            "recipes",
            "a_id",
            "b_id",
            "percent_a",
            "percent_b",
            "lab",
            "lightness",
            "hue",
            "chroma",
        )

        def __init__(self, index: int) -> None:
            self.index = index
            self.rgb = (index & 0xFF, (index >> 8) & 0xFF, (index >> 16) & 0xFF)
            self.color_hex = f"#{index:06X}"
            self.key = f"triple|{self.color_hex}"
            self.pair_index = -1
            self.ratio_text = "1 个配方"
            self.recipes = ()
            self.a_id = library[0].id
            self.b_id = library[1].id
            self.percent_a = 50
            self.percent_b = 50
            self.lab = (50.0, 0.0, 0.0)
            self.lightness = 50.0
            self.hue = 0.0
            self.chroma = 0.0

    class _StubView:
        def __init__(self, count: int) -> None:
            self._count = count

        def __len__(self) -> int:
            return self._count

        def __getitem__(self, index):
            if isinstance(index, slice):
                return [self[i] for i in range(*index.indices(self._count))]
            index = int(index)
            if index < 0:
                index += self._count
            if not 0 <= index < self._count:
                raise IndexError(index)
            return _StubColour(index)

        def index_of_key(self, key: str) -> int:
            text = key[7:] if key.startswith("triple|") else key
            try:
                return int(text.lstrip("#"), 16)
            except ValueError:
                return -1

    stub = _StubView(200_000)
    window._grid.resize(660, 400)
    window._grid.setRecipes(stub, grouped=False)
    app.processEvents()
    # The viewport still carries the scroll offset of whatever was shown before,
    # and _indexAt works in content coordinates.
    window._grid.verticalScrollBar().setValue(0)
    app.processEvents()
    check(window._grid._flat, "the grid switches to the flat layout above LAZY_LIMIT")
    check(window._grid._recipes is stub, "a huge lazy sequence is kept, not copied")
    check(
        not isinstance(window._grid._recipes, list),
        "no per-cell Python list is built for it",
    )
    index = window._grid._indexAt(
        QPoint(3 * _grid_module.CELL + 5, 7 * _grid_module.CELL + 5)
    )
    check(
        index == 7 * window._grid._columns + 3,
        f"a click still maps to the right cell (got {index})",
    )
    target = stub[index]
    window._grid.selectRecipe(target)
    check(
        window._grid.selectedRecipe().key == target.key,
        "selecting a lazy cell does not scan the whole table",
    )
    check(
        window._grid.indexOfKey("triple|#FFFFFF") == 0xFFFFFF,
        "a key maps back to its position without a scan",
    )
    window._refresh_grid()
    app.processEvents()

    # --- the export carries the mixes, and only the right library may adopt them
    import tempfile  # noqa: E402

    from app.core.library import Filament as _Filament  # noqa: E402
    from app.core.library import FilamentLibrary as _Library  # noqa: E402

    sidecar_dir = Path(tempfile.mkdtemp(prefix="bp-sidecar-"))
    archive = sidecar_dir / "lib.json"
    check(window._write_mix_sidecar(archive) is None, "a plain export writes no mix cache")
    triple_window = MainWindow(library=FilamentLibrary(list(library.filaments)))
    deadline = time.time() + 120
    while triple_window._triple_timer is not None and time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    written = triple_window._write_mix_sidecar(archive)
    check(
        written is not None and written.is_file() and written.name == "lib.mixes.npz",
        "once the three-filament table is built, the export carries the cache",
    )
    if written is not None:
        import numpy as np  # noqa: E402

        with np.load(written) as data:
            check("key" in data.files, "the cache records which library it belongs to")

        other_library = _Library()
        for position, filament in enumerate(library):
            other_library.add(
                _Filament(
                    id=filament.id,
                    brand=filament.brand,
                    material_type=filament.material_type,
                    color_hex="#010203" if position == 0 else filament.color_hex,
                )
            )
        other = MainWindow(library=other_library, auto_triples=False)
        check(
            not other._adopt_mix_sidecar(written),
            "a different library refuses a foreign mix cache",
        )
        other.close()

        # The same spools must pick the mixes up instead of recomputing them.
        fresh_library = _Library()
        for filament in library:
            fresh_library.add(filament)
        fresh = MainWindow(library=fresh_library, auto_triples=False)
        check(
            fresh._triples is None,
            "a window with the build suppressed starts without the mixes",
        )
        check(fresh._adopt_mix_sidecar(written), "the same spools adopt the exported cache")
        check(
            fresh._triples is not None and fresh._triples.built,
            "adopting it lands the window with a usable triple table",
        )
        check(
            len(fresh._recipes) == len(triple_window._combined),
            "the adopted table has exactly as many colours as the one that built it",
        )
        fresh.close()

        # --- item 3 end to end: importing an archive that carries its own mixes -
        # m00041 item 3: 「导入文件中包含了混色，却提示说还在计算，而且我的鼠标一直在
        # 转圈」.  The old _on_import started the three-filament build first and only
        # then noticed the sidecar, so the timer kept stepping a catalogue with no
        # working array while the wait cursor was never handed back.  This drives
        # the real import action, not _adopt_mix_sidecar.
        import app.ui.main_window as _mw  # noqa: E402
        from PySide6.QtWidgets import QApplication as _QApp  # noqa: E402

        from app.core import paths as _paths  # noqa: E402

        library.save(archive)
        # A cache directory of its own: the file this run already wrote for these
        # five spools would otherwise answer from the ordinary cache path and the
        # guard under test would never be asked anything.
        scratch = sidecar_dir / "import-scratch"
        (scratch / "cache").mkdir(parents=True, exist_ok=True)
        saved_data_dir = _paths.data_dir
        saved_chooser = _mw.QFileDialog
        saved_box = _mw.QMessageBox
        _paths.data_dir = lambda: scratch  # type: ignore[assignment]
        _mw.QFileDialog = _StubChooser(archive)
        _mw.QMessageBox = _StubMessageBox
        importer = MainWindow(library=_Library())
        try:
            importer._on_import()
            check(
                importer._triples is not None and importer._triples.built,
                "importing an archive with a cache lands with the mixes ready",
            )
            check(
                importer._triple_timer is None,
                "the three-filament timer is not left stepping a table with no workspace",
            )
            check(
                not importer._build_notice,
                "an import that brings its own cache leaves no build notice behind",
            )
            check(
                _QApp.overrideCursor() is None,
                "the wait cursor is handed back once the imported cache is in",
            )
            check(
                importer._pending_sidecar is None,
                "the pending sidecar is cleared after it has been used",
            )
            check(
                "正在计算三色混色" not in importer._status.text(),
                f"the status line does not claim to be computing, got {importer._status.text()!r}",
            )

            # Positive control: the same archive with no cache beside it must
            # really start the build, otherwise those checks pass for free.
            plain_scratch = sidecar_dir / "import-scratch-plain"
            (plain_scratch / "cache").mkdir(parents=True, exist_ok=True)
            _paths.data_dir = lambda: plain_scratch  # type: ignore[assignment]
            no_cache = sidecar_dir / "no-cache.json"
            no_cache.write_bytes(archive.read_bytes())
            _mw.QFileDialog = _StubChooser(no_cache)
            plain = MainWindow(library=_Library())
            try:
                plain._on_import()
                check(
                    plain._triple_timer is not None,
                    "without a cache beside it the import does start the three-filament build",
                )
            finally:
                plain._stop_triple_timer()
                plain._end_build_notice()
                plain.close()
        finally:
            _paths.data_dir = saved_data_dir  # type: ignore[assignment]
            _mw.QFileDialog = saved_chooser
            _mw.QMessageBox = saved_box
            importer.close()
    triple_window.close()

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
    check(len(window._recipes) == everything, "restoring the full catalogue after the blank click")

    # --- 全部颜色: mixes and raw spools in ONE list, with no checkboxes --------
    # m08102 item 2: rename 全部混色 to 全部颜色, fold the three-filament colours
    # and the spool colours in automatically, and DELETE both checkboxes —
    # ticking 三色混色 used to make 全部颜色 impossible to tick.
    from PySide6.QtWidgets import QLabel  # noqa: E402

    check(
        not hasattr(window, "_all_colours") and not hasattr(window, "_triples_box"),
        "the 全部颜色 / 三色混色 checkboxes are gone",
    )
    check(
        [w for w in window.findChildren(QLabel) if w.text() == "全部颜色"],
        "the panel is titled 全部颜色",
    )
    check(
        not [w for w in window.findChildren(QLabel) if w.text() == "全部混色"],
        "no 全部混色 heading is left anywhere",
    )
    check(expected_recipe_count(2) == 81, "two spools make 81 mixes")
    check(expected_recipe_count(2) + 2 == 83, "81 mixes + the 2 raw spools = the 83 colours asked for")

    check(
        len(window._recipes) == everything,
        f"the grid is the whole combined list, got {len(window._recipes)} for {everything}",
    )
    spools = [cell for cell in window._recipes if cell.filaments]
    check(len(spools) >= 1, "the spools' own colours are in the very same list")
    check(
        len([cell for cell in window._recipes if not cell.filaments]) >= merged,
        "every merged mix colour is in there too",
    )
    check(
        all(cell.color_hex == cell.filaments[0].color_hex for cell in spools),
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
    # The same control must order both halves, so an explicit colour sort has to
    # move the spools too rather than leaving them frozen at the top.
    from app.spectral import color as spectral_color  # noqa: E402

    window._sort_combo.setCurrentIndex(
        [key for key, _ in SORT_CHOICES].index("lightness")
    )
    app.processEvents()
    light_spools = [c for c in window._recipes if c.filaments]
    levels = [spectral_color.lab_from_rgb(c.rgb)[0] for c in light_spools]
    check(levels == sorted(levels, reverse=True), "排序 by lightness reorders the spool rows too")
    window._sort_combo.setCurrentIndex([key for key, _ in SORT_CHOICES].index("rgb"))
    app.processEvents()

    # Selecting a spool-only colour must explain itself rather than pretend to be
    # a mix; a spool colour that IS also a mixture is a colour like any other.
    pure = [cell for cell in spools if not cell.recipes]
    check(pure, "at least one spool colour is produced by no mixture at all")
    window._grid.selectRecipe(pure[0])
    window._on_grid_selected(pure[0])
    app.processEvents()
    check(window._detail.recipe() is None, "a spool-only colour is not shown as a mix recipe")
    check(window._detail.colour() is pure[0], "the detail panel shows that colour")
    check(window._detail.selectionKey() == pure[0].key, "the detail panel keys it by colour")
    check(window.nearest_recipe("#982C29") is not None, "找最接近的 still works with spools listed")

    window._search.setText("大简")
    app.processEvents()
    check(
        len(window._recipes) == everything,
        "a brand search keeps every spool row too",
    )
    window._search.clear()
    app.processEvents()
    check(len(window._recipes) == everything, "clearing the search restores the combined list")
    window._detail.clear()

    # --- a one-spool edit reuses the catalogue instead of rebuilding it --------
    before_catalog = window._catalog
    before_recipes = {m.key: m for m in before_catalog.recipes}
    extra = library.create(brand="大简", material_type="PETG HF", color_hex="#7F3FBF", name="测试紫")
    started = time.perf_counter()
    window._reload_library()
    add_seconds = time.perf_counter() - started
    check(
        window._catalog is before_catalog,
        "adding a spool reuses the catalogue object instead of replacing it",
    )
    check(
        window._catalog.recipe_count == expected_recipe_count(6),
        f"the new spool's pairs are folded in, got {window._catalog.recipe_count}",
    )
    reused = sum(1 for key, recipe in before_recipes.items()
                 if window._catalog.find_recipe(*_key_parts(key)) is recipe)
    check(reused == 810, f"every pre-existing recipe object survives the add, got {reused}")
    check(
        add_seconds < 1.0,
        f"a one-spool add is not a full rebuild, took {add_seconds:.3f}s",
    )

    library.remove(extra.id)
    started = time.perf_counter()
    window._reload_library()
    remove_seconds = time.perf_counter() - started
    check(
        window._catalog.recipe_count == expected_recipe_count(5),
        f"deleting the spool takes its pairs back out, got {window._catalog.recipe_count}",
    )
    check(
        remove_seconds < 1.0,
        f"a one-spool delete is not a full rebuild, took {remove_seconds:.3f}s",
    )
    check(
        len(window._recipes) == window._combined.colour_count,
        "the grid is back to the combined count after the round trip",
    )

    # --- no preset colour, no delete prompt, Del deletes -----------------------
    from PySide6.QtGui import QKeySequence  # noqa: E402
    from PySide6.QtWidgets import QMessageBox, QSpinBox  # noqa: E402

    from app.ui.filament_dialog import FilamentDialog  # noqa: E402

    add_dialog = FilamentDialog(None, [f.brand for f in library.filaments], [], window)
    # m09456 item 1 (1.6.28): 用户要的就是反过来 —— 打开就写着 0,0,0，直接能确认。
    # （m01647 item 1 当年要的是「不要预设颜色」，1.6.28 按用户的新要求改回来。）
    check(add_dialog._color.isSet(), "添加耗材 opens on a colour")
    check(
        add_dialog.values()["color_hex"] == "#000000",
        f"a fresh dialog starts on black, got {add_dialog.values()['color_hex']!r}",
    )
    check(add_dialog._ok_button.isEnabled(), "确认 is usable straight away")
    check(add_dialog._next_button.isEnabled(), "下一个 is usable straight away")
    check(
        [spin.text() for spin in add_dialog._color.findChildren(QSpinBox)] == ["0", "0", "0"],
        "the RGB boxes show the default 0, 0, 0",
    )
    # 数字框里直接敲 0：值就停在最小值上，valueChanged 不响，只有 textEdited 能接住。
    three = add_dialog._color._spins[0].lineEdit()
    three.setText("0")
    three.textEdited.emit("0")
    check(
        add_dialog._color.hex() == "#000000",
        f"typing 0 into a channel keeps black (m09456 第 1 条), got {add_dialog._color.hex()!r}",
    )
    check(
        "0, 0, 0" in add_dialog._hint.text() or "L*" in add_dialog._hint.text(),
        f"the hint describes the colour that will be saved, got {add_dialog._hint.text()!r}",
    )
    add_dialog._color.setValue("#C8342E")
    check(add_dialog._ok_button.isEnabled(), "确认 comes alive once the user enters a colour")
    check(add_dialog._next_button.isEnabled(), "下一个 comes alive too")
    check(add_dialog.values()["color_hex"] == "#C8342E", "the typed colour is what the spool gets")
    check(
        [spin.text() for spin in add_dialog._color.findChildren(QSpinBox)] == ["200", "52", "46"],
        "the RGB boxes show the chosen colour",
    )
    # A colour whose channels are 0 must still print "0", not a blank.
    add_dialog._color.setValue("#000000")
    check(
        [spin.text() for spin in add_dialog._color.findChildren(QSpinBox)] == ["0", "0", "0"],
        "pure black still shows 0 in every channel",
    )
    add_dialog._color.setValue("#C8342E")
    # The colour is entered on the panel itself — there is no colour-picker dialog.
    check(
        add_dialog._color.findChild(QSpinBox) is not None,
        "the colour field offers R / G / B spin boxes",
    )
    # 名称 has no example placeholder any more (m01647 item 7).
    check(
        add_dialog._name.placeholderText() == "",
        f"名称 carries no example text, got {add_dialog._name.placeholderText()!r}",
    )
    # No hover pop-ups anywhere in this dialog (m01647 item 4).
    check(
        add_dialog._next_button.toolTip() == "" and add_dialog._name.toolTip() == "",
        "the add dialog has no tooltips left",
    )
    # m10238 item 3: 下一个 saves this spool and reopens an empty form, so adding
    # ten spools is ten dialogs deep, not one dialog and nine trips to 添加耗材.
    check(not add_dialog.wants_another(), "确认 alone does not ask for another spool")
    from PySide6.QtWidgets import QHBoxLayout as _DialogRow

    button_row = next(
        layout
        for layout in add_dialog.findChildren(_DialogRow)
        if layout.indexOf(add_dialog._ok_button) >= 0
    )
    check(
        0 <= button_row.indexOf(add_dialog._next_button) == button_row.indexOf(add_dialog._ok_button) - 1,
        "下一个 sits immediately left of 确认",
    )
    add_dialog._next_button.click()
    check(add_dialog.wants_another(), "下一个 asks for another spool")
    check(
        add_dialog.result() == FilamentDialog.DialogCode.Accepted,
        "下一个 still keeps the spool the user just typed",
    )
    check(
        add_dialog.build_filament().color_hex == "#C8342E",
        "the spool kept by 下一个 is the typed colour",
    )
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

    # --- one spool: its own colour must still reach the middle grid -------------
    lone = FilamentLibrary([Filament(name="只有一种", material_type="PLA", color_hex="#0000FF")])
    lone_window = MainWindow(library=lone)
    lone_window.resize(1200, 800)
    lone_window.show()
    app.processEvents()
    check(lone_window._catalog.recipe_count == 0, "one spool really does make zero mixes")
    check(
        len(lone_window._recipes) == 1,
        f"a lone spool's own colour still reaches the grid, got {len(lone_window._recipes)}",
    )
    if lone_window._recipes:
        only = lone_window._recipes[0]
        check(only.filaments and only.filaments[0].id == lone[0].id, "the lone cell is that spool")
        check(only.color_hex == "#0000FF", "the lone cell carries the spool's colour")
        lone_window._grid.selectRecipe(only)
        lone_window._on_grid_selected(only)
        app.processEvents()
        check(
            lone_window._detail.colour() is only,
            "clicking the lone spool explains it in the detail panel",
        )
    check(len(lone_window._recipes) > 0, "the empty-state message is not what a lone spool gets")
    lone_window.close()

    # --- 材质开关 (m05557 item 1) -------------------------------------------------
    # 两种塑料不能混进同一根喷嘴：切换材质＝换一份配色清单，Every page follows.
    material_mix = FilamentLibrary(
        [
            Filament(name="A-PETG", brand="大简", material_type="PETG HF", color_hex="#F5D2B8"),
            Filament(name="B-PETG", brand="大简", material_type="PETG HF", color_hex="#C52A36"),
            Filament(name="C-PLA", brand="b", material_type="PLA", color_hex="#2C2A2C"),
            Filament(name="D-PLA", brand="b", material_type="PLA", color_hex="#FF6C2B"),
            # 只有一卷的材料：切过来以后两种料的混色必须一个都不剩 (m10318 item 1).
            Filament(name="E-ABS", brand="b", material_type="ABS", color_hex="#101010"),
        ]
    )
    material_window = MainWindow(library=material_mix, auto_triples=False)
    material_window.resize(1200, 800)
    material_window.show()
    app.processEvents()
    picker = material_window._material_combo
    check(
        [picker.itemText(index) for index in range(picker.count())]
        == ["PETG HF（2）", "PLA（2）", "ABS（1）"],
        "the 材质 picker lists one entry per material, with its spool count",
    )
    check(
        material_window.library.material == "PETG HF",
        f"the picker starts on the biggest material, got {material_window.library.material!r}",
    )
    petg_hexes = {recipe.color_hex.upper() for recipe in material_window._recipes}
    check({"#F5D2B8", "#C52A36"} <= petg_hexes, "PETG's two spools colour the grid to start")

    picker.setCurrentIndex(picker.findData("PLA"))
    app.processEvents()
    pla_hexes = {recipe.color_hex.upper() for recipe in material_window._recipes}
    check(
        material_window.library.material == "PLA" and len(material_window.library.visible) == 2,
        "choosing 材质 PLA switches the spool list the colours come from",
    )
    check(
        not ({"#F5D2B8", "#C52A36"} & pla_hexes) and {"#2C2A2C", "#FF6C2B"} <= pla_hexes,
        f"every PETG colour left the grid and PLA's arrived, got {sorted(pla_hexes)[:4]}",
    )
    rows = [material_window._list.item(i).text() for i in range(material_window._list.count())]
    expected_rows = sum(1 for filament in material_window.library if material_window.library.accepts(filament))
    check(
        material_window._list.count() == expected_rows,
        f"我的耗材 lists only the material in force, got {material_window._list.count()} rows",
    )
    check(
        all("现在不配色" not in row for row in rows) and not any("#F5D2B8" in row for row in rows),
        f"the other material's spools leave the list entirely (m09456 第 2 条), got {rows}",
    )
    check(
        material_window.library.accepts(Filament(name="E-", brand="b", material_type=" PLA ", color_hex="#FFFFFF")),
        "a trailing space in 耗材种类 does not hide the spool",
    )
    material_window._list.setCurrentRow(0)
    app.processEvents()
    check(
        not hasattr(material_window, "_spell_out") and "#F5D2B8 · " not in material_window._status.text(),
        f"picking a spool no longer spells it out on the bottom line (m09456 第 3 条), got {material_window._status.text()[:60]!r}",
    )
    check(
        material_window.library.hidden_count == 0,
        "switching 材质 never hides a spool (hiding is the user's own act)",
    )
    check(
        "现在按 PLA 配色" in material_window._status.text(),
        f"the status line says which material is in force, got {material_window._status.text()[:50]!r}",
    )

    # 换到只有一卷的材质：两色表要归零，「全部颜色」只剩这一卷的本色 (m10318 item 1).
    # sync() 曾经拿「旧的那批卷子」当新卷子的另一半，于是切过去的每一次都留下一对
    # 幽灵配方（用户库里是 42 卷：切到 1 卷的 PETG 后还剩 942 个颜色）。
    picker.setCurrentIndex(picker.findData("ABS"))
    app.processEvents()
    single = material_window
    check(
        single._catalog is not None
        and single._catalog.pair_count == 0
        and single._catalog.recipe_count == 0,
        "a one-spool material leaves no pair behind, got "
        f"{single._catalog.pair_count if single._catalog else '—'} pairs / "
        f"{single._catalog.recipe_count if single._catalog else '—'} recipes",
    )
    check(
        len(single._combined) == 1,
        f"全部颜色 holds the spool itself and nothing else, got {len(single._combined)}",
    )
    check(
        len(single._recipes) == 1,
        f"the grid draws exactly one cell for a one-spool material, got {len(single._recipes)}",
    )
    check(
        "去重后共 1 个颜色" in single._status.text(),
        f"the bottom line counts one colour, got {single._status.text()[:80]!r}",
    )
    picker.setCurrentIndex(picker.findData("PETG HF"))
    app.processEvents()
    check(
        single._catalog.pair_count == 1 and single._catalog.recipe_count == 81,
        "switching back rebuilds the surviving material's own pairs, got "
        f"{single._catalog.pair_count} pairs / {single._catalog.recipe_count} recipes",
    )
    check(
        len(single._combined) == 83,
        f"switching back cannot resurrect the ABS spool, got {len(single._combined)} colours",
    )

    # 一种材质下的卷被删光：这种材质要从下拉里消失，材质切回第一种，并且说出来
    # (m10789 item 1)。以前下拉悄悄落回第一种、状态栏却还印着被删掉的那种名字，
    # 设置里也还记着它 —— 用户看到的就是「材质无法切换了」。
    picker.setCurrentIndex(picker.findData("PLA"))
    app.processEvents()
    check(
        material_window.library.material == "PLA",
        f"the picker can still go back to another material, got {material_window.library.material!r}",
    )
    for _round in range(2):
        material_window._list.setCurrentRow(0)
        app.processEvents()
        material_window._on_delete()
        app.processEvents()
    check(
        material_window.library.material == "PETG HF",
        "deleting a material's last spool falls back to the first material, got "
        f"{material_window.library.material!r}",
    )
    check(
        [picker.itemText(index) for index in range(picker.count())] == ["PETG HF（2）", "ABS（1）"],
        "an emptied material leaves the 材质 picker, got "
        f"{[picker.itemText(index) for index in range(picker.count())]}",
    )
    check(
        picker.isEnabled() and picker.currentData() == "PETG HF",
        "the picker still works after the material it was on disappeared",
    )
    check(
        "「PLA」下的耗材已经全部删除" in material_window._status.text()
        and "已切回「PETG HF」" in material_window._status.text(),
        "the bottom line says why the material changed, got "
        f"{material_window._status.text()[:70]!r}",
    )
    check(
        "PLA" not in material_window._library_hint.text(),
        "the empty hint no longer names the deleted material, got "
        f"{material_window._library_hint.text()[:50]!r}",
    )
    check(
        len(material_window._combined) == 83 and material_window._catalog.pair_count == 1,
        "the surviving material took the colours over, got "
        f"{len(material_window._combined)} colours / {material_window._catalog.pair_count} pairs",
    )
    material_window.close()
    app.processEvents()

    # --- the launcher defers the first catalogue build ---------------------------
    # 41 spools is 66,420 recipes and ~3 s; doing that inside __init__ made the
    # window take 4.4 s to appear.  The real launcher passes defer_build=True.
    deferred = MainWindow(library=library, defer_build=True, auto_triples=False)
    deferred.resize(1200, 800)
    deferred.show()
    check(
        deferred._catalog is None and not deferred._recipes,
        "a deferred window comes up before the catalogue exists",
    )
    check(
        "正在计算" in deferred._status.text(),
        f"it says what it is doing, got {deferred._status.text()!r}",
    )
    check(
        deferred._build_notice and app.overrideCursor() is not None,
        "a deferred build shows a wait cursor",
    )
    check(
        deferred._grid._empty_text == "正在计算混色表…",
        f"the grid stops blaming the library mid-build, got "
        f"{deferred._grid._empty_text!r}",
    )
    app.processEvents()
    check(
        deferred._catalog is None,
        "the build waits for the window to paint, it is not on a zero timer",
    )
    deadline = time.perf_counter() + 30.0
    while deferred._catalog is None and time.perf_counter() < deadline:
        app.processEvents()
        time.sleep(0.01)
    check(
        deferred._catalog is not None and deferred._catalog.recipe_count == 810,
        f"the delayed build fills the catalogue, got "
        f"{None if deferred._catalog is None else deferred._catalog.recipe_count}",
    )
    check(
        len(deferred._recipes) == deferred._combined.colour_count,
        "and the grid fills in with the merged colours",
    )
    check(not deferred._build_notice, "the wait cursor is taken back down")
    check(app.overrideCursor() is None, "no override cursor is left behind")
    check(
        deferred._grid._empty_text == "还没有混色。请先添加至少两种耗材。",
        "the grid's own empty state comes back after the build",
    )
    check("正在计算" not in deferred._status.text(), "the notice is replaced by the real status")
    # The synchronous default still holds for every test and headless tool.
    check(not window._defer_build, "an ordinary window builds straight away")
    deferred.close()

    # --- an unnamed spool must not print its hex twice --------------------------
    unnamed = FilamentLibrary([Filament(color_hex="#000000")])
    unnamed_window = MainWindow(library=unnamed)
    app.processEvents()
    row_text = unnamed_window._list.item(0).text()
    check(
        not row_text.strip().endswith("#000000   #000000"),
        f"an unnamed spool does not repeat its hex, got {row_text!r}",
    )
    check("RGB 0, 0, 0" in row_text, f"the duplicate hex becomes the RGB numbers, got {row_text!r}")
    check(
        not row_text.startswith("#000000   #000000"),
        f"an unnamed spool does not repeat its hex at the front either, got {row_text!r}",
    )
    unnamed_window.close()

    # --- the colour is typed in, never picked from a dialog ---------------------
    from app.ui import swatch as swatch_module  # noqa: E402
    from app.ui.swatch import ColorField  # noqa: E402

    check(not hasattr(swatch_module, "ColorButton"), "the old colour-picker button is gone")
    check(not hasattr(swatch_module, "picker_start"), "the old picker helper is gone")
    check(not hasattr(swatch_module, "QColorDialog"), "no colour dialog is imported any more")
    blank = ColorField(None)
    check(not blank.isSet() and blank.hex() == "", "a colour field starts empty")
    check(blank.rgb() == (0, 0, 0), "an empty colour field reads as no colour, not white")
    blank._hex.setText("#c8342e")
    blank._on_hex_edited()
    check(blank.hex() == "#C8342E", f"a lower-case hex is uppercased, got {blank.hex()!r}")
    blank._hex.setText("nonsense")
    blank._on_hex_edited()
    check(blank.hex() == "#C8342E", "gibberish leaves the last good colour alone")
    blank._spins[2].setValue(255)
    blank._on_spin()
    check(blank.hex() == "#C834FF", f"the B spin box drives the colour, got {blank.hex()!r}")

    # m06704 item 1 (1.6.27): 打完 #RRGGBB 直接点「确认」时输入框还没失焦，界面就当
    # 没填颜色（确认按钮是灰的，灰按钮又不接焦点，editingFinished 永远等不到）。现在
    # 每敲一位就提交一次；三位简写要等到失焦才展开 —— 敲到 "#F5D" 时先按 #FF55DD 读
    # 会把后面的两位接不上去。
    typed = ColorField(None)
    typed._hex.setText("#F5D2B8")
    typed._on_hex_typed("#F5D2B8")
    check(
        typed.isSet() and typed.hex() == "#F5D2B8",
        f"typing a full hex commits it without leaving the field, got {typed.hex()!r}",
    )
    typed._hex.setText("#8cf")
    typed._on_hex_typed("#8cf")
    check(
        typed.hex() == "#F5D2B8",
        f"a three-digit shorthand is not read while it is still being typed, got {typed.hex()!r}",
    )
    typed._on_hex_edited()
    check(
        typed.hex() == "#88CCFF",
        f"it expands once the field is left, got {typed.hex()!r}",
    )
    pending = ColorField(None)
    pending._hex.setText("#123456")
    pending.commitPending()
    check(
        pending.isSet() and pending.hex() == "#123456",
        f"commitPending() reads the typed text without waiting for focus loss, got {pending.hex()!r}",
    )

    # m09456 item 1 (1.6.28): 数字框里直接敲 0 —— 值本来就停在最小值 0 上，valueChanged 一点
    # 动静都没有，用户看到的就是「0 打不进去」。
    zeros = ColorField(None)
    zeros._spins[0].lineEdit().textEdited.emit("0")
    check(
        zeros.isSet() and zeros.hex() == "#000000",
        f"typing 0 into a channel commits black, got {zeros.hex()!r}",
    )
    zeros._spins[1].setValue(0)
    second = zeros._spins[1].lineEdit()
    second.setText("0")
    second.textEdited.emit("0")
    check(zeros.hex() == "#000000", "typing the value a channel already has still commits")

    from app.ui.filament_dialog import FilamentDialog  # noqa: E402

    fresh = FilamentDialog(None)
    check(
        fresh._color.isSet() and fresh._color.hex() == "#000000",
        f"a new spool starts on black, got {fresh._color.hex()!r}",
    )
    check(fresh._ok_button.isEnabled(), "a new spool already has a usable 确认")
    fresh.close()

    # --- nearest match --------------------------------------------------------
    nearest = window.nearest_recipe("#982C29")
    check(nearest is not None, "nearest_recipe returns something for #982C29")
    if nearest is not None:
        check(len(nearest.color_hex) == 7, f"nearest recipe has a hex, got {nearest.color_hex!r}")
    # m02366 item 1 (1.6.17): the target colour starts PRE-SET to 000.  It used to
    # start blank, but a blank field still reads (0, 0, 0) — so 找最接近的颜色
    # quietly searched for black while the box looked empty and the user had no
    # way to see what was being searched for.
    check(window._target_color.hex() == "#000000", "the 目标颜色 field starts at 000")
    check(
        window._target_color.isSet() and window._target_color.rgb() == (0, 0, 0),
        "the target colour starts as a real, visible #000000",
    )
    # Deliberately clickable with nothing chosen, exactly like 确认 in 添加耗材:
    # pressing it early explains what is missing instead of being greyed out.
    check(window._find_button.isEnabled(), "找最接近的颜色 can be pressed right away")
    # With a colour highlighted in the table, the borrow comes from THERE — that is
    # the colour the user is actually looking at.
    window._target_color.clear()
    borrowed = window._recipes[len(window._recipes) // 2]
    window._grid.selectRecipe(borrowed)
    app.processEvents()
    window._find_button.click()
    app.processEvents()
    # m10238 item 2: it must SEARCH, not print an instruction.  With no target it
    # borrows the colour the user is looking at and says so.
    check(
        "目标颜色取自你正看着的" in window._status.text(),
        f"an empty target borrows the highlighted colour: {window._status.text()!r}",
    )
    check(
        borrowed.color_hex.upper() in window._status.text().upper(),
        f"the borrowed target is the colour in the table: {window._status.text()!r}",
    )
    check(
        "最接近" in window._status.text(),
        f"and the borrow really answers a colour: {window._status.text()!r}",
    )
    check(
        window._target_color.hex().upper() == borrowed.color_hex.upper(),
        f"the borrowed colour is shown in the 目标颜色 field, got {window._target_color.hex()!r}",
    )
    # Nothing highlighted and no target typed: the R / G / B numbers are the only
    # thing left to aim at, so they are used.
    window._target_color.clear()
    window._grid.clearSelection()
    app.processEvents()
    window._find_button.click()
    app.processEvents()
    check(
        "目标颜色取自数字框里的 RGB" in window._status.text(),
        f"with nothing picked it falls back to the R/G/B numbers: {window._status.text()!r}",
    )
    window._target_color.setValue("#808080")
    check(window._find_button.isEnabled(), "找最接近的颜色 stays enabled once a colour is entered")
    window._on_find_nearest()
    app.processEvents()
    check(window._detail.recipe() is not None, "找最接近的颜色 selects a recipe")

    # Issue 3: the search used to skip the spool colours themselves, so asking
    # for a colour the user already owned answered a WORSE mixture (asked for
    # #FFFFFF, answered #E5E5E5 at ΔE00 5.40).  The true nearest now wins.
    from app.spectral import color as _nearest_color  # noqa: E402

    def _gap(cell, target_hex: str) -> float:
        lab = _nearest_color.lab_from_rgb(_nearest_color.hex_to_rgb(target_hex))
        return _nearest_color.delta_e_2000(cell.lab, tuple(lab))

    for spool in list(window.library.filaments)[:3]:
        answer = window.nearest_recipe(spool.color_hex)
        check(answer is not None, f"{spool.color_hex} is findable")
        if answer is not None:
            check(
                _gap(answer, spool.color_hex) <= 1e-6,
                f"asking for the spool colour {spool.color_hex} answers an exact match, "
                f"got {answer.color_hex} at ΔE00 {_gap(answer, spool.color_hex):.2f}",
            )
    _spools = list(window.library.filaments)
    if len(_spools) >= 2:
        _sample = window._catalog.pair_recipes(_spools[0].id, _spools[1].id)[0]
        _answer = window.nearest_recipe(_sample.color_hex)
        check(
            _answer is not None and _gap(_answer, _sample.color_hex) <= 1e-6,
            f"asking for the mix {_sample.color_hex} answers itself, got "
            f"{None if _answer is None else _answer.color_hex}",
        )
    # ... and the answer is never a colour that is no longer on screen.
    window._search.setText("#8080")
    app.processEvents()
    _on_screen = {cell.key for cell in window._recipes}
    _answer = window.nearest_recipe("#808080")
    check(
        _answer is None or _answer.key in _on_screen,
        "the nearest colour is always one of the colours currently on screen",
    )
    window._clear_filters()
    app.processEvents()

    # --- the add/edit dialog builds a valid filament ---------------------------
    from app.ui.filament_dialog import FilamentDialog  # noqa: E402

    dialog = FilamentDialog(brands=("大简",), types=("PETG HF",))
    dialog.setWindowTitle("smoke")
    dialog._name.setText("测试耗材")
    dialog._brand.setCurrentText("大简")
    dialog._type.setCurrentText("PETG HF")
    dialog._color.setValue("#1234AB")
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
    # The picture page has its own tab, and Qt does not deliver mouse clicks to a
    # hidden widget — the 耗材方案 checks below click real radios.
    window._tabs.setCurrentWidget(page)
    app.processEvents()

    # --- 导入 SVG 当原图（m13907 item 2）---------------------------------
    # 矢量图没有分辨率：导入时弹窗问「渲成多少像素」（1.6.49，m20069），默认值就是它在文件
    # 里声明的那一档（这里是 60×30），所以直接回车就是 1.6.48 的行为。冒烟里没有人能点那个
    # 框，所以把 `_ask_vector_size` 换成一个脚本，再走后面的全流程。
    vector_target = TEMP / "vector.svg"
    vector_target.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="60" height="30" viewBox="0 0 60 30">'
        '<rect width="60" height="30" fill="#E8C7BD"/>'
        '<circle cx="15" cy="15" r="8" fill="#8C4A52"/>'
        '<circle cx="45" cy="15" r="8" fill="#2F2A33"/>'
        "</svg>",
        encoding="utf-8",
    )
    vector_asked = []

    def _vector_size_from_script(parent, path, declared):
        vector_asked.append((path.name, tuple(declared)))
        return (120, 60)

    real_ask_vector_size = _picture_page._ask_vector_size
    _picture_page._ask_vector_size = _vector_size_from_script
    try:
        page.loadImage(vector_target)
        app.processEvents()
    finally:
        _picture_page._ask_vector_size = real_ask_vector_size
    check(
        vector_asked == [("vector.svg", (60, 30))],
        f"导入 SVG 会先弹窗问「渲成多少像素」，默认值就是它在文件里声明的那一档（1.6.49），"
        f"got {vector_asked}",
    )
    check(
        page._image is not None and page._image.size == (120, 60),
        f"弹窗里写 120×60，渲出来就是 120×60 的位图（1.6.49），"
        f"got {None if page._image is None else page._image.size}",
    )
    check(
        page.result is not None and len(page.result.palette) >= 2,
        "SVG 渲染出来的图照样能匹配",
    )
    check(
        page._image is not None and page._span == max(page._image.size),
        f"导入之后匹配栅格取图片自己的像素数，got {page._span}",
    )
    bitmap_before = len(vector_asked)
    page.loadImage(sample)
    app.processEvents()
    check(
        len(vector_asked) == bitmap_before,
        f"位图导入不弹那个框 —— 放大只是插值（1.6.49），got {len(vector_asked)} 次",
    )
    kept = (page._image, page._image_name, page._span)
    _picture_page._ask_vector_size = lambda parent, path, declared: None
    try:
        page.loadImage(vector_target)
        app.processEvents()
    finally:
        _picture_page._ask_vector_size = real_ask_vector_size
    check(
        (page._image, page._image_name, page._span) == kept,
        "弹窗里按「取消」⇒ 这张 SVG 不导入，原图、图名、栅格一个都不动（1.6.49）",
    )

    # --- 删除「最大边长」：导入即按图片自身的像素数（m16941 第 1 条）-------------
    # 界面上不再有这个数字框：位图一进来栅格就是它自己的最长边，状态行写「一格就是一个
    # 原像素」。 也没有任何上限（m17910 第 2 条），缩小那一档只剩「打开的是老工程会碰到」，
    # 这里直接把 _span 压小来验证状态行仍然把「原图 → 匹配栅格（缩小几倍）」写出来。
    check(
        not hasattr(page, "_max_dimension"),
        "界面上不再有「最大边长」这个数字框（m16941 第 1 条）",
    )
    page.loadImage(sample)
    app.processEvents()
    check(
        page._span == max(page._image.size),
        f"导入位图之后匹配栅格 = 图片自己的像素数（m16941 第 1 条），"
        f"got {page._span} / {page._image.size}",
    )
    check(
        "一格就是一个原像素" in page._status.text(),
        f"导入之后状态行写「一格就是一个原像素」，got {page._status.text()!r}",
    )
    page._span = 60
    page._on_rematch()
    app.processEvents()
    check(
        page._original_panel.title_label.text() == "原图 160×120",
        f"「原图」面板的标题带原图尺寸，got {page._original_panel.title_label.text()!r}",
    )
    check(
        "原图 160×120 → 匹配 60×45（缩小 2.7 倍）" in page._status.text(),
        f"栅格被压小时状态行写明原图 → 匹配栅格，got {page._status.text()!r}",
    )
    page._span = max(page._image.size)
    page._on_rematch()
    app.processEvents()
    check(
        "一格就是一个原像素" in page._status.text(),
        f"栅格回到一比一时状态行也说清楚，got {page._status.text()!r}",
    )

    result = page.result
    check(result is not None, "the sample picture matched a palette")
    check(
        page._list.current() == -1 and not page._list.selected_indices(),
        f"刚匹配完不默认选中第一个颜色（m01885 #7），got {page._list.current()}",
    )
    if result is not None:
        from app.core.image_matching import build_palette

        full = build_palette(library, window._catalog)
        check(
            len(full) == len(library) + merged,
            f"the candidate palette is every spool plus every merged mix, got {len(full)}",
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
        # Clicking must NOT reshuffle the list: it used to move the clicked colour
        # to the top, so the row under the cursor became a different colour than
        # the one just clicked.
        check(
            rows.item(0).data(Qt.ItemDataRole.UserRole) == 0,
            "the list keeps its sort order — row 0 stays the first sorted colour",
        )
        check(
            page._list._row_of(last) == last,
            "the clicked colour is still in its own row",
        )
        check(
            rows.currentRow() == last,
            f"the current row is the clicked colour, got {rows.currentRow()}",
        )
        check(
            page._list.selected_indices() == [last],
            f"clicking one colour selects exactly that one: {page._list.selected_indices()}",
        )
        order = [rows.item(i).data(Qt.ItemDataRole.UserRole) for i in range(rows.count())]
        check(order == list(range(len(result.palette))), "no row moved because of the click")

        # Issue 6: 原图 must stay the uploaded picture.  It used to grey out the
        # moment a colour was selected, so it looked like a washed-out copy of
        # 预览 instead of the user's own file.
        check(page._original_view._plain, "原图 is in plain mode")
        check(
            page._original_view._selected == -1,
            f"selecting a colour does not touch 原图, got selected={page._original_view._selected}",
        )
        check(
            page._original_view._dim_pixmap() is None,
            "原图 never draws the dimming overlay",
        )
        check(
            page._original_view._base_image() is page._original_view._source,
            "原图 draws the uploaded image, not the matched render",
        )
        check(
            page._original_view._source is page._source_qimage,
            "原图 shares the uploaded picture the matcher was given",
        )
        # ... and the 预览 keeps reacting to the selection.
        check(page._view._selected == last, "预览 still highlights the chosen colour")
        check(page._view._dim_pixmap() is not None, "预览 still dims everything else")
        # Issue 5: every hint and recipe is text people paste into Bambu Studio.
        selectable_flag = Qt.TextInteractionFlag.TextSelectableByMouse

        def _unselectable(root):
            return [
                w.text()[:40]
                for w in root.findChildren(QLabel)
                if w.text().strip() and not (w.textInteractionFlags() & selectable_flag)
            ]

        labels = [w for w in window.findChildren(QLabel) if w.text().strip()]
        check(bool(labels), "the main window has text labels to check")
        blind = _unselectable(window)
        check(not blind, f"every label in the main window can be selected, blind: {blind[:6]}")
        blind_page = _unselectable(page)
        check(not blind_page, f"every label on 图像转换 can be selected, blind: {blind_page[:6]}")

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
                abs(
                    plate.parts[0].region_pixels * plate.pixel_size_mm**2
                    - result.printed_pixels * (plate.width_mm / result.width) ** 2
                )
                < 1e-6,
                "the base plate covers the whole silhouette",
            )
            # 1.6.23 (m02289 item 2): the picture prints face down by default, so it is
            # the colour layer that touches the bed and the base slab that is on top.
            check(
                plate.parts[0].z_bottom_mm == plate.colour_thickness_mm,
                "the base plate is the top of the slab now",
            )
            check(
                all(part.z_bottom_mm == 0.0 for part in plate.parts[1:]),
                "every colour prism starts on the bed",
            )

            # --- 1.6.22 (m01647): 「圆滑色块」 --------------------------------
            from app.mesh.plate import edge_health, plate_mesh

            check(
                page._smooth_box.text() == _picture_page.SMOOTH_LABEL,
                f"the export row carries the {_picture_page.SMOOTH_LABEL} box",
            )
            check(page._smooth_box.isChecked(), "it is on by default")
            check(page._plate_settings().smooth_edges, "… and it reaches PlateSettings")
            check(
                any("圆滑" in note for note in plate.notes),
                f"the build says the borders were smoothed, got {plate.notes}",
            )
            smooth_points, smooth_faces, _smooth_parts = plate_mesh(plate)
            check(
                edge_health(smooth_points, smooth_faces) == (0, 0),
                "a smoothed plate is still one closed solid",
            )
            page._smooth_box.setChecked(False)
            app.processEvents()
            check(not page._plate_settings().smooth_edges, "unchecking turns it off")
            plain = page.buildPlate()
            check(
                plain.pixel_size_mm > plate.pixel_size_mm
                and abs(plain.pixel_size_mm - plain.width_mm / result.width) < 1e-9,
                "without the box the mesh goes back to the picture's own pixel",
            )
            check(
                abs(plain.width_mm - plate.width_mm) < 1e-9
                and abs(plain.volume_mm3 - plate.volume_mm3) / plate.volume_mm3 < 0.02,
                "smoothing changes neither the width nor the volume",
            )
            page._smooth_box.setChecked(True)
            app.processEvents()
            plate = page.buildPlate()

            # --- 1.6.26 (m05557 item 3): the box drives the preview too --------
            # It used to reach PlateSettings only, so the one place it changed
            # anything was inside Bambu Studio, after a round trip.
            page._smooth_box.setChecked(False)
            app.processEvents()
            blocky = page._view._matched
            check(
                page._view._smooth is False and blocky is not None and blocky.width() == result.width,
                f"unchecking the box puts the preview back on the raw grid ({None if blocky is None else blocky.width()} px)",
            )
            page._smooth_box.setChecked(True)
            app.processEvents()
            rounded = page._view._matched
            check(
                page._view._smooth is True
                and rounded is not None
                and blocky is not None
                and rounded.width() > blocky.width(),
                f"checking it draws the refined grid the plate is cut from ({None if rounded is None else rounded.width()} px)",
            )
            page._smooth_box.setChecked(False)
            app.processEvents()
            check(page._view._matched is blocky, "toggling back reuses the cached picture")
            page._smooth_box.setChecked(True)
            app.processEvents()

            # --- 1.6.23 (m02289 item 2): 图片位置 底面 / 顶面 -------------------
            from app.mesh.plate import signed_volume as _signed_volume

            check(
                page._side_combo.currentText() == _picture_page.IMAGE_SIDE_BOTTOM,
                f"图片位置 defaults to {_picture_page.IMAGE_SIDE_BOTTOM}",
            )
            check(
                page._side_combo.count() == 2
                and page._side_combo.itemText(1) == _picture_page.IMAGE_SIDE_TOP,
                "… and offers exactly 底面 / 顶面",
            )
            check(page._plate_settings().image_on_bottom, "the default reaches PlateSettings")
            face_down = page.buildPlate()
            check(
                face_down.parts[0].z_bottom_mm == face_down.colour_thickness_mm
                and all(part.z_bottom_mm == 0.0 for part in face_down.parts[1:]),
                "底面 puts the colour layer on the bed and the base slab on top",
            )
            page._side_combo.setCurrentIndex(1)
            app.processEvents()
            check(not page._plate_settings().image_on_bottom, "顶面 turns it off")
            face_up = page.buildPlate()
            check(
                face_up.parts[0].z_bottom_mm == 0.0
                and all(
                    part.z_bottom_mm
                    == face_up.base_thickness_mm - face_up.colour_thickness_mm
                    for part in face_up.parts[1:]
                ),
                "顶面 is the 1.6.22 layout: base on the bed, picture on top",
            )
            check(
                abs(face_up.width_mm - face_down.width_mm) < 1e-9
                and abs(face_up.volume_mm3 - face_down.volume_mm3) / face_down.volume_mm3 < 1e-9,
                "flipping the picture changes neither the width nor the volume",
            )
            thickness = face_up.total_thickness_mm
            check(
                all(
                    abs(float(a.vertices[:, 2].min()) - (thickness - float(b.vertices[:, 2].max())))
                    < 1e-9
                    and abs(
                        float(a.vertices[:, 2].max()) - (thickness - float(b.vertices[:, 2].min()))
                    )
                    < 1e-9
                    for a, b in zip(face_down.parts, face_up.parts)
                ),
                "底面 part for part is 顶面 mirrored through the slab",
            )
            down_points, down_faces, _ = plate_mesh(face_down)
            check(
                edge_health(down_points, down_faces) == (0, 0)
                and _signed_volume(down_points, down_faces) > 0,
                "the face-down plate is still one closed solid with outward normals",
            )
            page._side_combo.setCurrentIndex(0)
            app.processEvents()
            check(page._side_combo.currentText() == _picture_page.IMAGE_SIDE_BOTTOM, "back to 底面")
            plate = page.buildPlate()

            # --- m15721 第 3 条：板长宽 = 图的像素数 × 0.4，厚度只有一个框 ---------
            check(
                not hasattr(page, "_width_spin") and not hasattr(page, "_colour_spin"),
                "「打印宽度」和「颜色层厚度」两个框都撤了（m15721 第 3 条）",
            )
            from app.mesh.plate import CELL_MM as _cell_mm

            check(
                abs(page._plate_settings().cell_mm - _cell_mm) < 1e-12
                and abs(plate.width_mm - page._result.width * _cell_mm) < 1e-9,
                f"一格固定 {_cell_mm} mm，板长宽由图片像素数决定，"
                f"got {page._plate_settings().cell_mm} mm / {plate.width_mm} mm",
            )
            check(
                abs(page._plate_settings().base_thickness_mm - page._base_spin.value()) < 1e-9
                and abs(plate.total_thickness_mm - page._base_spin.value()) < 1e-9,
                "「底板厚度」就是整块的厚度",
            )
            check(
                "整块" in page._base_spin.toolTip() and "0.4 mm" in page._base_spin.toolTip(),
                f"底板厚度框的提示写清整块里含图上那 0.4 mm 颜色，"
                f"got {page._base_spin.toolTip()!r}",
            )

            # --- m16941 第 2 条：底板颜色跟「更换颜色」用同一个挑选框 -----------------
            check(
                not hasattr(page, "_base_combo") and hasattr(page, "_base_button"),
                "底板颜色不再是只列图上几行的下拉框（m16941 第 2 条）",
            )
            check(
                page._base_entry is None,
                "没挑过底板颜色时是「自动」（跟着图上最大的一块走）",
            )
            _auto = page._shown_base_entry()
            check(
                _auto is not None
                and plate.parts[0].color_hex == _auto.color_hex
                and plate.base_slot is None,
                f"自动时底板取图上最大一块的颜色、不另占槽，"
                f"got {plate.parts[0].color_hex} / {_auto.color_hex if _auto else None}",
            )
            _printed = {str(entry.key) for entry in page._result.palette}
            _extra = next(
                (entry for entry in page._candidate_palette() if str(entry.key) not in _printed),
                None,
            )
            if _extra is None:
                check(False, "整套混色里找不到一个图外的颜色，没法验证底板另占一个槽")
            else:
                page._base_entry = _extra
                _own = page.buildPlate()
                _slot = _own.parts[0]
                _entries = _own.filament_entries(page._result.palette)
                check(
                    _slot.color_hex == _extra.color_hex
                    and _slot.extruder == len(page._result.palette) + 1,
                    f"挑了图上没有的颜色时，底板自己占一个耗材槽（m16941 第 2 条），"
                    f"got {_slot.color_hex} 槽 {_slot.extruder}",
                )
                check(
                    _own.base_slot is _extra and _entries[-1] is _extra,
                    "底板那一种被接在调色板后面交给 3MF（filament_entries）",
                )
                from app.mesh import threemf as _slots_mod

                _slots = _slots_mod.slots_from_palette(_entries, page.library)
                check(
                    _slots[-1].color_hex == _extra.color_hex,
                    f"3MF 的槽表里底板那一种带上了自己的颜色，got {_slots[-1].color_hex}",
                )
                page._base_entry = page._result.palette[0]
                _shared = page.buildPlate()
                check(
                    _shared.base_slot is None and _shared.parts[0].extruder == 1,
                    "挑的正好是图上某一卷时，底板跟它共用一个槽（老行为不变）",
                )
                page._base_entry = None
                page.buildPlate()
                app.processEvents()

            # m17910 第 1 条：按下去要真的把这个挑选框建出来。早先这里把 hex 字符串当 RGB
            # 传下去，框在 int('#') 上抛异常，界面上看起来就是「按钮点不动」。
            from app.ui import picture_page as _picker_mod

            _real_base_picker = _picker_mod.ColourPickerDialog
            _base_calls: list[tuple] = []

            class _RecordingBasePicker:
                def __init__(self, entries, target_rgb, **kwargs):
                    _base_calls.append(
                        (tuple(target_rgb), kwargs.get("title"), kwargs.get("target_label"))
                    )
                    self._entries = list(entries)

                def exec(self):
                    return _picker_mod.QDialog.DialogCode.Accepted

                def chosen(self):
                    printed = {str(e.key) for e in page._result.palette}
                    return next(
                        (e for e in self._entries if str(e.key) not in printed),
                        self._entries[0],
                    )

            _picker_mod.ColourPickerDialog = _RecordingBasePicker
            try:
                page._base_button.click()
            finally:
                _picker_mod.ColourPickerDialog = _real_base_picker
            app.processEvents()
            check(
                len(_base_calls) == 1
                and len(_base_calls[0][0]) == 3
                and all(isinstance(v, int) for v in _base_calls[0][0]),
                f"「底板颜色 → 更换…」按一下真的把框建出来了，而且传的是 RGB 三元组"
                f"（m17910 第 1 条），got {_base_calls}",
            )
            check(
                _base_calls and _base_calls[0][1] == "更换底板颜色"
                and _base_calls[0][2] == "现在的底板颜色",
                f"底板颜色的框标题与「现在这一种」的说明写清楚了，got {_base_calls}",
            )
            check(
                page._base_entry is not None
                and page._base_entry.color_hex != plate.parts[0].color_hex
                and "底板颜色换成" in page._status.text(),
                f"挑完底板颜色就记下来并写了状态行，got {page._base_entry} / {page._status.text()[:40]}",
            )
            page._base_entry = None
            page._plate = None
            page._refresh_base_view()
            app.processEvents()
            # m18149 第 2 条：按钮文字只说它干什么，「自动」挪进 tooltip。
            check(
                page._base_button.text() == "更换底板颜色"
                and "自动挑的" in page._base_button.toolTip(),
                f"按钮写「更换底板颜色」、自动与否写在 tooltip 里，got "
                f"{page._base_button.text()!r} / {page._base_button.toolTip()!r}",
            )

            # m19523：1.6.46 加的「宽/长」两个框与「按当前px重新导入图片」撤掉了；
            # 导入仍然按图片自己的像素数分色（1.6.43 起就是这条）。
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
                (int(page._result.width), int(page._result.height)) == _picture_size,
                f"栅格就是原图的像素数（一格一个像素），got "
                f"{page._result.width}×{page._result.height}",
            )
            # m18149 第 1 条：图片朝下时 X 也照一次镜子 —— 原图里靠左的色块跑到右边，
            # 翻过来看底面（X 再反一次）才跟原图一个方向。
            page._side_combo.setCurrentIndex(1)
            face_up = page.buildPlate()
            page._side_combo.setCurrentIndex(0)
            face_down = page.buildPlate()
            app.processEvents()
            up_x = {p.palette_index: float(p.vertices[:, 0].mean()) for p in face_up.parts[1:]}
            down_x = {p.palette_index: float(p.vertices[:, 0].mean()) for p in face_down.parts[1:]}
            left_up = min(up_x, key=up_x.get) if up_x else None
            left_down = min(down_x, key=down_x.get) if down_x else None
            check(
                len(down_x) >= 2
                and left_up is not None
                and left_down is not None
                and left_up != left_down
                and max(up_x, key=up_x.get) == left_down,
                f"图片朝下时左右翻了一次，翻过来看才是正的（m18149 第 1 条），got "
                f"朝上 {up_x} / 朝下 {down_x}",
            )

            # m17910 第 2 条：栅格不再有上限（1.6.43 里的 4096 也删了）。
            from app.core.image_matching import MatchSettings as _GridSettings

            check(
                _GridSettings(max_dimension=9999).clamped().max_dimension == 9999,
                f"栅格不封顶，got {_GridSettings(max_dimension=9999).clamped().max_dimension}",
            )
            check(
                not hasattr(_picker_mod, "SVG_RENDER_LIMIT"),
                "SVG 渲染也不再封 4096（m17910 第 2 条）",
            )

            # --- 选中的颜色: both RGBs, the recipe, and manual replacement -------
            from app.core.image_matching import MatchSettings, sort_palette
            from app.core.mixes import (
                SORT_CHOICES_WITH_SIMILARITY,
                SORT_SIMILARITY,
            )
            from app.spectral import color as _spectral_color
            from app.ui.colour_picker import ColourDetail, ColourPickerDialog, LIST_CHUNK

            # 1.6.13 #2: the detail is asked for with a double click and opens in a
            # window of its own; the column keeps 耗材方案 + 打印图片需要的颜色 only.
            check(isinstance(page._detail, ColourDetail), "the colour detail widget is still there")
            check(
                page._detail_window().isAncestorOf(page._detail)
                and not page._detail_window().isVisible(),
                "双击之前它待在关着的「颜色详情」窗口里",
            )
            check(
                not page._detail_window().isModal(),
                "那个窗口不是模态的：开着也能继续点色块",
            )
            page._on_colour_activated(0)
            app.processEvents()
            check(
                page._detail_window().isVisible()
                and "颜色详情" in page._detail_window().windowTitle(),
                f"看详情会打开一个写明颜色的窗口，got {page._detail_window().windowTitle()!r}",
            )
            page._detail_window().close()
            app.processEvents()
            check(not page._detail_window().isVisible(), "关掉它之后还留在原来的色块上")
            check(
                page._base is not None
                and page._base.palette == result.palette
                and not page._replaced
                and not page._merged
                and not page._deleted
                and not page._redo_stack,
                "a fresh match starts with the automatic palette and no edits"
                f"（palette 一致 {page._base is not None and page._base.palette == result.palette}，"
                f"replaced {len(page._replaced)}，merged {len(page._merged)}，"
                f"deleted {len(page._deleted)}，redo {len(page._redo_stack)}）",
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
            check(
                not page._detail.can_restore(),
                "nothing has been hand-picked before a change",
            )
            check(page._delete_button.isEnabled(), "删除 is live once a colour is picked")
            check(not page._merge_button.isEnabled(), "合并 needs two colours, not one")

            # The replacement menu is 「全部颜色」, ordered like the 颜色配方 grid.
            entries = build_palette(library, window._catalog, include_mixes=True)
            # At 41 spools this list is 66,461 entries and ~0.5 s to build.  The
            # match and every 「更换颜色…」 have to share one copy of it.
            cached = page._candidate_palette()
            check(cached is page._candidate_palette(), "the 全部颜色 list is cached, not rebuilt")
            check(
                {candidate.key for candidate in cached} == {candidate.key for candidate in entries},
                "the cached list is exactly the replacement menu's candidates",
            )
            # 「混色也算候选」 used to sit here: ticking it off built a smaller
            # list, so the cache had to key on it.  The box is gone — a spool
            # library without its mixes is half a palette — so the list is
            # always the full one and the cache is keyed on nothing but the
            # library itself.
            check(
                page._candidate_palette() is cached,
                "the candidate list stays cached; it has no checkbox to invalidate it",
            )
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

            # 41 spools makes 66,461 candidates; building a widget per candidate
            # took 26 s and Windows greyed the dialog out as 「(未响应)」.  Only
            # the first screenful is materialised, the rest arrive as you scroll.
            check(
                picker._list.count() == min(len(entries), LIST_CHUNK),
                f"the menu materialises one batch of rows, got {picker._list.count()}",
            )
            check(
                len(picker.visible_rows()) == len(entries),
                "but every candidate is still reachable in the menu",
            )
            check(
                picker._rows[0].key == ordered[0].key,
                "the materialised rows keep the similarity order",
            )
            picker._extend()
            check(
                picker._list.count() == min(len(entries), 2 * LIST_CHUNK),
                f"scrolling a screenful adds the next batch, got {picker._list.count()}",
            )
            check(
                picker._list.item(picker._list.count() - 1).text().startswith(
                    ordered[picker._list.count() - 1].color_hex.upper()
                )
                and picker._list.item(LIST_CHUNK).text().startswith(
                    ordered[LIST_CHUNK].color_hex.upper()
                ),
                "the appended batch continues where the first one stopped",
            )

            # Searching must not put the whole palette back into the list.
            picker._search.setText("白")
            picker._rebuild()
            found = picker.visible_rows()
            check(
                0 < len(found) < len(entries),
                f"a search narrows the list, got {len(found)} of {len(entries)}",
            )
            check(
                picker._list.count() == min(len(found), LIST_CHUNK),
                f"the hits are materialised a batch at a time, got {picker._list.count()}",
            )
            table = picker._haystack_table()
            check(
                all("白" in table[candidate.key] for candidate in found),
                "every search hit mentions 白",
            )
            check(
                any(candidate.is_mix for candidate in found),
                "searching a spool's name also finds the mixes it is part of",
            )
            picker._search.clear()
            picker._rebuild()
            check(
                len(picker.visible_rows()) == len(entries),
                "clearing the search brings 全部颜色 back",
            )
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
            # Snapshot first, then record — exactly what _on_replace does, which
            # is why 撤销 alone is enough to take a replacement back.
            page._push()
            page._replaced[0] = replacement
            page._apply_edits(keep=0)
            app.processEvents()
            check(
                page._result.palette[0].key == replacement.key,
                "the replacement lands in the palette the print uses",
            )
            check(page._detail.can_restore(), "the panel knows the colour was hand-picked")
            check(page._undo_button.isEnabled(), "a replacement can be undone with 撤销")
            check(page._replace_button.isEnabled(), "更换颜色 is live right next to 删除")
            check(
                page._replace_button.property("accent") in (None, "false"),
                "更换颜色 is a plain button like 删除 / 合并, not a blue one",
            )
            check(
                "手动更换" in page._status.text(),
                f"the status line reports the manual replacement: {page._status.text()}",
            )
            check(
                replacement.color_hex.upper()
                in {part.color_hex.upper() for part in page.buildPlate().parts},
                "the plate really uses the replacement colour",
            )

            page._undo()
            app.processEvents()
            check(
                page._result.palette[0].key == entry.key,
                "撤销 puts the automatic match back",
            )
            check(page._replaced == {}, "undoing forgets the replacement")
            check(not page._detail.can_restore(), "nothing hand-picked is left")
            check(
                entry.color_hex.upper()
                in {part.color_hex.upper() for part in page.buildPlate().parts},
                "the plate goes back to the automatic colour",
            )

            # 「删除」: the pixels stop being printed and the plate loses both the
            # prism and the base plate under them.
            import numpy as _np  # noqa: PLC0415

            printed_before = page._result.printed_pixels
            colours_before = len(page._result.palette)
            base_before = len(page.buildPlate().parts)
            live = page._result.indices
            placed = _np.bincount(live[live >= 0].ravel())
            doomed = int(placed.argmax())
            doomed_pixels = int(placed[doomed])
            page._list.select([doomed])
            check(page._delete_button.isEnabled(), "删除 lights up once a colour is picked")

            # 「多选」: on, two plain clicks keep BOTH colours (the whole point is
            # not having to know about Ctrl).  Off, the list holds exactly one row —
            # m02289 item 1: 「没有多选却多选了色块」 used to happen by three doors
            # (Ctrl+click, turning 多选 off with a set in hand, an edit reselecting
            # every block it touched); all three are shut now.
            from PySide6.QtWidgets import QAbstractItemView as _QAbstractItemView

            check(page._multi_button.isCheckable(), "多选 is a toggle")
            check(not page._multi_button.isChecked(), "多选 starts off")
            check(
                page._list._list.selectionMode()
                == _QAbstractItemView.SelectionMode.SingleSelection,
                "off means the list can only hold one row",
            )
            page._multi_button.setChecked(True)
            app.processEvents()
            check(
                page._list._list.selectionMode()
                == _QAbstractItemView.SelectionMode.MultiSelection,
                "多选 switches the list to plain-click multi-selection",
            )
            rows = [page._list._list.item(i) for i in range(page._list._list.count())]
            rows[0].setSelected(True)
            rows[1].setSelected(True)
            app.processEvents()
            check(
                len(page._list.selected_indices()) == 2,
                f"two plain clicks select two colours: {page._list.selected_indices()}",
            )
            check(page._merge_button.isEnabled(), "合并 lights up from a plain multi-selection")
            page._multi_button.setChecked(False)
            app.processEvents()
            check(
                len(page._list.selected_indices()) == 1,
                "turning 多选 off drops the extra rows, keeping one: "
                f"{page._list.selected_indices()}",
            )
            check(
                page._list._list.selectionMode()
                == _QAbstractItemView.SelectionMode.SingleSelection,
                "turning 多选 off restores the one-row list",
            )
            # The guard behind the mode: whatever put a second row in the selection,
            # 多选 off is only ever allowed to hold one (m02289 item 1).
            page._list.select([0, 1])
            app.processEvents()
            check(
                len(page._list.selected_indices()) == 1,
                f"多选 off cannot hold two rows even when asked: {page._list.selected_indices()}",
            )
            page._list.select([doomed])

            # The row reads in the order the work happens: how to change the
            # selection, then what to do with it, then how to take it back.
            from PySide6.QtWidgets import QHBoxLayout as _QHBoxLayout
            from PySide6.QtWidgets import QPushButton as _QPushButton

            edit_row = next(
                layout
                for layout in page.findChildren(_QHBoxLayout)
                if layout.indexOf(page._multi_button) >= 0
            )
            labels = [
                edit_row.itemAt(position).widget().text()
                for position in range(edit_row.count())
                if isinstance(edit_row.itemAt(position).widget(), _QPushButton)
            ]
            check(
                labels == ["多选", "删除", "合并", "更换颜色", "撤销", "恢复"],
                f"the edit row reads 多选 删除 合并 更换颜色 撤销 恢复, got {labels}",
            )
            # m02316 item 3: Del still deletes, it just is not part of the button's name
            # any more — 帮助 is where the key is written down now.
            check(
                "Del" in _picture_page.PAGE_HELP,
                "「删除」不再叫 (Del)，但帮助里还写着 Del 就是删除",
            )
            # m14360: 用户问「为什么预览的色块大小形状和原图的不一样」——帮助里得说清楚
            # 这是「颜色数量」压缩 + 「圆滑色块」细格叠加的正常结果，以及怎么看到原始格子。
            check(
                "形状跟原图对不上" in _picture_page.PAGE_HELP
                and "取消这个勾" in _picture_page.PAGE_HELP,
                "帮助里说明了预览色块的形状为什么和原图对不上、怎么切回原始格子（m14360）",
            )

            page._on_delete_colours()
            app.processEvents()
            check(
                len(page._result.palette) == colours_before - 1,
                f"删除 drops one colour, {colours_before} → {len(page._result.palette)}",
            )
            check(
                page._result.printed_pixels == printed_before - doomed_pixels,
                "删除 stops printing exactly the deleted pixels",
            )
            check(
                bool((page._result.indices < 0).any())
                and (page._result.indices < 0).sum() > 0,
                "the deleted region is transparent in the index map",
            )
            check(page._undo_button.isEnabled(), "删除 can be undone")
            check("删除" in page._status.text(), f"the status reports 删除: {page._status.text()}")
            # Building the plate moves the status bar onto its own progress line
            # (「正在生成模型：…」, 1.6.22), so ask about 删除 before that happens.
            check(
                len(page.buildPlate().parts) == base_before - 1,
                "the plate loses the deleted colour's prism",
            )
            check(
                page._plate.triangle_count > 0,
                "the plate still builds with a hole in it",
            )

            page._undo()
            app.processEvents()
            check(
                not page._deleted
                and len(page._result.palette) == colours_before
                and page._result.printed_pixels == printed_before,
                "Ctrl+Z brings the deleted colour and its pixels back",
            )
            check(page._redo_button.isEnabled(), "the undone step can be redone")
            page._redo()
            app.processEvents()
            check(
                len(page._result.palette) == colours_before - 1,
                "Ctrl+Y deletes it again",
            )
            page._undo()
            app.processEvents()
            check(
                len(page._result.palette) == colours_before and page._redo_button.isEnabled(),
                "a second Ctrl+Z is back to the clean match",
            )

            # --- m12426 item 1: 一行一种打印色 --------------------------------
            palette_keys = [entry.key for entry in page._result.palette]
            distinct = len(set(palette_keys))
            check(
                page._list._list.count() == distinct,
                f"「按颜色选择」一行一种打印色（{page._list._list.count()} 行 / {distinct} 种）",
            )
            check(
                f"共 {distinct} 种颜色" in page._list._summary.text(),
                f"小结也按打印色计数：{page._list._summary.text()[-14:]}",
            )
            if distinct < len(palette_keys):
                grouped = [
                    row
                    for row in range(page._list._list.count())
                    if "等" in page._list._list.item(row).text()
                ]
                check(
                    bool(grouped),
                    f"同一个配方的几处合成一行，行里写明「等 N 处」：{grouped}",
                )

            # --- m12426 item 2: 单选再点一下 = 什么都不选 ---------------------
            page._list.select([0])
            app.processEvents()
            page._select(page._list.selected_indices()[0])
            app.processEvents()
            first = page._list._list.item(0)
            page._list._press_index = int(first.data(Qt.ItemDataRole.UserRole))
            page._list._press_shown = page._list._pinned
            page._list._on_row_clicked(first)
            app.processEvents()
            check(
                not page._list.selected_indices() and page._shown < 0,
                "再点一下已经选中的颜色：什么都不选，不像以前那样退回上一次那个"
                f"（选中 {page._list.selected_indices()}、_shown {page._shown}）",
            )
            check(
                not page._delete_button.isEnabled() and not page._merge_button.isEnabled(),
                "一个都没选中时 删除 / 合并 都灰着",
            )

            # 合并: the merge itself, then undo, then the dialog's own plumbing.
            if len(page._result.palette) >= 2:
                keep_key = page._result.palette[1].key
                page._list.select([0])
                check(not page._merge_button.isEnabled(), "合并 needs two colours, not one")
                check(not page._merge_button.isEnabled(), "合并 stays off while 多选 is off")
                # m12261: 用户以为按钮坏了，其实是没开「多选」—— 这句话得挂在按钮上。
                check(
                    "多选" in page._merge_button.toolTip(),
                    "合并 的提示告诉他先开多选："
                    + page._merge_button.toolTip(),
                )
                page._multi_button.setChecked(True)
                page._list.select([0, 1])
                check(page._merge_button.isEnabled(), "合并 lights up with two colours picked")
                page._push()
                page._merged[page._live_map[0]] = page._live_map[1]
                page._apply_edits(keep=page._live_map[1])
                app.processEvents()
                check(
                    len(page._result.palette) == colours_before - 1
                    and page._result.palette[0].key == keep_key,
                    "merging two colours leaves the chosen one",
                )
                check(
                    page._result.printed_pixels == printed_before,
                    "merging prints every pixel the two colours covered",
                )
                # m01411 item 3: list and preview have to point at the same colour —
                # the list must never say 「没选中」 while 预览 draws a block.
                check(
                    page._list.current() >= 0
                    and page._view._selected == page._list.current()
                    and page._view._dim_pixmap() is not None,
                    "after a merge the list and the preview point at the same colour "
                    f"(row={page._list.current()}, view={page._view._selected})",
                )
                page._undo()
                app.processEvents()
                check(
                    not page._merged and len(page._result.palette) == colours_before,
                    "undoing the merge restores both colours",
                )
                page._multi_button.setChecked(False)
                app.processEvents()
            from app.ui.picture_page import MergeDialog  # noqa: PLC0415

            dialog = MergeDialog(page._result.palette[:3], parent=page)
            check(dialog.windowTitle() == "合并颜色", "the merge dialog is titled 合并颜色")
            check(dialog._list.count() == 3, "the merge dialog lists exactly the picked colours")
            dialog._list.setCurrentRow(2)
            check(dialog.chosen() == 2, "the merge dialog reports which colour was chosen")
            dialog.close()

            # Deleting every colour is a legal state — the picture simply becomes
            # nothing — so it must not raise until the export refuses it.
            page._push()
            page._deleted = set(range(len(page._base.palette)))
            page._merged = {}
            page._apply_edits()
            app.processEvents()
            check(len(page._result.palette) == 0, "deleting every colour empties the palette")
            check(bool((page._result.indices < 0).all()), "nothing at all is printed")
            check(not page._detail._match_card.isVisible(), "the detail panel clears itself")
            check(not page._delete_button.isEnabled(), "删除 goes dead with nothing left to pick")
            try:
                page.buildPlate()
                check(False, "an empty palette is refused by the exporter")
            except ValueError as error:
                check(
                    "没有一个可打印的颜色" in str(error),
                    f"the refusal is readable: {error}",
                )
            page._undo()
            app.processEvents()
            check(
                len(page._result.palette) == colours_before
                and page._result.printed_pixels == printed_before,
                "undo refills the palette after deleting everything",
            )

            # m11180 item 2: 改完颜色，方案表必须说清「这几个数字是改之前的」。
            # 删除 / 合并 / 更换颜色 只改「图上有什么颜色」，而行里的 平均 / 最差 /
            # 差块 是上一次匹配算出来的：用户删到只剩两种颜色，「差块 9」还挂在行上
            # （差块是按图片的颜色数算的，两种颜色最多两个差块）。行不改数字（不替用户
            # 重挑卷），但要标出来，并把「按当前色块重新匹配」指给他。
            stale_colours = len(page._result.palette)
            stale_rows = len(page._plans)
            check(stale_rows > 0, f"样图给出了可以检查的方案行，got {stale_rows}")
            check(not page._plans_stale, "刚匹配完的方案表是新鲜的")
            page._list.select([0])
            page._on_delete_colours()
            app.processEvents()
            check(
                len(page._result.palette) < stale_colours,
                "删掉一种颜色之后调色板确实少了",
            )
            check(page._plans_stale, "删掉一种颜色之后方案表被标成「改过了」")
            check(
                len(page._plans) == stale_rows,
                f"标出来而不是偷偷换掉行：行数没变，got {len(page._plans)}",
            )
            page._refresh_plans()
            stale_labels = [
                label.text()
                for label in page._plan_rows_layout.itemAt(0)
                .widget()
                .findChildren(_picture_page.QLabel)
                if "平均" in label.text()
            ]
            stale_row_text = stale_labels[0] if stale_labels else ""
            stale_tip = page._plan_group.button(0).toolTip()
            check(
                "色块改过了" in stale_row_text,
                f"第一行写着「色块改过了」：{stale_row_text}",
            )
            check(
                "按当前色块重新匹配" in stale_tip,
                f"提示里告诉他按哪个按钮重算：{stale_tip}",
            )
            page._on_rematch_blocks()
            app.processEvents()
            check(not page._plans_stale, "按下「按当前色块重新匹配」之后标记清掉了")
            check(
                all(option.defects <= len(page._result.palette) for option in page._plans),
                "重算之后每一行的差块都在图片颜色数之内："
                f"{[(option.key, option.defects) for option in page._plans]}",
            )
            # m11985：色块是用户自己的决定，再按一次不许把两块悄悄并成一块。
            keys_once = [entry.key for entry in page._base.palette]
            page._on_rematch_blocks()
            app.processEvents()
            check(
                [entry.key for entry in page._base.palette] == keys_once,
                f"再按一次手里的色块不变（{len(keys_once)} 块原样）",
            )
            page._undo()
            app.processEvents()
            page._undo()
            app.processEvents()
            check(
                len(page._plans) == stale_rows,
                f"撤销把方案表也放回去了，got {len(page._plans)} 行",
            )
            # 再退一步：把「删掉一种颜色」也退掉，后面的检查要在原来的图上跑。
            page._undo()
            app.processEvents()
            check(
                len(page._result.palette) == stale_colours and not page._plans_stale,
                "撤销回到改色块之前，调色板和方案表都回到原样",
            )

            # m13246 item 2：「颜色数量」是用户给的硬上限，「按当前色块重新匹配」也得守。
            # 收口只改「这些像素打印成什么」，色块与打印的像素一个不少。
            plain_keys = {entry.key for entry in page._result.palette}
            if len(plain_keys) > 3:
                count_box = page._colour_count
                was_count = count_box.value()
                painted_before = int((page._result.indices >= 0).sum())
                count_box.setValue(3)
                page._on_rematch_blocks()
                app.processEvents()
                capped_keys = {entry.key for entry in page._result.palette}
                check(
                    len(capped_keys) <= 3,
                    f"颜色数量收到 3 以后再重匹配，打印色不超过 3 种，got {len(capped_keys)}",
                )
                check(
                    int((page._result.indices >= 0).sum()) == painted_before,
                    f"收口以后打印的像素一个不少（{painted_before} 个）",
                )
                # 状态栏那句话也要跟小结、跟上限一个口径 (m13246 item 2)：位子数会比
                # 「种颜色」多，两个数并存用户就会以为上限又破了。
                check(
                    f"{len(capped_keys)} 种颜色" in page._status.text()
                    and f"共 {len(capped_keys)} 种颜色" in page._list._summary.text(),
                    f"状态栏和小结都按打印色计数（{len(capped_keys)} 种）",
                )
                count_box.setValue(was_count)
                page._undo()
                app.processEvents()
            else:
                check(
                    True,
                    f"样张的打印色本来就在 3 种以内（{len(plain_keys)} 种），跳过收口检查",
                )

            # Exercise the two export buttons, with the file chooser stubbed out.
            original_dialog = _picture_page.QFileDialog
            original_box = _picture_page.QMessageBox
            _picture_page.QMessageBox = _StubMessageBox
            try:
                # m12426 item 3: 「导出原图」 writes the edited original as a PNG, alpha
                # included — the picture the 原图 panel draws, not the preview.
                page._list.clear_selection()
                original_target = TEMP / "edited-original.png"
                _picture_page.QFileDialog = _StubChooser(original_target)
                page._on_export_original()
                app.processEvents()
                check(
                    original_target.is_file() and original_target.stat().st_size > 0,
                    f"导出原图 wrote {original_target.name}",
                )
                from PIL import Image as _Image

                exported = _Image.open(original_target).convert("RGBA")
                check(
                    exported.size == page._image.size,
                    f"导出的原图尺寸和图一致 {exported.size} == {page._image.size}",
                )
                check(
                    "导出原图" in page._status.text(),
                    f"状态栏说明导出了原图：{page._status.text()[:40]}",
                )
                _picture_page.QFileDialog = _StubChooser(TEMP / "no-image.png")
                saved_image = page._image
                page._image = None
                page._on_export_original()
                app.processEvents()
                check(
                    "先导入" in page._status.text(),
                    f"没有图的时候导出原图会拒绝：{page._status.text()[:24]}",
                )
                page._image = saved_image
                page._refresh_source()
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
                check(
                    app.overrideCursor() is None,
                    "the wait cursor is popped again after exporting",
                )

                # --- the 3MF carries the mixes' own recipes (m04136 item 2) -------
                # Bambu Studio reads a mixed filament from `filament_is_mixed`
                # plus `filament_mixed_components` and `_sublayer_ratios`, so the
                # exported plate has to name every component spool — including
                # the ones only the recipe uses, which become rows of their own.
                import json

                from app.mesh import threemf as _threemf

                with zipfile.ZipFile(TEMP / "plate.3mf") as archive:
                    written = json.loads(
                        archive.read("Metadata/project_settings.config").decode("utf-8")
                    )
                slots = _threemf.slots_from_palette(page.result.palette, page.library)
                project = _threemf.project_filaments(slots)
                check(
                    written["filament_colour"] == [slot.color_hex for slot in project.slots],
                    f"the 3MF lists all {project.count} filaments the recipes need",
                )
                check(
                    written["filament_is_mixed"]
                    == ["1" if index in project.mixed else "0" for index in range(project.count)],
                    f"the 3MF says which of them are mixed ({project.mixed_count} of {project.count})",
                )
                check(
                    all(
                        written["filament_mixed_components"][index]
                        == ",".join(str(parent) for parent in parents)
                        and written["filament_mixed_sublayer_ratios"][index]
                        == ",".join(f"{ratio:.4f}" for ratio in ratios)
                        for index, (parents, ratios) in project.mixed.items()
                    ),
                    "every mixed filament carries its components and their ratios",
                )
                if project.mixed_count:
                    check(
                        written["enable_mixed_color_sublayer"] == "1",
                        "mixed layers are split into sublayers (7 layers would band)",
                    )
                check(
                    written["filament_is_mixed"] == sorted(written["filament_is_mixed"])
                    and all(
                        index + 1 > max(parents)
                        for index, (parents, _ratios) in project.mixed.items()
                    ),
                    "every mixed row is written after the rows it mixes (Bambu Studio 2.8.2.61 "
                    "answers -18 'Invalid parameter value(s)' to anything else)",
                )
                plain = json.loads(
                    _threemf._project_settings(slots, {})
                )
                check(
                    "enable_mixed_color_sublayer" not in plain
                    and "0" * project.count == "".join(plain["filament_is_mixed"]),
                    "with no mixes at all the key is left out",
                )

                planted = (
                    _threemf.FilamentSlot(color_hex="#F5D2B8", material_type="PETG HF", vendor="大简"),
                    _threemf.FilamentSlot(
                        color_hex="#EA818A",
                        components=(
                            _threemf.FilamentComponent(
                                "#C52A36", material_type="PETG HF", vendor="大简", percent=38.0
                            ),
                            _threemf.FilamentComponent(
                                "#F5D2B8", material_type="PETG HF", vendor="大简", percent=62.0
                            ),
                        ),
                    ),
                )
                planted_project = _threemf.project_filaments(planted)
                planted_settings = json.loads(
                    _threemf._project_settings(planted_project.slots, planted_project.mixed)
                )
                check(
                    planted_settings["filament_colour"] == ["#F5D2B8", "#C52A36", "#EA818A"]
                    and planted_settings["filament_is_mixed"] == ["0", "0", "1"]
                    and planted_settings["filament_mixed_components"] == ["", "", "2,1"]
                    and planted_settings["filament_mixed_sublayer_ratios"]
                    == ["", "", "0.3800,0.6200"]
                    and planted_settings["enable_mixed_color_sublayer"] == "1",
                    "a planted recipe exports its components, ratios and sublayer switch",
                )
            finally:
                _picture_page.QFileDialog = original_dialog
                _picture_page.QMessageBox = original_box

            # --- 1.6.25 (m04136 item 1): the colour budget is spent -----------
            # The quantiser used to oversample six-fold and fold the result back
            # down, and the fold ate a ramp's evenly spaced clusters: 32 colours
            # asked for, 20 delivered.  The clustering now runs at the budget, so
            # the count the user typed is the count they get — and the merge that
            # folds near-identical colours compares against leaders only, so it
            # cannot chain a ramp into one colour.
            import numpy as _np
            from PIL import Image as _PILImage

            from app.core.image_matching import (
                MatchSettings as _KernelSettings,
                _merge_similar_colours as _merge_fold,
                reduce_colours as _reduce_colours,
            )

            gradient = _np.zeros((96, 128, 3), dtype=_np.uint8)
            gradient[..., 0] = _np.linspace(0, 255, 128, dtype=_np.uint8).reshape(1, 128)
            gradient[..., 1] = _np.linspace(0, 255, 96, dtype=_np.uint8).reshape(96, 1)
            gradient[..., 2] = _np.linspace(255, 0, 96, dtype=_np.uint8).reshape(96, 1)
            ramp = _PILImage.fromarray(gradient, "RGB").convert("RGBA")
            delivered = [
                len(_reduce_colours(ramp, _KernelSettings(max_colours=budget))[0])
                for budget in (8, 16, 32)
            ]
            check(
                delivered == [8, 16, 32],
                f"a gradient gets the colours it asked for ({delivered})",
            )
            greys = _np.array([(120 + 3 * index,) * 3 for index in range(6)], dtype=_np.uint8)
            folded, _ = _merge_fold(
                greys, _np.tile(_np.arange(6, dtype=_np.int64), (4, 6)), 2.0
            )
            check(
                len(folded) == 3,
                f"the merge keeps a ramp apart ({len(folded)} of 6 greys survive)",
            )

            # --- a match says what it is doing (m08102 item 2) ------------------
            # The exact-colour pass is a Python loop of CIEDE2000 calls: 41 spools
            # means 1.3–1.5 s of a frozen window.  The bar is deliberately NOT put
            # on screen at once — a 50 ms match would only flash it.
            check(
                page._progress.isHidden(),
                "the progress bar is hidden once a match is finished",
            )
            check(
                page._progress.maximum() == _picture_page.PROGRESS_STEPS,
                "the bar counts thousandths, so even a slow step moves it",
            )
            page._begin_match()
            check(page._busy_timer.isActive(), "the busy delay is armed when a match starts")
            check(page._progress.isHidden(), "… and a fast match never flashes the bar")
            page._show_progress()
            check(
                not page._progress.isHidden(),
                "the bar appears once the match outlasts the delay",
            )
            page._report_match("在耗材与混色库里寻找最接近的颜色", 0.42)
            check(
                page._progress.value() == int(0.42 * _picture_page.PROGRESS_STEPS),
                f"a report moves the bar, got {page._progress.value()}",
            )
            check(
                page._status.text() == "正在匹配：在耗材与混色库里寻找最接近的颜色…",
                f"the status line names the step, got {page._status.text()!r}",
            )
            page._end_match()
            check(
                page._progress.isHidden() and page._progress.value() == 0,
                "finishing hides the bar and resets it",
            )

            # A real match has to walk that bar forward, stage by stage.
            reports: list[tuple[str, float]] = []
            remember_report = page._report_match

            def _record(message: str, fraction: float) -> None:
                reports.append((message, fraction))
                remember_report(message, fraction)

            page._report_match = _record  # type: ignore[method-assign]
            try:
                page._on_rematch()
                app.processEvents()
            finally:
                page._report_match = remember_report  # type: ignore[method-assign]
            check(len(reports) >= 9, f"every stage of the match reports itself, got {len(reports)}")
            check(
                all(later[1] >= earlier[1] for earlier, later in zip(reports, reports[1:])),
                f"progress never goes backwards: {[round(f, 3) for _m, f in reports]}",
            )
            check(reports[-1][1] == 1.0, f"the last report is the end, got {reports[-1][1]}")
            check(
                sum(1 for message, _f in reports if "寻找最接近" in message) > 1,
                "the long colour search reports row by row, not only when it is over",
            )
            check(page._progress.isHidden(), "the bar is gone when the match ends")
            check(app.overrideCursor() is None, "no wait cursor is left over after a match")

            # 按当前色块重新匹配 is the button that froze the window (m00940 第 2 条):
            # it used to spend seconds grouping the whole three-filament pool
            # without saying anything, so the status line sat on the search.
            block_reports: list[tuple[str, float]] = []

            def _record_block(message: str, fraction: float) -> None:
                block_reports.append((message, fraction))
                remember_report(message, fraction)

            page._report_match = _record_block  # type: ignore[method-assign]
            try:
                page._on_rematch_blocks()
                app.processEvents()
            finally:
                page._report_match = remember_report  # type: ignore[method-assign]
            check(
                len(block_reports) >= 9,
                f"按当前色块重新匹配 reports every stage too, got {len(block_reports)}",
            )
            check(
                all(
                    later[1] >= earlier[1]
                    for earlier, later in zip(block_reports, block_reports[1:])
                ),
                f"… and its progress never goes backwards: "
                f"{[round(f, 3) for _m, f in block_reports]}",
            )
            check(
                block_reports[-1][1] == 1.0,
                f"… and it ends at 1.0, got {block_reports[-1][1]}",
            )
            check(
                any(message == "给每个色块找可替换的颜色" for message, _f in block_reports),
                "the stretch that used to be silent reports itself",
            )
            check(page._progress.isHidden(), "the bar is gone when the block re-match ends")

            # --- which rolls the picture spends (m08102 item 3) -----------------
            from PySide6.QtWidgets import QLabel as _QLabel  # noqa: PLC0415

            from app.core.image_matching import spool_usage as _spool_usage  # noqa: PLC0415

            uses = _spool_usage(page.result, library)
            check(bool(uses), "the match says which spools it spends")
            usage_text = page._usage.text()
            check(
                f"共 {len(uses)} 卷" in usage_text,
                f"the panel counts the rolls, got {usage_text!r}",
            )
            check(
                all(use.display_name in usage_text for use in uses),
                "every spool it spends is named in the panel",
            )
            check("%" in usage_text, "each spool says how much of the material it takes")
            check(
                page._usage.textFormat() == Qt.TextFormat.RichText
                and "background-color" in usage_text,
                "each spool is shown with its own colour",
            )
            # 1.6.13 #1: the rolls belong to the 方案 that spends them, so the list
            # lives inside 「耗材方案」 instead of in a card of its own.
            check(
                page._plan_box.isAncestorOf(page._usage),
                "the spool list lives in the 耗材方案 panel",
            )
            # An edit has to refresh the list, not leave the previous sentence up.
            usage_saved = page._usage.text()
            page._on_colour_selected(0)
            page._on_delete_colours()
            app.processEvents()
            check(page._usage.text() != usage_saved, "an edit refreshes the spool list")
            # m01411 item 3: the deleted colour was the selected one, so nothing is
            # selected now — and the preview must not keep drawing its block.
            check(
                page._view._selected == -1 and page._view._dim_pixmap() is None,
                "deleting the picked colour clears the preview highlight too "
                f"(view={page._view._selected})",
            )
            page._undo()
            app.processEvents()
            check(page._usage.text() == usage_saved, "undo puts the spool list back")

            # --- 点图片时：按颜色选择 / 按色块边缘选择 (m01411 item 4) --------------
            from app.core.settings import load_settings as _load_settings  # noqa: PLC0415

            labels = [page._pick_combo.itemText(i) for i in range(page._pick_combo.count())]
            check(
                labels
                == [
                    _picture_page.PICK_MODE_LABELS[_picture_page.PICK_COLOUR],
                    _picture_page.PICK_MODE_LABELS[_picture_page.PICK_EDGE],
                ],
                f"the toolbar offers both ways to pick a colour, got {labels}",
            )
            page._set_pick_mode(_picture_page.PICK_COLOUR, save=False)
            check(page._view.pickMode() == _picture_page.PICK_COLOUR, "按颜色选择 is what it did before")
            page._pick_combo.setCurrentIndex(1)
            app.processEvents()
            check(
                page._view.pickMode() == _picture_page.PICK_EDGE,
                f"choosing the other one reaches the preview, got {page._view.pickMode()!r}",
            )
            check(
                _load_settings().get("pick_mode") == _picture_page.PICK_EDGE,
                "the way the user picks is remembered for next time",
            )
            # A synthetic 5x5 match: one painted pixel at (2, 2), the rest unpainted.
            # The sample picture has no blank area for the model to draw, so the two
            # modes are compared where the difference is unambiguous: the cursor is on
            # a blank pixel whose nearest painted pixel is the one to its left.
            from dataclasses import replace as _dataclass_replace  # noqa: PLC0415

            import numpy as _np  # noqa: PLC0415

            lonely = _dataclass_replace(
                page._result,
                width=5,
                height=5,
                palette=[page._result.palette[0]],
                counts=_np.asarray([1], dtype=np.int64),
                indices=_np.full((5, 5), -1, dtype=np.int32),
            )
            lonely.indices[2, 2] = 0
            page._set_pick_mode(_picture_page.PICK_COLOUR, save=False)
            check(
                page._view._pick_index(lonely, 2, 2) == 0,
                "按颜色选择 reads the pixel under the cursor, and it is the block",
            )
            check(
                page._view._pick_index(lonely, 3, 2) == -1,
                "按颜色选择 answers with the pixel under the cursor, blank is blank",
            )
            page._set_pick_mode(_picture_page.PICK_EDGE, save=False)
            check(
                page._view._pick_index(lonely, 2, 2) == 0,
                "按色块边缘选择 reads the pixel under the cursor too",
            )
            check(
                page._view._pick_index(lonely, 3, 2) == 0,
                "按色块边缘选择 still finds the block next door",
            )
            # m13246 item 1：深棕色笔画里嵌着的小斑点也要点得到。预览里的圆滑色块把那点
            # 画得比真实像素大，用户点在画出来的斑点上时，脚下最近的像素仍属于周围那根
            # 大笔画，所以边缘选择先在 3 像素以内找「比脚下这一块更小」的块。
            speck = _np.full((5, 5), -1, dtype=_np.int32)
            speck[1:4, 1:4] = 0
            speck[2, 2] = 1
            speck_sizes = _np.asarray(
                [int((speck == 0).sum()), int((speck == 1).sum())], dtype=_np.int64
            )
            page._view.setBlockLabels(speck, speck_sizes)
            page._set_pick_mode(_picture_page.PICK_EDGE, save=False)
            check(
                page._view._pick_index(lonely, 2, 1) == 1,
                "边缘选择：贴着大块的小斑点先被选中（m13246 item 1）",
            )
            check(
                page._view._pick_index(lonely, 1, 1) == 1,
                "边缘选择：小斑点在自己的半径里赢过脚下的大块",
            )
            page._view.setBlockLabels(None, None)
            page._set_pick_mode(_picture_page.PICK_COLOUR, save=False)

            # --- 1.6.19 (m01566): Del is the 删除 button, and 删除/合并 edit 原图 ----
            from PySide6.QtCore import QEvent as _QEvent  # noqa: PLC0415
            from PySide6.QtGui import QKeyEvent as _QKeyEvent  # noqa: PLC0415
            from PySide6.QtTest import QTest as _QTest  # noqa: PLC0415

            check(
                page._delete_button.text() == "删除",
                f"the 删除 button is just 删除 again (m02316 item 3), got "
                f"{page._delete_button.text()!r}",
            )
            check(
                "Del" in _picture_page.PAGE_HELP,
                "… and 帮助 still says Del means 删除",
            )
            page._set_pick_mode(_picture_page.PICK_EDGE, save=False)
            app.processEvents()
            blocks = page._blocks
            # m14748 第 2 条：颜色表列出来的颜色，色块表里也要有一个位子。
            # 去噪并掉的是位置不是颜色 —— 以前一个颜色要是全由小于门槛的碎片组成
            # （那份工程里的红色：474 px 散成 214 片，最大 136 px），色块表里它
            # 会整个消失，而右边「打印图片需要的颜色」照样写着它。
            if blocks is not None and blocks.count:
                printed = {
                    int(value)
                    for value in _np.unique(page._result.indices).tolist()
                    if int(value) >= 0
                }
                in_blocks = {int(value) for value in blocks.values.tolist()}
                check(
                    printed <= in_blocks,
                    "色块表覆盖了颜色表列出的每一种颜色（m14748 第 2 条）："
                    f"印了 {sorted(printed)}，色块表只有 {sorted(in_blocks)}",
                )
            if blocks is not None and blocks.count:
                whole = blocks.mask(blocks.order()[0])
                page._list.clear_selection()
                page._on_region_clicked(blocks.order()[0])
                app.processEvents()
                before_del = page._result.printed_pixels
                page._list.setFocus()
                app.processEvents()
                check(not page._typing_in_field(), "with the list focused Del belongs to 删除")
                # The focus itself is checked in the selftest (this offscreen window is
                # never active, so ``focusWidget()`` here is not the spin box); what
                # matters is that a Del typed into a number box never deletes a block,
                # so hand the page the same guard the field would raise.
                typing = page._typing_in_field
                page._typing_in_field = lambda: True
                try:
                    app.sendEvent(
                        page,
                        _QKeyEvent(
                            _QEvent.Type.KeyPress,
                            Qt.Key.Key_Delete,
                            Qt.KeyboardModifier.NoModifier,
                        ),
                    )
                    app.processEvents()
                finally:
                    page._typing_in_field = typing
                check(
                    page._result.printed_pixels == before_del,
                    "typing Del in a number box does not delete a block",
                )
                page._list.setFocus()
                app.processEvents()
                _QTest.keyClick(page._list._list, Qt.Key.Key_Delete)
                app.processEvents()
                check(
                    page._result.printed_pixels == before_del - int(whole.sum()),
                    f"Del deletes what the button deletes, got "
                    f"{before_del - page._result.printed_pixels} of {int(whole.sum())} pixels",
                )
                check(page._original_edited, "a delete edits 原图 itself")
                check(
                    page._original_badge is not None and page._original_badge.isVisible(),
                    "原图 wears the 原图已修改 badge",
                )
                grid = _picture_page._to_image_grid(
                    _np.asarray(whole, dtype=bool), page._image.size
                )
                check(
                    int(_np.asarray(page._image.convert("RGBA"))[..., 3][grid].max()) == 0,
                    "the deleted pixels are transparent in 原图",
                )
                check(
                    page._original_means(page._result) is not None,
                    "the edited 原图 can still measure every block's own colour",
                )
                page._undo()
                app.processEvents()
                check(
                    not page._original_edited and not page._original_badge.isVisible(),
                    "Ctrl+Z puts 原图 back too, badge and all",
                )
                # The list is not the only door: a Del the page itself receives means
                # 删除 as well (the list can lose the focus to the picture).
                page._list.clear_selection()
                page._on_region_clicked(blocks.order()[0])
                app.processEvents()
                again = page._result.printed_pixels
                app.sendEvent(
                    page,
                    _QKeyEvent(
                        _QEvent.Type.KeyPress,
                        Qt.Key.Key_Delete,
                        Qt.KeyboardModifier.NoModifier,
                    ),
                )
                app.processEvents()
                check(
                    page._result.printed_pixels < again and page._original_edited,
                    "a Del delivered to the page itself still means 删除",
                )
                page._undo()
                app.processEvents()
            page._set_pick_mode(_picture_page.PICK_COLOUR, save=False)
            app.processEvents()
            # m02316 item 2: a colour-level delete edits 原图 too — the pixels become
            # transparent, exactly like a block-level delete.  原图 and 预览 must never
            # tell two different stories about the same picture.
            page._list.select([0])
            colour_image = page._image.tobytes()
            page._on_delete_colours()
            app.processEvents()
            check(
                page._image.tobytes() != colour_image
                and page._original_edited
                and page._original_badge.isVisible(),
                "deleting a colour edits 原图 too, badge and all",
            )
            page._undo()
            app.processEvents()
            check(
                page._image.tobytes() == colour_image and not page._original_edited,
                "Ctrl+Z puts a colour-level delete back as well",
            )
            page._list.select([0])
            wanted = next(
                (
                    entry
                    for entry in page._candidate_palette()
                    if entry.key != page._result.palette[0].key
                ),
                None,
            )
            if wanted is not None:
                picture_before = page._image.tobytes()
                real_picker = _picture_page.ColourPickerDialog
                _picture_page.ColourPickerDialog = lambda *args, **kwargs: _StubDialog(0, wanted)
                try:
                    page._on_replace()
                    app.processEvents()
                finally:
                    _picture_page.ColourPickerDialog = real_picker
                check(
                    page._image.tobytes() == picture_before,
                    "「更换颜色」 leaves 原图 alone",
                )
                page._undo()
                app.processEvents()

            # 1.6.21 (m01191 item 2): 合并 writes the surviving block's OWN colour back into
            # 原图, not the matched roll's colour.  The user's rule: 原图色块 c 并进原图色块 a
            # 之后，c 的像素在原图里用的是 a 自己的颜色——原图的颜色集合一点都不变，
            # 变的只是色块大小，所以「按当前色块重新匹配」拿到的还是原图自己那几种颜色。
            if len(page._result.palette) >= 2:
                merge_picked = page._list.row_ids()[:2]
                # Two colours at once only happen with 「多选」 on (m02289 item 1).
                page._multi_button.setChecked(True)
                page._list.select(merge_picked)
                app.processEvents()
                check(
                    len(page._list.selected_indices()) == 2,
                    "多选 on lets 合并 see both colours: "
                    f"{page._list.selected_indices()}",
                )
                keeper = int(merge_picked[1])
                merge_indices = _np.asarray(page._result.indices)
                touched = _np.isin(
                    merge_indices, _np.asarray(merge_picked, dtype=merge_indices.dtype)
                )
                colours_before_merge = {
                    tuple(int(value) for value in pixel)
                    for pixel in _np.unique(
                        _np.asarray(page._image.convert("RGBA"))[..., :3].reshape(-1, 3),
                        axis=0,
                    )
                }
                keeper_colour = page._picture_colour(merge_indices == keeper)
                before_merge_image = page._image.tobytes()
                real_merge = _picture_page.MergeDialog
                _picture_page.MergeDialog = lambda *args, **kwargs: _StubDialog(1)
                try:
                    page._on_merge_colours()
                    app.processEvents()
                finally:
                    _picture_page.MergeDialog = real_merge
                check(
                    page._original_edited and page._image.tobytes() != before_merge_image,
                    "merging colours edits 原图 too",
                )
                painted = _picture_page._to_image_grid(touched, page._image.size)
                merge_pixels = _np.asarray(page._image.convert("RGBA"))
                merged = merge_pixels[painted][merge_pixels[painted][:, 3] > 0][:, :3]
                check(
                    keeper_colour is not None
                    and merged.size > 0
                    and bool(
                        (merged == _np.asarray(keeper_colour, dtype=merged.dtype)).all()
                    ),
                    "合并 writes the surviving block's own colour into 原图 "
                    f"(expected {keeper_colour})",
                )
                colours_after_merge = {
                    tuple(int(value) for value in pixel)
                    for pixel in _np.unique(
                        _np.asarray(page._image.convert("RGBA"))[..., :3].reshape(-1, 3),
                        axis=0,
                    )
                }
                check(
                    colours_after_merge <= colours_before_merge,
                    "合并 never adds a colour 原图 did not already have "
                    f"({sorted(colours_after_merge - colours_before_merge)} are new)",
                )
                page._undo()
                app.processEvents()
                check(
                    page._image.tobytes() == before_merge_image,
                    "Ctrl+Z puts 原图 back after a merge",
                )
                page._multi_button.setChecked(False)
                app.processEvents()

            # 1.6.21 (m01191 item 1): an empty selection means 预览 draws nothing.  Clicking
            # the blank space under the rows (or unticking the last row in 多选) empties the
            # list without going through any of the paths that used to clear the highlight,
            # so the rule hangs off the selection itself now.
            empty_ids = page._list.row_ids()
            if empty_ids:
                page._list.select([empty_ids[0]])
                app.processEvents()
                check(page._view._dim_pixmap() is not None, "a picked colour is drawn in 预览")
                page._list._list.clearSelection()
                page._list._list.setCurrentRow(-1)
                app.processEvents()
                check(
                    page._view._selected == -1
                    and page._view._dim_pixmap() is None
                    and page._list.current() == -1
                    and not page._delete_button.isEnabled(),
                    "an empty selection takes the highlight off 预览 (点空白处那条路)",
                )
            page._set_pick_mode(_picture_page.PICK_EDGE, save=False)
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
                    and not page._merge_button.isEnabled(),
                    "an empty selection takes the block outline off 预览 too",
                )
                page._set_pick_mode(_picture_page.PICK_COLOUR, save=False)
                app.processEvents()

            # --- the plans the cap allows (1.6.12: 「耗材方案」) -----------------

            # This picture is flat enough to print exactly with four rolls, and
            # then every recipe agrees, so only one plan is worth offering.
            # Squeeze the cap to three so the panel has to show a real choice.
            remembered_cap = page._spool_cap.value()
            page._spool_cap.setValue(3)
            page._on_rematch()
            app.processEvents()
            check(
                bool(page._plans) and page._plan_key == page._plans[0].key,
                "a match offers 耗材方案 and starts on the first one",
            )
            check(page._base is page._plans[0].result, "the preview is that first plan")
            # m06704 item 3 (1.6.27): 没点名的时候用第一行，而第一行现在是「大色差色块
            # 优先」——先把差得最明显的色块认出来，平均差得少不再排前面。
            from app.core import image_matching as _matching  # noqa: PLC0415

            check(
                _matching._budget_objective is _matching._objective_priority,
                "the default yardstick is 「大色差色块优先」",
            )
            check(
                page._plans[0].key == "priority",
                f"the first row is that yardstick, got {page._plans[0].key!r}",
            )
            check(
                page._plans[0].title == _matching._PLAN_SPECS[0][1],
                "the first row is named after it",
            )
            check(
                _matching.plan_help().find(_matching._PLAN_SPECS[0][1]) >= 0,
                "the help text explains which row 重新匹配 lands on",
            )
            check(
                len(page._plans) >= 2,
                f"a tight cap offers several recipes, got {len(page._plans)}",
            )
            cap = page._spool_cap.value()
            over = [
                option.key for option in page._plans if cap and len(option.spools) > cap
            ]
            check(not over, f"no plan asks for more than 耗材上限 {cap}, got {over}")
            check(
                len(page._plans) == len({frozenset(option.spools) for option in page._plans}),
                "two plans never spend the very same rolls",
            )
            check(
                all(option.key != "extra" for option in page._plans),
                "the plan that asked for one roll more is gone",
            )
            # The button group is the live set: `_refresh_plans` rebuilds the rows
            # and forgets the old buttons on the spot, while the old *widgets*
            # linger as children until Qt runs their deferred deletion.
            buttons = page._plan_group.buttons()
            check(
                len(buttons) == len(page._plans),
                f"the panel has one radio per plan, got {len(buttons)} for "
                f"{len(page._plans)} plans",
            )
            check(
                sum(1 for button in buttons if button.isChecked()) == 1,
                "exactly one plan is checked",
            )
            check(
                sorted(button.text() for button in buttons)
                == sorted(option.title for option in page._plans),
                "each radio is named after its plan",
            )
            numbers = [
                label.text() for label in page._plan_box.findChildren(_QLabel)
            ]
            check(
                any("差块" in text and "用 " in text for text in numbers),
                f"each row shows 差块 and the roll count, got {numbers}",
            )
            check(
                all(
                    f"平均 ΔE00 {option.fidelity[0]:.2f}" in " ".join(numbers)
                    for option in page._plans
                ),
                "the average ΔE00 on screen is the plan's own",
            )
            check(
                page._usage.text().startswith("共")
                and all(use.display_name in page._usage.text() for use in _spool_usage(page._base, library)),
                "「用到的耗材」 names every roll of the selected plan",
            )
            check(
                page._plan_hint.text() == "" and not page._plan_hint.isVisibleTo(page),
                f"有方案时面板上不打印解释文本（m01885 #8），got {page._plan_hint.text()!r}",
            )
            # m01885 #1: 每一行自己带着这一套要装的那几卷的颜色方块。
            rows = [
                page._plan_rows_layout.itemAt(index).widget()
                for index in range(page._plan_rows_layout.count())
            ]
            swatches = [
                [label for label in row.findChildren(_QLabel) if label.width() == 14]
                for row in rows
            ]
            check(
                [len(row) for row in swatches] == [len(option.spools) for option in page._plans],
                f"行里的颜色方块数 = 这一套用几卷，got {[len(row) for row in swatches]}",
            )
            check(
                all("备选" not in option.title for option in page._plans),
                f"方案名里没有「（备选 N）」，got {[option.title for option in page._plans]}",
            )
            second = page._plans[1]
            check(
                buttons[1].isVisible() and buttons[1].isEnabled(),
                "the second plan's radio is on screen and clickable",
            )
            _QTest.mouseClick(buttons[1], Qt.MouseButton.LeftButton)
            app.processEvents()
            check(page._plan_key == second.key, "clicking a plan switches to it")
            check(page._base is second.result, "… and the picture follows it")
            check(
                buttons[1].isChecked() and not buttons[0].isChecked(),
                "… and the radio moved with it",
            )
            check(
                all(
                    use.display_name in page._usage.text()
                    for use in _spool_usage(page._base, library)
                ),
                "… and 「用到的耗材」 follows that plan",
            )
            # m00830 item 2: the preview must not keep the old picture's highlight.
            page._select(0)
            app.processEvents()
            check(page._view._selected == 0, "a clicked colour is what the preview highlights")
            _QTest.mouseClick(buttons[0], Qt.MouseButton.LeftButton)
            app.processEvents()
            check(
                page._view._selected == -1
                and page._list.current() == -1
                and not page._list.selected_indices()
                and page._view._dim_pixmap() is None,
                "switching 方案 clears the preview's highlight block and outline "
                f"(view={page._view._selected}, row={page._list.current()})",
            )
            check(
                page._plan_key == page._plans[0].key
                and page._base is page._plans[0].result,
                "clicking the first plan brings the first answer back",
            )
            # m02316 item 7: a project file only remembers the plan it was saved under,
            # so 「打开工程」 used to leave 耗材方案 with that single row and no way to see
            # the other 搭法 short of pressing 「按当前色块重新匹配」——但没人觉得「打开一个
            # 工程」要先按它。打开时顺手把同一批色块的其它搭法补齐，勾还在文件里那一套上。
            plans_project = TEMP / "saved-plans.bpproj"
            real_dialog = _picture_page.QFileDialog
            real_box = _picture_page.QMessageBox
            _picture_page.QFileDialog = _StubChooser(plans_project)
            _picture_page.QMessageBox = _StubMessageBox
            offered = len(page._plans)
            check(offered >= 2, f"先让这张图有几套搭配，才看得出补齐（{offered} 套）")
            page._on_save_project()
            app.processEvents()
            check(page._dirty is False and plans_project.is_file(), "把这一版存成工程")
            _StubMessageBox.answer = "不保存"
            page._on_open_project()
            app.processEvents()
            check(
                len(page._plans) == offered,
                f"打开工程之后方案表把别的搭法补齐了（{len(page._plans)} 行 / 存档时 {offered} 套）",
            )
            check(
                page._plan_key == page._plans[0].key and page._base is page._plans[0].result,
                "勾还在文件里那一套上",
            )
            check(
                len(page._plan_group.buttons()) == len(page._plans),
                "补出来的每一套都有自己的单选钮",
            )
            check(
                all(option.spools for option in page._plans),
                "补出来的方案都说着自己要用哪几卷",
            )
            # m10318 item 2: the row that came out of the file carries numbers from the
            # moment it was saved, so it says so on the row itself — otherwise an
            # upgrade looks like "the match still has not changed".
            check(
                page._plans[0].from_file
                and all(not option.from_file for option in page._plans[1:]),
                "只有文件里那一套被标成「从工程里读的」",
            )
            rows = [
                page._plan_rows_layout.itemAt(index).widget()
                for index in range(page._plan_rows_layout.count())
            ]
            texts = [
                child.text()
                for row in rows
                for child in row.findChildren(_picture_page.QLabel)
                if "从工程里读的那套" in child.text()
            ]
            check(
                len(texts) == 1 and "从工程里读的那套" in texts[0],
                f"那一行自己写着是从工程里读的（{texts}）",
            )
            _StubMessageBox.answer = "清空"
            page._project_path = None
            page._dirty = False
            _picture_page.QFileDialog = real_dialog
            _picture_page.QMessageBox = real_box
            page._spool_cap.setValue(remembered_cap)
            page._on_rematch()
            app.processEvents()
            # m18338 第 2 条把「耗材方案」从「同一把尺子下的前几名」扩成「几档省料」：
            # 一张本来就打得出来的图，多出来的那几行是「少用一个卷」「少用两个卷」，
            # 每一行都还能把整张图打完（省的是料，不是画面），卷的组合各不相同。
            _painted = {int((option.result.indices >= 0).sum()) for option in page._plans}
            check(
                len(page._plans) >= 1
                and page._plan_key == page._plans[0].key
                and len({frozenset(option.spools) for option in page._plans}) == len(page._plans)
                and len(_painted) == 1
                and all(option.spools for option in page._plans),
                f"an exactly printable picture offers the same spread as any other, got "
                f"{len(page._plans)} 行 {[(option.title, len(option.spools)) for option in page._plans]}"
                f" 打印像素 {_painted}",
            )
            check(
                all(not option.from_file for option in page._plans),
                "重新匹配之后就没有「从工程里读的」那一行了",
            )

            # --- 筛选耗材 / 按当前色块重新匹配 (1.6.13 #3/#4) --------------------
            from app.ui.plan_filter import (  # noqa: PLC0415
                FORBIDDEN as _FILTER_FORBIDDEN,
                REQUIRED as _FILTER_REQUIRED,
                SpoolFilterDialog,
            )

            check(
                page._filter_button.isEnabled() and "筛选耗材" == page._filter_button.text(),
                "「筛选耗材」还在「耗材方案」面板上",
            )
            # m15721 第 1 条：从前两个按钮只差「要不要重新分块」，现在只有一个 —— 分块
            # 是软件该自己认的事，颜色重找、色块不动。
            check(
                not hasattr(page, "_rematch_button")
                and page._rematch_blocks_button.text() == "按当前色块重新匹配",
                "「重新匹配」按钮撤了，只剩「按当前色块重新匹配」",
            )
            check(
                "色块的划分和编号都不动" in _picture_page.PAGE_HELP,
                "帮助里写清这个按钮不重新分块",
            )
            check(
                page._filter_hint.text() == "每一卷都可以用。",
                f"没筛选时说清楚，got {page._filter_hint.text()!r}",
            )
            demanded = sorted(page._plans[0].spools)[0]
            banned = next(
                (
                    filament.id
                    for filament in page._library.visible
                    if filament.id not in page._plans[0].spools
                ),
                "",
            )
            dialog = SpoolFilterDialog(
                library.visible,
                [demanded],
                [banned] if banned else [],
                cap=page._spool_cap.value(),
            )
            check(
                dialog.required() == (demanded,)
                and dialog.forbidden() == ((banned,) if banned else ()),
                "筛选窗口按传进去的答案摆好每一行",
            )
            row = dialog._list.item(0)
            dialog._on_row_clicked(row)
            check(
                int(row.data(Qt.ItemDataRole.UserRole + 1))
                in (_FILTER_REQUIRED, _FILTER_FORBIDDEN)
                and row.text().startswith(("必须有 · ", "不要用 · ")),
                f"点一行就换一档并写在行上，got {row.text()[:16]!r}",
            )
            dialog.close()
            page._require = frozenset({demanded})
            page._avoid = frozenset({banned}) if banned else frozenset()
            page._update_filter_hint()
            check(
                demanded in page._filter_hint.text() or page._spool_name(demanded) in page._filter_hint.text(),
                f"筛选条件写在面板上，got {page._filter_hint.text()!r}",
            )
            page._on_rematch()
            app.processEvents()
            check(
                bool(page._plans) and all(demanded in option.spools for option in page._plans),
                "「必须有」那一卷每一套都用上了",
            )
            check(
                not banned or all(banned not in option.spools for option in page._plans),
                "「不要用」那一卷一套都没碰",
            )
            keep = len(page._result.palette)
            page._list.select([0])
            page._on_delete_colours()
            app.processEvents()
            unpainted = page._result.total_pixels - page._result.printed_pixels
            before_plans = [option.key for option in page._plans]
            before_base = page._base
            before_key = page._plan_key
            before_undo = len(page._undo_stack)
            page._on_rematch_blocks()
            app.processEvents()
            check(
                bool(page._plans)
                and len(page._base.palette) < keep
                and page._base.total_pixels - page._base.printed_pixels == unpainted,
                "「按当前色块重新匹配」不再自己加回删掉的色块",
            )
            # m01411 item 2: every block still on screen has to survive the re-match —
            # 1.6.15 answered each block with its own nearest colour, and when those
            # colours needed more rolls than the cap allowed the surplus blocks were
            # folded away instead (the reporter's 3 blocks came back as 1).
            check(
                len(page._base.palette) == keep - 1,
                "it keeps every block that is left, none folded away "
                f"({len(page._base.palette)} of {keep - 1})",
            )
            # m01411 item 1: it replaces the whole match, so it has to be one Ctrl+Z.
            check(
                len(page._undo_stack) == before_undo + 1,
                f"and it is one step of history, not a dead end ({len(page._undo_stack)})",
            )
            check(
                all(demanded in option.spools for option in page._plans),
                "… 也没忘了「必须有」那一卷",
            )
            page._undo()
            app.processEvents()
            check(
                page._base is before_base
                and [option.key for option in page._plans] == before_plans
                and page._plan_key == before_key,
                "Ctrl+Z puts the replaced match, its plans and its ticked row back",
            )
            page._redo()
            app.processEvents()
            page._require = frozenset()
            page._avoid = frozenset()
            page._update_filter_hint()
            page._on_rematch()
            app.processEvents()
            # m02316 item 2: the colour-level delete already took those pixels out of
            # 原图, so a plain 重新匹配 starts from that picture and the colour does NOT
            # come back.  This is what the reporter asked for in 图三: the two views must
            # agree, and a deleted colour must stay deleted.
            opaque = float(
                (_np.asarray(page._image.convert("RGBA"))[..., 3] > 0).mean()
            )
            printed = page._result.printed_pixels / max(1, page._result.total_pixels)
            check(
                not page._deleted
                and len(page._result.palette) < keep
                and abs(printed - opaque) < 0.01,
                "从头匹配也从改过的原图出发，删掉的颜色没有回来"
                f"（不透明 {opaque:.1%}、打印 {printed:.1%}、{len(page._result.palette)} 种 < {keep}）",
            )

            # --- one project file per picture (m08102 item 5) -------------------
            import json  # noqa: PLC0415
            from PySide6.QtGui import QKeySequence as _QKeySequence  # noqa: PLC0415

            project = TEMP / "roundtrip.bpproj"
            _picture_page.QFileDialog = _StubChooser(project)
            _picture_page.QMessageBox = _StubMessageBox
            check(page._project_path is None, "the picture tab starts with no project file")
            # The rolls are read off the match that is on screen right now: the picture
            # has been edited since `uses` was measured (a colour delete edits 原图, so
            # a fresh match no longer brings that colour back).
            saving_uses = _spool_usage(page.result, library)
            page._on_save_project()
            app.processEvents()
            check(project.is_file(), "保存工程 wrote the project file")
            check(zipfile.is_zipfile(project), "a project is a real zip container")
            with zipfile.ZipFile(project) as package:
                members = set(package.namelist())
                check(
                    {"工程.json", "原图.png", "预览.png", "基线.npy"} <= members,
                    f"the project holds the picture, the preview and the document, got {sorted(members)}",
                )
                document = json.loads(package.read("工程.json").decode("utf-8"))
            check(document["app"] == "BambuPalette", "the document names the program")
            check(
                len(document["spools"]) == len(saving_uses),
                "the document lists the spools this picture needs",
            )
            check(
                all(row["color_hex"] for row in document["spools"]),
                "… with their colours, so the file is readable on a machine that lacks them",
            )
            check(page._project_path == project, "the page remembers where it saved")
            check(page._dirty is False, "saving clears the unsaved flag")
            check("已保存工程" in page._status.text(), f"saving says so: {page._status.text()!r}")

            # 1.6.26 (m05557 item 3): 预览.png is the picture the user is looking at.
            # With the box on that is the refined grid the plate is cut from, not the
            # raw match grid — the file and the panel used to disagree with the slicer.
            with zipfile.ZipFile(project) as package:
                from PIL import Image as _SavedPreview
                import io

                saved_preview = _SavedPreview.open(io.BytesIO(package.read("预览.png")))
            check(
                page._smooth_box.isChecked()
                and saved_preview.width == page._view._matched.width(),
                f"预览.png matches the panel ({saved_preview.width} px vs {page._view._matched.width()} px)",
            )

            # 1.6.22 (m01647): the 圆滑色块 tick travels with the project.
            # 1.6.23 (m02289 item 2): so does 图片位置.
            page._smooth_box.setChecked(False)
            page._side_combo.setCurrentIndex(1)
            page._on_save_project()
            app.processEvents()
            page._smooth_box.setChecked(True)
            page._side_combo.setCurrentIndex(0)
            _StubMessageBox.answer = "不保存"
            page._on_open_project()
            app.processEvents()
            check(
                page._smooth_box.isChecked() is False,
                "opening the project puts the 圆滑色块 tick back where it was saved",
            )
            check(
                page._side_combo.currentText() == _picture_page.IMAGE_SIDE_TOP,
                "opening the project puts 图片位置 back where it was saved: "
                f"{page._side_combo.currentText()!r}",
            )
            page._smooth_box.setChecked(True)
            page._side_combo.setCurrentIndex(0)

            saved_colours = len(page._result.palette)
            saved_printed = page._result.printed_pixels
            page._on_colour_selected(0)
            page._on_delete_colours()
            app.processEvents()
            edited_colours = len(page._result.palette)
            edited_printed = page._result.printed_pixels
            check(page._dirty is True, "an edit marks the project unsaved")
            check(edited_colours == saved_colours - 1, "the edit really changed the picture")

            # 取消 on 「要保存吗」 must leave the edit exactly where it is.
            _StubMessageBox.answer = "取消"
            page._on_open_project()
            app.processEvents()
            check(page._project_path == project, "cancelling the question keeps the project")
            check(
                len(page._result.palette) == edited_colours,
                "cancelling does not throw the edit away",
            )
            # 不保存 opens the file — i.e. the picture it was saved with.
            _StubMessageBox.answer = "不保存"
            page._on_open_project()
            app.processEvents()
            check(
                len(page._result.palette) == saved_colours
                and page._result.printed_pixels == saved_printed,
                "打开工程 restores the picture that was saved",
            )
            check(page._dirty is False, "an opened project is not dirty")
            check("已打开工程" in page._status.text(), f"opening says so: {page._status.text()!r}")
            # 先保存 writes the edit first, then reopens it.
            page._on_colour_selected(0)
            page._on_delete_colours()
            app.processEvents()
            _StubMessageBox.answer = "先保存"
            page._on_open_project()
            app.processEvents()
            check(page._dirty is False, "先保存 writes the edit before opening")
            check(
                len(page._result.palette) == saved_colours - 1,
                "… and the reopened project is that edited picture",
            )
            check(
                page._result.printed_pixels != saved_printed or edited_printed == saved_printed,
                "the reopened picture is the one that was written",
            )
            _StubMessageBox.answer = "清空"

            # The save dialog cannot produce a file we are then unable to open.
            _picture_page.QFileDialog = _StubChooser(TEMP / "noext")
            page._on_save_project(ask=True)
            app.processEvents()
            check((TEMP / "noext.bpproj").is_file(), "a name without .bpproj still saves openable")
            check(
                page._save_shortcut.key() == _QKeySequence(_QKeySequence.StandardKey.Save),
                "Ctrl+S is bound to 保存工程",
            )
            check(
                page._save_as_shortcut.key().toString() == "Ctrl+Shift+S",
                f"Ctrl+Shift+S asks for another name, got {page._save_as_shortcut.key().toString()}",
            )
            _picture_page.QFileDialog = original_dialog
            _picture_page.QMessageBox = original_box

            # 1.6.26 (m05557 item 2): a project carrying 色块级的改动 has to survive
            # 「打开 → 再保存」.  1.6.25 read one step of that file
            # ({"mask": …, "entry": …}) as a pair, so the colour came back as the
            # string "entry" and the save died with "'str' object has no attribute
            # 'key'" — which is the 保存工程失败 the user hit.
            block_project = TEMP / "block-edits.bpproj"
            page._set_pick_mode(_picture_page.PICK_EDGE, save=False)
            app.processEvents()
            blocks = page._blocks
            wanted = None
            if blocks is not None and blocks.count:
                # 「删除」 leaves the file with entry=None, which cannot show the bug
                # (it needs a colour object); 「更换颜色」 is what puts one in there.
                page._list.clear_selection()
                page._on_region_clicked(blocks.order()[0])
                app.processEvents()
                page._delete_button.click()
                app.processEvents()
                # 删除会把块重排，块对象和编号都换新的——必须再读一次，否则下面的
                # 编号在新块表里根本不存在，_picked_blocks() 只会空着手回来。
                blocks = page._blocks
                target = blocks.order()[0]
                wanted = next(
                    (
                        entry
                        for entry in page._candidate_palette()
                        if entry.key != page.result.palette[blocks.value(target)].key
                    ),
                    None,
                )
                page._list.clear_selection()
                page._on_region_clicked(target)
                app.processEvents()
                check(
                    bool(page._picked_blocks()[0]),
                    f"there is a block picked to recolour ({target})",
                )
                check(wanted is not None, "… and another colour to replace it with")
                if wanted is not None and page._picked_blocks()[0]:
                    real_picker = _picture_page.ColourPickerDialog
                    _picture_page.ColourPickerDialog = (
                        lambda *args, **kwargs: _StubDialog(0, wanted)
                    )
                    try:
                        page._on_replace()
                        app.processEvents()
                    finally:
                        _picture_page.ColourPickerDialog = real_picker
            check(
                bool(page._block_edits),
                f"a 色块级的改动 is in place ({len(page._block_edits)})",
            )
            page._project_path = block_project
            page._on_save_project()
            app.processEvents()
            check(
                block_project.is_file() and page._dirty is False,
                "a project with 色块级的改动 saves",
            )
            from app.core.project import load_project as _load_project  # noqa: PLC0415

            state = _load_project(block_project, library=library)
            page._adopt_project(state, block_project)
            app.processEvents()
            entries = [edit.entry for edit in page._block_edits]
            check(
                len(entries) == len(state.block_paints)
                and all(entry is None or hasattr(entry, "key") for entry in entries),
                "opening it back keeps those colours as colours: "
                f"{[type(entry).__name__ for entry in entries]}",
            )
            check(
                any(entry is not None for entry in entries),
                "… and one of them is a real colour object, not only a delete",
            )
            page._on_save_project()
            app.processEvents()
            check(page._dirty is False, "saving the reopened project works again")

            # --- m15721 第 1 条：按当前色块重新匹配 —— 色块划分和编号一个都不动 ----
            # 从前这个按钮只重新挑卷、颜色表是现成的（`_rematch_source`）。 现在它先
            # 从改过的原图重新分色，但块还必须留在原处，块级编辑（按像素掩码）才贴得
            # 回去；行级的「更换颜色」按旧表的行号记，新表一换就对不上，清掉。
            page._set_pick_mode(_picture_page.PICK_EDGE, save=False)
            app.processEvents()
            before_blocks = page._blocks
            before_labels = None if before_blocks is None else before_blocks.labels.copy()
            before_count = 0 if before_blocks is None else before_blocks.count
            before_edits = [edit.mask.copy() for edit in page._block_edits]
            # 行级的「更换颜色」只在颜色模式里记：它标的是颜色表的第几行。
            page._set_pick_mode(_picture_page.PICK_COLOUR, save=False)
            app.processEvents()
            # 1.6.56：0 像素的颜色不再列出来，第一行不一定是调色板第 0 位 —— 按第一行
            # 真正顶着的那一位来选，判据才对得上「选中的那一行」。
            first_value = int(page._list._list.item(0).data(Qt.ItemDataRole.UserRole))
            page._list.select([first_value])
            wanted = next(
                (
                    entry
                    for entry in page._candidate_palette()
                    if entry.key != page._result.palette[first_value].key
                ),
                None,
            )
            check(wanted is not None, "有一款和第一行不同的耗材可以换上")
            real_picker = _picture_page.ColourPickerDialog
            _picture_page.ColourPickerDialog = lambda *args, **kwargs: _StubDialog(0, wanted)
            try:
                page._on_replace()
                app.processEvents()
            finally:
                _picture_page.ColourPickerDialog = real_picker
            check(bool(page._replaced), "行级的「更换颜色」记下了（下面要看它被清掉）")
            page._on_rematch_blocks()
            app.processEvents()
            page._set_pick_mode(_picture_page.PICK_EDGE, save=False)
            app.processEvents()
            after_blocks = page._blocks
            check(
                after_blocks is not None
                and after_blocks.count == before_count
                and _np.array_equal(after_blocks.labels, before_labels),
                f"色块的划分和编号一块都没动（{before_count} 块）",
            )
            check(
                not page._replaced,
                f"行级的「更换颜色」清掉了，got {len(page._replaced)} 条",
            )
            check(
                len(page._block_edits) == len(before_edits)
                and all(
                    _np.array_equal(edit.mask, mask)
                    for edit, mask in zip(page._block_edits, before_edits)
                ),
                "块级改动（按像素掩码）原样留着",
            )
            # --- no menu bar left, and 帮助 is a button (m08102 item 6, renamed
            # by m01885 item 5; m01885 items 6/8 moved the explanations into it) ---
            from PySide6.QtWidgets import QMenuBar as _QMenuBar  # noqa: PLC0415

            check(
                not window.findChildren(_QMenuBar),
                "文件 / 视图 / 帮助 are gone — there is no menu bar left",
            )
            check(window._help_button.text() == "帮助", "「帮助」 is a plain button now")
            # 1.6.53（m21616 第 6/8 条）：帮助 / 设置 / 添加耗材从标题栏搬到了
            # 「目标颜色」那一行的右端 —— 判据改成「和颜色输入框同一个父控件」。
            check(
                window._help_button.parentWidget() is window._target_color.parentWidget()
                and window._settings_button.parentWidget() is window._target_color.parentWidget(),
                "帮助 / 设置 / 添加耗材 与「目标颜色」在同一行",
            )
            check(
                "BambuPalette" in window._about_text(),
                "帮助 still says what the program is, and where the algorithms come from",
            )
            window._on_help()
            app.processEvents()
            manual = window._help_window
            check(
                manual is not None and manual.isVisible() and manual.windowTitle() == "帮助",
                "点「帮助」打开一个标题为「帮助」的窗口",
            )
            if manual is not None:
                check(
                    manual.text() == window._help_text()
                    and window._picture_page.help_text() in manual.text(),
                    "窗口正文就是整本说明，图像转换那一段也在里面",
                )
                check(
                    manual._scroll.widget() is manual._body and manual._body.wordWrap(),
                    "正文放在可滚动区域里且自动换行",
                )
                manual.close()
            app.processEvents()

            # A spool can be deleted while its mixture is still in the detail
            # panel — that panel then holds a recipe whose parent is gone, which
            # used to reach the 0→100 % bar as "?" and raise out of hex_to_rgb
            # (清空耗材库 did exactly that).
            window._detail._bar.setPair("?", "?", 50, window._catalog.engine)
            check(
                window._detail._bar._stops == [],
                "a recipe whose parent is gone draws a flat bar instead of raising",
            )

    # --- the picture page only re-matches when it is on screen -----------------
    # At 41 spools a re-match is ~0.7 s against 66,461 candidates; spending it
    # while the 颜色配方 tab is showing buys the user nothing.
    page._stale = False
    window._tabs.setCurrentIndex(0)
    app.processEvents()
    check(not page.isVisible(), "the 颜色配方 tab is the one on screen")
    before_result = page.result
    spare = Filament(brand="大简", material_type="PETG HF", color_hex="#123456")
    library.add(spare)
    window._reload_library()
    app.processEvents()
    check(page._stale, "a library change while the picture tab is hidden defers the re-match")
    check(page.result is before_result, "the deferred re-match leaves the result alone")
    window._tabs.setCurrentIndex(1)
    app.processEvents()
    check(not page._stale, "coming back to the picture tab clears the deferred flag")
    check(
        page.result is not None and page.result is not before_result,
        "the picture is re-matched on the way back to the tab",
    )
    window._tabs.setCurrentIndex(0)
    library.remove(spare.id)
    window._reload_library()
    app.processEvents()

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
    check(
        not bare._clear_button.isEnabled(),
        "清空耗材库 is dead while there is nothing to clear",
    )
    bare.close()

    # --- 清空耗材库 asks first, and really empties this copy's archive ----------
    # m08102 item 1: the spools a fresh copy shows are not a preset, and since
    # 1.6.11 they belong to this copy alone, so the way to start empty is a button.
    from app.ui import main_window as _main_window  # noqa: PLC0415

    check(window._clear_button.isEnabled(), "清空耗材库 is live while there are spools")
    clear_box = _main_window.QMessageBox
    _main_window.QMessageBox = _StubMessageBox
    try:
        _StubMessageBox.answer = "取消"
        before_clear = len(window.library)
        window._on_clear_library()
        app.processEvents()
        check(len(window.library) == before_clear, "取消 keeps every spool")
        _StubMessageBox.answer = "清空"
        window._on_clear_library()
        app.processEvents()
        check(len(window.library) == 0, "清空 really empties the archive")
        check(not window._clear_button.isEnabled(), "the button goes dead afterwards")
        check(not window._list.isVisible(), "the empty list hides itself")
        check(window._library_hint.isVisible(), "the empty hint comes back")
        check(
            "已清空耗材库" in window._status.text(),
            f"it says what happened: {window._status.text()!r}",
        )
        check(len(FilamentLibrary.load()) == 0, "the archive on disk is empty too")
    finally:
        _main_window.QMessageBox = clear_box
        _StubMessageBox.answer = "清空"

    # --- 「导入」opens where the old shared archive lives ------------------------
    # 1.6.11: copies no longer share one library, so a copy that has not been
    # given spools yet offers the old per-machine file's folder — a path Explorer
    # never showed — instead of somewhere unrelated.
    from app.core import paths as _paths_module  # noqa: PLC0415

    legacy_file = TEMP / "legacy-library.json"
    legacy_file.write_text("{}", encoding="utf-8")
    ours_missing = TEMP / "no-library-yet.json"
    ours_missing.unlink(missing_ok=True)
    saved_legacy = _paths_module.legacy_shared_library
    saved_library_path = _paths_module.library_path
    _paths_module.legacy_shared_library = lambda: legacy_file
    _paths_module.library_path = lambda: ours_missing
    try:
        check(
            window._import_start_dir() == legacy_file.parent,
            "导入 opens at the old shared archive while this copy has none",
        )
        ours_missing.write_text("{}", encoding="utf-8")
        check(
            window._import_start_dir() == TEMP,
            "and uses this copy's own folder once it has an archive",
        )
    finally:
        _paths_module.legacy_shared_library = saved_legacy
        _paths_module.library_path = saved_library_path
        ours_missing.unlink(missing_ok=True)

    # --- 1.6.51：图像转换改成中键平移；模型配色页 -------------------------------
    import json as _json  # noqa: E402
    import struct as _struct  # noqa: E402

    from PIL import Image as _PILImage  # noqa: E402

    from app.ui.model_page import ModelPage  # noqa: E402
    from app.ui.picture_page import PicturePage  # noqa: E402

    pan_file = TEMP / "pan-probe.png"
    _PILImage.new("RGB", (400, 300), (30, 120, 220)).save(pan_file)
    # 平移要用主窗口里那一个（它才带耗材库）；空库的 PicturePage 连匹配都算不出来。
    for pan_name, pan_hex in (("探针白", "#F2F0EB"), ("探针蓝", "#2F5BB8")):
        window.library.create(brand="探针", material_type="PETG HF", color_hex=pan_hex, name=pan_name)
    window._reload_library()
    pan_page = window._picture_page
    pan_page.resize(240, 200)
    pan_page.show()
    pan_page.loadImage(pan_file)
    for _ in range(6):
        app.processEvents()
    check(pan_page._result is not None, "图像转换：探针图匹配出来了")
    pan_view = pan_page._view
    pan_view.zoom_by(2.5, QPointF(120, 100))
    app.processEvents()
    start_pan = QPointF(pan_view._pan)

    def _send(widget, event_type, x, y, button, buttons, modifier=Qt.KeyboardModifier.NoModifier):
        event = QMouseEvent(event_type, QPointF(x, y), QPointF(x, y), button, buttons, modifier)
        if event_type == QEvent.Type.MouseMove:
            widget.mouseMoveEvent(event)
        elif event_type == QEvent.Type.MouseButtonRelease:
            widget.mouseReleaseEvent(event)
        else:
            widget.mousePressEvent(event)

    _send(pan_view, QEvent.Type.MouseButtonPress, 100, 80, Qt.MouseButton.MiddleButton, Qt.MouseButtons.MiddleButton)
    check(pan_view._panning, "图像转换：按住中键开始平移")
    _send(pan_view, QEvent.Type.MouseMove, 124, 96, Qt.MouseButton.NoButton, Qt.MouseButtons.MiddleButton)
    _send(pan_view, QEvent.Type.MouseButtonRelease, 124, 96, Qt.MouseButton.MiddleButton, Qt.MouseButtons.NoButton)
    check(not pan_view._panning, "图像转换：松开中键结束平移")
    check(pan_view._pan != start_pan, f"图像转换：中键拖拽把视图平移了 {start_pan} -> {pan_view._pan}")
    check(not hasattr(pan_view, "_dragged"), "图像转换：左键不再兼任平移")
    left_pan = QPointF(pan_view._pan)
    _send(pan_view, QEvent.Type.MouseButtonPress, 60, 60, Qt.MouseButton.LeftButton, Qt.MouseButtons.LeftButton)
    _send(pan_view, QEvent.Type.MouseMove, 90, 90, Qt.MouseButton.NoButton, Qt.MouseButtons.LeftButton)
    _send(pan_view, QEvent.Type.MouseButtonRelease, 90, 90, Qt.MouseButton.LeftButton, Qt.MouseButtons.NoButton)
    check(pan_view._pan == left_pan, "图像转换：左键拖不改平移")

    def _write_probe_glb(path) -> None:
        corners = np.array(
            [
                [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
                [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
            ],
            dtype="<f4",
        )
        faces = np.array(
            [
                [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1],
                [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
            ],
            dtype="<u4",
        )
        blob = bytearray(corners.tobytes()) + bytearray(faces.tobytes())
        views = [
            {"buffer": 0, "byteOffset": 0, "byteLength": int(corners.nbytes)},
            {"buffer": 0, "byteOffset": int(corners.nbytes), "byteLength": int(faces.nbytes)},
        ]
        accessors = [
            {"bufferView": 0, "componentType": 5126, "count": int(len(corners)), "type": "VEC3"},
            {"bufferView": 1, "componentType": 5125, "count": int(faces.size), "type": "SCALAR"},
        ]
        document = {
            "asset": {"version": "2.0"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0, "name": "probe"}],
            "meshes": [
                {
                    "name": "probe",
                    "primitives": [
                        {"attributes": {"POSITION": 0}, "indices": 1, "material": 0}
                    ],
                }
            ],
            "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [0.8, 0.06, 0.06, 1.0]}}],
            "accessors": accessors,
            "bufferViews": views,
            "buffers": [{"byteLength": len(blob)}],
        }
        payload = _json.dumps(document).encode("utf-8")
        while len(payload) % 4:
            payload += b" "
        while len(blob) % 4:
            blob += b"\x00"
        total = 12 + 8 + len(payload) + 8 + len(blob)
        path.write_bytes(
            _struct.pack("<4sII", b"glTF", 2, total)
            + _struct.pack("<II", len(payload), 0x4E4F534A)
            + payload
            + _struct.pack("<II", len(blob), 0x004E4942)
            + bytes(blob)
        )

    check(window._tabs.count() == 3, f"主窗口有 3 个页签，got {window._tabs.count()}")
    check(window._tabs.tabText(2) == "模型配色", f"第三个页签是「模型配色」，got {window._tabs.tabText(2)!r}")
    from app.core.mixes import MixCatalog as _MixCatalog  # noqa: E402
    from app.core.library import FilamentLibrary as _FilamentLibrary  # noqa: E402

    probe_library = _FilamentLibrary()
    for name, hex_value in (
        ("探针白", "#F2F0EB"),
        ("探针黑", "#17181C"),
        ("探针红", "#C8342E"),
        ("探针绿", "#2FA84F"),
    ):
        probe_library.create(brand="探针", material_type="PETG HF", color_hex=hex_value, name=name)
    probe_catalog = _MixCatalog(probe_library.visible).build()
    model_page = ModelPage()
    model_page.setLibrary(probe_library, probe_catalog)
    glb_file = TEMP / "probe-model.glb"
    _write_probe_glb(glb_file)
    check(model_page.loadModel(glb_file), "模型配色页能打开 GLB")
    loaded_model = model_page.model
    check(loaded_model is not None and loaded_model.triangle_count == 12,
          f"GLB 读出 12 个三角形，got {None if loaded_model is None else loaded_model.triangle_count}")
    check(loaded_model is not None and len(loaded_model.primitives) == 1, "GLB 读出 1 个图元")
    model_match = model_page.match
    check(model_match is not None and len(model_match.colours) >= 1, "读出了模型上的颜色")
    preview_hexes = model_page.previewHexes()
    check(len(preview_hexes) == len(model_match.colours) if model_match else False,
          "预览色数等于配色色数")
    check(all(len(value) == 7 and value.startswith("#") for value in preview_hexes),
          f"预览色都是 #RRGGBB：{preview_hexes}")
    check(model_match is not None and model_match.spool_count >= 1, "配色至少用了一卷耗材")
    # 1.6.53（m21616 第 13 条）：那句「N 种颜色，平均 ΔE00 …」搬进了「耗材方案」，
    # 状态行现在只说「左边是原模型、右边是预览」。
    # 1.6.58（m26604 第 6 条）：那句话按用户要求整条撤掉，方案行自己带数字。
    check(not model_page._summary.isVisible(), "「N 种颜色，平均 ΔE00 …」那句不再显示")
    check("三角形" in model_page._model_label.text(), "模型名字行带三角形数")
    # 1.6.58（m26604 第 9 条）：`_export_button` 这个名字现在给了「导出」下拉按钮
    # （GLB / OBJ + MTL / Blender），所以旧名字要换个判据：配色表那一颗是彻底没了。
    check(not hasattr(model_page, "_export_table"), "「导出配色表」按钮已经删了（m21496 第 9 条）")
    check(isinstance(model_page._source_view.hasOpenGL(), bool), "三维视图能回答有没有 OpenGL")
    check(isinstance(model_page._source_view.available(), str), "三维视图能给出不可用原因")

    # --- 1.6.53（m21616）：整套界面不再弹黄色悬停框 --------------------------
    from app.ui import tips as _tips

    check(_tips.tooltips_disabled(), "悬停提示框全程序关掉了（ToolTip 事件被吃掉）")
    check(model_page._source_view.flipOrbit() is True, "模型配色页默认左右翻转")
    # 1.6.57（m26286 第 1 条）：方案改成图像转换页那种「圆钮 + 数字 + 卷的颜色」的行。
    from PySide6.QtWidgets import QLabel as _PlanLabel  # noqa: PLC0415
    from PySide6.QtWidgets import QRadioButton as _PlanRadio  # noqa: PLC0415

    _rows = [
        model_page._plan_rows_layout.itemAt(index).widget()
        for index in range(model_page._plan_rows_layout.count())
    ]
    _rows = [row for row in _rows if row is not None]
    check(len(_rows) == len(model_page._plans),
          f"耗材方案行数等于方案数，got {len(_rows)} / {len(model_page._plans)}")
    if _rows:
        _texts = [label.text() for label in _rows[0].findChildren(_PlanLabel)]
        check(any("ΔE00" in text and "卷" in text for text in _texts),
              f"方案行写的是数字：{_texts!r}")
        check(bool(_rows[0].findChildren(_PlanRadio)), "方案行是圆钮（跟图像转换页一致）")

    # --- 1.6.52：只匹配耗材 / 筛选 / 换色撤销 / 三个出口 -----------------------
    check(model_page._colour_count.value() == 32,
          f"颜色数量默认 32，got {model_page._colour_count.value()}")
    check(model_page._only_spools.isChecked() is False, "「只匹配耗材」默认不勾")
    mixed_kinds = {item.entry.kind for item in model_match.colours} if model_match else set()
    model_page._only_spools.setChecked(True)
    plain_match = model_page.match
    plain_kinds = {item.entry.kind for item in plain_match.colours} if plain_match else set()
    check(plain_kinds <= {"filament"}, f"只匹配耗材时全是本色，got {plain_kinds}")
    check("只用耗材" in model_page._filter_hint.text(),
          f"筛选行写明只用耗材：{model_page._filter_hint.text()!r}")
    model_page._only_spools.setChecked(False)
    check(model_page.match is not None and
          {item.entry.kind for item in model_page.match.colours} == mixed_kinds,
          "取消勾选后候选恢复成含混色的那一份")
    check(model_page._filter_button.isEnabled(), "「筛选耗材」可用")

    first_hex = model_page.match.colours[0].color_hex if model_page.match else ""
    pool = model_page._candidate_palette()
    spare = next((entry for entry in pool if entry.color_hex != model_page.override_of(first_hex)), None)
    check(spare is not None, "调色板里找得到另一种颜色")
    if spare is not None:
        model_page.replace_colour(first_hex, spare)
        check(model_page.override_of(first_hex) == spare.color_hex, "换色记在了这一种颜色上")
        check(model_page.previewHexes()[0] == spare.color_hex, "预览立刻跟着换")
        check(model_page.undo(), "撤销可用")
        check(model_page.override_of(first_hex) is None, "撤销把换色退回去了")
        check(model_page.redo(), "恢复可用")
        check(model_page.override_of(first_hex) == spare.color_hex, "恢复把换色又拿回来")

    plate = model_page.plate()
    check(plate is not None and plate.parts, "模型能摊成板")
    if plate is not None:
        check(plate.width_mm > 0 and plate.depth_mm > 0,
              f"板有尺寸：{plate.width_mm:.1f} × {plate.depth_mm:.1f} mm")
    check(model_page.glb_payload()[:4] == b"glTF", "导出的 GLB 有 glTF 头")
    # 1.6.58（m26604 第 9 条）：导出改成一颗按钮 + 下拉，三项都在。
    check(model_page._export_button.text() == "导出", "导出是一颗按钮")
    check(
        [action.text() for action in model_page._export_button.menu().actions()][:3]
        == ["导出 GLB（.glb，能再导入回来接着改）", "导出 OBJ + MTL", "导出 Blender（.blend）"],
        f"下拉三项：{[a.text() for a in model_page._export_button.menu().actions()]}",
    )
    check(model_page._export_3mf.text().startswith("生成 3MF"), "有生成 3MF 的按钮")
    check(
        model_page._export_button.height() == model_page._export_3mf.height(),
        f"底排按钮等高：{model_page._export_button.height()} / {model_page._export_3mf.height()}",
    )
    check(
        all(button.height() == model_page._export_button.height() for button in model_page._bottom_buttons),
        "底排每一颗（含两排视角按钮）都是同一个高度："
        + str(sorted({button.height() for button in model_page._bottom_buttons})),
    )
    check(model_page._save_shortcut.key().toString() in ("Ctrl+S", "Ctrl+S "),
          f"Ctrl+S 绑在导出 GLB 上，got {model_page._save_shortcut.key().toString()!r}")

    shot = window._picture_page
    shot._only_spools.setChecked(True)
    plain_palette = shot._candidate_palette()
    check({entry.kind for entry in plain_palette} <= {"filament"},
          "图像转换页勾上只匹配耗材后候选里没有混色")
    shot._only_spools.setChecked(False)
    check(any(entry.kind != "filament" for entry in shot._candidate_palette()),
          "图像转换页取消勾选后混色回来了")

    # --- 1.6.54（m22211）----------------------------------------------------
    # 第 1 条：那条只剩标题的顶栏撤了；第 3 条：模型页的状态行默认收起；
    # 第 2 条：右栏是一个纵向分割条；第 6 条：底色可换；第 7 条：设置里的快捷键与撞车检查。
    from PySide6.QtCore import Qt as _Qt  # noqa: E402, PLC0415
    from PySide6.QtGui import QShortcut as _QShortcut  # noqa: E402, PLC0415
    from PySide6.QtWidgets import QSplitter as _QSplitter, QWidget as _QWidget  # noqa: E402, PLC0415

    from app.ui.settings_dialog import SettingsDialog  # noqa: E402, PLC0415

    check(window.findChild(_QWidget, "topbar") is None, "顶栏那个大标题撤了")
    check("左边的原模型" not in model_page._status.text(),
          f"模型页不再常驻那句「左边的原模型…」（现在是 {model_page._status.text()!r}）")
    model_page._say("")
    check(model_page._status.isHidden(), "状态行空着的时候整行收起来")
    vertical = [
        splitter
        for splitter in model_page.findChildren(_QSplitter)
        if splitter.orientation() == _Qt.Orientation.Vertical
    ]
    check(bool(vertical), "模型页右栏是可拖的纵向分割条")
    check(model_page.background_hex() == "#B4C6D2",
          f"三维视图默认底色是 SolidWorks 的浅蓝灰，got {model_page.background_hex()!r}")
    model_page._apply_background("#101010")
    check(model_page.background_hex() == "#101010", "底色换得掉")
    model_page._apply_background("#b4c6d2")
    check(model_page.background_hex() == "#B4C6D2", "底色换得回来")
    settings_dialog = SettingsDialog()
    check(settings_dialog.conflicts() == "", "出厂快捷键同页不撞车")
    settings_dialog._shortcut_pic_import.setKeySequence(settings_dialog._shortcut_pic_rematch.keySequence())
    check("图像转换" in settings_dialog.conflicts(), "同页两个动作抢一个键会被指出来")
    settings_dialog.close()
    window_keys = {
        shortcut.key().toString()
        for shortcut in model_page.findChildren(_QShortcut)
        if shortcut.key().toString()
    }
    check({"Ctrl+O", "Ctrl+R"} <= window_keys, f"模型页打开了 Ctrl+O / Ctrl+R：{sorted(window_keys)}")

    # --- 1.6.55（m25568）----------------------------------------------------
    # 第 1 条：没打开模型时也能改颜色数量 / 耗材上限；第 2 条：两个视图套同款卡片；
    # 第 3 条：GLB 里记着方案与换色，再导入摆回来；第 4 条：能导入 3MF。
    import json as _json  # noqa: E402, PLC0415
    import zipfile as _zipfile  # noqa: E402, PLC0415
    from pathlib import Path as _Path  # noqa: E402, PLC0415
    from tempfile import TemporaryDirectory as _TemporaryDirectory  # noqa: E402, PLC0415

    from app.ui.model_page import MODEL_FILTER as _MODEL_FILTER, ModelPage as _ModelPage  # noqa: E402, PLC0415

    fresh = _ModelPage()
    check(fresh._colour_count.isEnabled(), "模型页没打开模型时「颜色数量」也能改")
    check(fresh._spool_cap.isEnabled(), "模型页没打开模型时「耗材上限」也能改")
    fresh._colour_count.setValue(20)
    check(fresh._colour_count.value() == 20, "没模型时改得动，不会弹错")
    check(fresh._source_view.parent().objectName() == "panel", "原模型视图套在同一款白底卡片里")
    check(fresh._preview_view.parent().objectName() == "panel", "预览视图套在同一款白底卡片里")
    check(fresh._open_button.text() == "打开模型", "按钮改成「打开模型」，不再只认 GLB")
    check("3mf" in _MODEL_FILTER, f"文件对话框认 3MF：{_MODEL_FILTER!r}")
    fresh.close()

    with _TemporaryDirectory() as folder:
        folder_path = _Path(folder)
        three_mf = folder_path / "two.3mf"
        payload = (
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
            "</resources><build><item objectid=\"1\"/><item objectid=\"2\"/></build></model>"
        )
        with _zipfile.ZipFile(three_mf, "w", _zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("[Content_Types].xml", "<Types/>")
            archive.writestr("3D/3dmodel.model", payload)
        check(model_page.loadModel(three_mf), "模型配色页能直接打开 3MF")
        check(model_page.model is not None and model_page.model.triangle_count == 4,
              "3MF 里的 4 个三角形都读到了")
        check(len(model_page.model.primitives) == 2, "3MF 的两种颜色分成两块")
        check(bool(model_page.previewHexes()), "3MF 的模型也能配色")

        # GLB 往返：方案、颜色、设置都要跟着文件走
        before_hexes = model_page.previewHexes()
        before_plan = model_page._match.key if model_page._match else ""
        before_count = model_page._colour_count.value()
        exported = folder_path / "round.glb"
        exported.write_bytes(model_page.glb_payload())
        model_page._colour_count.setValue(8)
        check(model_page.loadModel(exported), "导出的 GLB 能再导入回来")
        check(model_page.previewHexes() == before_hexes, "再导入后每一块的颜色和导出前一致")
        check(model_page._match is not None and model_page._match.key == before_plan,
              "再导入后还停在导出前那一套方案上")
        check(model_page._colour_count.value() == before_count,
              f"再导入后颜色数量摆回 {before_count}（现在是 {model_page._colour_count.value()}）")
        check(model_page.model is not None and model_page.model.saved is not None,
              "读回来的模型里带着 extras 记录")

    model_page.close()

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
    _code = main()
    # Same reason as app/selftest.py: once a window has adopted a project,
    # letting the interpreter tear Qt down afterwards exits with 0xC0000409
    # (STATUS_STACK_BUFFER_OVERRUN) on a good share of runs.  That happens
    # *after* the verdict is printed — and a harness that says "SMOKE OK" and
    # then dies is worse than no harness — so this path leaves through the
    # front door.
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:  # pragma: no cover - no streams in a windowed build
        pass
    os._exit(_code)

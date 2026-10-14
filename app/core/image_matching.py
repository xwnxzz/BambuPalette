"""Match an image's colours against the filament + mix palette.

The workflow the user asked for is: import a picture, let the program find which
of *their* spools (and which of the 81-ratio mixes) reproduce it, then print the
picture flat.  A printer has a handful of extruders, so the picture first has to
be reduced to a small set of printable colours and each of those colours has to
be answered with a filament or a two-filament recipe.

Everything here is deliberately deterministic: the same picture and the same
library always produce the same palette and the same pixel assignments.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
from math import comb
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

from ..spectral import color as _color
from .library import Filament, FilamentLibrary
from .mixes import (
    SORT_LABEL,
    SORT_PAIR,
    SORT_RGB,
    SORT_SIMILARITY,
    MixCatalog,
    MixRecipe,
    filament_sort_key,
    recipe_sort_key,
)

ProgressFn = Callable[[str, float], None]

#: How many candidates the cheap Lab-distance pass keeps before the exact
#: CIEDE2000 pass runs.  Pure Lab distance is a good but not perfect proxy, and
#: CIEDE2000 is far too slow to evaluate against a palette of tens of thousands.
#: Twenty-four was too few: on the 41-spool library it missed the true nearest
#: colour by up to 2.673 ΔE00 — a difference anyone can see — so the pass now
#: keeps 4,096 rows, where the worst miss measured against an exact scan of the
#: same pool is 0.010 ΔE00, for 1.35× the search time (m03703 第 1 条).
_SHORTLIST = 4096

#: ``rows × colours`` below which the search skips the shortlist altogether and
#: measures every row with CIEDE2000, so its answer *is* the global optimum.  The
#: two-filament pool is ~60k rows and lands here; the four-million-row
#: three-filament pool cannot, because measuring all of it honestly takes 29 s
#: for a nineteen-colour picture (m03703 第 1 条).
_EXACT_PAIRS = 1 << 22

#: Rows walked per chunk by that exact scan.
_EXACT_BLOCK = 1 << 15

#: Dtype of the per-pixel palette index.  It must be wide enough for the WHOLE
#: palette, not just for the colours a picture happens to use: a 41-spool
#: library already has 66,461 entries, so an ``int16`` index silently wraps
#: past 32,767 into a negative number — which reads as "transparent" and quietly
#: deletes that colour from the plate.  That is how a seven-colour pig came out
#: with two colours.
_INDEX_DTYPE = np.int32

#: The exact-colour pass is a Python loop of CIEDE2000 calls and is by far the
#: longest step of a match (1.3–1.5 s for a 41-spool library).  Reporting every
#: ``_PROGRESS_ROWS`` colours is what keeps a progress bar moving during it
#: instead of standing still for a second and a half.
_PROGRESS_ROWS = 4

#: Same idea for 「按当前色块重新匹配」: a block may have up to
#: :data:`_BUDGET_ROLL_LIMIT` rolls to spend, which is 253 roll sets to score it
#: against, and the alternatives step has to say something inside that walk too.
_PROGRESS_ROLL_SETS = 16


def _report(progress: ProgressFn | None, message: str, fraction: float) -> None:
    if progress is not None:
        progress(message, fraction)


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PaletteEntry:
    """One printable colour: a spool on its own, or a mixed colour.

    A mixed colour is ONE entry however many recipes reach it: the grid merges
    identical ``#RRGGBB`` values (m06962), so offering the same colour twice to
    the matcher or to the 更换颜色 list would only waste rows.  ``recipe`` stays
    the first recipe (what the compact label shows) and ``recipes`` carries all
    of them.
    """

    key: str
    color_hex: str
    rgb: tuple[int, int, int]
    lab: np.ndarray
    label: str
    kind: str  # "filament" | "mix"
    filament_ids: tuple[str, ...]
    recipe: MixRecipe | None = None
    filament: Filament | None = None
    recipes: tuple[MixRecipe, ...] = ()
    #: The recipe as ``((filament_id, percent), …)``, used when ``recipe`` cannot
    #: describe it: a three-filament mix has no ``MixRecipe`` (it is a
    #: :class:`~app.core.triples.TripleRecipe`), so the UI reads the parts for
    #: two and three parents alike instead of growing a second code path.
    parts: tuple[tuple[str, int], ...] = ()

    @property
    def is_mix(self) -> bool:
        return self.kind == "mix"

    @property
    def recipe_count(self) -> int:
        if self.recipes:
            return len(self.recipes)
        return 1 if self.recipe is not None else 0

    @property
    def ratio_text(self) -> str:
        if len(self.recipes) > 1:
            return f"{len(self.recipes)} 个配方"
        if self.recipe is not None:
            return self.recipe.ratio_text
        if self.parts:
            return " + ".join(f"{percent}%" for _parent, percent in self.parts)
        return ""

    @property
    def short_label(self) -> str:
        if self.filament is not None:
            return self.filament.display_name
        if self.recipe is not None:
            return f"{self.recipe.percent_a}% + {self.recipe.percent_b}%"
        if self.parts:
            return " + ".join(f"{percent}%" for _parent, percent in self.parts)
        return self.color_hex


def build_palette(
    library: FilamentLibrary,
    catalog: MixCatalog | None = None,
    *,
    include_mixes: bool = True,
) -> list[PaletteEntry]:
    """Every colour the user can print: each spool, then every distinct mix.

    Spools come first so that, when a mix lands exactly on a spool colour, the
    cheaper answer (one spool) wins the tie.

    Spools the user hid (料用完了，还没买回来) are left out entirely — see
    :attr:`FilamentLibrary.visible`.  The catalogue already excludes them as
    ingredients, so without this the hidden spool's own colour would still be
    offered as a printable colour, which is the one thing 「隐藏」 promises not
    to happen.

    ``visible`` also carries the 材质 filter (m05557 item 1): switching the
    材质 picker must move every colour on every page, and this palette is where
    「全部颜色」 and the 色块 colour picker read theirs from.
    """
    entries: list[PaletteEntry] = []
    for filament in library.visible:
        entries.append(
            PaletteEntry(
                key=f"filament:{filament.id}",
                color_hex=filament.color_hex,
                rgb=filament.rgb,
                lab=_color.lab_from_rgb(filament.rgb),
                label=filament.display_name,
                kind="filament",
                filament_ids=(filament.id,),
                filament=filament,
            )
        )

    if include_mixes and catalog is not None:
        for colour in catalog.colours:
            recipe = colour.first
            a = library.get(recipe.a_id)
            b = library.get(recipe.b_id)
            if a is None or b is None:
                continue
            # 配方表是照 visible 建的，但手里的这份可能是上一轮的（切换材质后还
            # 没重建）：宁可少给一个颜色，也别给一个打印不出来的颜色。
            if a.hidden or b.hidden:
                continue
            if not library.accepts(a) or not library.accepts(b):
                continue
            entries.append(
                PaletteEntry(
                    key=f"colour:{colour.color_hex}",
                    color_hex=colour.color_hex,
                    rgb=tuple(int(v) for v in colour.rgb),
                    lab=np.asarray(colour.lab, dtype=np.float64),
                    label=(
                        f"{a.display_name} {recipe.percent_a}% + "
                        f"{b.display_name} {recipe.percent_b}%"
                    ),
                    kind="mix",
                    filament_ids=(recipe.a_id, recipe.b_id),
                    recipe=recipe,
                    recipes=tuple(colour.recipes),
                )
            )
    return entries


def _entry_sort_key(entry: PaletteEntry, key: str, target_rgb):
    if entry.recipe is not None:
        return recipe_sort_key(key, entry.recipe, target_rgb=target_rgb)
    if entry.filament is not None:
        return filament_sort_key(key, entry.filament, target_rgb)
    return entry.rgb


def sort_palette(
    entries: Sequence[PaletteEntry], key: str = SORT_RGB, *, target_rgb=None
) -> list[PaletteEntry]:
    """Order 「全部颜色」 by the same keys the 颜色配方 grid offers.

    The list mixes raw spools with mixes, so the shapes of the two underlying
    sort keys have to line up. RGB / lightness / hue / similarity already do;
    「按名称」 and 「按母材组合」 mean different things for a spool and a mix, so
    those two group the spools first and compare inside each group.
    """
    items = list(entries)
    if key == SORT_SIMILARITY:
        # A 41-spool library is 66,461 entries, and this runs on every keystroke
        # of the picker's search box, so the distances are computed in one
        # vectorised CIEDE2000 call instead of 66,461 scalar ones.
        if target_rgb is None:
            raise ValueError(
                f"sort key {SORT_SIMILARITY!r} needs a target colour; pass target_rgb="
            )
        if not items:
            return []
        target_lab = _color.lab_from_rgb(tuple(target_rgb))
        labs = np.asarray([entry.lab for entry in items], dtype=np.float64)
        distances = np.asarray(_color.delta_e_2000(labs, target_lab), dtype=np.float64)
        order = np.argsort(distances, kind="stable")
        return [items[int(index)] for index in order]
    if key == SORT_LABEL:
        return sorted(
            items,
            key=lambda entry: (
                0 if entry.recipe is None else 1,
                entry.label.casefold(),
                entry.color_hex,
            ),
        )
    if key == SORT_PAIR:
        return sorted(
            items,
            key=lambda entry: (
                0 if entry.recipe is None else 1,
                0 if entry.recipe is None else entry.recipe.pair_index,
                0 if entry.recipe is None else entry.recipe.percent_a,
            ),
        )
    return sorted(items, key=lambda entry: _entry_sort_key(entry, key, target_rgb))


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MatchSettings:
    """Knobs the user controls from the picture page."""

    max_colours: int = 12
    max_dimension: int = 512
    alpha_threshold: int = 8
    dither: bool = False
    # Matched colours closer than this ΔE00 are merged into the most-used one.
    # Without it two spools a hair apart both claim slices of the same flat
    # region, the plate comes out speckled, and a filament change is spent on a
    # difference nobody can see.
    merge_delta_e: float = 2.0
    # Radius of the mean filter applied before the colour count is reduced.
    # Generated illustrations carry a few units of paper texture over their
    # flat fills; a quantiser fed that noise spends nearly every colour slot on
    # ten shades of the same beige and drops the small dark details — on the
    # bundled pig the eyes and the brown hair simply vanished.  Averaging the
    # texture away first leaves the slots for colours that are actually
    # different.  0 turns it off.
    smooth_radius: int = 2
    #: How many spools the whole plan may use.  A picture with 30 colours can
    #: still be printed from four spools if the mixes are chosen well, and the
    #: user who owns four spools asked for exactly that: keep the biggest
    #: regions, and never introduce a spool the budget cannot pay for.  0 means
    #: no limit.
    max_spools: int = 0

    def clamped(self) -> "MatchSettings":
        # Imported here rather than at module level: ``app.mesh.plate`` imports
        # this module, so a top-level import would close the loop.
        from ..mesh.threemf import MAX_PAINTED_FILAMENTS

        return replace(
            self,
            max_colours=max(2, min(MAX_PAINTED_FILAMENTS, int(self.max_colours))),
            max_dimension=max(32, min(4096, int(self.max_dimension))),
            alpha_threshold=max(0, min(255, int(self.alpha_threshold))),
            merge_delta_e=max(0.0, min(10.0, float(self.merge_delta_e))),
            smooth_radius=max(0, min(8, int(self.smooth_radius))),
            max_spools=max(0, min(MAX_PAINTED_FILAMENTS, int(self.max_spools))),
        )


@dataclass
class MatchResult:
    """A picture reduced to printable colours."""

    width: int
    height: int
    palette: list[PaletteEntry]
    indices: np.ndarray  # (H, W) int32, -1 where transparent
    counts: np.ndarray  # (len(palette),) pixel counts, largest first
    settings: MatchSettings
    source: str = ""
    # Mean sRGB of the picture's own pixels in each colour's region, shape
    # (len(palette), 3). This is the "图片中这个色块的颜色" the user compares the
    # matched spool/mix against, so it must travel with the result rather than be
    # recomputed from a differently-sized image later.
    means: np.ndarray | None = None
    # ΔE00 of every colour the picture was reduced to against the colour this
    # plan prints for it, and the pixels behind each of those colours, both shape
    # (kept colours,).  ``means`` answers "what colour is this block" for the
    # preview; these two answer "how wrong is this plan" at the resolution the
    # picture was reduced to, so painting a thin brown line with the background
    # costs what it costs instead of hiding inside a block average.
    source_values: np.ndarray | None = None
    source_counts: np.ndarray | None = None

    @property
    def total_pixels(self) -> int:
        return int(self.indices.size)

    @property
    def printed_pixels(self) -> int:
        return int(self.counts.sum())

    @property
    def coverage(self) -> float:
        if not self.total_pixels:
            return 0.0
        return self.printed_pixels / float(self.total_pixels)

    def region(self, index: int) -> np.ndarray:
        return self.indices == index

    def region_colour(self, index: int) -> tuple[int, int, int]:
        """The picture's own colour for ``index``, falling back to the match."""
        if self.means is not None and 0 <= index < len(self.means):
            return tuple(int(value) for value in self.means[index])
        return tuple(self.palette[index].rgb)

    def shares(self) -> np.ndarray:
        return self.counts / float(max(1, self.printed_pixels))


@dataclass(frozen=True)
class SpoolUse:
    """One spool a match actually consumes, and how much of it.

    ``share`` is a fraction of the material laid down, not of the pixels: a
    region printed with 「白 10% + 金 90%」 spends ten times as much 金 as 白, so
    counting regions (or pixels) per colour answers a different question than
    「这一版要拆哪几卷」.  See :func:`spool_usage`.
    """

    filament_id: str
    share: float
    filament: Filament | None = None

    @property
    def color_hex(self) -> str:
        if self.filament is not None:
            return self.filament.color_hex
        return ""

    @property
    def display_name(self) -> str:
        if self.filament is not None:
            return self.filament.display_name
        # A spool the library no longer has — a project file opened on another
        # machine, a spool the user cleared out — still has to be named: the
        # short id is the honest answer.
        return self.filament_id[:8]


def spool_usage(
    result: MatchResult | None, library: FilamentLibrary | None
) -> list[SpoolUse]:
    """How much of each spool one match spends, largest first.

    Every kept colour is either a spool on its own (100% of that spool) or a
    two/three-filament recipe (its percentages), and the per-colour shares add
    up to 1.  Splitting each share by the recipe's percentages therefore gives
    the spool totals the printer will actually load — which is what 「用到了哪些
    耗材」 has to mean if it is to decide which rolls go in the AMS.
    """
    if result is None or not result.palette:
        return []
    printed = max(1, int(result.printed_pixels))
    totals: dict[str, float] = {}
    for index, entry in enumerate(result.palette):
        share = int(result.counts[index]) / printed
        parts = tuple(entry.parts)
        if not parts:
            recipe = entry.recipe
            if recipe is not None:
                parts = ((recipe.a_id, recipe.percent_a), (recipe.b_id, recipe.percent_b))
            elif entry.filament_ids:
                parts = ((entry.filament_ids[0], 100),)
        for filament_id, percent in parts:
            parent = str(filament_id)
            totals[parent] = totals.get(parent, 0.0) + share * percent / 100.0
    ordered = sorted(totals.items(), key=lambda item: (-item[1], item[0]))
    return [
        SpoolUse(
            filament_id=filament_id,
            share=share,
            filament=library.get(filament_id) if library is not None else None,
        )
        for filament_id, share in ordered
    ]


#: How many palette rows the nearest-colour search looks at per block.  The
#: three-filament pool is millions of colours, so the distance to every one of
#: them is computed in blocks and only the best few per target are carried
#: forward — the full ``(targets × colours)`` matrix would be gigabytes.
_SEARCH_BLOCK = 1 << 16


def _palette_table(palette: Sequence[PaletteEntry]) -> np.ndarray:
    """The ``(N, 3)`` Lab table the nearest-colour search runs against.

    A palette that carries its own table is asked for it — the three-filament
    pool does, because building a :class:`PaletteEntry` for each of its four
    million colours is exactly what cannot be done.  Everything stays float64:
    the shortlist pass is only a proxy, but the CIEDE2000 pass that follows is
    the answer the UI reports, and rounding it through float32 would move
    colours by more than the differences it is supposed to resolve.
    """
    getter = getattr(palette, "lab_table", None)
    if callable(getter):
        return np.asarray(getter(), dtype=np.float64)
    return np.asarray([entry.lab for entry in palette], dtype=np.float64)


def _shortlist_indices(
    table: np.ndarray,
    targets: np.ndarray,
    shortlist: int,
    on_block: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """The ``shortlist`` rows of ``table`` closest to each row of ``targets``.

    Distances are Euclidean in Lab and computed in blocks: the three-filament
    pool is millions of rows, so the full ``(targets × rows)`` matrix would be
    gigabytes.  Only the running best ``shortlist`` rows per target are carried
    between blocks, so the working set is ``(targets × shortlist)`` however long
    the pool is — which is what makes a four-thousand-row shortlist affordable
    at all (m03703 第 1 条).

    ``on_block`` is called with ``(rows_done, rows_total)`` after every block.
    It is what keeps the bar moving across a four-million-row pool: the scan is
    one vectorised call per block, so without this nothing is reported between
    the start of the search and its answer (m00940 第 2 条).
    """
    count = int(table.shape[0])
    take = min(shortlist, count)
    if take <= 0:
        return np.zeros((len(targets), 0), dtype=np.int64)
    target_sq = (targets**2).sum(axis=1)[:, None]
    run_indices = np.zeros((len(targets), 0), dtype=np.int64)
    run_scores = np.zeros((len(targets), 0), dtype=np.float32)
    for start in range(0, count, _SEARCH_BLOCK):
        block = table[start : start + _SEARCH_BLOCK]
        square = target_sq + (block**2).sum(axis=1)[None, :] - 2.0 * (targets @ block.T)
        np.maximum(square, 0.0, out=square)
        here = min(take, block.shape[0])
        picked = np.argpartition(square, here - 1, axis=1)[:, :here]
        run_indices = np.concatenate([run_indices, picked + start], axis=1)
        run_scores = np.concatenate(
            [run_scores, np.take_along_axis(square, picked, axis=1)], axis=1
        )
        if run_scores.shape[1] > 2 * take:
            keep = np.argpartition(run_scores, take - 1, axis=1)[:, :take]
            run_indices = np.take_along_axis(run_indices, keep, axis=1)
            run_scores = np.take_along_axis(run_scores, keep, axis=1)
        if on_block is not None:
            on_block(min(start + _SEARCH_BLOCK, count), count)
    width = run_scores.shape[1]
    keep = np.argpartition(run_scores, min(take, width) - 1, axis=1)[:, :take]
    return np.take_along_axis(run_indices, keep, axis=1)


def _scan_exact(
    labs: np.ndarray,
    table: np.ndarray,
    on_block: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Index of the closest row of ``table`` to each row of ``labs``, exactly.

    Every row is measured with CIEDE2000 and the smallest one wins, so this is
    the global optimum by construction — no Lab proxy anywhere in the answer.
    It walks the colours one at a time so the working set stays a couple of
    megabytes whatever the pool size, and ``on_block`` is called with
    ``(pairs_done, pairs_total)`` as it goes, pairs being colour × row.
    """
    count = int(table.shape[0])
    rows = len(labs)
    out = np.zeros(rows, dtype=np.int64)
    if count == 0 or rows == 0:
        return out
    width = min(_EXACT_BLOCK, count)
    pairs_total = count * rows
    pairs_done = 0
    for row in range(rows):
        lab = np.asarray(labs[row], dtype=np.float64)[None, :]
        best = float("inf")
        chosen = 0
        for start in range(0, count, width):
            chunk = table[start : start + width]
            values = _color.delta_e_2000(lab, chunk)
            local = int(np.argmin(values))
            value = float(values[local])
            if value < best:
                best = value
                chosen = start + local
            pairs_done += len(chunk)
            if on_block is not None:
                on_block(pairs_done, pairs_total)
        out[row] = chosen
    return out


def _nearest_entries(
    labs: np.ndarray,
    entries: Sequence[PaletteEntry],
    table: np.ndarray | None = None,
    on_row: Callable[[int, int], None] | None = None,
    on_scan: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Index of the palette entry closest to each row of ``labs``.

    Two paths, and the pool size decides which one runs:

    * ``rows × colours`` within :data:`_EXACT_PAIRS` — every entry is measured
      with CIEDE2000, so the answer is the true nearest colour.  The
      two-filament pool (tens of thousands of colours) always lands here.
    * the four-million-row three-filament pool — a Euclidean pass in Lab keeps
      the :data:`_SHORTLIST` nearest rows per colour and CIEDE2000 picks among
      those.  That is a proxy, not a proof: measured against a fully exact scan
      of the same pool, nineteen picture colours came out at most 0.010 ΔE00
      away from the true optimum (41-spool library, m03703 第 1 条).

    ``on_scan`` gets ``(done, total)`` as the rows are walked — the Euclidean
    pass in the second path, the whole colour × row sweep in the first — and
    ``on_row`` the same shape as the colours are settled.  Two callbacks rather
    than one because they are two very different stretches of time.
    """
    if not entries:
        return np.zeros(len(labs), dtype=np.int64)
    if table is None:
        table = _palette_table(entries)
    count = int(table.shape[0])
    rows = len(labs)
    if count == 0 or rows == 0:
        return np.zeros(rows, dtype=np.int64)
    if count * rows <= _EXACT_PAIRS:
        out = _scan_exact(labs, table, on_scan)
        if on_row is not None:
            on_row(rows, rows)
        return out

    targets = np.asarray(labs, dtype=np.float32)
    best_indices = _shortlist_indices(table, targets, _SHORTLIST, on_scan)

    # A three-filament pool is a superset of the two-filament one, but a wider
    # table can push a two-filament colour out of the Euclidean shortlist — and
    # then ticking 「也算三色配方」 would quietly make a colour *worse*, which is
    # the one thing it must never do.  So the head gets its own shortlist and the
    # two are merged before the exact pass.
    head_size = int(getattr(entries, "head_size", 0) or 0)
    if 0 < head_size < count:
        head_indices = _shortlist_indices(table[:head_size], targets, _SHORTLIST)
        best_indices = np.concatenate([best_indices, head_indices], axis=1)

    out = np.zeros(rows, dtype=np.int64)
    for row in range(rows):
        candidates = np.asarray(best_indices[row], dtype=np.int64)
        values = _color.delta_e_2000(
            np.asarray(labs[row], dtype=np.float64)[None, :], table[candidates]
        )
        out[row] = int(candidates[int(np.argmin(values))])
        if on_row is not None and (row % _PROGRESS_ROWS == _PROGRESS_ROWS - 1 or row == rows - 1):
            on_row(row + 1, rows)
    return out


def _pack(rgb: np.ndarray) -> np.ndarray:
    channels = rgb.astype(np.int64)
    return (channels[..., 0] << 16) | (channels[..., 1] << 8) | channels[..., 2]


def _unpack(key: int) -> tuple[int, int, int]:
    return ((key >> 16) & 0xFF, (key >> 8) & 0xFF, key & 0xFF)


#: Lloyd iterations for the weighted clustering in ``reduce_colours``.  Twelve
#: is where the labels stop moving on every picture measured, and the loop also
#: stops as soon as they stop changing.
_KMEANS_ITERATIONS = 12

#: The seed pass runs median cut over the colour *histogram* rather than over the
#: picture, so a photograph is thinned to this many rows first.  Seeds only have
#: to land near the answer — the clustering is what places the colours.
_SEED_ROWS = 400_000


def _box_sum(values: np.ndarray, radius: int) -> np.ndarray:
    """Sum over the ``(2 * radius + 1)²`` window around every pixel.

    An integral image makes this independent of the radius; the window is
    clipped at the borders rather than zero-padded so edges do not darken.
    """
    height, width = values.shape[:2]
    integral = np.zeros((height + 1, width + 1) + values.shape[2:], dtype=np.float64)
    integral[1:, 1:] = values
    np.cumsum(integral, axis=0, out=integral)
    np.cumsum(integral, axis=1, out=integral)

    rows = np.arange(height)
    columns = np.arange(width)
    top = np.maximum(rows - radius, 0)
    bottom = np.minimum(rows + radius + 1, height)
    left = np.maximum(columns - radius, 0)
    right = np.minimum(columns + radius + 1, width)
    top_grid, left_grid = np.meshgrid(top, left, indexing="ij")
    bottom_grid, right_grid = np.meshgrid(bottom, right, indexing="ij")
    return (
        integral[bottom_grid, right_grid]
        - integral[top_grid, right_grid]
        - integral[bottom_grid, left_grid]
        + integral[top_grid, left_grid]
    )


def _smooth_colours(rgb: np.ndarray, keep: np.ndarray | None, radius: int) -> np.ndarray:
    """Mean filter that only ever averages pixels the caller cares about."""
    if radius <= 0:
        return rgb.astype(np.float64)
    weight = np.ones(rgb.shape[:2], dtype=np.float64)
    if keep is not None:
        weight = keep.astype(np.float64)
    total = _box_sum(rgb.astype(np.float64) * weight[:, :, None], radius)
    divisor = _box_sum(weight, radius)
    return total / np.maximum(divisor, 1.0)[:, :, None]


def _class_means(rgb: np.ndarray, classes: np.ndarray, count: int) -> np.ndarray:
    """Mean of the ORIGINAL pixels behind every class.

    The classes come from a smoothed image, so their own values are blurred;
    reporting the untouched pixels instead keeps the swatch and the ΔE00 the
    user judges against honest.
    """
    flat = classes.reshape(-1)
    pixels = rgb.reshape(-1, 3).astype(np.float64)
    valid = flat >= 0
    totals = np.zeros((count, 3), dtype=np.float64)
    np.add.at(totals, flat[valid], pixels[valid])
    counts = np.bincount(flat[valid], minlength=count).astype(np.float64)
    return np.rint(totals / np.maximum(counts, 1.0)[:, None]).astype(np.int64)


def _median_cut_seeds(keys: np.ndarray, counts: np.ndarray, count: int) -> np.ndarray:
    """Median cut over the colour histogram — the starting point of the clustering.

    Median cut splits whichever RGB box holds the most pixels, which is not the
    same question as where the eye sees a difference; it is, however, a cheap and
    stable way to get ``count`` colours that cover the picture, and the weighted
    clustering below then moves them onto the picture's own colours.
    """
    rows = np.array([_unpack(int(key)) for key in keys], dtype=np.uint8)
    repeat = np.maximum(np.asarray(counts, dtype=np.int64), 1)
    total = int(repeat.sum())
    if total > _SEED_ROWS:
        repeat = np.maximum(repeat // int(np.ceil(total / _SEED_ROWS)), 1)
    strip = np.repeat(rows, repeat, axis=0).reshape(1, -1, 3)
    quantized = Image.fromarray(strip, "RGB").quantize(
        colors=int(count), method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
    )
    palette = quantized.getpalette() or []
    seeds = np.asarray(palette[: count * 3], dtype=np.float64).reshape(-1, 3)
    return np.clip(np.rint(seeds), 0, 255).astype(np.uint8)


def _kmeans_labels(
    points: np.ndarray,
    counts: np.ndarray,
    seeds: np.ndarray,
    iterations: int = _KMEANS_ITERATIONS,
) -> np.ndarray:
    """Weighted Lloyd clustering of the unique colours, in a perceptual space.

    ``points`` is ``(U, 3)`` in the space the eye cares about (CIELAB here) and
    ``counts`` says how many pixels each unique colour stands for, so a colour
    covering half the picture pulls its centre far harder than a stray pixel.
    Returns one label per unique colour, drawn from ``0 … len(seeds) - 1``.

    An empty cluster is handed the colour it is farthest from, so the colour
    budget is never silently spent on fewer colours than the user asked for.
    """
    unique = len(points)
    if unique == 0 or len(seeds) == 0:
        return np.zeros(unique, dtype=np.int64)
    centres = np.asarray(seeds, dtype=np.float64).copy()
    weight = np.asarray(counts, dtype=np.float64)[:, None]
    labels = np.zeros(unique, dtype=np.int64)
    distance = ((points[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
    for step in range(max(1, iterations)):
        fresh = distance.argmin(axis=1)
        if step and np.array_equal(fresh, labels):
            break
        labels = fresh
        for index in range(len(centres)):
            mine = labels == index
            if not mine.any():
                farthest = int(distance.min(axis=1).argmax())
                centres[index] = points[farthest]
                labels[farthest] = index
                continue
            total = (points[mine] * weight[mine]).sum(axis=0)
            centres[index] = total / max(weight[mine].sum(), 1.0)
        distance = ((points[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
    return distance.argmin(axis=1)


def _inward_fill(rgb: np.ndarray, keep: np.ndarray, rounds: int = 8) -> np.ndarray:
    """Grow the artwork outwards into the transparent pixels by diffusion.

    ``convert("RGB")`` leaves BLACK where the picture is transparent.  Handing
    that black to a dithering quantiser is worse than handing it nothing: error
    diffusion carries a pixel's mistake into its right and lower neighbours, so
    the black would be pushed into the artwork's edge.  A few rounds of 「the
    mean of my opaque neighbours」 give those neighbours a plausible colour
    first, and whatever the rounds never reach (the far side of a big
    background) is filled with the artwork's own mean — a flat area diffuses
    nothing into the picture, and the mask throws it away afterwards anyway.
    """
    values = rgb.astype(np.float64)
    weight = keep.astype(np.float64)
    for _ in range(max(0, rounds)):
        total = _box_sum(values * weight[:, :, None], 1)
        divisor = _box_sum(weight, 1)
        fresh = (divisor > 0) & (weight == 0)
        if not fresh.any():
            break
        values[fresh] = (total / np.maximum(divisor, 1.0)[:, :, None])[fresh]
        weight[fresh] = 1.0
    remaining = weight == 0
    if remaining.any():
        mean = values[keep].mean(axis=0) if keep.any() else np.zeros(3)
        values[remaining] = mean
    return values


def _diffuse(rgb: np.ndarray, palette: np.ndarray) -> np.ndarray | None:
    """Floyd–Steinberg the picture onto a fixed palette.

    Returns the ``(H, W)`` int64 map of palette indices, or ``None`` when there
    is nothing to diffuse (fewer than two colours, or a shape mismatch).
    """
    rows = np.asarray(palette, dtype=np.int64).reshape(-1, 3)[:256]
    if len(rows) < 2:
        return None
    # The table has to be 256 entries wide, and anything past the palette must
    # REPEAT real colours: padded black would be a colour the artwork never had,
    # and the quantiser would happily hand dark pixels to it (measured: five
    # classes out of a four-colour palette).  The caller folds the indices back.
    tiled = np.tile(rows, (int(np.ceil(256 / len(rows))), 1))[:256]
    table = Image.new("P", (1, 1))
    table.putpalette([int(value) for value in tiled.reshape(-1)])
    source = Image.fromarray(np.clip(np.rint(rgb), 0, 255).astype(np.uint8), "RGB")
    mapped = np.asarray(
        source.quantize(palette=table, dither=Image.Dither.FLOYDSTEINBERG), dtype=np.int64
    )
    if mapped.shape != rgb.shape[:2]:
        return None
    return mapped % len(rows)


def _dither_masked(
    rgb: np.ndarray, keep: np.ndarray, palette: np.ndarray, pad: int = 2
) -> np.ndarray | None:
    """Diffuse the picture onto a fixed palette, inside the mask only.

    Error diffusion needs left and right neighbours, and the transparent
    surround has neither a colour nor a right to the colour budget: the picture
    is cropped to the artwork (plus a ``pad``-pixel skirt), filled inwards
    first, and diffused onto the palette the masked reduction already earned.
    Whatever diffusion does outside the mask is thrown away again below.

    Returns the ``(H, W)`` class map (``-1`` outside the mask), or ``None``
    when there is nothing to diffuse.
    """
    rows = np.flatnonzero(keep.any(axis=1))
    columns = np.flatnonzero(keep.any(axis=0))
    if not rows.size or not columns.size or len(palette) < 2:
        return None
    top = max(0, int(rows[0]) - pad)
    bottom = min(rgb.shape[0], int(rows[-1]) + 1 + pad)
    left = max(0, int(columns[0]) - pad)
    right = min(rgb.shape[1], int(columns[-1]) + 1 + pad)
    inside = keep[top:bottom, left:right]
    filled = _inward_fill(rgb[top:bottom, left:right], inside)
    mapped = _diffuse(filled, palette)
    if mapped is None:
        return None
    classes = np.full(rgb.shape[:2], -1, dtype=np.int64)
    classes[top:bottom, left:right] = np.where(inside, mapped, -1)
    return classes


def reduce_colours(
    image: Image.Image, settings: MatchSettings, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a picture to at most ``max_colours`` classes.

    Returns ``(colours, classes)`` where ``colours`` is ``(N, 3)`` int64 and
    ``classes`` is the ``(H, W)`` int64 class map (``-1`` outside ``mask``).

    ``mask`` matters more than it looks.  ``image.convert("RGB")`` turns a
    transparent background into solid BLACK, and on an illustration that
    background is usually most of the picture — on the bundled pig it is
    138,095 of 262,144 pixels.  A quantiser handed that image spends its whole
    colour budget on black and on the anti-aliased fringe where the artwork
    meets it, and real colours disappear into the black cluster.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)

    keep: np.ndarray | None = None
    if mask is not None:
        keep = np.asarray(mask, dtype=bool)
        if keep.shape != rgb.shape[:2] or not keep.any() or keep.all():
            keep = None

    smoothed = _smooth_colours(rgb, keep, settings.smooth_radius)
    # A picture that already fits the budget keeps its own colours EXACTLY, so
    # this test runs on the untouched pixels: smoothing a four-colour logo only
    # invents shades along its edges.
    keys = _pack(rgb[keep]) if keep is not None else _pack(rgb)

    unique_keys, inverse = np.unique(keys, return_inverse=True)
    if unique_keys.size <= settings.max_colours:
        colours = np.array([_unpack(int(key)) for key in unique_keys], dtype=np.int64)
        if keep is None:
            return colours, inverse.reshape(rgb.shape[:2]).astype(np.int64)
        classes = np.full(rgb.shape[:2], -1, dtype=np.int64)
        classes[keep] = inverse
        return colours, classes

    # The clustering below sees the SMOOTHED picture (that is what 平滑 is for)
    # while every class colour is the mean of the ORIGINAL pixels, see
    # ``_class_means``.
    #
    # Weighted k-means in CIELAB, seeded with median cut's own palette.
    #
    # Median cut splits whichever RGB box holds the most pixels, which is not
    # where the eye sees a difference: on a dusk poster it spends slot after
    # slot on a dozen near-identical blues and leaves the skin one average, and
    # on a line drawing it buys the anti-aliased fringe instead of the lines —
    # measured, 18.7 % of the pixels more than ΔE00 5 from the colour they were
    # given.  Clustering weighted by pixel count in a perceptual space fixes
    # both, and it also uses the whole budget instead of asking for
    # ``max_colours × 6`` and folding most of them away again.  Measured against
    # per-pixel ΔE00 (same pictures, same budgets): gradient 2.222 → 1.252 and
    # 8.1 % → 3.4 % at 16 colours, line drawing 0.634/37.5 p95/18.7 % →
    # 0.000/0.0 %/0.0 %, and faster — 0.15 s against 1.37 s on the gradient at
    # 32 colours.
    strip = np.clip(np.rint(smoothed), 0, 255).astype(np.uint8)
    cluster_keys, cluster_inverse = np.unique(
        _pack(strip[keep]) if keep is not None else _pack(strip), return_inverse=True
    )
    rows = np.array([_unpack(int(key)) for key in cluster_keys], dtype=np.uint8)
    counts = np.bincount(np.asarray(cluster_inverse).reshape(-1), minlength=len(cluster_keys))
    budget = int(min(256, settings.max_colours))
    seeds = _median_cut_seeds(cluster_keys, counts, budget)
    labels = _kmeans_labels(_color.lab_from_rgb(rows), counts, _color.lab_from_rgb(seeds))
    if keep is None:
        classes = np.asarray(labels)[cluster_inverse].reshape(rgb.shape[:2]).astype(np.int64)
    else:
        classes = np.full(rgb.shape[:2], -1, dtype=np.int64)
        classes[keep] = np.asarray(labels)[cluster_inverse]
    used = np.unique(classes[classes >= 0])
    lookup = np.full(int(classes.max()) + 1, -1, dtype=np.int64)
    lookup[used] = np.arange(len(used))
    classes = np.where(classes >= 0, lookup[np.clip(classes, 0, None)], -1)
    colours = _class_means(rgb, classes, len(used))
    # The colours come out of the clustering already budgeted, so the picture
    # gets the number of colours the user asked for.  It used to leave here one
    # step earlier: the quantiser was asked for ``max_colours × 6`` clusters and
    # ``_fold_clusters`` folded them back down, which silently handed back fewer
    # colours than requested (measured: 25 instead of 48 on a gradient, at a
    # third more error).  Near-identical source colours are still folded — in
    # ``_merge_similar_colours``, on the reduced palette, which is where the
    # speckle it guards against appears.
    if not settings.dither or len(colours) < 2:
        # Nothing to diffuse, or a single colour left to diffuse towards.
        return colours, classes
    # The compacted strip has no meaningful neighbour relationship, so error
    # diffusion always runs on the two-dimensional picture: over the whole crop
    # the artwork was reduced from, or — with a mask — inside the artwork.
    if keep is None:
        classes = _diffuse(rgb, colours)
    else:
        classes = _dither_masked(rgb, keep, colours)
    if classes is None:
        return colours, classes
    used = np.unique(classes[classes >= 0])
    if used.size < 2:
        return colours, classes
    lookup = np.full(int(classes.max()) + 1, -1, dtype=np.int64)
    lookup[used] = np.arange(len(used))
    classes = np.where(classes >= 0, lookup[np.clip(classes, 0, None)], -1)
    return _class_means(rgb, classes, len(used)), classes


def _class_counts(classes: np.ndarray, count: int) -> np.ndarray:
    """Pixels per class, ignoring the ``-1`` that marks the transparent surround."""
    flat = classes.reshape(-1)
    flat = flat[flat >= 0]
    return np.bincount(flat, minlength=count).astype(np.int64)


def _fold_clusters(
    colours: np.ndarray, classes: np.ndarray, limit: int, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Fold colours the eye cannot tell apart, keeping at most ``limit`` of them.

    This is the merge behind :func:`_merge_similar_colours` — and, before 1.6.25,
    the pass that folded an oversampled quantiser back down to the budget.

    A plain union-find over "everything within ``threshold``" is not usable
    here.  On a smooth gradient every colour is within ``threshold`` of its
    neighbour, so the union chains across the whole ramp and a 金→红 gradient
    collapses to a single colour — measured, and it is why this is a leader
    pass instead: a colour is only ever compared against the leaders, never
    against its neighbour, so a ramp keeps several representatives while a flat
    fill's near-identical shades still fold onto the first one.

    Colours are visited largest first, so the leaders are the colours the
    picture actually uses; once ``limit`` leaders exist every later colour joins
    its nearest one, which is what keeps a small-but-distinct region (the eyes)
    ahead of yet another shade of the background.
    """
    if len(colours) <= 1:
        return colours, classes

    lab = _color.lab_from_rgb(colours)
    counts = _class_counts(classes, len(colours))
    order = [int(index) for index in np.argsort(-counts, kind="stable")]

    leaders: list[int] = []
    for index in order:
        if len(leaders) >= limit:
            break
        if all(_color.delta_e_2000(lab[index], lab[leader]) >= threshold for leader in leaders):
            leaders.append(index)
    if not leaders:  # threshold <= 0 and limit <= 0 would land here
        leaders = [order[0]]

    survivors = sorted(leaders)
    lookup = np.empty(len(colours), dtype=np.int64)
    for index in range(len(colours)):
        lookup[index] = min(
            range(len(survivors)),
            key=lambda position: _color.delta_e_2000(lab[index], lab[survivors[position]]),
        )
    return (
        colours[survivors],
        np.where(classes >= 0, lookup[np.clip(classes, 0, None)], -1),
    )


def load_image(source: str | Path | Image.Image) -> Image.Image:
    if isinstance(source, Image.Image):
        return source.copy()
    with Image.open(source) as handle:
        handle.load()
        return handle.copy()


def _merge_similar_colours(
    colours: np.ndarray,
    classes: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fold reduced source colours that the eye cannot tell apart.

    The colour reduction happily spends two of its budget on two source shades a
    fraction of a ΔE00 apart. Each then maps to a *different* palette entry, and
    because the picture's own noise decides which, a flat region comes out
    speckled — the pink/cream stipple that made the plate look wrong. Merge them
    first, keeping the shade that covers the most pixels.

    ``_fold_clusters`` does the merging, and that matters: its leader pass folds
    a flat fill's near-identical shades onto the shade that covers it without
    chaining a smooth gradient into one colour (see its docstring for the
    measurement).
    """
    if threshold <= 0.0 or len(colours) < 2:
        return colours, classes
    return _fold_clusters(colours, classes, len(colours), threshold)


def _merge_indistinguishable(
    kept: list[PaletteEntry],
    counts: np.ndarray,
    indices: np.ndarray,
    threshold: float,
) -> tuple[list[PaletteEntry], np.ndarray, np.ndarray]:
    """Fold matched colours the eye cannot tell apart into the most-used one.

    Two palette entries a hair apart in Lab both win part of the same flat
    region, so the plate comes out speckled, the highlight turns to stripes,
    and the printer spends a filament change on an invisible difference.
    Union-find the pairs closer than ``threshold`` ΔE00, keep the most-used
    member of each group, then renumber largest-first as the caller expects.
    """
    if threshold <= 0.0 or len(kept) < 2:
        return kept, counts, indices

    labs = np.array([entry.lab for entry in kept], dtype=np.float64)
    parent = list(range(len(kept)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    merged = False
    for first in range(len(kept)):
        for second in range(first + 1, len(kept)):
            if _color.delta_e_2000(labs[first], labs[second]) >= threshold:
                continue
            root_a, root_b = find(first), find(second)
            if root_a != root_b:
                parent[root_b] = root_a
                merged = True
    if not merged:
        return kept, counts, indices

    groups: dict[int, list[int]] = {}
    for index in range(len(kept)):
        groups.setdefault(find(index), []).append(index)

    survivors = sorted(
        max(members, key=lambda member: (int(counts[member]), -member))
        for members in groups.values()
    )
    remap = np.full(len(kept), -1, dtype=_INDEX_DTYPE)
    for new_index, survivor in enumerate(survivors):
        for member in groups[find(survivor)]:
            remap[member] = new_index
    folded = np.where(indices >= 0, remap[np.clip(indices, 0, None)], -1).astype(_INDEX_DTYPE)

    folded_counts = np.array(
        [int(np.count_nonzero(folded == index)) for index in range(len(survivors))],
        dtype=np.int64,
    )
    order = np.argsort(-folded_counts, kind="stable")
    final = np.where(folded >= 0, np.argsort(order)[np.clip(folded, 0, None)], -1).astype(_INDEX_DTYPE)
    return (
        [kept[survivors[int(position)]] for position in order],
        folded_counts[order],
        final,
    )


def _packed_key(rgb: Sequence[int]) -> int:
    values = [int(value) & 0xFF for value in rgb]
    return (values[0] << 16) | (values[1] << 8) | values[2]


class TriplePalette(Sequence[PaletteEntry]):
    """``entries`` (spools and pairs) plus every three-filament colour.

    A 41-spool library has four million three-filament colours, so the tail is
    never materialised: the matcher asks for the Lab table — built once, in
    blocks, as float64 — and a :class:`PaletteEntry` is built only for the
    handful of colours the picture actually keeps.  Colours the head already
    offers are skipped, because one spool or one two-filament mix is the cheaper
    way to reach the same hex and the tie should go to the cheaper answer.
    """

    def __init__(self, entries, triples, library=None) -> None:
        self._head: list[PaletteEntry] = list(entries)
        self._triples = triples
        self._library = library
        self._tail: np.ndarray | None = None
        self._table: np.ndarray | None = None

    @property
    def head(self) -> list[PaletteEntry]:
        """The spools and two-filament mixes, as they were handed in."""
        return self._head

    @property
    def head_size(self) -> int:
        """How many leading rows are spools and two-filament mixes.

        The nearest-colour search gives these rows their own shortlist, so a
        much wider tail can never evict a cheaper two-filament answer.
        """
        return len(self._head)

    # -- lazy bookkeeping --------------------------------------------------
    def _tail_indices(self) -> np.ndarray:
        if self._tail is None:
            packed = np.asarray(self._triples.packed, dtype=np.int64)
            taken = np.fromiter(
                (_packed_key(entry.rgb) for entry in self._head),
                dtype=np.int64,
                count=len(self._head),
            )
            # ``packed`` is deduplicated and ascending, so the membership test
            # can assume uniqueness on both sides.
            keep = ~np.isin(packed, taken, assume_unique=True)
            self._tail = np.flatnonzero(keep)
        return self._tail

    def _name(self, filament_id: str) -> str:
        getter = getattr(self._library, "get", None)
        filament = getter(filament_id) if callable(getter) else None
        return getattr(filament, "display_name", "") or str(filament_id)

    def _entry_at(self, index: int) -> PaletteEntry:
        """Build one entry for a three-filament colour (the only ones built)."""
        recipes = self._triples.recipes_at(index)
        recipe = recipes[0] if recipes else None
        color_hex = self._triples.color_hex_at(index)
        parts: tuple[tuple[str, int], ...] = ()
        label = color_hex
        if recipe is not None:
            parts = (
                (str(recipe.a_id), int(recipe.percent_a)),
                (str(recipe.b_id), int(recipe.percent_b)),
                (str(recipe.c_id), int(recipe.percent_c)),
            )
            label = " + ".join(
                f"{self._name(parent)} {percent}%" for parent, percent in parts
            )
        rgb = tuple(int(value) for value in self._triples.rgb_at(index))
        return PaletteEntry(
            key=f"colour:{color_hex}",
            color_hex=color_hex,
            rgb=rgb,
            lab=np.asarray(_color.lab_from_rgb(rgb), dtype=np.float64),
            label=label,
            kind="mix",
            filament_ids=tuple(parent for parent, _percent in parts),
            recipe=None,
            recipes=(),
            parts=parts,
        )

    # -- Sequence ----------------------------------------------------------
    def __len__(self) -> int:
        return len(self._head) + int(self._tail_indices().size)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        position = int(index)
        if position < 0:
            position += len(self)
        if not 0 <= position < len(self):
            raise IndexError(index)
        if position < len(self._head):
            return self._head[position]
        return self._entry_at(int(self._tail_indices()[position - len(self._head)]))

    def lab_table(self) -> np.ndarray:
        """``(len(self), 3)`` Lab, cached for the block search.

        The tail reuses the catalogue's own cached table — it is the same Lab
        values in the same order as ``packed``, and converting four million
        colours twice would cost seconds for nothing.  Only the head, which the
        catalogue never saw, is stacked here.
        """
        if self._table is None:
            head = np.asarray([entry.lab for entry in self._head], dtype=np.float64)
            head = head.reshape(len(self._head), 3)
            tail = np.asarray(self._triples.lab_table(), dtype=np.float64)[
                self._tail_indices()
            ]
            self._table = np.concatenate([head, tail], axis=0)
        return self._table


def _spool_ids(entry: PaletteEntry) -> tuple[str, ...]:
    return tuple(str(value) for value in entry.filament_ids)


def _renumber_sources(
    kept: list[PaletteEntry], folded: np.ndarray
) -> tuple[list[PaletteEntry], np.ndarray, np.ndarray, list[int]]:
    """Recount ``folded`` and renumber largest-first, dropping empty entries.

    ``folded`` holds compact indices into ``kept``.  Entries no pixel points at
    any more are dropped outright — keeping one would put an unused filament in
    the plate's slot list, which is the one thing the budget promises not to do.

    The fourth value is which position of ``kept`` each surviving entry came
    from, in the same order.  「按当前色块重新匹配」 needs that link: two blocks can
    be folded into one region, and the region then keeps the picture colour of
    the block that survived rather than the average of everything in it.
    """
    live = np.unique(folded[folded >= 0])
    counts = np.array(
        [int(np.count_nonzero(folded == value)) for value in live], dtype=np.int64
    )
    order = np.argsort(-counts, kind="stable")
    # ``order[rank] = position_in_live``, so the inverse maps a live index to
    # the rank it ends up at.
    rank_of_live = np.empty(live.size, dtype=_INDEX_DTYPE)
    rank_of_live[order] = np.arange(live.size, dtype=_INDEX_DTYPE)
    final = np.full(folded.shape, -1, dtype=_INDEX_DTYPE)
    valid = folded >= 0
    final[valid] = rank_of_live[np.searchsorted(live, folded[valid])]
    sources = [int(live[int(position)]) for position in order]
    return [kept[source] for source in sources], counts[order], final, sources


def _renumber(
    kept: list[PaletteEntry], folded: np.ndarray
) -> tuple[list[PaletteEntry], np.ndarray, np.ndarray]:
    """Recount ``folded`` and renumber largest-first, dropping empty entries."""
    entries, counts, final, _sources = _renumber_sources(kept, folded)
    return entries, counts, final


def _affordable_nearest(
    entry: PaletteEntry, budget: int, palette: Sequence[PaletteEntry] | None
) -> PaletteEntry | None:
    """The closest colour the budget can pay for, or ``None``.

    Only the palette's own head is searched — spools and two-filament mixes.
    A three-filament colour needs three spools, so with a budget below three it
    can never be the answer, and walking the lazy tail to prove that would build
    millions of entries to reject all of them.
    """
    if palette is None:
        return None
    candidates = getattr(palette, "head", None)
    if candidates is None:
        candidates = palette
    best = None
    best_value = float("inf")
    for candidate in candidates:
        if len(_spool_ids(candidate)) > budget:
            continue
        value = _color.delta_e_2000(entry.lab, candidate.lab)
        if value < best_value:
            best_value = value
            best = candidate
    return best


def _nearest_spending(
    entry: PaletteEntry, spool: str, palette: Sequence[PaletteEntry] | None
) -> PaletteEntry | None:
    """The closest colour that is printed FROM ``spool``, or ``None``.

    「必须包含」 is a demand about the machine, not about the picture: the user has
    that roll loaded or wants it used up, so some region has to be printed with
    it.  This finds the colour that roll can print which is closest to what the
    region wanted.  Only the palette's head is searched — spools and
    two-filament mixes — for the same reason as :func:`_affordable_nearest`.
    """
    if palette is None:
        return None
    candidates = getattr(palette, "head", None)
    if candidates is None:
        candidates = palette
    best = None
    best_value = float("inf")
    for candidate in candidates:
        if spool not in _spool_ids(candidate):
            continue
        value = _color.delta_e_2000(entry.lab, candidate.lab)
        if value < best_value:
            best_value = value
            best = candidate
    return best


def _nearest_free(
    entry: PaletteEntry, banned: frozenset[str], palette: Sequence[PaletteEntry] | None
) -> PaletteEntry | None:
    """The closest colour that spends none of ``banned``, or ``None``.

    「不要用」 has to reach the colour picking itself, not just the roll list: a
    colour whose recipe names a banned spool is a colour this print cannot make.
    """
    if palette is None:
        return None
    candidates = getattr(palette, "head", None)
    if candidates is None:
        candidates = palette
    best = None
    best_value = float("inf")
    for candidate in candidates:
        if banned & set(_spool_ids(candidate)):
            continue
        value = _color.delta_e_2000(entry.lab, candidate.lab)
        if value < best_value:
            best_value = value
            best = candidate
    return best


#: How many different spools the 「耗材上限」 search is willing to choose between.
#: The regions the eye compares are the big ones and they are what asks for the
#: first rolls; the tail only multiplies the search.
_BUDGET_ROLL_LIMIT = 22

#: Above this many subsets the exhaustive walk gives way to growing a set roll
#: by roll and then swapping, which is the same answer on everything measured.
_BUDGET_SUBSET_LIMIT = 20_000

#: How many of a block's nearest colours 「按当前色块重新匹配」 keeps on hand.  The
#: block is printed with the nearest one that the 「耗材上限」 can afford, so this is
#: the depth of the trade the search may make to keep a block on the picture; six
#: was enough to hold every block of the reporter's pig inside four rolls.
_REMATCH_SHORTLIST = 6


#: ΔE00 at which a patch stops being 「同一种颜色差一点」 and becomes a patch the
#: eye reads as the wrong colour.  Five is the 「看得出来不一样」 line on a flat
#: print, and it is what :func:`_objective_distinct` counts.
_BAD_BLOCK_DELTA_E = 5.0

#: A way of ranking candidate roll sets: given each region's error under a set
#: of rolls (``worst``) and the pixels behind each region (``weights``), the
#: tuples are compared element by element.  There is no single right one — on
#: the user's pig every one of them loses something — which is exactly why the
#: page now offers several and lets the user pick.
BudgetObjective = Callable[[np.ndarray, np.ndarray], tuple[float, ...]]


def _objective_worst(worst: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """最不像的那一块先看：the region that looks wrong first, then the average.

    A picture is judged region by region, and the eye finds the one patch that
    is plainly the wrong colour long before it averages anything.  On the pig
    that patch was the nose: 0.2% of the pixels, and the whole complaint.  The
    pixel-weighted average alone answers 「总体最接近」, which gladly spends the
    nose to buy the background and prints a pink nose white — measured, the
    average-optimal four rolls put the nose at ΔE00 11.75 where the rolls that
    print it exactly cost 12% of the average.  So the worst surviving region
    leads and the average breaks the tie, which is the promise 「每一块都要像」
    read literally.
    """
    return float(worst.max()), float(weights @ worst)


def _objective_closest(worst: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """总体最接近：the pixel-weighted average leads and the worst block breaks the tie.

    This is what 「这张图整体像不像」 means, and it is the answer for a picture
    whose interest is a large area rather than a small feature.
    """
    return float(weights @ worst), float(worst.max())


def _objective_distinct(worst: np.ndarray, weights: np.ndarray) -> tuple[float, ...]:
    """小色块优先：count the blocks that are plainly wrong, then the average.

    0.2% of a picture is nothing to an average and everything to a nose.  This
    counts the regions past :data:`_BAD_BLOCK_DELTA_E` first, so a roll that
    rescues a small feature beats one that shaves the background.
    """
    return (
        float(np.count_nonzero(worst > _BAD_BLOCK_DELTA_E)),
        float(weights @ worst),
        float(worst.max()),
    )


def _objective_priority(worst: np.ndarray, weights: np.ndarray) -> tuple[float, ...]:
    """大色差色块优先：rescue the block that looks worst, then count the others.

    The user's own rule, in their words: the blocks with a big colour difference
    have to be recognised first, then those blocks get their colour under the
    remaining budget, and the 耗材上限 is spent rather than saved (m06704 第 3 条).

    The average cannot speak for a small feature.  On the user's pig the brow
    stroke is 1.6% of the pixels, so the average-led plan bought the background
    with the rolls that would have printed it and came out with a **brown stroke
    printed pink** (ΔE00 33 measured) while its average stayed at 2.29.  Leading
    with ``worst.max()`` and following with the count of plainly-wrong blocks
    (:data:`_BAD_BLOCK_DELTA_E`) puts a colour on the stroke first; the average
    only breaks the tie, which is why this is the default from 1.6.27 on.
    """
    return (
        float(worst.max()),
        float(np.count_nonzero(worst > _BAD_BLOCK_DELTA_E)),
        float(weights @ worst),
    )


#: What :func:`match_image` uses when the caller does not name a 方案, and the
#: objective behind the first entry :func:`match_options` offers.  The two agree
#: on purpose: the page shows 「重新匹配」 landing on the first row.
#:
#: 1.6.27 之前这里是 ``_objective_closest``（平均领先）。用户在自己的 41 卷库上看到
#: 「棕色的笔画被印成粉红」——那块只占 1.6% 像素，平均口径乐意拿它去换大面积，所以
#: 默认改成大色差优先（m06704 第 3 条）。平均领先的一档仍然是第二行，想整张图最像
#: 的用户点一下就有。
_budget_objective: BudgetObjective = _objective_priority


@dataclass(frozen=True)
class _Fit:
    """One way of spending 「耗材上限」: what it prints and which rolls it buys."""

    entries: list[PaletteEntry]
    counts: np.ndarray
    indices: np.ndarray
    #: Which position of the colour list each surviving entry came from, in the
    #: same order.  「按当前色块重新匹配」 needs it: a region that swallowed another
    #: one keeps the picture colour of the block that survived (the rule 合并
    #: follows), not the average of everything folded into it.
    sources: tuple[int, ...]
    #: Every roll the printed colours need.
    spools: frozenset[str]


def _budget_fits(
    kept: list[PaletteEntry],
    indices: np.ndarray,
    budget: int,
    palette: Sequence[PaletteEntry] | None = None,
    counts: Sequence[int] | np.ndarray | None = None,
    objective: BudgetObjective = _objective_closest,
    *,
    region_rgb: np.ndarray | None = None,
    limit: int = 1,
    require: frozenset[str] = frozenset(),
    avoid: frozenset[str] = frozenset(),
    candidates: Sequence[Sequence[PaletteEntry]] | None = None,
) -> list[_Fit]:
    """Every way of spending ``budget`` worth offering, best first.

    This is a choice, not a walk.  Taking the colours largest region first and
    refusing every one that needs a spool the budget has already spent is the
    greedy reading of 「最多用 N 卷料」, and it is measurably wrong: on a real
    41-spool library it scored weighted ΔE00 1.813 where the best four rolls the
    picture asked for score 1.249 — 31% worse — and the roll that prints a
    549-pixel nose sat one combination away from the answer it printed.  So the
    rolls the picture actually asks for are scored as subsets.  ``kept`` is
    expected largest-first with ``counts`` the pixels behind each entry.

    ``limit`` asks for the best ``limit`` different roll sets rather than only
    the winner: 「耗材方案太少了，多推荐几组」 is answered by ranking the same
    subsets the search already scores, not by a second search.  ``require`` and
    ``avoid`` are the user's 筛选: a plan that cannot spend a demanded roll, or
    that spends a forbidden one, is not a plan and is left out — the cap and the
    filter are instructions about this machine, not preferences to trade against
    picture quality.

    ``region_rgb`` is the picture colour behind each colour of ``kept``, and it is
    what makes the ranking agree with the numbers the page prints: a plan is
    scored on **every colour of the picture** — the ΔE00 between that colour and
    the colour this plan prints for it, weighted by its pixels — which is exactly
    the quantity :func:`match_fidelity_report` shows.

    Scoring the refused colours against the colour that replaces them is not the
    same thing, and averaging them is worse still.  Folding a colour into a block
    and judging the block by its mean lets a thin dark line disappear: the mean of
    a pale block barely moves, so the plan that paints the eyebrows with the
    background measured better than the plan that keeps them (1.6.14 shipped
    that, and the reporter's pig came back washed out — 平均 ΔE00 1.39 with 差块 0
    while the black eyes and brown brows were gone).  A per-colour penalty cannot
    be bought that cheaply: the line keeps its own ΔE00 whatever else is in the
    block.

    Returns ``[]`` when nothing has to be dropped, so the caller keeps the
    untouched result — which is also why the untouched rolls are checked against
    the filter first: a picture that already fits still has to obey 「不要用」.

    ``candidates`` says, per block, which colours that block may be printed with,
    nearest first.  Without it every block has exactly one colour — the one the
    picture (or the previous match) chose for it — and the search may only decide
    *which* blocks survive the cap.  With it a block may answer with a cheaper
    colour instead of being folded away, which is what 「按当前色块重新匹配」 needs:
    the blocks are the user's own decision there, and the reporter's pig came back
    from a merge with six blocks and, after a re-match, with two — because the
    nearest answer for the pink block wanted two rolls of its own and the three
    brown blocks wanted two more, so the cap folded four of them into the skin.
    Allowing the pink block to be printed with its second-nearest colour frees
    those rolls and keeps every block the user could still see.
    """
    if budget <= 0 or len(kept) < 2:
        return []

    pool = list(kept)
    #: Which position of ``kept`` each entry of ``pool`` came from.  The fallback
    #: colour appended below answers for position 0: it is chosen to match the
    #: largest region, which is what it stands in for.
    origin = list(range(len(pool)))
    #: The colours each position may be printed with, nearest first.  A caller that
    #: does not pass ``candidates`` gets exactly one per position — the colour the
    #: picture picked — and then this is the subset search it always was.
    if candidates is not None and len(candidates) == len(pool):
        options: list[list[PaletteEntry]] = [
            [entry for entry in entries] or [pool[position]]
            for position, entries in enumerate(candidates)
        ]
    else:
        options = [[entry] for entry in pool]
    option_spools = [
        [frozenset(_spool_ids(entry)) for entry in entries] for entries in options
    ]
    option_labs = [
        [np.asarray(entry.lab, dtype=np.float64) for entry in entries] for entries in options
    ]
    #: Pad every position to the same number of candidates by repeating the last
    #: one, so the ΔE tables below can be indexed with one candidate number per
    #: position; the repeats cost nothing because the search stops at the first
    #: minimum, and ``option_spools`` repeats with them, so they never widen the
    #: set of roll sets a block can be printed with.
    depth = max(len(entries) for entries in options)
    for entries, spools, rows in zip(options, option_spools, option_labs):
        while len(entries) < depth:
            entries.append(entries[-1])
            spools.append(spools[-1])
            rows.append(rows[-1])
    #: Each position's own wish list — what it costs if it is printed as intended.
    needs = [spools[0] for spools in option_spools]
    if counts is not None and len(counts) == len(pool):
        weights = np.asarray(counts, dtype=np.float64)
    else:
        weights = np.array(
            [int(np.count_nonzero(indices == position)) for position in range(len(pool))],
            dtype=np.float64,
        )

    #: The colour the picture actually has behind each colour of ``kept``.  This
    #: is what a plan is judged against; ``labs`` is what a plan may print.  A
    #: caller that cannot say (the legacy ``_fit_spool_budget``) gets each colour
    #: standing for itself, which is the same rule read one level down.
    if region_rgb is not None and len(region_rgb) == len(pool):
        picture_labs = list(
            _color.lab_from_rgb(
                np.rint(np.asarray(region_rgb, dtype=np.float64)).astype(np.uint8)
            )
        )
    else:
        picture_labs = [np.asarray(entry.lab, dtype=np.float64) for entry in pool]

    untouched = frozenset().union(*needs)
    if len(untouched) <= budget and require <= untouched and not (avoid & untouched):
        return []  # the picture already fits; there is nothing to decide

    # The rolls worth choosing between, in the order the picture asks for them, so
    # a cap can only ever cut the tail of the picture's own wish list.  Every block
    # offers its nearest colour first and its alternatives after, so no block's
    # first choice is squeezed out by another block's third.
    rolls: list[str] = []
    for wants in range(depth):
        for spools in option_spools:
            if wants >= len(spools):
                continue
            for spool in sorted(spools[wants]):
                if spool not in rolls and len(rolls) < _BUDGET_ROLL_LIMIT:
                    rolls.append(spool)
    if not rolls:
        return []

    picture_stack = np.asarray(picture_labs, dtype=np.float64)
    candidate_labs = np.asarray(option_labs, dtype=np.float64)  # (position, candidate, 3)
    #: ΔE00 the page would print for a block answered with candidate ``k`` …
    direct = np.empty((len(pool), depth), dtype=np.float64)
    #: … and for a block folded into one printed with candidate ``k`` of ``other``.
    fold = np.empty((len(pool), len(pool), depth), dtype=np.float64)
    for position in range(len(pool)):
        # np.atleast_1d: delta_e_2000 hands back a bare float when it compares one
        # pair, and both tables are indexed per candidate below.
        direct[position] = np.atleast_1d(
            _color.delta_e_2000(picture_stack[position], candidate_labs[position])
        )
        fold[position] = np.atleast_1d(
            _color.delta_e_2000(picture_stack[position], candidate_labs.reshape(len(pool) * depth, 3))
        ).reshape(len(pool), depth)

    def _printed(printable: Sequence[int], picked: Sequence[int]) -> list[PaletteEntry]:
        """The colours a plan prints — one entry per colour, not per block.

        Two blocks may end up answered with the same entry (one block's shortlist
        contains the winner of another).  They are then a single printed colour fed
        by two regions, which is exactly how the page reads a 合并 the user made by
        hand, so the answer is one row and not two identical ones.
        """
        colours: list[PaletteEntry] = []
        seen: set[str] = set()
        for rank, position in enumerate(printable):
            entry = options[position][picked[rank]]
            if entry.key in seen:
                continue
            seen.add(entry.key)
            colours.append(entry)
        return colours

    def _score(choice: Sequence[str]) -> tuple[tuple[float, ...], list[int], list[int]]:
        """How the picture reads under ``choice``, what fits, and printed with what.

        A block is printable when *one* of its candidates costs rolls the subset
        already holds, and it is printed with the nearest of those — so a block
        that would have had to be folded away for want of a roll may instead be
        answered with its second-nearest colour and stay on the picture.
        """
        subset = frozenset(choice)
        printable: list[int] = []
        picked: list[int] = []
        for position in range(len(pool)):
            best: tuple[float, int] | None = None
            for rank, need in enumerate(option_spools[position]):
                if need <= subset:
                    value = float(direct[position, rank])
                    if best is None or value < best[0]:
                        best = (value, rank)
            if best is not None:
                printable.append(position)
                picked.append(best[1])
        if not printable:
            return (float("inf"),), [], []
        rank_of = {position: rank for rank, position in enumerate(printable)}
        values = np.empty(len(pool), dtype=np.float64)
        for position in range(len(pool)):
            rank = rank_of.get(position)
            if rank is not None:
                values[position] = direct[position, picked[rank]]
                continue
            values[position] = min(
                fold[position, printable[other], picked[other]]
                for other in range(len(printable))
            )
        total = float(weights.sum()) or 1.0
        return objective(values, weights / total), printable, picked

    def _walk(
        available: Sequence[str], width: int
    ) -> tuple[tuple[float, ...], list[str], list[int], list[int]] | None:
        """Grow a set roll by roll, then swap rolls in and out until nothing improves."""
        chosen: list[str] = []
        best_score: tuple[float, ...] = (float("inf"),)
        best_printable: list[int] = []
        best_picked: list[int] = []
        while len(chosen) < width:
            gain: tuple[tuple[float, ...], str, list[int], list[int]] | None = None
            for spool in available:
                if spool in chosen:
                    continue
                score, printable, picked = _score([*chosen, spool])
                if gain is None or score < gain[0]:
                    gain = (score, spool, printable, picked)
            if gain is None:
                break
            chosen = [*chosen, gain[1]]
            best_score, best_printable, best_picked = gain[0], gain[2], gain[3]
        if not chosen:
            return None
        improved = True
        while improved:
            improved = False
            for position in range(len(chosen)):
                for spool in available:
                    if spool in chosen:
                        continue
                    trial = list(chosen)
                    trial[position] = spool
                    score, printable, picked = _score(trial)
                    if score < best_score:
                        chosen, best_score = trial, score
                        best_printable, best_picked = printable, picked
                        improved = True
        if not best_printable:
            return None
        return best_score, chosen, best_printable, best_picked

    size = min(budget, len(rolls))
    # With 「必须包含」 in force the answer is not necessarily a full ``size`` set:
    # the demanded rolls have to be spent, and a smaller set is allowed to be
    # that answer, so the smaller widths are walked as well.
    widths = [size]
    if require:
        widths = list(range(size, max(len(require), 1) - 1, -1))

    found: list[tuple[tuple[float, ...], list[str], list[int], list[int]]] = []
    for width in widths:
        if width < 1:
            continue
        if comb(len(rolls), width) <= _BUDGET_SUBSET_LIMIT:
            for candidate in combinations(rolls, width):
                score, printable, picked = _score(candidate)
                if printable:
                    found.append((score, list(candidate), printable, picked))
        else:
            # Too many combinations to walk: grow the set by whichever roll helps
            # most, then swap rolls in and out until nothing improves.  The
            # alternatives come from forbidding one roll of the winner at a time.
            walked = _walk(rolls, width)
            if walked is None:
                continue
            found.append(walked)
            if limit > 1:
                for spool in walked[1]:
                    rest = [value for value in rolls if value != spool]
                    if len(rest) < width:
                        continue
                    alternative = _walk(rest, width)
                    if alternative is not None:
                        found.append(alternative)
                    if len(found) >= limit:
                        break
    found.sort(key=lambda item: (item[0], len(item[1]), item[1]))

    def _build(printable: list[int], picked: list[int]) -> _Fit:
        """Print ``printable`` — each with the candidate ``picked`` chose — fold the rest."""
        colours = _printed(printable, picked)
        order = {colour.key: rank for rank, colour in enumerate(colours)}
        ranks = {
            position: order[options[position][picked[rank]].key]
            for rank, position in enumerate(printable)
        }
        remap = np.full(len(pool), -1, dtype=_INDEX_DTYPE)
        for position in range(len(pool)):
            if position in ranks:
                remap[position] = ranks[position]
                continue
            remap[position] = min(
                range(len(colours)),
                key=lambda rank: _color.delta_e_2000(
                    picture_labs[position], colours[rank].lab
                ),
            )
        folded = np.where(indices >= 0, remap[np.clip(indices, 0, None)], -1).astype(_INDEX_DTYPE)
        entries, region_counts, final, sources = _renumber_sources(colours, folded)
        return _Fit(
            entries=entries,
            counts=region_counts,
            indices=final,
            sources=tuple(origin[printable[source]] for source in sources),
            spools=frozenset(spool for entry in entries for spool in _spool_ids(entry)),
        )

    if not found:
        # Not one of the colours the picture picked can be printed inside the
        # budget — a budget of one against a picture whose dominant colour is a
        # mix, say.  Repainting with the closest affordable colour is the only
        # way to keep the promise; the largest region is what it has to match.
        fallback = _affordable_nearest(pool[0], budget, palette)
        if fallback is None:
            # Nothing affordable exists at all (a budget below the cheapest
            # recipe).  Keeping the largest colour beats deleting the picture.
            printable = [0]
        else:
            pool.append(fallback)
            origin.append(0)
            options.append([fallback])
            option_spools.append([frozenset(_spool_ids(fallback))])
            option_labs.append([np.asarray(fallback.lab, dtype=np.float64)])
            # It stands in for the largest region, so it is judged against that
            # region's picture colour.
            picture_labs.append(np.asarray(picture_labs[0], dtype=np.float64))
            printable = [len(pool) - 1]
        fit = _build(printable, [0])
        if require - fit.spools or (avoid & fit.spools):
            return []  # the only plan left breaks the filter; say so instead
        return [fit]

    fits: list[_Fit] = []
    seen: set[frozenset[str]] = set()
    for _value, _choice, printable, picked in found:
        if len(fits) >= limit:
            break
        spent = frozenset(
            spool for entry in _printed(printable, picked) for spool in _spool_ids(entry)
        )
        if require - spent or (avoid & spent):
            continue  # a plan that cannot obey the filter is not a plan
        if spent in seen:
            continue  # the same rolls under two objectives: list them once
        seen.add(spent)
        fits.append(_build(printable, picked))
    return fits


def _fit_spool_budget(
    kept: list[PaletteEntry],
    indices: np.ndarray,
    budget: int,
    palette: Sequence[PaletteEntry] | None = None,
    counts: Sequence[int] | np.ndarray | None = None,
    objective: BudgetObjective = _objective_closest,
) -> tuple[list[PaletteEntry], np.ndarray, np.ndarray] | None:
    """The best single plan inside ``budget``, or ``None`` when nothing changes.

    A thin wrapper over :func:`_budget_fits` kept for the callers and the tests
    that only ever want one answer.
    """
    fits = _budget_fits(kept, indices, budget, palette, counts, objective)
    if not fits:
        return None
    fit = fits[0]
    return fit.entries, fit.counts, fit.indices


@dataclass
class _Prepared:
    """Everything about a match that does not depend on which rolls get printed.

    Splitting the pipeline here is what makes offering several 耗材方案
    affordable: reading and reducing the picture, folding near-identical colours
    together and the exact palette search are the slow steps, and they are the
    same whichever rolls the user ends up printing.  Only the budget fit and the
    per-region means are redone per 方案.

    ``pictures`` is the colour each surviving block has to match, aligned with
    ``kept``.  It is ``None`` when the answer is the picture itself (the colours
    are then averaged out of ``image`` at the end), and set when the answer is
    the *current* blocks — 「按当前色块重新匹配」: there the block's colour already
    includes the user's 合并 and 更换颜色, so re-matching has to aim at that, not at
    the original pixels.
    """

    image: Image.Image | None
    palette: Sequence[PaletteEntry]
    kept: list[PaletteEntry]
    counts: np.ndarray
    indices: np.ndarray
    config: MatchSettings
    name: str
    pictures: np.ndarray | None = None
    #: Per block, the colours this block may be printed with, nearest first —
    #: ``alternatives[position][0]`` is ``kept[position]``.  A normal match leaves
    #: this ``None``: there the picture decides which colours are wanted and
    #: 「耗材上限」 only folds what it must.  On 「按当前色块重新匹配」 the blocks are the
    #: user's decision, so the roll search may answer one with its second-nearest
    #: colour instead of folding it away — see :func:`_budget_fits`.
    alternatives: list[list[PaletteEntry]] | None = None


def _prepare(
    source: str | Path | Image.Image,
    palette: Sequence[PaletteEntry],
    settings: MatchSettings | None,
    progress: ProgressFn | None,
) -> _Prepared:
    """Reduce the picture and answer every colour it keeps with its closest match."""
    if not palette:
        raise ValueError("调色板是空的，请先添加耗材。")
    config = (settings or MatchSettings()).clamped()
    name = str(source) if isinstance(source, (str, Path)) else ""

    _report(progress, "读取图片", 0.05)
    image = load_image(source).convert("RGBA")

    if max(image.size) > config.max_dimension:
        _report(progress, "缩放图片", 0.15)
        scale = config.max_dimension / float(max(image.size))
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(size, Image.LANCZOS)

    alpha = np.asarray(image.getchannel("A"), dtype=np.uint8)
    mask = alpha > config.alpha_threshold
    if not mask.any():
        if not alpha.any():
            # 整张图都是透明的：``convert("RGB")`` 会把这些像素变成纯黑，照旧当成
            # 有效像素匹配出去的就是一块纯黑底板 —— 用户拿到的东西和「一张全透明的
            # 图」毫无关系。宁可把话说清楚。
            raise ValueError("这张图整张都是透明的，没有可以打印的像素。")
        # 不透明度全都低于阈值（例如 alpha 被存成 1）：这时 RGB 还是画本身，当成
        # 整张图都有效比直接拒绝它合理。
        mask = np.ones(alpha.shape, dtype=bool)

    _report(progress, "减少图片颜色数量", 0.30)
    colours, classes = reduce_colours(image, config, mask)
    colours, classes = _merge_similar_colours(colours, classes, config.merge_delta_e)

    _report(progress, "在耗材与混色库里寻找最接近的颜色", 0.35)
    labs = _color.lab_from_rgb(colours)

    def _scan_rows(rows_done: int, rows_total: int) -> None:
        # The Euclidean pass over a four-million-row pool: silent for seconds
        # unless every block says something (m00940 第 2 条).
        _report(
            progress,
            "在耗材与混色库里寻找最接近的颜色",
            0.35 + 0.25 * rows_done / max(1, rows_total),
        )

    def _search_rows(rows_done: int, rows_total: int) -> None:
        # The exact pass is the long one: keep the bar moving through it.
        _report(
            progress,
            "在耗材与混色库里寻找最接近的颜色",
            0.60 + 0.15 * rows_done / max(1, rows_total),
        )

    chosen = _nearest_entries(
        labs, palette, _palette_table(palette), _search_rows, _scan_rows
    )

    _report(progress, "按像素分配颜色", 0.75)
    indices = np.where(mask, chosen[classes], -1).astype(_INDEX_DTYPE)

    # Drop palette entries the picture never used, then renumber largest first.
    used = sorted(int(value) for value in np.unique(indices) if value >= 0)
    counts = np.array([int(np.count_nonzero(indices == value)) for value in used], dtype=np.int64)
    order = np.argsort(-counts, kind="stable")
    # Indexed, never ``list(palette)``: with three-filament mixes on the pool is
    # millions of colours and materialising it is the one thing that cannot be
    # done.  ``used`` is at most the colour count, so this touches a handful.
    kept = [palette[used[int(position)]] for position in order]
    remap = np.full(len(palette), -1, dtype=_INDEX_DTYPE)
    for new_index, position in enumerate(order):
        remap[used[int(position)]] = new_index
    final = np.where(indices >= 0, remap[np.clip(indices, 0, None)], -1).astype(_INDEX_DTYPE)

    ordered_counts = counts[order]
    _report(progress, "合并相近颜色", 0.85)
    kept, ordered_counts, final = _merge_indistinguishable(
        kept, ordered_counts, final, config.merge_delta_e
    )
    return _Prepared(
        image=image,
        palette=palette,
        kept=kept,
        counts=ordered_counts,
        indices=final,
        config=config,
        name=name,
    )


def _region_means(prepared: _Prepared, fit: _Fit) -> np.ndarray | None:
    """The picture colour behind every surviving block, in ``fit`` order."""
    if not len(fit.entries):
        return None
    if prepared.pictures is not None:
        # ''按当前色块重新匹配'': the block colour is the target, so a region that
        # swallowed another one keeps the picture colour of the block that
        # survived instead of averaging the two.
        return np.asarray(
            [prepared.pictures[source] for source in fit.sources], dtype=np.uint8
        )
    if prepared.image is None:
        return None
    pixels = np.asarray(prepared.image.convert("RGB"), dtype=np.float64).reshape(-1, 3)
    flat = fit.indices.reshape(-1)
    valid = flat >= 0
    totals = np.zeros((len(fit.entries), 3), dtype=np.float64)
    np.add.at(totals, flat[valid], pixels[valid])
    totals /= np.maximum(fit.counts, 1)[:, None]
    return np.rint(totals).astype(np.uint8)


def _pool_region_means(prepared: _Prepared) -> np.ndarray | None:
    """The picture colour behind each colour of ``kept``, before the cap is spent.

    :func:`_region_means` averages the pixels of every block a *plan* ends up
    with; scoring a plan before it is built needs the same thing one level down,
    for the colours the cap starts from — each kept colour keeps its own picture
    colour, because that is what the plan is judged against.  These are kept as
    floats rather than rounded to bytes so that the rounding happens once, when
    the ΔE00 is measured.
    """
    if prepared.pictures is not None:
        return np.asarray(prepared.pictures, dtype=np.float64)
    if prepared.image is None:
        return None
    pixels = np.asarray(prepared.image.convert("RGB"), dtype=np.float64).reshape(-1, 3)
    flat = prepared.indices.reshape(-1)
    totals = np.zeros((len(prepared.kept), 3), dtype=np.float64)
    valid = flat >= 0
    np.add.at(totals, flat[valid], pixels[valid])
    return totals / np.maximum(prepared.counts, 1)[:, None]


def _source_report(
    prepared: _Prepared, fit: _Fit, picture_labs: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray] | None:
    """（每个图片颜色的 ΔE00，它背后的像素数）for the colours a plan prints.

    One entry per colour the picture was reduced to: how far the colour this plan
    prints for that region is from what the region actually looks like.  The page
    shows these three numbers — weighted average, worst, how many are past
    :data:`_BAD_BLOCK_DELTA_E` — and :func:`_budget_fits` ranks candidates with the
    same quantity, so a row can never promise something its own picture does not
    deliver.  A thin wrong line is one entry here, which is the point: averaging
    it into a big block is what hid it before.
    """
    if picture_labs is None or not len(fit.entries) or not len(prepared.kept):
        return None
    # Which surviving block each colour of ``kept`` was folded into.  Read off the
    # pixels: ``prepared.indices`` says which kept colour a pixel wanted and
    # ``fit.indices`` which printed block it got.
    rank_of_position = np.full(len(prepared.kept), -1, dtype=_INDEX_DTYPE)
    wanted = prepared.indices.reshape(-1)
    printed = fit.indices.reshape(-1)
    valid = (wanted >= 0) & (printed >= 0)
    if valid.any():
        rank_of_position[wanted[valid]] = printed[valid]
    values = np.zeros(len(prepared.kept), dtype=np.float64)
    entry_labs = [np.asarray(entry.lab, dtype=np.float64) for entry in fit.entries]
    for position in range(len(prepared.kept)):
        rank = int(rank_of_position[position])
        if 0 <= rank < len(entry_labs):
            values[position] = _color.delta_e_2000(
                picture_labs[position], entry_labs[rank]
            )
    return values, np.asarray(prepared.counts, dtype=np.float64)


def _identity_fit(prepared: _Prepared) -> _Fit:
    """The plan that changes nothing: print every surviving colour as it is."""
    return _Fit(
        entries=list(prepared.kept),
        counts=prepared.counts,
        indices=prepared.indices,
        sources=tuple(range(len(prepared.kept))),
        spools=frozenset(
            spool for entry in prepared.kept for spool in _spool_ids(entry)
        ),
    )


def _plans(
    prepared: _Prepared,
    max_spools: int,
    objective: BudgetObjective,
    *,
    limit: int = 1,
    require: frozenset[str] = frozenset(),
    avoid: frozenset[str] = frozenset(),
    progress: ProgressFn | None = None,
) -> list[MatchResult]:
    """Up to ``limit`` ways of printing ``prepared``, best first.

    「最多用 N 卷料」 comes after the merge: folding two invisible shades together
    frees spools for free, so the budget should only have to work on colours that
    actually survived.  An empty list means nothing can be printed under these
    instructions — the picture needs more rolls than the cap allows, or the 筛选
    of demanded / forbidden rolls cannot be met.
    """
    _report(progress, "按「耗材上限」合并颜色", 0.90)
    region_rgb = _pool_region_means(prepared)
    fits = _budget_fits(
        prepared.kept,
        prepared.indices,
        max_spools,
        prepared.palette,
        prepared.counts,
        objective,
        region_rgb=region_rgb,
        limit=limit,
        require=require,
        avoid=avoid,
        candidates=prepared.alternatives,
    )
    if not fits:
        # Nothing to decide: the picture already fits inside the cap and inside
        # the 筛选, so the colours survive exactly as prepared.  (When it does
        # not, the empty list is the honest answer: there is no plan to offer.)
        spools = frozenset(
            spool for entry in prepared.kept for spool in _spool_ids(entry)
        )
        capped = max_spools <= 0 or len(spools) <= max_spools
        if capped and require <= spools and not (avoid & spools):
            fits = [_identity_fit(prepared)]
    results: list[MatchResult] = []
    picture_labs = None
    if region_rgb is not None and len(region_rgb) == len(prepared.kept):
        picture_labs = _color.lab_from_rgb(
            np.rint(np.asarray(region_rgb, dtype=np.float64)).astype(np.uint8)
        )
    for fit in fits:
        _report(progress, "统计每种颜色的像素", 0.96)
        report = _source_report(prepared, fit, picture_labs)
        results.append(
            MatchResult(
                width=int(fit.indices.shape[1]),
                height=int(fit.indices.shape[0]),
                palette=fit.entries,
                indices=fit.indices,
                counts=fit.counts,
                settings=prepared.config,
                source=prepared.name,
                means=_region_means(prepared, fit),
                source_values=None if report is None else report[0],
                source_counts=None if report is None else report[1],
            )
        )
    return results


def _plan(
    prepared: _Prepared,
    max_spools: int,
    objective: BudgetObjective,
    progress: ProgressFn | None = None,
) -> MatchResult:
    """The single best way of printing ``prepared``; raises when there is none."""
    results = _plans(prepared, max_spools, objective, progress=progress)
    if not results:
        raise ValueError("这些条件下没有能用的耗材搭配。")
    return results[0]


def match_image(
    source: str | Path | Image.Image,
    palette: Sequence[PaletteEntry],
    settings: MatchSettings | None = None,
    *,
    progress: ProgressFn | None = None,
) -> MatchResult:
    """Reduce ``source`` to printable colours drawn from ``palette``.

    The default 方案; :func:`match_options` offers the alternatives the page
    shows, and its first entry is this same answer.
    """
    prepared = _prepare(source, palette, settings, progress)
    result = _plan(prepared, prepared.config.max_spools, _budget_objective, progress)
    _report(progress, "完成", 1.0)
    return result


def match_fidelity(result: MatchResult) -> tuple[float, float]:
    """（平均 ΔE00，最差 ΔE00）of a plan, the average pixel-weighted.

    These are the two numbers a 方案 is judged by on screen: the average is the
    overall promise, the worst is the patch that will look wrong.  Printing them
    beside each candidate lets the user compare plans the same way the program
    does, instead of taking its word for which one is best.
    """
    average, worst, _bad = match_fidelity_report(result)
    return average, worst


def _fidelity_values(result: MatchResult) -> tuple[np.ndarray, np.ndarray] | None:
    """（每种颜色的色差 ΔE00, 每种颜色的像素权重）that :func:`match_fidelity_report` reads.

    One entry per **colour of the picture** when the result carries the measurement
    (``source_values``, written by :func:`_plans`), otherwise one entry per printed
    block, read off its mean picture colour.  Both callers — the row's numbers and
    the ranking that puts the rows in order — go through here, so a row can never
    be ranked by one measurement and printed with another (the 1.6.14 complaint).
    """
    values = result.source_values
    counts = result.source_counts
    if values is not None and counts is not None and len(values) and len(values) == len(counts):
        total = float(np.sum(counts)) or 1.0
        return np.asarray(values, dtype=np.float64), np.asarray(counts, dtype=np.float64) / total
    if result.means is None or not len(result.palette):
        return None
    values = np.asarray(
        [
            _color.delta_e_2000(
                _color.lab_from_rgb(np.asarray(result.means[index], dtype=np.uint8)), entry.lab
            )
            for index, entry in enumerate(result.palette)
        ],
        dtype=np.float64,
    )
    total = float(np.sum(result.counts)) or 1.0
    return values, np.asarray(result.counts, dtype=np.float64) / total


def match_fidelity_report(result: MatchResult) -> tuple[float, float, int]:
    """（平均 ΔE00，最差 ΔE00，差块个数）of a plan.

    Measured per **colour of the picture**: the third number counts the colours
    off by more than :data:`_BAD_BLOCK_DELTA_E`, which is the 「混不出来」 the eye
    actually notices — the nose, the eyebrows, a whole small shape.  An average
    alone hides that choice, and an average *per printed block* hides it twice
    over: folding a thin dark line into a large pale block barely moves the
    block's mean, so a plan that erases the line used to look perfect (1.6.14).
    Counting colours keeps it honest whatever the block sizes are.

    Results built by :func:`_plans` carry the measurement (:attr:`MatchResult.
    source_values`); one built by hand falls back to comparing each block's mean
    picture colour with the colour that prints it, which is the same quantity
    read at block resolution.
    """
    measured = _fidelity_values(result)
    if measured is None:
        return 0.0, 0.0, 0
    values, weights = measured
    return (
        float(weights @ values),
        float(values.max()),
        int(np.count_nonzero(values > _BAD_BLOCK_DELTA_E)),
    )


@dataclass(frozen=True)
class BudgetOption:
    """One 耗材方案: which rolls, why it was picked, and what it prints."""

    key: str
    title: str
    note: str
    result: MatchResult

    @property
    def spools(self) -> tuple[str, ...]:
        """The rolls this plan loads, in the order the picture asks for them."""
        ordered: list[str] = []
        for entry in self.result.palette:
            for spool in _spool_ids(entry):
                if spool not in ordered:
                    ordered.append(spool)
        return tuple(ordered)

    @property
    def fidelity(self) -> tuple[float, float]:
        return match_fidelity(self.result)

    @property
    def defects(self) -> int:
        """How many printed colours are off by more than ΔE00 5 (差块个数)."""
        return match_fidelity_report(self.result)[2]


def _collapse(prepared: _Prepared) -> _Prepared:
    """Fold blocks that ended up with the same colour into one.

    Two blocks can come out identical: 「合并」 folds one into another, or the
    palette answers two regions with the very same recipe.  On the plate they are
    one colour and one row, and — the reason this matters here — one roll
    requirement instead of two, which is what lets a small 「耗材上限」 go further.
    The surviving block is the largest one and keeps its own picture colour,
    which is the rule 合并 follows.
    """
    entries = list(prepared.kept)
    groups: dict[str, list[int]] = {}
    order: list[str] = []
    for position, entry in enumerate(entries):
        if entry.key not in groups:
            groups[entry.key] = []
            order.append(entry.key)
        groups[entry.key].append(position)
    if len(order) == len(entries):
        return prepared
    rank = {key: index for index, key in enumerate(order)}
    remap = np.array([rank[entry.key] for entry in entries], dtype=_INDEX_DTYPE)
    folded = np.where(
        prepared.indices >= 0, remap[np.clip(prepared.indices, 0, None)], -1
    ).astype(_INDEX_DTYPE)
    survivors = [groups[key][0] for key in order]
    kept, counts, final, sources = _renumber_sources(
        [entries[position] for position in survivors], folded
    )
    pictures = None
    if prepared.pictures is not None:
        pictures = np.asarray(
            [prepared.pictures[survivors[int(source)]] for source in sources],
            dtype=np.uint8,
        )
    alternatives = None
    if prepared.alternatives is not None:
        alternatives = [
            list(prepared.alternatives[survivors[int(source)]]) for source in sources
        ]
    return replace(
        prepared,
        kept=kept,
        counts=counts,
        indices=final,
        pictures=pictures,
        alternatives=alternatives,
    )


def _with_required(
    prepared: _Prepared,
    require: frozenset[str],
    palette: Sequence[PaletteEntry] | None,
    *,
    pinned: frozenset[int] = frozenset(),
) -> _Prepared:
    """Swap colours in so every demanded roll is really spent.

    「必须包含」 is about the machine — this roll is loaded, or the user wants it
    used up — so it has to reach the recipes.  For each demanded roll the picture
    does not spend yet, the block that can take it most cheaply gives up its
    colour for the closest colour that roll can print.  The pixels do not move;
    only what a block is printed with does.  When no block can take it the
    palette simply cannot print that roll, and the plans are dropped later, which
    is what 「没有符合筛选的搭配」 means.
    """
    if not require:
        return prepared
    entries = list(prepared.kept)
    for spool in sorted(require):
        spent = frozenset(
            value for entry in entries for value in _spool_ids(entry)
        )
        if spool in spent:
            continue
        best: tuple[float, int, PaletteEntry] | None = None
        for position, entry in enumerate(entries):
            if position in pinned:
                continue
            replacement = _nearest_spending(entry, spool, palette)
            if replacement is None or replacement.key == entry.key:
                continue
            cost = float(prepared.counts[position]) * _color.delta_e_2000(
                entry.lab, replacement.lab
            )
            if best is None or cost < best[0]:
                best = (cost, position, replacement)
        if best is None:
            continue  # nothing in the palette prints this roll
        entries[best[1]] = best[2]
    return replace(prepared, kept=entries)


def _without_banned(
    prepared: _Prepared,
    avoid: frozenset[str],
    palette: Sequence[PaletteEntry] | None,
    *,
    pinned: frozenset[int] = frozenset(),
) -> _Prepared:
    """Take the forbidden rolls out of the recipes.

    「不要用」 is not advice about a list of rolls, it is about what may be
    printed: a colour whose recipe names a forbidden spool is a colour this print
    cannot make, so the block is re-pointed at the closest colour that spends
    none of them.  When nothing at all is left, the block keeps its colour and
    the plans that would need it are dropped — the same honest 「没有符合筛选的搭配」.
    """
    if not avoid:
        return prepared
    entries = list(prepared.kept)
    changed = False
    for position, entry in enumerate(entries):
        if position in pinned or not (avoid & set(_spool_ids(entry))):
            continue
        replacement = _nearest_free(entry, avoid, palette)
        if replacement is None or replacement.key == entry.key:
            continue
        entries[position] = replacement
        changed = True
    return replace(prepared, kept=entries) if changed else prepared


#: The 耗材方案 the page offers, in the order the user sees them.  ``key`` is what
#: a project file remembers about the chosen one.  Every one of them fits inside
#: 「耗材上限」: that cap is the user's own instruction about how many rolls this
#: print may load, so a plan that breaks it is not an option to offer, it is a
#: different setting.  The plans differ in *which* colours they sacrifice — on the
#: user's pig 「总体最接近」 keeps the average lowest and 「小色块优先」 costs average
#: to print the nose — and that is a choice about the picture, not a computable
#: fact, so the user makes it.  **1.6.27 起第一行是大色差色块优先**（用户 m06704 第 3 条
#: 给的顺序：先把大色差色块认出来，再在限制条件下给它配色，耗材上限优先用满）；平均
#: 领先的那一档退到第二行，想整张图最像的点一下就有。
#: ... and how many rows each one offers.  「耗材方案太少了，多推荐几组」 is answered
#: with more *different roll sets* inside the same rules: the same subsets the
#: search already scores are ranked, so the second row is the runner-up rather
#: than a worse objective.  1.6.15（m00830 第 4 条「耗材方案多给些选项」）把每一档都
#: 加宽：前两档各五组，后两档各四组。印出来是同一套卷的行仍然只留一行，所以卷库小、
#: 不少组合打平的图上会少几行。
_PLAN_SPECS: tuple[tuple[str, str, str, BudgetObjective, int], ...] = (
    (
        "priority",
        "大色差色块优先",
        "先把差得最明显的色块认出来，耗材上限优先用满，再用剩下的把整张图拉近。",
        _objective_priority,
        5,
    ),
    (
        "closest",
        "总体最接近",
        "整张图算下来最像；个别小色块可能差一点。",
        _objective_closest,
        5,
    ),
    (
        "worst",
        "最不像的色块优先",
        "先把差得最明显的那一块拉回来，平均会稍差。",
        _objective_worst,
        4,
    ),
    (
        "distinct",
        "小色块优先",
        "眼睛、鼻子这类小色块更准；超过 ΔE00 5 的色块最少。",
        _objective_distinct,
        4,
    ),
)


def _option_title(key: str, title: str, rank: int) -> tuple[str, str]:
    """（key，名称）for the ``rank``-th best rolls of one objective.

    The name carries no 「备选 N」.  Two rows of the same objective are told apart by
    the roll colours printed on them, which is the fact the user needs when
    choosing; 「备选 1」 named a difference nobody could see without counting rows.
    The ``#N`` stays in the *key*: a project file remembers which row it was saved
    with, so the row has to be identifiable even when it is not named.
    """
    if rank <= 0:
        return key, title
    return f"{key}#{rank}", title


def plan_caption(key: str) -> tuple[str, str]:
    """（名称，说明）for a 方案 key, 「工程里存的那一套」 for anything else.

    「打开工程」 hands the page the plan the file was saved with, and that key can
    name a 方案 this build no longer offers (an older file, a hand-edited one).
    The row still has to say something true, so it says where it came from.  A
    key ending in ``#N`` is the Nth alternative of the same objective and is named
    after that objective: the alternatives differ in which rolls they buy, and the
    row prints those.
    """
    base, _, _rank = key.partition("#")
    for spec_key, title, note, _objective, _count in _PLAN_SPECS:
        if spec_key != base:
            continue
        return title, note
    return "工程里存的那一套", "保存工程时用的搭配；点「重新匹配」可以再挑一套。"


def plan_help() -> str:
    """What the 耗材方案 panel offers, for the 帮助 window.

    The panel used to print one of these sentences under the rows and 「（备选 N）」
    in them; the user asked for the explanations to move into 帮助 so the panel
    stays about this picture.  The text lives beside the specs on purpose — a new
    方案 cannot be added without its explanation following it.
    """
    lines = [
        "「耗材方案」这几行：",
        "· 同一个「耗材上限」里有好几种搭法，每一行是一种。行里写着这一套的平均色差、最差色差、"
        "差块个数和用几卷，后面那几个小方块就是这一套要装哪几卷料。",
        f"· 「差块」是色差 ΔE00 超过 {_BAD_BLOCK_DELTA_E:g} 的颜色个数，按图片自己的颜色数："
        "图片被归并成多少种颜色，就按多少种算。平均色差看不出是哪一块画坏了，差块个数看得出——"
        "越少，越不容易一眼看出偏差。卷少、颜色多的时候这个数天生就大：几卷料印不出几十种颜色。",
    ]
    lines.extend(
        f"· {title}：{note}" for _key, title, note, _objective, _count in _PLAN_SPECS
    )
    lines.append(
        "· 同一种口径下的第二、第三套不再另起名字：按行里那几卷的颜色认就行，"
        "名称只说明它是按什么标准挑出来的。"
    )
    lines.append(
        f"· 不点名的时候用第一行（{_PLAN_SPECS[0][1]}）：「重新匹配」和打开工程时没带方案的图"
        "都落在这一行；想要别的口径，点那一行再按下面的「用这一套」就行。"
    )
    return "\n".join(lines)


def _ranked_plans(
    pool: Sequence[MatchResult], objective: BudgetObjective
) -> list[MatchResult]:
    """``pool`` sorted by ``objective``, using the numbers the rows print.

    The sort key is :func:`_fidelity_values` — the same measurement
    :func:`match_fidelity_report` turns into 「平均 / 最差 / 差块」 — so the row on
    top really is the best one under the name it carries.  A plan that prints the
    same picture with more rolls wins the tie: 「选择耗材时优先用满耗材上限」
    (m06704 第 3 条) says the cap is there to be spent, so of two ways that print
    the same colours the one that uses more of the allowance is the answer.
    """
    indexed: list[tuple[tuple[float, ...], int, MatchResult]] = []
    for index, result in enumerate(pool):
        measured = _fidelity_values(result)
        if measured is None:
            continue
        values, weights = measured
        score = objective(values, weights) + (-float(len(_option_rolls(result))),)
        indexed.append((score, index, result))
    indexed.sort(key=lambda item: (item[0], item[1]))
    return [result for _score, _index, result in indexed]


def _option_rolls(result: MatchResult) -> frozenset[str]:
    """Every spool a plan spends, whichever colour it spends it on."""
    return frozenset(spool for entry in result.palette for spool in _spool_ids(entry))


def _offer_options(
    prepared: _Prepared,
    *,
    require: frozenset[str] = frozenset(),
    avoid: frozenset[str] = frozenset(),
    progress: ProgressFn | None = None,
) -> list[BudgetOption]:
    """Turn the plans every spec asks for into the rows the page shows.

    Each objective searches for itself, and then the **union** of what all of them
    found is re-ranked per objective.  Searching separately is a heuristic (the
    budget walk is a beam, not an exhaustive enumeration), so one objective's top
    answer can also be sitting in another objective's list — on the pig the 小色块
    优先 rows came out at 差块 13 while a 总体最接近 row was 12, not because the
    ranking was wrong but because that search never walked to those rolls.  With
    the union, every objective's first row is the best of everything found.

    The rows are then handed out **round by round** — the best of each objective,
    then the second best of each, and so on — so no objective can eat the whole
    pool and leave another one only leftovers.  Because the best answer under one
    objective is often also the best under another (both 总体最接近 and 最不像的
    色块优先 want the plan with the smallest worst block here), a roll set another
    row already prints is skipped: the row under the second name is that
    objective's best *remaining* answer, which ties the one above rather than
    beating it, and no two rows print the same rolls.

    A candidate that cannot obey the 筛选 never reaches the list: the cap and the
    demanded / forbidden rolls are instructions about this machine, not
    preferences to trade against picture quality.
    """
    cap = prepared.config.max_spools
    total = max(1, sum(spec[4] for spec in _PLAN_SPECS))
    pool: list[MatchResult] = []
    done = 0
    for key, title, note, objective, count in _PLAN_SPECS:
        _report(progress, f"试着搭配耗材（{title}）", 0.88 + 0.1 * done / total)
        found = _plans(prepared, cap, objective, limit=count, require=require, avoid=avoid)
        done += len(found)
        for result in found:
            rolls = _option_rolls(result)
            if require - rolls or (avoid & rolls):
                continue  # a plan that breaks the 筛选 is not a plan
            pool.append(result)
    ranked = {
        key: _ranked_plans(pool, objective)
        for key, _title, _note, objective, _count in _PLAN_SPECS
    }
    cursors = {key: 0 for key, _title, _note, _objective, _count in _PLAN_SPECS}
    picks: dict[str, list[MatchResult]] = {key: [] for key in cursors}
    seen: set[frozenset[str]] = set()
    for round_index in range(max(spec[4] for spec in _PLAN_SPECS)):
        for key, title, note, _objective, count in _PLAN_SPECS:
            if round_index >= count:
                continue
            rows = ranked[key]
            while cursors[key] < len(rows):
                candidate = rows[cursors[key]]
                cursors[key] += 1
                rolls = _option_rolls(candidate)
                if rolls in seen:
                    continue  # the same rolls under two objectives: list them once
                picks[key].append(candidate)
                seen.add(rolls)
                break
    options: list[BudgetOption] = []
    for key, title, note, _objective, _count in _PLAN_SPECS:
        for rank, result in enumerate(picks[key]):
            option_key, option_title = _option_title(key, title, rank)
            options.append(
                BudgetOption(
                    key=option_key,
                    title=option_title,
                    note=note,
                    result=result,
                )
            )
    _report(progress, "完成", 1.0)
    return options


def match_options(
    source: str | Path | Image.Image,
    palette: Sequence[PaletteEntry],
    settings: MatchSettings | None = None,
    *,
    require: Iterable[str] = (),
    avoid: Iterable[str] = (),
    progress: ProgressFn | None = None,
) -> list[BudgetOption]:
    """Every 耗材方案 worth showing for ``source``, best first and deduplicated.

    The program does not pretend there is one right answer.  On the user's own
    pig, four rolls print the nose pale whichever way the search is pointed —
    one 方案 gets the average down, another spends a little average to print the
    nose exactly — and that is a choice about the picture, so it is offered
    rather than assumed.

    ``require`` and ``avoid`` are the user's 筛选: rolls this print must load and
    rolls it must not.  They are applied to the recipes before the rolls are
    chosen, so a demanded roll really is spent and a forbidden one really is not
    — and a plan that cannot manage that is left out instead of shown.
    """
    prepared = _prepare(source, palette, settings, progress)
    wanted = frozenset(require)
    banned = frozenset(avoid)
    if wanted:
        _report(progress, "换成「必须包含」的耗材", 0.86)
        prepared = _with_required(prepared, wanted, palette)
    if banned:
        _report(progress, "换掉「不要用」的耗材", 0.87)
        prepared = _without_banned(prepared, banned, palette)
    prepared = _collapse(prepared)
    return _offer_options(prepared, require=wanted, avoid=banned, progress=progress)


def _with_alternatives(
    prepared: _Prepared,
    palette: Sequence[PaletteEntry],
    *,
    locked: frozenset[int] = frozenset(),
    progress: ProgressFn | None = None,
) -> _Prepared:
    """Give every block of a re-match the other colours it could be printed with.

    「按当前色块重新匹配」 answers each block with the single closest colour the
    library can print it with, and then 「耗材上限」 folds away whatever those colours
    cost too many rolls.  On the reporter's pig that fold ate the picture: six
    blocks after a 合并 came back as two, and the eyes, brows and nose were gone —
    the pink block's nearest wanted one pair of rolls, the three brown blocks
    wanted two more pairs, and only the skin fitted inside four.

    The blocks are what the button promises to keep, so each block brings more than
    its one nearest colour.  The rolls the blocks ask for together are the rolls the
    search may spend (:data:`_BUDGET_ROLL_LIMIT` of them at most); within those, each
    block keeps the closest colour the library can print for **every** roll set it
    could be given — a single roll or a two-filament mix — nearest first.  The
    search can then spend its rolls where they buy the most and still print every
    block: the pink takes a mix that shares a roll with the skin, the blacks take
    one that shares a roll with the reds, and all six blocks fit in four rolls.
    """
    if len(prepared.kept) < 2:
        return prepared
    #: The rolls the blocks ask for, largest block first: the block whose colour is
    #: most of the picture gets its pick before the tail crowds the list.
    wish: list[str] = []
    for entry in prepared.kept:
        for spool in _spool_ids(entry):
            if spool not in wish and len(wish) < _BUDGET_ROLL_LIMIT:
                wish.append(spool)
    if not wish:
        return prepared
    if prepared.pictures is not None:
        labs = _color.lab_from_rgb(np.asarray(prepared.pictures, dtype=np.uint8))
    else:
        labs = np.asarray([entry.lab for entry in prepared.kept], dtype=np.float64)

    #: The palette by the rolls each colour spends, so a roll set can be asked for
    #: its best colour without walking four million triples.  Only a spool or a
    #: two-filament mix can answer a roll set here — every ``roll_sets`` key is one
    #: or two rolls — so a three-filament pool is asked for its head and never for
    #: its tail: grouping the tail was 4.3 M :class:`PaletteEntry` objects built to
    #: look up keys nothing asks for, and it is what froze the window on a
    #: 41-spool library (m00940 第 2 条).  ``head`` keeps the palette's own
    #: indices, so the group values are still palette positions.
    head = getattr(palette, "head", None)
    rows: Sequence[PaletteEntry] = head if head is not None else palette
    table = _palette_table(palette)
    groups: dict[tuple[str, ...], list[int]] = {}
    for index, entry in enumerate(rows):
        groups.setdefault(tuple(sorted(_spool_ids(entry))), []).append(index)

    roll_sets: list[tuple[str, ...]] = [(spool,) for spool in wish]
    roll_sets += [tuple(sorted(pair)) for pair in combinations(wish, 2)]

    total = max(1, len(prepared.kept))
    alternatives: list[list[PaletteEntry]] = []
    for position, entry in enumerate(prepared.kept):
        if position in locked:
            alternatives.append([entry])
            _report(progress, "给每个色块找可替换的颜色", 0.62 + 0.24 * (position + 1) / total)
            continue
        scored: list[tuple[float, int]] = []
        for step, key in enumerate(roll_sets):
            indices = groups.get(key)
            if not indices:
                continue
            values = np.atleast_1d(_color.delta_e_2000(labs[position], table[indices]))
            best = int(np.argmin(values))
            scored.append((float(values[best]), int(indices[best])))
            # A block with 22 rolls to spend has 253 roll sets to try, and a node
            # that only spoke at the end of it would leave the bar sitting still
            # for the whole stretch.
            if step % _PROGRESS_ROLL_SETS == _PROGRESS_ROLL_SETS - 1:
                _report(
                    progress,
                    "给每个色块找可替换的颜色",
                    0.62 + 0.24 * (position + (step + 1) / len(roll_sets)) / total,
                )
        scored.sort(key=lambda item: item[0])
        near = [palette[index] for _value, index in scored[:_REMATCH_SHORTLIST]]
        rest = [candidate for candidate in near if candidate.key != entry.key]
        alternatives.append([entry, *rest])
        _report(progress, "给每个色块找可替换的颜色", 0.62 + 0.24 * (position + 1) / total)
    return replace(prepared, alternatives=alternatives)


def rematch_options(
    result: MatchResult,
    palette: Sequence[PaletteEntry],
    settings: MatchSettings | None = None,
    *,
    pinned: Mapping[int, PaletteEntry] | None = None,
    require: Iterable[str] = (),
    avoid: Iterable[str] = (),
    progress: ProgressFn | None = None,
) -> list[BudgetOption]:
    """「按当前色块重新匹配」: keep the blocks on screen, re-choose the rolls.

    The picture is not consulted again.  The blocks are already the answer — 删除的
    颜色 are outside ``indices`` and a 合并 left its target's colour in ``means`` —
    so each one is answered with the closest colour the library can print and the
    rolls are fitted inside 「耗材上限」 exactly as a normal match.  That is what
    makes the button useful: fix the blocks first, then ask what to load.

    ``pinned`` holds the blocks the user replaced by hand (block index of
    ``result`` → the colour they chose): those keep their colour, so re-matching
    does not silently undo a 更换颜色.
    """
    if not palette:
        raise ValueError("调色板是空的，请先添加耗材。")
    if not len(result.palette):
        return []
    config = (settings or result.settings or MatchSettings()).clamped()
    wanted = frozenset(require)
    banned = frozenset(avoid)
    choices = dict(pinned or {})

    targets = np.asarray(
        [result.region_colour(index) for index in range(len(result.palette))],
        dtype=np.uint8,
    )
    _report(progress, "读取当前色块", 0.08)
    labs = _color.lab_from_rgb(targets)

    def _scan_rows(rows_done: int, rows_total: int) -> None:
        _report(
            progress,
            "在耗材与混色库里寻找最接近的颜色",
            0.10 + 0.30 * rows_done / max(1, rows_total),
        )

    def _search_rows(rows_done: int, rows_total: int) -> None:
        _report(
            progress,
            "在耗材与混色库里寻找最接近的颜色",
            0.40 + 0.15 * rows_done / max(1, rows_total),
        )

    chosen = _nearest_entries(
        labs, palette, _palette_table(palette), _search_rows, _scan_rows
    )
    entries = [
        choices.get(index, palette[int(chosen[index])])
        for index in range(len(result.palette))
    ]
    prepared = _Prepared(
        image=None,
        palette=palette,
        kept=entries,
        counts=np.asarray(result.counts, dtype=np.int64),
        indices=np.asarray(result.indices, dtype=_INDEX_DTYPE).copy(),
        config=config,
        name=result.source,
        pictures=targets,
    )
    prepared = _collapse(prepared)
    pinned_keys = {entry.key for entry in choices.values()}
    held = frozenset(
        position
        for position, entry in enumerate(prepared.kept)
        if entry.key in pinned_keys
    )
    if wanted:
        _report(progress, "换成「必须包含」的耗材", 0.58)
        prepared = _with_required(prepared, wanted, palette, pinned=held)
    if banned:
        _report(progress, "换掉「不要用」的耗材", 0.60)
        prepared = _without_banned(prepared, banned, palette, pinned=held)
    prepared = _with_alternatives(prepared, palette, locked=held, progress=progress)
    return _offer_options(prepared, require=wanted, avoid=banned, progress=progress)

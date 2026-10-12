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
from typing import Callable, Sequence

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
_SHORTLIST = 24

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
    """
    entries: list[PaletteEntry] = []
    for filament in library:
        if filament.hidden:
            continue
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
    table: np.ndarray, targets: np.ndarray, shortlist: int
) -> np.ndarray:
    """The ``shortlist`` rows of ``table`` closest to each row of ``targets``.

    Distances are Euclidean in Lab and computed in blocks: the three-filament
    pool is millions of rows, so the full ``(targets × rows)`` matrix would be
    gigabytes.
    """
    count = int(table.shape[0])
    take = min(shortlist, count)
    target_sq = (targets**2).sum(axis=1)[:, None]
    best_indices = np.zeros((len(targets), 0), dtype=np.int64)
    best_scores = np.zeros((len(targets), 0), dtype=np.float32)
    for start in range(0, count, _SEARCH_BLOCK):
        block = table[start : start + _SEARCH_BLOCK]
        square = target_sq + (block**2).sum(axis=1)[None, :] - 2.0 * (targets @ block.T)
        np.maximum(square, 0.0, out=square)
        picked = np.argpartition(square, take - 1, axis=1)[:, :take]
        best_indices = np.concatenate([best_indices, picked + start], axis=1)
        best_scores = np.concatenate(
            [best_scores, np.take_along_axis(square, picked, axis=1)], axis=1
        )
    width = best_scores.shape[1]
    keep = np.argpartition(best_scores, min(shortlist, width) - 1, axis=1)[:, :shortlist]
    return np.take_along_axis(best_indices, keep, axis=1)


def _nearest_entries(
    labs: np.ndarray,
    entries: Sequence[PaletteEntry],
    table: np.ndarray | None = None,
    on_row: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Index of the palette entry closest to each row of ``labs``.

    A Euclidean pass in Lab narrows the field, then CIEDE2000 picks the winner so
    the answer agrees with the ΔE the UI reports.

    ``on_row`` is called with ``(rows_done, rows_total)`` as the exact pass
    walks the colours — it is the only step long enough to need saying so.
    """
    if not entries:
        return np.zeros(len(labs), dtype=np.int64)
    if table is None:
        table = _palette_table(entries)
    count = int(table.shape[0])
    if count == 0:
        return np.zeros(len(labs), dtype=np.int64)

    targets = np.asarray(labs, dtype=np.float32)
    best_indices = _shortlist_indices(table, targets, _SHORTLIST)

    # A three-filament pool is a superset of the two-filament one, but a wider
    # table can push a two-filament colour out of the Euclidean shortlist — and
    # then ticking 「也算三色配方」 would quietly make a colour *worse*, which is
    # the one thing it must never do.  So the head gets its own shortlist and the
    # two are merged before the exact pass.
    head_size = int(getattr(entries, "head_size", 0) or 0)
    if 0 < head_size < count:
        head_indices = _shortlist_indices(table[:head_size], targets, _SHORTLIST)
        best_indices = np.concatenate([best_indices, head_indices], axis=1)

    rows = len(targets)
    out = np.zeros(rows, dtype=np.int64)
    for row in range(rows):
        candidates = best_indices[row]
        best_index = int(candidates[0])
        best = float("inf")
        for index in candidates:
            value = _color.delta_e_2000(labs[row], table[index])
            if value < best:
                best = value
                best_index = int(index)
        out[row] = best_index
        if on_row is not None and (row % _PROGRESS_ROWS == _PROGRESS_ROWS - 1 or row == rows - 1):
            on_row(row + 1, rows)
    return out


def _pack(rgb: np.ndarray) -> np.ndarray:
    channels = rgb.astype(np.int64)
    return (channels[..., 0] << 16) | (channels[..., 1] << 8) | channels[..., 2]


def _unpack(key: int) -> tuple[int, int, int]:
    return ((key >> 16) & 0xFF, (key >> 8) & 0xFF, key & 0xFF)


#: How many clusters to ask the quantiser for per colour the user requested.
#: See ``reduce_colours`` for why overshooting and folding back is better than
#: asking for the final number directly.
_OVERSAMPLE = 6


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


def _dither_masked(
    rgb: np.ndarray, keep: np.ndarray, palette: np.ndarray, pad: int = 2
) -> np.ndarray | None:
    """Diffuse the picture onto a fixed palette, inside the mask only.

    The masked reduction compacts the opaque pixels into a one-row strip, and a
    strip has no left or right neighbour for Floyd–Steinberg to use — that is
    why 抖动 used to be refused outright on a picture with transparency.  Here
    the two-dimensional picture is kept instead, cropped to the artwork (plus a
    ``pad``-pixel skirt) and filled inwards first, and the palette is the one the
    picture already earned rather than a fresh median cut — so the transparent
    background cannot win back the colour budget the mask exists to protect.

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
    source = Image.fromarray(np.clip(np.rint(filled), 0, 255).astype(np.uint8), "RGB")

    rows = np.asarray(palette, dtype=np.int64).reshape(-1, 3)[:256]
    if len(rows) < 2:
        return None
    # The table has to be 256 entries wide, and anything past the palette must
    # REPEAT real colours: padded black would be a colour the artwork never had,
    # and the quantiser would happily hand dark pixels to it (measured: five
    # classes out of a four-colour palette).  Indices are folded back below.
    tiled = np.tile(rows, (int(np.ceil(256 / len(rows))), 1))[:256]
    table = Image.new("P", (1, 1))
    table.putpalette([int(value) for value in tiled.reshape(-1)])
    mapped = np.asarray(
        source.quantize(palette=table, dither=Image.Dither.FLOYDSTEINBERG), dtype=np.int64
    )
    if mapped.shape != inside.shape:
        return None
    mapped %= len(rows)
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
    if keep is not None:
        # Compact the opaque pixels into a one-row strip so the quantiser sees
        # the artwork and nothing else.  Median cut works from the colour
        # histogram, so the arrangement is irrelevant to the result.
        source = Image.fromarray(
            np.clip(np.rint(smoothed[keep]), 0, 255).astype(np.uint8).reshape(1, -1, 3), "RGB"
        )
        keys = _pack(rgb[keep])
    else:
        source = Image.fromarray(np.clip(np.rint(smoothed), 0, 255).astype(np.uint8), "RGB")
        keys = _pack(rgb)

    unique_keys, inverse = np.unique(keys, return_inverse=True)
    if unique_keys.size <= settings.max_colours:
        colours = np.array([_unpack(int(key)) for key in unique_keys], dtype=np.int64)
        if keep is None:
            return colours, inverse.reshape(rgb.shape[:2]).astype(np.int64)
        classes = np.full(rgb.shape[:2], -1, dtype=np.int64)
        classes[keep] = inverse
        return colours, classes

    # Ask for more clusters than the user did, then collapse the ones a person
    # cannot tell apart.  Median cut splits whichever box holds the most
    # pixels, so on a big flat fill it happily spends every slot on a dozen
    # shades of the same beige and never reaches the 1,700 dark pixels that are
    # the eyes — the picture came out with no dark colour at all.  Overshooting
    # and folding afterwards gives the small distinct colours a cluster to
    # themselves while the near-identical ones still collapse to one.
    budget = min(256, max(settings.max_colours, settings.max_colours * _OVERSAMPLE))
    dither = Image.Dither.NONE
    if keep is None and settings.dither:
        dither = Image.Dither.FLOYDSTEINBERG
    # The compacted strip has no meaningful neighbour relationship, so error
    # diffusion there would scatter noise instead of smoothing gradients.
    quantized = source.quantize(
        colors=budget,
        method=Image.Quantize.MEDIANCUT,
        dither=dither,
    )
    class_map = np.asarray(quantized, dtype=np.int64)
    if keep is None:
        classes = class_map
        region = classes
    else:
        classes = np.full(rgb.shape[:2], -1, dtype=np.int64)
        classes[keep] = class_map.reshape(-1)
        region = classes
    used = np.unique(region[region >= 0])
    lookup = np.full(int(region.max()) + 1, -1, dtype=np.int64)
    lookup[used] = np.arange(len(used))
    classes = np.where(region >= 0, lookup[np.clip(region, 0, None)], -1)
    colours = _class_means(rgb, classes, len(used))
    result = _fold_clusters(colours, classes, settings.max_colours, settings.merge_delta_e)
    if keep is None or not settings.dither or len(result[0]) < 2:
        # Without a mask the quantiser above already dithered; with a single
        # colour left there is nothing for the error to diffuse towards.
        return result
    classes = _dither_masked(rgb, keep, result[0])
    if classes is None:
        return result
    used = np.unique(classes[classes >= 0])
    if used.size < 2:
        return result
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
    """Fold the oversampled clusters down to at most ``limit`` of them.

    A plain union-find over "everything within ``threshold``" is not usable
    here.  On a smooth gradient every cluster is within ``threshold`` of its
    neighbour, so the union chains across the whole ramp and a 金→红 gradient
    collapses to a single colour — measured, and it is why this is a leader
    pass instead: a cluster is only ever compared against the leaders, never
    against its neighbour, so a ramp keeps several representatives while a flat
    fill's near-identical shades still fold onto the first one.

    Clusters are visited largest first, so the leaders are the colours the
    picture actually uses; once ``limit`` leaders exist every later cluster
    joins its nearest one, which is what keeps a small-but-distinct region (the
    eyes) ahead of yet another shade of the background.
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
    """
    if threshold <= 0.0 or len(colours) < 2:
        return colours, classes

    labs = _color.lab_from_rgb(colours)
    parent = list(range(len(colours)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    merged = False
    for first in range(len(colours)):
        for second in range(first + 1, len(colours)):
            if _color.delta_e_2000(labs[first], labs[second]) >= threshold:
                continue
            root_a, root_b = find(first), find(second)
            if root_a != root_b:
                parent[root_b] = root_a
                merged = True
    if not merged:
        return colours, classes

    # ``-1`` marks a pixel outside the opacity mask.  Clipping it to 0 would
    # credit every transparent pixel to the first colour and could make the
    # wrong shade the survivor of a merge.
    flat = classes.reshape(-1)
    flat = flat[flat >= 0]
    counts = np.bincount(flat, minlength=len(colours))
    groups: dict[int, list[int]] = {}
    for index in range(len(colours)):
        groups.setdefault(find(index), []).append(index)
    survivors = sorted(
        max(members, key=lambda member: (int(counts[member]), -member))
        for members in groups.values()
    )
    remap = np.zeros(len(colours), dtype=np.int32)
    for new_index, survivor in enumerate(survivors):
        for member in groups[find(survivor)]:
            remap[member] = new_index
    # ``remap[-1]`` would silently dress the transparent pixels up as a real
    # class, so the mask has to survive the fold.
    return colours[survivors], np.where(
        classes >= 0, remap[np.clip(classes, 0, None)], -1
    )


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


def _renumber(
    kept: list[PaletteEntry], folded: np.ndarray
) -> tuple[list[PaletteEntry], np.ndarray, np.ndarray]:
    """Recount ``folded`` and renumber largest-first, dropping empty entries.

    ``folded`` holds compact indices into ``kept``.  Entries no pixel points at
    any more are dropped outright — keeping one would put an unused filament in
    the plate's slot list, which is the one thing the budget promises not to do.
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
    return [kept[int(live[int(position)])] for position in order], counts[order], final


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


#: How many different spools the 「耗材上限」 search is willing to choose between.
#: The regions the eye compares are the big ones and they are what asks for the
#: first rolls; the tail only multiplies the search.
_BUDGET_ROLL_LIMIT = 22

#: Above this many subsets the exhaustive walk gives way to growing a set roll
#: by roll and then swapping, which is the same answer on everything measured.
_BUDGET_SUBSET_LIMIT = 20_000


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


#: What :func:`match_image` uses when the caller does not name a 方案, and the
#: objective behind the first entry :func:`match_options` offers.  The two agree
#: on purpose: the page shows 「重新匹配」 landing on the 总体最接近 plan.
_budget_objective: BudgetObjective = _objective_closest


def _fit_spool_budget(
    kept: list[PaletteEntry],
    indices: np.ndarray,
    budget: int,
    palette: Sequence[PaletteEntry] | None = None,
    counts: Sequence[int] | np.ndarray | None = None,
    objective: BudgetObjective = _objective_closest,
) -> tuple[list[PaletteEntry], np.ndarray, np.ndarray] | None:
    """Re-pick ``kept`` so the plan never asks for more than ``budget`` spools.

    This is a choice, not a walk.  Taking the colours largest region first and
    refusing every one that needs a spool the budget has already spent is the
    greedy reading of 「最多用 N 卷料」, and it is measurably wrong: on a real
    41-spool library it scored weighted ΔE00 1.813 where the best four rolls the
    picture asked for score 1.249 — 31% worse — and the roll that prints a
    549-pixel nose sat one combination away from the answer it printed.  So the
    rolls the picture actually asks for are scored as subsets and the best one
    is printed.  ``kept`` is expected largest-first with ``counts`` the pixels
    behind each entry.  Returns ``None`` when nothing has to be dropped, so the
    caller keeps the untouched result.
    """
    if budget <= 0 or len(kept) < 2:
        return None

    pool = list(kept)
    labs = [np.asarray(entry.lab, dtype=np.float64) for entry in pool]
    needs = [set(_spool_ids(entry)) for entry in pool]
    if counts is not None and len(counts) == len(pool):
        weights = np.asarray(counts, dtype=np.float64)
    else:
        weights = np.array(
            [int(np.count_nonzero(indices == position)) for position in range(len(pool))],
            dtype=np.float64,
        )

    if len(set().union(*needs)) <= budget:
        return None  # the picture already fits; there is nothing to decide

    # The rolls worth choosing between, in the order the picture asks for them,
    # so a cap can only ever cut the tail of the picture's own wish list.
    rolls: list[str] = []
    for need in needs:
        for spool in sorted(need):
            if spool not in rolls and len(rolls) < _BUDGET_ROLL_LIMIT:
                rolls.append(spool)
    if not rolls:
        return None

    matrix: np.ndarray | None = None

    def _table() -> np.ndarray:
        nonlocal matrix
        if matrix is None:
            matrix = np.empty((len(pool), len(pool)), dtype=np.float64)
            for position in range(len(pool)):
                matrix[position] = [
                    _color.delta_e_2000(labs[position], other) for other in labs
                ]
        return matrix

    def _score(choice: Sequence[str]) -> tuple[tuple[float, ...], list[int]]:
        """How the picture reads under ``choice``, and what fits inside it."""
        subset = set(choice)
        printable = [position for position, need in enumerate(needs) if need <= subset]
        if not printable:
            return (float("inf"),), []
        if len(printable) == len(pool):
            return (0.0, 0.0), printable
        worst = np.zeros(len(pool), dtype=np.float64)
        surplus = np.ones(len(pool), dtype=bool)
        surplus[printable] = False
        columns = _table()[:, printable]
        worst[surplus] = columns[surplus].min(axis=1)
        return objective(worst, weights), printable

    size = min(budget, len(rolls))
    best_score: tuple[float, ...] = (float("inf"),)
    best_printable: list[int] = []
    chosen: list[str] = []
    if comb(len(rolls), size) <= _BUDGET_SUBSET_LIMIT:
        for candidate in combinations(rolls, size):
            score, printable = _score(candidate)
            if score < best_score:
                best_score, chosen, best_printable = score, list(candidate), printable
    else:
        # Too many combinations to walk: grow the set by whichever roll helps
        # most, then swap rolls in and out until nothing improves.
        while len(chosen) < size:
            gain: tuple[tuple[float, ...], str, list[int]] | None = None
            for spool in rolls:
                if spool in chosen:
                    continue
                score, printable = _score([*chosen, spool])
                if gain is None or score < gain[0]:
                    gain = (score, spool, printable)
            if gain is None:
                break
            chosen = [*chosen, gain[1]]
            best_score, best_printable = gain[0], gain[2]
        improved = True
        while improved:
            improved = False
            for position in range(len(chosen)):
                for spool in rolls:
                    if spool in chosen:
                        continue
                    trial = list(chosen)
                    trial[position] = spool
                    score, printable = _score(trial)
                    if score < best_score:
                        chosen, best_score, best_printable = trial, score, printable
                        improved = True

    if best_printable:
        printable = best_printable
        if len(printable) == len(pool):
            return None
    else:
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
            labs.append(np.asarray(fallback.lab, dtype=np.float64))
            printable = [len(pool) - 1]

    # Everything the search refused is repainted with the surviving colour it
    # is closest to; ``remap`` holds positions within ``printable``, so the
    # entries are re-expressed before renumbering, which indexes the list it is
    # handed.
    ranks = {position: rank for rank, position in enumerate(printable)}
    remap = np.full(len(pool), -1, dtype=_INDEX_DTYPE)
    for position in range(len(pool)):
        if position in ranks:
            remap[position] = ranks[position]
            continue
        remap[position] = min(
            range(len(printable)),
            key=lambda rank: _color.delta_e_2000(labs[position], labs[printable[rank]]),
        )

    folded = np.where(indices >= 0, remap[np.clip(indices, 0, None)], -1).astype(_INDEX_DTYPE)
    return _renumber([pool[position] for position in printable], folded)


@dataclass
class _Prepared:
    """Everything about a match that does not depend on which rolls get printed.

    Splitting the pipeline here is what makes offering several 耗材方案
    affordable: reading and reducing the picture, folding near-identical colours
    together and the exact palette search are the slow steps, and they are the
    same whichever rolls the user ends up printing.  Only the budget fit and the
    per-region means are redone per 方案.
    """

    image: Image.Image
    palette: Sequence[PaletteEntry]
    kept: list[PaletteEntry]
    counts: np.ndarray
    indices: np.ndarray
    config: MatchSettings
    name: str


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
        mask = np.ones(alpha.shape, dtype=bool)

    _report(progress, "减少图片颜色数量", 0.30)
    colours, classes = reduce_colours(image, config, mask)
    colours, classes = _merge_similar_colours(colours, classes, config.merge_delta_e)

    _report(progress, "在耗材与混色库里寻找最接近的颜色", 0.35)
    labs = _color.lab_from_rgb(colours)

    def _search_rows(rows_done: int, rows_total: int) -> None:
        # The exact pass is the long one: keep the bar moving through it.
        _report(
            progress,
            "在耗材与混色库里寻找最接近的颜色",
            0.35 + 0.35 * rows_done / max(1, rows_total),
        )

    chosen = _nearest_entries(labs, palette, _palette_table(palette), _search_rows)

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


def _plan(
    prepared: _Prepared,
    max_spools: int,
    objective: BudgetObjective,
    progress: ProgressFn | None = None,
) -> MatchResult:
    """Print ``prepared`` with at most ``max_spools`` rolls, ranked by ``objective``.

    「最多用 N 卷料」 comes after the merge: folding two invisible shades together
    frees spools for free, so the budget should only have to work on colours that
    actually survived.
    """
    kept, ordered_counts, final = prepared.kept, prepared.counts, prepared.indices
    _report(progress, "按「耗材上限」合并颜色", 0.90)
    budgeted = _fit_spool_budget(
        kept, final, max_spools, prepared.palette, ordered_counts, objective
    )
    if budgeted is not None:
        kept, ordered_counts, final = budgeted
    means = None
    if len(kept):
        _report(progress, "统计每种颜色的像素", 0.96)
        pixels = np.asarray(prepared.image.convert("RGB"), dtype=np.float64).reshape(-1, 3)
        flat = final.reshape(-1)
        valid = flat >= 0
        totals = np.zeros((len(kept), 3), dtype=np.float64)
        np.add.at(totals, flat[valid], pixels[valid])
        totals /= np.maximum(ordered_counts, 1)[:, None]
        means = np.rint(totals).astype(np.uint8)
    # The picture's own average colour per region, so the UI can show
    # 「图片 #xxxxxx → 匹配 #yyyyyy」 and the user can judge the match.
    return MatchResult(
        width=int(final.shape[1]),
        height=int(final.shape[0]),
        palette=kept,
        indices=final,
        counts=ordered_counts,
        settings=prepared.config,
        source=prepared.name,
        means=means,
    )


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


def match_fidelity_report(result: MatchResult) -> tuple[float, float, int]:
    """（平均 ΔE00，最差 ΔE00，差块个数）of a plan.

    The third number counts the colours off by more than
    :data:`_BAD_BLOCK_DELTA_E`, which is the 「混不出来」 the eye actually
    notices: on the user's pig every four-roll 方案 keeps the average near 1.5,
    and they differ in whether the nose is one of the two or three blocks that
    are visibly off.  An average alone hides that choice.
    """
    if result.means is None or not len(result.palette):
        return 0.0, 0.0, 0
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
    weights = np.asarray(result.counts, dtype=np.float64) / total
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


#: The 耗材方案 the page offers, in the order the user sees them.  ``key`` is what
#: a project file remembers about the chosen one.  Every one of them fits inside
#: 「耗材上限」: that cap is the user's own instruction about how many rolls this
#: print may load, so a plan that breaks it is not an option to offer, it is a
#: different setting.  The plans differ in *which* colours they sacrifice — on the
#: user's pig 「总体最接近」 keeps the average lowest and 「小色块优先」 costs average
#: to print the nose — and that is a choice about the picture, not a computable
#: fact, so the user makes it.
_PLAN_SPECS: tuple[tuple[str, str, str, BudgetObjective], ...] = (
    (
        "closest",
        "总体最接近",
        "整张图算下来最像；个别小色块可能差一点。",
        _objective_closest,
    ),
    (
        "worst",
        "最不像的色块优先",
        "先把差得最明显的那一块拉回来，平均会稍差。",
        _objective_worst,
    ),
    (
        "distinct",
        "小色块优先",
        "眼睛、鼻子这类小色块更准；超过 ΔE00 5 的色块最少。",
        _objective_distinct,
    ),
)


def plan_caption(key: str) -> tuple[str, str]:
    """（名称，说明）for a 方案 key, 「工程里存的那一套」 for anything else.

    「打开工程」 hands the page the plan the file was saved with, and that key can
    name a 方案 this build no longer offers (an older file, a hand-edited one).
    The row still has to say something true, so it says where it came from.
    """
    for spec_key, title, note, _objective in _PLAN_SPECS:
        if spec_key == key:
            return title, note
    return "工程里存的那一套", "保存工程时用的搭配；点「重新匹配」可以再挑一套。"


def match_options(
    source: str | Path | Image.Image,
    palette: Sequence[PaletteEntry],
    settings: MatchSettings | None = None,
    *,
    progress: ProgressFn | None = None,
) -> list[BudgetOption]:
    """Every 耗材方案 worth showing for ``source``, best first and deduplicated.

    The program does not pretend there is one right answer.  On the user's own
    pig, four rolls print the nose pale whichever way the search is pointed —
    one 方案 gets the average down, another spends a little average to print the
    nose exactly — and that is a choice about the picture, so it is offered
    rather than assumed.  Two candidates that end up printing the same rolls are
    the same 方案 and only the first is kept, so a picture that fits inside
    「耗材上限」 with room to spare is offered one option, and every option obeys
    the cap.
    """
    prepared = _prepare(source, palette, settings, progress)
    cap = prepared.config.max_spools
    total = len(_PLAN_SPECS)
    options: list[BudgetOption] = []
    seen: set[frozenset[str]] = set()
    for position, (key, title, note, objective) in enumerate(_PLAN_SPECS):
        _report(progress, f"试着搭配耗材（{title}）", 0.88 + 0.1 * position / total)
        result = _plan(prepared, cap, objective)
        rolls = frozenset(spool for entry in result.palette for spool in _spool_ids(entry))
        if rolls in seen:
            continue
        seen.add(rolls)
        options.append(BudgetOption(key=key, title=title, note=note, result=result))
    _report(progress, "完成", 1.0)
    return options

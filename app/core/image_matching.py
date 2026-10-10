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


def _report(progress: ProgressFn | None, message: str, fraction: float) -> None:
    if progress is not None:
        progress(message, fraction)


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PaletteEntry:
    """One printable colour: a spool on its own, or a two-spool recipe."""

    key: str
    color_hex: str
    rgb: tuple[int, int, int]
    lab: np.ndarray
    label: str
    kind: str  # "filament" | "mix"
    filament_ids: tuple[str, ...]
    recipe: MixRecipe | None = None
    filament: Filament | None = None

    @property
    def is_mix(self) -> bool:
        return self.kind == "mix"

    @property
    def ratio_text(self) -> str:
        return self.recipe.ratio_text if self.recipe is not None else ""

    @property
    def short_label(self) -> str:
        if self.filament is not None:
            return self.filament.display_name
        if self.recipe is not None:
            return f"{self.recipe.percent_a}% + {self.recipe.percent_b}%"
        return self.color_hex


def build_palette(
    library: FilamentLibrary,
    catalog: MixCatalog | None = None,
    *,
    include_mixes: bool = True,
) -> list[PaletteEntry]:
    """Every colour the user can print: each spool, then every two-spool mix.

    Spools come first so that, when a mix lands exactly on a spool colour, the
    cheaper answer (one spool) wins the tie.
    """
    entries: list[PaletteEntry] = []
    for filament in library:
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
        for recipe in catalog.recipes:
            a = library.get(recipe.a_id)
            b = library.get(recipe.b_id)
            if a is None or b is None:
                continue
            entries.append(
                PaletteEntry(
                    key=f"mix:{recipe.key}",
                    color_hex=recipe.color_hex,
                    rgb=tuple(int(v) for v in recipe.rgb),
                    lab=np.asarray(recipe.lab, dtype=np.float64),
                    label=f"{a.display_name} {recipe.percent_a}% + {b.display_name} {recipe.percent_b}%",
                    kind="mix",
                    filament_ids=(recipe.a_id, recipe.b_id),
                    recipe=recipe,
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
    """Order 「全部颜色」 by the same keys the 混色配方 grid offers.

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


def _nearest_entries(labs: np.ndarray, entries: Sequence[PaletteEntry]) -> np.ndarray:
    """Index of the palette entry closest to each row of ``labs``.

    A Euclidean pass in Lab narrows the field, then CIEDE2000 picks the winner so
    the answer agrees with the ΔE the UI reports.
    """
    if not entries:
        return np.zeros(len(labs), dtype=np.int64)
    table = np.stack([entry.lab for entry in entries])
    out = np.zeros(len(labs), dtype=np.int64)
    for row, lab in enumerate(labs):
        distances = np.linalg.norm(table - lab, axis=1)
        if distances.size > _SHORTLIST:
            candidates = np.argpartition(distances, _SHORTLIST)[:_SHORTLIST]
        else:
            candidates = np.arange(distances.size)
        best_index = int(candidates[0])
        best = float("inf")
        for index in candidates:
            value = _color.delta_e_2000(lab, table[index])
            if value < best:
                best = value
                best_index = int(index)
        out[row] = best_index
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
    return _fold_clusters(colours, classes, settings.max_colours, settings.merge_delta_e)


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


def match_image(
    source: str | Path | Image.Image,
    palette: Sequence[PaletteEntry],
    settings: MatchSettings | None = None,
    *,
    progress: ProgressFn | None = None,
) -> MatchResult:
    """Reduce ``source`` to printable colours drawn from ``palette``."""
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

    _report(progress, "在耗材与混色库里寻找最接近的颜色", 0.55)
    labs = _color.lab_from_rgb(colours)
    chosen = _nearest_entries(labs, palette)

    _report(progress, "按像素分配颜色", 0.80)
    indices = np.where(mask, chosen[classes], -1).astype(_INDEX_DTYPE)

    # Drop palette entries the picture never used, then renumber largest first.
    used = sorted(int(value) for value in np.unique(indices) if value >= 0)
    counts = np.array([int(np.count_nonzero(indices == value)) for value in used], dtype=np.int64)
    order = np.argsort(-counts, kind="stable")
    kept = [list(palette)[used[int(position)]] for position in order]
    remap = np.full(len(palette), -1, dtype=_INDEX_DTYPE)
    for new_index, position in enumerate(order):
        remap[used[int(position)]] = new_index
    final = np.where(indices >= 0, remap[np.clip(indices, 0, None)], -1).astype(_INDEX_DTYPE)

    # The picture's own average colour per region, so the UI can show
    # 「图片 #xxxxxx → 匹配 #yyyyyy」 and the user can judge the match.
    means = None
    ordered_counts = counts[order]
    kept, ordered_counts, final = _merge_indistinguishable(
        kept, ordered_counts, final, config.merge_delta_e
    )
    if len(kept):
        pixels = np.asarray(image.convert("RGB"), dtype=np.float64).reshape(-1, 3)
        flat = final.reshape(-1)
        valid = flat >= 0
        totals = np.zeros((len(kept), 3), dtype=np.float64)
        np.add.at(totals, flat[valid], pixels[valid])
        totals /= np.maximum(ordered_counts, 1)[:, None]
        means = np.rint(totals).astype(np.uint8)

    _report(progress, "完成", 1.0)
    return MatchResult(
        width=int(final.shape[1]),
        height=int(final.shape[0]),
        palette=kept,
        indices=final,
        counts=ordered_counts,
        settings=config,
        source=name,
        means=means,
    )

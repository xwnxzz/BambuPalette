"""Turn a matched picture into a flat, printable, multi-material plate.

The user's requirement is explicit: the export must be a *flat* print, not a
relief.  Two thicknesses are therefore all we need —
* the slab is ``base_thickness_mm`` thick in the base colour, so the object is a
  solid with exactly the picture's outline;
* the top ``colour_thickness_mm`` of that same slab carries the picture's colours,
  one filament per colour region.

The slab is built as **one watertight solid**, not as one closed prism stacked on
another.  Stacking looks equivalent and is not: the colour prisms' undersides sit
exactly on the base's top face and their side walls touch each other, so the
assembled mesh is full of interior walls and coincident faces.  A slicer unions
those away silently, but anything that *counts* them — Bambu Studio's mesh
statistics, for one — reports thousands of non-manifold edges, and the model is
not actually a manifold solid.

Geometry is built on the pixel grid and compressed into maximal rectangles, so a
512×512 picture becomes thousands of quads rather than hundreds of thousands.
T-junctions left by that compression are repaired exactly, in one pass over the
whole solid, because a per-part repair cannot see the edge its neighbour owns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

import numpy as np

from ..core.image_matching import MatchResult, PaletteEntry, ProgressFn

__all__ = [
    "PlatePart",
    "PlateModel",
    "PlateSettings",
    "rectangles",
    "run_rectangles",
    "smooth_factor",
    "smooth_indices",
    "smooth_lattice",
    "prism_mesh",
    "stitch_t_junctions",
    "merge_parts",
    "plate_mesh",
    "edge_health",
    "signed_volume",
    "build_plate",
]

#: Tag used for the faces that print in the base colour, distinct from every
#: palette index (which are ``>= 0``) because the base colour is normally *also*
#: one of the palette entries and the two must not be confused.
BASE_TAG = -1

Point = tuple[float, float, float]


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
#: 一格的宽度：0.4 mm 的喷嘴铺一层就是一格。 1.6.42 起板的长宽由它推出
#: （图片像素数 × 这一格），不再问「打印宽度」；也不再按 250 mm 的打印范围自动缩，
#: 用户明确要求「不限制大小，允许超出」，超出多少由切片器去说。
CELL_MM = 0.4
#: 图那一层颜色占多厚。 混色是 Bambu Studio 自己的事（它按 ``filament_mixed_sublayer_ratios``
#: 在层内排），我们只要留出够放住那些颜色的一段，所以这一项也不再暴露给界面。
COLOUR_BAND_MM = 0.4


@dataclass(frozen=True)
class PlateSettings:
    """Everything the user can change about the exported slab.

    1.6.42 起界面只问两件事：**底板厚度**和图片印在哪一面。 以前那两个输入框 ——
    「打印宽度」和「颜色层厚度」—— 都不再是用户的活：

    * 一格打多宽由喷嘴定（:data:`CELL_MM`），所以板的长宽跟着图片像素数走。
    * 颜色层厚度由 Bambu Studio 自己管，这里固定成 :data:`COLOUR_BAND_MM`；
      于是 ``base_thickness_mm`` 变成**整块**的厚度（含那一层颜色）。
    """

    #: 一格多少毫米。 0.4 = 一个 0.4 mm 喷嘴刚好铺满一格，是全套假设里唯一和机器有关的一条。
    cell_mm: float = CELL_MM
    #: 整块底板的厚度，含图那一层颜色（1.6.42 之前这里是「底板色那段」的厚度）。
    base_thickness_mm: float = 1.4
    #: 图那一层颜色占多厚。 界面上没有这个输入，留着是为了工具和测试能改它看重叠行为。
    colour_thickness_mm: float = COLOUR_BAND_MM
    #: Palette index whose filament also prints the base plate; -1 = the colour
    #: that covers the most pixels.
    base_index: int = -1
    #: 导出时把色块边界圆滑 (1.6.22 / 1.6.27)。 匹配栅格上的色块边界是「一格一格的台阶」，
    #: 150 mm 宽、512 像素的图上约 0.29 mm 一级，在 Bambu Studio 里放大看就是锯齿。
    #: 打开之后先在匹配栅格上把每种颜色糊一遍再细分若干倍：1.6.27 起直接描出那条边界线
    #: （斜的边是直弦、弯的边是折线，一步约 0.07 mm，见 smooth_lattice），描不动时才退回
    #: 按细格重新分类。 界面上的勾默认打着。
    smooth_edges: bool = True
    #: 图片印在底面还是顶面 (1.6.23)。 默认底面：图片贴着打印板，顶面是一整片底板色，
    #: 打出来顶面平整、不被色块的边线划过。 打开工程时这一项随工程走。
    image_on_bottom: bool = True

    def clamped(self) -> "PlateSettings":
        colour = max(0.2, min(20.0, float(self.colour_thickness_mm)))
        return PlateSettings(
            cell_mm=max(0.1, min(5.0, float(self.cell_mm))),
            # 整块厚度：至少要给底板色那段留下 0.2 mm，否则那一层会被颜色层吃掉。
            base_thickness_mm=max(colour + 0.2, min(20.0, float(self.base_thickness_mm))),
            colour_thickness_mm=colour,
            base_index=int(self.base_index),
            smooth_edges=bool(self.smooth_edges),
            image_on_bottom=bool(self.image_on_bottom),
        )


def settings_for_width(width_mm: float, pixels: int, **kwargs) -> PlateSettings:
    """按「整块要多宽」造一份设置：``cell_mm = width_mm / pixels``。

    1.6.42 起界面上只有「底板厚度」一个尺寸输入，板的一格就是 :data:`CELL_MM`
    （图片像素数 × 0.4 mm）。 这个帮手留给需要指定整块宽度的场合：样例导出、
    几何测试，以及把老工程里记的「打印宽度」换算过来看。
    """
    return PlateSettings(cell_mm=float(width_mm) / max(1, int(pixels)), **kwargs)


#: 圆滑色块时把匹配栅格放大几倍，最多这么多。 4 倍把 0.29 mm 的台阶变成 0.07 mm，
#: 比 0.4 mm 的喷嘴细一档；再细只是把导出时间拉长，打印上区分不出来。
SMOOTH_FACTOR = 4#: 细栅格的像元上限。 一张 512×512 的图放大 4 倍是 419 万个像元，正是这一步的开销
#: 所在（每种颜色都要在细栅格上做一次双线性取样），比这更大就少放几倍。
SMOOTH_MAX_CELLS = 4_500_000
#: 「细格像元 × 颜色数」的预算：颜色越多，每种颜色那一趟也越贵。 32 色的图在 512²
#: 下退到 2 倍（0.15 mm 一格，还是比喷嘴细），导出不至于从 3 秒涨到 9 秒。
SMOOTH_SLOT_BUDGET = 48_000_000
#: 但至少放 2 倍：这是「圆滑」和「不圆滑」的分界，除非图本身已经大得放不动了。
SMOOTH_ALWAYS_CELLS = 16_000_000
#: 放大前的高斯 σ，单位是匹配栅格的像元。 它决定边界的「圆」到什么程度，也决定多细
#: 的东西会被磨掉，所以是量出来的：一块 64×64 的色块里放一条 1 像元宽的线，细分 4 倍
#: 后 σ=0.4/0.5 留下 99%，σ=0.6 只剩 49%，σ=0.8 直接没了（2 像元宽以上各档都还在）。
#: 所以取 0.5：比一像元细的胡须，描边照样印得出来，台阶的直角仍旧被磨圆。
SMOOTH_SIGMA = 0.5


@dataclass
class PlatePart:
    """One filament's share of the slab.

    A part carries its own copy of every corner it uses, including the corners it
    shares with a neighbour, because the builder stitches each region's outline
    independently.  The copies sit at bit-identical positions, so
    :func:`merge_parts` folds them back into the single solid they were cut from.
    Exporters must go through that merge: splicing the parts together verbatim
    hands every part-border edge two different index pairs, which is a mesh the
    slicer reads as open even though no geometry is missing.
    """

    name: str
    color_hex: str
    palette_index: int
    extruder: int
    vertices: np.ndarray  # (V, 3) float64, millimetres
    triangles: np.ndarray  # (T, 3) int64
    region_pixels: int
    z_bottom_mm: float
    z_top_mm: float

    @property
    def triangle_count(self) -> int:
        return int(len(self.triangles))


@dataclass
class PlateModel:
    """A flat plate ready to be written as 3MF or OBJ."""

    parts: list[PlatePart]
    width_mm: float
    depth_mm: float
    pixel_size_mm: float
    base_thickness_mm: float
    colour_thickness_mm: float
    notes: list[str] = field(default_factory=list)
    source: str = ""

    @property
    def total_thickness_mm(self) -> float:
        #: 1.6.42 起 ``base_thickness_mm`` 记的就是整块厚度（颜色层算在里面）。
        return self.base_thickness_mm

    @property
    def extruder_count(self) -> int:
        return max((part.extruder for part in self.parts), default=0)

    @property
    def triangle_count(self) -> int:
        return sum(part.triangle_count for part in self.parts)

    @property
    def vertex_count(self) -> int:
        """Vertices the parts carry between them, before coincident ones merge."""
        return sum(int(len(part.vertices)) for part in self.parts)

    @property
    def shared_vertex_count(self) -> int:
        """Vertices the written mesh actually carries.

        Border corners are shared, so this is smaller than :attr:`vertex_count`
        and it is the number a reader of the exported file would count.
        """
        if not self.parts:
            return 0
        return int(len(merge_parts(self.parts)[0]))

    @property
    def volume_mm3(self) -> float:
        cell = self.pixel_size_mm * self.pixel_size_mm
        total = 0.0
        for part in self.parts:
            total += part.region_pixels * cell * (part.z_top_mm - part.z_bottom_mm)
        return total

    def stats(self) -> dict:
        return {
            "parts": len(self.parts),
            "extruders": self.extruder_count,
            "triangles": self.triangle_count,
            "vertices": self.shared_vertex_count,
            "width_mm": round(self.width_mm, 3),
            "depth_mm": round(self.depth_mm, 3),
            "thickness_mm": round(self.total_thickness_mm, 3),
            "pixel_mm": round(self.pixel_size_mm, 4),
            "volume_mm3": round(self.volume_mm3, 2),
        }


# ---------------------------------------------------------------------------
# Raster helpers
# ---------------------------------------------------------------------------
def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Half-open ``(start, stop)`` runs of ``True`` in a 1-D boolean array."""
    if not flags.any():
        return []
    padded = np.concatenate(([False], flags.astype(bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(edges[0::2], edges[1::2])]


def rectangles(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Decompose a boolean mask into maximal rectangles ``(x, y, w, h)``.

    Greedy row-major expansion: run right while the row keeps the colour, then
    pull the whole run down while every covered row matches exactly.  The result
    tiles the mask with no overlap and no gaps.
    """
    if mask.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    height, width = mask.shape
    visited = np.zeros((height, width), dtype=bool)
    out: list[tuple[int, int, int, int]] = []
    for y in range(height):
        row = mask[y]
        if not row.any():
            continue
        x = 0
        while x < width:
            if not row[x] or visited[y, x]:
                x += 1
                continue
            x_end = x
            while x_end + 1 < width and row[x_end + 1] and not visited[y, x_end + 1]:
                x_end += 1
            y_end = y
            while y_end + 1 < height:
                segment = mask[y_end + 1, x : x_end + 1]
                if not segment.all() or visited[y_end + 1, x : x_end + 1].any():
                    break
                y_end += 1
            visited[y : y_end + 1, x : x_end + 1] = True
            out.append((x, y, x_end - x + 1, y_end - y + 1))
            x = x_end + 1
    return out


def run_rectangles(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Tile a mask with rectangles found from vectorised row runs.

    Same contract as :func:`rectangles` — the rectangles cover the mask exactly
    once, no overlap and no gaps — but the runs of every row come out of one
    numpy pass and are then pulled down while the same span keeps repeating.  The
    greedy walk costs a Python step per *cell*, which is fine on the match grid
    and far too slow on the refined grid of :func:`smooth_indices` (a 512×512
    picture refined 4× has four million cells).
    """
    occupied = np.asarray(mask, dtype=bool)
    if occupied.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    height, width = occupied.shape
    if height == 0 or width == 0 or not occupied.any():
        return []

    # Pad with a false border: then every run of the row is a false→true
    # transition followed by a true→false one, in row-major order.
    padded = np.zeros((height, width + 2), dtype=bool)
    padded[:, 1:-1] = occupied
    changes = np.flatnonzero(padded[:, 1:] != padded[:, :-1])
    rows, columns = np.divmod(changes, width + 1)
    starts = columns[0::2]
    stops = columns[1::2]
    row_of = rows[0::2]

    out: list[tuple[int, int, int, int]] = []
    running: dict[tuple[int, int], tuple[int, int]] = {}
    for y, a, b in zip(row_of.tolist(), starts.tolist(), stops.tolist()):
        span = (a, b)
        previous = running.get(span)
        if previous is not None and previous[1] == y - 1:
            running[span] = (previous[0], y)
            continue
        if previous is not None:
            out.append((span[0], previous[0], span[1] - span[0], previous[1] - previous[0] + 1))
            del running[span]
        running[span] = (y, y)
    for span, (y_start, y_stop) in running.items():
        out.append((span[0], y_start, span[1] - span[0], y_stop - y_start + 1))
    out.sort(key=lambda item: (item[1], item[0]))
    return out


def _merge_stacked(rects: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """Fuse rectangles that share an x-range and touch vertically.

    The greedy pass can leave a column split into several rectangles when a
    neighbouring region changes; fusing them removes T-junctions and triangles
    for free.
    """
    by_span: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for x, y, w, h in rects:
        by_span.setdefault((x, w), []).append((y, h))
    merged: list[tuple[int, int, int, int]] = []
    for (x, w), spans in by_span.items():
        spans.sort()
        y_start, y_stop = spans[0][0], spans[0][0] + spans[0][1]
        for y, h in spans[1:]:
            if y == y_stop:
                y_stop = y + h
            else:
                merged.append((x, y_start, w, y_stop - y_start))
                y_start, y_stop = y, y + h
        merged.append((x, y_start, w, y_stop - y_start))
    merged.sort(key=lambda item: (item[1], item[0]))
    return merged


# ---------------------------------------------------------------------------
# Round edges (1.6.22)
# ---------------------------------------------------------------------------
def smooth_factor(height: int, width: int, colours: int = 1) -> int:
    """How many times to refine the match grid before re-classifying it.

    :data:`SMOOTH_FACTOR` unless the refined grid would pass one of two budgets:
    the absolute size cap :data:`SMOOTH_MAX_CELLS`, or the per-picture
    :data:`SMOOTH_SLOT_BUDGET` shared out over the colours (the field sampling
    costs one pass per colour).  Even then it keeps at least two steps, which is
    what makes the difference between a staircase and a border, unless the match
    grid itself is already past :data:`SMOOTH_ALWAYS_CELLS`.
    """
    cells = max(1, int(height)) * max(1, int(width))
    allowed = min(SMOOTH_MAX_CELLS, SMOOTH_SLOT_BUDGET // max(1, int(colours)))
    allowed = max(allowed, min(cells * 2 * 2, SMOOTH_ALWAYS_CELLS))
    steps = SMOOTH_FACTOR
    while steps > 1 and cells * steps * steps > allowed:
        steps -= 1
    return max(1, steps)


def _gaussian_kernel(sigma: float) -> np.ndarray:
    """Normalised 1-D Gaussian, cut at three standard deviations."""
    radius = max(1, int(math.ceil(3.0 * float(sigma))))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    weights = np.exp(-(offsets * offsets) / (2.0 * float(sigma) ** 2))
    return weights / weights.sum()


def _blur_separable(field: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Blur a 2-D float field, repeating the outermost row and column."""
    radius = len(kernel) // 2
    height, width = field.shape
    spread = np.pad(field, ((radius, radius), (0, 0)), mode="edge")
    rows = np.zeros_like(field)
    for index, weight in enumerate(kernel):
        rows += weight * spread[index : index + height]
    spread = np.pad(rows, ((0, 0), (radius, radius)), mode="edge")
    out = np.zeros_like(field)
    for index, weight in enumerate(kernel):
        out += weight * spread[:, index : index + width]
    return out


def _sample_axis(
    centres: np.ndarray, size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lower/upper index and blend weight for sampling a coarse axis."""
    clamped = np.clip(centres, 0.0, float(size - 1))
    lower = np.floor(clamped).astype(np.intp)
    upper = np.minimum(lower + 1, size - 1)
    return lower, upper, (clamped - lower).astype(np.float32)


def smooth_indices(
    indices: np.ndarray,
    *,
    factor: int | None = None,
    sigma: float = SMOOTH_SIGMA,
    progress: ProgressFn | None = None,
) -> np.ndarray:
    """Re-classify a match grid on a finer lattice so block borders come out round.

    Every block border is a curve sampled on the match grid, so a border that
    runs at an angle is a one-cell staircase — 0.293 mm of step on a 512-pixel
    picture printed 150 mm wide, which is a plainly visible saw in the slicer.

    Smoothing the *polygons* of the two blocks that share such a border cannot
    work: the slab has to stay one closed solid, and two independently smoothed
    outlines meet at vertices that land on different lattice cells, which opens
    the seam (that is what produced the 14,650 open edges of 1.6.5).  Refining
    the label raster instead cannot open anything, because every refined cell
    belongs to exactly one colour:

    * each colour becomes an indicator field on the match grid, blurred with a
      small Gaussian (``mode="edge"``, so the border of the picture repeats);
    * each field is sampled bilinearly at the centres of the refined lattice
      (``(index + 0.5) / factor - 0.5`` in match-grid coordinates);
    * every refined cell takes the colour with the strongest field.

    The border therefore crosses *between* refined cells instead of along the
    match grid's cell lines, and the step size drops to one refined cell
    (a quarter of a match pixel, ≈0.07 mm on a 150 mm plate), which is finer
    than the nozzle.  Transparent picture
    pixels are one more "colour" (slot 0) and stay ``-1``.
    """
    labels = np.asarray(indices)
    if labels.ndim != 2:
        raise ValueError("indices must be two-dimensional")
    height, width = (int(size) for size in labels.shape)
    steps = max(1, int(factor)) if factor else smooth_factor(height, width)
    if steps <= 1 or height == 0 or width == 0:
        return labels.astype(np.int32, copy=True)

    # Slot 0 is the transparent/背景 area, palette colour ``i`` is slot ``i + 1``.
    slots = np.where(labels < 0, 0, labels + 1).astype(np.int32)
    kernel = _gaussian_kernel(sigma)
    fine_height, fine_width = height * steps, width * steps
    rows_low, rows_high, row_weight = _sample_axis(
        (np.arange(fine_height, dtype=np.float32) + 0.5) / steps - 0.5, height
    )
    cols_low, cols_high, col_weight = _sample_axis(
        (np.arange(fine_width, dtype=np.float32) + 0.5) / steps - 0.5, width
    )
    row_weight = row_weight[:, None]

    # Every field is non-negative and they sum to one everywhere, so -1 is
    # strictly below all of them and the first slot of a tie keeps the cell.
    best = np.full((fine_height, fine_width), -1.0, dtype=np.float32)
    chosen = np.zeros((fine_height, fine_width), dtype=np.int32)
    present = np.unique(slots)
    for position, slot in enumerate(present):
        if progress is not None:
            progress(
                f"圆滑色块边界 {position + 1}/{len(present)}",
                (position + 1) / len(present),
            )
        field = _blur_separable((slots == slot).astype(np.float32), kernel)
        rows_a = np.take(field, rows_low, axis=0)
        rows_b = np.take(field, rows_high, axis=0)
        chosen_slot = np.take(rows_a, cols_low, axis=1)
        blended = np.take(rows_a, cols_high, axis=1)
        np.multiply(chosen_slot, 1.0 - col_weight, out=chosen_slot)
        np.multiply(blended, col_weight, out=blended)
        chosen_slot += blended
        np.multiply(chosen_slot, 1.0 - row_weight, out=chosen_slot)
        blended = np.take(rows_b, cols_low, axis=1)
        tail = np.take(rows_b, cols_high, axis=1)
        np.multiply(blended, 1.0 - col_weight, out=blended)
        np.multiply(tail, col_weight, out=tail)
        blended += tail
        np.multiply(blended, row_weight, out=blended)
        chosen_slot += blended
        better = chosen_slot > best
        if not better.any():
            continue
        best[better] = chosen_slot[better]
        chosen[better] = int(slot)

    return np.where(chosen == 0, np.int32(-1), chosen - 1).astype(np.int32)


# ---------------------------------------------------------------------------
# Tracing the border (1.6.27)
# ---------------------------------------------------------------------------
#: Lattice coordinates are integers and crossings sit between them, so a
#: difference below this is "the same point".
_TRACE_EPS = 1e-9

#: A border that runs exactly along a lattice line makes the two blurred fields
#: equal there, and the sampled labels then differ from cell to cell by float
#: noise: the crossing lands on the lattice point itself and leaves a piece of
#: some 1e-8 cells of area.  Those bits are folded into their neighbour (see
#: :func:`_trace_square`), and a point this close to a lattice point *is* that
#: lattice point: the interpolation of a crossing that really sits on a line
#: comes out as much as a few thousandths of a cell away from it (measured:
#: 267.9999 instead of 268 on the selftest picture, 12.99866 instead of 13 on a
#: 16 px stress case), and that hair is enough to keep two "same" points apart in
#: the assembler.  A fiftieth of a cell is far below the resolution the matcher
#: works at, and twenty times below one fine cell, the thinnest real feature.
_TRACE_SNAP = 5e-3

#: Below this area a piece of a square is one of those lattice-line bits, not a
#: colour: a single fine cell is 0.0625 of a square, and the thinnest strip a
#: border can cut through one is some 0.03, so this is nearly two orders of
#: magnitude clear of anything a picture can actually draw.
_TRACE_MIN_AREA = 1e-3

#: Coordinates the traced geometry is emitted on.  A crossing near a lattice
#: point (a border that only just misses it) comes out of the field
#: interpolation a hair away from it, and a piece and the wall beside it then
#: carry two points a nanometre apart: the assembler welds by exact equality, so
#: "the same point" stays two vertices, the triangulator cuts a zero-area face
#: between them, and the slicer reports the plate as open.  Everything is
#: therefore put on this grid first — a hair below the assembler's own micrometre
#: and still three orders of magnitude below the thinnest real feature, a fine
#: cell (0.03 mm on a 512 px picture at four steps).
_TRACE_QUANTUM = 1e-5


def _snap_trace(value: float) -> float:
    """Put one traced coordinate on the lattice, or on :data:`_TRACE_QUANTUM`."""
    nearest = round(value)
    if abs(value - nearest) <= _TRACE_SNAP:
        return float(nearest)
    return round(value / _TRACE_QUANTUM) * _TRACE_QUANTUM


def _label_border(labels: np.ndarray) -> np.ndarray:
    """Match-grid cells that have a neighbour with another label.

    The picture's edge repeats (``mode="edge"``, the same as the blur), so a
    border that runs into the edge of the picture does not look like a change of
    label just because there is nothing outside it.
    """
    height, width = labels.shape
    padded = np.pad(labels, 1, mode="edge")
    out = np.zeros((height, width), dtype=bool)
    for dy in range(3):
        for dx in range(3):
            if dy == 1 and dx == 1:
                continue
            out |= padded[dy : dy + height, dx : dx + width] != labels
    return out


def _grow(mask: np.ndarray) -> np.ndarray:
    """Dilate a mask by one cell in all eight directions."""
    height, width = mask.shape
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    out = np.zeros((height, width), dtype=bool)
    for dy in range(3):
        for dx in range(3):
            out |= padded[dy : dy + height, dx : dx + width]
    return out


def _sample_lattice(
    fields: list[tuple[int, np.ndarray]],
    ring: np.ndarray,
    steps: int,
    coarse: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample every blurred field at the lattice points of the ring's cells.

    This is the cheap half of :func:`smooth_indices`: that one samples the whole
    refined grid (``colors × steps²`` passes over every match-grid cell), while a
    border only ever runs through the ring, so the sample count here is
    O(border × steps) and the fields are already blurred.
    """
    height, width = coarse.shape
    squares = np.repeat(np.repeat(ring, steps, axis=0), steps, axis=1)
    point = np.zeros((height * steps + 1, width * steps + 1), dtype=bool)
    point[:-1, :-1] |= squares
    point[:-1, 1:] |= squares
    point[1:, :-1] |= squares
    point[1:, 1:] |= squares
    rows, cols = np.nonzero(point)
    y = rows.astype(np.float32) / steps - 0.5
    x = cols.astype(np.float32) / steps - 0.5
    rows_low, rows_high, row_weight = _sample_axis(y, height)
    cols_low, cols_high, col_weight = _sample_axis(x, width)

    best = np.full(rows.shape, -1.0, dtype=np.float32)
    second = np.zeros(rows.shape, dtype=np.float32)
    chosen = np.zeros(rows.shape, dtype=np.int32)
    for slot, field in fields:
        value = (
            field[rows_low, cols_low] * (1.0 - row_weight) * (1.0 - col_weight)
            + field[rows_low, cols_high] * (1.0 - row_weight) * col_weight
            + field[rows_high, cols_low] * row_weight * (1.0 - col_weight)
            + field[rows_high, cols_high] * row_weight * col_weight
        )
        better = value > best
        second = np.where(better, np.maximum(best, 0.0), np.maximum(second, value))
        best = np.where(better, value, best)
        chosen[better] = slot

    # Away from the ring every point keeps the label of the match-grid cell it
    # falls in: the ring is wide enough that the border cannot reach its edge.
    row_cell = np.minimum(np.arange(height * steps + 1) // steps, height - 1)
    col_cell = np.minimum(np.arange(width * steps + 1) // steps, width - 1)
    corners = coarse[np.ix_(row_cell, col_cell)].astype(np.int32, copy=True)
    margin = np.zeros(corners.shape, dtype=np.float32)
    corners[rows, cols] = np.where(chosen == 0, np.int32(-1), chosen - 1)
    margin[rows, cols] = best - second
    return corners, margin


def _lattice_is_closed(corners: np.ndarray, ring: np.ndarray, steps: int) -> bool:
    """Is the border clear of the outer edge of the sampled region?

    A crossing *on* that edge would need a vertex the merged rectangles on the
    other side know nothing about: a T-junction, or a gap where the two sides of
    the border disagree.  The caller grows the ring until this holds.
    """
    squares = np.repeat(np.repeat(ring, steps, axis=0), steps, axis=1)
    if squares.shape != (corners.shape[0] - 1, corners.shape[1] - 1):
        return False
    # Vertical lattice edges, between the squares (row, col) and (row, col + 1):
    # both ends of the edge are corners of column col + 1.
    left = corners[:, :-1]
    border = squares[:, :-1] != squares[:, 1:]
    if np.any(border & (left[:-1, 1:] != left[1:, 1:])):
        return False
    # Horizontal lattice edges, between (row, col) and (row + 1, col): both ends
    # are corners of row + 1.
    top = corners[:-1, :]
    border = squares[:-1, :] != squares[1:, :]
    if np.any(border & (top[1:, :-1] != top[1:, 1:])):
        return False
    return True


def smooth_lattice(
    indices: np.ndarray,
    *,
    factor: int | None = None,
    sigma: float = SMOOTH_SIGMA,
    progress: ProgressFn | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Refine a match grid and hand back the lattice the exporter traces.

    :func:`smooth_indices` re-labels refined *cells*, which trades the match
    grid's staircase for a four-times finer one — but a staircase all the same,
    and a slicer zoomed to 0.02 mm per pixel still shows 0.07 mm of saw tooth
    (measured on the 512 px pig at 4 steps).  Steps are made of cells, so no
    number of extra cells removes them.

    The border is traced instead.  Each colour is an indicator field blurred on
    the match grid, so the border between two colours is where their fields are
    equal: it crosses the refined lattice edges, not the cells.  Every lattice
    edge whose two ends disagree carries one crossing point, placed by linear
    interpolation of ``winner − runner-up``; joining the crossings of a cell
    (marching squares) gives the border as it really is.  Two crossings make the
    cell two straight pieces, so a slanted border comes out as one chord instead
    of a staircase, and a curve as a chord chain at one refined cell per link.

    Returns ``(fine, corners, margin)``:

    * ``fine`` — refined labels for pixel counts, the match-grid label repeated
      ``steps`` times.  Repeating (rather than re-classifying near the border)
      keeps the printed area identical to the unsmoothed export, which is what
      the tests have always asserted; the border geometry carries the shape.
    * ``corners`` — the winning label at every refined lattice *point*, sampled
      at ``index / steps − 0.5`` in match-grid coordinates (shape one larger
      than ``fine`` on both axes), and ``margin`` — winner minus runner-up
      there, for placing the crossings.
    * ``(labels, None, None)`` when there is nothing to refine — and
      ``(raster, None, None)`` with the plain :func:`smooth_indices` result if
      the border stays ambiguous three cells out, which the walked geometry
      cannot close and the re-labelled raster can.
    """
    labels = np.asarray(indices)
    if labels.ndim != 2:
        raise ValueError("indices must be two-dimensional")
    height, width = (int(size) for size in labels.shape)
    steps = max(1, int(factor)) if factor else smooth_factor(height, width)
    if steps <= 1 or height == 0 or width == 0:
        return labels.astype(np.int32, copy=True), None, None

    slots = np.where(labels < 0, 0, labels + 1).astype(np.int32)
    kernel = _gaussian_kernel(sigma)
    present = np.unique(slots)
    fields: list[tuple[int, np.ndarray]] = []
    # Every field is non-negative and they sum to one everywhere, so -1 is
    # strictly below all of them and the first slot of a tie keeps the cell.
    best = np.full((height, width), -1.0, dtype=np.float32)
    second = np.zeros((height, width), dtype=np.float32)
    chosen = np.zeros((height, width), dtype=np.int32)
    for position, slot in enumerate(present):
        if progress is not None:
            progress(
                f"圆滑色块边界 {position + 1}/{len(present)}",
                (position + 1) / len(present),
            )
        field = _blur_separable((slots == slot).astype(np.float32), kernel)
        fields.append((int(slot), field))
        better = field > best
        second = np.where(better, np.maximum(best, 0.0), np.maximum(second, field))
        best = np.where(better, field, best)
        chosen[better] = int(slot)

    coarse = np.where(chosen == 0, np.int32(-1), chosen - 1).astype(np.int32)
    ring = _grow(_label_border(coarse))
    corners: np.ndarray | None = None
    margin: np.ndarray | None = None
    for _ in range(3):
        corners, margin = _sample_lattice(fields, ring, steps, coarse)
        if _lattice_is_closed(corners, ring, steps):
            break
        ring = _grow(ring)
    else:
        return smooth_indices(labels, factor=steps, progress=progress), None, None

    fine = np.repeat(np.repeat(coarse, steps, axis=0), steps, axis=1)
    return fine, corners, margin


class _Outline:
    """The material's outline, gathered edge by edge from every solid region.

    A face on its own says nothing about where the slab is open; what says it is
    *the edge walked once*.  Every solid piece (a traced piece, the perimeter of a
    refined block) walks its own boundary with the material on its left and hands
    each directed edge in here:

    * the same undirected edge walked twice is two faces put back to back, so it is
      material inside — drop it;
    * an edge walked once is the outline — wall it.

    Two colours meeting simply walk the seam twice and it disappears, which is why
    no colour-to-colour rule is needed; a chord, a fan spoke and a lattice line all
    go through the same check, which is why the degenerate cases stopped needing
    rules of their own.  The only care is T-junctions: a block's edge is longer than
    the traced pieces that touch it, so every long edge is cut at the vertices
    already recorded on its line before the count is taken.
    """

    def __init__(self, quantum: float = _TRACE_QUANTUM) -> None:
        self._quantum = quantum
        self._edges: dict[tuple[int, int, int, int], tuple[int, int, int, int, int]] = {}
        self._columns: dict[int, set[int]] = {}
        self._rows: dict[int, set[int]] = {}

    def _index(self, point: tuple[float, float]) -> tuple[int, int]:
        return (
            int(round(point[0] / self._quantum)),
            int(round(point[1] / self._quantum)),
        )

    def add(self, start: tuple[float, float], stop: tuple[float, float], colour: int) -> None:
        """One directed edge of one solid region: the material sits on its left."""
        first = self._index(start)
        second = self._index(stop)
        if first == second:
            return
        key = (min(first[0], second[0]), min(first[1], second[1]),
               max(first[0], second[0]), max(first[1], second[1]))
        if key in self._edges:
            del self._edges[key]
        else:
            self._edges[key] = (first[0], first[1], second[0], second[1], colour)
        self._columns.setdefault(first[0], set()).add(first[1])
        self._columns.setdefault(second[0], set()).add(second[1])
        self._rows.setdefault(first[1], set()).add(first[0])
        self._rows.setdefault(second[1], set()).add(second[0])

    def _keep(
        self,
        cut: dict[tuple[int, int, int, int], tuple[int, int, int, int, int]],
        first: tuple[int, int],
        second: tuple[int, int],
        colour: int,
    ) -> None:
        key = (min(first[0], second[0]), min(first[1], second[1]),
               max(first[0], second[0]), max(first[1], second[1]))
        if key in cut:
            del cut[key]
        else:
            cut[key] = (first[0], first[1], second[0], second[1], colour)

    def _cut(self, edge: tuple[int, int, int, int, int]) -> list[tuple[int, int, int, int, int]]:
        """One registered edge, cut at the vertices already recorded on its line.

        The pieces keep the edge's own direction: the stops are built from low to
        high and reversed when the edge runs the other way, or a descending edge
        hands back (high, low) pieces that match no neighbour at all.
        """
        first_x, first_y, second_x, second_y, colour = edge
        if first_x == second_x:
            line = sorted(self._columns.get(first_x, ()))
            low, high = sorted((first_y, second_y))
            stops = [low] + [value for value in line if low < value < high] + [high]
            if first_y > second_y:
                stops.reverse()
            return [(first_x, one, first_x, other, colour) for one, other in zip(stops, stops[1:])]
        if first_y == second_y:
            line = sorted(self._rows.get(first_y, ()))
            low, high = sorted((first_x, second_x))
            stops = [low] + [value for value in line if low < value < high] + [high]
            if first_x > second_x:
                stops.reverse()
            return [(one, first_y, other, first_y, colour) for one, other in zip(stops, stops[1:])]
        return [edge]

    def walls(self) -> list[tuple[float, float, float, float, int]]:
        """The outline as (x0, y0, x1, y1, colour) in lattice coordinates.

        Every registered edge is cut first and the pieces are counted afterwards.
        Cutting while counting lets a long block edge re-insert a piece a traced
        piece had already cancelled — the walk-in order decided the answer, and a
        seam that should have disappeared came back as a wall.
        """
        pieces: list[tuple[int, int, int, int, int]] = []
        for edge in self._edges.values():
            pieces.extend(self._cut(edge))
        cut: dict[tuple[int, int, int, int], tuple[int, int, int, int, int]] = {}
        for first_x, first_y, second_x, second_y, colour in pieces:
            self._keep(cut, (first_x, first_y), (second_x, second_y), colour)
        quantum = self._quantum
        return [
            (first_x * quantum, first_y * quantum, second_x * quantum, second_y * quantum, colour)
            for first_x, first_y, second_x, second_y, colour in cut.values()
        ]


def _trace_square(
    row: int,
    col: int,
    corners: np.ndarray,
    margin: np.ndarray,
    columns: int,
    rows: int,
    z_base_top: float,
    z_colour_top: float,
    add: Callable[[list[Point], int], None],
    rim: Callable[[tuple[float, float], tuple[float, float], int], None],
    base_rim: Callable[[tuple[float, float], tuple[float, float], int], None],
    *,
    labels: np.ndarray | None = None,
    uniform: np.ndarray | None = None,
) -> None:
    """Build the pieces of one refined square whose four corners disagree.

    Marching squares, walked counter-clockwise so the solid is on the left of
    every edge: two crossings give two pieces that meet along the chord between
    them — a straight border is then exactly straight, with no vertex in the
    middle of it.  Three or four crossings meet inside the square, so each piece
    takes the middle of the square as a fan tip instead, which keeps the piece
    count equal to the crossing count and leaves no gap.

    Walls belong to the piece that is *not* transparent, along its own direction
    (the winding carries the solid on the left, which is what makes the outward
    normal come out right), plus the stretches of the piece that run along the
    outer edge of the picture: outside the picture is not solid, so the slab has
    to be closed there too.
    """
    x0, y0 = float(col), float(row)
    points = ((x0, y0), (x0 + 1.0, y0), (x0 + 1.0, y0 + 1.0), (x0, y0 + 1.0))
    label = (
        int(corners[row, col]),
        int(corners[row, col + 1]),
        int(corners[row + 1, col + 1]),
        int(corners[row + 1, col]),
    )
    weight = (
        float(margin[row, col]),
        float(margin[row, col + 1]),
        float(margin[row + 1, col + 1]),
        float(margin[row + 1, col]),
    )

    def tidy(body: list[tuple[float, float]], snap: float = _TRACE_EPS) -> list[tuple[float, float]]:
        """Drop a point that repeats its neighbour.

        A border that runs along a lattice line puts two crossings on the same
        point, and the piece between them is a sliver of no area: leaving the
        repeats in gives the triangulator triangles of area zero, and leaving the
        slivers in stacks two faces on the same edges.
        """
        out: list[tuple[float, float]] = []
        for point in body:
            if out and abs(out[-1][0] - point[0]) <= snap and abs(out[-1][1] - point[1]) <= snap:
                continue
            out.append(point)
        while (
            len(out) > 1
            and abs(out[0][0] - out[-1][0]) <= snap
            and abs(out[0][1] - out[-1][1]) <= snap
        ):
            out.pop()
        return out

    def area_of(body: list[tuple[float, float]]) -> float:
        total = 0.0
        for index in range(len(body)):
            x0, y0 = body[index]
            x1, y1 = body[(index + 1) % len(body)]
            total += x0 * y1 - x1 * y0
        return abs(total) * 0.5

    crossings: dict[int, tuple[float, float]] = {}
    for index in range(4):
        following = (index + 1) % 4
        if label[index] == label[following]:
            continue
        total = weight[index] + weight[following]
        share = 0.5 if total <= 0.0 else weight[index] / total
        if index == 0:
            crossings[index] = (_snap_trace(x0 + share), _snap_trace(y0))
        elif index == 1:
            crossings[index] = (_snap_trace(x0 + 1.0), _snap_trace(y0 + share))
        elif index == 2:
            crossings[index] = (_snap_trace(x0 + 1.0 - share), _snap_trace(y0 + 1.0))
        else:
            crossings[index] = (_snap_trace(x0), _snap_trace(y0 + 1.0 - share))
    if len(crossings) < 2:
        return

    walk: list[tuple[tuple[float, float], int, bool]] = []
    for index in range(4):
        walk.append((points[index], label[index], True))
        if index in crossings:
            walk.append((crossings[index], label[index], False))
    first = next(index for index, token in enumerate(walk) if not token[2])
    walk = walk[first:] + walk[:first]
    marks = [index for index, token in enumerate(walk) if not token[2]]

    pieces: list[tuple[list[tuple[float, float]], int]] = []
    for number, begin in enumerate(marks):
        if number + 1 < len(marks):
            body = walk[begin : marks[number + 1]] + [walk[marks[number + 1]]]
        else:
            body = walk[begin:] + [walk[0]]
        colour = next(token[1] for token in body if token[2])
        pieces.append(([token[0] for token in body], colour))
    if len(pieces) > 2:
        middle = (x0 + 0.5, y0 + 0.5)
        pieces = [(body + [middle], colour) for body, colour in pieces]
    pieces = [(tidy(body, _TRACE_SNAP), colour) for body, colour in pieces]

    # 边界线本身要走的两条弧，合并之前先留一份：一张色块边界贴着格线走的时候，靠边那件窄到
    # 没有面积，会被并进邻居、面不该发（见下），可这条边界线还得有人立起来 —— 墙按弧发，
    # 不按发过面的件发。
    arcs = [(tidy(body, _TRACE_SNAP), colour) for body, colour in pieces]
    pieces = list(arcs)

    # 边界正好压在格线上时，有些件是「贴着格线的薄片」，面积小到没有（见 _TRACE_SNAP）：
    # 它和相邻那件共用那个交点，并进去以后两件合成一整块 —— 面才发得对。
    merged: list[tuple[list[tuple[float, float]], int]] = []
    for body, colour in pieces:
        if merged and area_of(body) <= _TRACE_MIN_AREA:
            previous, previous_colour = merged[-1]
            merged[-1] = (tidy(previous + body[1:], _TRACE_SNAP), previous_colour)
            continue
        merged.append((body, colour))
    while len(merged) > 1 and area_of(merged[0][0]) <= _TRACE_MIN_AREA:
        body, _colour = merged.pop(0)
        previous, previous_colour = merged[-1]
        merged[-1] = (tidy(previous + body[1:], _TRACE_SNAP), previous_colour)
    pieces = merged

    solid = [
        (body, colour)
        for body, colour in pieces
        if colour >= 0 and len(body) >= 3 and area_of(body) > _TRACE_MIN_AREA
    ]

    # Walls are not decided here any more: every edge of every solid piece goes into
    # the outline (see :class:`_Outline`), which cancels the ones that are material
    # inside and hands back the rest.  Two crossings, four crossings, a border lying
    # exactly on a lattice line and a 0.07-cell sliver all take the same path.
    # 顶面那层和底板那层各记一份：底板整片是一个色（BASE_TAG），它自己的内部缝要
    # 自己抵消，不然底板层的墙会沿着色块之间的缝冒出来。
    for body, colour in solid:
        add([(x, y, z_colour_top) for x, y in body], colour)
        # 底面整片是底板色：同一件反过来绕一遍（外法线朝 −z）。
        add([(x, y, 0.0) for x, y in reversed(body)], BASE_TAG)
        for index in range(len(body)):
            rim(body[index], body[(index + 1) % len(body)], colour)
            base_rim(body[index], body[(index + 1) % len(body)], colour)


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------
_EPS = 1e-9

#: Coordinates are indexed on an integer lattice so "does a vertex sit on this
#: line" is an exact test rather than a tolerance guess.  The scale has to be fine
#: enough to keep the two heights apart: rounding to whole numbers puts the base
#: layer's top (0.8 mm) and the colour layer's top (1.4 mm) on the same integer,
#: and the index then hands a top-surface edge a vertex from the layer below it —
#: which tears the polygon into a shape no triangulation can close.
_SCALE = 1_000_000


def _newell(points: list[Point]) -> tuple[float, float, float]:
    """A planar polygon's area-weighted normal."""
    nx = ny = nz = 0.0
    count = len(points)
    for index in range(count):
        x0, y0, z0 = points[index]
        x1, y1, z1 = points[(index + 1) % count]
        nx += (y0 - y1) * (z0 + z1)
        ny += (z0 - z1) * (x0 + x1)
        nz += (x0 - x1) * (y0 + y1)
    return nx, ny, nz


def _polygon_area(flat: list[tuple[float, float]]) -> float:
    """Twice the signed area of a 2-D polygon; positive when counter-clockwise."""
    total = 0.0
    count = len(flat)
    for index in range(count):
        x0, y0 = flat[index]
        x1, y1 = flat[(index + 1) % count]
        total += x0 * y1 - x1 * y0
    return total


def _cross2(o, a, b) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _inside_triangle(p, a, b, c) -> bool:
    """Point-in-triangle for a counter-clockwise triangle, boundary included."""
    return (
        _cross2(a, b, p) >= 0.0
        and _cross2(b, c, p) >= 0.0
        and _cross2(c, a, p) >= 0.0
    )


class _Stitcher:
    """Weld coincident vertices, then split every edge a third vertex sits on.

    A rectangle decomposition inevitably leaves T-junctions: a long edge of one
    rectangle meets the corner of two neighbours, and a slicer reads that as an
    open boundary.  Every polygon here is axis-aligned in pixel space, so "this
    vertex lies strictly inside that edge" is an exact test.

    Triangulating a split loop by fanning it from one of its **own corners** emits
    zero-area slivers whenever that corner shares an original edge with the
    inserted points — and those slivers are what turned a handful of real
    T-junctions into thousands of reported open and non-manifold edges.  A loop is
    therefore ear-clipped rather than fanned; see :meth:`triangulate`.
    """

    def __init__(self, points: np.ndarray) -> None:
        # 焊接按微米整数格做，不按精确浮点。描出来的交点只保证落在格线之间、走的是
        # 同一套 1e-5 取整，不保证两次算出来的浮点逐位相同：按精确浮点焊接会把同一个
        # 点留成两个顶点，T 型接点插进去的那个和邻居自己用的那个对不上号，面上就多出
        # 一条「开口」边（1.6.27 自检里 160×120 那张图报过 3940 条）。
        integer = np.rint(np.asarray(points, dtype=np.float64) * _SCALE).astype(np.int64)
        welded, inverse = np.unique(integer, axis=0, return_inverse=True)
        self.points: list[Point] = [tuple(row) for row in (welded / _SCALE).tolist()]
        self.inverse = np.asarray(inverse, dtype=np.int64).reshape(-1)
        self._integer = welded
        self._lines: dict[tuple[str, int, int], list[int]] = {}
        for index, (x, y, z) in enumerate(self._integer.tolist()):
            self._lines.setdefault(("x", y, z), []).append(index)
            self._lines.setdefault(("y", x, z), []).append(index)
            self._lines.setdefault(("z", x, y), []).append(index)
        self._cache: dict[tuple[int, int], list[int]] = {}

    def welded(self, index: int) -> int:
        """The welded vertex id of the raw vertex ``index``."""
        return int(self.inverse[index])

    def between(self, a: int, b: int) -> list[int]:
        """Welded vertices strictly inside the edge ``a→b``, in travel order.

        The cache always stores the ascending order, because the same edge is
        asked for from both of its faces and only one of them may travel
        downwards.
        """
        key = (a, b) if a <= b else (b, a)
        cached = self._cache.get(key)
        if cached is None:
            pa = self._integer[a]
            pb = self._integer[b]
            delta = np.abs(pb - pa)
            # Only axis-aligned edges can carry T-junctions.  A fan diagonal is
            # not axis-aligned, and the line index below would happily return
            # vertices that lie on the diagonal's *bounding line* rather than on
            # the diagonal, which would tear the polygon open.
            if int(np.count_nonzero(delta)) != 1:
                cached = []
            else:
                axis = int(np.argmax(delta))
                names = ("x", "y", "z")
                others = [int(pa[i]) for i in range(3) if i != axis]
                candidates = self._lines.get((names[axis], others[0], others[1]), [])
                low = int(min(pa[axis], pb[axis]))
                high = int(max(pa[axis], pb[axis]))
                values = self._integer[candidates, axis]
                found = [
                    int(candidates[i])
                    for i in np.flatnonzero((values > low) & (values < high))
                ]
                found.sort(key=lambda index: int(self._integer[index, axis]))
                cached = found
            self._cache[key] = cached
        return list(reversed(cached)) if a > b else cached

    def loop(self, polygon: list[int]) -> list[int]:
        """The polygon's boundary with every T-junction vertex inserted."""
        out: list[int] = []
        count = len(polygon)
        for position in range(count):
            a = int(polygon[position])
            b = int(polygon[(position + 1) % count])
            out.append(a)
            out.extend(self.between(a, b))
        return out

    def _fan(self, loop: list[int]) -> list[tuple[int, int, int]]:
        """Last-resort fan from a centroid, used only when ear clipping stalls."""
        count = len(loop)
        centre = len(self.points)
        self.points.append(
            tuple(
                sum(self.points[index][axis] for index in loop) / count
                for axis in range(3)
            )
        )
        return [(centre, loop[k], loop[(k + 1) % count]) for k in range(count)]

    def triangulate(self, loop: list[int]) -> list[tuple[int, int, int]]:
        """Ear-clip a planar loop, keeping every T-junction vertex and no slivers.

        Fanning from a fixed corner or from the centroid both eventually cut a
        triangle whose edge lies *along* an existing boundary edge: a wall's
        vertical edge, or the segment the loop's own average lands on once a
        neighbour has split its top edge.  Ear clipping cannot do that — an ear is
        only taken when no other vertex of the loop sits inside it (boundary
        included), so a diagonal never lands on a vertex that is not its endpoint.

        Collinear vertices are deliberately **not** clipped away: they are exactly
        the vertices a neighbouring polygon's corner needs to find.  Dropping them
        to get tidier triangles would put the T-junction straight back.
        """
        count = len(loop)
        if count < 3:
            return []
        if count == 3:
            return [(loop[0], loop[1], loop[2])]

        coords = [self.points[index] for index in loop]
        normal = _newell(coords)
        axis = max(range(3), key=lambda item: abs(normal[item]))
        if abs(normal[axis]) <= _EPS:
            return self._fan(loop)
        other = [item for item in range(3) if item != axis]
        flat = [(point[other[0]], point[other[1]]) for point in coords]

        order = list(range(count))
        flipped = _polygon_area(flat) < 0.0
        if flipped:
            order.reverse()

        out: list[tuple[int, int, int]] = []
        while len(order) > 3:
            clipped = False
            for position in range(len(order)):
                before = order[position - 1]
                here = order[position]
                after = order[(position + 1) % len(order)]
                a, b, c = flat[before], flat[here], flat[after]
                if _cross2(a, b, c) <= _EPS:
                    continue  # reflex or collinear — not an ear
                if any(
                    _inside_triangle(flat[j], a, b, c)
                    for j in order
                    if j not in (before, here, after)
                ):
                    continue
                out.append((loop[before], loop[here], loop[after]))
                order.pop(position)
                clipped = True
                break
            if not clipped:
                out.extend(self._fan([loop[index] for index in order]))
                order = []
                break
        if len(order) == 3:
            out.append((loop[order[0]], loop[order[1]], loop[order[2]]))
        return [(a, c, b) for a, b, c in out] if flipped else out

    def build(
        self, polygons: list[list[int]], tags: list[int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Triangulate indexed polygons, carrying one tag through to each face."""
        faces: list[tuple[int, int, int]] = []
        face_tags: list[int] = []
        for polygon, tag in zip(polygons, tags):
            triangles = self.triangulate(self.loop(polygon))
            faces.extend(triangles)
            face_tags.extend([tag] * len(triangles))
        return (
            np.array(self.points, dtype=np.float64).reshape(-1, 3),
            np.array(faces, dtype=np.int64).reshape(-1, 3),
            np.array(face_tags, dtype=np.int64).reshape(-1),
        )


def _assemble(
    polygons: list[list[Point]], tags: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weld, T-junction-stitch and triangulate a tagged polygon soup.

    The whole solid goes through **one** stitcher.  Stitching part by part cannot
    work: the long edge belongs to one part and the two corner vertices that
    should split it belong to its neighbour, so each part would keep half a
    T-junction and the weld would show an open edge.
    """
    flat = [point for polygon in polygons for point in polygon]
    stitcher = _Stitcher(np.array(flat, dtype=np.float64).reshape(-1, 3))
    indexed: list[list[int]] = []
    cursor = 0
    for polygon in polygons:
        indexed.append(
            [stitcher.welded(cursor + offset) for offset in range(len(polygon))]
        )
        cursor += len(polygon)
    return stitcher.build(indexed, tags)


def stitch_t_junctions(
    points: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Weld coincident vertices and split every edge a third vertex sits on.

    A triangle-level entry point for callers that already hold a triangle soup;
    the plate builder works on polygons instead, through :func:`_assemble`.
    """
    stitcher = _Stitcher(points)
    indexed = [
        [stitcher.welded(int(index)) for index in triangle]
        for triangle in np.asarray(faces, dtype=np.int64).tolist()
    ]
    vertices, triangles, _tags = stitcher.build(indexed, [0] * len(indexed))
    return vertices, triangles


def _prism_polygons(
    mask: np.ndarray, *, z_bottom: float, z_top: float
) -> list[list[Point]]:
    """The six-sided boundary of a closed prism over ``mask``, as quads."""
    height, width = mask.shape
    polygons: list[list[Point]] = []

    for x, y, w, h in _merge_stacked(rectangles(mask)):
        x0, x1 = float(x), float(x + w)
        y0, y1 = float(y), float(y + h)
        # top face, normal +Z
        polygons.append([(x0, y0, z_top), (x1, y0, z_top), (x1, y1, z_top), (x0, y1, z_top)])
        # bottom face, normal -Z
        polygons.append(
            [(x0, y1, z_bottom), (x1, y1, z_bottom), (x1, y0, z_bottom), (x0, y0, z_bottom)]
        )

    occupied = mask.astype(bool)
    north = occupied & ~np.vstack([np.zeros((1, width), dtype=bool), occupied[:-1]])
    south = occupied & ~np.vstack([occupied[1:], np.zeros((1, width), dtype=bool)])
    west = occupied & ~np.hstack([np.zeros((height, 1), dtype=bool), occupied[:, :-1]])
    east = occupied & ~np.hstack([occupied[:, 1:], np.zeros((height, 1), dtype=bool)])

    for y in np.flatnonzero(north.any(axis=1)):
        plane = float(y)
        for x0, x1 in _runs(north[y]):
            a, b = float(x0), float(x1)
            # normal -Y
            polygons.append(
                [(a, plane, z_bottom), (b, plane, z_bottom), (b, plane, z_top), (a, plane, z_top)]
            )

    for y in np.flatnonzero(south.any(axis=1)):
        plane = float(y) + 1.0
        for x0, x1 in _runs(south[y]):
            a, b = float(x0), float(x1)
            # normal +Y
            polygons.append(
                [(b, plane, z_bottom), (a, plane, z_bottom), (a, plane, z_top), (b, plane, z_top)]
            )

    # The side walls run down the picture, so their runs are gathered per column
    # and merged vertically — that keeps them aligned with the top and bottom
    # faces instead of leaving a staircase of one-pixel steps.
    for x in np.flatnonzero(west.any(axis=0)):
        plane = float(x)
        for y0, y1 in _runs(west[:, x]):
            ya, yb = float(y0), float(y1)
            # normal -X
            polygons.append(
                [(plane, yb, z_bottom), (plane, ya, z_bottom), (plane, ya, z_top), (plane, yb, z_top)]
            )

    for x in np.flatnonzero(east.any(axis=0)):
        plane = float(x) + 1.0
        for y0, y1 in _runs(east[:, x]):
            ya, yb = float(y0), float(y1)
            # normal +X
            polygons.append(
                [(plane, ya, z_bottom), (plane, yb, z_bottom), (plane, yb, z_top), (plane, ya, z_top)]
            )

    return polygons


def prism_mesh(
    mask: np.ndarray,
    *,
    z_bottom: float,
    z_top: float,
    pixel_mm: float,
    flip_y: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """A closed prism over ``mask``: top + bottom over the mask, sides on its rim.

    Returns ``(vertices, triangles)`` with outward-facing winding.  The sides are
    only emitted where the mask meets the outside, so the prism carries no
    internal walls.

    The faces are wound in pixel space, where ``y`` grows downwards.  Placing the
    picture upright on the build plate needs ``y`` mirrored, and a mirror inverts
    every normal, so the winding is reversed at the very end to compensate.
    """
    if z_top <= z_bottom:
        raise ValueError("z_top must be greater than z_bottom")
    height = mask.shape[0]
    polygons = _prism_polygons(mask, z_bottom=z_bottom, z_top=z_top)
    points, faces, _tags = _assemble(polygons, [0] * len(polygons))
    points[:, 0] *= pixel_mm
    points[:, 1] *= pixel_mm
    if flip_y:
        points[:, 1] = height * pixel_mm - points[:, 1]
        faces = faces[:, ::-1]
    return points, faces


# ---------------------------------------------------------------------------
# Mesh health
# ---------------------------------------------------------------------------
def edge_health(
    points: np.ndarray, faces: np.ndarray, *, weld: bool = False
) -> tuple[int, int]:
    """``(open_edges, non_manifold_edges)`` counted the way a reader counts them.

    Counting follows the **index list**: an edge is shared only when both of its
    triangles name the same two vertex indices.  Two vertices at identical
    coordinates but different indices therefore leave that edge open.  This is
    what a slicer's mesh statistics see — Bambu Studio reported 14,650 open
    edges for a plate whose geometry was closed, and 14,650 is exactly the
    index-level count — so this is the number that must gate acceptance.

    ``weld=True`` first reunites vertices that share a position, which answers a
    different question ("is the *geometry* closed?") and is only useful when
    diagnosing the index list.  It is never the acceptance number: welding here
    is what hid the defect that shipped in 1.6.5.
    """
    vertices = np.asarray(points, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if weld:
        _welded, inverse = np.unique(vertices, axis=0, return_inverse=True)
        triangles = np.asarray(inverse, dtype=np.int64).reshape(-1)[triangles]
    counts: dict[tuple[int, int], int] = {}
    for a, b, c in triangles.tolist():
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            counts[key] = counts.get(key, 0) + 1
    return (
        sum(1 for count in counts.values() if count == 1),
        sum(1 for count in counts.values() if count > 2),
    )


def signed_volume(points: np.ndarray, faces: np.ndarray) -> float:
    """Six times the signed volume, over six — positive when normals face out."""
    vertices = np.asarray(points, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    a = vertices[triangles[:, 0]]
    b = vertices[triangles[:, 1]]
    c = vertices[triangles[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def merge_parts(parts: Sequence[PlatePart]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Fold the per-part vertex lists into ONE shared pool, with per-part faces.

    Parts each keep their own copy of the corners they share with a neighbour, so
    writing them back to back gives every part-border edge two different index
    pairs.  The solid is geometrically closed but topologically open, and a reader
    that counts edges by index — which is exactly what Bambu Studio's mesh
    statistics do — reports the whole part border as open edges.  That is the bug
    that shipped in 1.6.5: the user's file reported 14,650 open edges while
    containing no gap at all.

    Vertices are matched on the builder's own integer lattice,
    ``round(coordinate * _SCALE)`` with ``_SCALE = 1_000_000``.  The 3MF writer
    prints exactly those six decimals, so a vertex here and the vertex text in
    the file agree one for one and the two welds cannot drift apart.

    Returns the merged ``(N, 3)`` float64 points in first-seen order, and one
    ``(T, 3)`` int64 triangle array per part indexing into that pool.
    """
    merged: list[Point] = []
    slot_of: dict[tuple[int, int, int], int] = {}
    per_part: list[np.ndarray] = []
    for part in parts:
        vertices = np.asarray(part.vertices, dtype=np.float64)
        local = np.empty(len(vertices), dtype=np.int64)
        for position, (x, y, z) in enumerate(vertices.tolist()):
            key = (round(x * _SCALE), round(y * _SCALE), round(z * _SCALE))
            slot = slot_of.get(key)
            if slot is None:
                slot = len(merged)
                slot_of[key] = slot
                merged.append((x, y, z))
            local[position] = slot
        per_part.append(local[np.asarray(part.triangles, dtype=np.int64)])
    points = np.asarray(merged, dtype=np.float64).reshape(-1, 3)
    return points, per_part


def plate_mesh(plate: PlateModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The whole plate as one mesh plus a per-triangle extruder list.

    The parts are merged into a single shared vertex pool, so the result is the
    closed solid the exporters have to write — call :func:`edge_health` on it
    without welding and it must report no open edges.
    """
    if not plate.parts:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.int64),
            np.zeros(0, dtype=np.int64),
        )
    points, per_part = merge_parts(plate.parts)
    faces = np.vstack(per_part)
    extruders = np.concatenate(
        [
            np.full(len(part.triangles), part.extruder, dtype=np.int64)
            for part in plate.parts
        ]
    )
    return points, faces, extruders


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def build_plate(
    result: MatchResult,
    settings: PlateSettings | None = None,
    *,
    base_label: str = "底板",
    progress: ProgressFn | None = None,
) -> PlateModel:
    """Build the picture as ONE watertight slab, then cut it into per-filament parts.

    The solid has exactly four kinds of face:

    * the underside, over the whole silhouette, in the base colour;
    * the top surface at ``base_thickness_mm`` (the slab's whole thickness, 1.6.42),
      tiled by colour region and tagged with that region's filament;
    * the outside rim below ``base_thickness - colour_thickness``, in the base colour;
    * the outside rim above it, in the colour of the region that owns that
      stretch of rim.

    There are no interior walls and no coincident faces, so the assembly is a
    genuine 2-manifold with a well-defined inside and outside.
    """
    config = (settings or PlateSettings()).clamped()
    if not result.palette:
        raise ValueError("没有一个可打印的颜色，无法生成模型。")

    notes: list[str] = []
    picture_width = max(1, int(result.width))
    picture_height = max(1, int(result.height))
    # 一格就是一格喷嘴宽 (1.6.42)：板的长宽 = 图片像素数 × 这一格，不设上限、不自动缩。
    pixel = config.cell_mm

    # --- 导出时圆滑色块 (1.6.22 / 1.6.27) -------------------------------------
    # 色块边界原本落在匹配栅格的格线上（150 mm 宽、512 像素的图约 0.293 mm 一格），斜着走的地方
    # 就是台阶。1.6.22 把栅格放大 steps 倍重新分类，台阶细到 1/steps 格；1.6.27 改成直接描那条
    # 边界线（见 smooth_lattice）：交点落在格边上，斜边是一根直弦、弯边是一串折线，一级台阶也
    # 剩下。两条路都留着：常走描线，边界在三格外仍说不清时才退回重新分类。
    steps = 1
    lattice: tuple[np.ndarray, np.ndarray] | None = None
    if config.smooth_edges:
        steps = smooth_factor(picture_height, picture_width, len(result.palette))
        if steps > 1:
            indices, corners, margin = smooth_lattice(
                result.indices, factor=steps, progress=progress
            )
            counts = np.array(
                [
                    int(np.count_nonzero(indices == index))
                    for index in range(len(result.palette))
                ],
                dtype=np.asarray(result.counts).dtype,
            )
            result = replace(
                result,
                width=int(indices.shape[1]),
                height=int(indices.shape[0]),
                indices=indices,
                counts=counts,
            )
            if corners is None or margin is None:
                notes.append(
                    f"色块边界已圆滑：在 {steps} 倍细分栅格上重新分类，网格 {pixel / steps:.3f} mm"
                )
            else:
                lattice = (corners, margin)
                notes.append(
                    f"色块边界已圆滑：在 {steps} 倍细分栅格上描出边界线，"
                    f"跨一格 {pixel / steps:.3f} mm，斜的边是直的"
                )

    cell = pixel / steps
    width_mm = result.width * cell
    depth_mm = result.height * cell

    base_index = config.base_index
    if base_index < 0 or base_index >= len(result.palette):
        base_index = int(np.argmax(result.counts))
    base_extruder = base_index + 1

    z_colour_top = config.base_thickness_mm
    z_base_top = config.base_thickness_mm - config.colour_thickness_mm

    silhouette = result.indices >= 0
    if not silhouette.any():
        raise ValueError("图片没有可见像素，无法生成模型。")

    height, width = result.indices.shape
    polygons: list[list[Point]] = []
    tags: list[int] = []

    def add(polygon: list[Point], tag: int) -> None:
        polygons.append(polygon)
        tags.append(tag)

    def tiles(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Cover a mask with rectangles.

        The refined grid of :func:`smooth_indices` has 16× the cells of the
        match grid, and :func:`rectangles` walks them one by one in Python;
        :func:`run_rectangles` finds the same runs with numpy.  The coarse path
        keeps the greedy tiling, so an export without the smoothing switch is
        byte-for-byte what it always was.

        A traced border (``lattice``) covers itself below, so this returns
        nothing then.
        """
        if lattice is not None:
            return []
        if steps > 1:
            return run_rectangles(mask)
        return _merge_stacked(rectangles(mask))

    # --- the traced border (1.6.27) ------------------------------------------
    # 四个角标签一样的细格：边界不会穿过它，照旧合并成大矩形。剩下的一圈（薄薄一条边带）交给
    # marching squares 描件：矩形和描出来的件都在整数格点上，谁也不会切到别人的边上。
    if lattice is not None:
        corners, margin = lattice
        lower_left = corners[:-1, :-1]
        lower_right = corners[:-1, 1:]
        upper_right = corners[1:, 1:]
        upper_left = corners[1:, :-1]
        inside = (
            (lower_left == lower_right)
            & (lower_right == upper_right)
            & (upper_right == upper_left)
        )
        # ``inside`` is already in fine cells: a fine square whose four lattice
        # corners agree is the one the trace never touches.  It doubles as the
        # working mask, because the trace hands back the squares where the border
        # collapsed onto a lattice line (see :func:`_trace_square`): a square like
        # that is solid or empty along its whole width after all, so it is filled
        # and walled by exactly the same two passes as a square that agreed from
        # the start.
        uniform_fine = inside
        outline = _Outline()
        base_outline = _Outline()
        for row, col in np.argwhere(~inside.copy()):
            _trace_square(
                int(row),
                int(col),
                corners,
                margin,
                width,
                height,
                z_base_top,
                z_colour_top,
                add,
                outline.add,
                base_outline.add,
                labels=result.indices,
                uniform=inside,
            )
        for x, y, w, h in run_rectangles(inside & (lower_left >= 0)):
            x0, x1 = float(x), float(x + w)
            y0, y1 = float(y), float(y + h)
            add([(x0, y1, 0.0), (x1, y1, 0.0), (x1, y0, 0.0), (x0, y0, 0.0)], BASE_TAG)
            # 底板那层的四边走底板自己的账：底板整片是一个色，色块之间的缝在它这里
            # 根本不存在，两边各走一遍就抵消了。
            base_outline.add((x0, y0), (x1, y0), BASE_TAG)
            base_outline.add((x1, y0), (x1, y1), BASE_TAG)
            base_outline.add((x1, y1), (x0, y1), BASE_TAG)
            base_outline.add((x0, y1), (x0, y0), BASE_TAG)
        for index in range(len(result.palette)):
            mask = inside & (lower_left == index)
            if not mask.any():
                continue
            for x, y, w, h in run_rectangles(mask):
                x0, x1 = float(x), float(x + w)
                y0, y1 = float(y), float(y + h)
                add(
                    [
                        (x0, y0, z_colour_top),
                        (x1, y0, z_colour_top),
                        (x1, y1, z_colour_top),
                        (x0, y1, z_colour_top),
                    ],
                    index,
                )
                # 整块的四条边同样走一遍（材料在左，和描出来的件一个方向）：相邻整块、
                # 整块和贴着它的窄条都会互相抵消，剩下的才是材料的外形。
                outline.add((x0, y0), (x1, y0), int(index))
                outline.add((x1, y0), (x1, y1), int(index))
                outline.add((x1, y1), (x0, y1), int(index))
                outline.add((x0, y1), (x0, y0), int(index))
        for start_x, start_y, stop_x, stop_y, colour in outline.walls():
            add(
                [
                    (start_x, start_y, z_base_top),
                    (stop_x, stop_y, z_base_top),
                    (stop_x, stop_y, z_colour_top),
                    (start_x, start_y, z_colour_top),
                ],
                colour,
            )
        for start_x, start_y, stop_x, stop_y, _colour in base_outline.walls():
            add(
                [
                    (start_x, start_y, 0.0),
                    (stop_x, stop_y, 0.0),
                    (stop_x, stop_y, z_base_top),
                    (start_x, start_y, z_base_top),
                ],
                BASE_TAG,
            )

    # --- the base colour: the underside of the slab --------------------------
    for x, y, w, h in tiles(silhouette):
        x0, x1 = float(x), float(x + w)
        y0, y1 = float(y), float(y + h)
        add([(x0, y1, 0.0), (x1, y1, 0.0), (x1, y0, 0.0), (x0, y0, 0.0)], BASE_TAG)

    # --- one filament per colour: its patch of the top surface ---------------
    for index in range(len(result.palette)):
        mask = result.indices == index
        if not mask.any():
            continue
        for x, y, w, h in tiles(mask):
            x0, x1 = float(x), float(x + w)
            y0, y1 = float(y), float(y + h)
            add(
                [
                    (x0, y0, z_colour_top),
                    (x1, y0, z_colour_top),
                    (x1, y1, z_colour_top),
                    (x0, y1, z_colour_top),
                ],
                index,
            )

    # --- the outside rim, split at the base layer ----------------------------
    # A wall belongs to a colour only where that colour touches the *outside* of
    # the silhouette.  Where two colours merely touch each other there is no wall
    # at all: it would be an interior face nobody can see and every slicer has to
    # repair.  描出边界线的时候这趟不跑：墙全部由 :class:`_Outline` 从「只被走一次的边」
    # 里发出来（整块的周长和描出来的件走同一张表，互相抵消），图片最外圈也自然包含在内。
    if lattice is None:
        shade = silhouette
        above = np.vstack([np.zeros((1, width), dtype=bool), shade[:-1]])
        below = np.vstack([shade[1:], np.zeros((1, width), dtype=bool)])
        left = np.hstack([np.zeros((height, 1), dtype=bool), shade[:, :-1]])
        right = np.hstack([shade[:, 1:], np.zeros((height, 1), dtype=bool)])

        for index in range(len(result.palette)):
            mask = result.indices == index
            if not mask.any():
                continue
            north = mask & ~above
            south = mask & ~below
            west = mask & ~left
            east = mask & ~right

            for y in np.flatnonzero(north.any(axis=1)):
                plane = float(y)
                for x0, x1 in _runs(north[y]):
                    a, b = float(x0), float(x1)
                    add(
                        [(a, plane, 0.0), (b, plane, 0.0), (b, plane, z_base_top), (a, plane, z_base_top)],
                        BASE_TAG,
                    )
                    add(
                        [
                            (a, plane, z_base_top),
                            (b, plane, z_base_top),
                            (b, plane, z_colour_top),
                            (a, plane, z_colour_top),
                        ],
                        index,
                    )

            for y in np.flatnonzero(south.any(axis=1)):
                plane = float(y) + 1.0
                for x0, x1 in _runs(south[y]):
                    a, b = float(x0), float(x1)
                    add(
                        [(b, plane, 0.0), (a, plane, 0.0), (a, plane, z_base_top), (b, plane, z_base_top)],
                        BASE_TAG,
                    )
                    add(
                        [
                            (b, plane, z_base_top),
                            (a, plane, z_base_top),
                            (a, plane, z_colour_top),
                            (b, plane, z_colour_top),
                        ],
                        index,
                    )

            for x in np.flatnonzero(west.any(axis=0)):
                plane = float(x)
                for y0, y1 in _runs(west[:, x]):
                    ya, yb = float(y0), float(y1)
                    add(
                        [(plane, yb, 0.0), (plane, ya, 0.0), (plane, ya, z_base_top), (plane, yb, z_base_top)],
                        BASE_TAG,
                    )
                    add(
                        [
                            (plane, yb, z_base_top),
                            (plane, ya, z_base_top),
                            (plane, ya, z_colour_top),
                            (plane, yb, z_colour_top),
                        ],
                        index,
                    )

            for x in np.flatnonzero(east.any(axis=0)):
                plane = float(x) + 1.0
                for y0, y1 in _runs(east[:, x]):
                    ya, yb = float(y0), float(y1)
                    add(
                        [(plane, ya, 0.0), (plane, yb, 0.0), (plane, yb, z_base_top), (plane, ya, z_base_top)],
                        BASE_TAG,
                    )
                    add(
                        [
                            (plane, ya, z_base_top),
                            (plane, yb, z_base_top),
                            (plane, yb, z_colour_top),
                            (plane, ya, z_colour_top),
                        ],
                        index,
                    )

    points, faces, face_tags = _assemble(polygons, tags)

    points[:, 0] *= cell
    points[:, 1] *= cell
    points[:, 1] = height * cell - points[:, 1]
    faces = faces[:, ::-1]

    # --- 图片印在哪一面 (1.6.23) ---------------------------------------------
    # 上面的多边形天生是「图片朝上」的：底面整片是底板色，z_colour_top 那一层按色块铺开。
    # 要让图片朝下，就把整块实体在 Z 上照一次镜子：顶面 ↔ 底面、色块层 ↔ 底板层，一件都
    # 不少。 镜像会翻朝向，所以绕序要再翻一次，法线才仍旧朝外；两侧的 z 说明也跟着对调。
    if config.image_on_bottom:
        thickness = z_colour_top
        points[:, 2] = thickness - points[:, 2]
        faces = faces[:, ::-1]
        base_z = (config.colour_thickness_mm, thickness)
        colour_z = (0.0, config.colour_thickness_mm)
    else:
        base_z = (0.0, z_base_top)
        colour_z = (z_base_top, z_colour_top)

    # --- cut the one solid into per-filament parts ---------------------------
    parts: list[PlatePart] = [
        _slice_part(
            BASE_TAG,
            points,
            faces,
            face_tags,
            name=base_label,
            color_hex=result.palette[base_index].color_hex,
            palette_index=base_index,
            extruder=base_extruder,
            region_pixels=int(np.count_nonzero(silhouette)),
            z_bottom_mm=base_z[0],
            z_top_mm=base_z[1],
        )
    ]

    for index, entry in enumerate(result.palette):
        count = int(np.count_nonzero(result.indices == index))
        if not count:
            continue
        parts.append(
            _slice_part(
                index,
                points,
                faces,
                face_tags,
                name=_part_name(entry, index),
                color_hex=entry.color_hex,
                palette_index=index,
                extruder=index + 1,
                region_pixels=count,
                z_bottom_mm=colour_z[0],
                z_top_mm=colour_z[1],
            )
        )

    if z_base_top < 0.4:
        notes.append(
            f"底板色那段只有 {z_base_top:.2f} mm，太薄容易撕裂；底板厚度建议至少 "
            f"{config.colour_thickness_mm + 0.4:.1f} mm"
        )

    return PlateModel(
        parts=parts,
        width_mm=width_mm,
        depth_mm=depth_mm,
        pixel_size_mm=cell,
        base_thickness_mm=config.base_thickness_mm,
        colour_thickness_mm=config.colour_thickness_mm,
        notes=notes,
        source=result.source,
    )


def _slice_part(
    tag: int,
    points: np.ndarray,
    faces: np.ndarray,
    face_tags: np.ndarray,
    **kwargs,
) -> PlatePart:
    """One filament's faces, re-indexed into a private vertex block."""
    selected = faces[face_tags == tag]
    used, local = np.unique(selected, return_inverse=True)
    return PlatePart(
        vertices=points[used],
        triangles=np.asarray(local, dtype=np.int64).reshape(-1, 3),
        **kwargs,
    )


def _part_name(entry: PaletteEntry, index: int) -> str:
    label = entry.label
    if len(label) > 42:
        label = label[:39] + "..."
    return f"{index + 1:02d}_{entry.color_hex}_{label}"

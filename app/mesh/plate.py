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

from dataclasses import dataclass, field

import numpy as np

from ..core.image_matching import MatchResult, PaletteEntry

__all__ = [
    "PlatePart",
    "PlateModel",
    "PlateSettings",
    "rectangles",
    "prism_mesh",
    "stitch_t_junctions",
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
@dataclass(frozen=True)
class PlateSettings:
    """Everything the user can change about the exported slab."""

    target_width_mm: float = 150.0
    base_thickness_mm: float = 0.8
    colour_thickness_mm: float = 0.6
    #: Palette index whose filament also prints the base plate; -1 = the colour
    #: that covers the most pixels.
    base_index: int = -1
    #: Keep the plate inside this footprint even if ``target_width_mm`` is larger.
    max_footprint_mm: float = 250.0
    #: Cells finer than this are below what an 0.4 mm nozzle resolves.
    min_pixel_mm: float = 0.35

    def clamped(self) -> "PlateSettings":
        return PlateSettings(
            target_width_mm=max(10.0, min(400.0, float(self.target_width_mm))),
            base_thickness_mm=max(0.2, min(20.0, float(self.base_thickness_mm))),
            colour_thickness_mm=max(0.2, min(20.0, float(self.colour_thickness_mm))),
            base_index=int(self.base_index),
            max_footprint_mm=max(20.0, min(1000.0, float(self.max_footprint_mm))),
            min_pixel_mm=max(0.05, min(2.0, float(self.min_pixel_mm))),
        )


@dataclass
class PlatePart:
    """One filament's share of the slab, as its own indexed triangle window.

    Parts share no vertex indices — each one carries its own copy of the corners
    it uses — because every exporter writes them as separate groups and offsets
    the indices per part.  The copies sit at identical positions, so welding the
    parts back together reproduces the single solid they were cut from.
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
        return self.base_thickness_mm + self.colour_thickness_mm

    @property
    def extruder_count(self) -> int:
        return max((part.extruder for part in self.parts), default=0)

    @property
    def triangle_count(self) -> int:
        return sum(part.triangle_count for part in self.parts)

    @property
    def vertex_count(self) -> int:
        return sum(int(len(part.vertices)) for part in self.parts)

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
            "vertices": self.vertex_count,
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
        welded, inverse = np.unique(
            np.asarray(points, dtype=np.float64), axis=0, return_inverse=True
        )
        self.points: list[Point] = [tuple(row) for row in welded.tolist()]
        self.inverse = np.asarray(inverse, dtype=np.int64).reshape(-1)
        self._integer = np.rint(welded * _SCALE).astype(np.int64)
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
def edge_health(points: np.ndarray, faces: np.ndarray) -> tuple[int, int]:
    """``(open_edges, non_manifold_edges)`` after welding coincident vertices.

    Bambu Studio welds an imported object's parts together before it reports mesh
    statistics, so welding first is the number the user actually sees.
    """
    welded, inverse = np.unique(
        np.asarray(points, dtype=np.float64), axis=0, return_inverse=True
    )
    del welded
    mapped = np.asarray(inverse, dtype=np.int64).reshape(-1)[
        np.asarray(faces, dtype=np.int64)
    ]
    counts: dict[tuple[int, int], int] = {}
    for a, b, c in mapped.tolist():
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


def plate_mesh(plate: PlateModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The whole plate as one mesh plus a per-triangle extruder list.

    Vertices are concatenated with a per-part offset, exactly as the exporters
    write them; call :func:`edge_health` to weld them back together.
    """
    if not plate.parts:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.int64),
            np.zeros(0, dtype=np.int64),
        )
    points: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    extruders: list[np.ndarray] = []
    offset = 0
    for part in plate.parts:
        points.append(part.vertices)
        faces.append(part.triangles + offset)
        extruders.append(
            np.full(len(part.triangles), part.extruder, dtype=np.int64)
        )
        offset += int(len(part.vertices))
    return np.vstack(points), np.vstack(faces), np.concatenate(extruders)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def build_plate(
    result: MatchResult,
    settings: PlateSettings | None = None,
    *,
    base_label: str = "底板",
) -> PlateModel:
    """Build the picture as ONE watertight slab, then cut it into per-filament parts.

    The solid has exactly four kinds of face:

    * the underside, over the whole silhouette, in the base colour;
    * the top surface at ``base_thickness + colour_thickness``, tiled by colour
      region and tagged with that region's filament;
    * the outside rim below ``base_thickness``, in the base colour;
    * the outside rim above it, in the colour of the region that owns that
      stretch of rim.

    There are no interior walls and no coincident faces, so the assembly is a
    genuine 2-manifold with a well-defined inside and outside.
    """
    config = (settings or PlateSettings()).clamped()
    if not result.palette:
        raise ValueError("没有一个可打印的颜色，无法生成模型。")

    notes: list[str] = []
    pixel = config.target_width_mm / max(1, result.width)
    if pixel * result.height > config.max_footprint_mm:
        pixel = config.max_footprint_mm / max(1, result.height)
        notes.append(
            f"按目标宽度会超出 {config.max_footprint_mm:.0f} mm 的打印范围，已自动缩小"
        )
    if pixel < config.min_pixel_mm:
        notes.append(
            f"每个像素只有 {pixel:.3f} mm，比 0.4 mm 喷嘴还细；相邻颜色在打印时会互相覆盖"
        )

    width_mm = result.width * pixel
    depth_mm = result.height * pixel

    base_index = config.base_index
    if base_index < 0 or base_index >= len(result.palette):
        base_index = int(np.argmax(result.counts))
    base_extruder = base_index + 1

    z_base_top = config.base_thickness_mm
    z_colour_top = config.base_thickness_mm + config.colour_thickness_mm

    silhouette = result.indices >= 0
    if not silhouette.any():
        raise ValueError("图片没有可见像素，无法生成模型。")

    height, width = result.indices.shape
    polygons: list[list[Point]] = []
    tags: list[int] = []

    def add(polygon: list[Point], tag: int) -> None:
        polygons.append(polygon)
        tags.append(tag)

    # --- the base colour: the underside of the slab --------------------------
    for x, y, w, h in _merge_stacked(rectangles(silhouette)):
        x0, x1 = float(x), float(x + w)
        y0, y1 = float(y), float(y + h)
        add([(x0, y1, 0.0), (x1, y1, 0.0), (x1, y0, 0.0), (x0, y0, 0.0)], BASE_TAG)

    # --- one filament per colour: its patch of the top surface ---------------
    for index in range(len(result.palette)):
        mask = result.indices == index
        if not mask.any():
            continue
        for x, y, w, h in _merge_stacked(rectangles(mask)):
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
    # repair.
    above = np.vstack([np.zeros((1, width), dtype=bool), silhouette[:-1]])
    below = np.vstack([silhouette[1:], np.zeros((1, width), dtype=bool)])
    left = np.hstack([np.zeros((height, 1), dtype=bool), silhouette[:, :-1]])
    right = np.hstack([silhouette[:, 1:], np.zeros((height, 1), dtype=bool)])

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

    points[:, 0] *= pixel
    points[:, 1] *= pixel
    points[:, 1] = height * pixel - points[:, 1]
    faces = faces[:, ::-1]

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
            z_bottom_mm=0.0,
            z_top_mm=z_base_top,
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
                z_bottom_mm=z_base_top,
                z_top_mm=z_colour_top,
            )
        )

    if config.colour_thickness_mm < 0.4:
        notes.append("颜色层不足 0.4 mm，打印时可能在层间混色，建议至少 0.4 mm")
    if config.base_thickness_mm < 0.4:
        notes.append("底板不足 0.4 mm，太薄容易撕裂，建议至少 0.4 mm")

    return PlateModel(
        parts=parts,
        width_mm=width_mm,
        depth_mm=depth_mm,
        pixel_size_mm=pixel,
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

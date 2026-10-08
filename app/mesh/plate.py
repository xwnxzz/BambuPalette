"""Turn a matched picture into a flat, printable, multi-material plate.

The user's requirement is explicit: the export must be a *flat* print, not a
relief.  Two thicknesses are therefore all we need —

* a **base plate** covering the picture's silhouette at
  ``base_thickness_mm``, so the object is a solid slab with exactly the
  picture's outline (nothing thinner than a printable layer);
* one **cone per colour** stacked on top of the base at
  ``colour_thickness_mm``, so the top surface reproduces the picture and every
  colour region can be assigned to its own filament.

Geometry is built on the pixel grid and compressed into maximal rectangles, so a
512×512 picture becomes thousands of quads rather than hundreds of thousands.
Every part is a closed prism, and the whole model is watertight up to the
T-junctions a rectangle decomposition necessarily leaves — the mesh repair step
every slicer runs on import closes those.
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
    "build_plate",
]


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
    """One printed object: a closed prism with its own filament."""

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
    only emitted where the mask meets a different region or the outside, so the
    mesh carries no internal walls.

    The faces are wound in pixel space, where ``y`` grows downwards.  Placing the
    picture upright on the build plate needs ``y`` mirrored, and a mirror inverts
    every normal, so the winding is reversed at the very end to compensate.
    """
    if z_top <= z_bottom:
        raise ValueError("z_top must be greater than z_bottom")
    height, width = mask.shape
    rects = _merge_stacked(rectangles(mask))

    vertices: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []

    def quad(p0, p1, p2, p3) -> None:
        base = len(vertices)
        vertices.extend((p0, p1, p2, p3))
        triangles.append((base, base + 1, base + 2))
        triangles.append((base, base + 2, base + 3))

    for x, y, w, h in rects:
        x0, x1 = float(x), float(x + w)
        y0, y1 = float(y), float(y + h)
        # top face, normal +Z
        quad((x0, y0, z_top), (x1, y0, z_top), (x1, y1, z_top), (x0, y1, z_top))
        # bottom face, normal -Z
        quad((x0, y1, z_bottom), (x1, y1, z_bottom), (x1, y0, z_bottom), (x0, y0, z_bottom))

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
            quad((a, plane, z_bottom), (b, plane, z_bottom), (b, plane, z_top), (a, plane, z_top))

    for y in np.flatnonzero(south.any(axis=1)):
        plane = float(y) + 1.0
        for x0, x1 in _runs(south[y]):
            a, b = float(x0), float(x1)
            # normal +Y
            quad((b, plane, z_bottom), (a, plane, z_bottom), (a, plane, z_top), (b, plane, z_top))

    # The side walls run down the picture, so their runs are gathered per column
    # and merged vertically — that keeps them aligned with the top and bottom
    # faces instead of leaving a staircase of one-pixel steps.
    for x in np.flatnonzero(west.any(axis=0)):
        plane = float(x)
        for y0, y1 in _runs(west[:, x]):
            ya, yb = float(y0), float(y1)
            # normal -X
            quad((plane, yb, z_bottom), (plane, ya, z_bottom), (plane, ya, z_top), (plane, yb, z_top))

    for x in np.flatnonzero(east.any(axis=0)):
        plane = float(x) + 1.0
        for y0, y1 in _runs(east[:, x]):
            ya, yb = float(y0), float(y1)
            # normal +X
            quad((plane, ya, z_bottom), (plane, yb, z_bottom), (plane, yb, z_top), (plane, ya, z_top))

    points = np.array(vertices, dtype=np.float64).reshape(-1, 3)
    faces = np.array(triangles, dtype=np.int64).reshape(-1, 3)
    points, faces = stitch_t_junctions(points, faces)
    points[:, 0] *= pixel_mm
    points[:, 1] *= pixel_mm
    if flip_y:
        points[:, 1] = height * pixel_mm - points[:, 1]
        faces = faces[:, ::-1]
    return points, faces


def stitch_t_junctions(
    points: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Weld coincident vertices and split every edge a third vertex sits on.

    A rectangle decomposition inevitably leaves T-junctions: a long edge of one
    rectangle meets the corner of two neighbours.  A slicer sees the long edge as
    an open boundary there.  Every face here is axis-aligned in pixel space, so
    "this vertex lies strictly inside that edge" is an exact test and the repair
    is a fan re-triangulation.  Afterwards each part is watertight.
    """
    welded, inverse = np.unique(np.asarray(points, dtype=np.float64), axis=0, return_inverse=True)
    welded_int = np.rint(welded).astype(np.int64)
    remapped = inverse[np.asarray(faces, dtype=np.int64)]

    lines: dict[tuple[str, int, int], list[int]] = {}
    for index, (x, y, z) in enumerate(welded_int.tolist()):
        lines.setdefault(("x", y, z), []).append(index)
        lines.setdefault(("y", x, z), []).append(index)
        lines.setdefault(("z", x, y), []).append(index)

    cache: dict[tuple[int, int], list[int]] = {}

    def between(a: int, b: int) -> list[int]:
        """Welded vertices strictly inside the edge ``a→b``, in travel order.

        The cache always stores the ascending order, because the same edge is
        asked for from both of its faces and only one of them may travel
        downwards.
        """
        key = (a, b) if a <= b else (b, a)
        cached = cache.get(key)
        if cached is None:
            pa = welded_int[a]
            pb = welded_int[b]
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
                candidates = lines.get((names[axis], others[0], others[1]), [])
                low = int(min(pa[axis], pb[axis]))
                high = int(max(pa[axis], pb[axis]))
                values = welded_int[candidates, axis]
                found = [int(candidates[i]) for i in np.flatnonzero((values > low) & (values < high))]
                found.sort(key=lambda index: int(welded_int[index, axis]))
                cached = found
            cache[key] = cached
        return list(reversed(cached)) if a > b else cached

    out: list[tuple[int, int, int]] = []
    for triangle in remapped.tolist():
        polygon: list[int] = []
        for i in range(3):
            a, b = int(triangle[i]), int(triangle[(i + 1) % 3])
            polygon.append(a)
            polygon.extend(between(a, b))
        for k in range(1, len(polygon) - 1):
            out.append((polygon[0], polygon[k], polygon[k + 1]))

    return welded, np.array(out, dtype=np.int64).reshape(-1, 3)



# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def build_plate(
    result: MatchResult,
    settings: PlateSettings | None = None,
    *,
    base_label: str = "底板",
) -> PlateModel:
    """Assemble the base slab plus one stacked prism per colour."""
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

    z_base_top = config.base_thickness_mm
    z_colour_top = config.base_thickness_mm + config.colour_thickness_mm

    parts: list[PlatePart] = []

    # --- the base plate: the picture's silhouette -----------------------------
    silhouette = result.indices >= 0
    if not silhouette.any():
        raise ValueError("图片没有可见像素，无法生成模型。")
    vertices, triangles = prism_mesh(
        silhouette, z_bottom=0.0, z_top=z_base_top, pixel_mm=pixel
    )
    base_entry = result.palette[base_index]
    parts.append(
        PlatePart(
            name=base_label,
            color_hex=base_entry.color_hex,
            palette_index=base_index,
            extruder=base_index + 1,
            vertices=vertices,
            triangles=triangles,
            region_pixels=int(np.count_nonzero(silhouette)),
            z_bottom_mm=0.0,
            z_top_mm=z_base_top,
        )
    )

    # --- one stacked prism per colour ----------------------------------------
    for index, entry in enumerate(result.palette):
        mask = result.indices == index
        count = int(np.count_nonzero(mask))
        if not count:
            continue
        vertices, triangles = prism_mesh(
            mask, z_bottom=z_base_top, z_top=z_colour_top, pixel_mm=pixel
        )
        parts.append(
            PlatePart(
                name=_part_name(entry, index),
                color_hex=entry.color_hex,
                palette_index=index,
                extruder=index + 1,
                vertices=vertices,
                triangles=triangles,
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


def _part_name(entry: PaletteEntry, index: int) -> str:
    label = entry.label
    if len(label) > 42:
        label = label[:39] + "..."
    return f"{index + 1:02d}_{entry.color_hex}_{label}"

"""Mesh tests: rectangle decomposition, prism winding, and the assembled plate.

The numbers here are geometric identities, not recorded observations — a signed
volume, a projected area, an edge-incidence count — so they hold for any mask
and any size.

Run with::

    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.engines import ENGINE_BAMBU  # noqa: E402
from app.core.image_matching import MatchSettings, build_palette, match_image  # noqa: E402
from app.core.library import Filament, FilamentLibrary  # noqa: E402
from app.core.mixes import MixCatalog  # noqa: E402
from app.mesh.plate import (  # noqa: E402
    PlateSettings,
    _merge_stacked,
    build_plate,
    edge_health,
    plate_mesh,
    prism_mesh,
    rectangles,
    signed_volume,
)

SPOOLS = (
    ("耗材白", "#F2F0EB"),
    ("耗材黑", "#17181C"),
    ("耗材金", "#D9A441"),
    ("耗材红", "#C8342E"),
)


def _library() -> FilamentLibrary:
    return FilamentLibrary(
        [
            Filament(name=name, brand="测试", material_type="PETG HF", color_hex=value)
            for name, value in SPOOLS
        ]
    )


def _quadrant_image(size: int = 20) -> Image.Image:
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    half = size // 2
    for x, y, value in (
        (0, 0, SPOOLS[0][1]),
        (half, 0, SPOOLS[1][1]),
        (0, half, SPOOLS[2][1]),
        (half, half, SPOOLS[3][1]),
    ):
        colour = tuple(int(value.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
        image.paste(Image.new("RGBA", (half, half), colour + (255,)), (x, y))
    return image


def _paint(mask: np.ndarray, rects) -> np.ndarray:
    canvas = np.zeros_like(mask, dtype=np.int64)
    for x, y, w, h in rects:
        canvas[y : y + h, x : x + w] += 1
    return canvas


def _signed_volume(points: np.ndarray, faces: np.ndarray) -> float:
    a = points[faces[:, 0]]
    b = points[faces[:, 1]]
    c = points[faces[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def _edge_health(points: np.ndarray, faces: np.ndarray) -> tuple[int, int]:
    """``(boundary_edges, non_manifold_edges)`` after welding by coordinate."""
    welded, inverse = np.unique(np.round(points, 6), axis=0, return_inverse=True)
    assert len(welded) == len(np.unique(inverse))
    remapped = inverse[faces]
    counts: Counter = Counter()
    for tri in remapped:
        for i in range(3):
            a, b = int(tri[i]), int(tri[(i + 1) % 3])
            counts[(min(a, b), max(a, b))] += 1
    boundary = sum(1 for value in counts.values() if value == 1)
    non_manifold = sum(1 for value in counts.values() if value > 2)
    return boundary, non_manifold


class RectangleTests(unittest.TestCase):
    def test_a_solid_block_becomes_one_rectangle(self):
        mask = np.zeros((7, 9), dtype=bool)
        mask[1:6, 2:8] = True
        self.assertEqual(rectangles(mask), [(2, 1, 6, 5)])

    def test_the_decomposition_tiles_the_mask_exactly(self):
        rng = np.random.default_rng(20260506)
        for shape in ((9, 9), (17, 31), (40, 23)):
            with self.subTest(shape=shape):
                mask = rng.random(shape) > 0.45
                canvas = _paint(mask, rectangles(mask))
                self.assertTrue(np.array_equal(canvas == 1, mask))
                self.assertTrue(np.array_equal(canvas > 1, np.zeros(shape, dtype=bool)))

    def test_merging_keeps_the_mask_and_never_adds_rectangles(self):
        rng = np.random.default_rng(7)
        mask = rng.random((24, 24)) > 0.5
        plain = rectangles(mask)
        merged = _merge_stacked(plain)
        self.assertTrue(np.array_equal(_paint(mask, merged) == 1, mask))
        self.assertLessEqual(len(merged), len(plain))

    def test_a_staircase_is_compressed(self):
        mask = np.zeros((20, 20), dtype=bool)
        for row in range(20):
            mask[row, : row + 1] = True
        merged = _merge_stacked(rectangles(mask))
        self.assertEqual(len(merged), 20)
        self.assertTrue(np.array_equal(_paint(mask, merged) == 1, mask))


class PrismTests(unittest.TestCase):
    def test_a_solid_prism_is_watertight_and_outward(self):
        mask = np.zeros((6, 8), dtype=bool)
        mask[:, :] = True
        points, faces = prism_mesh(mask, z_bottom=0.0, z_top=2.0, pixel_mm=0.5)
        self.assertEqual(len(faces), 12)  # one box: 6 quads
        boundary, non_manifold = _edge_health(points, faces)
        self.assertEqual(boundary, 0)
        self.assertEqual(non_manifold, 0)
        volume = _signed_volume(points, faces)
        self.assertAlmostEqual(volume, 8 * 6 * 0.5 * 0.5 * 2.0, places=9)
        self.assertGreater(volume, 0.0)

    def test_projected_top_area_equals_the_mask_area(self):
        rng = np.random.default_rng(99)
        mask = rng.random((13, 11)) > 0.4
        points, faces = prism_mesh(mask, z_bottom=0.0, z_top=1.0, pixel_mm=0.25)
        top = faces[np.all(np.isclose(points[faces, 2], 1.0), axis=1)]
        p = points[top]
        area = 0.5 * np.abs(
            (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
            - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1])
        ).sum()
        self.assertAlmostEqual(area, int(np.count_nonzero(mask)) * 0.25 * 0.25, places=9)

    def test_the_picture_ends_up_upright(self):
        mask = np.zeros((4, 4), dtype=bool)
        mask[0, :] = True  # the top row of the picture
        points, faces = prism_mesh(mask, z_bottom=0.0, z_top=1.0, pixel_mm=1.0, flip_y=True)
        self.assertAlmostEqual(float(points[:, 1].max()), 4.0, places=9)
        self.assertAlmostEqual(float(points[:, 1].min()), 3.0, places=9)

    def test_no_flip_keeps_pixel_order(self):
        mask = np.zeros((4, 4), dtype=bool)
        mask[0, :] = True
        points, _faces = prism_mesh(mask, z_bottom=0.0, z_top=1.0, pixel_mm=1.0, flip_y=False)
        self.assertAlmostEqual(float(points[:, 1].min()), 0.0, places=9)
        self.assertAlmostEqual(float(points[:, 1].max()), 1.0, places=9)

    def test_a_scattered_mask_stays_closed(self):
        rng = np.random.default_rng(3)
        mask = rng.random((16, 16)) > 0.5
        points, faces = prism_mesh(mask, z_bottom=0.0, z_top=0.6, pixel_mm=0.4)
        boundary, _non_manifold = _edge_health(points, faces)
        self.assertEqual(boundary, 0, "a prism must have no open edges")

    def test_outward_winding_survives_enclosed_holes(self):
        mask = np.ones((12, 12), dtype=bool)
        mask[4:8, 4:8] = False
        points, faces = prism_mesh(mask, z_bottom=0.0, z_top=1.0, pixel_mm=1.0)
        volume = _signed_volume(points, faces)
        self.assertAlmostEqual(volume, (12 * 12 - 16) * 1.0, places=9)
        self.assertGreater(volume, 0.0)

    def test_a_thinner_than_zero_prism_is_refused(self):
        with self.assertRaises(ValueError):
            prism_mesh(np.ones((2, 2), dtype=bool), z_bottom=1.0, z_top=1.0, pixel_mm=1.0)


class PlateTests(unittest.TestCase):
    def setUp(self):
        self.library = _library()
        self.catalog = MixCatalog(self.library.filaments, engine=ENGINE_BAMBU).build()
        self.palette = build_palette(self.library, self.catalog)
        self.result = match_image(_quadrant_image(20), self.palette, MatchSettings(max_colours=8))

    def test_parts_are_the_base_plus_every_colour(self):
        plate = build_plate(self.result, PlateSettings(target_width_mm=100.0))
        self.assertEqual(len(plate.parts), 1 + len(self.result.palette))
        self.assertEqual(plate.parts[0].name, "底板")
        self.assertEqual(plate.extruder_count, len(self.result.palette))
        self.assertEqual(plate.parts[0].z_bottom_mm, 0.0)
        for part in plate.parts[1:]:
            self.assertAlmostEqual(part.z_bottom_mm, plate.base_thickness_mm, places=9)
            self.assertAlmostEqual(part.z_top_mm, plate.total_thickness_mm, places=9)

    def test_the_base_uses_the_largest_colour_by_default(self):
        plate = build_plate(self.result, PlateSettings())
        self.assertEqual(plate.parts[0].extruder, int(np.argmax(self.result.counts)) + 1)

    def test_an_explicit_base_colour_is_honoured(self):
        plate = build_plate(self.result, PlateSettings(base_index=2))
        self.assertEqual(plate.parts[0].extruder, 3)

    def test_geometry_scales_with_the_target_width(self):
        narrow = build_plate(self.result, PlateSettings(target_width_mm=80.0))
        wide = build_plate(self.result, PlateSettings(target_width_mm=160.0))
        self.assertAlmostEqual(narrow.width_mm, 80.0, places=9)
        self.assertAlmostEqual(wide.width_mm, 160.0, places=9)
        self.assertAlmostEqual(wide.pixel_size_mm, narrow.pixel_size_mm * 2.0, places=9)
        # Twice the width means four times the area at the same thicknesses.
        self.assertAlmostEqual(wide.volume_mm3, narrow.volume_mm3 * 4.0, places=6)

    def test_the_footprint_is_capped(self):
        plate = build_plate(self.result, PlateSettings(target_width_mm=100.0, max_footprint_mm=30.0))
        self.assertLessEqual(plate.depth_mm, 30.0 + 1e-9)
        self.assertTrue(any("打印范围" in note for note in plate.notes))

    def test_a_fine_print_resolution_is_reported(self):
        # ``PlateSettings.clamped`` floors the target width at 10 mm, so ask for a
        # coarser nozzle instead of a narrower plate.
        plate = build_plate(
            self.result,
            PlateSettings(target_width_mm=10.0, min_pixel_mm=0.9),
        )
        self.assertTrue(any("喷嘴" in note for note in plate.notes), plate.notes)

    def test_the_whole_plate_is_one_closed_outward_solid(self):
        """The plate is a single manifold solid, not a stack of closed prisms.

        A part on its own is *not* closed any more — it is a slice of the solid and
        is open exactly where it meets its neighbours.  What has to hold is that
        welding the parts back together (which is what an importing slicer does
        before it counts anything) gives a closed surface with no interior walls.
        """
        plate = build_plate(self.result, PlateSettings(target_width_mm=100.0))
        points, faces, extruders = plate_mesh(plate)
        self.assertEqual(len(faces), plate.triangle_count)
        self.assertEqual(len(extruders), len(faces))

        boundary, non_manifold = edge_health(points, faces)
        self.assertEqual(boundary, 0, "组装后不该有开放边")
        self.assertEqual(non_manifold, 0, "组装后不该有非流形边")
        self.assertGreater(signed_volume(points, faces), 0.0)

    def test_the_assembled_volume_is_the_printed_volume(self):
        """Volume = printed pixels x cell area x total thickness, exactly."""
        plate = build_plate(self.result, PlateSettings(target_width_mm=100.0))
        points, faces, _ = plate_mesh(plate)
        expected = (
            int(np.count_nonzero(self.result.indices >= 0))
            * plate.pixel_size_mm**2
            * plate.total_thickness_mm
        )
        self.assertAlmostEqual(signed_volume(points, faces), expected, places=6)

    def test_no_face_is_a_sliver(self):
        """No zero-area triangle, and no edge lying across another vertex.

        A fan over a T-junction-split polygon emits slivers whose edges run along
        the polygon's own boundary; they inflate both the open-edge and the
        non-manifold-edge count and are what Bambu Studio reported.
        """
        plate = build_plate(self.result, PlateSettings(target_width_mm=100.0))
        points, faces, _ = plate_mesh(plate)
        corners = points[faces]
        areas = np.linalg.norm(
            np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1
        )
        self.assertTrue(bool((areas > 1e-9).all()), "出现了零面积三角形")

        # Every edge that the triangulation produced must be free of other
        # vertices: welding keeps identical positions, so a vertex sitting inside
        # an edge shows up as an edge that is not an edge of any single triangle.
        welded, inverse = np.unique(points, axis=0, return_inverse=True)
        mapped = np.asarray(inverse).reshape(-1)[faces]
        edges: set[tuple[int, int]] = set()
        for a, b, c in mapped.tolist():
            for u, v in ((a, b), (b, c), (c, a)):
                edges.add((u, v) if u < v else (v, u))
        for u, v in edges:
            start = welded[u]
            end = welded[v]
            span = end - start
            length = float(np.linalg.norm(span))
            if length <= 1e-9:
                continue
            relative = (welded - start) @ span / (length * length)
            offset = np.linalg.norm(
                (welded - start) - np.outer(relative, span), axis=1
            )
            inside = (relative > 1e-9) & (relative < 1.0 - 1e-9) & (offset < 1e-6)
            self.assertEqual(
                int(np.count_nonzero(inside)), 0, f"边 {u}->{v} 上还压着别的顶点"
            )

    def test_region_pixels_cover_the_printed_pixels(self):
        plate = build_plate(self.result, PlateSettings())
        # parts[0] is the base plate covering the whole silhouette; the colour
        # prisms on top of it must tile exactly the printed pixels.
        self.assertEqual(
            sum(part.region_pixels for part in plate.parts[1:]),
            int(self.result.counts.sum()),
        )
        self.assertEqual(
            plate.parts[0].region_pixels,
            int(np.count_nonzero(self.result.indices >= 0)),
        )

    def test_stats_describe_the_plate(self):
        plate = build_plate(self.result, PlateSettings(target_width_mm=100.0))
        stats = plate.stats()
        self.assertEqual(stats["parts"], 1 + len(self.result.palette))
        self.assertEqual(stats["extruders"], len(self.result.palette))
        self.assertGreater(stats["triangles"], 0)
        self.assertAlmostEqual(stats["width_mm"], 100.0, places=3)

    def test_an_empty_result_is_refused(self):
        empty = match_image(_quadrant_image(8), self.palette, MatchSettings(max_colours=8))
        empty.palette = []
        empty.indices = np.full_like(empty.indices, -1)
        with self.assertRaises(ValueError):
            build_plate(empty)


if __name__ == "__main__":
    unittest.main()

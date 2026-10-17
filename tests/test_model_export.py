"""模型配色的三个出口：摊成板、写 GLB、筛选耗材（1.6.52）。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.core import glb, model_export
from app.core.glb import GlbModel, GlbPrimitive
from app.core.image_matching import MatchSettings, PaletteEntry
from app.core.model_export import MM_PER_UNIT, glb_bytes, plate_from_model, write_glb
from app.core.model_matching import chart_image, match_model


def box(low, high, colour):
    """一个轴对齐的长方体，顶点 8 个、三角面 12 个。"""
    x0, y0, z0 = low
    x1, y1, z1 = high
    corners = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float64,
    )
    quads = [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]
    triangles = []
    for a, b, c, d in quads:
        triangles.append([a, b, c])
        triangles.append([a, c, d])
    return GlbPrimitive(
        name="box",
        material="m",
        positions=corners,
        normals=np.zeros_like(corners),
        triangles=np.array(triangles, dtype=np.int64),
        colour=colour,
    )


def two_boxes() -> GlbModel:
    first = box((0.0, 0.0, 0.0), (0.01, 0.02, 0.03), (255, 0, 0))
    second = box((0.02, 0.0, 0.0), (0.03, 0.02, 0.03), (0, 0, 255))
    low = np.array([0.0, 0.0, 0.0])
    high = np.array([0.03, 0.02, 0.03])
    return GlbModel(name="两只盒子", primitives=[first, second], low=low, high=high)


def entry(color_hex: str, rgb: tuple[int, int, int]) -> PaletteEntry:
    return PaletteEntry(
        key=color_hex,
        color_hex=color_hex,
        rgb=rgb,
        lab=(0.0, 0.0, 0.0),
        label=color_hex,
        kind="filament",
        filament_ids=(color_hex,),
    )


class PlateTests(unittest.TestCase):
    def test_a_part_per_primitive_in_millimetres(self) -> None:
        plate = plate_from_model(two_boxes(), [(255, 0, 0), (0, 0, 255)])
        self.assertEqual(len(plate.parts), 2)
        self.assertAlmostEqual(plate.width_mm, 30.0, places=3)
        self.assertAlmostEqual(plate.depth_mm, 30.0, places=3)
        self.assertEqual(plate.parts[0].color_hex, "#FF0000")
        self.assertEqual(plate.parts[1].color_hex, "#0000FF")
        # glTF 的 Y 朝上、单位是米：0.02 的 y 变成 20 mm 的高度
        heights = [part.z_top_mm - part.z_bottom_mm for part in plate.parts]
        self.assertAlmostEqual(max(heights), 20.0, places=3)
        self.assertAlmostEqual(min(plate.parts[0].vertices[:, 2]), 0.0, places=6)

    def test_the_bottom_centre_lands_on_the_origin(self) -> None:
        plate = plate_from_model(two_boxes(), [(255, 0, 0), (0, 0, 255)])
        points = np.vstack([part.vertices for part in plate.parts])
        self.assertAlmostEqual(float(points[:, 0].min() + points[:, 0].max()) / 2.0, 0.0, places=6)
        self.assertAlmostEqual(float(points[:, 1].min() + points[:, 1].max()) / 2.0, 0.0, places=6)
        self.assertAlmostEqual(float(points[:, 2].min()), 0.0, places=6)

    def test_slots_and_palette_indices_travel_with_the_parts(self) -> None:
        plate = plate_from_model(
            two_boxes(),
            [(255, 0, 0), (0, 0, 255)],
            slots=[1, 2],
            palette_indices=[5, 7],
        )
        self.assertEqual([part.extruder for part in plate.parts], [1, 2])
        self.assertEqual([part.palette_index for part in plate.parts], [5, 7])

    def test_an_untouched_glb_travels_through_millimetres(self) -> None:
        plate = plate_from_model(two_boxes(), [(1, 2, 3), (4, 5, 6)])
        span = plate.parts[0].z_top_mm - plate.parts[0].z_bottom_mm
        self.assertAlmostEqual(span, 0.02 * MM_PER_UNIT, places=3)

    def test_a_model_without_triangles_is_refused(self) -> None:
        empty = GlbModel(name="空的", primitives=[], low=np.zeros(3), high=np.zeros(3))
        with self.assertRaises(ValueError):
            plate_from_model(empty, [])


class GlbRoundTripTests(unittest.TestCase):
    def test_the_exported_bytes_are_a_glb(self) -> None:
        payload = glb_bytes(two_boxes(), [(255, 0, 0), (0, 0, 255)])
        self.assertEqual(payload[:4], b"glTF")
        self.assertEqual(len(payload) % 4, 0)

    def test_reloading_keeps_the_triangles_and_the_colours(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "out.glb"
            path.write_bytes(glb_bytes(two_boxes(), [(255, 0, 0), (0, 0, 255)]))
            again = glb.load_model(path)
        self.assertEqual(again.triangle_count, 24)
        self.assertEqual(len(again.primitives), 2)
        self.assertEqual(sorted(item.colour for item in again.primitives), [(0, 0, 255), (255, 0, 0)])

    def test_writing_leaves_no_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = write_glb(two_boxes(), [(1, 2, 3), (4, 5, 6)], Path(folder) / "model")
            self.assertTrue(path.exists())
            self.assertEqual(path.suffix, ".glb")
            self.assertEqual(sorted(Path(folder).rglob("*.tmp")), [])


class MatchModelFilterTests(unittest.TestCase):
    def _palette(self) -> list[PaletteEntry]:
        return [entry("#FF0000", (255, 0, 0)), entry("#0000FF", (0, 0, 255))]

    def test_a_required_spool_that_does_not_exist_is_refused(self) -> None:
        colours = [glb.ModelColour(rgb=(255, 0, 0), area=1.0, primitives=1)]
        with self.assertRaises(ValueError):
            match_model(
                colours,
                self._palette(),
                MatchSettings(max_colours=4, max_dimension=64, max_spools=2),
                require={"nope"},
            )

    def test_requiring_the_red_spool_paints_everything_with_it(self) -> None:
        palette = [
            PaletteEntry(
                key="#FF0000",
                color_hex="#FF0000",
                rgb=(255, 0, 0),
                lab=(53.0, 80.0, 67.0),
                label="红",
                kind="filament",
                filament_ids=("red",),
            ),
            PaletteEntry(
                key="#0000FF",
                color_hex="#0000FF",
                rgb=(0, 0, 255),
                lab=(32.0, 79.0, -108.0),
                label="蓝",
                kind="filament",
                filament_ids=("blue",),
            ),
        ]
        colours = [
            glb.ModelColour(rgb=(200, 40, 40), area=2.0, primitives=1),
            glb.ModelColour(rgb=(60, 60, 200), area=1.0, primitives=1),
        ]
        found = match_model(
            colours,
            palette,
            MatchSettings(max_colours=3, max_dimension=64, max_spools=2),
            require={palette[0].filament_ids[0]},
        )
        # 「必须有」是硬规矩：红那卷一定出现在用到的耗材里
        self.assertTrue(found.colours)
        used = {fid for entry_used in found.result.palette for fid in entry_used.filament_ids}
        self.assertIn("red", used)

    def test_the_chart_keeps_the_band_order(self) -> None:
        colours = [
            glb.ModelColour(rgb=(255, 0, 0), area=3.0, primitives=1),
            glb.ModelColour(rgb=(0, 0, 255), area=1.0, primitives=1),
        ]
        picture, ranges = chart_image(colours)
        self.assertEqual(picture.size, (model_export_size(), 9))
        self.assertEqual(len(ranges), 2)
        self.assertLess(ranges[0][0], ranges[1][0])
        self.assertEqual(picture.getpixel((ranges[0][0], 0)), (255, 0, 0))


def model_export_size() -> int:
    from app.core.model_matching import CHART_WIDTH

    return CHART_WIDTH


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

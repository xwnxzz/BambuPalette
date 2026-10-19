"""3MF 读入 + GLB 里「这次怎么配的」往返（1.6.55，m25568 第 3 / 4 条）。"""

from __future__ import annotations

import struct
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from app.core.glb import GlbModel, GlbPrimitive, load_model, model_colours
from app.core.model_export import glb_bytes
from app.core.threemf_import import load_3mf_model


def _box_3mf(
    path: Path,
    *,
    unit: str = "millimeter",
    colours: tuple[str, ...] = ("#C8342E", "#2FA84F"),
    broken_triangle: bool = False,
) -> Path:
    """两个各 12 个三角形的方块，各用一种颜色；可选塞一个越界三角形。"""
    resources: list[str] = []
    items: list[str] = []
    for position, _colour in enumerate(colours):
        verts = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    verts.append(
                        (
                            position * 20.0 + sx * 7.5,
                            sy * 7.5,
                            sz * 7.5,
                        )
                    )
        faces = [
            (0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5),
            (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6),
            (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3),
        ]
        if broken_triangle:
            faces.append((0, 1, 99))
        resources.append(
            f'<object id="{position + 1}" type="model" name="part{position + 1}" '
            f'pid="1" pindex="{position}"><mesh><vertices>'
            + "".join(f'<vertex x="{x}" y="{y}" z="{z}"/>' for x, y, z in verts)
            + "</vertices><triangles>"
            + "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in faces)
            + "</triangles></mesh></object>"
        )
        items.append(f'<item objectid="{position + 1}"/>')
    materials = (
        '<basematerials id="1">'
        + "".join(f'<base name="c{n}" displaycolor="{c}"/>' for n, c in enumerate(colours))
        + "</basematerials>"
    )
    payload = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<model unit="{unit}" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">'
        f"<resources>{materials}{''.join(resources)}</resources>"
        f"<build>{''.join(items)}</build></model>"
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("3D/3dmodel.model", payload)
    return path


class ThreeMfImportTests(unittest.TestCase):
    def test_reads_two_coloured_parts(self) -> None:
        with TemporaryDirectory() as folder:
            path = _box_3mf(Path(folder) / "two.3mf")
            model = load_3mf_model(path)
        self.assertEqual(len(model.primitives), 2)
        self.assertEqual(model.triangle_count, 24)
        self.assertEqual(model.skipped, ())
        colours = model_colours(model)
        self.assertEqual(len(colours), 2)
        self.assertEqual({item.rgb for item in colours}, {(200, 52, 46), (47, 168, 79)})

    def test_millimetres_become_units(self) -> None:
        # 15 mm 高 → 0.015 单位（1 单位 = 1 米，1.6.52 的约定）
        with TemporaryDirectory() as folder:
            path = _box_3mf(Path(folder) / "mm.3mf")
            model = load_3mf_model(path)
        self.assertAlmostEqual(float(model.size[1]), 0.015, places=6)

    def test_inch_unit_scales(self) -> None:
        with TemporaryDirectory() as folder:
            path = _box_3mf(Path(folder) / "inch.3mf", unit="inch")
            model = load_3mf_model(path)
        self.assertAlmostEqual(float(model.size[1]), 0.381, places=4)

    def test_bad_triangle_is_skipped_with_note(self) -> None:
        with TemporaryDirectory() as folder:
            path = _box_3mf(Path(folder) / "broken.3mf", broken_triangle=True)
            model = load_3mf_model(path)
        self.assertTrue(any("越界" in note for note in model.skipped))
        self.assertEqual(model.triangle_count, 24)

    def test_not_a_zip_is_a_value_error(self) -> None:
        with TemporaryDirectory() as folder:
            path = Path(folder) / "fake.3mf"
            path.write_bytes(b"not a zip at all")
            with self.assertRaises(ValueError):
                load_3mf_model(path)

    def test_missing_check_component(self) -> None:
        with TemporaryDirectory() as folder:
            path = Path(folder) / "empty.3mf"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("3D/3dmodel.model", "<model/>")
            model = load_3mf_model(path)
        self.assertEqual(model.primitives, [])
        self.assertTrue(model.is_empty)


class GlbExtrasRoundTripTests(unittest.TestCase):
    def _model(self) -> GlbModel:
        verts = np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32
        )
        primitives = [
            GlbPrimitive(
                name="a",
                material="#111111",
                positions=verts,
                normals=np.zeros((0, 3), dtype=np.float32),
                triangles=np.asarray([[0, 1, 2]], dtype=np.int64),
                colour=(0x11, 0x11, 0x11),
            ),
            GlbPrimitive(
                name="b",
                material="#222222",
                positions=verts,
                normals=np.zeros((0, 3), dtype=np.float32),
                triangles=np.asarray([[1, 3, 2]], dtype=np.int64),
                colour=(0x22, 0x22, 0x22),
            ),
        ]
        return GlbModel(name="t", primitives=primitives, low=verts.min(axis=0), high=verts.max(axis=0))

    def test_extras_survive_a_round_trip(self) -> None:
        model = self._model()
        extras = {
            "bambupalette": {
                "version": 1,
                "plan": "colour:abc",
                "settings": {"colours": 16, "spools": 3},
                "primitives": [{"index": 0, "original": "#111111", "entry": "colour:abc"}],
            }
        }
        payload = glb_bytes(model, [(0x30, 0x30, 0x30), (0x40, 0x40, 0x40)], extras=extras)
        with TemporaryDirectory() as folder:
            path = Path(folder) / "round.glb"
            path.write_bytes(payload)
            back = load_model(path)
        self.assertIsNotNone(back.saved)
        assert back.saved is not None
        self.assertEqual(back.saved["plan"], "colour:abc")
        self.assertEqual(back.saved["settings"]["colours"], 16)
        self.assertEqual(back.saved["primitives"][0]["original"], "#111111")

    def test_no_extras_means_none(self) -> None:
        model = self._model()
        payload = glb_bytes(model, [(0x30, 0x30, 0x30), (0x40, 0x40, 0x40)])
        with TemporaryDirectory() as folder:
            path = Path(folder) / "plain.glb"
            path.write_bytes(payload)
            back = load_model(path)
        self.assertIsNone(back.saved)

    def test_painted_colours_land_in_materials(self) -> None:
        model = self._model()
        payload = glb_bytes(model, [(0x30, 0x30, 0x30), (0x40, 0x40, 0x40)])
        with TemporaryDirectory() as folder:
            path = Path(folder) / "painted.glb"
            path.write_bytes(payload)
            back = load_model(path)
        self.assertEqual({tuple(item.colour) for item in back.primitives}, {(0x30, 0x30, 0x30), (0x40, 0x40, 0x40)})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

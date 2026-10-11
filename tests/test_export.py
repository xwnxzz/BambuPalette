"""Export tests: the 3MF package layout and the OBJ/MTL pair.

The 3MF assertions follow the structure recorded from a real Bambu Studio
project file (``BambuStudio-02.06.00.51``): OPC plumbing, a single painted
object with the colours on its triangles, and every extruder assignment in
``Metadata/model_settings.config``.

Run with::

    .venv\\Scripts\\python.exe -m unittest tests.test_export -v
"""

from __future__ import annotations

import sys
import unittest
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.image_matching import MatchSettings, build_palette, match_image  # noqa: E402
from app.core.library import Filament, FilamentLibrary  # noqa: E402
from app.mesh.objfile import write_obj  # noqa: E402
from app.mesh.plate import (  # noqa: E402
    PlateSettings,
    build_plate,
    edge_health,
    plate_mesh,
    signed_volume,
)
from app.mesh.threemf import write_3mf  # noqa: E402

SPOOLS = (
    ("耗材白", "#F2F0EB"),
    ("耗材黑", "#17181C"),
    ("耗材金", "#D9A441"),
    ("耗材红", "#C8342E"),
)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


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


class ExportTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.library = _library()
        palette = build_palette(self.library, None, include_mixes=False)
        self.result = match_image(_quadrant_image(), palette, MatchSettings(max_colours=4))
        self.plate = build_plate(self.result, PlateSettings(target_width_mm=100.0))
        self._temp = TemporaryDirectory()
        self.tmp = Path(self._temp.name)

    def tearDown(self) -> None:
        self._temp.cleanup()


class ThreeMfTests(ExportTestCase):
    def _write(self, **kwargs) -> Path:
        return write_3mf(self.plate, self.tmp / "plate", **kwargs)

    def test_the_extension_is_applied_and_the_file_is_a_zip(self):
        path = write_3mf(self.plate, self.tmp / "no-extension")
        self.assertEqual(path.suffix, ".3mf")
        self.assertTrue(zipfile.is_zipfile(path))

    def test_the_required_parts_are_present(self):
        with zipfile.ZipFile(self._write()) as archive:
            names = set(archive.namelist())
        self.assertIn("[Content_Types].xml", names)
        self.assertIn("_rels/.rels", names)
        self.assertIn("3D/3dmodel.model", names)
        self.assertIn("Metadata/model_settings.config", names)

    def test_meshes_are_inline_rather_than_spread_over_part_files(self):
        # Regression guard.  The first implementation copied Bambu Studio's own
        # layout (one 3D/Objects/object_N.model per part, referenced through the
        # 3MF production extension).  Bambu Studio 02.08.02.61 accepted the
        # metadata but discarded every component, so the model imported with no
        # geometry at all.  Core 3MF with inline meshes round-trips correctly.
        with zipfile.ZipFile(self._write()) as archive:
            names = set(archive.namelist())
            root = ET.fromstring(archive.read("3D/3dmodel.model"))
        self.assertFalse(
            [name for name in names if name.startswith("3D/Objects/")],
            f"no external geometry parts are expected, found {sorted(names)}",
        )
        self.assertNotIn("requiredextensions", root.attrib)
        meshes = [item for item in root.iter() if _local(item.tag) == "mesh"]
        self.assertEqual(len(meshes), 1, "the plate is one object, not one per colour")

    def test_content_types_declare_the_model_extension(self):
        with zipfile.ZipFile(self._write()) as archive:
            root = ET.fromstring(archive.read("[Content_Types].xml"))
        defaults = {item.get("Extension"): item.get("ContentType") for item in root}
        self.assertIn("3dmanufacturing-3dmodel+xml", defaults["model"])

    def test_the_relationship_points_at_the_model(self):
        with zipfile.ZipFile(self._write()) as archive:
            root = ET.fromstring(archive.read("_rels/.rels"))
        targets = [item.get("Target") for item in root]
        self.assertIn("/3D/3dmodel.model", targets)

    def test_every_xml_part_is_well_formed(self):
        with zipfile.ZipFile(self._write()) as archive:
            for name in archive.namelist():
                if name.endswith(".model") or name.endswith(".xml"):
                    ET.fromstring(archive.read(name))
                elif name == "Metadata/model_settings.config":
                    ET.fromstring(archive.read(name))

    def test_there_is_no_project_settings_config(self):
        """Bambu Studio must not be handed a partial project config.

        Measured against 02.08.02.61: any partial ``project_settings.config``
        makes ``Plater::load_files`` print 「3mf文件配置无效，仅加载几何数据」 and
        then paint the plate from its own spools.  With the entry absent the
        loader keeps the mesh's colour data and there is no dialog at all, which
        is why the writer stopped emitting it.
        """
        with zipfile.ZipFile(self._write()) as archive:
            self.assertNotIn("Metadata/project_settings.config", archive.namelist())

    def _model(self):
        with zipfile.ZipFile(self._write()) as archive:
            return ET.fromstring(archive.read("3D/3dmodel.model"))

    def _by_extruder(self) -> dict[int, str]:
        """The colour each extruder slot is painted with, from the parts."""
        mapping: dict[int, str] = {}
        for part in self.plate.parts:
            mapping[part.extruder] = part.color_hex
        return mapping

    def test_the_plate_is_a_single_painted_object(self):
        """The user's requirement: ONE plate with colours painted on it.

        Every colour region used to be its own ``<part>``, which Bambu Studio
        showed as a stack of child objects under 混色底板 rather than a plate.
        Now there is one object, one mesh, and the colours live on the triangles.
        """
        root = self._model()
        objects = [item for item in root.iter() if _local(item.tag) == "object"]
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0].get("id"), "1")
        self.assertFalse(
            [item for item in root.iter() if _local(item.tag) == "component"],
            "one inline mesh replaces the per-colour component list",
        )
        build = next(item for item in root.iter() if _local(item.tag) == "build")
        items = [item for item in build if _local(item.tag) == "item"]
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].get("objectid"), "1")

    def test_every_triangle_names_its_filament(self):
        """pid/p1 and Bambu Studio's own paint_color must agree, triangle by triangle."""
        from app.mesh.threemf import BASEMATERIALS_ID, paint_code

        root = self._model()
        triangles = [item for item in root.iter() if _local(item.tag) == "triangle"]
        self.assertEqual(len(triangles), self.plate.triangle_count)

        expected: dict[str, int] = {}
        for part in self.plate.parts:
            code = paint_code(part.extruder)
            expected[code] = expected.get(code, 0) + part.triangle_count

        seen: dict[str, int] = {}
        for triangle in triangles:
            self.assertEqual(triangle.get("pid"), str(BASEMATERIALS_ID))
            self.assertIsNotNone(triangle.get("p1"))
            code = triangle.get("paint_color")
            self.assertIn(code, expected)
            seen[code] = seen.get(code, 0) + 1
        self.assertEqual(seen, expected)

    def test_the_basematerials_table_lists_one_entry_per_filament(self):
        root = self._model()
        tables = [item for item in root.iter() if _local(item.tag) == "basematerials"]
        self.assertEqual(len(tables), 1)
        bases = [item for item in tables[0] if _local(item.tag) == "base"]
        by_extruder = self._by_extruder()
        self.assertEqual(len(bases), len(by_extruder))
        self.assertEqual(sorted(by_extruder), list(range(1, len(by_extruder) + 1)))
        for index, base in enumerate(bases, start=1):
            self.assertEqual(base.get("displaycolor"), f"{by_extruder[index].upper()}FF")

    def test_geometry_round_trips_vertex_for_vertex(self):
        root = self._model()
        mesh = next(item for item in root.iter() if _local(item.tag) == "mesh")
        vertices = [item for item in mesh.iter() if _local(item.tag) == "vertex"]
        triangles = [item for item in mesh.iter() if _local(item.tag) == "triangle"]
        self.assertEqual(len(vertices), sum(len(part.vertices) for part in self.plate.parts))
        self.assertEqual(len(triangles), self.plate.triangle_count)
        for triangle in triangles:
            for key in ("v1", "v2", "v3"):
                self.assertLess(int(triangle.get(key)), len(vertices))

    def test_the_merged_vertices_keep_each_part_contiguous(self):
        """Concatenation must offset indices, not renumber them across parts."""
        root = self._model()
        mesh = next(item for item in root.iter() if _local(item.tag) == "mesh")
        triangles = [item for item in mesh.iter() if _local(item.tag) == "triangle"]
        cursor = 0
        offset = 0
        for part in self.plate.parts:
            window = triangles[cursor : cursor + part.triangle_count]
            self.assertEqual(len(window), part.triangle_count)
            for triangle in window:
                for key in ("v1", "v2", "v3"):
                    index = int(triangle.get(key))
                    self.assertGreaterEqual(index, offset)
                    self.assertLess(index, offset + len(part.vertices))
            cursor += part.triangle_count
            offset += len(part.vertices)

    def test_the_last_written_vertex_is_the_model_vertex_on_the_bed(self):
        root = self._model()
        mesh = next(item for item in root.iter() if _local(item.tag) == "mesh")
        vertices = [item for item in mesh.iter() if _local(item.tag) == "vertex"]
        last = vertices[-1]
        expected = self.plate.parts[-1].vertices[-1]
        self.assertAlmostEqual(float(last.get("x")) - expected[0], (256.0 - self.plate.width_mm) / 2.0, places=4)
        self.assertAlmostEqual(float(last.get("y")) - expected[1], (256.0 - self.plate.depth_mm) / 2.0, places=4)
        self.assertAlmostEqual(float(last.get("z")) - expected[2], 0.0, places=6)

    def test_the_settings_file_describes_one_object_with_one_part(self):
        with zipfile.ZipFile(self._write()) as archive:
            root = ET.fromstring(archive.read("Metadata/model_settings.config"))
        objects = [item for item in root.iter() if _local(item.tag) == "object"]
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0].get("id"), "1")
        parts = [item for item in objects[0].iter() if _local(item.tag) == "part"]
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0].get("subtype"), "normal_part")
        metadata = {
            item.get("key"): item.get("value")
            for item in objects[0]
            if _local(item.tag) == "metadata"
        }
        self.assertEqual(metadata["extruder"], str(self.plate.parts[0].extruder))
        # Bambu writes the face count as a KEYLESS metadata attribute rather
        # than a key/value pair; a real project file carries it the same way.
        face_counts = [
            item.get("face_count")
            for item in objects[0]
            if _local(item.tag) == "metadata" and item.get("key") is None
        ]
        self.assertEqual(face_counts, [str(self.plate.triangle_count)])

    def test_the_mesh_carries_one_base_material_per_colour(self):
        """The colours travel in the mesh now that the project config is gone."""
        root = self._model()
        table = next(
            item for item in root.iter() if _local(item.tag) == "basematerials"
        )
        entries = [item for item in table if _local(item.tag) == "base"]
        self.assertEqual(len(entries), self.plate.extruder_count)
        for part in self.plate.parts:
            entry = entries[part.extruder - 1]
            self.assertEqual(entry.get("displaycolor"), f"{part.color_hex.upper()}FF")

    def test_the_plate_records_one_instance_of_the_object(self):
        with zipfile.ZipFile(self._write()) as archive:
            root = ET.fromstring(archive.read("Metadata/model_settings.config"))
        plate = next(item for item in root.iter() if _local(item.tag) == "plate")
        metadata = {
            item.get("key"): item.get("value") for item in plate if _local(item.tag) == "metadata"
        }
        self.assertEqual(metadata["filament_map_mode"], "Auto For Flush")
        instances = [item for item in plate if _local(item.tag) == "model_instance"]
        self.assertEqual(len(instances), 1)
        instance = {
            item.get("key"): item.get("value") for item in instances[0] if _local(item.tag) == "metadata"
        }
        self.assertEqual(instance["object_id"], "1")

    def test_object_names_are_xml_escaped(self):
        path = write_3mf(self.plate, self.tmp / "escaped", object_name='a & b <c> "d"')
        with zipfile.ZipFile(path) as archive:
            root = ET.fromstring(archive.read("Metadata/model_settings.config"))
        names = [
            item.get("value")
            for item in root.iter()
            if _local(item.tag) == "metadata" and item.get("key") == "name"
        ]
        self.assertIn('a & b <c> "d"', names)

    def test_an_empty_plate_is_refused(self):
        empty = build_plate(self.result, PlateSettings(target_width_mm=100.0))
        empty.parts = []
        with self.assertRaises(ValueError):
            write_3mf(empty, self.tmp / "empty")

    def test_the_object_is_centred_on_the_bed(self):
        path = write_3mf(self.plate, self.tmp / "centred", plate_size_mm=256.0)
        with zipfile.ZipFile(path) as archive:
            root = ET.fromstring(archive.read("3D/3dmodel.model"))
        objects = [item for item in root.iter() if _local(item.tag) == "object"]
        first = next(item for item in objects if item.get("id") == "1")
        vertices = [item for item in first.iter() if _local(item.tag) == "vertex"]
        xs = [float(item.get("x")) for item in vertices]
        ys = [float(item.get("y")) for item in vertices]
        self.assertAlmostEqual(min(xs), (256.0 - self.plate.width_mm) / 2.0, places=4)
        self.assertAlmostEqual(min(ys), (256.0 - self.plate.depth_mm) / 2.0, places=4)
        self.assertAlmostEqual(max(xs) - min(xs), self.plate.width_mm, places=4)


class ObjTests(ExportTestCase):
    def _write(self) -> Path:
        return write_obj(self.plate, self.tmp / "plate")

    def test_the_material_library_is_written_next_to_the_obj(self):
        path = self._write()
        self.assertTrue(path.exists())
        self.assertTrue(path.with_suffix(".mtl").exists())

    def test_the_obj_references_the_material_library(self):
        text = self._write().read_text(encoding="utf-8")
        self.assertIn("mtllib plate.mtl", text)

    def test_vertex_and_face_counts_match_the_plate(self):
        lines = self._write().read_text(encoding="utf-8").splitlines()
        vertices = [line for line in lines if line.startswith("v ")]
        faces = [line for line in lines if line.startswith("f ")]
        self.assertEqual(len(vertices), self.plate.vertex_count)
        self.assertEqual(len(faces), self.plate.triangle_count)

    def test_face_indices_stay_inside_the_vertex_list(self):
        lines = self._write().read_text(encoding="utf-8").splitlines()
        total = sum(1 for line in lines if line.startswith("v "))
        for line in lines:
            if not line.startswith("f "):
                continue
            for token in line.split()[1:]:
                index = int(token.split("/")[0])
                self.assertGreaterEqual(index, 1)
                self.assertLessEqual(index, total)

    def test_every_part_has_a_group_and_a_material(self):
        lines = self._write().read_text(encoding="utf-8").splitlines()
        groups = [line for line in lines if line.startswith("g ")]
        uses = [line for line in lines if line.startswith("usemtl ")]
        self.assertEqual(len(groups), len(self.plate.parts))
        self.assertEqual(len(uses), len(self.plate.parts))

    def test_the_material_colours_match_the_parts(self):
        mtl = self._write().with_suffix(".mtl").read_text(encoding="utf-8").splitlines()
        declared = [line.split(None, 1)[1] for line in mtl if line.startswith("newmtl ")]
        self.assertEqual(len(declared), len(self.plate.parts))
        diffuses = [line for line in mtl if line.startswith("Kd ")]
        self.assertEqual(len(diffuses), len(self.plate.parts))
        for line, part in zip(diffuses, self.plate.parts):
            values = [float(token) for token in line.split()[1:]]
            expected = [int(part.color_hex.lstrip("#")[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]
            for actual, wanted in zip(values, expected):
                self.assertAlmostEqual(actual, wanted, places=3)

    def test_the_obj_and_the_3mf_describe_the_same_solid(self):
        obj_lines = self._write().read_text(encoding="utf-8").splitlines()
        obj_vertices = [line for line in obj_lines if line.startswith("v ")]
        obj_faces = [line for line in obj_lines if line.startswith("f ")]
        path = write_3mf(self.plate, self.tmp / "same")
        with zipfile.ZipFile(path) as archive:
            root = ET.fromstring(archive.read("3D/3dmodel.model"))
        total_vertices = sum(1 for item in root.iter() if _local(item.tag) == "vertex")
        total_faces = sum(1 for item in root.iter() if _local(item.tag) == "triangle")
        self.assertEqual(len(obj_vertices), total_vertices)
        self.assertEqual(len(obj_faces), total_faces)

    def test_an_empty_plate_is_refused(self):
        empty = build_plate(self.result, PlateSettings(target_width_mm=100.0))
        empty.parts = []
        with self.assertRaises(ValueError):
            write_obj(empty, self.tmp / "empty")


class GeometrySanityTests(ExportTestCase):
    def test_the_exported_solids_are_closed_and_positive(self):
        """The exported plate is one closed, outward-facing solid.

        The parts are slices of a single solid, so a part on its own is open where
        it meets its neighbour — the export has to be checked assembled, which is
        also how an importing slicer sees it.
        """
        points, faces, _extruders = plate_mesh(self.plate)
        open_edges, non_manifold = edge_health(points, faces)
        self.assertEqual(open_edges, 0)
        self.assertEqual(non_manifold, 0)
        self.assertGreater(signed_volume(points, faces), 0.0)

    def test_the_printed_volume_matches_the_reported_volume(self):
        cell = self.plate.pixel_size_mm**2
        total = 0.0
        for part in self.plate.parts:
            a = part.vertices[part.triangles[:, 0]]
            b = part.vertices[part.triangles[:, 1]]
            c = part.vertices[part.triangles[:, 2]]
            total += float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)
        self.assertAlmostEqual(total, self.plate.volume_mm3, places=3)
        self.assertGreater(cell, 0.0)


if __name__ == "__main__":
    unittest.main()

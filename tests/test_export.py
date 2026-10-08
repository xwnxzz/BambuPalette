"""Export tests: the 3MF package layout and the OBJ/MTL pair.

The 3MF assertions follow the structure recorded from a real Bambu Studio
project file (``BambuStudio-02.06.00.51``): OPC plumbing, a separate geometry
file per part pulled in through ``<components>``, and every extruder assignment
in ``Metadata/model_settings.config``.

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
from app.mesh.plate import PlateSettings, build_plate  # noqa: E402
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
        self.assertEqual(len(meshes), len(self.plate.parts))

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
                if name.endswith(".model") or name.endswith(".config") or name.endswith(".xml"):
                    ET.fromstring(archive.read(name))

    def _model(self):
        with zipfile.ZipFile(self._write()) as archive:
            return ET.fromstring(archive.read("3D/3dmodel.model"))

    def test_the_assembly_pulls_in_one_component_per_part(self):
        root = self._model()
        objects = [item for item in root.iter() if _local(item.tag) == "object"]
        self.assertEqual(len(objects), len(self.plate.parts) + 1)
        assembly = objects[-1]
        self.assertEqual(assembly.get("id"), str(len(self.plate.parts) + 1))
        components = [item for item in assembly.iter() if _local(item.tag) == "component"]
        self.assertEqual(len(components), len(self.plate.parts))
        for index, component in enumerate(components, start=1):
            self.assertEqual(component.get("objectid"), str(index))
            self.assertIsNone(component.get("p:path"))
        build = next(item for item in root.iter() if _local(item.tag) == "build")
        items = [item for item in build if _local(item.tag) == "item"]
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].get("objectid"), assembly.get("id"))

    def test_every_component_resolves_to_an_inline_object(self):
        """The invariant Bambu Studio actually choked on: ids must resolve."""
        root = self._model()
        objects = [item for item in root.iter() if _local(item.tag) == "object"]
        mesh_objects = {
            item.get("id"): item
            for item in objects
            if any(_local(child.tag) == "mesh" for child in item)
        }
        self.assertEqual(len(mesh_objects), len(self.plate.parts))
        assembly = objects[-1]
        for component in (item for item in assembly.iter() if _local(item.tag) == "component"):
            self.assertIn(component.get("objectid"), mesh_objects)

    def test_geometry_round_trips_vertex_for_vertex(self):
        root = self._model()
        objects = [item for item in root.iter() if _local(item.tag) == "object"]
        mesh_objects = [item for item in objects if any(_local(c.tag) == "mesh" for c in item)]
        self.assertEqual(len(mesh_objects), len(self.plate.parts))
        for index, (element, part) in enumerate(zip(mesh_objects, self.plate.parts), start=1):
            with self.subTest(part=part.name):
                self.assertEqual(element.get("id"), str(index))
                vertices = [item for item in element.iter() if _local(item.tag) == "vertex"]
                triangles = [item for item in element.iter() if _local(item.tag) == "triangle"]
                self.assertEqual(len(vertices), len(part.vertices))
                self.assertEqual(len(triangles), part.triangle_count)
                for triangle in triangles:
                    for key in ("v1", "v2", "v3"):
                        self.assertLess(int(triangle.get(key)), len(vertices))

    def test_the_last_written_vertex_is_the_model_vertex_on_the_bed(self):
        root = self._model()
        objects = [item for item in root.iter() if _local(item.tag) == "object"]
        first = next(item for item in objects if item.get("id") == "1")
        vertices = [item for item in first.iter() if _local(item.tag) == "vertex"]
        last = vertices[-1]
        expected = self.plate.parts[0].vertices[-1]
        self.assertAlmostEqual(float(last.get("x")) - expected[0], (256.0 - self.plate.width_mm) / 2.0, places=4)
        self.assertAlmostEqual(float(last.get("y")) - expected[1], (256.0 - self.plate.depth_mm) / 2.0, places=4)
        self.assertAlmostEqual(float(last.get("z")) - expected[2], 0.0, places=6)

    def test_every_part_gets_its_own_extruder_slot(self):
        with zipfile.ZipFile(self._write()) as archive:
            root = ET.fromstring(archive.read("Metadata/model_settings.config"))
        parts = [item for item in root.iter() if _local(item.tag) == "part"]
        self.assertEqual(len(parts), len(self.plate.parts))
        for index, element in enumerate(parts, start=1):
            self.assertEqual(element.get("id"), str(index))
            self.assertEqual(element.get("subtype"), "normal_part")
            metadata = {
                item.get("key"): item.get("value")
                for item in element
                if _local(item.tag) == "metadata"
            }
            self.assertEqual(metadata["extruder"], str(self.plate.parts[index - 1].extruder))
            self.assertEqual(metadata["name"], self.plate.parts[index - 1].name)

    def test_the_plate_records_one_instance_of_the_assembly(self):
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
        self.assertEqual(instance["object_id"], str(len(self.plate.parts) + 1))

    def test_object_names_are_xml_escaped(self):
        plate = build_plate(self.result, PlateSettings(target_width_mm=100.0))
        plate.parts[0].name = 'a & b <c> "d"'
        path = write_3mf(plate, self.tmp / "escaped")
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
        for part in self.plate.parts:
            welded, inverse = np.unique(np.round(part.vertices, 6), axis=0, return_inverse=True)
            faces = inverse[part.triangles]
            counts: dict[tuple[int, int], int] = {}
            for triangle in faces:
                for i in range(3):
                    a, b = int(triangle[i]), int(triangle[(i + 1) % 3])
                    key = (min(a, b), max(a, b))
                    counts[key] = counts.get(key, 0) + 1
            open_edges = sum(1 for count in counts.values() if count == 1)
            self.assertEqual(open_edges, 0, part.name)
            a = part.vertices[part.triangles[:, 0]]
            b = part.vertices[part.triangles[:, 1]]
            c = part.vertices[part.triangles[:, 2]]
            volume = float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)
            self.assertGreater(volume, 0.0, part.name)

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

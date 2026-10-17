"""Export tests: the 3MF package layout and the OBJ/MTL pair.

The 3MF assertions follow the structure recorded from a real Bambu Studio
project file (``BambuStudio-02.06.00.51``): OPC plumbing, a single painted
object with the colours on its triangles, and every extruder assignment in
``Metadata/model_settings.config``.

Run with::

    .venv\\Scripts\\python.exe -m unittest tests.test_export -v
"""

from __future__ import annotations

import json
import sys
import unittest
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.engines import ENGINE_BAMBU  # noqa: E402
from app.core.image_matching import MatchSettings, build_palette, match_image  # noqa: E402
from app.core.library import Filament, FilamentLibrary  # noqa: E402
from app.core.mixes import MixCatalog  # noqa: E402
from app.mesh import objfile, threemf  # noqa: E402
from app.mesh.objfile import write_obj  # noqa: E402
from app.mesh.plate import (  # noqa: E402
    PlateSettings,
    build_plate,
    edge_health,
    merge_parts,
    plate_mesh,
    settings_for_width,
    signed_volume,
)
from app.mesh.threemf import (  # noqa: E402
    FilamentComponent,
    FilamentSlot,
    _project_settings,
    paint_code,
    project_filaments,
    slots_from_palette,
    write_3mf,
)

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


def _leftover_temporaries(folder: Path) -> list[Path]:
    """落盘用到的一半文件：名字里带 ``.tmp`` 的都不该留下。"""
    return sorted(Path(folder).rglob("*.tmp"))


class ExportTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.library = _library()
        palette = build_palette(self.library, None, include_mixes=False)
        self.result = match_image(_quadrant_image(), palette, MatchSettings(max_colours=4))
        self.plate = build_plate(self.result, settings_for_width(100.0, self.result.width))
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

    def test_a_failed_export_keeps_the_previous_file(self):
        """覆盖上一次那份导出时，中途出错不能把旧的 3MF 弄成半截（1.6.50）。"""
        path = self._write()
        before = path.read_bytes()

        def explode(*args, **kwargs):
            raise OSError("磁盘满了")

        # _model_settings 是在压缩包打开之后才求值的：老实现以写入模式直接打开目标，
        # 走到这里磁盘上已经只剩一个写了一半的 zip。
        with mock.patch.object(threemf, "_model_settings", explode):
            with self.assertRaises(OSError):
                self._write()

        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(_leftover_temporaries(self.tmp), [])

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

    def test_application_names_a_client_bambu_studio_trusts(self):
        """``Application`` is the switch that makes the project config count.

        Measured on 02.08.02.61, one build per fresh process: ``BambuPalette``
        and ``BambuStudio-BambuPalette-1.6.7`` both make it discard
        ``Metadata/project_settings.config`` and report 「3mf文件配置无效，仅加载
        几何数据」, while ``BambuStudio-2.3.0`` and ``BambuStudio-02.07.01.62``
        are accepted.  The part after the dash has to parse as a version.
        """
        from app.mesh.threemf import THREEMF_APPLICATION, THREEMF_GENERATOR

        root = self._model()
        values = {
            item.get("name"): item.text
            for item in root.iter()
            if _local(item.tag) == "metadata"
        }
        self.assertEqual(values.get("Application"), THREEMF_APPLICATION)
        self.assertTrue(THREEMF_APPLICATION.startswith("BambuStudio-"))
        version = THREEMF_APPLICATION.split("-", 1)[1]
        self.assertRegex(version, r"^\d+(\.\d+)+$")
        # The honest writer name is still recorded, in slice_info.config.
        with zipfile.ZipFile(self._write()) as archive:
            header = archive.read("Metadata/slice_info.config").decode("utf-8")
        self.assertIn(f'value="{THREEMF_GENERATOR}"', header)

    def test_the_project_config_carries_the_palette(self):
        """The colour list Bambu Studio reads is our palette, in slot order."""
        with zipfile.ZipFile(self._write()) as archive:
            names = archive.namelist()
            self.assertIn("Metadata/project_settings.config", names)
            settings = json.loads(
                archive.read("Metadata/project_settings.config").decode("utf-8")
            )
        expected = [
            self._by_extruder()[slot] for slot in sorted(self._by_extruder())
        ]
        self.assertEqual(settings["filament_colour"], expected)

    def test_every_per_filament_array_matches_the_palette_length(self):
        """A short array is what makes Bambu Studio call the config invalid.

        The template is the Blender 3MF add-on's stock Orca/Bambu project: 511
        keys, most arrays holding one entry per filament.  Each has to be
        resized to our filament count exactly as ``orca.py`` does.
        """
        from app.mesh.threemf import _PER_FILAMENT_ARRAYS

        with zipfile.ZipFile(self._write()) as archive:
            settings = json.loads(
                archive.read("Metadata/project_settings.config").decode("utf-8")
            )
        count = len(self._by_extruder())
        self.assertGreater(count, 0)
        resized = [
            key
            for key, value in settings.items()
            if isinstance(value, list)
            and value
            and (key.startswith("filament_") or key in _PER_FILAMENT_ARRAYS)
        ]
        self.assertTrue(resized, "the template should supply per-filament arrays")
        wrong = {key: len(settings[key]) for key in resized if len(settings[key]) != count}
        self.assertEqual(wrong, {})
        # And the document has to stay complete — a handful of keys is rejected.
        self.assertGreater(len(settings), 400)
        # Bambu's own spelling, or it substitutes the value and says so.
        self.assertEqual(settings["ensure_vertical_shell_thickness"], "enabled")

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
        # The file carries the MERGED pool, not one block per part: coincident
        # border vertices become a single index, which is what makes the written
        # mesh closed when a reader counts edges by index.
        merged, _per_part = merge_parts(self.plate.parts)
        self.assertEqual(len(vertices), len(merged))
        self.assertLess(len(vertices), sum(len(part.vertices) for part in self.plate.parts))
        self.assertEqual(len(triangles), self.plate.triangle_count)
        for triangle in triangles:
            for key in ("v1", "v2", "v3"):
                self.assertLess(int(triangle.get(key)), len(vertices))

    def test_the_parts_share_the_vertices_on_their_borders(self):
        """Merging must reunite border corners, not just renumber the parts.

        The 1.6.5 export concatenated the parts, so every border edge belonged to
        two index pairs and Bambu Studio reported 14,650 open edges on a solid
        that had none.  The written mesh has to reference a shared index instead.
        """
        root = self._model()
        mesh = next(item for item in root.iter() if _local(item.tag) == "mesh")
        vertices = [
            (item.get("x"), item.get("y"), item.get("z"))
            for item in mesh.iter()
            if _local(item.tag) == "vertex"
        ]
        triangles = [item for item in mesh.iter() if _local(item.tag) == "triangle"]
        counts: dict[tuple[str, str, str], int] = {}
        for triangle in triangles:
            for key in ("v1", "v2", "v3"):
                point = vertices[int(triangle.get(key))]
                counts[point] = counts.get(point, 0) + 1
        self.assertTrue(counts, "the mesh has no vertices")
        shared = [point for point, seen in counts.items() if seen > 1]
        self.assertTrue(
            shared,
            "no vertex is referenced by more than one triangle — the parts were "
            "concatenated again instead of merged",
        )
        # The merge is what closes the mesh, so measure that directly.
        points, _per_part = merge_parts(self.plate.parts)
        self.assertEqual(len(vertices), len(points))

    def test_the_last_written_vertex_is_the_model_vertex_on_the_bed(self):
        root = self._model()
        mesh = next(item for item in root.iter() if _local(item.tag) == "mesh")
        vertices = [item for item in mesh.iter() if _local(item.tag) == "vertex"]
        last = vertices[-1]
        # The merge keeps first-seen order, so the final vertex is the last point
        # no earlier part had already contributed.
        points, _per_part = merge_parts(self.plate.parts)
        expected = points[-1]
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
        empty = build_plate(self.result, settings_for_width(100.0, self.result.width))
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

    def test_a_failed_export_keeps_the_previous_pair(self):
        """两份都写完才改名：中途出错，磁盘上还是上一次那一对（1.6.50）。"""
        path = self._write()
        before = (path.read_bytes(), path.with_suffix(".mtl").read_bytes())

        real_open = objfile.atomic_path
        calls = []

        def second_call_fails(target, **kwargs):
            calls.append(Path(target))
            if len(calls) >= 2:
                raise OSError("磁盘满了")
            return real_open(target, **kwargs)

        with mock.patch.object(objfile, "atomic_path", second_call_fails):
            with self.assertRaises(OSError):
                self._write()

        self.assertEqual(len(calls), 2, calls)
        self.assertEqual(path.read_bytes(), before[0])
        self.assertEqual(path.with_suffix(".mtl").read_bytes(), before[1])
        self.assertEqual(_leftover_temporaries(self.tmp), [])

    def test_the_obj_references_the_material_library(self):
        text = self._write().read_text(encoding="utf-8")
        self.assertIn("mtllib plate.mtl", text)

    def test_vertex_and_face_counts_match_the_plate(self):
        lines = self._write().read_text(encoding="utf-8").splitlines()
        vertices = [line for line in lines if line.startswith("v ")]
        faces = [line for line in lines if line.startswith("f ")]
        merged, _per_part = merge_parts(self.plate.parts)
        self.assertEqual(len(vertices), len(merged))
        self.assertLess(len(vertices), self.plate.vertex_count)
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
        empty = build_plate(self.result, settings_for_width(100.0, self.result.width))
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


class _RecipeEntry:
    """Just enough of a ``PaletteEntry`` for the export path.

    ``slots_from_palette`` reads entries through ``getattr`` on purpose (this
    module must not import ``app.core``), so a mix can be described here by the
    two fields the recipe lives in and nothing else.
    """

    def __init__(self, color_hex: str, parts: tuple = (), recipe=None) -> None:
        self.color_hex = color_hex
        self.kind = "mix"
        self.label = "混色"
        self.parts = parts
        self.recipe = recipe
        self.filament_ids = tuple(parent for parent, _percent in parts)


class MixedFilamentTests(ExportTestCase):
    """The exported project carries the recipes behind the mixed colours.

    Bambu Studio keeps a mixed colour as one row of its project filament list
    plus one row per spool it mixes, tied together by ``filament_is_mixed`` /
    ``filament_mixed_components`` / ``filament_mixed_sublayer_ratios``.  The
    shapes asserted here come from a project Bambu Studio exported itself
    (「猪包」, 2026-10-11): parents 1-based in recipe order, ratios four decimals
    summing to 1.
    """

    def _mix(self):
        catalog = MixCatalog(self.library.filaments, engine=ENGINE_BAMBU).build()
        palette = build_palette(self.library, catalog)
        return next(entry for entry in palette if entry.is_mix)

    def test_a_mix_becomes_a_mixed_row_with_its_two_spools(self):
        entry = self._mix()
        slots = slots_from_palette([entry], self.library)
        self.assertTrue(slots[0].is_mixed, "a two-spool mix keeps its recipe")
        project = project_filaments(slots)
        self.assertEqual(project.count, 3, "the mix and its two spools")
        self.assertEqual(project.added, 2)
        self.assertEqual(project.mixed_count, 1)
        # The two spools take rows 1 and 2 and the mixed row closes the list.
        self.assertEqual(project.order, {1: 3})
        parents, ratios = project.mixed[2]
        self.assertEqual(parents, (1, 2))
        self.assertEqual(len(ratios), 2)
        self.assertAlmostEqual(sum(ratios), 1.0, places=6)

    def test_a_three_spool_recipe_lists_three_parents(self):
        ids = [filament.id for filament in self.library.filaments[:3]]
        entry = _RecipeEntry(
            "#808080", parts=((ids[0], 50), (ids[1], 30), (ids[2], 20))
        )
        slots = slots_from_palette([entry], self.library)
        self.assertEqual(len(slots[0].components), 3)
        project = project_filaments(slots)
        parents, ratios = project.mixed[project.count - 1]
        self.assertEqual(len(parents), 3)
        self.assertEqual(len(set(parents)), 3)
        self.assertEqual([round(ratio, 4) for ratio in ratios], [0.5, 0.3, 0.2])

    def test_spools_already_in_the_list_are_reused_not_duplicated(self):
        entry = self._mix()
        parents = [self.library.get(filament_id) for filament_id in entry.filament_ids]
        painted = [
            FilamentSlot(
                color_hex=parent.color_hex,
                material_type=parent.material_type,
                vendor=parent.brand,
            )
            for parent in parents
        ]
        project = project_filaments(painted + slots_from_palette([entry], self.library))
        self.assertEqual(project.added, 0)
        self.assertEqual(project.count, 3)
        self.assertEqual(project.mixed[2][0], (1, 2))

    def test_a_mix_whose_spools_are_gone_is_exported_flat(self):
        """A project opened on a machine without the spools still paints right.

        The plate is painted with the mix's own colour, so a 混色耗材 that named
        a spool the reader does not have would be worse than the flat colour the
        user picked.
        """
        entry = self._mix()
        slots = slots_from_palette([entry], None)
        self.assertFalse(slots[0].is_mixed)
        self.assertEqual(project_filaments(slots).added, 0)

    def test_the_settings_carry_the_mix_the_way_bambu_studio_writes_it(self):
        entry = self._mix()
        project = project_filaments(slots_from_palette([entry], self.library))
        settings = json.loads(_project_settings(project.slots, project.mixed))
        parents, ratios = project.mixed[2]
        self.assertEqual(settings["filament_is_mixed"], ["0", "0", "1"])
        self.assertEqual(settings["filament_mixed_components"], ["", "", "1,2"])
        self.assertEqual(
            settings["filament_mixed_sublayer_ratios"],
            ["", "", ",".join(f"{ratio:.4f}" for ratio in ratios)],
        )
        self.assertEqual(settings["filament_colour"], settings["filament_multi_colour"])
        self.assertEqual(settings["enable_mixed_color_sublayer"], "1")
        # 渐变混色耗材 is a ratio that changes with height; a plate prints one
        # colour per block, so every gradient key stays off.
        for key in (
            "filament_mixed_gradient",
            "filament_mixed_gradient_per_part",
        ):
            self.assertEqual(settings[key], ["0", "0", "0"])
        for key in ("filament_mixed_gradient_curve", "filament_mixed_gradient_range"):
            self.assertEqual(settings[key], ["", "", ""])

    def test_a_project_without_mixes_writes_no_mixed_rows(self):
        colours = ["#F2F0EB", "#17181C"]
        settings = json.loads(_project_settings(colours))
        self.assertEqual(settings["filament_is_mixed"], ["0", "0"])
        self.assertEqual(settings["filament_mixed_components"], ["", ""])
        self.assertNotIn("enable_mixed_color_sublayer", settings)

    def test_the_written_project_lists_the_mix_and_the_spools_it_needs(self):
        entry = self._mix()
        painted = slots_from_palette(
            build_palette(self.library, None, include_mixes=False), self.library
        )
        painted[0] = slots_from_palette([entry], self.library)[0]
        path = write_3mf(self.plate, self.tmp / "mixed", filaments=painted)
        with zipfile.ZipFile(path) as archive:
            settings = json.loads(
                archive.read("Metadata/project_settings.config").decode("utf-8")
            )
            model = archive.read("Metadata/model_settings.config").decode("utf-8")
        count = len(settings["filament_colour"])
        # Every spool this plate paints with here happens to be one of the
        # recipe's parents, so the list stays at the painted rows — with the mix
        # moved from the front to the end, which is where Bambu Studio reads it.
        self.assertEqual(count, len(painted))
        self.assertEqual(settings["filament_is_mixed"][-1], "1")
        self.assertEqual(settings["filament_is_mixed"][:-1], ["0"] * (count - 1))
        parents = settings["filament_mixed_components"][-1].split(",")
        self.assertEqual(len(parents), 2)
        expected = [
            self.library.get(filament_id).color_hex.upper()
            for filament_id in entry.filament_ids
        ]
        for parent, colour in zip(parents, expected):
            index = int(parent) - 1
            self.assertLess(index, count - 1, "a parent is a row before the mix")
            self.assertEqual(settings["filament_is_mixed"][index], "0")
            self.assertEqual(settings["filament_colour"][index], colour)
        ratios = [
            float(text)
            for text in settings["filament_mixed_sublayer_ratios"][-1].split(",")
        ]
        self.assertAlmostEqual(sum(ratios), 1.0, places=4)
        self.assertIn(f'key="filament_maps" value="{" ".join(["1"] * count)}"', model)
        self.assertIn(
            f'key="filament_volume_maps" value="{" ".join(["0"] * count)}"', model
        )

    def test_a_spool_only_the_recipe_uses_becomes_a_row_of_its_own(self):
        """The case 「猪包」 showed: a mix whose parent nothing paints.

        ``filament_mixed_components`` points at 1-based rows of the same list, so
        the spool has to be there even though no triangle is painted with it —
        Bambu Studio creates that row itself when its own 拆色 makes a mix.
        """
        spare = Filament(name="耗材蓝", brand="测试", material_type="PETG HF", color_hex="#1B4F9C")
        library = FilamentLibrary(list(self.library.filaments) + [spare])
        palette = build_palette(library, None, include_mixes=False)
        result = match_image(_quadrant_image(), palette, MatchSettings(max_colours=4))
        plate = build_plate(result, settings_for_width(100.0, result.width))
        painted = slots_from_palette(palette, library)[: plate.extruder_count]
        self.assertNotIn(
            spare.color_hex.upper(),
            [slot.color_hex.upper() for slot in painted],
            "the spare spool is not painted anywhere",
        )
        mix_only = FilamentSlot(
            color_hex="#6E4A28",
            material_type="PETG HF",
            vendor="测试",
            components=(
                FilamentComponent(
                    color_hex=self.library.filaments[1].color_hex,
                    material_type="PETG HF",
                    vendor="测试",
                    percent=57.0,
                ),
                FilamentComponent(
                    color_hex=spare.color_hex,
                    material_type="PETG HF",
                    vendor="测试",
                    percent=43.0,
                ),
            ),
        )
        painted[0] = mix_only
        project = project_filaments(painted)
        self.assertEqual(
            project.order[1], project.count, "the painted mix closes the list"
        )
        self.assertEqual(project.count, len(painted) + project.added)
        self.assertGreaterEqual(project.added, 1, "the spool nothing paints is still a row")
        parents, ratios = project.mixed[project.count - 1]
        self.assertEqual(len(parents), 2)
        self.assertEqual([round(ratio, 4) for ratio in ratios], [0.57, 0.43])
        self.assertTrue(
            all(parent < project.count for parent in parents),
            "every parent is a row before the mix",
        )
        appended = project.slots[parents[1] - 1]
        self.assertEqual(appended.color_hex, spare.color_hex)
        self.assertFalse(appended.is_mixed)

        settings = json.loads(_project_settings(project.slots, project.mixed))
        self.assertEqual(settings["filament_is_mixed"][project.count - 1], "1")
        self.assertEqual(
            settings["filament_mixed_components"][project.count - 1],
            ",".join(str(parent) for parent in parents),
        )
        self.assertEqual(
            settings["filament_mixed_sublayer_ratios"][project.count - 1],
            "0.5700,0.4300",
        )

        path = write_3mf(plate, self.tmp / "spare", filaments=painted)
        with zipfile.ZipFile(path) as archive:
            written = json.loads(
                archive.read("Metadata/project_settings.config").decode("utf-8")
            )
            model = archive.read("Metadata/model_settings.config").decode("utf-8")
        count = len(written["filament_colour"])
        self.assertEqual(count, plate.extruder_count + project.added)
        self.assertEqual(written["filament_is_mixed"][-1], "1")
        self.assertEqual(
            written["filament_colour"][parents[1] - 1], spare.color_hex.upper()
        )
        self.assertIn(f'key="filament_maps" value="{" ".join(["1"] * count)}"', model)

    def test_the_mixed_rows_are_written_after_every_other_row(self):
        """The reader refuses a project whose mixed rows are interleaved.

        Measured against the installed Bambu Studio (``02.08.02.61``): a project
        that lists a mixed row before rows it does not need exports as
        「Invalid parameter value(s) included in the 3mf file」 and writes nothing
        (``return_code`` -18), while the same package with every mixed row moved
        to the end exports fine.  A mixed row may itself be painted — the codes
        just have to move with it, or the plate would print a different spool.
        """
        entry = self._mix()
        painted = slots_from_palette(
            build_palette(self.library, None, include_mixes=False), self.library
        )
        painted[0] = slots_from_palette([entry], self.library)[0]
        project = project_filaments(painted)
        path = write_3mf(self.plate, self.tmp / "order", filaments=painted)
        with zipfile.ZipFile(path) as archive:
            settings = json.loads(
                archive.read("Metadata/project_settings.config").decode("utf-8")
            )
            model = archive.read("3D/3dmodel.model").decode("utf-8")

        flags = settings["filament_is_mixed"]
        self.assertEqual(flags[-1], "1")
        self.assertNotIn("1", flags[:-1], "no mixed row survives in the middle")
        for index, components in enumerate(settings["filament_mixed_components"]):
            if components:
                self.assertEqual(flags[index], "1")
                self.assertTrue(
                    all(int(parent) < index + 1 for parent in components.split(",")),
                    "a parent is always a row before the mix that names it",
                )
        # The plate is painted with the mix, so its triangles carry the code of
        # the row the mix ended up in — the last row, which is a mixed one.
        last = paint_code(len(flags))
        self.assertIn(f'paint_color="{last}"', model)


if __name__ == "__main__":
    unittest.main()

"""Acceptance test: hand our 3MF to the real Bambu Studio and check what comes back.

Usage::

    .venv\\Scripts\\python.exe tools\\verify_bambu_roundtrip.py
    .venv\\Scripts\\python.exe tools\\verify_bambu_roundtrip.py samples\\sample-plate.3mf

With no argument it builds a small plate of its own; pass a ``.3mf`` path to run
the same check against an existing export (for example the shipped sample).

Bambu Studio is a PrusaSlicer descendant, so it accepts ``--export-3mf <out> <in>``:
it loads the input, builds a scene and writes a fresh project file.  That round
trip is the strongest check available without clicking through the GUI — if the
reader had rejected our structure, the re-exported model would come back with no
geometry at all.

This is exactly how the first exporter was caught: the production-extension
``<component p:path=…>`` layout was accepted metadata-wise but every component
was dropped, leaving ``face_count="0"``.  The current core-3MF inline layout is
verified by this script.

Exit code 0 means Bambu Studio loaded and reproduced the geometry.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core import paths  # noqa: E402

TEMP = Path(tempfile.mkdtemp(prefix="fcs-bambu-"))
paths.library_path = lambda: TEMP / "filament-library.json"  # type: ignore[assignment]
paths.data_dir = lambda: TEMP  # type: ignore[assignment]

from app.core.image_matching import MatchSettings, build_palette, match_image  # noqa: E402
from app.core.library import Filament, FilamentLibrary  # noqa: E402
from app.mesh.plate import PlateSettings, build_plate  # noqa: E402
from app.mesh.threemf import slots_from_palette, write_3mf  # noqa: E402

BAMBU_CANDIDATES = (
    Path(r"C:\Program Files\Bambu Studio\bambu-studio.exe"),
    Path(r"C:\Program Files (x86)\Bambu Studio\bambu-studio.exe"),
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Bambu Studio" / "bambu-studio.exe",
)

SPOOLS = (
    ("大简 PETG HF 白", "#F2F0EB"),
    ("大简 PETG HF 黑", "#17181C"),
    ("大简 PETG HF 金", "#D9A441"),
    ("大简 PETG HF 青", "#2FA8B8"),
    ("大简 PETG HF 红", "#C8342E"),
)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def find_bambu() -> Path | None:
    for candidate in BAMBU_CANDIDATES:
        if candidate and candidate.is_file():
            return candidate
    return None


def build_sample() -> Path:
    from PIL import Image, ImageDraw

    library = FilamentLibrary([
        Filament(name=name, brand="大简", material_type="PETG HF", color_hex=value)
        for name, value in SPOOLS
    ])
    picture = TEMP / "roundtrip.png"
    image = Image.new("RGB", (160, 120), "#F2F0EB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 75, 65), fill="#C8342E")
    draw.ellipse((85, 10, 150, 60), fill="#D9A441")
    draw.polygon([(20, 110), (80, 75), (140, 110)], fill="#2FA8B8")
    image.save(picture)

    palette = build_palette(library, None, include_mixes=False)
    result = match_image(picture, palette, MatchSettings(max_colours=8))
    plate = build_plate(result, PlateSettings(target_width_mm=120.0))
    return write_3mf(
        plate,
        TEMP / "source.3mf",
        object_name="往返验证底板",
        filaments=slots_from_palette(result.palette, library),
    )


def count_geometry(path: Path) -> dict:
    """Everything we care about in a 3MF, however Bambu Studio chose to lay it out."""
    report = {
        "vertices": 0,
        "triangles": 0,
        "objects": 0,
        "parts": [],
        "meshes": 0,
        "paints": {},  # paint_color string -> triangle count
        "unpainted": 0,
        "filaments": [],
    }
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        for name in names:
            if not name.endswith(".model"):
                continue
            root = ET.fromstring(archive.read(name))
            report["meshes"] += sum(1 for item in root.iter() if _local(item.tag) == "mesh")
            report["vertices"] += sum(1 for item in root.iter() if _local(item.tag) == "vertex")
            for item in root.iter():
                if _local(item.tag) != "triangle":
                    continue
                report["triangles"] += 1
                code = item.get("paint_color")
                if code:
                    report["paints"][code] = report["paints"].get(code, 0) + 1
                else:
                    report["unpainted"] += 1
            report["objects"] += sum(1 for item in root.iter() if _local(item.tag) == "object")
        if "Metadata/model_settings.config" in names:
            root = ET.fromstring(archive.read("Metadata/model_settings.config"))
            for element in root.iter():
                if _local(element.tag) != "part":
                    continue
                metadata = {
                    item.get("key"): item.get("value")
                    for item in element
                    if _local(item.tag) == "metadata"
                }
                report["parts"].append((element.get("id"), metadata.get("extruder"), metadata.get("name")))
        if "Metadata/project_settings.config" in names:
            try:
                settings = json.loads(archive.read("Metadata/project_settings.config"))
            except (ValueError, UnicodeDecodeError):
                settings = {}
            colours = settings.get("filament_colour")
            if isinstance(colours, list):
                report["filaments"] = [str(item) for item in colours]
    return report


def mesh_health(path: Path) -> tuple[int, int]:
    """Open / non-manifold edge counts of a 3MF, welded the way a slicer welds it.

    Bambu Studio reports these two numbers on the object panel, so they are the
    numbers a user judges the file by.  Every part is written with its own vertex
    block, so the positions have to be reunited before counting — otherwise each
    part looks like a bag of open edges.
    """
    import numpy as np

    from app.mesh.plate import edge_health

    points: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    offset = 0
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.endswith(".model"):
                continue
            root = ET.fromstring(archive.read(name))
            for mesh in (item for item in root.iter() if _local(item.tag) == "mesh"):
                base = offset
                for vertex in mesh.iter():
                    if _local(vertex.tag) != "vertex":
                        continue
                    points.append(
                        (
                            float(vertex.get("x", 0.0)),
                            float(vertex.get("y", 0.0)),
                            float(vertex.get("z", 0.0)),
                        )
                    )
                offset += sum(
                    1 for item in mesh.iter() if _local(item.tag) == "vertex"
                )
                for item in mesh.iter():
                    if _local(item.tag) != "triangle":
                        continue
                    faces.append(
                        (
                            int(item.get("v1", 0)) + base,
                            int(item.get("v2", 0)) + base,
                            int(item.get("v3", 0)) + base,
                        )
                    )
    if not faces:
        return 0, 0
    return edge_health(
        np.array(points, dtype=np.float64).reshape(-1, 3),
        np.array(faces, dtype=np.int64).reshape(-1, 3),
    )


def main() -> int:
    bambu = find_bambu()
    if bambu is None:
        print("SKIP: Bambu Studio was not found in any of the usual install locations")
        for candidate in BAMBU_CANDIDATES:
            print("      looked for", candidate)
        return 2

    print("bambu studio:", bambu)
    if len(sys.argv) > 1:
        source = Path(sys.argv[1]).resolve()
        if not source.is_file():
            print(f"FAIL: {source} does not exist")
            return 1
    else:
        source = build_sample()
    source_report = count_geometry(source)
    print(
        f"source      : {source.stat().st_size:>8,} B  "
        f"{source_report['vertices']:,} vertices  {source_report['triangles']:,} triangles  "
        f"{len(source_report['parts'])} parts  {len(source_report['paints'])} paint codes"
    )
    for part_id, extruder, name in source_report["parts"]:
        print(f"              part {part_id}: extruder {extruder}  {name}")
    print(f"              filaments: {' '.join(source_report['filaments']) or '(none)'}")
    print(
        "              paints: "
        + "  ".join(f"{code}×{count:,}" for code, count in sorted(source_report["paints"].items()))
    )
    source_open, source_non_manifold = mesh_health(source)
    print(
        f"              mesh: {source_open:,} open edges  "
        f"{source_non_manifold:,} non-manifold edges"
    )

    exported = TEMP / "bambu-roundtrip.3mf"
    if exported.exists():
        exported.unlink()
    completed = subprocess.run(
        [str(bambu), "--export-3mf", str(exported), str(source)],
        capture_output=True,
        text=True,
        timeout=600,
        # Bambu Studio drops a ``result.json`` in its working directory; run it
        # inside the scratch directory so it never litters the project root.
        cwd=str(TEMP),
    )
    if not exported.is_file():
        print(f"FAIL: Bambu Studio did not write {exported} (exit {completed.returncode})")
        print(completed.stdout[-2000:])
        print(completed.stderr[-2000:])
        return 1

    returned = count_geometry(exported)
    print(
        f"round trip  : {exported.stat().st_size:>8,} B  "
        f"{returned['vertices']:,} vertices  {returned['triangles']:,} triangles  "
        f"{len(returned['parts'])} parts  {returned['meshes']} meshes"
    )
    for part_id, extruder, name in returned["parts"]:
        print(f"              part {part_id}: extruder {extruder}  {name}")
    print(f"              filaments: {' '.join(returned['filaments']) or '(none)'}")
    print(
        "              paints: "
        + "  ".join(f"{code}×{count:,}" for code, count in sorted(returned["paints"].items()))
    )

    failures: list[str] = []
    warnings: list[str] = []
    if returned["triangles"] == 0:
        failures.append("Bambu Studio returned no triangles — the geometry did not survive the import")
    if returned["vertices"] == 0:
        failures.append("Bambu Studio returned no vertices")
    if returned["triangles"] < source_report["triangles"] * 0.5:
        failures.append(
            f"only {returned['triangles']} of {source_report['triangles']} triangles came back"
        )
    if not returned["parts"]:
        failures.append("Bambu Studio kept no <part> entries, so the object did not survive")
    if source_report["paints"] and not returned["paints"]:
        failures.append(
            "every paint_color was lost, so the plate came back unpainted — the colours "
            "would have to be assigned by hand again"
        )
    missing = set(source_report["paints"]) - set(returned["paints"])
    if missing:
        failures.append(f"paint codes {sorted(missing)} did not survive the round trip")
    if source_open:
        failures.append(
            f"the exported plate has {source_open:,} open edges — Bambu Studio shows these "
            f"as 信息: 发现 N 个开放边 and offers to repair the model"
        )
    if source_non_manifold:
        failures.append(
            f"the exported plate has {source_non_manifold:,} non-manifold edges — Bambu Studio "
            f"shows these as 错误: N 非流形边"
        )
    if source_report["filaments"] and returned["filaments"] != source_report["filaments"]:
        # NOT a failure, and for our own exports it no longer even applies: the
        # writer stopped emitting ``Metadata/project_settings.config``, because
        # Bambu Studio rejects any partial project config and then paints the
        # plate from its own spools (see the comment in ``app/mesh/threemf.py``).
        # The colours that travel are the ``<basematerials>`` table and the
        # ``paint_color`` codes on the triangles, and those are checked above.
        warnings.append(
            f"the filament list is not carried by a foreign-project import "
            f"({source_report['filaments']} -> {returned['filaments']}); "
            f"the plate keeps its colours through basematerials/paint_color instead"
        )

    print()
    if warnings:
        print("WARNINGS")
        for item in warnings:
            print("  !", item)
        print()
    if failures:
        print("FAILED")
        for item in failures:
            print("  -", item)
        return 1
    print("BAMBU ROUND TRIP OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

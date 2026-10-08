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
from app.mesh.threemf import write_3mf  # noqa: E402

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
    return write_3mf(plate, TEMP / "source.3mf", object_name="往返验证底板")


def count_geometry(path: Path) -> dict:
    """Vertices, triangles and per-part extruders of a 3MF, however it is laid out."""
    report = {"vertices": 0, "triangles": 0, "objects": 0, "parts": [], "meshes": 0}
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        for name in names:
            if not name.endswith(".model"):
                continue
            root = ET.fromstring(archive.read(name))
            report["meshes"] += sum(1 for item in root.iter() if _local(item.tag) == "mesh")
            report["vertices"] += sum(1 for item in root.iter() if _local(item.tag) == "vertex")
            report["triangles"] += sum(1 for item in root.iter() if _local(item.tag) == "triangle")
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
    return report


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
        f"{len(source_report['parts'])} parts"
    )
    for part_id, extruder, name in source_report["parts"]:
        print(f"              part {part_id}: extruder {extruder}  {name}")

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

    failures: list[str] = []
    if returned["triangles"] == 0:
        failures.append("Bambu Studio returned no triangles — the geometry did not survive the import")
    if returned["vertices"] == 0:
        failures.append("Bambu Studio returned no vertices")
    if returned["triangles"] < source_report["triangles"] * 0.5:
        failures.append(
            f"only {returned['triangles']} of {source_report['triangles']} triangles came back"
        )
    if not returned["parts"]:
        failures.append("Bambu Studio kept no <part> entries, so no extruder assignment survived")

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

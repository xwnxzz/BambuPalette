"""Write a ready-to-open sample plate into ``samples/``.

Usage::

    .venv\\Scripts\\python.exe tools\\export_sample_plate.py

Generates a test picture, matches it against a five-spool library, builds the
flat multi-colour plate and writes both exporters' output so the files can be
opened in Bambu Studio (or inspected in any other 3D tool) without running the
GUI.  Nothing touches the user's real library.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core import paths  # noqa: E402

TEMP = Path(tempfile.mkdtemp(prefix="fcs-sample-"))
paths.library_path = lambda: TEMP / "filament-library.json"  # type: ignore[assignment]
paths.data_dir = lambda: TEMP  # type: ignore[assignment]

from app.core.image_matching import MatchSettings, build_palette, match_image  # noqa: E402
from app.core.library import Filament, FilamentLibrary  # noqa: E402
from app.mesh.objfile import write_obj  # noqa: E402
from app.mesh.plate import PlateSettings, build_plate  # noqa: E402
from app.mesh.threemf import write_3mf  # noqa: E402

SPOOLS = (
    ("大简 PETG HF 白", "#F2F0EB"),
    ("大简 PETG HF 黑", "#17181C"),
    ("大简 PETG HF 金", "#D9A441"),
    ("大简 PETG HF 青", "#2FA8B8"),
    ("大简 PETG HF 红", "#C8342E"),
)

OUT = ROOT / "samples"


def make_picture(target: Path) -> Path:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (320, 240), "#F2F0EB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((16, 16, 150, 130), fill="#C8342E")
    draw.ellipse((170, 20, 300, 120), fill="#D9A441")
    draw.polygon([(40, 220), (160, 150), (280, 220)], fill="#2FA8B8")
    for x in range(16, 300):
        shade = int(32 + (x - 16) / 284 * 200)
        draw.line([(x, 132), (x, 145)], fill=(shade, shade, shade))
    image.save(target)
    return target


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    picture = make_picture(OUT / "sample-picture.png")

    library = FilamentLibrary([
        Filament(name=name, brand="大简", material_type="PETG HF", color_hex=value)
        for name, value in SPOOLS
    ])
    palette = build_palette(library, None, include_mixes=False)
    result = match_image(picture, palette, MatchSettings(max_colours=12))
    print(f"matched {len(result.palette)} colours from {result.printed_pixels} solid pixels")

    plate = build_plate(result, PlateSettings(target_width_mm=150.0))
    print("plate:", plate.stats())
    for note in plate.notes:
        print("note:", note)

    three_mf = write_3mf(plate, OUT / "sample-plate.3mf", object_name="混色底板示例")
    obj = write_obj(plate, OUT / "sample-plate.obj")
    print("wrote", three_mf, f"{three_mf.stat().st_size} B")
    print("wrote", obj, f"{obj.stat().st_size} B")
    print("wrote", OUT / "sample-plate.mtl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Self-check: prove the whole pipeline works in this build.

Run from the packaged application with ``--selftest``.  It is deliberately
independent of the test suite so a user (or support) can confirm that numpy,
Pillow, Qt and the mesh writers all function in the frozen bundle:

* build a five-spool library and the full mix catalogue,
* render a small picture and match it against that library,
* build the flat multi-colour plate,
* write both a 3MF and an OBJ + MTL,
* re-open the 3MF and check its structure.

The report is written to ``selftest-report.txt`` and, when a console exists,
also printed.  Exit code 0 means everything passed.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

SPOOLS = (
    ("大简 PETG HF 白", "#F2F0EB"),
    ("大简 PETG HF 黑", "#17181C"),
    ("大简 PETG HF 金", "#D9A441"),
    ("大简 PETG HF 青", "#2FA8B8"),
    ("大简 PETG HF 红", "#C8342E"),
)


def run_selftest(*, report_path: Path | None = None) -> int:
    lines: list[str] = []
    failures: list[str] = []

    def check(condition: bool, label: str) -> None:
        lines.append(f"[{'ok' if condition else 'FAIL'}] {label}")
        if not condition:
            failures.append(label)

    from .core import paths

    try:
        # Bounded, unlike tempfile.mkdtemp: a refused temp directory must not
        # turn the self-check into an unkillable retry loop.
        scratch = paths.scratch_dir("fcs-selftest-")
    except OSError:
        scratch = paths.data_dir() / "selftest"
        scratch.mkdir(parents=True, exist_ok=True)
    paths.library_path = lambda: scratch / "filament-library.json"  # type: ignore[assignment]
    paths.data_dir = lambda: scratch  # type: ignore[assignment]

    from . import APP_NAME, __version__

    lines.append(f"app         {APP_NAME} {__version__}")
    lines.append(f"python      {sys.version.split()[0]}")
    lines.append(f"frozen      {bool(getattr(sys, 'frozen', False))}")
    lines.append(f"scratch     {scratch}")
    lines.append("")

    try:
        import numpy

        lines.append(f"numpy       {numpy.__version__}")
        from PIL import __version__ as pillow_version

        lines.append(f"Pillow      {pillow_version}")
        from PySide6 import __version__ as qt_version

        lines.append(f"PySide6     {qt_version}")
    except Exception as error:  # pragma: no cover - only on a broken bundle
        check(False, f"importing the dependencies failed: {error}")

    from .core.library import Filament, FilamentLibrary
    from .main import build_application

    app = build_application([])

    from .core.engines import ENGINES, ENGINE_BAMBU, ENGINE_MIXER
    from .core.image_matching import MatchSettings, build_palette, match_image
    from .core.mixes import expected_recipe_count
    from .mesh.objfile import write_obj
    from .mesh.plate import PlateSettings, build_plate
    from .mesh.threemf import write_3mf
    from .ui.main_window import MainWindow

    lines.append(f"engines     {', '.join(sorted(ENGINES))}")
    lines.append("")

    library = FilamentLibrary([
        Filament(name=name, brand="大简", material_type="PETG HF", color_hex=value)
        for name, value in SPOOLS
    ])
    window = MainWindow(library=library)
    window.resize(1200, 760)
    window.show()
    app.processEvents()

    check(len(window._recipes) == expected_recipe_count(5) == 810, "5 spools produce 810 mixes")
    check(window._catalog.pair_count == 10, "5 spools produce 10 pairs")
    check(window._catalog.engine.id == ENGINE_MIXER, "the Bambu 2.8 mixer engine is the default")
    check(
        ENGINES[ENGINE_MIXER].mix_rgb_pair((0, 33, 133), (252, 211, 0), 50) == (47, 141, 56),
        "the bundled pigment polynomial reproduces its reference sample",
    )
    check(
        ENGINES[ENGINE_BAMBU].mix_hex("#FFFFFF", "#000000", 90) == "#E5E5E5",
        "the legacy sRGB engine is still selectable and exact",
    )
    check(
        all(
            len({*engine.mix_rgb_pair((255, 255, 255), (0, 0, 0), percent)}) == 1
            for engine in ENGINES.values()
            for percent in (10, 50, 90)
        ),
        "every engine keeps a white + black mix neutral (no blue cast)",
    )
    check(
        not window._target_color.isSet(),
        "the 目标颜色 picker starts with no preset colour",
    )
    check(
        window._picture_page is not None and window._tabs.count() == 2,
        "both pages are present",
    )

    # A picture with four flat fields and a gradient.
    from PIL import Image, ImageDraw

    picture = scratch / "selftest.png"
    image = Image.new("RGB", (160, 120), "#F2F0EB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 75, 65), fill="#C8342E")
    draw.ellipse((85, 10, 150, 60), fill="#D9A441")
    draw.polygon([(20, 110), (80, 75), (140, 110)], fill="#2FA8B8")
    for x in range(8, 150):
        shade = int(32 + (x - 8) / 142 * 200)
        draw.line([(x, 66), (x, 72)], fill=(shade, shade, shade))
    image.save(picture)

    palette = build_palette(library, window._catalog)
    check(len(palette) == 5 + 810, f"the candidate palette has every spool and mix ({len(palette)})")

    result = match_image(picture, palette, MatchSettings(max_colours=12))
    check(result.printed_pixels > 0, f"the picture matched ({result.printed_pixels} solid pixels)")
    check(len(result.palette) >= 3, f"{len(result.palette)} colours were needed")

    page = window._picture_page
    page.loadImage(picture)
    app.processEvents()
    check(page.result is not None, "the picture page matched the picture")

    page.buildPlate()
    plate = page.plate
    check(plate is not None, "a plate was built")
    if plate is not None:
        check(plate.width_mm > 0 and plate.depth_mm > 0, f"the plate is {plate.width_mm} × {plate.depth_mm} mm")
        check(plate.triangle_count > 0, f"the plate has {plate.triangle_count} triangles")

        three_mf = write_3mf(plate, scratch / "selftest.3mf", object_name="自检底板")
        obj = write_obj(plate, scratch / "selftest.obj")
        check(three_mf.is_file() and three_mf.stat().st_size > 0, f"3MF written ({three_mf.stat().st_size} B)")
        check(obj.is_file() and obj.stat().st_size > 0, f"OBJ written ({obj.stat().st_size} B)")
        check((scratch / "selftest.mtl").is_file(), "MTL written")

        if zipfile.is_zipfile(three_mf):
            with zipfile.ZipFile(three_mf) as archive:
                names = set(archive.namelist())
            needed = {
                "[Content_Types].xml",
                "_rels/.rels",
                "3D/3dmodel.model",
                "Metadata/model_settings.config",
            }
            check(needed <= names, "the 3MF holds every required entry")
        else:
            check(False, "the 3MF is a zip package")

    # Render the real window once, so a broken Qt platform plugin or a missing
    # font shows up here rather than in front of the user.
    shot = None
    try:
        shot = Path.cwd() / "selftest-ui.png"
        check(window.grab().save(str(shot)), "the user interface rendered to a PNG")
    except Exception as error:  # pragma: no cover - only on a broken bundle
        check(False, f"rendering the user interface failed: {error}")

    window.close()

    lines.append("")
    if shot is not None and shot.is_file():
        lines.append(f"ui          {shot} ({shot.stat().st_size} B)")
    if failures:
        lines.append(f"FAILED ({len(failures)})")
        lines.extend(f"  - {item}" for item in failures)
    else:
        lines.append("SELFTEST OK")

    text = "\n".join(lines) + "\n"
    target = report_path or (Path.cwd() / "selftest-report.txt")
    try:
        target.write_text(text, encoding="utf-8")
    except OSError:  # pragma: no cover - read-only working directory
        target = scratch / "selftest-report.txt"
        target.write_text(text, encoding="utf-8")

    if sys.stdout is not None:  # a windowed build has no console
        try:
            sys.stdout.write(text)
            sys.stdout.write(f"report: {target}\n")
        except Exception:  # pragma: no cover
            pass

    return 1 if failures else 0

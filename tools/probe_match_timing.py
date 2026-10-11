"""How long one ``match_image`` really takes, step by step.

The 1.6.11 round adds a progress bar over 「图像转换」/「重新匹配」.  Before
drawing one, measure: the core already reports five coarse steps (读取图片 →
缩放图片 → 减少图片颜色数量 → 寻找最接近的颜色 → 按像素分配颜色), but a bar that
stands still for two seconds reads as a hang, so the coarse steps are only good
enough if none of them dominates.

This probe runs the same shape of match the picture page runs — a spool library,
its two-colour mixes, and the lazy three-filament tail — and prints the wall
time spent inside each reported step, so the number under each message is the
number the user would see frozen on the bar.

Run with::

    .venv\\Scripts\\python.exe tools\\probe_match_timing.py
    .venv\\Scripts\\python.exe tools\\probe_match_timing.py --spools 41 --no-triples
"""

from __future__ import annotations

import argparse
import colorsys
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image  # noqa: E402

from app.core.engines import ENGINE_BAMBU, ENGINE_MIXER  # noqa: E402
from app.core.image_matching import (  # noqa: E402
    MatchSettings,
    TriplePalette,
    build_palette,
    match_image,
)
from app.core.library import Filament, FilamentLibrary  # noqa: E402
from app.core.mixes import MixCatalog  # noqa: E402
from app.core.triples import TripleCatalog  # noqa: E402


def _hue_library(count: int) -> FilamentLibrary:
    """``count`` spools spread around the hue circle, all one material type.

    Bambu Studio only mixes filaments of the same 耗材种类, so a probe library
    that wants mixes at all has to be single-material.
    """
    filaments = []
    for index in range(count):
        hue = (index / max(count, 1)) % 1.0
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.62, 0.86)
        hex_value = "#{:02X}{:02X}{:02X}".format(
            round(red * 255), round(green * 255), round(blue * 255)
        )
        filaments.append(
            Filament(
                name=f"探针 {index + 1:02d}",
                brand="探针",
                material_type="PETG HF",
                color_hex=hex_value,
            )
        )
    return FilamentLibrary(filaments)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spools", type=int, default=8, help="how many spools in the probe library")
    parser.add_argument("--image", default=str(PROJECT_ROOT / "samples" / "dajian-pig.png"))
    parser.add_argument("--colours", type=int, default=32, help="「颜色数量」")
    parser.add_argument("--dimension", type=int, default=512, help="「最大边长」")
    parser.add_argument("--spool-cap", type=int, default=4, help="「耗材上限」")
    parser.add_argument("--engine", default="bambu", choices=("bambu", "mixer"))
    parser.add_argument("--no-triples", action="store_true", help="skip the three-filament tail")
    parser.add_argument("--dither", action="store_true")
    arguments = parser.parse_args()

    engine = ENGINE_BAMBU if arguments.engine == "bambu" else ENGINE_MIXER
    library = _hue_library(arguments.spools)

    started = time.perf_counter()
    catalog = MixCatalog(library.filaments, engine=engine).build()
    palette = build_palette(library, catalog, include_mixes=True)
    build_seconds = time.perf_counter() - started
    print(f"library        : {len(library)} spools, engine {engine}")
    print(f"candidate head : {len(palette):,} entries ({build_seconds:.2f}s to build)")

    tail = ""
    search: object = palette
    if not arguments.no_triples:
        started = time.perf_counter()
        triples = TripleCatalog(library.filaments, engine=engine).build()
        triple_seconds = time.perf_counter() - started
        search = TriplePalette(palette, triples, library)
        tail = f", tail {len(triples.packed):,} colours ({triple_seconds:.2f}s to build)"
    print(f"search pool    : head + {tail or 'no tail'}")

    image = Image.open(arguments.image).convert("RGBA")
    settings = MatchSettings(
        max_colours=arguments.colours,
        max_dimension=arguments.dimension,
        dither=arguments.dither,
        max_spools=arguments.spool_cap,
    ).clamped()
    print(
        f"image          : {arguments.image} {image.width}x{image.height} "
        f"-> max {settings.max_dimension}px, {settings.max_colours} colours, "
        f"{settings.max_spools or '不限'} 卷, dither {settings.dither}"
    )

    marks: list[tuple[float, str]] = []
    last = time.perf_counter()

    def progress(message: str, fraction: float) -> None:
        nonlocal last
        now = time.perf_counter()
        marks.append((now - last, f"{fraction * 100:5.1f}%  {message}"))
        last = now

    started = time.perf_counter()
    result = match_image(image, search, settings, progress=progress)
    total = time.perf_counter() - started
    marks.append((time.perf_counter() - last, "          (after the last report)"))

    print("\nstep by step (seconds spent since the previous report):")
    for seconds, label in marks:
        print(f"  {seconds:6.2f}  {label}")
    print(f"  {total:6.2f}  TOTAL match_image")
    kept = result.palette
    print(
        f"\nkept           : {len(kept)} colours, "
        f"{result.printed_pixels:,} printed px, coverage {result.coverage:.1%}"
    )
    for index, entry in enumerate(kept):
        share = result.counts[index] / max(result.total_pixels, 1)
        print(f"  {index:2d}  {entry.color_hex}  {share:6.1%}  {entry.short_label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

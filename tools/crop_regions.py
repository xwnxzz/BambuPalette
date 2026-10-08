"""Crop and zoom regions of the Bambu Studio mixing screenshots for visual QA.

Usage:  python tools/crop_regions.py <image> <out.png> <x0> <y0> <x1> <y1> [<scale>]
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image


def main() -> int:
    source, destination = Path(sys.argv[1]), Path(sys.argv[2])
    x0, y0, x1, y1 = (int(v) for v in sys.argv[3:7])
    scale = int(sys.argv[7]) if len(sys.argv) > 7 else 2
    image = Image.open(source).convert("RGB").crop((x0, y0, x1, y1))
    image = image.resize((image.width * scale, image.height * scale), Image.NEAREST)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)
    print(f"{destination}  {image.width}x{image.height}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

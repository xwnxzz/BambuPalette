"""Measure the two Bambu Studio "添加混色耗材" screenshots pixel by pixel.

The dialog shows, for a two-filament mix:
  * a 效果预览 swatch (the colour at the selected ratio),
  * a 比例 gradient bar spanning 90% -> 10% of filament 1,
  * a 混色推荐 grid of suggested mixes.

The gradient bar is the interesting one: it is the model's own output sampled
across the whole ratio sweep, so its *shape* tells us whether the underlying
mixing maths is a pigment-level Kubelka-Munk blend (strongly non-linear, black
dominated) or a simple linear blend.

Usage:  python tools/measure_screenshots.py <image> [<image> ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _hex(rgb: np.ndarray) -> str:
    r, g, b = (int(v) for v in rgb[:3])
    return f"#{r:02X}{g:02X}{b:02X}"


def background_colour(px: np.ndarray) -> np.ndarray:
    """The dialog background: the single most common colour in the image."""
    flat = px.reshape(-1, 3)
    packed = (flat[:, 0].astype(np.int32) << 16) | (flat[:, 1].astype(np.int32) << 8) | flat[:, 2]
    values, counts = np.unique(packed, return_counts=True)
    best = values[counts.argmax()]
    return np.array([(best >> 16) & 255, (best >> 8) & 255, best & 255], dtype=np.uint8)


def foreground_mask(px: np.ndarray, bg: np.ndarray, tolerance: int = 12) -> np.ndarray:
    distance = np.abs(px.astype(np.int16) - bg.astype(np.int16)).max(axis=2)
    return distance > tolerance


def find_long_runs(mask: np.ndarray, min_length: int) -> list[tuple[int, int, int]]:
    """Return (row, start_x, end_x) for every row holding a horizontal run."""
    found: list[tuple[int, int, int]] = []
    for y in range(mask.shape[0]):
        row = mask[y]
        best_start = best_end = -1
        start = None
        for x in range(row.shape[0] + 1):
            inside = x < row.shape[0] and row[x]
            if inside and start is None:
                start = x
            elif not inside and start is not None:
                if x - start >= min_length and (x - start) > (best_end - best_start):
                    best_start, best_end = start, x
                start = None
        if best_start >= 0:
            found.append((y, best_start, best_end))
    return found


def group_rows(rows: list[tuple[int, int, int]]) -> list[tuple[int, int, int, int]]:
    """Collapse consecutive rows into (y0, y1, x0, x1) bands."""
    bands: list[tuple[int, int, int, int]] = []
    for y, x0, x1 in rows:
        if bands and y == bands[-1][1] + 1:
            by0, _, bx0, bx1 = bands[-1]
            bands[-1] = (by0, y, min(bx0, x0), max(bx1, x1))
        else:
            bands.append((y, y, x0, x1))
    return bands


def report(path: Path) -> None:
    image = Image.open(path).convert("RGB")
    px = np.asarray(image)
    height, width = px.shape[:2]
    bg = background_colour(px)
    print(f"=== {path.name}  {width}x{height}  background={_hex(bg)}")

    mask = foreground_mask(px, bg)
    wide = [row for row in find_long_runs(mask, min_length=width // 3)]
    for y0, y1, x0, x1 in group_rows(wide):
        if y1 - y0 < 4:
            continue
        mid = (y0 + y1) // 2
        strip = px[mid, x0:x1].astype(np.float64)
        has_ramp = strip.std(axis=0).max() > 6
        kind = "GRADIENT" if has_ramp else "flat"
        print(f"  band y={y0}..{y1} x={x0}..{x1} h={y1 - y0 + 1}  {kind}")
        if not has_ramp:
            print(f"      colour {_hex(np.median(strip, axis=0))}")

    # The gradient bar is the widest ramp band; sample it at fixed fractions.
    ramps = [b for b in group_rows(wide) if b[1] - b[0] >= 4]
    for y0, y1, x0, x1 in ramps:
        mid = (y0 + y1) // 2
        row = px[mid, x0:x1].astype(np.float64)
        if row.std(axis=0).max() <= 6:
            continue
        print(f"  ramp samples (y={mid}, x={x0}..{x1}):")
        for i in range(11):
            j = int(round(i / 10 * (len(row) - 1)))
            lo, hi = max(0, j - 2), min(len(row), j + 3)
            mean = row[lo:hi].mean(axis=0)
            cumulative = j / (len(row) - 1) * 100
            print(f"      {i * 10:3d}% of bar ({cumulative:5.1f}%) -> {_hex(mean)}")

    # Preview swatch: the biggest square-ish foreground blob in the left column.
    left = mask[:, : int(width * 0.33)]
    tall = [row for row in find_long_runs(left, min_length=20)]
    for y0, y1, x0, x1 in group_rows(tall):
        if y1 - y0 < 20:
            continue
        block = px[y0:y1 + 1, x0:x1 + 1].reshape(-1, 3)
        median = np.median(block, axis=0)
        print(f"  blob y={y0}..{y1} x={x0}..{x1} h={y1 - y0 + 1} w={x1 - x0 + 1} -> {_hex(median)}")


def main() -> int:
    for argument in sys.argv[1:]:
        report(Path(argument))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

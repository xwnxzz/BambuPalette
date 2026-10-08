"""Fit the 比例 gradient bar in the Bambu Studio mixing screenshots.

The bar shows the mixed colour sampled across the whole ratio sweep. Its shape
discriminates between candidate mixing models:

  * a linear blend in sRGB display space  -> each channel is a straight line in x
  * a linear blend in linear-light RGB    -> straight after undoing the sRGB curve
  * pigment-level Kubelka-Munk (the port of OrcaSlicer-FullSpectrum)
                                          -> a violent curve, near black almost
                                             immediately once any dark filament
                                             is present

Usage:  python tools/analyse_mix_bar.py <image> [<image> ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image


def srgb_to_linear(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64) / 255.0
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(v: np.ndarray) -> np.ndarray:
    v = np.clip(np.asarray(v, dtype=np.float64), 0.0, 1.0)
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * v ** (1 / 2.4) - 0.055) * 255.0


def locate_bar(px: np.ndarray) -> tuple[int, int, int]:
    """Return (row, x0, x1) of the widest smooth horizontal colour ramp."""
    height, width = px.shape[:2]
    best: tuple[int, int, int] = (-1, -1, -1)
    for y in range(30, height):
        row = px[y].astype(np.float64)
        # A bar row is non-uniform but locally smooth: no hard edges except ends.
        row_std = row.std(axis=0).max()
        if row_std < 25:
            continue
        ramp = np.abs(np.diff(row, axis=0)).max(axis=1)
        # Count pixels that are NOT part of a long smooth run.
        interior = ramp[1:-1]
        if interior.size == 0:
            continue
        abrupt = int((interior > 6).sum())
        if abrupt > 4:
            continue
        span = width
        if span > (best[2] - best[1]):
            best = (y, 0, width)
    return best


def report(path: Path) -> None:
    image = Image.open(path).convert("RGB")
    px = np.asarray(image)
    height, width = px.shape[:2]
    print(f"=== {path.name}  {width}x{height}")

    # Locate the bar as the row with the longest run of non-white pixels whose
    # interior has no abrupt jumps.
    candidates: list[tuple[int, int, int, int]] = []
    background = px[2, 2].astype(np.int16)
    for y in range(height):
        row = px[y].astype(np.int16)
        far = np.abs(row - background).max(axis=1) > 12
        run_start = run_end = -1
        start = None
        for x in range(width + 1):
            inside = x < width and far[x]
            if inside and start is None:
                start = x
            elif not inside and start is not None:
                if run_end - run_start < x - start:
                    run_start, run_end = start, x
                start = None
        if run_start < 0 or run_end - run_start < 150:
            continue
        segment = px[y, run_start:run_end].astype(np.float64)
        if segment.std(axis=0).max() < 20:
            continue  # a flat swatch, not a gradient
        interior_jump = np.abs(np.diff(segment, axis=0)).max(axis=1)[1:-1]
        abrupt = int((interior_jump > 6).sum()) if interior_jump.size else 0
        candidates.append((y, run_start, run_end, abrupt))

    bars = [c for c in candidates if c[3] <= 4]
    if not bars:
        print("  no clean ramp row found")
        return
    # Prefer the tallest contiguous band; here just take the median row of the
    # band that contains the most clean rows.
    rows = sorted(c[0] for c in bars)
    band = [rows[0]]
    bands: list[list[int]] = []
    for y in rows[1:]:
        if y == band[-1] + 1:
            band.append(y)
        else:
            bands.append(band)
            band = [y]
    bands.append(band)
    band = max(bands, key=len)
    chosen = [c for c in bars if c[0] in band]
    y = chosen[len(chosen) // 2][0]
    x0 = int(np.median([c[1] for c in chosen]))
    x1 = int(np.median([c[2] for c in chosen]))
    # Trim the borders of the widget.
    x0 += 3
    x1 -= 3
    print(f"  bar band rows {band[0]}..{band[-1]}, sampled row {y}, x {x0}..{x1}")

    segment = px[y, x0:x1].astype(np.float64)
    position = np.linspace(0.0, 1.0, segment.shape[0])

    for name, values in (
        ("sRGB", segment),
        ("linear", srgb_to_linear(segment)),
    ):
        for channel, label in enumerate("RGB"):
            slope, intercept = np.polyfit(position, values[:, channel], 1)
            predicted = slope * position + intercept
            residual = values[:, channel] - predicted
            rms = float(np.sqrt((residual ** 2).mean()))
            # A straight line leaves a random residual; a curve leaves a slow
            # systematic swing. Compare the smoothed residual to its own RMS.
            window = max(5, values.shape[0] // 10)
            kernel = np.ones(window) / window
            smoothed = np.convolve(residual, kernel, mode="valid")
            swing = float(smoothed.max() - smoothed.min())
            print(
                f"  {name:>6} {label}: slope={slope:+8.3f}  rms_resid={rms:6.2f}"
                f"  systematic_swing={swing:6.2f}"
            )

    print("  bar endpoints:")
    for fraction, index in ((0.0, 0), (0.5, len(segment) // 2), (1.0, len(segment) - 1)):
        value = segment[index]
        print(f"      {fraction:>4.0%} -> #{int(value[0]):02X}{int(value[1]):02X}{int(value[2]):02X}")


def main() -> int:
    for argument in sys.argv[1:]:
        report(Path(argument))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

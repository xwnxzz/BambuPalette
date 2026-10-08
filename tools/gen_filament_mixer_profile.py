"""Bake ``filament_mixer``'s polynomial coefficients into a Python module.

The mixer Bambu Studio 02.08 uses for previewing a mixed filament is the
header-only ``filament_mixer`` library (MIT, Copyright (c) 2026 Justin Hayes),
which approaches Mixbox behaviour with a degree-4 polynomial regression over
330 features and 7 inputs -- Mean Delta-E ~2.07 against Mixbox.  Bambu bundles
it as ``src/libslic3r/FilamentMixerModel.hpp`` and calls it through
``Slic3r::filament_mixer_lerp``.

This script parses that header and emits the coefficient tables as NumPy
arrays so the application carries no C++ and no file it cannot read.  The
reference header itself is kept out of the shipped bundle.

Usage:
    python tools/gen_filament_mixer_profile.py            # rewrite the module
    python tools/gen_filament_mixer_profile.py --check     # verify only
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REFERENCE = ROOT / "tools" / "reference" / "FilamentMixerModel.hpp"
TARGET = ROOT / "app" / "spectral" / "filament_mixer_profile.py"

# The upstream header documents these two values as its own smoke test.
SAMPLE_INPUT = (0, 33, 133, 252, 211, 0, 0.5)
SAMPLE_OUTPUT = (47, 141, 56)

EXPECTED_FEATURES = 330
EXPECTED_INPUTS = 7


def _block(text: str, header: str) -> str:
    """Return the brace-delimited body that follows ``header``."""
    start = text.index(header)
    start = text.index("{", start)
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : index]
    raise ValueError(f"unbalanced braces after {header!r}")


def _rows(body: str, width: int, label: str) -> list[list[float]]:
    rows: list[list[float]] = []
    for group in re.findall(r"\{([^{}]*)\}", body):
        values = [float(part) for part in group.split(",") if part.strip()]
        if len(values) != width:
            raise ValueError(f"{label}: expected {width} values, found {len(values)}")
        rows.append(values)
    return rows


def parse(path: Path) -> tuple[list[list[int]], list[list[float]], list[float]]:
    text = path.read_text(encoding="utf-8")
    text = text.replace("// BEGIN AUTO-GENERATED COEFFICIENTS", "")

    powers = _rows(_block(text, "POWERS["), EXPECTED_INPUTS, "POWERS")
    coef = _rows(_block(text, "COEF["), 3, "COEF")
    intercept_body = _block(text, "INTERCEPT[")
    intercept = [float(part) for part in intercept_body.replace("\n", " ").split(",") if part.strip()]

    if len(powers) != EXPECTED_FEATURES:
        raise ValueError(f"POWERS: expected {EXPECTED_FEATURES} rows, found {len(powers)}")
    if len(coef) != EXPECTED_FEATURES:
        raise ValueError(f"COEF: expected {EXPECTED_FEATURES} rows, found {len(coef)}")
    if len(intercept) != 3:
        raise ValueError(f"INTERCEPT: expected 3 values, found {len(intercept)}")

    powers_int = [[int(value) for value in row] for row in powers]
    for row in powers_int:
        if any(value < 0 or value > 4 for value in row):
            raise ValueError(f"POWERS out of range: {row}")
        if sum(row) > 4:
            raise ValueError(f"POWERS row exceeds degree 4: {row}")
    return powers_int, coef, intercept


def reference_mix(
    powers: list[list[int]], coef: list[list[float]], intercept: list[float], x: tuple[float, ...]
) -> tuple[int, int, int]:
    """A direct transcription of the header's ``lerp`` for verification."""
    out: list[int] = []
    for channel in range(3):
        total = intercept[channel]
        for index, row in enumerate(powers):
            feature = 1.0
            for exponent, base in zip(row, x):
                if exponent:
                    feature *= base**exponent
            total += feature * coef[index][channel]
        out.append(max(0, min(255, int(total))))
    return out[0], out[1], out[2]


def render(powers: list[list[int]], coef: list[list[float]], intercept: list[float]) -> str:
    power_rows = ",\n    ".join("(" + ", ".join(str(v) for v in row) + ")" for row in powers)
    coef_rows = ",\n    ".join(
        "(" + ", ".join(repr(v) for v in row) + ")" for row in coef
    )
    return f'''"""Polynomial coefficients for the ``filament_mixer`` pigment model.

GENERATED FILE -- do not edit by hand.  Regenerate with::

    python tools/gen_filament_mixer_profile.py

Source: the header-only ``filament_mixer`` library, MIT License,
Copyright (c) 2026 Justin Hayes, as bundled by Bambu Studio 02.08 as
``src/libslic3r/FilamentMixerModel.hpp`` (version ``v02.08.02.61``).
The model is a degree-4 polynomial regression over {EXPECTED_FEATURES} features and
{EXPECTED_INPUTS} inputs, fitted to approximate Mixbox behaviour.

The input vector is ``(r1, g1, b1, r2, g2, b2, t)`` with the colours in 0-255
and ``t`` in 0.0-1.0, where ``t = 0`` returns the first colour and ``t = 1``
the second.
"""

from __future__ import annotations

import numpy as np

POLY_DEGREE = 4
N_FEATURES = {EXPECTED_FEATURES}
N_INPUTS = {EXPECTED_INPUTS}

#: ``(r1, g1, b1, r2, g2, b2, t)`` -- the documented smoke test of the source.
SAMPLE_INPUT = {SAMPLE_INPUT!r}
#: What ``filament_mixer::lerp`` returns for :data:`SAMPLE_INPUT` (blue + yellow -> green).
SAMPLE_OUTPUT = {SAMPLE_OUTPUT!r}

#: Exponent applied to each input for every feature: shape ``({EXPECTED_FEATURES}, {EXPECTED_INPUTS})``.
POWERS = np.array(
    [
    {power_rows}
    ],
    dtype=np.int64,
)

#: Regression coefficients: shape ``({EXPECTED_FEATURES}, 3)``.
COEF = np.array(
    [
    {coef_rows}
    ],
    dtype=np.float64,
)

#: Per-channel offset added before the dot product.
INTERCEPT = np.array({intercept!r}, dtype=np.float64)

__all__ = [
    "COEF",
    "INTERCEPT",
    "N_FEATURES",
    "N_INPUTS",
    "POLY_DEGREE",
    "POWERS",
    "SAMPLE_INPUT",
    "SAMPLE_OUTPUT",
]
'''


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify without rewriting")
    args = parser.parse_args(argv[1:])

    if not REFERENCE.exists():
        print(f"reference header not found: {REFERENCE}")
        print("download it with tools/decode_github_content.py from FilamentMixerModel.hpp")
        return 1

    powers, coef, intercept = parse(REFERENCE)

    got = reference_mix(powers, coef, intercept, SAMPLE_INPUT)
    print(f"features      {len(powers)} x {len(powers[0])}")
    print(f"coefficients  {len(coef)} x {len(coef[0])}")
    print(f"sample lerp   {SAMPLE_INPUT[:6]} t={SAMPLE_INPUT[6]} -> {got}")
    if got != SAMPLE_OUTPUT:
        print(f"FAIL: the source documents {SAMPLE_OUTPUT}")
        return 1
    print("sample check  OK")

    rendered = render(powers, coef, intercept)
    if args.check:
        if not TARGET.exists():
            print(f"FAIL: {TARGET} does not exist")
            return 1
        if TARGET.read_text(encoding="utf-8") != rendered:
            print(f"FAIL: {TARGET} is stale; regenerate it")
            return 1
        print(f"check         {TARGET.name} is up to date")
        return 0

    TARGET.write_text(rendered, encoding="utf-8")
    print(f"wrote         {TARGET} ({len(rendered):,} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

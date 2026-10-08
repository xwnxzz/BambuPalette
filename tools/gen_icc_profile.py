#!/usr/bin/env python3
"""Generate ``app/spectral/icc_profile.py`` from the upstream ICC polynomial profile header.

Input : ``tools/reference/FullSpectrumICCPolynomialProfile.h``
Output: ``app/spectral/icc_profile.py``

The header is the generated artefact distributed with
``ratdoux/OrcaSlicer-FullSpectrum`` (``src/libslic3r/FullSpectrumICCPolynomialProfile.h``),
which in turn is derived from the International Color Consortium's
"Munsell Glossy D50 XYZ polynomial estimator" ICC profile.

Run from the project root::

    python tools/gen_icc_profile.py

The script is dependency-free and idempotent; re-running it rewrites the module
with byte-identical content when the source header is unchanged.
"""

from __future__ import annotations

import re
import struct
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_HEADER = PROJECT_ROOT / "tools" / "reference" / "FullSpectrumICCPolynomialProfile.h"
TARGET = PROJECT_ROOT / "app" / "spectral" / "icc_profile.py"

TERM_COUNT = 20
SPECTRUM_SIZE = 31
FIRST_WAVELENGTH_NM = 400
LAST_WAVELENGTH_NM = 700
WAVELENGTH_STEP_NM = 10

FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def _f32(value: float) -> float:
    """Round to IEEE-754 binary32, matching the C++ header's ``float`` literals.

    Upstream declares ``COEFFICIENTS`` as ``std::array<std::array<float, 20>, 31>``
    and promotes each entry to ``double`` during accumulation. Round-tripping
    through binary32 here reproduces those promoted values exactly.
    """
    return struct.unpack("<f", struct.pack("<f", value))[0]

# Term order is fixed by the upstream `polynomial_terms()` implementation.
TERM_ORDER = (
    "1", "X", "Y", "Z",
    "X2", "Y2", "Z2", "XY", "XZ", "YZ",
    "X3", "Y3", "Z3", "X2Y", "X2Z", "XY2", "XZ2", "Y2Z", "YZ2", "XYZ",
)

# The upstream CIE 1931 10-degree observer (400..700 nm, 10 nm) and the D65
# relative spectral power distribution are emitted here verbatim so the runtime
# engine needs no external tables.
CIEXYZ_10DEG_400_700_10NM = (
    (0.019110, 0.002004, 0.086011), (0.084736, 0.008756, 0.389366),
    (0.204492, 0.021391, 0.972542), (0.314679, 0.038676, 1.553480),
    (0.383734, 0.062077, 1.967280), (0.370702, 0.089456, 1.994800),
    (0.302273, 0.128201, 1.745370), (0.195618, 0.185190, 1.317560),
    (0.080507, 0.253589, 0.772125), (0.016172, 0.339133, 0.415254),
    (0.003816, 0.460777, 0.218502), (0.037465, 0.606741, 0.112044),
    (0.117749, 0.761757, 0.060709), (0.236491, 0.875211, 0.030451),
    (0.376772, 0.961988, 0.013676), (0.529826, 0.991761, 0.003988),
    (0.705224, 0.997340, 0.000000), (0.878655, 0.955552, 0.000000),
    (1.014160, 0.868934, 0.000000), (1.118520, 0.777405, 0.000000),
    (1.124000, 0.658341, 0.000000), (1.030480, 0.527963, 0.000000),
    (0.856297, 0.398057, 0.000000), (0.647467, 0.283493, 0.000000),
    (0.431567, 0.179828, 0.000000), (0.268329, 0.107633, 0.000000),
    (0.152568, 0.060281, 0.000000), (0.081261, 0.031800, 0.000000),
    (0.040851, 0.015905, 0.000000), (0.019941, 0.007749, 0.000000),
    (0.009577, 0.003718, 0.000000),
)

D65_400_700_10NM = (
    82.7549, 91.4860, 93.4318, 86.6823, 104.8650, 117.0080, 117.8120,
    114.8610, 115.9230, 108.8110, 109.3540, 107.8020, 104.7900, 107.6890,
    104.4050, 104.0460, 100.0000, 96.3342, 95.7880, 88.6856, 90.0062,
    89.5991, 87.6987, 83.2886, 83.6992, 80.0268, 80.2146, 82.2778, 78.2842,
    69.7213, 71.6091,
)

SOURCE_PROFILE_SHA256 = "8291983ea02ca7b7adf023a1f3ddd3fc618a853ef20ffd01eb145504a92ff2e4"
SOURCE_PROFILE_ID = "5436fbfce5f7414dc520bb6e5d9c1516"
SOURCE_PROFILE_URL = "https://www.color.org/resources/spectral/xyz2PolyEstimateRefV2.icc"
SOURCE_PROFILE_LICENSE = "https://registry.color.org/profile-library/"
SOURCE_HEADER_REPO = "https://github.com/ratdoux/OrcaSlicer-FullSpectrum"


def parse_coefficients(text: str) -> list[list[float]]:
    """Extract the 31x20 coefficient table from the C++ header."""
    anchor = text.index("COEFFICIENTS = {{")
    end = text.index("}};", anchor)
    body = text[anchor + len("COEFFICIENTS = {{"):end]
    # Drop block delimiters so only numeric literals remain.
    body = body.replace("{", " ").replace("}", " ")
    body = re.sub(r"//[^\n]*", " ", body)

    values: list[float] = []
    for token in body.split(","):
        token = token.strip()
        if not token:
            continue
        if not token.endswith("f"):
            raise ValueError(f"unexpected token in coefficient table: {token!r}")
        token = token[:-1]
        match = FLOAT_RE.fullmatch(token)
        if not match:
            raise ValueError(f"cannot parse coefficient literal: {token!r}")
        values.append(_f32(float(token)))

    expected = SPECTRUM_SIZE * TERM_COUNT
    if len(values) != expected:
        raise ValueError(f"expected {expected} coefficients, parsed {len(values)}")

    return [values[i * TERM_COUNT:(i + 1) * TERM_COUNT] for i in range(SPECTRUM_SIZE)]


def _format_table(table: list[list[float]]) -> str:
    lines = []
    for row in table:
        cells = ", ".join(repr(value) for value in row)
        lines.append(f"    ({cells}),")
    return "\n".join(lines)


def render(coefficients: list[list[float]]) -> str:
    return f'''"""ICC polynomial reflectance estimator coefficients — GENERATED FILE, DO NOT EDIT.

Generated by ``tools/gen_icc_profile.py`` from
``tools/reference/FullSpectrumICCPolynomialProfile.h``
({SOURCE_HEADER_REPO}, ``src/libslic3r/FullSpectrumICCPolynomialProfile.h``).

That header is derived from the International Color Consortium's
"Munsell Glossy D50 XYZ polynomial estimator" ICC profile:

    source profile : {SOURCE_PROFILE_URL}
    SHA-256        : {SOURCE_PROFILE_SHA256}
    profile ID     : {SOURCE_PROFILE_ID}
    copyright      : Copyright 2022 International Color Consortium.
    profile license: {SOURCE_PROFILE_LICENSE}

The estimator maps a D50 XYZ tristimulus value (Y = 100 scale) to a reflectance
spectrum of {SPECTRUM_SIZE} samples covering {FIRST_WAVELENGTH_NM}-{LAST_WAVELENGTH_NM} nm in
{WAVELENGTH_STEP_NM} nm steps, using a 20-term polynomial basis.
"""

from __future__ import annotations

TERM_COUNT = {TERM_COUNT}
SPECTRUM_SIZE = {SPECTRUM_SIZE}
FIRST_WAVELENGTH_NM = {FIRST_WAVELENGTH_NM}
LAST_WAVELENGTH_NM = {LAST_WAVELENGTH_NM}
WAVELENGTH_STEP_NM = {WAVELENGTH_STEP_NM}

SOURCE_PROFILE_SHA256 = "{SOURCE_PROFILE_SHA256}"
SOURCE_PROFILE_ID = "{SOURCE_PROFILE_ID}"
SOURCE_PROFILE_URL = "{SOURCE_PROFILE_URL}"
SOURCE_PROFILE_LICENSE = "{SOURCE_PROFILE_LICENSE}"
SOURCE_HEADER_REPO = "{SOURCE_HEADER_REPO}"

#: Basis order, identical to the upstream ``polynomial_terms()``.
TERM_ORDER = {TERM_ORDER!r}

#: COEFFICIENTS[wavelength_index][term_index], 31 rows x 20 columns.
COEFFICIENTS: tuple[tuple[float, ...], ...] = (
{_format_table(coefficients)}
)

#: CIE 1931 10-degree standard observer, 400-700 nm at 10 nm, as (x_bar, y_bar, z_bar).
CIEXYZ_10DEG_400_700_10NM: tuple[tuple[float, float, float], ...] = (
{chr(10).join("    " + repr(row) + "," for row in CIEXYZ_10DEG_400_700_10NM)}
)

#: CIE standard illuminant D65 relative spectral power distribution, same sampling.
D65_400_700_10NM: tuple[float, ...] = (
{chr(10).join("    " + ", ".join(repr(v) for v in D65_400_700_10NM[i:i + 5]) + "," for i in range(0, len(D65_400_700_10NM), 5))}
)
'''


def main() -> int:
    if not SOURCE_HEADER.exists():
        print(f"missing source header: {SOURCE_HEADER}", file=sys.stderr)
        return 1

    text = SOURCE_HEADER.read_text(encoding="utf-8")
    coefficients = parse_coefficients(text)

    # Sanity: the table must contain the upstream sentinel values.
    assert len(coefficients) == SPECTRUM_SIZE
    assert all(len(row) == TERM_COUNT for row in coefficients)

    rendered = render(coefficients)
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"wrote {TARGET} ({len(rendered)} chars, {SPECTRUM_SIZE}x{TERM_COUNT} coefficients)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

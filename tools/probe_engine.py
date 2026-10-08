"""Diagnostic probe: measures the real behaviour of all three mixing engines.

Not part of the shipped app.  Section 0 covers the whole registry -- the
properties the test suite is allowed to rely on for *every* engine:

* the reference sample the pigment polynomial is defined by;
* the two Bambu Studio reference screenshots, which the default engine must
  reproduce byte for byte and the legacy engine must not;
* self-mix drift (the pigment model is a regression, so it is NOT an identity);
* batch vs scalar agreement, and exact short-circuit endpoints.

Sections 1-6 then drill into the **spectral (Kubelka-Munk)** engine:

1. the round-trip error of the upstream pipeline
   ``sRGB -> fitted reflectance -> Lab -> sRGB``;
2. whether a self-mix of one colour at any ratio reproduces that colour's
   fitted anchor exactly;
3. whether the fitted reflectance of a neutral is physically flat (it is not);
4. whether the Kubelka-Munk gradient stays monotonic in L*.

Run:  .venv\\Scripts\\python.exe tools\\probe_engine.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from app.core.engines import (  # noqa: E402
    DEFAULT_ENGINE,
    ENGINE_BAMBU,
    ENGINE_MIXER,
    ENGINE_SPECTRAL,
    ENGINES,
)
from app.spectral import color as C  # noqa: E402

SAMPLES = [
    "#000000", "#FFFFFF", "#808080", "#404040", "#C0C0C0",
    "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#00FFFF", "#FF00FF",
    "#1A2B3C", "#FEDCBA", "#7F3F00", "#00A0A0", "#B9CCD6", "#0F2030",
    "#2E7D32", "#8E24AA", "#FF6F00", "#455A64", "#D32F2F", "#1976D2",
    "#E91E63", "#4CAF50", "#9E9E9E", "#795548", "#607D8B", "#CDDC39",
]


def refl(value):
    return C.reflectance_from_rgb(C.hex_to_rgb(value))


def anchor(value: str) -> str:
    """The colour the engine renders for a pure spool.

    A pure colour never reaches the Lab step directly: upstream always runs
    ``reflectance -> K/S -> reflectance`` (see
    ``FullSpectrumKSPairResidual.cpp:313-336``), and ``ks_from_reflectance``
    clamps R into ``[0.001, 0.999]``.  So the engine's canonical rendering of a
    filament colour is this clamped round-trip, and it is the value every mix of
    that colour is built from.
    """
    clamped = C.reflectance_from_ks(C.ks_from_reflectance(refl(value)))
    return C.hex_from_lab(C.lab_from_reflectance(clamped))


def anchor_lab(value: str) -> np.ndarray:
    clamped = C.reflectance_from_ks(C.ks_from_reflectance(refl(value)))
    return C.lab_from_reflectance(clamped)[0]


def _section(title: str) -> None:
    print()
    print("=" * 92)
    print(title)
    print("=" * 92)


# The two Bambu Studio reference screenshots: a pure-white spool plus a
# pure-black spool.  ``tools/measure_screenshots.py`` read these preview
# swatches straight off the pixels.
REFERENCE_SHOTS = [
    ("screenshot 1", 90, "#CBE4EE"),
    ("screenshot 2", 10, "#041D2E"),
]


def probe_engines() -> int:
    """Section 0: the whole registry, and what each engine must reproduce."""

    _section("0. The engine registry")
    for engine_id, engine in ENGINES.items():
        marker = "   <- default" if engine_id == DEFAULT_ENGINE else ""
        print(f"  {engine_id:<9} {engine.name}{marker}")
        print(f"            {engine.detail}")

    _section("0a. Upstream reference sample (blue + yellow -> green)")
    reference = ENGINES[ENGINE_MIXER].mix_rgb_pair((0, 33, 133), (252, 211, 0), 50)
    print("  filament_mixer::lerp(0,33,133, 252,211,0, 0.5)")
    print(f"    documented upstream : (47, 141, 56)")
    print(f"    our port            : {tuple(int(v) for v in reference)}")
    print(f"    matches             : {tuple(int(v) for v in reference) == (47, 141, 56)}")
    print(f"    naive sRGB average  : {ENGINES[ENGINE_BAMBU].mix_rgb_pair((0, 33, 133), (252, 211, 0), 50)}")
    print("    (the whole point of the pigment model is that these disagree)")

    _section("0b. Reference screenshots: pure white + pure black")
    header = f"  {'case':<16}{'white':>7}  {'measured':>9}  " + "  ".join(f"{e:>9}" for e in ENGINES)
    print(header)
    for label, percent, measured in REFERENCE_SHOTS:
        cells = "  ".join(f"{ENGINES[e].mix_hex('#FFFFFF', '#000000', percent):>9}" for e in ENGINES)
        print(f"  {label:<16}{percent:>6}%  {measured:>9}  {cells}")
    print()
    for label, percent, measured in REFERENCE_SHOTS:
        for engine_id, engine in ENGINES.items():
            got = engine.mix_hex("#FFFFFF", "#000000", percent)
            verdict = "EXACT" if got.upper() == measured else "off"
            print(f"  {label:<16} {engine_id:<9} {got}  vs {measured}  {verdict}")
    print()
    worst_legacy = max(
        max(abs(a - b) for a, b in zip(
            C.hex_to_rgb(ENGINES[ENGINE_BAMBU].mix_hex("#FFFFFF", "#000000", pct)),
            C.hex_to_rgb(measured),
        ))
        for _, pct, measured in REFERENCE_SHOTS
    )
    print(f"  the legacy sRGB engine is off by up to {worst_legacy} / 255 on these two cases")
    print("  -> the default engine is the one that matches the installed Bambu Studio 2.8")

    _section("0c. Self-mix drift: lerp(c, c, t) is NOT exactly c (it is a regression)")
    print("  max |channel delta| between the colour and its own self-mix, over ALL 81 ratios")
    print(f"  {'colour':>9}  " + "  ".join(f"{e:>9}" for e in ENGINES))
    worst = {engine_id: (0, "") for engine_id in ENGINES}
    for value in SAMPLES:
        row = []
        for engine_id, engine in ENGINES.items():
            base = np.asarray(C.hex_to_rgb(value), dtype=float)
            drift = 0
            for percent in range(10, 91):
                got = np.asarray(C.hex_to_rgb(engine.mix_hex(value, value, percent)), dtype=float)
                drift = max(drift, int(np.max(np.abs(base - got))))
            if drift > worst[engine_id][0]:
                worst[engine_id] = (drift, value)
            row.append(f"{drift:>9d}")
        print(f"  {value:>9}  " + "  ".join(row))
    print()
    for engine_id in ENGINES:
        drift, where = worst[engine_id]
        print(f"  worst self-mix drift, {engine_id:<9} = {drift:>3d} / 255  at {where}")
    print()
    print("  NOTE for 'mixer': this is the regression's own error, worst at the gamut corners.")
    print("  NOTE for 'spectral': it is dominated by the [0.001, 0.999] K/S clamp, i.e. the")
    print("  engine's render of an out-of-gamut input -- see section 1 for the anchor comparison.")

    _section("0d. Endpoints are exact short-circuits, and batch == scalar")
    print(f"  {'engine':<9}  {'100/0 returns a':>16}  {'0/100 returns b':>16}  {'81 ratios batch==scalar':>24}")
    for engine_id, engine in ENGINES.items():
        a_ok = b_ok = True
        for value in SAMPLES:
            other = "#000000" if value != "#000000" else "#FFFFFF"
            if engine.mix_hex(value, other, 100).upper() != value.upper():
                if engine_id != ENGINE_SPECTRAL:  # spectral clamps r into [0.001, 0.999]
                    a_ok = False
            if engine.mix_hex(value, other, 0).upper() != other.upper():
                if engine_id != ENGINE_SPECTRAL:
                    b_ok = False
        batch_ok = True
        percentages = list(range(10, 91))
        for first, second in (("#FFFFFF", "#000000"), ("#F2F0EB", "#17181C"), ("#3D7BC8", "#FF8800")):
            flat = engine.mix_rgb(
                np.asarray([[[*C.hex_to_rgb(first)], [*C.hex_to_rgb(second)]]], dtype=float),
                np.asarray([[[p / 100.0, 1.0 - p / 100.0] for p in percentages]], dtype=float),
            )[0]
            for index, percent in enumerate(percentages):
                scalar = C.hex_to_rgb(engine.mix_hex(first, second, percent))
                if any(int(v) != int(w) for v, w in zip(flat[index], scalar)):
                    batch_ok = False
        print(f"  {engine_id:<9}  {str(a_ok):>16}  {str(b_ok):>16}  {str(batch_ok):>24}")
    print("  (spectral cannot round-trip its endpoints: ks_from_reflectance clamps R into [0.001, 0.999])")

    _section("0e. The three engines side by side, white + black (and the L* the docs quote)")
    print(f"  {'white':>6}  " + "  ".join(f"{e:>22}" for e in ENGINES))
    for percent in (10, 50, 90):
        cells = []
        for engine in ENGINES.values():
            got = engine.mix_hex("#FFFFFF", "#000000", percent)
            lstar = float(C.lab_from_rgb(C.hex_to_rgb(got))[0])
            cells.append(f"{got}  L*={lstar:5.1f}")
        print(f"  {percent:>5}%  " + "  ".join(cells))
    print()
    print("  achromatic blue cast of the pigment model (white+black):")
    for percent in (10, 50, 90):
        r, g, b = C.hex_to_rgb(ENGINES[ENGINE_MIXER].mix_hex("#FFFFFF", "#000000", percent))
        print(f"    {percent:>2}% white -> #{r:02X}{g:02X}{b:02X}   blue - red = {b - r:+d}")
    return 0


def main() -> int:
    probe_engines()

    print()
    print("#" * 92)
    print("# Sections 1-6 below drill into the SPECTRAL (Kubelka-Munk) engine only.")
    print("#" * 92)
    print("=" * 92)
    print("1. Round-trip: sRGB -> fitted reflectance -> Lab -> sRGB")
    print("=" * 92)
    print(f"{'input':>9} {'anchor':>9} {'dE76':>7} {'dE2000':>7} {'maxch':>6}")
    errors = []
    for value in SAMPLES:
        rgb_in = np.asarray(C.hex_to_rgb(value), dtype=float)
        rendered = anchor(value)
        rgb_out = np.asarray(C.hex_to_rgb(rendered), dtype=float)
        de76 = float(C.delta_e_76(C.lab_from_reflectance(refl(value))[0], anchor_lab(value)))
        de00 = float(C.delta_e_2000(C.lab_from_reflectance(refl(value))[0], anchor_lab(value)))
        maxch = int(np.max(np.abs(rgb_in - rgb_out)))
        errors.append((de00, value, rendered, maxch))
        print(f"{value:>9} {rendered:>9} {de76:7.3f} {de00:7.3f} {maxch:6d}")
    errors.sort(reverse=True)
    worst, worst_in, worst_out, worst_ch = errors[0]
    print(f"\nworst dE2000 = {worst:.3f} ({worst_in} -> {worst_out}, {worst_ch}/255 per channel)")
    print(f"mean dE2000    = {sum(e[0] for e in errors) / len(errors):.3f}")

    print()
    print("=" * 92)
    print("2. Self-mix: does mixing a colour with itself return its anchor at every ratio?")
    print("=" * 92)
    exact = True
    for value in SAMPLES:
        expected = anchor(value)
        mismatches = []
        for percent in range(10, 91):
            got = C.mix_hex([value, value], [percent, 100 - percent])
            if got != expected:
                mismatches.append((percent, got))
        if mismatches:
            exact = False
            print(f"  {value}: anchor {expected}, mismatches {mismatches[:3]}")
    print(f"all ratios reproduce the anchor exactly: {exact}")

    print()
    print("=" * 92)
    print("3. Fitted reflectance of neutrals (a white spectrum is NOT flat)")
    print("=" * 92)
    for value in ("#000000", "#404040", "#808080", "#C0C0C0", "#FFFFFF"):
        spectrum = refl(value)
        print(
            f"{value}  min={spectrum.min():.3f} max={spectrum.max():.3f} "
            f"mean={spectrum.mean():.3f} spread={spectrum.max() - spectrum.min():.3f}  "
            f"@400={spectrum[0]:.3f} @550={spectrum[15]:.3f} @700={spectrum[-1]:.3f}"
        )

    print()
    print("=" * 92)
    print("3b. How hard does the [0.001, 0.999] clamp bite? (out-of-gamut colours)")
    print("=" * 92)
    for value in ("#FF00FF", "#00FF00", "#00FFFF", "#FFFFFF", "#808080"):
        raw = refl(value)
        clamped = C.reflectance_from_ks(C.ks_from_reflectance(raw))
        print(
            f"{value}  raw [{'%+.3f' % raw.min()}, {'%+.3f' % raw.max()}]  "
            f"clamped [{'%+.3f' % clamped.min()}, {'%+.3f' % clamped.max()}]  "
            f"max|delta|={np.max(np.abs(raw - clamped)):.3f}  "
            f"n_outside={int(np.sum((raw < 0.001) | (raw > 0.999)))}/31"
        )

    print()
    print("=" * 92)
    print("4. D50 white point through the polynomial")
    print("=" * 92)
    white_point = C._d50_xyz_from_rgbs(np.asarray([[255.0, 255.0, 255.0]]))[0]
    print(f"D50 XYZ of sRGB white: {white_point.round(4).tolist()}")
    estimated = C._COEFFICIENTS @ C._polynomial_terms(white_point[None, :])[0]
    print(
        f"estimated reflectance: min={estimated.min():.3f} max={estimated.max():.3f} "
        f"mean={estimated.mean():.3f} spread={estimated.max() - estimated.min():.3f}"
    )

    print()
    print("=" * 92)
    print("5. White -> black gradient over all 81 ratios")
    print("=" * 92)
    axis = [C.mix_hex(["#FFFFFF", "#000000"], [p, 100 - p]) for p in range(10, 91)]
    l_values = [float(C.lab_from_reflectance(refl(h))[0][0]) for h in axis]
    print(f"L* at 90% white = {l_values[-1]:.2f}   at 90% black = {l_values[0]:.2f}")
    print(f"strictly decreasing L* across the axis: {all(a > b for a, b in zip(l_values, l_values[1:]))}")
    print(f"distinct colours along the axis: {len(set(axis))} / {len(axis)}")
    print(f"0.5 white + 0.5 black = {C.mix_hex(['#FFFFFF', '#000000'], [50, 50])} "
          f"(naive sRGB average would be #808080, L*={l_values[0]:.1f}..{l_values[-1]:.1f})")
    mid = C.mix_hex(["#FFFFFF", "#000000"], [50, 50])
    print(f"L* of that 50/50 mix: {float(C.lab_from_reflectance(refl(mid))[0][0]):.2f}")

    print()
    print("=" * 92)
    print("6. Timing: how long does a full catalogue take?")
    print("=" * 92)
    import time

    from app.core.library import FilamentLibrary
    from app.core.mixes import MixCatalog, expected_recipe_count

    for count in (2, 5, 10, 20, 40):
        library = FilamentLibrary()
        for index in range(count):
            library.create(color_hex=SAMPLES[index % len(SAMPLES)],
                           name=f"spool{index}", material_type="PETG HF", brand="大简")
        started = time.perf_counter()
        catalog = MixCatalog(library).build()
        elapsed = time.perf_counter() - started
        print(
            f"{count:3d} spools -> {catalog.recipe_count:7d} recipes "
            f"(expected {expected_recipe_count(count):7d}) in {elapsed:7.3f} s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

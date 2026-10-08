"""Identify the colour-mixing model behind Bambu Studio's 比例 gradient bar.

Two screenshots of the "添加混色耗材" dialog were measured pixel by pixel
(tools/measure_screenshots.py, tools/analyse_mix_bar.py). Each bar samples the
mixed colour across the ratio sweep, so the whole curve is available.

For every candidate mixing model we solve for the two parent colours that best
reproduce the measured bar, then compare residuals. The model that fits with
plausible parents AND a small residual is the one Bambu actually uses.

Candidate models
  srgb    linear blend of the gamma-encoded sRGB values  (Bambu 2.5.x dialog)
  mixer   the filament_mixer degree-4 pigment polynomial (Bambu 2.8.x dialog)
  linear  linear blend of linear-light RGB
  gamma2  linear blend of squared values (a plain "physical" blend)
  lab     linear blend in CIELAB
  km      additive Kubelka-Munk K/S blend (the OrcaSlicer-FullSpectrum port)
  kmgeo   geometric-mean K/S blend (linear blend of optical density)

Reading the bar: Bambu draws it as ``for x in width: blend_colors(col_a, col_b,
1.0 - x/width)``, so the LEFT end is the pure first filament and the right end
the pure second. Which 11 positions the measurement sampled is not recorded
exactly, so the sweep below refits every model under several span assumptions;
a model only wins if it wins under all of them.

Usage:  python tools/fit_mix_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.spectral import color as C  # noqa: E402
from app.spectral import filament_mixer as MX  # noqa: E402

# Measured bars, left to right.
BARS = {
    "shot1 (labels 90% -> 10%)": [
        "#C7E0EC", "#B2CBDD", "#9AB3CC", "#839DBB", "#6D87A8", "#567193",
        "#405C7C", "#2B4664", "#173149", "#021A2B", "#00070F",
    ],
    "shot2 (labels 10% -> 90%)": [
        "#E3FDFD", "#CEE6EF", "#B6CEE0", "#9DB6CF", "#86A0BE", "#708AAB",
        "#5A7597", "#456081", "#2F4A68", "#1A344E", "#082235",
    ],
}

# Span assumptions for the 11 samples, as fractions of the bar from left to right.
SPANS = {
    "10-90%": np.linspace(0.10, 0.90, 11),
    "05-95%": np.linspace(0.05, 0.95, 11),
    "02-98%": np.linspace(0.02, 0.98, 11),
    "00-100%": np.linspace(0.00, 1.00, 11),
}

FRACTIONS = SPANS["10-90%"]


def srgb_to_linear(values: np.ndarray) -> np.ndarray:
    v = np.clip(values, 0.0, 255.0) / 255.0
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(values: np.ndarray) -> np.ndarray:
    v = np.clip(values, 0.0, 1.0)
    out = np.where(v <= 0.0031308, v * 12.92, 1.055 * v ** (1 / 2.4) - 0.055)
    return np.clip(out, 0.0, 1.0) * 255.0


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    return C.lab_from_reflectance(C.reflectance_from_rgb(rgb))


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    return np.clip(C.lab_to_rgb(lab), 0.0, 255.0)


def model_srgb(c1, c2, t):
    return (1 - t) * c1 + t * c2


def model_linear(c1, c2, t):
    return linear_to_srgb((1 - t) * srgb_to_linear(c1) + t * srgb_to_linear(c2))


def model_gamma2(c1, c2, t):
    a = (np.clip(c1, 0, 255) / 255.0) ** 2.0
    b = (np.clip(c2, 0, 255) / 255.0) ** 2.0
    return np.sqrt((1 - t) * a + t * b) * 255.0


def model_lab(c1, c2, t):
    return lab_to_rgb((1 - t) * rgb_to_lab(c1) + t * rgb_to_lab(c2))


def _ks(rgb):
    refl = C.reflectance_from_rgb(np.clip(rgb, 0, 255))
    return C.ks_from_reflectance(refl)


def _rgb_from_ks(ks):
    return np.clip(C.lab_to_rgb(C.lab_from_reflectance(C.reflectance_from_ks(ks))), 0, 255)


def model_km(c1, c2, t):
    return _rgb_from_ks((1 - t) * _ks(c1) + t * _ks(c2))


def model_kmgeo(c1, c2, t):
    a, b = _ks(c1), _ks(c2)
    exponent = np.clip((1 - t), 1e-6, 1 - 1e-6)
    return _rgb_from_ks(np.exp(exponent * np.log(np.maximum(a, 1e-9)) + t * np.log(np.maximum(b, 1e-9))))


def model_mixer(c1, c2, t):
    return MX.mix_pair_batch(
        np.array([c1], dtype=np.float64),
        np.array([c2], dtype=np.float64),
        np.array([t], dtype=np.float64),
    )[0].astype(np.float64)


MODELS = {
    "srgb": model_srgb,
    "mixer": model_mixer,
    "linear": model_linear,
    "gamma2": model_gamma2,
    "lab": model_lab,
    "km": model_km,
    "kmgeo": model_kmgeo,
}


def predict(model, params: np.ndarray) -> np.ndarray:
    c1, c2 = params[:3], params[3:]
    if model is model_mixer:
        count = FRACTIONS.size
        return MX.mix_pair_batch(
            np.repeat(c1[None, :], count, axis=0),
            np.repeat(c2[None, :], count, axis=0),
            FRACTIONS,
        ).astype(np.float64)
    return np.array([model(c1, c2, float(t)) for t in FRACTIONS])


def fit(model, observed: np.ndarray) -> tuple[np.ndarray, float]:
    """Coordinate descent over the six parent-channel values."""
    best_params, best_error = None, float("inf")
    initials = [
        np.array([*observed[0], *observed[-1]], dtype=np.float64),
        np.array([250, 252, 255, 5, 10, 20], dtype=np.float64),
        np.array([255, 255, 255, 0, 0, 0], dtype=np.float64),
        np.array([*observed[0], 0, 0, 0], dtype=np.float64),
        np.array([200, 230, 250, 20, 40, 60], dtype=np.float64),
    ]
    for initial in initials:
        params = initial.copy()
        step = 64.0
        error = float(((predict(model, params) - observed) ** 2).mean())
        while step > 0.05:
            improved = False
            for index in range(6):
                for direction in (1.0, -1.0):
                    trial = params.copy()
                    trial[index] = np.clip(trial[index] + direction * step, 0.0, 255.0)
                    trial_error = float(((predict(model, trial) - observed) ** 2).mean())
                    if trial_error < error - 1e-9:
                        params, error, improved = trial, trial_error, True
            if not improved:
                step *= 0.5
        if error < best_error:
            best_params, best_error = params, error
    return best_params, best_error


def describe(params: np.ndarray) -> str:
    c1 = C.rgb_to_hex(np.round(params[:3]).astype(int))
    c2 = C.rgb_to_hex(np.round(params[3:]).astype(int))
    return f"c1={c1} c2={c2}"


def main() -> int:
    global FRACTIONS
    totals = {name: 0.0 for name in MODELS}

    for name, hexes in BARS.items():
        observed = np.array([C.hex_to_rgb(h) for h in hexes], dtype=np.float64)
        print(f"=== {name}")
        for span_name, span in SPANS.items():
            FRACTIONS = span
            results = []
            for model_name, model in MODELS.items():
                params, error = fit(model, observed)
                results.append((np.sqrt(error), model_name, params))
            best = min(results)[1]
            line = "  ".join(
                f"{model_name}={rms:5.2f}" for rms, model_name, _ in sorted(results, key=lambda r: r[1])
            )
            print(f"   span {span_name:>8}  {line}   <- best {best}")
            for rms, model_name, _ in results:
                totals[model_name] += rms
        # Detail for the two models that matter, under the middle span.
        FRACTIONS = SPANS["05-95%"]
        for model_name in ("srgb", "mixer"):
            params, error = fit(MODELS[model_name], observed)
            print(f"   [{model_name} @ 05-95%] rms={np.sqrt(error):5.2f}  {describe(params)}")
        print()

    print("=== total rms across both bars and all spans (lower is better)")
    for model_name, total in sorted(totals.items(), key=lambda item: item[1]):
        print(f"   {model_name:>7}  {total:8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

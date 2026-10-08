"""Spectral colour mixing engine.

This module is a line-by-line port of the generic (non-database) colour path used
by Bambu Studio's *add mixed filament* feature. The reference implementation is
the FullSpectrum fork of OrcaSlicer, which Bambu Lab has publicly acknowledged as
the basis of its colour-prediction code:

    https://github.com/ratdoux/OrcaSlicer-FullSpectrum
    src/libslic3r/FullSpectrumICCPolynomialEstimator.cpp
    src/libslic3r/FullSpectrumKSPairResidual.cpp

Pipeline (identical to upstream):

    sRGB hex
      -> linear sRGB
      -> D65 XYZ                       (ICC sRGB colorimetric matrix)
      -> D50 XYZ                       (ICC c2sp linearised Bradford adaptation, x100)
      -> 20-term polynomial basis
      -> reflectance spectrum          (ICC "Munsell Glossy D50 XYZ polynomial estimator")
      -> Kubelka-Munk K/S              K/S = (1-R)^2 / 2R
      -> weighted K/S blend            weights = normalised print percentages
      -> reflectance                   R = 1 + f - sqrt(f^2 + 2f),  f = max(0, K/S)
      -> CIE 1931 10-degree Lab (D65)
      -> sRGB hex

Upstream skips the measured-material database whenever a colour has no known
material index (``material_index = nullopt``), in which case the pair/triple/
quadruple residual terms are all no-ops and ``material_composition`` stays zero.
That is exactly the situation for a user-entered filament colour, so the residual
machinery is intentionally not ported: this engine implements only the documented
generic branch.

Both scalar and vectorised entry points are provided; the vectorised ones are
used to enumerate the full 81-step gradient of every filament pair.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import icc_profile as _profile

SPECTRUM_SIZE = _profile.SPECTRUM_SIZE
TERM_COUNT = _profile.TERM_COUNT

# --------------------------------------------------------------------------------------
# Static matrices and tables
# --------------------------------------------------------------------------------------

#: ICC sRGB colorimetric profile: linear sRGB -> D65 XYZ (Y = 1 scale).
_SRGB_TO_D65 = np.array(
    [
        [0.412348487282463, 0.357601378848728, 0.180450133868809],
        [0.212617188755020, 0.715202757697456, 0.072180053547523],
        [0.019328835341365, 0.119200459616243, 0.950370705042392],
    ],
    dtype=np.float64,
)

#: ICC c2sp linearised Bradford adaptation, D65 -> D50 PCS.
_D65_TO_D50 = np.array(
    [
        [1.047907381710167, 0.022933384554211, -0.050201634798010],
        [0.029605959417717, 0.990456039910784, -0.017075529195870],
        [-0.009246794326782, 0.015062680140149, 0.751791232609078],
    ],
    dtype=np.float64,
)

_SRGB_TO_XYZ_D50 = _D65_TO_D50 @ _SRGB_TO_D65

#: Reflectance estimator coefficients, COEFFICIENTS[wave][term].
_COEFFICIENTS = np.array(_profile.COEFFICIENTS, dtype=np.float64)

#: CIE 1931 10-degree observer, shape (31, 3).
_CMF = np.array(_profile.CIEXYZ_10DEG_400_700_10NM, dtype=np.float64)

#: CIE D65 relative spectral power distribution, shape (31,).
_D65 = np.array(_profile.D65_400_700_10NM, dtype=np.float64)

# Pre-derived white point used by the reflectance -> Lab step.
_Y_WEIGHT = float(np.sum(_D65 * _CMF[:, 1]))
_XN_WEIGHT = float(np.sum(_D65 * _CMF[:, 0]))
_ZN_WEIGHT = float(np.sum(_D65 * _CMF[:, 2]))
_LAB_K = 100.0 / _Y_WEIGHT
_WHITE_X = _LAB_K * _XN_WEIGHT
_WHITE_Y = 100.0
_WHITE_Z = _LAB_K * _ZN_WEIGHT

# Lab -> sRGB uses the hard-coded D65 10-degree white point from upstream.
_D65_10_X = 94.811
_D65_10_Y = 100.0
_D65_10_Z = 107.304

#: XYZ(D65, Y=1) -> linear sRGB.
_XYZ_TO_SRGB = np.array(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ],
    dtype=np.float64,
)

_DELTA = 6.0 / 29.0
_DELTA3 = _DELTA * _DELTA * _DELTA

# --------------------------------------------------------------------------------------
# Transfer functions
# --------------------------------------------------------------------------------------


def srgb_to_linear(value: float) -> float:
    """Remove sRGB companding. ``value`` is on the 0..1 scale."""
    srgb = float(value)
    if srgb <= 0.04045:
        return srgb / 12.92
    return ((srgb + 0.055) / 1.055) ** 2.4


def srgb8_to_linear(value: float) -> float:
    """Remove sRGB companding from an 8-bit channel value (0..255)."""
    return srgb_to_linear(float(value) / 255.0)


def linear_to_srgb(value: float) -> float:
    """sRGB companding. Not clamped; callers clamp the result to 0..1."""
    if value <= 0.0031308:
        return 12.92 * value
    return 1.055 * (value ** (1.0 / 2.4)) - 0.055


def _srgb_to_linear_array(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb_array(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.where(values <= 0.0031308, 12.92 * values, 1.055 * np.power(values, 1.0 / 2.4) - 0.055)


# --------------------------------------------------------------------------------------
# Hex / RGB helpers
# --------------------------------------------------------------------------------------


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    """Parse ``#RRGGBB`` (case insensitive). Raises ``ValueError`` when malformed."""
    if not isinstance(value, str):
        raise ValueError(f"colour must be a string, got {type(value).__name__}")
    text = value.strip()
    if len(text) == 7 and text[0] == "#":
        body = text[1:]
    elif len(text) == 6:
        body = text
    else:
        raise ValueError(f"colour must be '#RRGGBB', got {value!r}")
    try:
        r = int(body[0:2], 16)
        g = int(body[2:4], 16)
        b = int(body[4:6], 16)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"colour must be '#RRGGBB', got {value!r}") from exc
    return (r, g, b)


def normalize_hex(value: str) -> str:
    """Return the canonical upper-case ``#RRGGBB`` form of ``value``."""
    r, g, b = hex_to_rgb(value)
    return f"#{r:02X}{g:02X}{b:02X}"


def rgb_to_hex(rgb) -> str:
    r, g, b = (int(max(0, min(255, round(float(c))))) for c in rgb)
    return f"#{r:02X}{g:02X}{b:02X}"


# --------------------------------------------------------------------------------------
# Spectral estimate
# --------------------------------------------------------------------------------------


def _d50_xyz_from_rgbs(rgbs: np.ndarray) -> np.ndarray:
    """``(N, 3)`` uint8/float sRGB -> ``(N, 3)`` D50 XYZ on the Y = 100 scale."""
    linear = _srgb_to_linear_array(np.asarray(rgbs, dtype=np.float64) / 255.0)
    return 100.0 * (linear @ _SRGB_TO_XYZ_D50.T)


def _polynomial_terms(xyz: np.ndarray) -> np.ndarray:
    """``(N, 3)`` D50 XYZ -> ``(N, 20)`` polynomial basis in upstream term order."""
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    x2, y2, z2 = x * x, y * y, z * z
    return np.stack(
        [
            np.ones_like(x), x, y, z,
            x2, y2, z2, x * y, x * z, y * z,
            x2 * x, y2 * y, z2 * z, x2 * y, x2 * z,
            x * y2, x * z2, y2 * z, y * z2, x * y * z,
        ],
        axis=1,
    )


def reflectances_from_rgbs(rgbs) -> np.ndarray:
    """Estimate reflectance spectra. ``(N, 3)`` sRGB -> ``(N, 31)`` reflectance."""
    xyz = _d50_xyz_from_rgbs(np.asarray(rgbs, dtype=np.float64))
    terms = _polynomial_terms(xyz)
    return terms @ _COEFFICIENTS.T


def reflectance_from_rgb(rgb) -> np.ndarray:
    """Estimate the reflectance spectrum of a single sRGB triple -> ``(31,)``."""
    return reflectances_from_rgbs(np.asarray([rgb], dtype=np.float64))[0]


# --------------------------------------------------------------------------------------
# Kubelka-Munk
# --------------------------------------------------------------------------------------


def ks_from_reflectance(reflectance) -> np.ndarray:
    """K/S = (1-R)^2 / 2R with R clamped to [0.001, 0.999] (upstream behaviour)."""
    r = np.clip(np.asarray(reflectance, dtype=np.float64), 0.001, 0.999)
    return ((1.0 - r) ** 2) / (2.0 * r)


def reflectance_from_ks(ks) -> np.ndarray:
    """R = clamp01(1 + f - sqrt(f^2 + 2f)) with f = max(0, K/S)."""
    f = np.maximum(0.0, np.asarray(ks, dtype=np.float64))
    return np.clip(1.0 + f - np.sqrt(f * f + 2.0 * f), 0.0, 1.0)


# --------------------------------------------------------------------------------------
# Reflectance -> Lab -> RGB
# --------------------------------------------------------------------------------------


def _lab_pivot(values: np.ndarray) -> np.ndarray:
    return np.where(values > _DELTA3, np.cbrt(values), values / (3.0 * _DELTA * _DELTA) + 4.0 / 29.0)


def lab_from_reflectance(reflectance) -> np.ndarray:
    """``(N, 31)`` reflectance -> ``(N, 3)`` CIE L*a*b* (D65, 10-degree observer)."""
    spectrum = np.atleast_2d(np.asarray(reflectance, dtype=np.float64))
    spectrum = np.maximum(0.0, spectrum)

    x = _LAB_K * np.sum(spectrum * _D65 * _CMF[:, 0], axis=1)
    y = _LAB_K * np.sum(spectrum * _D65 * _CMF[:, 1], axis=1)
    z = _LAB_K * np.sum(spectrum * _D65 * _CMF[:, 2], axis=1)

    fx = _lab_pivot(x / _WHITE_X)
    fy = _lab_pivot(y / _WHITE_Y)
    fz = _lab_pivot(z / _WHITE_Z)
    return np.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], axis=1)


def _pivot_lab_to_xyz(values: np.ndarray) -> np.ndarray:
    cubed = values * values * values
    return np.where(cubed > 0.008856, cubed, (values - 16.0 / 116.0) / 7.787)


def lab_to_rgb(lab) -> np.ndarray:
    """``(N, 3)`` Lab -> ``(N, 3)`` float sRGB in 0..255 (unrounded)."""
    values = np.atleast_2d(np.asarray(lab, dtype=np.float64))
    fy = (values[:, 0] + 16.0) / 116.0
    fx = values[:, 1] / 500.0 + fy
    fz = fy - values[:, 2] / 200.0

    x = _D65_10_X * _pivot_lab_to_xyz(fx) / 100.0
    y = _D65_10_Y * _pivot_lab_to_xyz(fy) / 100.0
    z = _D65_10_Z * _pivot_lab_to_xyz(fz) / 100.0

    linear = np.stack([x, y, z], axis=1) @ _XYZ_TO_SRGB.T
    srgb = np.clip(_linear_to_srgb_array(np.maximum(linear, 0.0)), 0.0, 1.0)
    return np.clip(srgb, 0.0, 1.0) * 255.0


_SRGB_TO_XYZ_D65 = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)

#: The white point of the whole pipeline.
#:
#: The CMF/illuminant tables this module mixes with are the CIE 1931 10° pair,
#: whose white is (94.811, 100.000, 107.304).  ``lab_to_rgb`` and
#: ``lab_from_reflectance`` already use it, so ``lab_from_rgb`` must use the very
#: same one — otherwise ``rgb_from_lab(lab_from_rgb(x))`` no longer returns ``x``
#: and saturated colours drift by up to 17 sRGB steps.
_WHITE_D65 = np.array([_D65_10_X / 100.0, _D65_10_Y / 100.0, _D65_10_Z / 100.0], dtype=np.float64)


def lab_from_rgb(rgb) -> np.ndarray:
    """CIELAB of 8-bit sRGB byte values, on this module's D65 10° white point.

    Accepts a single ``(3,)`` triple or any leading shape and converts each
    element.  Used so that every mixing engine can be compared and sorted in one
    consistent colour space regardless of how it produced its bytes — including
    the byte path of the sRGB engine, whose output never touches a spectrum.
    """
    values = np.asarray(rgb, dtype=np.float64) / 255.0
    linear = _srgb_to_linear_array(values)
    xyz = linear @ _SRGB_TO_XYZ_D65.T
    scaled = xyz / _WHITE_D65
    delta = 6.0 / 29.0
    scaled = np.where(
        scaled > delta ** 3,
        np.cbrt(np.maximum(scaled, 0.0)),
        scaled / (3.0 * delta * delta) + 4.0 / 29.0,
    )
    fx, fy, fz = scaled[..., 0], scaled[..., 1], scaled[..., 2]
    return np.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], axis=-1)


def rgb_from_lab(lab) -> tuple[int, int, int]:
    r, g, b = lab_to_rgb(lab)[0]
    return (
        int(max(0, min(255, round(r)))),
        int(max(0, min(255, round(g)))),
        int(max(0, min(255, round(b)))),
    )


def hex_from_lab(lab) -> str:
    return rgb_to_hex(lab_to_rgb(lab)[0])


# --------------------------------------------------------------------------------------
# Mixing
# --------------------------------------------------------------------------------------


def _normalize_weights(percents) -> np.ndarray:
    weights = np.asarray(percents, dtype=np.float64)
    if np.any(weights < 0):
        raise ValueError("percentages must not be negative")
    total = float(np.sum(weights))
    if total <= 1e-12:
        raise ValueError("at least one percentage must be positive")
    return weights / total


def _as_recipe_stack(colours) -> np.ndarray:
    """Normalise ``colours`` to a numeric ``(M, K, 3)`` sRGB array.

    Accepted input shapes:

    * ``(M, K)``   — hex strings
    * ``(M, K, 3)`` — sRGB triples
    """
    array = np.asarray(colours, dtype=object)
    if array.ndim == 2:
        rows = array.shape[0]
        out = np.empty((rows, array.shape[1], 3), dtype=np.float64)
        for r in range(rows):
            for c in range(array.shape[1]):
                out[r, c] = hex_to_rgb(str(array[r, c]))
        return out
    if array.ndim == 3:
        if array.shape[2] != 3:
            raise ValueError(f"colour triples must have 3 channels, got shape {array.shape}")
        return array.astype(np.float64)
    raise ValueError(
        f"colours must be a (M, K) grid of hex strings or a (M, K, 3) grid of sRGB triples, "
        f"got shape {array.shape}"
    )


def mix_lab_batch(colours, percents) -> np.ndarray:
    """Mix many recipes at once.

    ``colours``  : ``(M, K)`` hex strings or ``(M, K, 3)`` sRGB triples.
    ``percents`` : ``(K,)`` shared across recipes, or ``(M, K)`` per recipe.

    Returns ``(M, 3)`` Lab. Rows whose percentages leave fewer than two
    positively-weighted components are returned as ``nan``, mirroring upstream's
    ``std::nullopt`` for single-component input.
    """
    rgb = _as_recipe_stack(colours)
    m, k = rgb.shape[0], rgb.shape[1]

    percent_array = np.asarray(percents, dtype=np.float64)
    if percent_array.ndim == 1:
        if percent_array.shape[0] != k:
            raise ValueError(f"percents has {percent_array.shape[0]} entries, expected {k}")
        percent_array = np.broadcast_to(percent_array, (m, k))
    elif percent_array.ndim == 2:
        if percent_array.shape != (m, k):
            raise ValueError(f"percents shape {percent_array.shape} does not match colours {(m, k)}")
    else:
        raise ValueError(f"percents must be 1-D or 2-D, got {percent_array.ndim}-D")
    if np.any(percent_array < 0):
        raise ValueError("percentages must not be negative")

    spectra = reflectances_from_rgbs(rgb.reshape(-1, 3)).reshape(m, k, SPECTRUM_SIZE)
    ks = ks_from_reflectance(spectra)

    out = np.full((m, 3), np.nan, dtype=np.float64)
    for row in range(m):
        mask = percent_array[row] > 0
        if int(np.count_nonzero(mask)) < 2:
            continue
        weights = _normalize_weights(percent_array[row][mask])
        out[row] = lab_from_reflectance(reflectance_from_ks(weights @ ks[row][mask]))[0]
    return out


def mix_rgb_batch(colours, percents) -> np.ndarray:
    """Like :func:`mix_lab_batch` but returns ``(M, 3)`` integer sRGB."""
    lab = mix_lab_batch(colours, percents)
    return np.rint(np.nan_to_num(lab_to_rgb(lab), nan=0.0)).astype(np.int32)


def mix_hex_batch(colours, percents) -> list[str | None]:
    """Batch mix returning ``#RRGGBB`` strings, ``None`` where upstream would fail."""
    lab = mix_lab_batch(colours, percents)
    out: list[str | None] = []
    for row in lab:
        out.append(None if np.any(np.isnan(row)) else hex_from_lab(row))
    return out


def _wrap_single(colours) -> np.ndarray:
    """Turn a single recipe's components into a ``(1, K[, 3])`` grid."""
    array = np.asarray(colours, dtype=object)
    if array.ndim == 1:
        return array[None, :]
    if array.ndim == 2:
        return array[None, :, :]
    raise ValueError(f"a single recipe must be 1-D or 2-D, got shape {array.shape}")


def mix_lab(colours, percents) -> np.ndarray:
    """Mix a single recipe. ``colours`` is a sequence of hex strings or sRGB triples."""
    return mix_lab_batch(_wrap_single(colours), [[float(p) for p in percents]])[0]


def mix_rgb(colours, percents) -> tuple[int, int, int]:
    """Single-recipe mix as an integer sRGB triple."""
    lab = mix_lab(colours, percents)
    if np.any(np.isnan(lab)):
        raise ValueError("a mixture needs at least two components with positive percentage")
    return rgb_from_lab(lab)


def mix_hex(colours, percents) -> str:
    """Mix a single recipe and return ``#RRGGBB``; the scalar form of the engine."""
    return rgb_to_hex(mix_rgb(colours, percents))


# --------------------------------------------------------------------------------------
# Perceptual difference
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Lab:
    """CIE L*a*b* triple."""

    l: float
    a: float
    b: float

    @classmethod
    def from_array(cls, values) -> "Lab":
        l, a, b = (float(v) for v in values)
        return cls(l, a, b)

    def to_array(self) -> np.ndarray:
        return np.array([self.l, self.a, self.b], dtype=np.float64)


def delta_e_76(lab_a, lab_b) -> float:
    """CIE76 colour difference (straight Euclidean distance in Lab)."""
    a = np.asarray(lab_a, dtype=np.float64)
    b = np.asarray(lab_b, dtype=np.float64)
    return float(np.sqrt(np.sum((a - b) ** 2)))


def delta_e_2000(lab_a, lab_b) -> float:
    """CIEDE2000 colour difference (Sharma et al. formulation).

    ``lab_a`` / ``lab_b`` may be single ``(3,)`` triples.
    """
    a = np.atleast_2d(np.asarray(lab_a, dtype=np.float64))
    b = np.atleast_2d(np.asarray(lab_b, dtype=np.float64))
    if a.shape != b.shape:
        a, b = np.broadcast_arrays(a, b)

    l1, a1, b1 = a[:, 0], a[:, 1], a[:, 2]
    l2, a2, b2 = b[:, 0], b[:, 1], b[:, 2]

    c1 = np.sqrt(a1 * a1 + b1 * b1)
    c2 = np.sqrt(a2 * a2 + b2 * b2)
    c_bar = 0.5 * (c1 + c2)
    c_bar7 = c_bar ** 7
    g = 0.5 * (1.0 - np.sqrt(c_bar7 / (c_bar7 + 25.0 ** 7)))

    a1p = (1.0 + g) * a1
    a2p = (1.0 + g) * a2
    c1p = np.sqrt(a1p * a1p + b1 * b1)
    c2p = np.sqrt(a2p * a2p + b2 * b2)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

    d_lp = l2 - l1
    d_cp = c2p - c1p

    c1p_c2p = c1p * c2p
    d_hp_raw = h2p - h1p
    d_hp = np.where(
        c1p_c2p == 0.0,
        0.0,
        np.where(
            np.abs(d_hp_raw) <= 180.0,
            d_hp_raw,
            np.where(d_hp_raw > 180.0, d_hp_raw - 360.0, d_hp_raw + 360.0),
        ),
    )
    d_big_hp = 2.0 * np.sqrt(c1p_c2p) * np.sin(np.radians(d_hp / 2.0))

    l_bar_p = 0.5 * (l1 + l2)
    c_bar_p = 0.5 * (c1p + c2p)

    h_sum = h1p + h2p
    h_diff = np.abs(h1p - h2p)
    h_bar_p = np.where(
        c1p_c2p == 0.0,
        h_sum,
        np.where(
            h_diff <= 180.0,
            0.5 * h_sum,
            np.where(h_sum < 360.0, 0.5 * (h_sum + 360.0), 0.5 * (h_sum - 360.0)),
        ),
    )

    t = (
        1.0
        - 0.17 * np.cos(np.radians(h_bar_p - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * h_bar_p))
        + 0.32 * np.cos(np.radians(3.0 * h_bar_p + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * h_bar_p - 63.0))
    )

    d_theta = 30.0 * np.exp(-(((h_bar_p - 275.0) / 25.0) ** 2))
    r_c = 2.0 * np.sqrt(c_bar_p ** 7 / (c_bar_p ** 7 + 25.0 ** 7))
    s_l = 1.0 + (0.015 * (l_bar_p - 50.0) ** 2) / np.sqrt(20.0 + (l_bar_p - 50.0) ** 2)
    s_c = 1.0 + 0.045 * c_bar_p
    s_h = 1.0 + 0.015 * c_bar_p * t
    r_t = -np.sin(np.radians(2.0 * d_theta)) * r_c

    term_l = d_lp / s_l
    term_c = d_cp / s_c
    term_h = d_big_hp / s_h

    result = np.sqrt(term_l ** 2 + term_c ** 2 + term_h ** 2 + r_t * term_c * term_h)
    return float(result[0]) if result.size == 1 else result

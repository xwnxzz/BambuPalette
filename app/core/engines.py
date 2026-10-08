"""Mixing engines — what colour do two filaments make at a given ratio?

Three independent models are implemented, and the UI lets the user pick.

``ENGINE_MIXER`` — 「Bambu 混色预览」 (the default)
    A transcription of the arithmetic Bambu Studio uses to paint the
    *添加混色耗材* dialog as of **v02.08.02.61**.  Verified against the shipped
    source of that tag, ``src/slic3r/GUI/MixedFilamentDialog.cpp``::

        static wxColour blend_colors(const wxColour& a, const wxColour& b, double ratio_a)
        {
            unsigned char r, g, bl;
            Slic3r::filament_mixer_lerp(a.Red(), a.Green(), a.Blue(),
                                        b.Red(), b.Green(), b.Blue(),
                                        static_cast<float>(1.0 - ratio_a), &r, &g, &bl);
            return wxColour(r, g, bl);
        }

    ``filament_mixer_lerp`` wraps the header-only ``filament_mixer`` library
    (MIT, Copyright (c) 2026 Justin Hayes), a degree-4 polynomial regression
    trained to approximate Mixbox pigment behaviour.  See
    :mod:`app.spectral.filament_mixer` for the transcription and
    :mod:`app.spectral.filament_mixer_profile` for the baked coefficients.

    This is the default because the whole point of the tool is to tell the user
    *which two spools and what ratio*, so they can reproduce it in Bambu Studio:
    whatever their installed Bambu Studio paints is the reference.  It also fits
    the reference screenshots roughly 36 % better than the old sRGB average
    (total rms 45.1 vs 71.0 across both screenshots and four sampling spans).

``ENGINE_BAMBU`` — 「Bambu 2.5 旧版预览（sRGB 平均）」
    The model Bambu Studio used **before** 02.08, at v02.05.03.62::

        static wxColour blend_colors(const wxColour& a, const wxColour& b, double ratio_a)
        {
            double rb = 1.0 - ratio_a;
            return wxColour(
                (unsigned char)(a.Red()   * ratio_a + b.Red()   * rb),
                (unsigned char)(a.Green() * ratio_a + b.Green() * rb),
                (unsigned char)(a.Blue()  * ratio_a + b.Blue()  * rb));
        }

    A straight weighted mean of the 8-bit sRGB values, and, for three
    components, ``blend_n_colors`` (same mean, then ``std::clamp`` before the
    cast).  Two consequences worth knowing:

    * The cast to ``unsigned char`` TRUNCATES, it does not round.
    * The mean is taken on **gamma-encoded** bytes, so it is a straight line in
      sRGB space and *not* in linear light: ``90 % white + 10 % black``
      previews as ``#E5E5E5`` (L* ≈ 91).

    Kept selectable so a user still on a 2.5.x build can match their own dialog.

``ENGINE_SPECTRAL`` — 「光谱 KM 模型」
    Kubelka-Munk mixing of estimated reflectance spectra: the sRGB value is
    fitted to a 31-band reflectance curve (400–700 nm, 10 nm) with the ICC
    "Munsell Glossy D50" polynomial estimator, converted to K/S, blended
    linearly by weight, and converted back to a colour.

    This is the physically motivated model and predicts dark, pigment-like
    results (``90 % white + 10 % black`` ≈ L* 19).  It is what a *printed*
    blend tends to look like, but it does NOT match Bambu Studio's dialog.

All engines share one interface::

    engine.mix_rgb(colours, weights) -> np.ndarray  # (P, R, 3) int

``colours`` is ``(P, K, 3)`` — ``P`` recipes, ``K`` parent filaments, RGB bytes —
and ``weights`` is ``(P, R, K)`` for ``R`` ratio steps.  Weights must sum to 1
along the last axis.

Neutral preservation
--------------------
Every engine's result passes through :func:`_preserve_neutral`, which forces a
**grey result for a grey input pair**.  This is a deliberate correction, not a
property of the models: the ``filament_mixer`` polynomial is a least-squares
regression, and it tints the achromatic axis blue — ``#FFFFFF`` + ``#000000`` at
50 % comes out ``#637E9F``.  Physically two neutral spools must mix to a neutral
grey, so the raw polynomial is wrong exactly there.  The guard keeps the mix's
own lightness (it averages the parents' channel means) and only discards the
spurious hue, so saturated pigment mixing (blue + yellow → green) is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..spectral import color as _color
from ..spectral import filament_mixer as _mixer

ENGINE_MIXER = "mixer"
ENGINE_BAMBU = "bambu"
ENGINE_SPECTRAL = "spectral"
DEFAULT_ENGINE = ENGINE_MIXER

#: Both parents must be this close to grey (channel spread, 0–255) before the
#: neutral guard engages.  12 covers real "white" and "black" spools, which are
#: never exactly neutral — the sample library's ``#F2F0EB`` (spread 7) and
#: ``#17181C`` (spread 5) both qualify — while staying far away from any
#: deliberately tinted filament.
NEUTRAL_SPREAD = 12

#: Engine ids that older builds wrote into ``settings.json``.  A saved id that
#: is no longer offered must still load, so ``get_engine`` maps these across.
_LEGACY_ENGINE_ALIASES: dict[str, str] = {
    # Before 02.08 the product called the sRGB average just "bambu"; anyone who
    # explicitly saved that choice keeps it.
    "bambu-srgb": ENGINE_BAMBU,
    "srgb": ENGINE_BAMBU,
    "km": ENGINE_SPECTRAL,
    "kubelka-munk": ENGINE_SPECTRAL,
}

__all__ = [
    "ENGINE_MIXER",
    "ENGINE_BAMBU",
    "ENGINE_SPECTRAL",
    "DEFAULT_ENGINE",
    "MixEngine",
    "FilamentMixerEngine",
    "BambuSRGBEngine",
    "SpectralKMEngine",
    "ENGINES",
    "ENGINE_CHOICES",
    "NEUTRAL_SPREAD",
    "get_engine",
    "resolve_engine_id",
]


def _validate(colours: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shared shape guard for every engine's ``mix_rgb``."""
    colours = np.asarray(colours, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if colours.ndim != 3 or weights.ndim != 3:
        raise ValueError(
            f"colours {colours.shape} and weights {weights.shape} must both be 3-D"
        )
    if colours.shape[0] != weights.shape[0] or colours.shape[1] != weights.shape[2]:
        raise ValueError(
            f"colours {colours.shape} and weights {weights.shape} disagree on (P, K)"
        )
    return colours, weights


def _preserve_neutral(
    colours: np.ndarray, weights: np.ndarray, mixed: np.ndarray
) -> np.ndarray:
    """Force a grey result when both parents are (near) neutral.

    Two neutral spools cannot make a tinted blend, but a least-squares pigment
    model can still invent one.  When every parent of a recipe is within
    :data:`NEUTRAL_SPREAD` of grey, the result is replaced by the weight-averaged
    **channel mean** of those parents — a colourless grey at the mix's own
    lightness.  Chromatic pairs are returned untouched, so pigment behaviour is
    preserved.

    The mean (not a luma weighting) is deliberate: for a neutral colour the three
    channels are equal, so the mean is exactly that grey level, whereas Rec. 709
    coefficients do not sum to 1.0 in binary floating point and would shave
    ``#FFFFFF`` down to ``#FEFEFE``.

    A single parent (``K < 2``) is left alone: there is nothing to mix, and the
    engines reproduce that input verbatim.
    """
    if colours.shape[1] < 2:
        return mixed
    spread = colours.max(axis=-1) - colours.min(axis=-1)         # (P, K)
    neutral = spread.max(axis=-1) <= NEUTRAL_SPREAD              # (P,)
    if not neutral.any():
        return mixed
    level = np.floor(np.einsum("prk,pk->pr", weights, colours.mean(axis=-1)))
    level = np.clip(level, 0, 255).astype(np.int64)              # (P, R)
    grey = np.repeat(level[:, :, None], 3, axis=2)
    return np.where(neutral[:, None, None], grey, mixed)


class MixEngine:
    """Base class: maps parent colours + weights to a blended RGB byte triple."""

    id: str = ""
    name: str = ""
    detail: str = ""

    def mix_rgb(self, colours: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Blend ``colours`` (P, K, 3) with ``weights`` (P, R, K) → (P, R, 3) int."""
        colours, weights = _validate(colours, weights)
        return _preserve_neutral(colours, weights, self._mix_rgb(colours, weights))

    def _mix_rgb(self, colours: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """The engine's own arithmetic; ``colours``/``weights`` are validated."""
        raise NotImplementedError

    def mix_rgb_pair(self, rgb_a, rgb_b, percent_a: float) -> tuple[int, int, int]:
        """Convenience wrapper for the common two-filament case."""
        colours = np.asarray([[rgb_a, rgb_b]], dtype=np.float64)
        ratio_a = float(percent_a) / 100.0
        weights = np.asarray([[[ratio_a, 1.0 - ratio_a]]], dtype=np.float64)
        out = self.mix_rgb(colours, weights)
        r, g, b = (int(v) for v in out[0, 0])
        return (r, g, b)

    def mix_hex(self, hex_a: str, hex_b: str, percent_a: float) -> str:
        r, g, b = self.mix_rgb_pair(_color.hex_to_rgb(hex_a), _color.hex_to_rgb(hex_b), percent_a)
        return f"#{r:02X}{g:02X}{b:02X}"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} {self.id}>"


@dataclass(frozen=True)
class FilamentMixerEngine(MixEngine):
    """``filament_mixer``'s degree-4 pigment polynomial — Bambu Studio 02.08+."""

    id: str = ENGINE_MIXER
    name: str = "Bambu 混色预览"
    detail: str = (
        "与 Bambu Studio 2.8（02.08.x）「添加混色耗材」对话框一致的颜料混色模型："
        "Bambu 内置的 filament_mixer 四次多项式（Mixbox 近似，MIT 许可）。"
        "这是默认模型 —— 你在 Bambu Studio 里看到的预览色，这里应当一样。"
    )

    def _mix_rgb(self, colours: np.ndarray, weights: np.ndarray) -> np.ndarray:
        if colours.shape[1] < 2:
            # A single parent is a no-op; the polynomial is undefined for K < 2.
            only = np.clip(np.rint(colours[:, 0, :]), 0, 255).astype(np.int64)
            return np.repeat(only[:, None, :], weights.shape[1], axis=1)
        return _mixer.mix_rgb(colours, weights)


@dataclass(frozen=True)
class BambuSRGBEngine(MixEngine):
    """Weighted mean of 8-bit sRGB values — Bambu Studio's pre-02.08 dialog."""

    id: str = ENGINE_BAMBU
    name: str = "Bambu 2.5 旧版预览（sRGB 平均）"
    detail: str = (
        "Bambu Studio 2.5 及更早版本「添加混色耗材」对话框的算法："
        "对 8 位 sRGB 数值做加权平均后截断取整。"
        "如果你用的还是 2.5.x，选这一项才能对上你看到的预览色；"
        "2.8 已经改用颜料混色模型（即默认的那一项）。"
    )

    def _mix_rgb(self, colours: np.ndarray, weights: np.ndarray) -> np.ndarray:
        mixed = np.einsum("prk,pkc->prc", weights, colours)
        # std::clamp then the C cast to unsigned char: both truncate toward zero.
        return np.clip(mixed, 0.0, 255.0).astype(np.int64)


@dataclass(frozen=True)
class SpectralKMEngine(MixEngine):
    """Kubelka-Munk blend of estimated reflectance spectra."""

    id: str = ENGINE_SPECTRAL
    name: str = "光谱 KM 模型"
    detail: str = (
        "物理模型：先把每个耗材的 sRGB 拟合为 400–700nm 的反射光谱，"
        "换算成 Kubelka-Munk 的 K/S 值后按重量线性混合，再转回颜色。"
        "结果更接近颜料混合的真实观感（深色、低明度），"
        "与 Bambu Studio 对话框的预览色并不一致。"
    )

    def _mix_rgb(self, colours: np.ndarray, weights: np.ndarray) -> np.ndarray:
        flat_colours = colours.reshape(-1, 3)
        spectra = _color.reflectances_from_rgbs(flat_colours).reshape(
            *colours.shape[:-1], _color.SPECTRUM_SIZE
        )                                                          # (P, K, 31)
        ks = _color.ks_from_reflectance(spectra)                   # (P, K, 31)
        mixed = np.einsum("prk,pkw->prw", weights, ks)             # (P, R, 31)
        flat = mixed.reshape(-1, _color.SPECTRUM_SIZE)
        lab = _color.lab_from_reflectance(_color.reflectance_from_ks(flat))
        rgb = np.clip(np.rint(_color.lab_to_rgb(lab)), 0, 255).astype(np.int64)
        return rgb.reshape(mixed.shape[0], mixed.shape[1], 3)


ENGINES: dict[str, MixEngine] = {
    engine.id: engine
    for engine in (FilamentMixerEngine(), BambuSRGBEngine(), SpectralKMEngine())
}

#: ``(id, label)`` pairs for combo boxes, default first.
ENGINE_CHOICES: tuple[tuple[str, str], ...] = tuple(
    (engine.id, engine.name) for engine in ENGINES.values()
)


def get_engine(engine: str | MixEngine | None = None) -> MixEngine:
    """Look up an engine by id; ``None`` yields the default.

    An unknown id raises, but ids that older builds shipped are translated
    first so a saved settings file never becomes unloadable.
    """
    if isinstance(engine, MixEngine):
        return engine
    if engine is None:
        return ENGINES[DEFAULT_ENGINE]
    key = resolve_engine_id(engine)
    if key is None:
        raise ValueError(
            f"unknown mixing engine {engine!r}; known: {', '.join(ENGINES)}"
        )
    return ENGINES[key]


def resolve_engine_id(engine_id: object) -> str | None:
    """Translate a possibly-legacy engine id; ``None`` when it is unknown."""
    if not isinstance(engine_id, str):
        return None
    key = _LEGACY_ENGINE_ALIASES.get(engine_id, engine_id)
    return key if key in ENGINES else None

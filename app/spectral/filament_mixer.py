"""The ``filament_mixer`` pigment model, in NumPy.

Bambu Studio's 添加混色耗材 dialog previews a mixed filament with the
header-only ``filament_mixer`` library (MIT, Copyright (c) 2026 Justin Hayes),
which regresses Mixbox behaviour with a degree-4 polynomial over 330 features
and the seven inputs ``(r1, g1, b1, r2, g2, b2, t)``.  Bambu bundles it as
``src/libslic3r/FilamentMixerModel.hpp`` and exposes it as
``Slic3r::filament_mixer_lerp``; the dialog's ``blend_colors`` calls it with
``t = 1 - ratio_a``, so ``t`` is the *second* colour's share.

This module is a direct transcription of that arithmetic.  The coefficients
live in :mod:`app.spectral.filament_mixer_profile`, which
``tools/gen_filament_mixer_profile.py`` bakes from the upstream header.

Two details are deliberately copied because they change the output:

* ``t <= 0`` and ``t >= 1`` short-circuit to the exact input colours rather
  than evaluating the polynomial.
* Each channel is **truncated** toward zero after clamping -- the source does
  ``static_cast<int>(sum)``, not a round.

For three or more colours the reference composites sequentially, quantising to
8 bits between steps; :func:`mix_rgb` reproduces that order.
"""

from __future__ import annotations

import numpy as np

from .filament_mixer_profile import COEF, INTERCEPT, N_FEATURES, N_INPUTS, POWERS

__all__ = ["mix_pair", "mix_pair_batch", "mix_rgb", "mix_rgb_flat"]

#: Exponent matrix split into the columns that are actually used, so the feature
#: build can skip the many zero exponents the degree-4 expansion produces.
_NONZERO = [(j, POWERS[:, j] > 0, POWERS[POWERS[:, j] > 0, j]) for j in range(N_INPUTS)]

#: The highest exponent any single input carries in the degree-4 expansion.
_MAX_EXPONENT = int(POWERS.max())


def _build_layout() -> tuple[list[tuple[int, int, int, int, np.ndarray]], list[int], int]:
    """Order the degree-4 monomials so each one costs a single multiply.

    The 330 features are *every* monomial of total degree ≤ 4 over the seven
    inputs.  A monomial therefore has a natural parent: the same monomial with
    its highest non-zero exponent **removed**, whose product chain is one factor
    shorter.  Multiply the parent by that input's full power and you get the
    child — and because the parent's chain is exactly the child's minus its last
    factor, the floating-point rounding order is the one the profile's own
    column order would have used.

    The layout walks the inputs and exponents in ascending order and, inside a
    group, appends children in *parent-position* order.  That makes each group a
    contiguous slice of the output while its parent columns still read forward.
    """
    # POWERS is (N_FEATURES, N_INPUTS): one exponent per input, per feature.
    lookup = {tuple(int(v) for v in POWERS[feature]): feature for feature in range(N_FEATURES)}
    zero = (0,) * N_INPUTS
    layout: list[tuple[int, ...]] = [zero]
    position = {zero: 0}
    writes: list[tuple[int, int, int, int, np.ndarray]] = []

    for index in range(N_INPUTS):
        for exponent in range(1, _MAX_EXPONENT + 1):
            members: list[tuple[int, tuple[int, ...]]] = []
            for monomial in lookup:
                if monomial[index] != exponent:
                    continue
                if any(monomial[j] for j in range(index + 1, N_INPUTS)):
                    continue
                parent = list(monomial)
                parent[index] = 0
                members.append((position[tuple(parent)], monomial))
            if not members:
                continue
            members.sort()
            start = len(layout)
            for _, monomial in members:
                position[monomial] = len(layout)
                layout.append(monomial)
            parents = np.array([parent for parent, _ in members], dtype=np.int64)
            writes.append((index, exponent, start, len(members), parents))

    assert len(layout) == N_FEATURES, "the monomial walk must cover every feature exactly once"
    order = [lookup[monomial] for monomial in layout]
    return writes, order, len(layout)


#: ``(input, exponent, start, count, parent columns)`` per multiply.
_WRITES, _ORDER, _COLUMNS = _build_layout()

#: ``COEF`` reordered to match the layout :func:`_build_layout` produced, which
#: is why the feature matrix never has to be put back into the profile's order.
COEF_ORDERED = COEF[_ORDER]

_CHUNK = 2_048


def _features(x: np.ndarray) -> np.ndarray:
    """Evaluate the 330 polynomial features for each row of ``x`` (N, 7).

    Returns the columns in :data:`_ORDER` order, not the profile's own order —
    :data:`COEF_ORDERED` is permuted to match, so the polynomial is unchanged.

    ``features[:, mask] *= x[:, column:column+1] ** exponents`` is the obvious
    spelling, but with a *vector* right-hand side NumPy calls ``pow()`` once per
    element — roughly a thousand powers per row across the 330 monomials, and
    about 2.7 s of a 41-spool catalogue build.  Multiplying each monomial out of
    its parent turns that into 330 multiplies over one column each.
    """
    features = np.empty((x.shape[0], _COLUMNS), dtype=np.float64)
    features[:, 0] = 1.0
    for index, exponent, start, count, parents in _WRITES:
        features[:, start : start + count] = (
            features[:, parents] * np.power(x[:, index : index + 1], exponent)
        )
    return features


def _evaluate(x: np.ndarray) -> np.ndarray:
    """Polynomial body shared by every entry point; ``x`` is (N, 7) -> (N, 3)."""
    out = np.empty((x.shape[0], 3), dtype=np.int64)
    for start in range(0, x.shape[0], _CHUNK):
        block = x[start : start + _CHUNK]
        raw = _features(block) @ COEF_ORDERED + INTERCEPT
        # `astype` truncates toward zero, matching the reference cast; clamping
        # first makes a negative value behave the same way it does in C++.
        out[start : start + block.shape[0]] = np.clip(raw, 0.0, 255.0).astype(np.int64)
    return out


def mix_pair_batch(first: np.ndarray, second: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Mix many colour pairs at once.

    ``first`` and ``second`` are ``(N, 3)`` colours in 0-255; ``t`` is ``(N,)``
    in ``[0, 1]`` and is the share of ``second``.  Returns ``(N, 3)`` integers.
    """
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)

    x = np.concatenate([first, second, t[:, None]], axis=1)
    out = _evaluate(x)

    # The reference returns the untouched input at the ends of the range.
    low = t <= 0.0
    high = t >= 1.0
    if low.any():
        out[low] = first[low].astype(np.int64)
    if high.any():
        out[high] = second[high].astype(np.int64)
    return out


def mix_pair(
    rgb_a: tuple[int, int, int], rgb_b: tuple[int, int, int], percent_a: float
) -> tuple[int, int, int]:
    """Mix two colours given the first one's percentage (0-100)."""
    t = 1.0 - max(0.0, min(100.0, float(percent_a))) / 100.0
    out = mix_pair_batch(
        np.array([rgb_a], dtype=np.float64),
        np.array([rgb_b], dtype=np.float64),
        np.array([t], dtype=np.float64),
    )[0]
    return int(out[0]), int(out[1]), int(out[2])


def mix_rgb(colours: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Vectorised engine entry point.

    ``colours`` is ``(P, K, 3)`` and ``weights`` is ``(P, R, K)``; the result is
    ``(P, R, 3)`` integers.  ``K == 2`` takes the polynomial directly; larger
    ``K`` composites left to right exactly as the reference does, quantising
    the accumulator to 8 bits between steps.
    """
    colours = np.asarray(colours, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    pairs, components, _channels = colours.shape
    if weights.shape[0] != pairs or weights.shape[2] != components:
        raise ValueError(
            f"colours {colours.shape} and weights {weights.shape} disagree on (P, K)"
        )
    ratios = weights.shape[1]

    first = np.repeat(colours[:, 0, :][:, None, :], ratios, axis=1).reshape(-1, 3)
    second = np.repeat(colours[:, 1, :][:, None, :], ratios, axis=1).reshape(-1, 3)
    share = weights[:, :, 1]

    accumulated = weights[:, :, 0] + share
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(accumulated > 0.0, share / accumulated, 0.5)

    result = mix_pair_batch(first, second, t.reshape(-1)).reshape(pairs, ratios, 3).astype(np.float64)
    if components == 2:
        return result.astype(np.int64)

    running = accumulated
    for index in range(2, components):
        weight = weights[:, :, index]
        total = running + weight
        with np.errstate(invalid="ignore", divide="ignore"):
            step = np.where(total > 0.0, weight / total, 0.0)
        flat = mix_pair_batch(
            result.reshape(-1, 3), np.repeat(colours[:, index, :][:, None, :], ratios, axis=1).reshape(-1, 3), step.reshape(-1)
        )
        result = flat.reshape(pairs, ratios, 3).astype(np.float64)
        running = total
    return result.astype(np.int64)


def mix_rgb_flat(
    rgb_a: tuple[int, int, int], rgb_b: tuple[int, int, int], percentages: np.ndarray
) -> np.ndarray:
    """Sweep one pair across many ``percent_a`` values; returns ``(R, 3)``."""
    percentages = np.asarray(percentages, dtype=np.float64)
    t = 1.0 - percentages / 100.0
    first = np.repeat(np.array([rgb_a], dtype=np.float64), t.size, axis=0)
    second = np.repeat(np.array([rgb_b], dtype=np.float64), t.size, axis=0)
    return mix_pair_batch(first, second, t)

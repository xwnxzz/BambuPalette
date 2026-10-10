"""Engine tests: colour transforms, the two mixing engines, catalogue arithmetic.

Every expected value in this file was measured against the implementation *and*
cross-checked against its source of truth:

* the Bambu numbers against ``MixedFilamentDialog.cpp`` of Bambu Studio
  ``v02.05.03.62`` (``blend_colors`` = 8-bit sRGB weighted mean, truncated);
* the CIE values against the published CIEDE2000 (Sharma et al.) test set and
  the CIE 1931 10° / D65 white point.

Run with::

    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
    # or, with pytest installed:
    .venv\\Scripts\\python.exe -m pytest tests -q
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.engines import (  # noqa: E402
    DEFAULT_ENGINE,
    ENGINE_BAMBU,
    ENGINE_CHOICES,
    ENGINE_MIXER,
    ENGINE_SPECTRAL,
    ENGINES,
    NEUTRAL_SPREAD,
    MixEngine,
    get_engine,
    resolve_engine_id,
)
from app.core.library import Filament, FilamentLibrary  # noqa: E402
from app.core.mixes import (  # noqa: E402
    MIX_RATIOS,
    SORT_CHOICES,
    MixCatalog,
    expected_recipe_count,
    sorted_cells,
    sorted_filaments,
)
from app.spectral import color as C  # noqa: E402
from app.spectral import filament_mixer as FM  # noqa: E402

PALETTE = (
    "#FFFFFF", "#000000", "#FF0000", "#00FF00", "#0000FF",
    "#808080", "#1A2B3C", "#FEDCBA", "#7F3F00", "#00A0A0",
)


def _batch(*hex_values):
    """Stack hex colours into the ``(P, K, 3)`` array the engines expect."""
    return np.array([[C.hex_to_rgb(value) for value in hex_values]], dtype=np.float64)


def _weights(*percents):
    """Stack percentages into the ``(P, R, K)`` array the engines expect."""
    return np.array([[[percent / 100.0 for percent in percents]]], dtype=np.float64)


class TransferFunctionTests(unittest.TestCase):
    def test_srgb_roundtrip(self):
        for value in (0.0, 0.001, 0.04045, 0.2, 0.5, 0.9, 1.0):
            with self.subTest(value=value):
                self.assertAlmostEqual(C.linear_to_srgb(C.srgb_to_linear(value)), value, places=6)

    def test_srgb8_matches_scalar(self):
        for channel in (0, 1, 17, 128, 254, 255):
            self.assertAlmostEqual(C.srgb8_to_linear(channel), C.srgb_to_linear(channel / 255.0), places=15)

    def test_hex_helpers_roundtrip(self):
        self.assertEqual(C.hex_to_rgb("#ff8800"), (255, 136, 0))
        self.assertEqual(C.normalize_hex("ff8800"), "#FF8800")
        self.assertEqual(C.rgb_to_hex((255, 136, 0)), "#FF8800")
        self.assertEqual(C.rgb_to_hex((-10, 300, 12.6)), "#00FF0D")

    def test_hex_rejects_malformed(self):
        for bad in ("", "#12345", "#1234567", "zzzzzz", "#GGGGGG"):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    C.hex_to_rgb(bad)


class ReflectanceEstimateTests(unittest.TestCase):
    """What the ICC polynomial estimator actually returns.

    The estimate is a *raw polynomial* result, so it is not confined to the
    physical [0, 1] range.  ``ks_from_reflectance`` clamps it to
    [0.001, 0.999]; that clamp is what makes the spectral engine lossy for
    saturated colours, and it is recorded here so the behaviour cannot change
    silently.
    """

    def test_shape_and_term_count(self):
        spectrum = C.reflectance_from_rgb((255, 255, 255))
        self.assertEqual(spectrum.shape, (C.SPECTRUM_SIZE,))
        self.assertEqual(C.SPECTRUM_SIZE, 31)
        self.assertEqual(C.TERM_COUNT, 20)

    def test_white_reflects_more_than_black_everywhere(self):
        white = C.reflectance_from_rgb((255, 255, 255))
        black = C.reflectance_from_rgb((0, 0, 0))
        for index in range(C.SPECTRUM_SIZE):
            self.assertGreater(white[index], black[index])
        self.assertGreater(float(white.mean()), 0.40)
        self.assertLess(float(black.mean()), 0.10)

    def test_estimate_leaves_the_physical_range(self):
        # Documented, expected: the polynomial overshoots above 1 for white and
        # dips below 0 for black.
        white = C.reflectance_from_rgb((255, 255, 255))
        black = C.reflectance_from_rgb((0, 0, 0))
        self.assertGreater(float(white.max()), 1.0)
        self.assertLess(float(black.min()), 0.0)

    def test_ks_roundtrip_of_a_mid_reflectance(self):
        reflectance = np.array([0.4, 0.5, 0.6])
        np.testing.assert_allclose(C.reflectance_from_ks(C.ks_from_reflectance(reflectance)),
                                   reflectance, atol=1e-12)


class LabTests(unittest.TestCase):
    def test_white_and_black_anchors(self):
        # This module's white point is the CIE 1931 10° D65 white used by the
        # CMF/illuminant tables, not the 2° one the sRGB matrix rows sum to, so
        # a neutral carries a sub-unit a/b offset.  Lightness is exact.
        white = C.lab_from_rgb((255, 255, 255))
        self.assertAlmostEqual(float(white[0]), 100.0, places=4)
        self.assertAlmostEqual(float(white[1]), 0.4145, places=3)
        self.assertAlmostEqual(float(white[2]), -0.9762, places=3)
        np.testing.assert_allclose(C.lab_from_rgb((0, 0, 0)), [0.0, 0.0, 0.0], atol=1e-9)

    def test_greys_stay_neutral(self):
        for value in (32, 96, 160, 224):
            with self.subTest(grey=value):
                lab = C.lab_from_rgb((value, value, value))
                self.assertLess(abs(float(lab[1])), 1.0)
                self.assertLess(abs(float(lab[2])), 1.0)

    def test_mid_grey_lightness(self):
        self.assertAlmostEqual(float(C.lab_from_rgb((128, 128, 128))[0]), 53.585, places=3)

    def test_rgb_lab_roundtrip(self):
        for value in PALETTE:
            with self.subTest(color=value):
                back = C.rgb_from_lab(C.lab_from_rgb(C.hex_to_rgb(value)))
                for expected, actual in zip(C.hex_to_rgb(value), back):
                    self.assertLessEqual(abs(expected - actual), 2)


class ColourDifferenceTests(unittest.TestCase):
    def test_zero_distance(self):
        self.assertAlmostEqual(C.delta_e_2000((50.0, 10.0, -20.0), (50.0, 10.0, -20.0)), 0.0, places=12)
        self.assertAlmostEqual(C.delta_e_76((50.0, 10.0, -20.0), (50.0, 10.0, -20.0)), 0.0, places=12)

    def test_ciede2000_sharma_reference_pairs(self):
        # Reference pairs published with the CIEDE2000 formulation (Sharma et al.).
        reference = (
            ((50.0000, 2.6772, -79.7751), (50.0000, 0.0000, -82.7485), 2.0425),
            ((50.0000, 3.1571, -77.2803), (50.0000, 0.0000, -82.7485), 2.8615),
            ((50.0000, 2.8361, -74.0200), (50.0000, 0.0000, -82.7485), 3.4412),
            ((50.0000, -1.3802, -84.2814), (50.0000, 0.0000, -82.7485), 1.0000),
            ((50.0000, 0.0000, 0.0000), (50.0000, -1.0000, 2.0000), 2.3669),
        )
        for lab_a, lab_b, expected in reference:
            with self.subTest(pair=(lab_a, lab_b)):
                self.assertAlmostEqual(C.delta_e_2000(lab_a, lab_b), expected, places=4)

    def test_symmetry(self):
        a, b = (55.0, 30.0, -40.0), (60.0, -20.0, 15.0)
        self.assertAlmostEqual(C.delta_e_2000(a, b), C.delta_e_2000(b, a), places=10)
        self.assertAlmostEqual(C.delta_e_76(a, b), C.delta_e_76(b, a), places=10)


class EngineRegistryTests(unittest.TestCase):
    def test_registry_and_choices_agree(self):
        self.assertEqual({choice[0] for choice in ENGINE_CHOICES}, set(ENGINES))
        self.assertIn(DEFAULT_ENGINE, ENGINES)
        self.assertEqual(DEFAULT_ENGINE, ENGINE_MIXER)
        # The default must be offered first, so a fresh combo box selects it.
        self.assertEqual(ENGINE_CHOICES[0][0], DEFAULT_ENGINE)

    def test_get_engine_resolves_strings_instances_and_none(self):
        self.assertIs(get_engine(None), ENGINES[DEFAULT_ENGINE])
        self.assertIs(get_engine(DEFAULT_ENGINE), ENGINES[DEFAULT_ENGINE])
        self.assertIs(get_engine(ENGINES[ENGINE_SPECTRAL]), ENGINES[ENGINE_SPECTRAL])
        with self.assertRaises(ValueError):
            get_engine("no-such-engine")

    def test_legacy_engine_ids_still_resolve(self):
        self.assertEqual(resolve_engine_id("srgb"), ENGINE_BAMBU)
        self.assertEqual(resolve_engine_id("kubelka-munk"), ENGINE_SPECTRAL)
        self.assertEqual(resolve_engine_id(ENGINE_MIXER), ENGINE_MIXER)
        self.assertIsNone(resolve_engine_id("no-such-engine"))
        self.assertIsNone(resolve_engine_id(None))
        self.assertIs(get_engine("srgb"), ENGINES[ENGINE_BAMBU])

    def test_shape_guard(self):
        for engine_id, engine in ENGINES.items():
            with self.subTest(engine=engine_id):
                with self.assertRaises(ValueError):
                    engine.mix_rgb(np.zeros((1, 2, 3)), np.zeros((1, 1, 3)))
                with self.assertRaises(ValueError):
                    engine.mix_rgb(np.zeros((1, 2, 3)), np.zeros((2, 1, 2)))
                with self.assertRaises(ValueError):
                    engine.mix_rgb(np.zeros((2, 2, 3)), np.zeros((1, 1, 2)))

    def test_engines_are_exact_at_the_endpoints(self):
        # 100/0 must hand back the input unchanged.  The two Bambu models do
        # that exactly; the spectral engine cannot, because a pure input has to
        # make a round trip through a reflectance that ks_from_reflectance
        # clamps into [0.001, 0.999] — so it is only ever close (measured worst
        # case across the palette is ΔE00 20.5).
        for engine_id in (ENGINE_MIXER, ENGINE_BAMBU):
            for value in ("#3D7BC8", "#FF8800", "#123456", "#EEEEEE"):
                with self.subTest(engine=engine_id, color=value):
                    engine = ENGINES[engine_id]
                    rgbs = _batch(value, value)
                    self.assertEqual(
                        engine.mix_rgb(rgbs, _weights(100, 0))[0, 0].tolist(),
                        list(C.hex_to_rgb(value)),
                    )
                    self.assertEqual(
                        engine.mix_rgb(rgbs, _weights(0, 100))[0, 0].tolist(),
                        list(C.hex_to_rgb(value)),
                    )

        for value in ("#3D7BC8", "#FF8800", "#123456", "#EEEEEE"):
            with self.subTest(engine=ENGINE_SPECTRAL, color=value):
                engine = ENGINES[ENGINE_SPECTRAL]
                rgbs = _batch(value, value)
                target = C.lab_from_rgb(C.hex_to_rgb(value))
                for percents in ((100, 0), (0, 100)):
                    mixed = engine.mix_rgb(rgbs, _weights(*percents))[0, 0]
                    self.assertLessEqual(C.delta_e_2000(target, C.lab_from_rgb(mixed)), 25.0)

    def test_bambu_and_spectral_are_exactly_symmetric(self):
        for engine_id in (ENGINE_BAMBU, ENGINE_SPECTRAL):
            engine = ENGINES[engine_id]
            with self.subTest(engine=engine_id):
                self.assertEqual(engine.mix_hex("#3D7BC8", "#FF8800", 30),
                                 engine.mix_hex("#FF8800", "#3D7BC8", 70))

    def test_the_pigment_model_is_only_nearly_symmetric(self):
        # filament_mixer is a least-squares regression, not an identity, so
        # swapping the operands moves the result by a unit or two.  Do not
        # assert equality — assert the bound, so a real regression is caught.
        engine = ENGINES[ENGINE_MIXER]
        worst = 0
        for percent in range(10, 91):
            forward = engine.mix_rgb_pair(C.hex_to_rgb("#3D7BC8"), C.hex_to_rgb("#FF8800"), percent)
            reverse = engine.mix_rgb_pair(C.hex_to_rgb("#FF8800"), C.hex_to_rgb("#3D7BC8"), 100 - percent)
            worst = max(worst, max(abs(a - b) for a, b in zip(forward, reverse)))
        self.assertLessEqual(worst, 4, f"worst channel disagreement was {worst}")


class FilamentMixerEngineTests(unittest.TestCase):
    """The default model: Bambu Studio 02.08's pigment polynomial.

    ``MixedFilamentDialog::blend_colors`` calls ``Slic3r::filament_mixer_lerp``,
    which wraps the header-only ``filament_mixer`` library (MIT, Copyright (c)
    2026 Justin Hayes).  These tests pin the transcription against values taken
    from the upstream header and from the shipped Bambu sources.
    """

    def setUp(self):
        self.engine = ENGINES[ENGINE_MIXER]

    def test_upstream_reference_sample(self):
        # Pinned by the library's own documentation: blue + yellow -> green,
        # and specifically NOT the naive RGB interpolation (126, 122, 67).
        self.assertEqual(FM.mix_pair((0, 33, 133), (252, 211, 0), 50), (47, 141, 56))
        self.assertEqual(
            self.engine.mix_rgb_pair((0, 33, 133), (252, 211, 0), 50), (47, 141, 56)
        )

    def test_the_endpoints_come_back_untouched(self):
        # The reference short-circuits t <= 0 and t >= 1 before the polynomial,
        # so these are exact even though the polynomial itself would drift.
        for value in ("#3D7BC8", "#FF8800", "#123456", "#EEEEEE", "#000000", "#FFFFFF"):
            rgb = C.hex_to_rgb(value)
            with self.subTest(color=value):
                self.assertEqual(self.engine.mix_rgb_pair(rgb, (18, 24, 28), 100), rgb)
                self.assertEqual(self.engine.mix_rgb_pair((18, 24, 28), rgb, 0), rgb)

    def test_a_self_mix_is_close_to_the_input(self):
        # Mixing a colour with itself must be nearly a no-op.  It is not exact:
        # the polynomial is a least-squares regression, and its error is largest
        # at the gamut corners.  Sweep EVERY ratio, not just 50/50 -- at 50/50
        # the worst is only 8, but over the whole 10..90 sweep pure blue drifts
        # by 21 (measured; #FF00FF 20, #FF0000 and #00FF00 13).  A bound rather
        # than an equality, so a genuinely broken transcription still fails.
        worst = 0
        where = 0
        for value in PALETTE:
            rgb = C.hex_to_rgb(value)
            drift = 0
            for percent in MIX_RATIOS:
                mixed = self.engine.mix_rgb_pair(rgb, rgb, percent)
                drift = max(drift, max(abs(a - b) for a, b in zip(rgb, mixed)))
            with self.subTest(color=value):
                self.assertLessEqual(drift, 24, f"{value} drifted {drift}")
            if drift > worst:
                worst, where = drift, percent
        self.assertLessEqual(worst, 24, f"self-mix drift grew to {worst}")
        self.assertGreaterEqual(worst, 5, "the regression should not be an exact identity either")

    def test_self_mix_drift_does_not_depend_on_which_side_is_first(self):
        # Both parents are the same colour, so the short-circuit at t<=0/t>=1 is
        # the only asymmetry left: 100/0 and 0/100 both return the input exactly,
        # while the interior of the sweep is where the regression's error lives.
        for value in PALETTE:
            rgb = C.hex_to_rgb(value)
            with self.subTest(color=value):
                self.assertEqual(self.engine.mix_rgb_pair(rgb, rgb, 100), rgb)
                self.assertEqual(self.engine.mix_rgb_pair(rgb, rgb, 0), rgb)

    def test_batch_and_scalar_paths_agree_for_every_ratio(self):
        a, b = C.hex_to_rgb("#F2F0EB"), C.hex_to_rgb("#17181C")
        batch = FM.mix_rgb_flat(a, b, np.array(list(MIX_RATIOS), dtype=np.float64))
        for row, percent in enumerate(MIX_RATIOS):
            with self.subTest(percent=percent):
                self.assertEqual(tuple(int(v) for v in batch[row]), FM.mix_pair(a, b, percent))

    def test_the_engine_entry_point_accepts_the_catalogue_shapes(self):
        # A chromatic pair on purpose: a neutral pair is overridden by the grey
        # guard, so it would not equal the raw model (see NeutralGuardTests).
        colours = _batch("#2FA8B8", "#C8342E")
        weights = np.array([[[r / 100.0, 1.0 - r / 100.0] for r in MIX_RATIOS]], dtype=np.float64)
        out = self.engine.mix_rgb(colours, weights)
        self.assertEqual(out.shape, (1, len(MIX_RATIOS), 3))
        self.assertEqual(
            out[0, 0].tolist(),
            list(FM.mix_pair(C.hex_to_rgb("#2FA8B8"), C.hex_to_rgb("#C8342E"), MIX_RATIOS[0])),
        )

    def test_a_single_parent_is_a_no_op(self):
        colours = np.asarray([[[61, 123, 200]]], dtype=np.float64)
        weights = np.asarray([[[1.0], [0.5]]], dtype=np.float64)
        out = self.engine.mix_rgb(colours, weights)
        self.assertEqual(out.shape, (1, 2, 3))
        self.assertEqual(out[0, 0].tolist(), [61, 123, 200])
        self.assertEqual(out[0, 1].tolist(), [61, 123, 200])

    def test_it_disagrees_with_the_legacy_model(self):
        # If the two Bambu models ever agree everywhere the choice is cosmetic.
        legacy = ENGINES[ENGINE_BAMBU]
        differing = sum(
            1
            for value in PALETTE
            if self.engine.mix_rgb_pair(C.hex_to_rgb(value), (18, 24, 28), 50)
            != legacy.mix_rgb_pair(C.hex_to_rgb(value), (18, 24, 28), 50)
        )
        self.assertGreater(differing, len(PALETTE) // 2)

    def test_the_raw_polynomial_tints_neutral_greys_blue(self):
        """The model's real defect, measured on the RAW polynomial.

        ``filament_mixer`` is a least-squares regression, and it drifts off the
        achromatic axis: a white + black mix comes out blue.  This is genuine
        upstream behaviour (not a transcription bug), which is exactly why the
        product has to correct it — see :class:`NeutralGuardTests`.
        """
        self.assertEqual(FM.mix_pair((255, 255, 255), (0, 0, 0), 90), (203, 228, 238))
        self.assertEqual(FM.mix_pair((255, 255, 255), (0, 0, 0), 50), (99, 126, 159))
        self.assertEqual(FM.mix_pair((255, 255, 255), (0, 0, 0), 10), (4, 29, 46))
        r, _g, b = FM.mix_pair((255, 255, 255), (0, 0, 0), 90)
        self.assertGreater(b, r, "the raw model pushes a neutral mix towards blue")

    def test_the_raw_polynomial_matches_the_reference_screenshots(self):
        """Where the default engine's provenance comes from.

        The user's two Bambu Studio screenshots show a white + black mix at
        90/10 and at 10/90.  Their 效果预览 swatches, measured off the PNGs by
        ``tools/measure_screenshots.py``, are ``#CBE4EE`` and ``#041D2E``.  The
        RAW filament_mixer polynomial reproduces BOTH exactly, while the legacy
        sRGB average returns ``#E5E5E5`` / ``#191919``.

        Note what this does NOT claim: the engine itself deliberately returns
        grey for this pair, because a white/black mix that renders blue is wrong
        (``NeutralGuardTests``).  What is pinned here is that the coefficients we
        ship really are Bambu Studio 02.08's.

        The screenshots live at ``samples/bambu-mix-90-10.png`` and
        ``samples/bambu-mix-10-90.png``.
        """
        self.assertEqual(FM.mix_pair((255, 255, 255), (0, 0, 0), 90), (203, 228, 238))
        self.assertEqual(FM.mix_pair((255, 255, 255), (0, 0, 0), 10), (4, 29, 46))

        legacy = ENGINES[ENGINE_BAMBU]
        self.assertEqual(legacy.mix_hex("#FFFFFF", "#000000", 90), "#E5E5E5")
        self.assertEqual(legacy.mix_hex("#FFFFFF", "#000000", 10), "#191919")
        for percent in (90, 10):
            modern = FM.mix_pair((255, 255, 255), (0, 0, 0), percent)
            old = C.hex_to_rgb(legacy.mix_hex("#FFFFFF", "#000000", percent))
            drift = max(abs(a - b) for a, b in zip(modern, old))
            with self.subTest(percent=percent):
                self.assertGreaterEqual(drift, 7, "the two models should differ visibly here")

    def test_the_white_to_black_sweep_is_monotonic_in_lightness(self):
        previous = 101.0
        for percent in range(100, -1, -5):
            rgb = self.engine.mix_rgb_pair((255, 255, 255), (0, 0, 0), percent)
            lightness = float(C.lab_from_rgb(rgb)[0])
            with self.subTest(percent=percent):
                self.assertLessEqual(lightness, previous + 1e-9)
            previous = lightness


class MixerFeatureLayoutTests(unittest.TestCase):
    """The shipped feature build is a *reordered* copy of the profile's own.

    ``filament_mixer`` lays the 330 monomials out by input and exponent so each
    one costs a single multiply (``_build_layout``), then folds the resulting
    permutation into ``COEF_ORDERED`` instead of permuting the matrix back.  That
    is only correct if the reordered features read against the reordered
    coefficients are the same polynomial the profile stores — which is what
    these check, straight from ``POWERS`` and ``COEF``.

    A whole-catalogue build at 41 spools went from 3.14 s to 0.82 s this way, and
    the 66,420 mixed colours it produces are bit-for-bit the ones the previous
    spelling produced; this test is what keeps that true.
    """

    def test_the_layout_covers_every_feature_exactly_once(self):
        self.assertEqual(len(FM._ORDER), FM.N_FEATURES)
        self.assertEqual(sorted(FM._ORDER), list(range(FM.N_FEATURES)))

    def test_the_reordered_features_are_the_profile_features(self):
        powers = FM.POWERS
        rng = np.random.default_rng(20261010)
        x = rng.uniform(0.0, 255.0, size=(64, powers.shape[1]))

        by_profile = np.ones((x.shape[0], FM.N_FEATURES))
        for feature in range(FM.N_FEATURES):
            for input_index, exponent in enumerate(powers[feature]):
                if exponent:
                    by_profile[:, feature] *= x[:, input_index] ** int(exponent)
        expected = np.clip(by_profile @ FM.COEF + FM.INTERCEPT, 0.0, 255.0).astype(np.int64)

        self.assertTrue(np.array_equal(expected, FM._evaluate(x)))

    def test_the_shortcut_paths_still_agree_with_the_full_polynomial(self):
        # t <= 0 and t >= 1 short-circuit to the untouched input colours, and t
        # is the *second* colour's share, so 0% of the first returns the second.
        self.assertEqual(FM.mix_pair((12, 200, 71), (240, 3, 90), 100), (12, 200, 71))
        self.assertEqual(FM.mix_pair((12, 200, 71), (240, 3, 90), 0), (240, 3, 90))
        self.assertEqual(FM.mix_pair((0, 0, 0), (255, 255, 255), 50),
                         tuple(FM.mix_pair_batch(
                             np.array([[0.0, 0.0, 0.0]]),
                             np.array([[255.0, 255.0, 255.0]]),
                             np.array([0.5]),
                         )[0].tolist()))


class NeutralGuardTests(unittest.TestCase):
    """Two neutral spools must mix to a neutral grey.

    A colourless pair cannot produce a tint, but the pigment polynomial does:
    ``#FFFFFF`` + ``#000000`` at 50 % comes out ``#637E9F``.  Every engine
    therefore passes its result through the same guard.  When **both** parents of
    a recipe are within :data:`NEUTRAL_SPREAD` of grey, the invented hue is
    discarded and the mix becomes the weight-averaged luma of those parents, so
    the engine keeps its own lightness and loses only the false colour.
    """

    def test_black_and_white_mix_to_pure_grey(self):
        # The exact bug the user reported: entering only white and black showed
        # a blue grid.  Every engine now agrees on where the greys are.
        for engine in ENGINES.values():
            for percent, expected in ((90, "#E5E5E5"), (50, "#7F7F7F"), (10, "#191919")):
                with self.subTest(engine=engine.id, percent=percent):
                    self.assertEqual(engine.mix_hex("#FFFFFF", "#000000", percent), expected)

    def test_every_ratio_of_a_neutral_pair_is_colourless(self):
        for engine in ENGINES.values():
            for percent in MIX_RATIOS:
                r, g, b = engine.mix_rgb_pair((255, 255, 255), (0, 0, 0), percent)
                with self.subTest(engine=engine.id, percent=percent):
                    self.assertEqual((r, g, b), (r, r, r))

    def test_a_near_neutral_pair_is_neutralised_too(self):
        # Real spools are never exactly neutral.  These are the sample library's
        # 大简 PETG HF 白 #F2F0EB (spread 7) and 黑 #17181C (spread 5); the raw
        # model tints their 50/50 mix #758396.
        for engine in ENGINES.values():
            for percent in MIX_RATIOS:
                r, g, b = engine.mix_rgb_pair((242, 240, 235), (23, 24, 28), percent)
                with self.subTest(engine=engine.id, percent=percent):
                    self.assertEqual((r, g, b), (r, r, r))

    def test_a_neutral_mix_equals_the_plain_srgb_average(self):
        # For a colourless pair the guard must land exactly where a straight
        # weighted mean would: a neutral mix IS the average of the grey levels.
        # Asserted against the legacy engine rather than against a formula,
        # because the float product 0.6 * 255 truncates to 152, not 153.
        mixer, legacy = ENGINES[ENGINE_MIXER], ENGINES[ENGINE_BAMBU]
        for parents in (((255, 255, 255), (0, 0, 0)), ((242, 240, 235), (23, 24, 28))):
            for percent in MIX_RATIOS:
                with self.subTest(parents=parents, percent=percent):
                    self.assertEqual(
                        mixer.mix_rgb_pair(*parents, percent),
                        legacy.mix_rgb_pair(*parents, percent),
                    )

    def test_a_chromatic_pair_is_left_alone(self):
        # Pigment mixing is the whole point of the default engine, so the guard
        # must not soften it: blue + yellow is still the library's green sample.
        engine = ENGINES[ENGINE_MIXER]
        pair = ((0, 33, 133), (252, 211, 0))
        self.assertEqual(engine.mix_rgb_pair(*pair, 50), (47, 141, 56))
        self.assertEqual(engine.mix_rgb_pair(*pair, 50), FM.mix_pair(*pair, 50))

    def test_one_chromatic_parent_disables_the_guard(self):
        # Black is neutral, #3D7BC8 is not, so this is a real pigment mix.
        engine = ENGINES[ENGINE_MIXER]
        for percent in (10, 50, 90):
            with self.subTest(percent=percent):
                self.assertEqual(
                    engine.mix_rgb_pair((61, 123, 200), (0, 0, 0), percent),
                    FM.mix_pair((61, 123, 200), (0, 0, 0), percent),
                )

    def test_the_threshold_boundary_is_exact(self):
        self.assertEqual(NEUTRAL_SPREAD, 12)
        engine = ENGINES[ENGINE_MIXER]
        # Spread 12 is "neutral": forced to grey.
        r, g, b = engine.mix_rgb_pair((255, 243, 243), (0, 0, 0), 50)
        self.assertEqual((r, g, b), (r, r, r))
        # Spread 13 is a tinted filament: the model is left alone, blue included.
        self.assertEqual(engine.mix_rgb_pair((255, 242, 242), (0, 0, 0), 50), (101, 121, 153))

    def test_a_single_parent_is_never_neutralised(self):
        # K < 2 is nothing to mix; the engine reproduces the input verbatim even
        # when it is a near-neutral colour that would be collapsed otherwise.
        colours = np.asarray([[[242, 240, 235]]], dtype=np.float64)
        weights = np.asarray([[[1.0]]], dtype=np.float64)
        out = ENGINES[ENGINE_MIXER].mix_rgb(colours, weights)
        self.assertEqual(out[0, 0].tolist(), [242, 240, 235])

    def test_the_guard_is_applied_by_the_shared_base_class(self):
        # Any engine added later gets the behaviour for free; a subclass that
        # overrode mix_rgb instead of _mix_rgb would silently skip it.
        for engine in ENGINES.values():
            with self.subTest(engine=engine.id):
                self.assertIs(type(engine).mix_rgb, MixEngine.mix_rgb)


class BambuEngineTests(unittest.TestCase):
    """The production model: Bambu Studio's own dialog arithmetic.

    ``blend_colors`` averages the 8-bit sRGB channels and *truncates* toward
    zero, so 90 % white + 10 % black is 229 (not 230) and 50 % grey is 127.
    """

    def setUp(self):
        self.engine = ENGINES[ENGINE_BAMBU]

    def test_white_black_sweep_matches_the_source(self):
        self.assertEqual(self.engine.mix_hex("#FFFFFF", "#000000", 90), "#E5E5E5")
        self.assertEqual(self.engine.mix_hex("#FFFFFF", "#000000", 50), "#7F7F7F")
        self.assertEqual(self.engine.mix_hex("#FFFFFF", "#000000", 10), "#191919")

    def test_cast_truncates_rather_than_rounds(self):
        self.assertEqual(self.engine.mix_hex("#000000", "#010101", 50), "#000000")

    def test_yellow_and_red(self):
        self.assertEqual(self.engine.mix_hex("#FF0000", "#FFFF00", 50), "#FF7F00")

    def test_sweep_is_a_straight_line_in_srgb(self):
        values = [C.hex_to_rgb(self.engine.mix_hex("#FFFFFF", "#000000", percent))[0]
                  for percent in range(10, 91)]
        steps = [values[index] - values[index + 1] for index in range(len(values) - 1)]
        self.assertLessEqual(max(steps) - min(steps), 1)

    def test_ten_percent_black_barely_darkens(self):
        # The opposite of the spectral engine: Bambu's preview stays very light.
        lab = C.lab_from_rgb(C.hex_to_rgb(self.engine.mix_hex("#FFFFFF", "#000000", 90)))
        self.assertGreater(float(lab[0]), 88.0)

    def test_self_mix_is_exactly_the_original_colour(self):
        for value in PALETTE:
            for percent in (10, 50, 90):
                with self.subTest(color=value, percent=percent):
                    self.assertEqual(self.engine.mix_hex(value, value, percent), value)

    def test_vectorised_matches_scalar(self):
        rgbs = _batch("#3D7BC8", "#FF8800")
        result = self.engine.mix_rgb(rgbs, _weights(30, 70))[0, 0]
        self.assertEqual(C.rgb_to_hex(result), self.engine.mix_hex("#3D7BC8", "#FF8800", 30))

    def test_batch_handles_several_pairs_at_once(self):
        colours = np.array([
            [C.hex_to_rgb("#FFFFFF"), C.hex_to_rgb("#000000")],
            [C.hex_to_rgb("#FF0000"), C.hex_to_rgb("#0000FF")],
        ], dtype=np.float64)
        weights = np.zeros((2, 2, 2), dtype=np.float64)
        weights[:, 0, :] = (0.9, 0.1)
        weights[:, 1, :] = (0.5, 0.5)
        mixed = self.engine.mix_rgb(colours, weights)
        self.assertEqual(mixed.shape, (2, 2, 3))
        self.assertEqual(C.rgb_to_hex(mixed[0, 0]), "#E5E5E5")
        self.assertEqual(C.rgb_to_hex(mixed[0, 1]), "#7F7F7F")
        self.assertEqual(C.rgb_to_hex(mixed[1, 0]), "#E50019")
        self.assertEqual(C.rgb_to_hex(mixed[1, 1]), "#7F007F")


class SpectralEngineTests(unittest.TestCase):
    """The optional physical model, and its documented limits."""

    def setUp(self):
        self.engine = ENGINES[ENGINE_SPECTRAL]

    def test_self_mix_is_close_but_not_exact(self):
        # The reflectance clamp in ks_from_reflectance costs accuracy for
        # saturated colours; measured worst case across the palette is ~20.5.
        for value in PALETTE:
            with self.subTest(color=value):
                mixed = self.engine.mix_hex(value, value, 50)
                distance = C.delta_e_2000(C.lab_from_rgb(C.hex_to_rgb(mixed)),
                                          tuple(C.lab_from_rgb(C.hex_to_rgb(value))))
                self.assertLess(distance, 25.0)

    def test_white_to_black_is_monotonic_in_lightness(self):
        lightness = [float(C.mix_lab(["#FFFFFF", "#000000"], [percent, 100 - percent])[0])
                     for percent in MIX_RATIOS]
        self.assertTrue(all(a < b for a, b in zip(lightness, lightness[1:])))

    def test_ten_percent_black_is_already_very_dark(self):
        # The Kubelka-Munk signature, and the opposite of the Bambu preview.
        lab = C.mix_lab(["#FFFFFF", "#000000"], [90, 10])
        self.assertLess(float(lab[0]), 40.0)

    def test_a_mix_is_far_darker_than_the_srgb_average(self):
        spectral = float(C.mix_lab(["#FFFFFF", "#000000"], [50, 50])[0])
        srgb_average = float(C.lab_from_rgb(C.hex_to_rgb(
            ENGINES[ENGINE_BAMBU].mix_hex("#FFFFFF", "#000000", 50)))[0])
        # Measured: 5.44 versus 53.19 — a difference of nearly 48 L* units.
        self.assertLess(spectral, 20.0)
        self.assertGreater(srgb_average, 50.0)
        self.assertGreater(srgb_average - spectral, 40.0)

    def test_mid_grey_with_itself_stays_grey(self):
        grey = C.hex_from_lab(C.mix_lab(["#808080", "#808080"], [50, 50]))
        for expected, actual in zip(C.hex_to_rgb("#808080"), C.hex_to_rgb(grey)):
            self.assertLessEqual(abs(expected - actual), 2)

    def test_swapping_arguments_swaps_the_result(self):
        self.assertEqual(C.mix_hex(["#F0F0F0", "#101820"], [70, 30]),
                         C.mix_hex(["#101820", "#F0F0F0"], [30, 70]))

    def test_zero_percent_component_is_ignored(self):
        self.assertEqual(C.mix_hex(["#8899AA", "#221100", "#8899AA"], [50, 50, 0]),
                         C.mix_hex(["#8899AA", "#221100"], [50, 50]))

    def test_single_component_reports_failure(self):
        # Upstream returns std::nullopt when fewer than two components are positive.
        self.assertTrue(bool(np.isnan(C.mix_lab(["#FF0000", "#00FF00"], [100, 0])).any()))
        with self.assertRaises(ValueError):
            C.mix_hex(["#FF0000", "#00FF00"], [100, 0])


class CatalogTests(unittest.TestCase):
    def _library(self, count: int) -> FilamentLibrary:
        library = FilamentLibrary()
        for index in range(count):
            library.create(
                brand="大简",
                material_type="PETG HF",
                color_hex=PALETTE[index % len(PALETTE)],
                name=f"Spool {index + 1}",
            )
        return library

    def test_ratio_axis_is_81_steps(self):
        self.assertEqual(len(MIX_RATIOS), 81)
        self.assertEqual(MIX_RATIOS[0], 10)
        self.assertEqual(MIX_RATIOS[-1], 90)

    def test_expected_counts(self):
        for count in (0, 1, 2, 3, 4, 5, 6, 8, 10):
            with self.subTest(filaments=count):
                self.assertEqual(expected_recipe_count(count), count * (count - 1) // 2 * 81)

    def test_catalog_recipe_counts(self):
        for count in (2, 3, 4, 5):
            with self.subTest(filaments=count):
                catalog = MixCatalog(self._library(count)).build()
                self.assertEqual(catalog.recipe_count, expected_recipe_count(count))
                self.assertEqual(catalog.pair_count, count * (count - 1) // 2)
                self.assertEqual(catalog.pair_count * 81, catalog.recipe_count)

    def test_every_recipe_links_back_to_its_parents(self):
        library = self._library(4)
        catalog = MixCatalog(library).build()
        for recipe in catalog.recipes:
            self.assertIn(recipe.a_id, library)
            self.assertIn(recipe.b_id, library)
            self.assertEqual(recipe.percent_a + recipe.percent_b, 100)
            self.assertIn(recipe.percent_a, MIX_RATIOS)
            self.assertRegex(recipe.color_hex, r"^#[0-9A-F]{6}$")
            self.assertEqual(recipe.color_hex, C.rgb_to_hex(recipe.rgb))
            self.assertEqual(recipe.engine, DEFAULT_ENGINE)

    def test_catalog_records_the_selected_engine(self):
        catalog = MixCatalog(self._library(2), engine=ENGINE_SPECTRAL).build()
        self.assertEqual(catalog.engine.id, ENGINE_SPECTRAL)
        self.assertEqual({recipe.engine for recipe in catalog.recipes}, {ENGINE_SPECTRAL})

    def test_three_spools_give_243_mixes(self):
        catalog = MixCatalog(self._library(3)).build()
        self.assertEqual(catalog.recipe_count, 243)
        self.assertEqual(catalog.pair_count, 3)

    def test_pair_lookup_returns_the_same_group(self):
        library = self._library(3)
        catalog = MixCatalog(library).build()
        a, b = library[0].id, library[1].id
        group = catalog.pair_recipes(a, b)
        self.assertEqual(len(group), 81)
        self.assertEqual(len(catalog.pair_recipes(b, a)), 81)
        self.assertEqual(len(catalog.pair_recipes(a, a)), 0)
        self.assertIs(catalog.find_recipe(a, b, 55), group[45])
        self.assertEqual(group[0].percent_a, 10)
        self.assertEqual(group[-1].percent_a, 90)

    def test_nearest_ratio_finds_the_exact_recipe(self):
        library = self._library(2)
        catalog = MixCatalog(library).build()
        a, b = library[0].id, library[1].id
        target = catalog.find_recipe(a, b, 42)
        self.assertEqual(catalog.nearest_ratio(a, b, target.color_hex).key, target.key)

    def test_describe_names_both_parents_and_the_ratio(self):
        library = self._library(2)
        catalog = MixCatalog(library).build()
        text = catalog.describe(catalog.find_recipe(library[0].id, library[1].id, 90))
        for fragment in ("Spool 1", "Spool 2", "90%", "10%"):
            self.assertIn(fragment, text)

    def test_default_sort_is_by_rgb(self):
        catalog = MixCatalog(self._library(3)).build()
        rgb_order = [recipe.rgb for recipe in catalog.sorted_recipes("rgb")]
        self.assertEqual(rgb_order, sorted(rgb_order))

    def test_sort_orders_are_total(self):
        catalog = MixCatalog(self._library(5)).build()
        self.assertEqual({key for key, _ in SORT_CHOICES},
                         {"rgb", "lightness", "hue", "pair", "label"})
        for key, _ in SORT_CHOICES:
            with self.subTest(sort=key):
                self.assertEqual(len(catalog.sorted_recipes(key)), catalog.recipe_count)

    def test_recipes_are_reproducible(self):
        library = self._library(3)
        first = MixCatalog(library).build().recipes
        second = MixCatalog(library).build().recipes
        self.assertEqual(
            [(m.a_id, m.b_id, m.percent_a, m.color_hex) for m in first],
            [(m.a_id, m.b_id, m.percent_a, m.color_hex) for m in second],
        )

    def test_stats(self):
        catalog = MixCatalog(self._library(4)).build()
        self.assertEqual(catalog.stats(),
                         {"filaments": 4, "pairs": 6, "recipes": 486, "ratios": 81})

    # -- incremental sync --------------------------------------------------------
    def test_sync_adding_a_spool_matches_a_full_rebuild(self):
        library = self._library(4)
        catalog = MixCatalog(library).build()
        before = catalog.recipe_count

        library.create(brand="大简", material_type="PETG HF", color_hex="#123456", name="Spool 5")
        catalog.sync(library)

        rebuilt = MixCatalog(library).build()
        self.assertEqual(catalog.recipe_count, expected_recipe_count(5))
        self.assertGreater(catalog.recipe_count, before)
        self.assertEqual(catalog.recipe_count, rebuilt.recipe_count)
        self.assertEqual(catalog.pair_count, rebuilt.pair_count)
        self.assertEqual(
            sorted((m.a_id, m.b_id, m.percent_a, m.color_hex) for m in catalog.recipes),
            sorted((m.a_id, m.b_id, m.percent_a, m.color_hex) for m in rebuilt.recipes),
        )

    def test_sync_reuses_the_recipes_it_already_had(self):
        """The point of sync: untouched recipes keep their identity, so the
        window does not rebuild 66,420 frozen objects on a one-spool edit."""
        library = self._library(4)
        catalog = MixCatalog(library).build()
        untouched = {m.key: m for m in catalog.recipes}

        library.create(brand="大简", material_type="PETG HF", color_hex="#123456", name="Spool 5")
        catalog.sync(library)

        reused = 0
        for recipe in catalog.recipes:
            if recipe.key in untouched:
                reused += 1
                self.assertIs(recipe, untouched[recipe.key])
        self.assertEqual(reused, expected_recipe_count(4))

    def test_sync_deleting_a_spool_drops_only_its_pairs(self):
        library = self._library(5)
        catalog = MixCatalog(library).build()
        victim = library[2]
        survivors = {f.id for f in library if f.id != victim.id}

        library.remove(victim.id)
        catalog.sync(library)

        self.assertEqual(catalog.recipe_count, expected_recipe_count(4))
        self.assertEqual(catalog.pair_count, 6)
        for recipe in catalog.recipes:
            self.assertIn(recipe.a_id, survivors)
            self.assertIn(recipe.b_id, survivors)
        self.assertEqual(catalog.pair_recipes(library[0].id, library[1].id).__len__(), 81)

    def test_sync_rebuilds_when_a_spool_changes_colour(self):
        library = self._library(4)
        catalog = MixCatalog(library).build()
        edited = library[0]
        library.update(edited.id, color_hex="#00FF00")
        catalog.sync(library)

        rebuilt = MixCatalog(library).build()
        self.assertEqual(
            sorted((m.a_id, m.b_id, m.percent_a, m.color_hex) for m in catalog.recipes),
            sorted((m.a_id, m.b_id, m.percent_a, m.color_hex) for m in rebuilt.recipes),
        )

    def test_sync_on_a_brand_new_catalog_builds_it(self):
        library = self._library(3)
        catalog = MixCatalog(library)
        catalog.sync(library)
        self.assertEqual(catalog.recipe_count, 243)

    def test_sync_handles_a_shrinking_then_growing_library(self):
        library = self._library(3)
        catalog = MixCatalog(library).build()
        for filament in list(library):
            library.remove(filament.id)
        catalog.sync(library)
        self.assertEqual(catalog.recipe_count, 0)

        library.create(brand="大简", material_type="PETG HF", color_hex="#654321", name="New")
        library.create(brand="大简", material_type="PETG HF", color_hex="#ABCDEF", name="Newer")
        catalog.sync(library)
        self.assertEqual(catalog.recipe_count, expected_recipe_count(2))


class SortedFilamentTests(unittest.TestCase):
    """「全部颜色」 lists raw spools beside the mixes, under the same 排序 control.

    So the spool ordering has to answer to the same keys as
    :meth:`MixCatalog.sorted_recipes`, with the same directions; otherwise
    flipping the sort would leave the spool rows out of step with everything
    below them.
    """

    def setUp(self) -> None:
        self.filaments = [
            Filament(name="白", brand="b", material_type="PLA", color_hex="#F2F0EB"),
            Filament(name="黑", brand="b", material_type="PLA", color_hex="#17181C"),
            Filament(name="金", brand="b", material_type="PLA", color_hex="#D9A441"),
            Filament(name="红", brand="b", material_type="PLA", color_hex="#C8342E"),
            Filament(name="灰", brand="b", material_type="PLA", color_hex="#808080"),
        ]

    def test_rgb_ascending(self):
        ordered = sorted_filaments(self.filaments, "rgb")
        self.assertEqual([f.rgb for f in ordered], sorted(f.rgb for f in self.filaments))

    def test_lightness_descending_like_the_catalogue(self):
        """The catalogue sorts mixes by ``-lightness``; the spools must match."""
        ordered = sorted_filaments(self.filaments, "lightness")
        levels = [C.lab_from_rgb(f.rgb)[0] for f in ordered]
        self.assertEqual(levels, sorted(levels, reverse=True))

    def test_hue_ascending_like_the_catalogue(self):
        ordered = sorted_filaments(self.filaments, "hue")
        hues = []
        for filament in ordered:
            _, a, b = C.lab_from_rgb(filament.rgb)
            hues.append(0.0 if abs(a) < 1e-9 and abs(b) < 1e-9 else float(np.degrees(np.arctan2(b, a)) % 360.0))
        self.assertEqual(hues, sorted(hues))

    def test_label_is_case_insensitive_by_name(self):
        filaments = [
            Filament(name="Zeta", material_type="PLA", color_hex="#111111"),
            Filament(name="alpha", material_type="PLA", color_hex="#222222"),
            Filament(name="Beta", material_type="PLA", color_hex="#333333"),
        ]
        self.assertEqual(
            [f.name for f in sorted_filaments(filaments, "label")], ["alpha", "Beta", "Zeta"]
        )

    def test_pair_sort_is_a_no_op_for_a_lone_spool(self):
        """A spool is not a parent pair, so 按母材组合 leaves the order alone."""
        self.assertIs(sorted_filaments(self.filaments, "pair")[0], self.filaments[0])
        self.assertEqual(
            [f.id for f in sorted_filaments(self.filaments, "pair")],
            [f.id for f in self.filaments],
        )

    def test_every_sort_key_is_accepted(self):
        for key, _ in SORT_CHOICES:
            with self.subTest(sort=key):
                self.assertEqual(len(sorted_filaments(self.filaments, key)), len(self.filaments))

    def test_an_unknown_key_is_refused(self):
        with self.assertRaises(ValueError):
            sorted_filaments(self.filaments, "nonsense")

    def test_the_input_is_not_mutated(self):
        before = [f.id for f in self.filaments]
        sorted_filaments(self.filaments, "hue")
        self.assertEqual([f.id for f in self.filaments], before)


class SortedCellTests(unittest.TestCase):
    """「全部颜色」 puts spools and mixes in ONE list, ordered by ONE control.

    The user asked for the spools to stop being pinned on top: ticking the
    checkbox must not break the colour order the 排序 combo just established.
    """

    def setUp(self) -> None:
        from app.ui.mix_grid import spool_cell

        self.filaments = [
            Filament(name="白", brand="b", material_type="PLA", color_hex="#F2F0EB"),
            Filament(name="黑", brand="b", material_type="PLA", color_hex="#17181C"),
            Filament(name="金", brand="b", material_type="PLA", color_hex="#D9A441"),
        ]
        self.library = FilamentLibrary()
        for filament in self.filaments:
            self.library.add(filament)
        self.catalog = MixCatalog(self.library).build()
        self.spools = [spool_cell(f) for f in self.filaments]
        self.label_of = lambda fid: (  # noqa: E731 - mirrors MainWindow._label_of
            self.library.get(fid).display_name.casefold()
            if self.library.get(fid) is not None
            else fid
        )

    def _merged(self, key):
        return sorted_cells(
            self.spools + self.catalog.sorted_recipes(key),
            key,
            label_of=self.label_of,
        )

    def test_rgb_ascending_across_both_kinds(self):
        cells = self._merged("rgb")
        self.assertEqual([c.rgb for c in cells], sorted(c.rgb for c in cells))

    def test_a_spool_is_not_pinned_to_the_top(self):
        cells = self._merged("rgb")
        spool_keys = {c.key for c in self.spools}
        self.assertEqual(len(cells), len(self.spools) + self.catalog.recipe_count)
        # Pinning would put every spool in rows 0..2. Under a real RGB order the
        # dark spool does come early, but the pale #F2F0EB cannot: it belongs far
        # down among the pale mixes, so at least one spool sits well past the old
        # pinned block.
        positions = [i for i, c in enumerate(cells) if c.key in spool_keys]
        self.assertEqual(positions, sorted(positions))
        self.assertGreater(max(positions), len(self.spools))

    def test_lightness_descending_across_both_kinds(self):
        cells = self._merged("lightness")
        levels = [c.lightness for c in cells]
        self.assertEqual(levels, sorted(levels, reverse=True))

    def test_hue_ascending_across_both_kinds(self):
        cells = self._merged("hue")
        hues = [c.hue for c in cells]
        self.assertEqual(hues, sorted(hues))

    def test_label_sorts_spools_and_mixes_by_spool_name(self):
        cells = self._merged("label")
        heads = [self.label_of(c.a_id) for c in cells]
        self.assertEqual(heads, sorted(heads))

    def test_similarity_needs_a_target(self):
        with self.assertRaises(ValueError):
            sorted_cells(self.spools, "similarity")

    def test_a_spool_cell_is_shaped_like_a_recipe(self):
        spool = self.spools[0]
        for attribute in (
            "rgb",
            "lab",
            "lightness",
            "hue",
            "chroma",
            "pair_index",
            "percent_a",
            "a_id",
            "b_id",
            "key",
        ):
            with self.subTest(attribute=attribute):
                self.assertTrue(hasattr(spool, attribute))
        self.assertEqual(spool.a_id, spool.filament.id)
        self.assertEqual(spool.b_id, spool.filament.id)
        self.assertLess(spool.pair_index, 0)
        self.assertEqual(spool.ratio_text, "单色 100%")
        self.assertTrue(np.allclose(np.asarray(spool.lab), np.asarray(C.lab_from_rgb(spool.rgb))))

    def test_every_sort_key_keeps_every_cell(self):
        for key, _ in SORT_CHOICES:
            with self.subTest(sort=key):
                self.assertEqual(
                    len(self._merged(key)),
                    len(self.spools) + self.catalog.recipe_count,
                )

    def test_the_input_is_not_mutated(self):
        before = [c.key for c in self.spools]
        self._merged("hue")
        self.assertEqual([c.key for c in self.spools], before)


class LibraryPersistenceTests(unittest.TestCase):
    def test_roundtrip(self):
        library = FilamentLibrary()
        library.create(brand="大简", material_type="PETG HF", color_hex="#123456", note="备注 test")
        library.create(brand="Bambu Lab", material_type="PLA", color_hex="#ABCDEF")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "lib.json"
            library.save(path)
            restored = FilamentLibrary.load(path)
        self.assertEqual(len(restored), 2)
        self.assertEqual(restored[0].color_hex, "#123456")
        self.assertEqual(restored[0].brand, "大简")
        self.assertEqual(restored[0].note, "备注 test")
        self.assertEqual(restored[1].material_type, "PLA")

    def test_missing_file_is_empty_library(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(len(FilamentLibrary.load(Path(temp) / "absent.json")), 0)

    def test_duplicate_ids_are_repaired(self):
        library = FilamentLibrary.from_dict(
            {"filaments": [{"id": "same", "colorHex": "#111111"}, {"id": "same", "colorHex": "#222222"}]}
        )
        self.assertEqual(len(library), 2)
        self.assertNotEqual(library[0].id, library[1].id)

    def test_malformed_entries_are_skipped(self):
        library = FilamentLibrary.from_dict(
            {"filaments": [{"id": "a", "colorHex": "#111111"}, "nonsense", 42]}
        )
        self.assertEqual(len(library), 1)

    def test_update_and_remove(self):
        library = FilamentLibrary()
        filament = library.create(color_hex="#111111", brand="大简")
        updated = library.update(filament.id, color_hex="#222222")
        self.assertEqual(updated.color_hex, "#222222")
        self.assertIs(library.get(filament.id), updated)
        library.remove(filament.id)
        self.assertIsNone(library.get(filament.id))
        self.assertEqual(len(library), 0)

    def test_find_duplicate(self):
        library = FilamentLibrary()
        library.create(brand="大简", material_type="PETG HF", color_hex="#123456")
        self.assertIsNotNone(library.find_duplicate("#123456", "PETG HF", "大简"))
        self.assertIsNone(library.find_duplicate("#123456", "PLA", "大简"))
        self.assertIsNone(library.find_duplicate("#654321", "PETG HF", "大简"))

    def test_display_name_fallback(self):
        self.assertEqual(Filament(brand="大简", material_type="PETG HF").display_name, "大简 PETG HF")
        self.assertEqual(Filament(brand="大简").display_name, "大简")
        self.assertEqual(Filament(color_hex="#010203").display_name, "#010203")
        self.assertEqual(Filament(name="金色", brand="大简").display_name, "金色")

    def test_subtitle_and_sort_by_colour(self):
        filament = Filament(name="金色", brand="大简", material_type="PETG HF", color_hex="#D9A441")
        self.assertEqual(filament.subtitle, "#D9A441 · 大简 · PETG HF")
        library = FilamentLibrary([
            Filament(color_hex="#FFFFFF"),
            Filament(color_hex="#000000"),
            Filament(color_hex="#FF0000"),
        ])
        self.assertEqual([f.color_hex for f in library.sort_by_color()],
                         ["#000000", "#FF0000", "#FFFFFF"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

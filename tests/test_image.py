"""Picture tests: colour reduction, palette matching, and the flat-plate mesh.

Every expected value was measured against the implementation and is justified
by the geometry it describes (pixel counts, triangle counts, voxel-aligned
rectangle decomposition), so a failure here means a real behaviour change.

Run with::

    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.engines import ENGINE_BAMBU, ENGINE_MIXER  # noqa: E402
from app.core.image_matching import (  # noqa: E402
    _PLAN_SPECS,
    MatchSettings,
    TriplePalette,
    _fit_spool_budget,
    _fold_clusters,
    _merge_indistinguishable,
    _merge_similar_colours,
    build_palette,
    match_fidelity,
    match_fidelity_report,
    match_image,
    match_options,
    reduce_colours,
    sort_palette,
)
from app.core.library import Filament, FilamentLibrary  # noqa: E402
from app.core.mixes import (  # noqa: E402
    SORT_CHOICES,
    SORT_CHOICES_WITH_SIMILARITY,
    SORT_SIMILARITY,
    MixCatalog,
)
from app.core.triples import TripleCatalog  # noqa: E402
from app.mesh.threemf import MAX_PAINTED_FILAMENTS  # noqa: E402
from app.spectral import color as _color  # noqa: E402

SPOOLS = (
    ("耗材白", "#F2F0EB"),
    ("耗材黑", "#17181C"),
    ("耗材金", "#D9A441"),
    ("耗材红", "#C8342E"),
)


def _library() -> FilamentLibrary:
    return FilamentLibrary(
        [
            Filament(name=name, brand="测试", material_type="PETG HF", color_hex=value)
            for name, value in SPOOLS
        ]
    )


def _quadrant_image(size: int = 40) -> Image.Image:
    """Four solid quadrants, each exactly one of the test spools."""
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    half = size // 2
    blocks = [(0, 0, SPOOLS[0][1]), (half, 0, SPOOLS[1][1]), (0, half, SPOOLS[2][1]), (half, half, SPOOLS[3][1])]
    for x, y, value in blocks:
        colour = tuple(int(value.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
        image.paste(Image.new("RGBA", (half, half), colour + (255,)), (x, y))
    return image


class ColourReductionTests(unittest.TestCase):
    def test_few_colours_are_kept_exactly(self):
        image = _quadrant_image(20)
        colours, classes = reduce_colours(image.convert("RGBA"), MatchSettings(max_colours=8))
        self.assertEqual(len(colours), 4)
        self.assertEqual(classes.shape, (20, 20))
        self.assertEqual(sorted(set(int(v) for v in np.unique(classes))), [0, 1, 2, 3])
        found = {tuple(int(c) for c in row) for row in colours}
        self.assertEqual(
            found,
            {
                (242, 240, 235),
                (23, 24, 28),
                (217, 164, 65),
                (200, 52, 46),
            },
        )

    def test_a_photo_is_reduced_to_the_requested_count(self):
        ramp = np.zeros((64, 64, 3), dtype=np.uint8)
        ramp[..., 0] = np.arange(256, dtype=np.uint8).reshape(4, 64)[np.arange(64) % 4]
        ramp[..., 1] = np.arange(64, dtype=np.uint8).reshape(64, 1)
        ramp[..., 2] = np.arange(64, dtype=np.uint8).reshape(1, 64)
        image = Image.fromarray(ramp, "RGB").convert("RGBA")
        colours, classes = reduce_colours(image, MatchSettings(max_colours=12))
        self.assertLessEqual(len(colours), 12)
        self.assertGreater(len(colours), 2)
        self.assertEqual(classes.shape, (64, 64))
        self.assertLess(int(classes.max()), len(colours))

    def test_dithering_a_transparent_picture_stays_inside_the_mask(self):
        # 抖动 used to be refused whenever the picture had transparency.  It now
        # runs on the two-dimensional picture with the palette the masked
        # reduction already earned, so the mask still wins.
        cols, rows = 40, 20
        ramp = np.zeros((rows, cols, 3), dtype=np.uint8)
        ramp[..., 0] = np.linspace(60, 250, cols, dtype=np.uint8).reshape(1, cols)
        ramp[..., 1] = np.linspace(30, 200, rows, dtype=np.uint8).reshape(rows, 1)
        ramp[..., 2] = 120
        image = Image.fromarray(ramp, "RGB").convert("RGBA")
        mask = np.ones((rows, cols), dtype=bool)
        mask[:, :6] = False  # a transparent strip on the left

        plain, _ = reduce_colours(image, MatchSettings(max_colours=4, dither=False), mask)
        colours, classes = reduce_colours(image, MatchSettings(max_colours=4, dither=True), mask)
        self.assertLessEqual(len(colours), 4)
        self.assertTrue((classes[:, :6] == -1).all(), "the transparent strip stays unprinted")
        self.assertTrue((classes[:, 6:] >= 0).all(), "the artwork is printed in full")
        self.assertLess(int(classes.max()), len(colours))

    def test_a_masked_dither_really_changes_the_class_map(self):
        # A no-op would satisfy every invariant above, so compare against the
        # undithered reduction: error diffusion has to move pixels.
        cols, rows = 48, 24
        ramp = np.zeros((rows, cols, 3), dtype=np.uint8)
        ramp[..., 0] = np.linspace(40, 240, cols, dtype=np.uint8).reshape(1, cols)
        ramp[..., 1] = 90
        ramp[..., 2] = np.linspace(200, 60, rows, dtype=np.uint8).reshape(rows, 1)
        image = Image.fromarray(ramp, "RGB").convert("RGBA")
        mask = np.ones((rows, cols), dtype=bool)
        mask[0, :] = False  # transparency along the top edge only
        settings = MatchSettings(max_colours=3, smooth_radius=0)
        _, plain = reduce_colours(image, settings, mask)
        colours, dithered = reduce_colours(
            image, replace(settings, dither=True), mask
        )
        self.assertTrue((dithered[:, :] >= 0).all() or (dithered[mask] >= 0).all())
        self.assertTrue((dithered[~mask] == -1).all())
        self.assertGreater(int(np.count_nonzero(dithered != plain)), 0)
        # A dithered picture spreads over the palette instead of using one flat
        # class per band, which is the whole point of the switch.
        self.assertGreater(len(np.unique(dithered[mask])), 1)


class PaletteTests(unittest.TestCase):
    def test_palette_is_every_spool_then_every_mix(self):
        library = _library()
        catalog = MixCatalog(library.filaments, engine=ENGINE_BAMBU).build()
        palette = build_palette(library, catalog)
        # 4 spools + one entry per DISTINCT mixed colour (identical hexes merge,
        # so this is never more than C(4,2) pairs × 81 ratios).
        mixed = [entry for entry in palette if entry.is_mix]
        self.assertEqual(len(palette), 4 + len(mixed))
        self.assertLessEqual(len(mixed), 6 * 81)
        self.assertEqual(len({entry.color_hex for entry in mixed}), len(mixed))
        self.assertEqual([entry.kind for entry in palette[:4]], ["filament"] * 4)
        self.assertTrue(all(entry.kind == "mix" for entry in palette[4:]))
        self.assertTrue(all(len(entry.filament_ids) == 2 for entry in palette[4:]))
        self.assertTrue(all(entry.filament_ids[0] != entry.filament_ids[1] for entry in palette[4:]))

    def test_a_merged_entry_carries_every_recipe_behind_it(self):
        library = _library()
        catalog = MixCatalog(library.filaments, engine=ENGINE_BAMBU).build()
        palette = build_palette(library, catalog)
        mixed = [entry for entry in palette if entry.is_mix]
        self.assertEqual(sum(entry.recipe_count for entry in mixed), 6 * 81)
        for entry in mixed:
            self.assertEqual(entry.recipes, tuple(catalog.colour_for(entry.color_hex).recipes))
            self.assertTrue(all(r.color_hex == entry.color_hex for r in entry.recipes))
        many = [entry for entry in mixed if entry.recipe_count > 1]
        if many:
            self.assertEqual(many[0].ratio_text, f"{len(many[0].recipes)} 个配方")
        single = [entry for entry in mixed if entry.recipe_count == 1]
        self.assertTrue(single)
        self.assertEqual(single[0].ratio_text, single[0].recipe.ratio_text)

    def test_palette_can_exclude_mixes(self):
        library = _library()
        catalog = MixCatalog(library.filaments, engine=ENGINE_BAMBU).build()
        palette = build_palette(library, catalog, include_mixes=False)
        self.assertEqual(len(palette), 4)
        self.assertTrue(all(not entry.is_mix for entry in palette))

    def test_spool_entries_report_their_filament(self):
        library = _library()
        palette = build_palette(library, None)
        self.assertEqual(palette[0].color_hex, "#F2F0EB")
        self.assertEqual(palette[0].rgb, (242, 240, 235))
        self.assertEqual(palette[0].label, "耗材白")
        self.assertEqual(palette[0].short_label, "耗材白")


class MatchTests(unittest.TestCase):
    def setUp(self):
        self.library = _library()
        self.catalog = MixCatalog(self.library.filaments, engine=ENGINE_BAMBU).build()
        self.palette = build_palette(self.library, self.catalog)

    def test_exact_spool_colours_match_their_own_spool(self):
        result = match_image(_quadrant_image(40), self.palette, MatchSettings(max_colours=8))
        self.assertEqual(len(result.palette), 4)
        self.assertEqual(sorted(int(v) for v in result.counts), [400, 400, 400, 400])
        self.assertEqual(result.coverage, 1.0)
        self.assertTrue(all(not entry.is_mix for entry in result.palette))
        self.assertEqual({entry.color_hex for entry in result.palette}, {v for _, v in SPOOLS})

    def test_counts_are_sorted_largest_first(self):
        result = match_image(_quadrant_image(40), self.palette, MatchSettings(max_colours=8))
        self.assertEqual(list(result.counts), sorted(result.counts, reverse=True))
        shares = result.shares()
        self.assertAlmostEqual(float(shares.sum()), 1.0, places=9)

    def test_transparent_pixels_are_left_out(self):
        image = _quadrant_image(40)
        image.paste(Image.new("RGBA", (20, 20), (0, 0, 0, 0)), (0, 0))
        result = match_image(image, self.palette, MatchSettings(max_colours=8))
        self.assertEqual(int(np.count_nonzero(result.indices < 0)), 400)
        self.assertEqual(int(result.counts.sum()), 1600 - 400)
        self.assertAlmostEqual(result.coverage, 0.75, places=9)

    def test_a_gradient_respects_the_colour_budget(self):
        # A gradient between the two tinted spools (金 → 红): the picture needs
        # more than one colour, but must still fit inside the budget.
        #
        # Deliberately NOT a synthetic ramp with a pinned blue channel.  Such a
        # picture is genuinely nearest to the near-black spool for every pixel —
        # which now collapses it to a single colour, because the neutral guard
        # removed the spurious blue tint that used to make grey mixes look like a
        # closer match.  That would assert the guard's absence, not the budget.
        height = width = 48
        top = np.array([217, 164, 65], dtype=np.float64)     # 耗材金 #D9A441
        bottom = np.array([200, 52, 46], dtype=np.float64)   # 耗材红 #C8342E
        blend = np.linspace(0.0, 1.0, height)[:, None, None]
        ramp = np.rint(top[None, None, :] * (1.0 - blend) + bottom[None, None, :] * blend)
        ramp = np.repeat(ramp, width, axis=1).astype(np.uint8)
        image = Image.fromarray(ramp, "RGB").convert("RGBA")
        result = match_image(image, self.palette, MatchSettings(max_colours=6))
        self.assertLessEqual(len(result.palette), 6)
        self.assertGreaterEqual(len(result.palette), 2)
        self.assertEqual(int(result.counts.sum()), 48 * 48)

    def test_matching_is_deterministic(self):
        first = match_image(_quadrant_image(32), self.palette, MatchSettings(max_colours=8))
        second = match_image(_quadrant_image(32), self.palette, MatchSettings(max_colours=8))
        self.assertTrue(np.array_equal(first.indices, second.indices))
        self.assertEqual([e.key for e in first.palette], [e.key for e in second.palette])
        self.assertEqual(list(first.counts), list(second.counts))

    def test_the_image_is_downscaled_to_the_budget(self):
        image = Image.new("RGBA", (800, 400), (242, 240, 235, 255))
        result = match_image(image, self.palette, MatchSettings(max_dimension=200))
        self.assertEqual((result.width, result.height), (200, 100))

    def test_a_mix_can_win_over_a_spool(self):
        # Only two spools, so the mid tone of black and white is a mix, not a spool.
        library = FilamentLibrary(
            [
                Filament(name="白", brand="测试", material_type="PETG HF", color_hex="#FFFFFF"),
                Filament(name="黑", brand="测试", material_type="PETG HF", color_hex="#000000"),
            ]
        )
        catalog = MixCatalog(library.filaments, engine=ENGINE_BAMBU).build()
        palette = build_palette(library, catalog)
        image = Image.new("RGBA", (8, 8), (127, 127, 127, 255))
        result = match_image(image, palette, MatchSettings(max_colours=4))
        self.assertEqual(len(result.palette), 1)
        entry = result.palette[0]
        self.assertTrue(entry.is_mix, f"expected a mix, got {entry.label}")
        self.assertTrue(set(entry.filament_ids) == {f.id for f in library})
        self.assertIn("50%", entry.label)

    def test_an_empty_palette_is_refused(self):
        with self.assertRaises(ValueError):
            match_image(_quadrant_image(8), [])


class RegionColourTests(unittest.TestCase):
    """The UI shows 「图片里的颜色 → 匹配到的颜色」, so the region mean must be real."""

    def setUp(self) -> None:
        library = _library()
        self.palette = build_palette(library, MixCatalog(library.filaments).build())

    def test_the_mean_is_the_picture_colour_not_the_palette_colour(self):
        # One flat region of a colour that is nobody's spool: the region mean has
        # to report the PICTURE's colour, which is the whole point of showing it
        # next to the match.
        image = Image.new("RGBA", (16, 16), (250, 212, 181, 255))
        result = match_image(image, self.palette, MatchSettings(max_colours=4))
        self.assertEqual(len(result.palette), 1)
        self.assertEqual(result.region_colour(0), (250, 212, 181))
        self.assertNotEqual(result.region_colour(0), result.palette[0].rgb)

    def test_every_region_has_a_mean(self):
        result = match_image(_quadrant_image(40), self.palette, MatchSettings(max_colours=8))
        self.assertIsNotNone(result.means)
        self.assertEqual(result.means.shape, (len(result.palette), 3))
        for index in range(len(result.palette)):
            self.assertIsInstance(result.region_colour(index), tuple)
            self.assertEqual(len(result.region_colour(index)), 3)

    def test_without_means_it_falls_back_to_the_menu_colour(self):
        # ``means`` is optional: a result built without it (an older file, a
        # hand-made one) still has to answer, using the matched colour.
        result = match_image(_quadrant_image(16), self.palette, MatchSettings(max_colours=8))
        bare = replace(result, means=None)
        self.assertEqual(bare.region_colour(0), bare.palette[0].rgb)


class SortPaletteTests(unittest.TestCase):
    """「全部颜色」 in the replacement dialog, ordered like the 颜色配方 grid."""

    def setUp(self) -> None:
        library = _library()
        self.catalog = MixCatalog(library.filaments, engine=ENGINE_BAMBU).build()
        self.entries = build_palette(library, self.catalog, include_mixes=True)
        self.target = (250, 212, 181)

    def test_similarity_leads_with_the_closest_colour(self):
        ordered = sort_palette(self.entries, SORT_SIMILARITY, target_rgb=self.target)
        differences = [
            float(_color.delta_e_2000(entry.lab, _color.lab_from_rgb(self.target)))
            for entry in ordered
        ]
        self.assertEqual(differences, sorted(differences))
        self.assertEqual(len(ordered), len(self.entries))

    def test_similarity_needs_a_target_colour(self):
        with self.assertRaises(ValueError):
            sort_palette(self.entries, SORT_SIMILARITY)

    def test_the_dialog_list_is_the_grid_list_plus_similarity(self):
        self.assertEqual(
            [key for key, _ in SORT_CHOICES_WITH_SIMILARITY],
            [key for key, _ in SORT_CHOICES] + [SORT_SIMILARITY],
        )
        # The 颜色配方 page keeps its five keys: similarity is meaningless there.
        self.assertNotIn(SORT_SIMILARITY, [key for key, _ in SORT_CHOICES])

    def test_every_key_keeps_every_entry(self):
        for key, _ in SORT_CHOICES_WITH_SIMILARITY:
            with self.subTest(sort=key):
                ordered = sort_palette(self.entries, key, target_rgb=self.target)
                self.assertEqual(len(ordered), len(self.entries))
                self.assertEqual({e.key for e in ordered}, {e.key for e in self.entries})

    def test_spools_and_mixes_are_grouped_not_interleaved_by_label(self):
        ordered = sort_palette(self.entries, "label", target_rgb=self.target)
        kinds = [entry.is_mix for entry in ordered]
        self.assertEqual(kinds, sorted(kinds), "spools come first under 按名称")


class IndistinguishableColourTests(unittest.TestCase):
    """Two colours a hair apart must not both claim part of one flat region.

    The picture's own noise decides which of two near-equal palette entries wins
    each pixel, so without a merge a flat region comes out speckled and the
    highlight turns to stripes. Measured on a real illustration: 107 index
    changes along one 512-pixel row before the merge, 7 after.
    """

    def setUp(self) -> None:
        library = _library()
        self.palette = build_palette(library, MixCatalog(library.filaments).build())

    def _noisy_flat(self, size: int = 64, sigma: float = 1.6) -> Image.Image:
        rng = np.random.default_rng(7)
        base = np.array([250, 212, 181], dtype=np.float64)
        pixels = np.clip(np.rint(base + rng.normal(0.0, sigma, (size, size, 3))), 0, 255)
        return Image.fromarray(pixels.astype(np.uint8), "RGB").convert("RGBA")

    def test_fine_source_noise_does_not_speckle_a_flat_region(self):
        result = match_image(self._noisy_flat(), self.palette, MatchSettings(max_colours=12))
        transitions = int(np.count_nonzero(result.indices[32][1:] != result.indices[32][:-1]))
        self.assertLessEqual(transitions, 2, f"the region is speckled: {transitions} changes")

    def test_switching_the_merge_off_brings_the_speckle_back(self):
        # Proves the merge is what fixes it, rather than the picture being easy.
        # Smoothing is switched off too, because averaging the texture away is
        # the *other* defence against speckle and would hide this one.
        settings = MatchSettings(max_colours=12, merge_delta_e=0.0, smooth_radius=0)
        result = match_image(self._noisy_flat(), self.palette, settings)
        transitions = int(np.count_nonzero(result.indices[32][1:] != result.indices[32][:-1]))
        self.assertGreater(transitions, 2)

    def test_smoothing_alone_already_calms_the_speckle(self):
        settings = MatchSettings(max_colours=12, merge_delta_e=0.0, smooth_radius=2)
        result = match_image(self._noisy_flat(), self.palette, settings)
        transitions = int(np.count_nonzero(result.indices[32][1:] != result.indices[32][:-1]))
        self.assertLessEqual(transitions, 2, f"the region is speckled: {transitions} changes")

    def test_smoothing_cannot_push_a_pixel_outside_the_picture(self):
        # The mean filter must not average the transparent surround in, and it
        # must leave the mask intact.
        image = self._noisy_flat(size=32)
        mask = np.zeros((32, 32), dtype=bool)
        mask[8:24, 8:24] = True
        colours, classes = reduce_colours(image, MatchSettings(smooth_radius=3), mask)
        self.assertTrue((classes[mask] >= 0).all())
        self.assertTrue((classes[~mask] == -1).all())
        self.assertGreater(len(colours), 0)

    def test_source_clusters_closer_than_the_threshold_are_folded(self):
        colours = np.array([[250, 212, 181], [250, 212, 183], [10, 10, 10]], dtype=np.uint8)
        classes = np.zeros((2, 6), dtype=np.int32)
        classes[1, 0] = 1  # the second shade owns one pixel
        classes[1, 1:] = 2
        merged, remapped = _merge_similar_colours(colours, classes, 2.0)
        self.assertEqual(len(merged), 2, "the two near-identical shades collapse")
        # The shade covering the most pixels survives, so the odd pixel moves.
        self.assertEqual(tuple(merged[0]), (250, 212, 181))
        self.assertEqual(int(remapped[1, 0]), 0)

    def test_a_threshold_of_zero_changes_nothing(self):
        colours = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
        classes = np.zeros((2, 2), dtype=np.int32)
        kept, remapped = _merge_similar_colours(colours, classes, 0.0)
        self.assertTrue(np.array_equal(kept, colours))
        self.assertTrue(np.array_equal(remapped, classes))

    def test_matched_colours_closer_than_the_threshold_are_folded(self):
        library = _library()
        palette = build_palette(library, MixCatalog(library.filaments).build())
        # Two entries that are genuinely indistinguishable, plus one that is not.
        pale = palette[0]
        near = pale.__class__(**{**pale.__dict__, "key": pale.key + "|twin", "color_hex": "#F2F0EC"})
        kept = [pale, near]
        counts = np.array([10, 3], dtype=np.int64)
        indices = np.zeros((2, 8), dtype=np.int16)
        indices[1, :3] = 1
        merged, new_counts, new_indices = _merge_indistinguishable(kept, counts, indices, 2.0)
        self.assertEqual(len(merged), 1)
        self.assertEqual(int(new_counts[0]), 16)
        self.assertEqual(int(np.count_nonzero(new_indices != 0)), 0)

    def test_the_merge_threshold_is_clamped(self):
        self.assertEqual(MatchSettings(merge_delta_e=-5.0).clamped().merge_delta_e, 0.0)
        self.assertEqual(MatchSettings(merge_delta_e=99.0).clamped().merge_delta_e, 10.0)
        self.assertEqual(MatchSettings(merge_delta_e=1.5).clamped().merge_delta_e, 1.5)


class RecipeTextTests(unittest.TestCase):
    """The colour list has to show the whole recipe, percentages included.

    Measured in the real window: the full form
    「配方：Bambu Lab PLA Basic (#FCF4F0) 71%  +  Sunlu PLA Silk (#E8A0A8) 29%」
    is about 70 characters wide and QListWidget elided it, clipping the second
    percentage off. The compact form keeps both percentages; the full one stays
    in the tooltip and in the 「选中的颜色」 panel.
    """

    def setUp(self) -> None:
        library = _library()
        self.library = library
        self.catalog = MixCatalog(library.filaments, engine=ENGINE_BAMBU).build()
        self.entries = build_palette(library, self.catalog, include_mixes=True)

    def test_compact_keeps_both_percentages_and_the_full_form_keeps_the_hex(self):
        from app.ui.colour_picker import entry_recipe_text

        mix = next(entry for entry in self.entries if entry.is_mix)
        full = entry_recipe_text(mix, self.library)
        compact = entry_recipe_text(mix, self.library, compact=True)
        self.assertIn("配方", full)
        self.assertIn("(#", full, "the full form spells out the parent colours")
        self.assertIn("配方", compact)
        self.assertNotIn("(#", compact, "the compact form drops the hex to make room")
        for text in (full, compact):
            self.assertIn(f"{mix.recipe.percent_a}%", text)
            self.assertIn(f"{mix.recipe.percent_b}%", text)
        self.assertLess(len(compact), len(full))

    def test_a_raw_spool_is_never_called_a_recipe(self):
        from app.ui.colour_picker import entry_recipe_text

        spool = next(entry for entry in self.entries if not entry.is_mix)
        for text in (
            entry_recipe_text(spool, self.library),
            entry_recipe_text(spool, self.library, compact=True),
        ):
            self.assertIn("耗材本色", text)
            self.assertNotIn("配方", text)

    def test_a_missing_library_still_renders(self):
        from app.ui.colour_picker import entry_recipe_text

        mix = next(entry for entry in self.entries if entry.is_mix)
        text = entry_recipe_text(mix, None)
        self.assertIn("配方", text)
        self.assertIn("%", text)


class LargePaletteTests(unittest.TestCase):
    """A palette far bigger than one picture ever uses.

    The per-pixel class map used to be ``int16``, so any index past 32,767
    wrapped negative — and negative means "transparent" everywhere downstream,
    so those colours were silently deleted and a five-colour pig came out with
    two.  These tests pin the dtype to the palette, not to the number of colours
    a picture happens to need.

    Since identical ``#RRGGBB`` values are merged (m06962) the palette has to be
    built from a WELL-SPREAD library to cross 32,767 at all: 30 spools scattered
    over the RGB cube give ~33,200 entries, where 30 collinear spools would
    collapse to ~14,500.
    """

    @staticmethod
    def _wide_library(count: int = 30) -> FilamentLibrary:
        # Deterministic, but spread over the whole cube so the mixes stay apart.
        rng = np.random.default_rng(20261011)
        filaments = []
        for index in range(count):
            r, g, b = (int(value) for value in rng.integers(0, 256, size=3))
            filaments.append(
                Filament(
                    name=f"P{index}",
                    brand="测试",
                    material_type="PETG HF",
                    color_hex=f"#{r:02X}{g:02X}{b:02X}",
                )
            )
        return FilamentLibrary(filaments)

    def test_the_palette_is_bigger_than_the_int16_range(self):
        library = self._wide_library()
        catalog = MixCatalog(library.filaments, engine=ENGINE_BAMBU).build()
        palette = build_palette(library, catalog)
        self.assertGreater(len(palette), np.iinfo(np.int16).max)

    def test_no_matched_colour_is_lost_past_index_32767(self):
        library = self._wide_library()
        catalog = MixCatalog(library.filaments, engine=ENGINE_BAMBU).build()
        palette = build_palette(library, catalog)

        # Five solid bands.  Run them through the picture-matching path, then
        # check every band still owns at least one pixel: a wrapped index reads
        # as transparent and the band would come back empty.
        bands = [(250, 212, 181), (222, 76, 96), (60, 50, 45), (240, 236, 226), (120, 170, 200)]
        size = 40
        row = np.zeros((size, size, 3), dtype=np.uint8)
        for index, colour in enumerate(bands):
            row[index * (size // len(bands)) : (index + 1) * (size // len(bands)), :] = colour
        image = Image.fromarray(row, "RGB").convert("RGBA")

        result = match_image(image, palette, MatchSettings(max_colours=8, smooth_radius=0))
        self.assertEqual(int(np.count_nonzero(result.indices < 0)), 0)
        self.assertLess(int(result.indices.max()), len(result.palette))
        self.assertEqual(int(result.counts.sum()), size * size)
        for colour in bands:
            key = "".join(f"{value:02X}" for value in colour)
            index = next(
                (i for i, entry in enumerate(result.palette) if entry.color_hex.endswith(key)),
                None,
            )
            if index is None:
                continue  # the matcher is allowed to pick a mix instead
            self.assertGreater(int(result.counts[index]), 0, colour)


class SimilaritySortScaleTests(unittest.TestCase):
    """「按跟图片目标颜色最相似排序」 over a whole 35,265-entry palette.

    Regression: the 更换颜色 dialog listed every candidate as its own widget and
    called ``delta_e_2000`` once per candidate in a Python loop, so on the bundled
    41-spool preset it took 26 seconds to open and Windows greyed it out as
    「(未响应)」.  The sort is now one vectorised CIEDE2000 call; this pins that it
    is still a *global* sort, not just a sorted prefix of the first candidates.
    """

    def test_the_similarity_order_is_the_global_minimum_first(self):
        library = LargePaletteTests._wide_library()
        catalog = MixCatalog(library.filaments, engine=ENGINE_BAMBU).build()
        palette = build_palette(library, catalog)
        target = (250, 212, 181)

        ordered = sort_palette(palette, SORT_SIMILARITY, target_rgb=target)
        self.assertEqual(len(ordered), len(palette))

        labs = np.asarray([entry.lab for entry in palette], dtype=np.float64)
        target_lab = _color.lab_from_rgb(target)
        every = np.asarray(_color.delta_e_2000(labs, target_lab), dtype=np.float64)

        by_key = {entry.key: position for position, entry in enumerate(palette)}
        self.assertEqual(
            [by_key[entry.key] for entry in ordered[:20]],
            [int(position) for position in np.argsort(every, kind="stable")[:20]],
        )
        self.assertAlmostEqual(
            float(_color.delta_e_2000(ordered[0].lab, target_lab)),
            float(every.min()),
            places=9,
        )

    def test_an_empty_palette_sorts_to_nothing(self):
        self.assertEqual(sort_palette([], SORT_SIMILARITY, target_rgb=(1, 2, 3)), [])


class SmallDistinctRegionTests(unittest.TestCase):
    """A tiny distinct region must survive a big flat field.

    Median cut splits the box with the most pixels, so a flat body with a few
    units of texture eats the whole colour budget and the eyes come back light
    grey — the user's 「颜色识别错误，不止两个颜色」.
    """

    def test_a_small_dark_patch_survives_a_big_flat_field(self):
        # A behavioural guard for the user's «不止两个颜色» complaint: a region
        # covering 1 % of the picture must still reach the palette.
        library = _library()
        catalog = MixCatalog(library.filaments, engine=ENGINE_BAMBU).build()
        palette = build_palette(library, catalog)

        rng = np.random.default_rng(20261009)
        shades = np.array([(250, 212, 181), (249, 210, 179), (251, 213, 182), (248, 209, 178),
                           (250, 211, 180), (252, 214, 183), (247, 208, 177), (249, 212, 181),
                           (251, 211, 179), (248, 212, 182), (250, 213, 180), (249, 209, 178)],
                          dtype=np.int16)
        field = np.empty((200, 200, 3), dtype=np.int16)
        for row_index, row_block in enumerate(np.array_split(np.arange(200), len(shades))):
            field[row_block, :, :] = shades[row_index]
        field += rng.integers(-1, 2, size=field.shape, dtype=np.int16)
        field[90:110, 90:110] = (60, 50, 45)  # 400 px out of 40,000 = 1 %
        image = Image.fromarray(np.clip(field, 0, 255).astype(np.uint8), "RGB").convert("RGBA")

        result = match_image(image, palette, MatchSettings(max_colours=6, smooth_radius=0))
        self.assertLessEqual(len(result.palette), 6)

        darkest = min(entry.rgb for entry in result.palette)
        self.assertLess(
            max(darkest),
            140,
            f"the dark patch was folded away: {[e.color_hex for e in result.palette]}",
        )
        # And it must actually own pixels, not just sit in the kept list.
        index = result.palette.index(next(e for e in result.palette if e.rgb == darkest))
        self.assertGreater(int(result.counts[index]), 100)

    def test_a_flat_field_is_not_shredded_into_shades(self):
        rng = np.random.default_rng(7)
        field = np.full((120, 120, 3), (233, 196, 168), dtype=np.int16)
        field += rng.integers(-2, 3, size=field.shape, dtype=np.int16)
        image = Image.fromarray(np.clip(field, 0, 255).astype(np.uint8), "RGB").convert("RGBA")

        library = _library()
        palette = build_palette(library, MixCatalog(library.filaments, engine=ENGINE_BAMBU).build())
        result = match_image(image, palette, MatchSettings(max_colours=8))
        self.assertLessEqual(len(result.palette), 3)


class ClusterFoldingTests(unittest.TestCase):
    """``_fold_clusters`` must not chain.

    A union-find over "everything within the threshold" collapses a smooth
    gradient to one colour, because every cluster is within the threshold of
    its neighbour.  A leader pass only ever compares a cluster to the leaders,
    so the ends of a ramp stay apart while a flat fill still folds to one.
    """

    @staticmethod
    def _fold(labs, limit=8, threshold=2.0):
        colours = np.array([_color.rgb_from_lab(lab) for lab in labs], dtype=np.int64)
        classes = np.repeat(np.arange(len(labs), dtype=np.int64), 10)
        classes = np.repeat(classes.reshape(1, -1), 10, axis=0)
        return _fold_clusters(colours, classes, limit, threshold)

    def test_a_chain_of_close_colours_does_not_collapse_to_one(self):
        labs = [[50.0, 0.0, 0.0], [51.5, 0.0, 0.0], [53.0, 0.0, 0.0], [54.5, 0.0, 0.0]]
        for near, far in zip(labs, labs[1:]):
            self.assertLess(_color.delta_e_2000(near, far), 2.0)
        self.assertGreaterEqual(_color.delta_e_2000(labs[0], labs[-1]), 2.0)

        colours, classes = self._fold(labs)
        # Union-find would give 1 here; the leader pass gives 50 and 53.
        self.assertEqual(len(colours), 2)
        self.assertEqual(sorted(set(int(v) for v in np.unique(classes))), [0, 1])

    def test_colours_all_within_the_threshold_still_fold_to_one(self):
        # 8-bit rounding of ``rgb_from_lab`` moves ΔE00 by a few hundredths, so
        # the spread stays well inside the threshold rather than on its edge.
        labs = [[50.0, 0.0, 0.0], [50.4, 0.0, 0.0], [50.8, 0.0, 0.0], [51.2, 0.0, 0.0]]
        for lab in labs[1:]:
            self.assertLess(_color.delta_e_2000(labs[0], lab), 2.0)
        colours, classes = self._fold(labs)
        self.assertEqual(len(colours), 1)
        self.assertEqual(set(int(v) for v in np.unique(classes)), {0})

    def test_the_budget_caps_the_number_of_leaders(self):
        labs = [[float(20 + 12 * index), 0.0, 0.0] for index in range(8)]
        colours, _ = self._fold(labs, limit=3)
        self.assertEqual(len(colours), 3)
        # The largest-first walk makes the first cluster a leader, so colour 0
        # is always kept; nothing here is size-weighted, they are all equal.
        self.assertEqual(len(self._fold(labs, limit=1)[0]), 1)

    def test_a_single_cluster_is_returned_untouched(self):
        colours = np.array([[10, 20, 30]], dtype=np.int64)
        classes = np.zeros((4, 4), dtype=np.int64)
        out_colours, out_classes = _fold_clusters(colours, classes, 8, 2.0)
        self.assertEqual(len(out_colours), 1)
        self.assertEqual(int(out_colours[0][0]), 10)
        self.assertEqual(int(out_classes[0, 0]), 0)

    def test_transparent_pixels_stay_transparent(self):
        labs = [[40.0, 0.0, 0.0], [80.0, 0.0, 0.0]]
        colours = np.array([_color.rgb_from_lab(lab) for lab in labs], dtype=np.int64)
        classes = np.array([[0, 1, -1, -1], [0, 1, -1, -1]], dtype=np.int64)
        _, out_classes = _fold_clusters(colours, classes, 8, 2.0)
        self.assertEqual(list(out_classes[0]), [0, 1, -1, -1])


class TriplePalettePoolTests(unittest.TestCase):
    """「也算三色配方」: a bigger pool without materialising every colour.

    The four-spool test library has four three-filament combinations, which is
    small enough that every assertion below can look at the whole tail; the
    point being tested is the structure (head first, tail lazy, three parents
    per tail colour), not the size.
    """

    def setUp(self):
        self.library = _library()
        self.catalog = MixCatalog(self.library.filaments, engine=ENGINE_BAMBU).build()
        self.triples = TripleCatalog(
            self.library.filaments, engine=ENGINE_MIXER
        ).build()
        self.head = build_palette(self.library, self.catalog)
        self.palette = TriplePalette(self.head, self.triples, self.library)

    def _tail_entries(self):
        return [self.palette[index] for index in range(len(self.head), len(self.palette))]

    def test_the_head_comes_first_untouched(self):
        self.assertEqual(self.palette[: len(self.head)], self.head)
        self.assertIs(self.palette.head, self.palette.head)

    def test_the_pool_really_is_bigger_than_the_two_filament_one(self):
        self.assertGreater(self.triples.colour_count, 0)
        self.assertGreater(len(self.palette), len(self.head))

    def test_a_tail_colour_names_three_spools(self):
        entry = self.palette[len(self.head)]
        self.assertTrue(entry.is_mix)
        self.assertIsNone(entry.recipe)
        self.assertEqual(entry.recipes, ())
        self.assertEqual(len(entry.parts), 3)
        self.assertEqual(len(entry.filament_ids), 3)
        for _parent, percent in entry.parts:
            self.assertGreater(percent, 0)
        self.assertEqual(sum(percent for _parent, percent in entry.parts), 100)
        self.assertIn("%", entry.ratio_text)

    def test_a_slice_is_the_same_as_indexing_one_by_one(self):
        # Compare keys, not entries: two tail entries carry numpy Lab vectors,
        # so ``==`` on them is ambiguous by design.
        start = len(self.head)
        self.assertEqual(
            [entry.key for entry in self.palette[start : start + 3]],
            [self.palette[start + offset].key for offset in range(3)],
        )

    def test_the_head_colour_is_never_offered_twice(self):
        head_hexes = {entry.color_hex for entry in self.head}
        tail_hexes = {entry.color_hex for entry in self._tail_entries()}
        self.assertEqual(head_hexes & tail_hexes, set())
        self.assertEqual(len(tail_hexes), len(self._tail_entries()))

    def test_the_lab_table_lines_up_with_the_entries(self):
        table = self.palette.lab_table()
        self.assertEqual(table.shape, (len(self.palette), 3))
        for index in (0, len(self.head), len(self.head) + 1, len(self.palette) - 1):
            self.assertTrue(np.allclose(table[index], self.palette[index].lab))

    def test_the_table_is_built_once(self):
        self.assertIs(self.palette.lab_table(), self.palette.lab_table())

    def test_the_head_declares_its_size(self):
        # The nearest-colour search gives these rows their own shortlist, so a
        # wide tail cannot evict a cheaper two-filament answer.
        self.assertEqual(self.palette.head_size, len(self.head))
        self.assertLess(self.palette.head_size, len(self.palette))

    def test_the_bigger_pool_never_matches_worse(self):
        # A colour no spool owns: the pool is a superset of the head, so the
        # error can only go down.
        image = Image.new("RGBA", (16, 16), (190, 60, 90, 255))
        settings = MatchSettings(max_colours=3, max_dimension=64)
        head_result = match_image(image, self.head, settings)
        pool_result = match_image(image, self.palette, settings)

        def worst(result) -> float:
            return max(
                _color.delta_e_2000(
                    _color.lab_from_rgb(tuple(result.means[index])), entry.lab
                )
                for index, entry in enumerate(result.palette)
            )

        self.assertLessEqual(worst(pool_result), worst(head_result) + 1e-9)


class SpoolBudgetTests(unittest.TestCase):
    """「这个模型最多用几卷耗材」: 0 = 不限, otherwise a hard cap on spools."""

    def setUp(self):
        self.library = _library()
        self.catalog = MixCatalog(self.library.filaments, engine=ENGINE_BAMBU).build()
        self.palette = build_palette(self.library, self.catalog)

    @staticmethod
    def _spools(result) -> set[str]:
        return {
            parent for entry in result.palette for parent in entry.filament_ids
        }

    def test_zero_means_no_limit(self):
        result = match_image(
            _quadrant_image(), self.palette, MatchSettings(max_colours=4, max_spools=0)
        )
        self.assertEqual(len(self._spools(result)), 4)

    def test_the_cap_is_the_number_of_distinct_spools(self):
        result = match_image(
            _quadrant_image(), self.palette, MatchSettings(max_colours=4, max_spools=2)
        )
        self.assertEqual(len(self._spools(result)), 2)
        self.assertEqual(len(result.palette), 2)

    def test_a_budget_of_one_leaves_one_colour(self):
        result = match_image(
            _quadrant_image(), self.palette, MatchSettings(max_colours=4, max_spools=1)
        )
        self.assertEqual(len(self._spools(result)), 1)
        self.assertEqual(len(result.palette), 1)

    def test_the_surviving_colour_is_the_biggest_region(self):
        # Two quadrants are twice the area of the other two, so the budget has
        # to keep one of those and repaint everything with it.
        image = Image.new("RGBA", (40, 40), (0, 0, 0, 255))
        image.paste(Image.new("RGBA", (20, 40), (200, 200, 195, 255)), (20, 0))
        result = match_image(image, self.palette, MatchSettings(max_colours=4, max_spools=1))
        self.assertEqual(len(result.palette), 1)
        self.assertEqual(int(result.counts[0]), 40 * 40)

    def test_a_refused_colour_never_reaches_the_result(self):
        # Both picks want two spools, so with a budget of one they are refused
        # and the promise can only be kept by repainting with a single spool.
        mixes = [entry for entry in self.palette if entry.is_mix][:2]
        self.assertEqual(len(mixes), 2)
        for entry in mixes:
            self.assertEqual(len(entry.filament_ids), 2)
        indices = np.zeros((4, 4), dtype=np.int32)
        indices[:, 2:] = 1
        kept, counts, final = _fit_spool_budget(list(mixes), indices, 1, self.palette)
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(kept[0].filament_ids), 1)
        self.assertNotIn(kept[0].key, {entry.key for entry in mixes})
        self.assertEqual(int(counts.sum()), 16)
        self.assertEqual(sorted(set(int(value) for value in final.reshape(-1))), [0])

    def test_the_budget_keeps_everything_when_it_fits(self):
        mixes = [entry for entry in self.palette if entry.is_mix][:2]
        indices = np.zeros((4, 4), dtype=np.int32)
        indices[:, 2:] = 1
        # Four spools is the whole library, so nothing has to be refused.
        self.assertIsNone(
            _fit_spool_budget(list(mixes), indices, len(self.library.filaments), self.palette)
        )

    def test_no_budget_means_no_work(self):
        mixes = [entry for entry in self.palette if entry.is_mix][:2]
        indices = np.zeros((4, 4), dtype=np.int32)
        self.assertIsNone(_fit_spool_budget(list(mixes), indices, 0, self.palette))

    def test_a_budget_of_one_falls_back_to_a_single_spool(self):
        # Both picks need two spools, so the promise can only be kept by
        # repainting with the closest colour that uses one.
        mixes = [entry for entry in self.palette if entry.is_mix][:2]
        for entry in mixes:
            self.assertEqual(len(entry.filament_ids), 2)
        indices = np.zeros((4, 4), dtype=np.int32)
        indices[:, 2:] = 1
        kept, _counts, _final = _fit_spool_budget(list(mixes), indices, 1, self.palette)
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(kept[0].filament_ids), 1)

    def test_transparent_pixels_stay_transparent(self):
        mixes = [entry for entry in self.palette if entry.is_mix][:2]
        indices = np.array([[0, 1, -1, -1], [0, 1, -1, -1]], dtype=np.int32)
        _kept, _counts, final = _fit_spool_budget(list(mixes), indices, 1, self.palette)
        self.assertEqual(list(final[0]), [0, 0, -1, -1])

    def test_the_setting_is_clamped_like_the_others(self):
        self.assertEqual(MatchSettings(max_spools=-4).clamped().max_spools, 0)
        self.assertEqual(
            MatchSettings(max_spools=10_000).clamped().max_spools, MAX_PAINTED_FILAMENTS
        )


class BudgetOptionTests(unittest.TestCase):
    """「多给用户选项，让用户自己选择用那一组耗材」.

    The program must not pretend one 方案 is right: it offers a few, each with
    its promise, and the user picks.  What is tested here is the promise of the
    list itself — the first entry is what 「重新匹配」 would have used, no entry
    ever spends more than 「耗材上限」 rolls (that cap is the user's instruction,
    not a suggestion the program may overshoot), and no two entries are the same
    方案 twice.
    """

    def setUp(self):
        self.library = _library()
        self.catalog = MixCatalog(self.library.filaments, engine=ENGINE_BAMBU).build()
        self.palette = build_palette(self.library, self.catalog)

    def test_the_first_option_is_what_the_plain_match_gives(self):
        settings = MatchSettings(max_colours=4, max_spools=2)
        plain = match_image(_quadrant_image(), self.palette, settings)
        options = match_options(_quadrant_image(), self.palette, settings)

        self.assertEqual(options[0].key, "closest")
        self.assertEqual(
            [entry.key for entry in options[0].result.palette],
            [entry.key for entry in plain.palette],
        )
        self.assertEqual(
            [int(value) for value in options[0].result.counts],
            [int(value) for value in plain.counts],
        )

    def test_a_picture_that_fits_the_cap_is_offered_once(self):
        # Four spools and a cap of four: nothing has to be refused, so every
        # 方案 prints the same rolls and the extra roll buys nothing.
        options = match_options(
            _quadrant_image(), self.palette, MatchSettings(max_colours=4, max_spools=4)
        )
        self.assertEqual(len(options), 1)
        self.assertEqual(len(options[0].spools), 4)

    def test_an_unlimited_cap_is_offered_once_and_never_asks_for_an_extra_roll(self):
        result = match_image(
            _quadrant_image(), self.palette, MatchSettings(max_colours=4, max_spools=0)
        )
        options = match_options(
            _quadrant_image(), self.palette, MatchSettings(max_colours=4, max_spools=0)
        )
        self.assertEqual([option.key for option in options], ["closest"])
        wanted = {parent for entry in result.palette for parent in entry.filament_ids}
        self.assertEqual(set(options[0].spools), wanted)

    def test_no_option_ever_spends_more_than_the_cap(self):
        # 「耗材方案只能比等于小于耗材上限，不能大于耗材上限」: the cap is the user's
        # instruction about this machine, so it is a hard ceiling for every 方案.
        for cap in (1, 2, 3, 4):
            options = match_options(
                _quadrant_image(), self.palette, MatchSettings(max_colours=4, max_spools=cap)
            )
            self.assertTrue(options, cap)
            for option in options:
                self.assertLessEqual(len(option.spools), cap, (cap, option.key))
                self.assertNotEqual(option.key, "extra")

    def test_every_option_obeys_the_cap_of_the_settings_it_was_given(self):
        settings = MatchSettings(max_colours=4, max_spools=2)
        options = match_options(_quadrant_image(), self.palette, settings)
        for option in options:
            self.assertLessEqual(len(option.spools), settings.max_spools, option.key)

    def test_two_options_never_print_the_same_rolls(self):
        options = match_options(
            _quadrant_image(), self.palette, MatchSettings(max_colours=4, max_spools=3)
        )
        rolls = [set(option.spools) for option in options]
        self.assertEqual(len(rolls), len({frozenset(entry) for entry in rolls}))

    def test_every_option_carries_a_title_a_note_and_its_rolls(self):
        options = match_options(
            _quadrant_image(), self.palette, MatchSettings(max_colours=4, max_spools=2)
        )
        known = {key for key, _title, _note, _objective in _PLAN_SPECS}
        self.assertEqual([option.key for option in options][0], "closest")
        for option in options:
            self.assertIn(option.key, known)
            self.assertTrue(option.title.strip())
            self.assertTrue(option.note.strip())
            self.assertNotIn("…", option.title + option.note)
            self.assertTrue(set(option.spools) <= {filament.id for filament in self.library.filaments})

    def test_the_defect_count_is_the_third_fidelity_number(self):
        options = match_options(
            _quadrant_image(), self.palette, MatchSettings(max_colours=4, max_spools=2)
        )
        average, worst, bad = match_fidelity_report(options[0].result)
        self.assertEqual(match_fidelity(options[0].result), (average, worst))
        self.assertEqual(options[0].defects, bad)
        self.assertGreaterEqual(bad, 0)

    def test_a_perfect_match_scores_zero(self):
        result = match_image(
            _quadrant_image(), self.palette, MatchSettings(max_colours=4, max_spools=0)
        )
        average, worst = match_fidelity(result)
        self.assertLess(average, 0.01)
        self.assertLess(worst, 0.01)
        self.assertEqual(match_fidelity(result), (average, worst))
        self.assertEqual(match_fidelity_report(result)[:2], (average, worst))
        self.assertEqual(match_fidelity_report(result)[2], 0)

    def test_the_average_is_weighted_by_pixels_and_bounded_by_the_worst(self):
        # A 40x40 black field with a small odd patch: the average must land much
        # closer to the exact region than to the patch, and never outside the
        # range of the per-region errors.
        image = Image.new("RGBA", (40, 40), (23, 24, 28, 255))
        image.paste(Image.new("RGBA", (10, 10), (120, 190, 60, 255)), (0, 0))
        result = match_image(image, self.palette, MatchSettings(max_colours=4, max_spools=0))
        average, worst = match_fidelity(result)
        errors = [
            _color.delta_e_2000(
                _color.lab_from_rgb(tuple(result.means[index])), entry.lab
            )
            for index, entry in enumerate(result.palette)
        ]
        self.assertAlmostEqual(worst, max(errors), places=9)
        self.assertGreaterEqual(average, min(errors) - 1e-9)
        self.assertLessEqual(average, max(errors) + 1e-9)
        self.assertLess(average, min(worst, 2.0))

    def test_an_empty_result_has_no_fidelity_to_report(self):
        # Nothing kept: the page greys the panel out rather than printing a
        # meaningless 0.00.
        result = match_image(
            _quadrant_image(), self.palette, MatchSettings(max_colours=4, max_spools=0)
        )
        empty = replace(result, palette=[], counts=np.array([], dtype=np.int64), means=None)
        self.assertEqual(match_fidelity(empty), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()

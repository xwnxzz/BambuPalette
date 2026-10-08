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
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.engines import ENGINE_BAMBU  # noqa: E402
from app.core.image_matching import (  # noqa: E402
    MatchSettings,
    build_palette,
    match_image,
    reduce_colours,
)
from app.core.library import Filament, FilamentLibrary  # noqa: E402
from app.core.mixes import MixCatalog  # noqa: E402

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


class PaletteTests(unittest.TestCase):
    def test_palette_is_every_spool_then_every_mix(self):
        library = _library()
        catalog = MixCatalog(library.filaments, engine=ENGINE_BAMBU).build()
        palette = build_palette(library, catalog)
        # 4 spools + C(4,2) pairs × 81 ratios
        self.assertEqual(len(palette), 4 + 6 * 81)
        self.assertEqual([entry.kind for entry in palette[:4]], ["filament"] * 4)
        self.assertTrue(all(entry.kind == "mix" for entry in palette[4:]))
        self.assertTrue(all(len(entry.filament_ids) == 2 for entry in palette[4:]))
        self.assertTrue(all(entry.filament_ids[0] != entry.filament_ids[1] for entry in palette[4:]))

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


if __name__ == "__main__":
    unittest.main()

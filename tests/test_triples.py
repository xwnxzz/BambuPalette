"""三色混色：2556 个配比、按颜色去重、惰性视图和缓存。"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.core.engines import ENGINE_BAMBU, ENGINE_MIXER
from app.core.library import Filament, FilamentLibrary
from app.core.triples import (
    TRIPLE_CACHE_KEEP,
    TRIPLE_CACHE_PREFIX,
    TRIPLE_RATIO_COUNT,
    OrderedTripleColours,
    TripleCatalog,
    TripleColours,
    load_triple_cache,
    mix_sidecar_path,
    prune_triple_caches,
    save_triple_cache,
    triple_cache_key,
    triple_pair_count,
    triple_ratios,
    triple_recipe_count,
)


def _library(*colours: str) -> FilamentLibrary:
    library = FilamentLibrary()
    for index, color_hex in enumerate(colours):
        library.add(
            Filament(
                id=f"f{index}",
                brand="测试",
                material_type="PLA",
                color_hex=color_hex,
            )
        )
    return library


SIX = ("#F7F3F3", "#2C2A2C", "#A4A3A3", "#60B2DE", "#FF6C2B", "#047EBA")


class TripleRatioTests(unittest.TestCase):
    def test_the_grid_is_the_2556_the_user_worked_out(self):
        ratios = triple_ratios()
        self.assertEqual(ratios.shape, (TRIPLE_RATIO_COUNT, 3))
        self.assertEqual(TRIPLE_RATIO_COUNT, 2556)
        self.assertTrue(np.all(ratios.sum(axis=1) == 100))
        self.assertEqual(int(ratios.min()), 10)
        self.assertEqual(int(ratios.max()), 80)
        self.assertEqual(len({tuple(row) for row in ratios.tolist()}), 2556)
        # 交换用量算不同配方，所以 30/40/30 和 40/30/30 都在里面。
        rows = {tuple(row) for row in ratios.tolist()}
        self.assertIn((30, 40, 30), rows)
        self.assertIn((40, 30, 30), rows)

    def test_a_higher_floor_shrinks_the_grid(self):
        ratios = triple_ratios(20)
        self.assertTrue(np.all(ratios.min(axis=1) >= 20))

    def test_the_counts_are_the_binomial_ones(self):
        self.assertEqual(triple_pair_count(3), 1)
        self.assertEqual(triple_pair_count(4), 4)
        self.assertEqual(triple_pair_count(41), 10660)
        self.assertEqual(triple_recipe_count(3), 2556)
        self.assertEqual(triple_recipe_count(41), 10660 * 2556)


class TripleCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = _library(*SIX)
        cls.catalog = TripleCatalog(cls.library.filaments, engine=ENGINE_MIXER).build()

    def test_the_table_holds_every_recipe_exactly_once(self):
        catalog = self.catalog
        self.assertTrue(catalog.built)
        self.assertEqual(catalog.triple_count, 20)
        self.assertEqual(catalog.recipe_count, 20 * 2556)
        total = sum(catalog.recipe_count_at(index) for index in range(catalog.colour_count))
        self.assertEqual(total, catalog.recipe_count)
        self.assertLess(catalog.colour_count, catalog.recipe_count)

    def test_the_packed_colours_are_sorted_and_the_offsets_line_up(self):
        catalog = self.catalog
        self.assertTrue(np.all(np.diff(catalog.packed) > 0), "colours are unique")
        self.assertEqual(int(catalog._offsets[0]), 0)
        self.assertEqual(int(catalog._offsets[-1]), catalog.recipe_count)
        self.assertTrue(np.all(np.diff(catalog._offsets) >= 1), "every colour has a recipe")

    def test_every_recipe_really_makes_its_colour(self):
        catalog = self.catalog
        for index in (0, 1, catalog.colour_count // 2, catalog.colour_count - 1):
            hex_value = catalog.color_hex_at(index)
            recipes = catalog.recipes_at(index)
            self.assertTrue(recipes)
            self.assertEqual(len(recipes), catalog.recipe_count_at(index))
            for recipe in recipes:
                self.assertEqual(recipe.color_hex, hex_value)
                self.assertEqual(sum(recipe.percents), 100)
                self.assertGreaterEqual(min(recipe.percents), 10)
                self.assertEqual(len(recipe.parent_ids), 3)
                self.assertEqual(len(set(recipe.parent_ids)), 3)

    def test_an_exact_colour_is_its_own_recipe(self):
        # 100 % of one spool is outside the 10 % floor, so the darkest colour has
        # to be a genuine blend, never a raw spool.
        catalog = self.catalog
        self.assertEqual(catalog.recipe_count_at(0), len(catalog.recipes_at(0)))
        self.assertEqual(catalog.rgb_at(0), catalog.colour_at(0).rgb)

    def test_colours_is_a_lazy_sequence(self):
        view = self.catalog.colours()
        self.assertIsInstance(view, TripleColours)
        self.assertEqual(len(view), self.catalog.colour_count)
        self.assertEqual(view[0].key, self.catalog.colour_at(0).key)
        self.assertEqual(view[-1].key, self.catalog.colour_at(len(view) - 1).key)
        self.assertEqual([c.key for c in view[2:5]], [view[i].key for i in (2, 3, 4)])
        self.assertEqual(view.index_of_key(view[7].key), 7)
        self.assertEqual(view.index_of_key("triple|#123456"), -1)
        with self.assertRaises(IndexError):
            view[len(view)]

    def test_ordered_colours_walks_the_order(self):
        ordered = self.catalog.ordered_colours("rgb")
        self.assertIsInstance(ordered, OrderedTripleColours)
        self.assertEqual(len(ordered), self.catalog.colour_count)
        packed = [
            (colour.rgb[0] << 16) | (colour.rgb[1] << 8) | colour.rgb[2]
            for colour in ordered[:40]
        ]
        self.assertEqual(packed, sorted(packed))
        target = ordered[11]
        self.assertEqual(ordered.index_of_key(target.key), 11)
        self.assertEqual(ordered.index_of_key("triple|#123456"), -1)

    def test_lightness_order_is_descending(self):
        ordered = self.catalog.ordered_colours("lightness")
        values = [colour.lightness for colour in ordered[:30]]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_nearest_finds_a_real_colour(self):
        found = self.catalog.nearest("#047EBA")
        self.assertIsNotNone(found)
        self.assertIn(found.color_hex, {c.color_hex for c in self.catalog.colours()})

    def test_build_in_steps_matches_a_full_build(self):
        stepped = TripleCatalog(self.library.filaments, engine=ENGINE_MIXER)
        stepped.begin()
        guard = 0
        while not stepped.step(0.01):
            guard += 1
            self.assertLess(guard, 100_000)
        stepped.finish()
        fresh = TripleCatalog(self.library.filaments, engine=ENGINE_MIXER).build()
        self.assertTrue(np.array_equal(stepped.packed, fresh.packed))
        self.assertTrue(np.array_equal(stepped._codes, fresh._codes))


class TripleCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.library = _library(*SIX[:4])
        self.catalog = TripleCatalog(self.library.filaments, engine=ENGINE_MIXER).build()

    def test_the_key_follows_the_colours_and_the_engine(self):
        base = triple_cache_key(self.library.filaments, ENGINE_MIXER)
        self.assertEqual(base, triple_cache_key(self.library.filaments, ENGINE_MIXER))
        self.assertNotEqual(base, triple_cache_key(self.library.filaments, ENGINE_BAMBU))
        other = _library(*SIX[:4])
        other.filaments[0].color_hex = "#000001"
        self.assertNotEqual(
            base, triple_cache_key(other.filaments, ENGINE_MIXER)
        )

    def test_a_missing_or_broken_cache_is_just_none(self):
        path = Path(self.tmp.name) / "nothing.npz"
        self.assertIsNone(load_triple_cache(path, self.library.filaments, ENGINE_MIXER))
        path.write_bytes(b"not an npz at all")
        self.assertIsNone(load_triple_cache(path, self.library.filaments, ENGINE_MIXER))
    def test_save_and_load_round_trips_the_colours(self):
        path = Path(self.tmp.name) / "cache" / "triples.npz"
        save_triple_cache(self.catalog, path)
        self.assertTrue(path.is_file())
        loaded = load_triple_cache(path, self.library.filaments, ENGINE_MIXER)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.colour_count, self.catalog.colour_count)
        self.assertEqual(loaded.recipe_count, self.catalog.recipe_count)
        self.assertTrue(np.array_equal(loaded.packed, self.catalog.packed))
        last = loaded.colour_count - 1
        self.assertEqual(loaded.recipes_at(last)[0].key, self.catalog.recipes_at(last)[0].key)

    def test_the_cache_records_which_library_it_belongs_to(self):
        # 导出档案旁边的那份缓存只有带着键，导入时才敢认领。
        path = Path(self.tmp.name) / "triples.npz"
        save_triple_cache(self.catalog, path)
        with np.load(path) as data:
            self.assertIn("key", data.files)
            self.assertEqual(
                str(data["key"].item()),
                triple_cache_key(self.library.filaments, ENGINE_MIXER),
            )

    def test_a_cache_from_another_library_is_refused(self):
        # 指纹对不上就当作没有缓存——颜色绝不会张冠李戴。
        path = Path(self.tmp.name) / "cache" / "triples.npz"
        save_triple_cache(self.catalog, path)
        self.assertIsNotNone(
            load_triple_cache(path, self.library.filaments, ENGINE_MIXER)
        )
        other = _library(*SIX[:4])
        other.filaments[0].color_hex = "#010203"
        self.assertIsNone(load_triple_cache(path, other.filaments, ENGINE_MIXER))
        self.assertIsNone(load_triple_cache(path, self.library.filaments, ENGINE_BAMBU))

    def test_old_caches_are_pruned_so_the_disk_does_not_fill_up(self):
        folder = Path(self.tmp.name) / "cache"
        folder.mkdir()
        for index in range(5):
            item = folder / f"{TRIPLE_CACHE_PREFIX}aaaaaaa{index}.npz"
            item.write_bytes(b"x")
            os.utime(item, (1_700_000_000 + index, 1_700_000_000 + index))
        (folder / f"{TRIPLE_CACHE_PREFIX}bbbbbbbb.npz.tmp").write_bytes(b"x")
        (folder / "unrelated.npz").write_bytes(b"x")

        removed = prune_triple_caches(folder, keep=2)

        left = sorted(item.name for item in folder.glob("*.npz"))
        self.assertEqual(
            left,
            [
                f"{TRIPLE_CACHE_PREFIX}aaaaaaa3.npz",
                f"{TRIPLE_CACHE_PREFIX}aaaaaaa4.npz",
                "unrelated.npz",
            ],
        )
        self.assertIn(f"{TRIPLE_CACHE_PREFIX}aaaaaaa0.npz", removed)
        self.assertEqual(list(folder.glob("*.tmp")), [])

    def test_saving_a_cache_prunes_the_siblings(self):
        folder = Path(self.tmp.name) / "cache"
        folder.mkdir()
        for index in range(5):
            (folder / f"{TRIPLE_CACHE_PREFIX}stale{index}.npz").write_bytes(b"x")
        target = folder / f"{TRIPLE_CACHE_PREFIX}live.npz"
        save_triple_cache(self.catalog, target)
        names = sorted(item.name for item in folder.glob("*.npz"))
        self.assertIn(target.name, names)
        self.assertLessEqual(len(names), TRIPLE_CACHE_KEEP)
        # 导出用的 sidecar 是另一个前缀，不该顺手删掉别人的缓存。
        sidecar_folder = Path(self.tmp.name) / "export"
        sidecar_folder.mkdir()
        for index in range(4):
            (sidecar_folder / f"{TRIPLE_CACHE_PREFIX}keep{index}.npz").write_bytes(b"x")
        save_triple_cache(self.catalog, sidecar_folder / "档案.mixes.npz")
        self.assertEqual(len(list(sidecar_folder.glob("*.npz"))), 5)


class SidecarPathTests(unittest.TestCase):
    def test_the_sidecar_sits_beside_the_archive(self):
        self.assertEqual(
            mix_sidecar_path(Path("C:/x/大简-PETG-HF-耗材档案.json")).name,
            "大简-PETG-HF-耗材档案.mixes.npz",
        )
        self.assertEqual(mix_sidecar_path("lib.JSON").name, "lib.mixes.npz")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

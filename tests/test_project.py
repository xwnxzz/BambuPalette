"""Project-file tests: a saved ``.bpproj`` must come back exactly as it left.

The file is the only thing standing between 「改了一下午」 and starting over, so
these tests are written the way the user would check it: save a match with
edits, read it back, and compare the picture, the colours, the recipes and the
settings — not just that the zip opens.

Run with::

    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
import zipfile
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.engines import ENGINE_BAMBU  # noqa: E402
from app.core.image_matching import (  # noqa: E402
    MatchResult,
    MatchSettings,
    PaletteEntry,
    build_palette,
    spool_usage,
)
from app.core.library import Filament, FilamentLibrary  # noqa: E402
from app.core.mixes import MixCatalog  # noqa: E402
from app.core.project import (  # noqa: E402
    PROJECT_SUFFIX,
    ProjectError,
    _live_palette,
    colour_list,
    entry_from_json,
    entry_to_json,
    load_project,
    render_preview,
    save_project,
    spool_list,
)
from app.mesh.plate import PlateSettings, smooth_factor  # noqa: E402
from app.spectral import color as _color  # noqa: E402

SPOOLS = (
    ("工程白", "#F2F0EB"),
    ("工程黑", "#17181C"),
    ("工程金", "#D9A441"),
    ("工程红", "#C8342E"),
)


def _library() -> FilamentLibrary:
    return FilamentLibrary(
        [
            Filament(name=name, brand="测试", material_type="PETG HF", color_hex=value)
            for name, value in SPOOLS
        ]
    )


def _palette(library: FilamentLibrary) -> list[PaletteEntry]:
    catalog = MixCatalog(library.filaments, engine=ENGINE_BAMBU).build()
    return build_palette(library, catalog, include_mixes=True)


def _base_match(library: FilamentLibrary, picks: int = 3) -> MatchResult:
    """A small hand-built match: ``picks`` colours over a 4×6 grid.

    Built by hand rather than by :func:`match_image` so the expected pixel
    counts are stated here instead of being whatever the matcher decided — a
    project round-trip test has to know the answer it is checking.
    """
    palette = _palette(library)
    chosen = palette[: picks - 1] + [entry for entry in palette if entry.is_mix][:1]
    indices = np.full((6, 4), -1, dtype=np.int32)
    indices[0:3, 1:3] = 0
    indices[3:5, 1:3] = 1
    indices[1:5, 3] = 2
    counts = np.asarray(
        [int(np.count_nonzero(indices == index)) for index in range(len(chosen))],
        dtype=np.int64,
    )
    means = np.asarray([entry.rgb for entry in chosen], dtype=np.uint8)
    return MatchResult(
        width=4,
        height=6,
        palette=chosen,
        indices=indices,
        counts=counts,
        settings=MatchSettings(
            max_colours=len(chosen), max_dimension=64, dither=False, max_spools=2
        ),
        source="测试来源",
        means=means,
    )


def _picture(size: int = 8) -> Image.Image:
    image = Image.new("RGBA", (size, 6), (200, 200, 200, 255))
    for x in range(size):
        image.putpixel((x, 0), (0, 0, 0, 0))
    return image


class ProjectRoundTripTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = Path(self.enterContext(_temp_dir()))

    def test_round_trip_keeps_the_picture_colours_and_settings(self):
        library = _library()
        base = _base_match(library)
        path = self._dir / "工程"
        written = save_project(
            path,
            image=_picture(),
            image_name="测试图",
            base=base,
            library=library,
            plate=PlateSettings(
                cell_mm=2.0, base_thickness_mm=1.0, base_index=1
            ),
            version="9.9.9",
        )
        self.assertTrue(written.name.endswith(PROJECT_SUFFIX), written)

        state = load_project(written, library=library)
        self.assertEqual(state.image_name, "测试图")
        self.assertEqual((state.image.width, state.image.height), (8, 6))
        self.assertEqual(state.image.getpixel((0, 0))[3], 0)
        self.assertEqual(state.version, "9.9.9")
        self.assertTrue(state.saved_at)
        self.assertEqual(
            [entry.color_hex for entry in state.base.palette],
            [entry.color_hex for entry in base.palette],
        )
        self.assertEqual([entry.kind for entry in state.base.palette],
                         [entry.kind for entry in base.palette])
        self.assertTrue(np.array_equal(state.base.indices, base.indices))
        self.assertTrue(np.array_equal(state.base.counts, base.counts))
        self.assertTrue(np.array_equal(state.base.means, base.means))
        self.assertEqual(state.base.source, "测试来源")
        self.assertEqual(state.base.settings.max_colours, base.settings.max_colours)
        self.assertEqual(state.base.settings.max_spools, 2)
        self.assertEqual(state.base.settings.max_dimension, 64)
        self.assertFalse(state.base.settings.dither)
        self.assertEqual(state.plate.cell_mm, 2.0)
        self.assertEqual(state.plate.base_thickness_mm, 1.0)
        # The base colour was saved as a hex, because a palette position means
        # nothing after the user deletes a colour or swaps one out.
        self.assertEqual(state.plate.base_index, 1)

    def test_the_spool_recipe_that_was_picked_is_remembered(self):
        # 「耗材方案」:工程文件要记住当初挑的是哪一套,不然「打开工程」只能干说一句
        # 「工程里存的那一套」。
        library = _library()
        written = save_project(
            self._dir / "plan",
            image=_picture(),
            image_name="方案",
            base=_base_match(library),
            library=library,
            plan="distinct",
        )
        self.assertEqual(load_project(written, library=library).plan, "distinct")

        # 1.6.11 及更早的工程没有 plan 这个键:读成空字符串,而不是拒绝整个文件。
        older = self._dir / "older.bpproj"
        with zipfile.ZipFile(written) as source, zipfile.ZipFile(
            older, "w", zipfile.ZIP_DEFLATED
        ) as target:
            for name in source.namelist():
                payload = source.read(name)
                if name == "工程.json":
                    document = json.loads(payload.decode("utf-8"))
                    document.pop("plan")
                    payload = json.dumps(document, ensure_ascii=False).encode("utf-8")
                target.writestr(name, payload)
        self.assertEqual(load_project(older, library=library).plan, "")

    def test_an_older_project_keeps_its_total_thickness(self):
        """1.6.41 及以前的工程：那是「底板色 + 颜色层」两个数，整块是两者之和。

        1.6.42（m15721 第 3 条）起 ``base_thickness_mm`` 是**整块**的厚度，读盘时按和
        换算，所以升级前后打出来的厚度一模一样。 那时候记的「打印宽度」不再认：板长宽
        = 图片像素数 × 一格（0.4 mm），所以文件里那些键丢掉、cell_mm 用默认值。
        """
        library = _library()
        written = save_project(
            self._dir / "oldplate",
            image=_picture(),
            image_name="旧板",
            base=_base_match(library),
            library=library,
            plate=PlateSettings(smooth_edges=False, image_on_bottom=False),
        )
        older = self._dir / "oldplate-1.6.41.bpproj"
        with zipfile.ZipFile(written) as source, zipfile.ZipFile(
            older, "w", zipfile.ZIP_DEFLATED
        ) as target:
            for name in source.namelist():
                payload = source.read(name)
                if name == "工程.json":
                    document = json.loads(payload.decode("utf-8"))
                    document["plate"] = {
                        "target_width_mm": 120.0,
                        "base_thickness_mm": 0.8,
                        "colour_thickness_mm": 0.6,
                        "max_footprint_mm": 250.0,
                        "min_pixel_mm": 0.35,
                        "smooth_edges": False,
                        "image_on_bottom": False,
                    }
                    payload = json.dumps(document, ensure_ascii=False).encode("utf-8")
                target.writestr(name, payload)
        plate = load_project(older, library=library).plate
        self.assertAlmostEqual(plate.base_thickness_mm, 1.4)
        self.assertAlmostEqual(plate.colour_thickness_mm, 0.4)
        self.assertAlmostEqual(plate.cell_mm, 0.4)
        self.assertFalse(plate.smooth_edges)
        self.assertFalse(plate.image_on_bottom)

    def test_the_edited_picture_and_its_badge_survive_the_round_trip(self):
        # 1.6.19 (m01566 的答复): 删除/合并 改的是原图本身，文件里存的就是改过的那张；
        # original_edited 只回答「标题旁边要不要写『原图已修改』」。
        library = _library()
        picture = _picture()
        picture.putpixel((3, 3), (10, 20, 30, 255))
        written = save_project(
            self._dir / "edited",
            image=picture,
            image_name="改过的图",
            base=_base_match(library),
            library=library,
            original_edited=True,
        )
        state = load_project(written, library=library)
        self.assertTrue(state.original_edited)
        self.assertEqual(state.image.getpixel((3, 3))[:3], (10, 20, 30))
        self.assertEqual(state.image.getpixel((0, 0))[3], 0)

        plain = save_project(
            self._dir / "plain",
            image=_picture(),
            image_name="没改过",
            base=_base_match(library),
            library=library,
        )
        self.assertFalse(load_project(plain, library=library).original_edited)

        # 1.6.18 及更早的工程没有这个键：读成「没改过」，而不是拒绝整个文件。
        older = self._dir / "older-picture.bpproj"
        with zipfile.ZipFile(written) as source, zipfile.ZipFile(
            older, "w", zipfile.ZIP_DEFLATED
        ) as target:
            for name in source.namelist():
                payload = source.read(name)
                if name == "工程.json":
                    document = json.loads(payload.decode("utf-8"))
                    document["image"].pop("original_edited")
                    payload = json.dumps(document, ensure_ascii=False).encode("utf-8")
                target.writestr(name, payload)
        self.assertFalse(load_project(older, library=library).original_edited)

    def test_mixed_recipes_come_back_with_their_percentages(self):
        library = _library()
        base = _base_match(library)
        written = save_project(
            self._dir / "recipe", image=_picture(), image_name="配方", base=base, library=library
        )
        state = load_project(written, library=library)
        mixed = [entry for entry in state.base.palette if entry.is_mix]
        self.assertEqual(len(mixed), 1)
        original = [entry for entry in base.palette if entry.is_mix][0]
        self.assertEqual(mixed[0].color_hex, original.color_hex)
        self.assertEqual(mixed[0].parts, ())
        self.assertIsNotNone(mixed[0].recipe)
        self.assertEqual(mixed[0].recipe.a_id, original.recipe.a_id)
        self.assertEqual(mixed[0].recipe.b_id, original.recipe.b_id)
        self.assertEqual(mixed[0].recipe.percent_a + mixed[0].recipe.percent_b, 100)
        self.assertEqual(mixed[0].filament_ids, original.filament_ids)
        # The filenames, not just the ids, have to survive: the list shows them.
        self.assertIn(library.get(original.filament_ids[0]).display_name, mixed[0].label)

    def test_edits_survive_and_do_not_depend_on_this_machine(self):
        library = _library()
        base = _base_match(library)
        deleted = {1}
        merged = {2: 0}
        replacement = _palette(library)[3]
        replaced = {0: replacement}
        # What the page hands over as ``live``: merging 2 into 0 makes 2 a
        # printed-with-0 colour, so 2 stops being a colour of its own (``_compose``
        # drops every merge SOURCE from the list) and index 1 is transparent.
        live_palette = [replacement]
        indices = np.full((6, 4), -1, dtype=np.int32)
        indices[0:3, 1:3] = 0
        indices[1:5, 3] = 0
        counts = np.asarray(
            [int(np.count_nonzero(indices == index)) for index in range(1)], dtype=np.int64
        )
        live = replace(base, palette=live_palette, indices=indices, counts=counts)
        written = save_project(
            self._dir / "edits",
            image=_picture(),
            image_name="改过",
            base=base,
            live=live,
            deleted=deleted,
            merged=merged,
            replaced=replaced,
            library=library,
        )
        # The same file read with NO library: the colours and percentages are in
        # the document, so a machine with an empty library still shows the edit.
        state = load_project(written, library=None)
        self.assertEqual(state.deleted, deleted)
        self.assertEqual(state.merged, merged)
        self.assertEqual(set(state.replaced), {0})
        self.assertEqual(state.replaced[0].color_hex, replacement.color_hex)
        self.assertIsNone(state.replaced[0].filament)
        # …and recomposing by the page's rule gives back the live picture.
        rebuilt = _live_palette(
            state.base,
            {"deleted": sorted(state.deleted), "merged": {str(k): v for k, v in state.merged.items()}},
            state.replaced,
        )
        self.assertEqual([entry.color_hex for entry in rebuilt], [e.color_hex for e in live_palette])
        self.assertEqual(len(rebuilt), len(live.palette))

    def test_preview_and_lists_describe_the_live_picture(self):
        library = _library()
        base = _base_match(library)
        preview = render_preview(base)
        self.assertEqual(preview.size, (4, 6))
        self.assertEqual(preview.getpixel((0, 0))[3], 0)
        self.assertEqual(preview.getpixel((1, 0))[:3], base.palette[0].rgb)

        rows = colour_list(base)
        self.assertEqual(len(rows), len(base.palette))        # The shares are rounded to six decimals to keep the JSON readable, so
        # three of them cannot be expected to add up to exactly 1.0.
        self.assertAlmostEqual(sum(row["share_of_pixels"] for row in rows), 1.0, delta=1e-4)
        self.assertEqual(rows[0]["hex"], base.palette[0].color_hex)
        self.assertEqual(rows[0]["pixels"], int(base.counts[0]))
        self.assertEqual(rows[0]["kind"], base.palette[0].kind)

        spools = spool_list(base, library)
        self.assertTrue(spools)
        self.assertAlmostEqual(sum(row["share"] for row in spools), 1.0, delta=1e-4)
        self.assertTrue(all(row["in_library"] for row in spools))
        self.assertTrue(all(row["name"] for row in spools))
        shares = {row["id"]: row["share"] for row in spools}
        self.assertEqual(list(shares), [row["id"] for row in spools])
        # Material shares, not pixel shares: the mixed colour spends its parents'
        # material in the recipe's ratio, so a parent of a mix is used for less
        # than the mix's whole area but for more than nothing.
        mix = [entry for entry in base.palette if entry.is_mix][0]
        for parent, percent in (
            (mix.recipe.a_id, mix.recipe.percent_a),
            (mix.recipe.b_id, mix.recipe.percent_b),
        ):
            self.assertGreater(shares[parent], 0.0)
            self.assertGreaterEqual(percent, 0)

    def test_the_smoothed_preview_is_the_grid_the_plate_is_cut_from(self):
        """圆滑色块 also decides what 预览.png looks like (m05557 item 3).

        The panel and the saved preview used to show the raw match grid — the
        one-cell staircase — while the plate was cut from the refined grid, so
        the only place the box did anything was inside Bambu Studio.
        """
        library = _library()
        base = _base_match(library)
        raw = render_preview(base)
        smooth = render_preview(base, smooth=True)
        steps = smooth_factor(*base.indices.shape, colours=len(base.palette))
        self.assertEqual(smooth.size, (raw.width * steps, raw.height * steps))
        # Same picture, finer lattice: the colours are the ones the match chose,
        # every pixel is one of them, and the alpha the raw grid had is kept.
        allowed = {(*entry.rgb, 255) for entry in base.palette}
        allowed.add((0, 0, 0, 0))
        pixels = np.asarray(smooth.convert("RGBA")).reshape(-1, 4)
        self.assertTrue({tuple(int(v) for v in row) for row in pixels} <= allowed)
        self.assertEqual(
            smooth.getpixel((0, 0))[3],
            0 if raw.getpixel((0, 0))[3] == 0 else 255,
        )

    def test_spools_this_machine_no_longer_has_are_marked_not_invented(self):
        library = _library()
        base = _base_match(library)
        written = save_project(
            self._dir / "gone", image=_picture(), image_name="缺料", base=base, library=library
        )
        state = load_project(written, library=None)
        # The file remembers what these spools were called when it was saved —
        # that is the list the user reads when they ask what to load.
        self.assertTrue(state.spools)
        self.assertTrue(all(row["name"] for row in state.spools))
        self.assertTrue(all(row["color_hex"] for row in state.spools))
        # Recomputing the list on a machine without those spools must say so
        # rather than invent a colour for an id this library never heard of.
        rows = spool_list(state.base, None)
        self.assertTrue(rows)
        self.assertTrue(all(row["in_library"] is False for row in rows))
        self.assertTrue(all(row["id"] for row in rows))
        self.assertTrue(all(row["color_hex"] == "" for row in rows))
        uses = spool_usage(state.base, None)
        self.assertTrue(all(use.filament is None for use in uses))
        self.assertTrue(all(use.display_name for use in uses))

    def test_a_three_parent_colour_survives_the_round_trip(self):
        entry = PaletteEntry(
            key="colour:#8D545F",
            color_hex="#8D545F",
            rgb=(141, 84, 95),
            lab=_color.lab_from_rgb((141, 84, 95)),
            label="白 27% + 黑 36% + 红 37%",
            kind="mix",
            filament_ids=("a", "b", "c"),
            parts=(("a", 27), ("b", 36), ("c", 37)),
        )
        back = entry_from_json(entry_to_json(entry))
        self.assertEqual(back.kind, "mix")
        self.assertEqual(back.color_hex, "#8D545F")
        self.assertEqual(back.rgb, (141, 84, 95))
        self.assertEqual(back.parts, (("a", 27), ("b", 36), ("c", 37)))
        self.assertEqual(sum(percent for _parent, percent in back.parts), 100)
        self.assertEqual(back.key, "colour:#8D545F")
        self.assertIsNone(back.recipe)
        self.assertEqual(back.ratio_text, "27% + 36% + 37%")

    def test_a_spool_entry_keeps_its_identity_without_a_library(self):
        entry = PaletteEntry(
            key="filament:xyz",
            color_hex="#123456",
            rgb=(0x12, 0x34, 0x56),
            lab=_color.lab_from_rgb((0x12, 0x34, 0x56)),
            label="某卷",
            kind="filament",
            filament_ids=("xyz",),
        )
        back = entry_from_json(entry_to_json(entry), library=None)
        self.assertEqual(back.key, "filament:xyz")
        self.assertEqual(back.filament_ids, ("xyz",))
        self.assertIsNone(back.filament)
        self.assertTrue(np.allclose(back.lab, entry.lab))

    def test_the_suffix_is_forced_so_the_dialog_cannot_make_a_readable_file_we_cannot_open(self):
        library = _library()
        base = _base_match(library)
        written = save_project(
            self._dir / "没有扩展名.txt",
            image=_picture(),
            image_name="扩展名",
            base=base,
            library=library,
        )
        self.assertEqual(written.suffix, PROJECT_SUFFIX)
        self.assertEqual(load_project(written).image_name, "扩展名")

    def test_the_zip_holds_the_picture_the_preview_and_the_document(self):
        library = _library()
        base = _base_match(library)
        written = save_project(
            self._dir / "members",
            image=_picture(),
            image_name="成员",
            base=base,
            library=library,
        )
        with zipfile.ZipFile(written) as archive:
            names = set(archive.namelist())
            self.assertEqual(names, {"工程.json", "原图.png", "预览.png", "基线.npy"})
            document = json.loads(archive.read("工程.json").decode("utf-8"))
        self.assertEqual(document["app"], "BambuPalette")
        self.assertEqual(document["format"], 1)
        self.assertEqual(document["image"]["file"], "原图.png")
        self.assertEqual(document["colours"], colour_list(base))
        for row in document["spools"]:
            self.assertTrue(row["color_hex"])
            self.assertIn("share", row)


class BlockPaintTests(unittest.TestCase):
    """色块级改动（1.6.17）的存取：删除留空、换成某个颜色，撤销要能一步步倒回去。"""

    def setUp(self) -> None:
        self._dir = Path(self.enterContext(_temp_dir()))

    def _masks(self, base: MatchResult) -> tuple[np.ndarray, np.ndarray]:
        first = np.zeros(base.indices.shape, dtype=bool)
        first[0:3, 1:3] = True
        second = np.zeros(base.indices.shape, dtype=bool)
        second[1:5, 3] = True
        return first, second

    def test_block_edits_survive_the_round_trip(self):
        library = _library()
        base = _base_match(library)
        first, second = self._masks(base)
        replacement = _palette(library)[3]
        written = save_project(
            self._dir / "blocks",
            image=_picture(),
            image_name="色块",
            base=base,
            library=library,
            block_paints=[(first, replacement), (second, None)],
        )
        with zipfile.ZipFile(written) as archive:
            names = set(archive.namelist())
        # 一张图层就装得下所有色块改动；没有改动时这个成员不出现。
        self.assertIn("色块图层.npy", names)

        state = load_project(written, library=None)
        self.assertEqual(len(state.block_paints), 2)
        # 第二块是「删除留空」，读回来还是空的——不能被当成「没改过」。
        self.assertIsNone(state.block_paints[1]["entry"])
        self.assertTrue(np.array_equal(state.block_paints[0]["mask"], first))
        self.assertTrue(np.array_equal(state.block_paints[1]["mask"], second))
        self.assertEqual(state.block_paints[0]["entry"].color_hex, replacement.color_hex)
        self.assertEqual(state.block_paints[0]["entry"].key, replacement.key)

    def test_the_last_edit_of_a_pixel_is_the_one_that_sticks(self):
        # 两个遮罩叠在一起时，后面的那一步说了算：文件里一张图层，读回来也该是
        # 「后来改的盖住先前改的」——不然撤销一次会看到颜色跳错地方。
        library = _library()
        base = _base_match(library)
        wide = np.zeros(base.indices.shape, dtype=bool)
        wide[0:4, 1:3] = True
        narrow = np.zeros(base.indices.shape, dtype=bool)
        narrow[0:2, 1:2] = True
        replacement = _palette(library)[3]
        written = save_project(
            self._dir / "overlap",
            image=_picture(),
            image_name="重叠",
            base=base,
            library=library,
            block_paints=[(wide, replacement), (narrow, None)],
        )
        state = load_project(written, library=None)
        self.assertEqual(len(state.block_paints), 2)
        self.assertTrue(np.array_equal(state.block_paints[1]["mask"], narrow))
        self.assertTrue(
            np.array_equal(state.block_paints[0]["mask"], wide & ~narrow),
            "先前那一步的遮罩至少该让出后来被改过的像素",
        )
        overlap = wide & narrow
        self.assertFalse(np.any(state.block_paints[0]["mask"] & overlap))

    def test_a_block_colour_joins_the_live_palette(self):
        # 色块级编辑可以带进来一个原本不在色表里的颜色；打开工程时它要排在后面，
        # 和页面的 _compose 一个顺序，不然 base_hex 会指错色。
        library = _library()
        base = _base_match(library)
        first, _second = self._masks(base)
        replacement = _palette(library)[3]
        self.assertNotIn(replacement.key, [entry.key for entry in base.palette])
        written = save_project(
            self._dir / "newcolour",
            image=_picture(),
            image_name="新颜色",
            base=base,
            library=library,
            block_paints=[(first, replacement)],
        )
        state = load_project(written, library=None)
        rebuilt = _live_palette(state.base, {"deleted": [], "merged": {}}, {}, state.block_paints)
        self.assertEqual(
            [entry.color_hex for entry in rebuilt],
            [entry.color_hex for entry in base.palette] + [replacement.color_hex],
        )

    def test_a_project_without_block_edits_still_opens(self):
        # 1.6.16 及更早的工程没有这个键、也没有图层成员：读成「没有色块改动」。
        library = _library()
        base = _base_match(library)
        written = save_project(
            self._dir / "plain",
            image=_picture(),
            image_name="旧工程",
            base=base,
            library=library,
        )
        with zipfile.ZipFile(written) as archive:
            self.assertNotIn("色块图层.npy", archive.namelist())
        self.assertEqual(load_project(written, library=library).block_paints, [])

    def test_a_project_whose_block_layer_is_missing_is_refused(self):
        # 文档说有改动、图层却不在：宁可说清楚，也不能装作没改过——那是用户一下午的活。
        library = _library()
        base = _base_match(library)
        first, _second = self._masks(base)
        written = save_project(
            self._dir / "lost",
            image=_picture(),
            image_name="缺图层",
            base=base,
            library=library,
            block_paints=[(first, _palette(library)[3])],
        )
        stripped = self._dir / "lost2.bpproj"
        with zipfile.ZipFile(written) as source, zipfile.ZipFile(
            stripped, "w", zipfile.ZIP_DEFLATED
        ) as target:
            for name in source.namelist():
                if name != "色块图层.npy":
                    target.writestr(name, source.read(name))
        with self.assertRaises(ProjectError) as caught:
            load_project(stripped)
        self.assertIn("色块级改动", str(caught.exception))

    def test_a_block_layer_of_the_wrong_size_is_refused(self):
        library = _library()
        base = _base_match(library)
        first, _second = self._masks(base)
        written = save_project(
            self._dir / "size",
            image=_picture(),
            image_name="尺寸不对",
            base=base,
            library=library,
            block_paints=[(first, _palette(library)[3])],
        )
        broken = self._dir / "size2.bpproj"
        import io

        buffer = io.BytesIO()
        np.save(buffer, np.zeros((3, 3), dtype=np.int32))
        with zipfile.ZipFile(written) as source, zipfile.ZipFile(
            broken, "w", zipfile.ZIP_DEFLATED
        ) as target:
            for name in source.namelist():
                payload = buffer.getvalue() if name == "色块图层.npy" else source.read(name)
                target.writestr(name, payload)
        with self.assertRaises(ProjectError) as caught:
            load_project(broken)
        self.assertIn("尺寸不符", str(caught.exception))

    def test_a_colour_painted_nowhere_keeps_its_step(self):
        # 换颜色换成了跟原来一样的颜色：图层上那一步什么都不盖。它仍然占一步，
        # 这样撤销的次数和用户做过的操作对得上。
        library = _library()
        base = _base_match(library)
        empty = np.zeros(base.indices.shape, dtype=bool)
        replacement = _palette(library)[3]
        written = save_project(
            self._dir / "empty",
            image=_picture(),
            image_name="空改动",
            base=base,
            library=library,
            block_paints=[(empty, replacement)],
        )
        state = load_project(written, library=None)
        self.assertEqual(len(state.block_paints), 1)
        self.assertFalse(np.any(state.block_paints[0]["mask"]))
        self.assertEqual(state.block_paints[0]["entry"].color_hex, replacement.color_hex)

    def test_absurdly_many_block_steps_are_refused(self):
        # 图层是 int16：说清楚上限，比写出一张悄悄错位的图好。
        from app.core.project import _paint_layer

        library = _library()
        base = _base_match(library)
        paints = [(np.zeros(base.indices.shape, dtype=bool), None)] * 32001
        with self.assertRaises(ProjectError) as caught:
            _paint_layer(base.indices.shape, paints)
        self.assertIn("32000", str(caught.exception))


class ProjectErrorTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = Path(self.enterContext(_temp_dir()))

    def test_a_file_that_is_not_a_zip_is_refused(self):
        broken = self._dir / "broken.bpproj"
        broken.write_bytes(b"not a zip at all")
        with self.assertRaises(ProjectError) as caught:
            load_project(broken)
        self.assertIn("不是有效的工程文件", str(caught.exception))

    def test_a_zip_missing_the_members_is_refused(self):
        library = _library()
        base = _base_match(library)
        written = save_project(
            self._dir / "trimmed", image=_picture(), image_name="缺件", base=base, library=library
        )
        trimmed = self._dir / "trimmed2.bpproj"
        with zipfile.ZipFile(written) as source, zipfile.ZipFile(
            trimmed, "w", zipfile.ZIP_DEFLATED
        ) as target:
            target.writestr("工程.json", source.read("工程.json"))
        with self.assertRaises(ProjectError) as caught:
            load_project(trimmed)
        self.assertIn("缺少", str(caught.exception))

    def test_another_apps_file_is_refused(self):
        path = self._dir / "other.bpproj"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("工程.json", json.dumps({"app": "SomethingElse", "format": 1}))
            archive.writestr("原图.png", b"")
            archive.writestr("基线.npy", b"")
        with self.assertRaises(ProjectError) as caught:
            load_project(path)
        self.assertIn("不是 BambuPalette 工程", str(caught.exception))

    def test_a_newer_format_is_refused_instead_of_half_read(self):
        library = _library()
        base = _base_match(library)
        written = save_project(
            self._dir / "future", image=_picture(), image_name="未来", base=base, library=library
        )
        bumped = self._dir / "future2.bpproj"
        with zipfile.ZipFile(written) as source, zipfile.ZipFile(
            bumped, "w", zipfile.ZIP_DEFLATED
        ) as target:
            for name in source.namelist():
                payload = source.read(name)
                if name == "工程.json":
                    document = json.loads(payload.decode("utf-8"))
                    document["format"] = 2
                    payload = json.dumps(document, ensure_ascii=False).encode("utf-8")
                target.writestr(name, payload)
        with self.assertRaises(ProjectError) as caught:
            load_project(bumped)
        self.assertIn("这个版本只认 1", str(caught.exception))


def _temp_dir():
    @contextmanager
    def manager():
        with tempfile.TemporaryDirectory(prefix="fcs-project-") as name:
            yield name

    return manager()


if __name__ == "__main__":
    unittest.main()

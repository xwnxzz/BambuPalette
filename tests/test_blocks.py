"""色块标注的测试：一张索引图被切成连成一片的同色区域。

1.6.17 的「按色块边缘选择」全部建立在这上面——列表一行一块、预览高亮一块、删除 /
合并 / 更换颜色都是按块来的——所以这里盯住三件事：

* 切得对不对：四邻接、颜色不同就断开、没打印的像素不属于任何一块，且和一遍朴素的
  洪水填充给出一模一样的结果（不是「差不多」）；
* 去噪守规矩：只并小块、不丢像素、不把一大块拆开、孤岛留着；
* 门槛随图片大小走。

Run with::

    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.blocks import (  # noqa: E402
    _MAX_ABSORB_PASSES,
    MIN_BLOCK_SHARE,
    label_blocks,
    min_block_pixels,
)


def _grid(rows: str) -> np.ndarray:
    """A picture from text: ``.`` = 不打印, a letter = a palette index."""
    lines = [line.strip() for line in rows.strip().splitlines() if line.strip()]
    letters = sorted({letter for line in lines for letter in line if letter != "."})
    codes = {letter: index for index, letter in enumerate(letters)}
    return np.asarray(
        [[-1 if letter == "." else codes[letter] for letter in line] for line in lines],
        dtype=np.int32,
    )


def _flood(grid: np.ndarray) -> tuple[np.ndarray, list[int], list[int]]:
    """The obvious answer: one pass of flood fill per connected piece."""
    height, width = (int(grid.shape[0]), int(grid.shape[1]))
    labels = np.full((height, width), -1, dtype=np.int32)
    sizes: list[int] = []
    values: list[int] = []
    for y in range(height):
        for x in range(width):
            if grid[y, x] < 0 or labels[y, x] >= 0:
                continue
            value = int(grid[y, x])
            label = len(sizes)
            labels[y, x] = label
            stack = [(y, x)]
            size = 0
            while stack:
                cy, cx = stack.pop()
                size += 1
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        if labels[ny, nx] < 0 and grid[ny, nx] == value:
                            labels[ny, nx] = label
                            stack.append((ny, nx))
            sizes.append(size)
            values.append(value)
    return labels, sizes, values


def _touching(labels: np.ndarray, mask: np.ndarray) -> int:
    """How many *other* printed pixels sit right next to ``mask`` (4-neighbours, no wrap)."""
    other = (labels >= 0) & ~mask
    total = 0
    total += int(np.count_nonzero(mask[:, :-1] & other[:, 1:]))
    total += int(np.count_nonzero(mask[:, 1:] & other[:, :-1]))
    total += int(np.count_nonzero(mask[:-1, :] & other[1:, :]))
    total += int(np.count_nonzero(mask[1:, :] & other[:-1, :]))
    return total


class BlockLabelTests(unittest.TestCase):
    def test_one_colour_in_two_places_is_two_blocks(self):
        # 同一个调色板下标出现在两处，那就是两块——这正是「按色块选择」和「按颜色
        # 选择」的区别：颜色只有一种，块有两块。
        grid = _grid(
            """
            A.A
            ...
            A.A
            """
        )
        blocks = label_blocks(grid)
        self.assertEqual(blocks.count, 4)
        self.assertEqual(sorted(blocks.sizes.tolist()), [1, 1, 1, 1])
        self.assertEqual(set(blocks.values.tolist()), {0})

    def test_touching_only_at_a_corner_is_not_one_block(self):
        # 四邻接：只在对角碰到的两块各算各的（八邻接会把它们并成一块）。
        grid = _grid(
            """
            A.
            .A
            """
        )
        self.assertEqual(label_blocks(grid).count, 2)

    def test_unprinted_pixels_belong_to_no_block(self):
        grid = _grid(
            """
            A.
            ..
            """
        )
        blocks = label_blocks(grid)
        self.assertEqual(blocks.count, 1)
        self.assertEqual(blocks.size(0), 1)
        self.assertFalse(blocks.mask(0)[1, 1])

    def test_sizes_values_and_order_are_reported(self):
        grid = _grid(
            """
            AAA.B
            AAA.B
            """
        )
        blocks = label_blocks(grid)
        self.assertEqual(blocks.count, 2)
        # order() 是列表用的顺序：大的在前，一样大按编号。
        self.assertEqual(blocks.order(), [0, 1])
        self.assertEqual([blocks.size(label) for label in blocks.order()], [6, 2])
        self.assertEqual([blocks.value(label) for label in blocks.order()], [0, 1])

    def test_it_agrees_with_a_flood_fill(self):
        # 「差不多」不算数：编号、大小、颜色都要和一遍朴素洪水填充一样。编号按光栅
        # 顺序发，所以连 labels 图本身都该逐像素相同——列表和预览就是靠它对齐的。
        rng = np.random.default_rng(20250117)
        for attempt in range(40):
            grid = rng.integers(-1, 4, size=(7, 9)).astype(np.int32)
            blocks = label_blocks(grid)
            labels, sizes, values = _flood(grid)
            self.assertTrue(
                np.array_equal(blocks.labels, labels),
                f"第 {attempt} 张图和洪水填充的结果不一样",
            )
            self.assertEqual(blocks.sizes.tolist(), sizes)
            self.assertEqual(blocks.values.tolist(), values)

    def test_a_one_dimensional_picture_works(self):
        grid = _grid("A.B.A")
        blocks = label_blocks(grid)
        self.assertEqual(blocks.count, 3)
        self.assertEqual(blocks.sizes.tolist(), [1, 1, 1])


class DespeckleTests(unittest.TestCase):
    """比门槛小的碎块并进邻居；大块和孤岛不动。"""

    def test_a_speck_is_folded_into_the_neighbour_it_touches_most(self):
        # 中间一颗 1 像素的 B：上下左右都是 A，并进 A。
        grid = _grid(
            """
            AAAAA
            AAAAA
            AABAA
            AAAAA
            AAAAA
            """
        )
        blocks = label_blocks(grid, min_pixels=4)
        self.assertEqual(blocks.count, 1)
        self.assertEqual(blocks.size(0), 25)
        self.assertEqual(blocks.value(0), 0)

    def test_no_pixel_is_lost_and_none_is_invented(self):
        rng = np.random.default_rng(7)
        grid = rng.integers(-1, 3, size=(24, 24)).astype(np.int32)
        blocks = label_blocks(grid, min_pixels=6)
        self.assertTrue(np.array_equal(blocks.labels >= 0, grid >= 0))
        self.assertEqual(int(blocks.sizes.sum()), int(np.count_nonzero(grid >= 0)))
        self.assertTrue(np.array_equal(blocks.labels.shape, grid.shape))

    def test_a_block_bigger_than_the_threshold_is_never_split(self):
        rng = np.random.default_rng(11)
        for attempt in range(20):
            grid = rng.integers(-1, 4, size=(18, 18)).astype(np.int32)
            raw = label_blocks(grid)
            blocks = label_blocks(grid, min_pixels=5)
            for label in range(raw.count):
                if raw.size(label) < 5:
                    continue
                mine = raw.mask(label)
                self.assertEqual(
                    len(set(blocks.labels[mine].tolist())),
                    1,
                    f"第 {attempt} 张图的第 {label} 块（{raw.size(label)} 像素）被拆开了",
                )

    def test_an_isolated_island_stays_a_block_of_its_own(self):
        # 四周什么都没有的孤岛没有邻居可并——留着，不能因为小就凭空消失。
        grid = _grid(
            """
            .....
            .....
            ..A..
            .....
            .....
            """
        )
        blocks = label_blocks(grid, min_pixels=9)
        self.assertEqual(blocks.count, 1)
        self.assertEqual(blocks.size(0), 1)

    def test_despeckling_leaves_only_isolated_leftovers(self):
        # 去噪之后不该剩下「能并却没并」的碎块：比门槛小的块只能是四周真的什么都没有
        # 的孤岛（没邻居可并），不能是漏并的碎点。
        rng = np.random.default_rng(23)
        grid = rng.integers(-1, 3, size=(20, 20)).astype(np.int32)
        blocks = label_blocks(grid, min_pixels=8)
        for label in range(blocks.count):
            if blocks.size(label) >= 8:
                continue
            self.assertEqual(
                _touching(blocks.labels, blocks.mask(label)),
                0,
                f"第 {label} 块（{blocks.size(label)} 像素）比门槛小，却还有邻居",
            )

    def test_the_result_is_the_same_every_time(self):
        # 同一张图跑两次必须一模一样，否则列表说的「第 3 块」在预览里会指到别处。
        grid = _grid(
            """
            AABAA
            ABBBA
            AABAA
            """
        )
        first = label_blocks(grid, min_pixels=3)
        second = label_blocks(grid, min_pixels=3)
        self.assertTrue(np.array_equal(first.labels, second.labels))
        self.assertEqual(first.values.tolist(), second.values.tolist())

    def test_despeckling_gives_up_rather_than_looping_forever(self):
        # 门槛比整张图还大、又没有邻居可并的时候不能转圈：给个上限，超了就照现状收工。
        self.assertGreater(_MAX_ABSORB_PASSES, 0)
        grid = _grid("A.")
        blocks = label_blocks(grid, min_pixels=1000)
        self.assertEqual(blocks.count, 1)


class ThresholdTests(unittest.TestCase):
    def test_small_pictures_get_the_floor_not_a_fraction_of_a_pixel(self):
        self.assertEqual(min_block_pixels(0), 2)
        self.assertEqual(min_block_pixels(1), 2)
        self.assertEqual(min_block_pixels(5000), 2)

    def test_big_pictures_scale_and_stop_at_the_cap(self):
        # 512×512 的猪图：约 12.2 万实心像素 → 24 像素，正好吃掉一圈抗锯齿的碎点。
        self.assertEqual(min_block_pixels(122358), 24)
        self.assertEqual(min_block_pixels(1_000_000), 64)
        self.assertEqual(MIN_BLOCK_SHARE, 0.0002)

    def test_it_never_shrinks_as_the_picture_grows(self):
        values = [min_block_pixels(count) for count in range(0, 400000, 977)]
        self.assertEqual(values, sorted(values))

    def test_the_threshold_is_what_makes_a_speck_disappear(self):
        grid = _grid(
            """
            AAAAA
            AABAA
            AAAAA
            """
        )
        self.assertEqual(label_blocks(grid).count, 2)
        self.assertEqual(label_blocks(grid, min_pixels=min_block_pixels(24)).count, 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

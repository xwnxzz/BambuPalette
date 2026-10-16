"""色块：打印图里连成一片的同一种颜色，一块一块地数出来。

「按颜色选择」把一张图看成几张色表——一种颜色是「眉毛、眼睛、描边」加起来；
「按色块边缘选择」（1.6.17）要的是能指着屏幕上那一块说话的东西，所以这里把
:class:`~app.core.image_matching.MatchResult` 的索引图切成连通区域：四邻接、
颜色不同就断开、没打印的像素（``-1``）不属于任何一块。

**为什么还要去噪。** 直接数连通区域，512×512 的猪图会给 953 块，其中 773 块
只有一个像素——每一道抗锯齿的边都会掉出一圈碎点。这种表没法看，用户心里的
「一块」是耳朵、眼睛、刘海这种形状。所以比 ``min_pixels`` 还小的碎块会被并进
它最贴着的邻居（贴着谁的边最长就跟谁），再去噪后重新编号；只有四周什么都没有
的孤岛留着自己一块。``min_pixels`` 按打印像素数取千分之零点二（512 px 的图约
24 像素），小图自动放宽，5×5 的合成图不会因此并成一块。

编号按光栅顺序（从上到下、从左到右第一次出现）发，同一次调用永远给同一批编号，
列表和预览才说得上是「同一块」；``sizes`` 是每块的像素数，界面按它从大到小排。
``values`` 是每块用的调色板下标——去噪会让碎点的像素改用邻居的颜色，所以块的
颜色必须由这里给，不能回头去读原图。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["BlockMap", "label_blocks", "min_block_pixels"]

#: 去噪门槛占打印像素的比例：512×512（约 12 万实心像素）的图画出来是 24 像素，
#: 比一个 5×5 的点还小；再小的东西在成品上就是噪点而不是形状。
MIN_BLOCK_SHARE = 0.0002
_MIN_BLOCK_CAP = 64
_MAX_ABSORB_PASSES = 12


def min_block_pixels(printed_pixels: int) -> int:
    """去噪门槛：小图不至于被并成一块，大图的碎点一定被吃掉。"""
    return max(2, min(_MIN_BLOCK_CAP, round(int(printed_pixels) * MIN_BLOCK_SHARE)))


@dataclass(frozen=True)
class BlockMap:
    """A picture cut into blocks; ``labels`` is the H×W map, ``-1`` = not printed."""

    labels: np.ndarray
    sizes: np.ndarray
    values: np.ndarray

    @property
    def count(self) -> int:
        return int(len(self.sizes))

    def order(self) -> list[int]:
        """Label ids largest first, ties by label — the order the list shows."""
        return [
            int(label)
            for label in sorted(range(self.count), key=lambda i: (-int(self.sizes[i]), i))
        ]

    def mask(self, label: int) -> np.ndarray:
        """The pixels of ``label`` as a bool mask (all ``False`` when it is gone)."""
        return self.labels == int(label)

    def size(self, label: int) -> int:
        return int(self.sizes[int(label)])

    def value(self, label: int) -> int:
        """The palette index this block prints with."""
        return int(self.values[int(label)])


def label_blocks(
    indices: np.ndarray, *, min_pixels: int = 0, keep_colours: bool = False
) -> BlockMap:
    """Cut ``indices`` (H, W; ``-1`` = not printed) into blocks.

    ``min_pixels`` is the despeckle threshold: blocks smaller than it are folded
    into the neighbour they share the longest border with.  ``0`` keeps every
    connected piece, which is what the tests and the raw measurements want.

    ``keep_colours`` keeps one entry per colour that is still printed (m14748
    第 2 条).  它只给界面用：去噪之后一个位子都不剩的颜色，会补一行**散点块**，
    免得「打印图片需要的颜色」列出了它、色块表里却没有它。
    """
    grid = np.asarray(indices)
    if grid.ndim != 2:
        raise ValueError("色块标注只认二维索引图。")
    work = grid.astype(np.int32, copy=True)
    labels, sizes = _runs(work)
    if min_pixels > 1 and sizes.size:
        work, labels, sizes = _absorb_small(work, labels, sizes, int(min_pixels))
    if keep_colours:
        work, labels, sizes = _keep_every_colour(work, labels, sizes, grid)
    return BlockMap(labels=labels, sizes=sizes, values=_block_values(work, labels, len(sizes)))


def revalue_blocks(
    labels: np.ndarray, indices: np.ndarray, *, previous: BlockMap | None = None
) -> BlockMap:
    """同一份色块划分，按新的分色结果重算每块打哪一色（m15721 第 1 条）。

    「按当前色块重新匹配」按一次要重新分色，但色块的划分和编号不许动：用户在色块模式里
    点过、改过的那些块还得在原处，块级编辑（按像素掩码）才贴得回去。 所以这里拿旧划分
    的标签图 ``labels`` 和新分色的索引图 ``indices`` 拼一份 :class:`BlockMap`：块数和
    编号照旧，每块的颜色仍按 :func:`_block_values` 那条规矩取 —— 光栅序里第一个像素的
    颜色，和没重新分色时一样，块的颜色不会因为换了一套分色就换个取法。

    ``previous`` 是这几块本来的那份 :class:`BlockMap`：新分色里某一块的格子全成了「不
    打印」（``-1``）时，它没有颜色可取了，就留本来那一种，而不是从别人那里顺手拿一个。

    尺寸对不上（用户改过「最大边长」）就别调这个：那样两套块根本不是同一张图上的东西，
    调用方该退回重新分块。
    """
    grid = np.asarray(indices)
    tag = np.asarray(labels)
    if grid.shape != tag.shape:
        raise ValueError("色块划分和新的分色结果不是同一个尺寸。")
    count = int(tag.max()) + 1 if tag.size else 0
    sizes = np.bincount(
        tag[tag >= 0].ravel(), minlength=max(0, count)
    )[:count].astype(np.int64)
    values = _block_values(grid.astype(np.int32, copy=False), tag, count)
    if previous is not None and count:
        # 这一块在新分色里还有没有一格是有颜色的？ 没有就留原来那一种。
        live = np.bincount(tag[grid >= 0].ravel(), minlength=count)[:count]
        silent = live == 0
        keep = np.asarray(previous.values)
        limit = min(count, int(keep.size))
        if silent[:limit].any():
            values[:limit] = np.where(silent[:limit], keep[:limit], values[:limit])
    return BlockMap(labels=tag.copy(), sizes=sizes, values=values.astype(np.int32))


def _keep_every_colour(
    work: np.ndarray, labels: np.ndarray, sizes: np.ndarray, original: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """每种在图里出现过的颜色都留一个位子（m14748 第 2 条）。

    去噪并掉的是**位置**，不是颜色：颜色表按像素算，照样把这个颜色列进「打印图片需要
    的颜色」，色块表却一个位子都不剩 —— 用户看到的就是「图一有红色，图二没有红色色块」。
    实测那一份工程：红色 474 px，散成 214 片，最大一片 136 px，全部低于 64 px 的门槛，
    于是被吃得干干净净（黑色也整片消失）。

    这里给还缺位子的颜色补一个**散点块**：像素彼此不相连，但一个颜色一行，两本账就
    一样了。散点块不是新画的东西 —— 颜色表里那个 0.0% 的颜色本来就在打，只是以前
    它没有可以点的地方：「删除」删不掉、「更换颜色」换不了。
    """
    present = [int(value) for value in np.unique(original) if int(value) >= 0]
    if not present:
        return work, labels, sizes
    values = _block_values(work, labels, len(sizes))
    have = {int(value) for value in values.tolist()}
    missing = [value for value in present if value not in have]
    if not missing:
        return work, labels, sizes
    count = int(len(sizes))
    filled = False
    for value in missing:
        mask = (original == value) & (labels >= 0)
        if not mask.any():
            continue
        work = work.copy() if not filled else work
        labels = labels.copy() if not filled else labels
        filled = True
        work[mask] = value
        labels[mask] = count
        count += 1
    if not filled:
        return work, labels, sizes
    sizes = np.bincount(labels[labels >= 0].ravel(), minlength=count)[:count].astype(np.int64)
    return work, labels, sizes


def _runs(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row runs + union-find: the connected pieces of ``grid`` in raster order."""
    height, width = int(grid.shape[0]), int(grid.shape[1])
    labels = np.full((height, width), -1, dtype=np.int32)
    if height == 0 or width == 0:
        return labels, np.zeros(0, dtype=np.int64)

    #: A run is a maximal stretch of one row holding one colour.  Ids are handed
    #: out row by row, so a set's root is always its earliest run and the compact
    #: numbering below comes out in raster order.
    parent: list[int] = []

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    def union(left: int, right: int) -> None:
        first, second = find(left), find(right)
        if first != second:
            parent[max(first, second)] = min(first, second)

    rows: list[list[tuple[int, int, int]]] = []
    previous: list[tuple[int, int, int, int]] = []
    for y in range(height):
        row = grid[y]
        edges = np.flatnonzero(row[1:] != row[:-1]) + 1
        if edges.size:
            starts = np.concatenate(([0], edges))
            stops = np.concatenate((edges, [width]))
        else:
            starts = np.array([0], dtype=np.int64)
            stops = np.array([width], dtype=np.int64)
        current: list[tuple[int, int, int, int]] = []
        for start, stop in zip(starts.tolist(), stops.tolist()):
            value = int(row[start])
            if value < 0:
                continue
            run = len(parent)
            parent.append(run)
            current.append((start, stop, run, value))
            # Touching runs of the row above belong together.  Overlap is "share
            # at least one column", which is exactly 4-connectivity.
            for pstart, pstop, prun, pvalue in previous:
                if pvalue == value and pstart < stop and start < pstop:
                    union(run, prun)
        rows.append([(start, stop, run) for start, stop, run, _ in current])
        previous = current

    roots = [find(run) for run in range(len(parent))]
    numbering = {root: position for position, root in enumerate(sorted(set(roots)))}
    sizes = np.zeros(len(numbering), dtype=np.int64)
    for y, runs in enumerate(rows):
        for start, stop, run in runs:
            label = numbering[roots[run]]
            labels[y, start:stop] = label
            sizes[label] += stop - start
    return labels, sizes


def _borders(labels: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every 4-adjacent pair of different blocks with how many pixels they share."""
    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    for horizontal in (True, False):
        if horizontal:
            first, second = labels[:, :-1], labels[:, 1:]
        else:
            first, second = labels[:-1, :], labels[1:, :]
        keep = (first >= 0) & (second >= 0) & (first != second)
        if keep.any():
            left.append(first[keep].astype(np.int64))
            right.append(second[keep].astype(np.int64))
    if not left:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty
    first = np.concatenate(left)
    second = np.concatenate(right)
    # Both directions, so a block's neighbourhood is complete from its own side.
    both = np.concatenate((first, second))
    other = np.concatenate((second, first))
    keys, counts = np.unique(both * np.int64(count) + other, return_counts=True)
    return keys // count, keys % count, counts


def _absorb_small(
    work: np.ndarray, labels: np.ndarray, sizes: np.ndarray, min_pixels: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fold blocks below ``min_pixels`` into the neighbour they touch most."""
    for _ in range(_MAX_ABSORB_PASSES):
        small = np.flatnonzero(sizes < min_pixels)
        if small.size == 0:
            return work, labels, sizes
        values = _block_values(work, labels, len(sizes))
        first, second, counts = _borders(labels, len(sizes))
        # Longest shared border wins; ties go to the bigger block, then the lower
        # label — one deterministic answer per picture.
        ranking = sorted(
            zip(first.tolist(), second.tolist(), counts.tolist()),
            key=lambda item: (-item[2], -int(sizes[item[1]]), item[1]),
        )
        winner: dict[int, int] = {}
        small_set = set(small.tolist())
        for one, other, _ in ranking:
            if one in small_set and one not in winner:
                winner[one] = other
        if not winner:
            return work, labels, sizes  # 只剩孤岛：没邻居可并，留着
        # 折叠的方向由 winner 说了算：小片并进哪一块，留下的就是哪一块的颜色。
        # 以前用并查集的根（两个里编号小的那个）取颜色，于是「光栅序排在前面」的
        # 碎点会把整块大色区改色 —— m14748 第 2 条在猪信工程里量到 4 块：块里
        # 1921 px 的原始像素是 #2C2A2C，报出来的却是 #A1A1A1；颜色表说黑色打了
        # 7604 px，色块表里黑色只剩两块 698 / 678 px。
        destination = list(range(len(sizes)))

        def land(node: int) -> int:
            trail: list[int] = []
            while destination[node] != node:
                trail.append(node)
                node = destination[node]
            for step in trail:
                destination[step] = node
            return node

        for one, other in winner.items():
            mine, theirs = land(one), land(other)
            if mine != theirs:
                destination[mine] = theirs
        dest = np.asarray([land(label) for label in range(len(sizes))], dtype=np.int64)
        merged = np.where(labels >= 0, dest[labels], -1)
        work = np.where(merged >= 0, values[merged], -1).astype(np.int32)
        labels, sizes = _runs(work)
    return work, labels, sizes


def _block_values(work: np.ndarray, labels: np.ndarray, count: int) -> np.ndarray:
    """The palette index each block prints with (blocks hold one value each)."""
    if count == 0:
        return np.zeros(0, dtype=np.int32)
    flat_labels = labels.ravel()
    keep = flat_labels >= 0
    keys = flat_labels[keep]
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    sorted_values = work.ravel()[keep][order]
    starts = np.searchsorted(sorted_keys, np.arange(count, dtype=np.int32), side="left")
    starts = np.minimum(starts, max(0, sorted_values.size - 1))
    return sorted_values[starts].astype(np.int32)

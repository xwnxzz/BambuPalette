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


def label_blocks(indices: np.ndarray, *, min_pixels: int = 0) -> BlockMap:
    """Cut ``indices`` (H, W; ``-1`` = not printed) into blocks.

    ``min_pixels`` is the despeckle threshold: blocks smaller than it are folded
    into the neighbour they share the longest border with.  ``0`` keeps every
    connected piece, which is what the tests and the raw measurements want.
    """
    grid = np.asarray(indices)
    if grid.ndim != 2:
        raise ValueError("色块标注只认二维索引图。")
    work = grid.astype(np.int32, copy=True)
    labels, sizes = _runs(work)
    if min_pixels > 1 and sizes.size:
        work, labels, sizes = _absorb_small(work, labels, sizes, int(min_pixels))
    return BlockMap(labels=labels, sizes=sizes, values=_block_values(work, labels, len(sizes)))


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
        # Follow the chain (small → small → big) so one pass merges whole groups.
        target = list(range(len(sizes)))

        def root(node: int) -> int:
            while target[node] != node:
                target[node] = target[target[node]]
                node = target[node]
            return node

        for one, other in winner.items():
            mine, theirs = root(one), root(other)
            if mine != theirs:
                target[max(mine, theirs)] = min(mine, theirs)
        resolved = np.asarray([root(label) for label in range(len(sizes))], dtype=np.int64)
        merged = np.where(labels >= 0, resolved[labels], -1)
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

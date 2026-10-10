"""三色混色：C(n,3) 组三卷耗材 × 2556 个整数配比。

两份耗材是「两两混色」；三份耗材的配比空间要大得多。设三卷的用量是
``a`` / ``b`` / ``c`` 个百分点，要求

    a + b + c = 100        三卷加起来正好是一整卷
    a, b, c >= 10          任何一卷都不低于 10%

凡是整数解都算一条配方（交换用量算不同配方）。令 ``a' = a - 10`` 等，
则 ``a' + b' + c' = 70``，非负整数解的个数是 ``C(70 + 3 - 1, 3 - 1) = C(72, 2)``
= **2556**。因为三卷都 >= 10%，任何一卷自动 <= 80%，所以「每卷 <= 80%」这条
约束永远不会生效。

41 卷大简 PETG HF 展开是 ``C(41,3) = 10,660`` 组三卷，
``10,660 x 2556 = 27,246,960`` 条配方，去重后约 429 万个颜色 ——
所以这个模块是 numpy 支撑的：每条配方只存一个 32 位编号，
颜色按 RGB 排好序、配方按颜色分组存成 CSR，
只有真的要显示某个颜色时才把它的配方还原成对象。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from ..spectral import color as _color
from .engines import ENGINES, DEFAULT_ENGINE
from .library import Filament

# 三卷配比的最小百分比。10% 是用户给的约束，也是两色混色用的同一个下限。
TRIPLE_MIN_PERCENT = 10
# C(72, 2)：三卷各 >= 10% 的整数配比个数。
TRIPLE_RATIO_COUNT = 2556
# 每次丢给混色引擎的三卷组数。24 x 2556 = 61,344 行，够快也够省内存。
TRIPLE_CHUNK = 24
# 磁盘缓存的文件名前缀，以及最多保留几份。
TRIPLE_CACHE_PREFIX = "triples-"
TRIPLE_CACHE_KEEP = 3


def triple_ratios(minimum: int = TRIPLE_MIN_PERCENT) -> np.ndarray:
    """返回 ``(2556, 3)`` 的整数配比表，顺序是 a 升序、再 b 升序。

    顺序固定，所以缓存下来的配方编号可以跨进程复用。
    """
    rows: list[tuple[int, int, int]] = []
    for a in range(minimum, 101 - 2 * minimum):
        for b in range(minimum, 101 - minimum - a):
            rows.append((a, b, 100 - a - b))
    return np.asarray(rows, dtype=np.int64)


def triple_pair_count(spool_count: int) -> int:
    """``n`` 卷耗材能排出多少组三卷（``C(n,3)``）。"""
    if spool_count < 3:
        return 0
    return spool_count * (spool_count - 1) * (spool_count - 2) // 6


def triple_recipe_count(spool_count: int) -> int:
    """``n`` 卷耗材一共能排出多少条三色配方。"""
    return triple_pair_count(spool_count) * TRIPLE_RATIO_COUNT


def hue_of(lab: Sequence[float]) -> float:
    """CIELAB 的色相角，单位度，落在 ``[0, 360)``。"""
    return math.degrees(math.atan2(float(lab[2]), float(lab[1]))) % 360.0


def chroma_of(lab: Sequence[float]) -> float:
    return float((float(lab[1]) ** 2 + float(lab[2]) ** 2) ** 0.5)


@dataclass(frozen=True)
class TripleRecipe:
    """一条三色配方：三卷母材 + 三个百分比。

    故意和 :class:`~app.core.mixes.MixRecipe` 暴露同一批只读属性名
    （``color_hex`` / ``rgb`` / ``lab`` / ``engine`` / ``key`` / ``ratio_text``），
    这样详情面板和配方清单可以同时吃两种配方。
    """

    triple_index: int
    a_id: str
    b_id: str
    c_id: str
    percent_a: int
    percent_b: int
    percent_c: int
    color_hex: str
    rgb: tuple[int, int, int]
    lab: tuple[float, float, float]
    engine: str

    @property
    def key(self) -> str:
        return f"{self.a_id}|{self.b_id}|{self.c_id}|{self.percent_a}|{self.percent_b}"

    @property
    def parent_ids(self) -> tuple[str, str, str]:
        return (self.a_id, self.b_id, self.c_id)

    @property
    def percents(self) -> tuple[int, int, int]:
        return (self.percent_a, self.percent_b, self.percent_c)

    @property
    def ratio_text(self) -> str:
        return f"{self.percent_a}% + {self.percent_b}% + {self.percent_c}%"

    @property
    def lightness(self) -> float:
        return float(self.lab[0])

    @property
    def hue(self) -> float:
        return hue_of(self.lab)

    @property
    def chroma(self) -> float:
        return chroma_of(self.lab)

    @property
    def pair_index(self) -> int:
        # 三色配方不属于任何「一对耗材」，和耗材本色一样用 -1。
        return -1


class TripleColour:
    """一个去重后的三色混色颜色。

    ``recipes`` 是**懒加载**的：429 万个颜色不可能每个都先建好对象，
    只有真的被打开（详情面板、候选表）时才去 ``store.recipes_at()`` 还原，
    还原结果留在 ``_recipes`` 里。
    """

    __slots__ = ("_store", "index", "rgb", "color_hex", "lab", "_recipes")

    def __init__(self, store: "TripleCatalog", index: int) -> None:
        self._store = store
        self.index = int(index)
        value = int(store.packed[index])
        self.rgb = ((value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF)
        self.color_hex = _color.rgb_to_hex(self.rgb)
        self.lab = _tuple_lab(_color.lab_from_rgb(self.rgb))
        self._recipes: tuple[TripleRecipe, ...] | None = None

    @property
    def recipes(self) -> tuple[TripleRecipe, ...]:
        if self._recipes is None:
            self._recipes = self._store.recipes_at(self.index)
        return self._recipes

    @property
    def key(self) -> str:
        return f"triple|{self.color_hex}"

    @property
    def recipe_count(self) -> int:
        return self._store.recipe_count_at(self.index)

    @property
    def first(self) -> TripleRecipe:
        return self.recipes[0]

    @property
    def a_id(self) -> str:
        return self.first.a_id

    @property
    def b_id(self) -> str:
        return self.first.b_id

    @property
    def percent_a(self) -> int:
        return self.first.percent_a

    @property
    def percent_b(self) -> int:
        return self.first.percent_b

    @property
    def ratio_text(self) -> str:
        return f"{self.recipe_count} 个配方" if self.recipe_count > 1 else self.first.ratio_text

    @property
    def lightness(self) -> float:
        return float(self.lab[0])

    @property
    def hue(self) -> float:
        return hue_of(self.lab)

    @property
    def chroma(self) -> float:
        return chroma_of(self.lab)

    @property
    def pair_index(self) -> int:
        return -1

    @property
    def engine(self) -> str:
        return self._store.engine_id

    @property
    def is_spool(self) -> bool:
        return False

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<TripleColour {self.color_hex} x{self.recipe_count}>"


class TripleCatalog:
    """``C(n,3) x 2556`` 条三色配方的只读索引。

    存储：

    * ``packed`` —— 去重后按 RGB 升序的颜色，``(C,) int32``，值是 ``(r<<16)|(g<<8)|b``；
    * ``offsets`` —— CSR 分组边界，``(C+1,) int64``；
    * ``codes`` —— 排好序的配方编号，``(M,) uint32``。

    配方编号 ``r`` 代表 ``triples[r // R]`` 这组三卷用 ``ratios[r % R]`` 这组配比。
    """

    def __init__(
        self,
        filaments: Sequence[Filament],
        engine: str = DEFAULT_ENGINE,
        ratios: np.ndarray | None = None,
    ) -> None:
        self._filaments = list(filaments)
        self._engine_id = engine
        self._engine = ENGINES[engine]
        self._ratios = triple_ratios() if ratios is None else ratios
        self._parent_index = _triples(len(self._filaments))
        self.packed = np.empty(0, dtype=np.int32)
        self._offsets = np.zeros(1, dtype=np.int64)
        self._codes = np.empty(0, dtype=np.uint32)
        self._labs: np.ndarray | None = None
        self._built = False
        self.build_seconds = 0.0

    # -- 基本信息 ---------------------------------------------------------

    @property
    def engine(self):
        return self._engine

    @property
    def engine_id(self) -> str:
        return self._engine_id

    @property
    def filaments(self) -> list[Filament]:
        return list(self._filaments)

    @property
    def ratios(self) -> np.ndarray:
        return self._ratios

    @property
    def triple_count(self) -> int:
        return int(len(self._parent_index))

    @property
    def recipe_count(self) -> int:
        return int(len(self._parent_index) * len(self._ratios))

    @property
    def colour_count(self) -> int:
        return int(len(self.packed))

    @property
    def built(self) -> bool:
        return self._built

    def triples(self) -> list[tuple[str, str, str]]:
        """所有三卷组合，按耗材 id 给出。"""
        return [
            tuple(self._filaments[int(index)].id for index in row)  # type: ignore[misc]
            for row in self._parent_index
        ]

    # -- 建表 -------------------------------------------------------------

    def build(self, progress: Callable[[int, int], None] | None = None) -> "TripleCatalog":
        """算完所有三色配方并按颜色去重。

        ``progress(done, total)`` 每块调一次；``total`` 是配方总数
        （41 卷时 27,246,960），用来驱动进度条。
        """
        import time

        started = time.perf_counter()
        self.begin()
        while not self.step(1.0):
            if progress is not None:
                progress(self.mixed_done, self.recipe_total)
        if progress is not None:
            progress(self.mixed_done, self.recipe_total)
        self.finish()
        self.build_seconds = time.perf_counter() - started
        return self

    # -- 分步建表（界面用，避免几秒钟里窗口不动） --------------------------

    def begin(self) -> None:
        """准备分步建表：``begin()`` → 反复 ``step()`` → ``finish()``。"""
        self._mixed_total = len(self._parent_index) * len(self._ratios)
        self._mixed = np.empty(self._mixed_total, dtype=np.int64)
        self._base = np.asarray([f.rgb for f in self._filaments], dtype=np.float64)
        self._weights = self._ratios.astype(np.float64) / 100.0
        self._block = 0
        self._mixed_done = 0

    def step(self, budget_seconds: float = 0.20) -> bool:
        """混色最多 ``budget_seconds`` 秒；返回是否已经全部算完。"""
        import time

        deadline = time.perf_counter() + max(0.005, float(budget_seconds))
        count = len(self._parent_index)
        if count == 0:
            return True
        while self._block < count:
            start = self._block
            block = self._parent_index[start : start + TRIPLE_CHUNK]
            parents = self._base[block]
            block_weights = np.broadcast_to(
                self._weights, (parents.shape[0], len(self._ratios), 3)
            )
            flat = self._engine.mix_rgb(parents, block_weights).reshape(-1, 3)
            self._mixed[self._mixed_done : self._mixed_done + len(flat)] = (
                (flat[:, 0].astype(np.int64) << 16)
                | (flat[:, 1].astype(np.int64) << 8)
                | flat[:, 2].astype(np.int64)
            )
            self._mixed_done += len(flat)
            self._block = start + TRIPLE_CHUNK
            if self._block < count and time.perf_counter() >= deadline:
                break
        return self._block >= count

    @property
    def recipe_total(self) -> int:
        """这一批三色配方一共多少条（进度条的分母）。"""
        return int(getattr(self, "_mixed_total", len(self._parent_index) * len(self._ratios)))

    @property
    def mixed_done(self) -> int:
        """已经算完多少条配方（进度条的分子）。"""
        return int(getattr(self, "_mixed_done", 0))

    def finish(self) -> "TripleCatalog":
        """排序去重，把中间结果收成 CSR 三件套。"""
        import time

        started = time.perf_counter()
        mixed = self._mixed[: self._mixed_done]
        order = np.argsort(mixed, kind="stable")
        sorted_packed = mixed[order]
        keep = np.empty(len(sorted_packed), dtype=bool)
        keep[0] = True
        np.not_equal(sorted_packed[1:], sorted_packed[:-1], out=keep[1:])
        starts = np.flatnonzero(keep)
        self.packed = sorted_packed[starts].astype(np.int32)
        self._offsets = np.append(starts, len(sorted_packed)).astype(np.int64)
        self._codes = order.astype(np.uint32)
        self._labs = None
        self._built = True
        self._mixed = None
        self.merge_seconds = time.perf_counter() - started
        return self

    def build_from_codes(
        self, packed: np.ndarray, offsets: np.ndarray, codes: np.ndarray
    ) -> "TripleCatalog":
        """从缓存文件直接装载（``files.py`` 用）。"""
        self.packed = np.asarray(packed, dtype=np.int32)
        self._offsets = np.asarray(offsets, dtype=np.int64)
        self._codes = np.asarray(codes, dtype=np.uint32)
        self._labs = None
        self._built = True
        return self

    # -- 导出 / 装载 -------------------------------------------------------

    def dumps(self) -> dict[str, np.ndarray]:
        """拆成可以直接 ``np.savez`` 的数组。"""
        return {
            "packed": self.packed,
            "offsets": self._offsets,
            "codes": self._codes,
            "parents": self._parent_index.astype(np.int32),
            "ratios": self._ratios.astype(np.int32),
        }

    @classmethod
    def loads(
        cls,
        filaments: Sequence[Filament],
        data,
        engine: str = DEFAULT_ENGINE,
    ) -> "TripleCatalog":
        catalog = cls(filaments, engine=engine, ratios=np.asarray(data["ratios"], dtype=np.int64))
        catalog.build_from_codes(data["packed"], data["offsets"], data["codes"])
        return catalog

    # -- 查询 -------------------------------------------------------------

    def rgb_at(self, index: int) -> tuple[int, int, int]:
        value = int(self.packed[index])
        return ((value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF)

    def color_hex_at(self, index: int) -> str:
        return _color.rgb_to_hex(self.rgb_at(index))

    def recipe_count_at(self, index: int) -> int:
        return int(self._offsets[index + 1] - self._offsets[index])

    def recipes_at(self, index: int) -> tuple[TripleRecipe, ...]:
        """还原第 ``index`` 个颜色的全部配方。"""
        start = int(self._offsets[index])
        stop = int(self._offsets[index + 1])
        value = int(self.packed[index])
        rgb = ((value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF)
        color_hex = _color.rgb_to_hex(rgb)
        lab = _tuple_lab(_color.lab_from_rgb(rgb))
        ratios = self._ratios
        span = len(ratios)
        out: list[TripleRecipe] = []
        for position in range(start, stop):
            code = int(self._codes[position])
            triple_index, ratio_index = divmod(code, span)
            parents = self._parent_index[triple_index]
            percent = ratios[ratio_index]
            out.append(
                TripleRecipe(
                    triple_index=int(triple_index),
                    a_id=self._filaments[int(parents[0])].id,
                    b_id=self._filaments[int(parents[1])].id,
                    c_id=self._filaments[int(parents[2])].id,
                    percent_a=int(percent[0]),
                    percent_b=int(percent[1]),
                    percent_c=int(percent[2]),
                    color_hex=color_hex,
                    rgb=rgb,
                    lab=lab,
                    engine=self._engine_id,
                )
            )
        return tuple(out)

    def colour_at(self, index: int) -> TripleColour:
        return TripleColour(self, index)

    def colours(self) -> "TripleColours":
        """A lazy view over every distinct colour, safe to hand to the grid."""
        return TripleColours(self)

    def ordered_colours(
        self, key: str = "rgb", target_rgb: Sequence[int] | None = None
    ) -> "OrderedTripleColours":
        """同一个懒视图，但按 ``order()`` 给的顺序取。"""
        return OrderedTripleColours(self.colours(), self.order(key, target_rgb))

    def lab_table(self) -> np.ndarray:
        """所有颜色的 Lab，``(C, 3)``，算一次就缓存。"""
        if self._labs is None or len(self._labs) != self.colour_count:
            rgb = np.stack(
                [
                    (self.packed >> 16) & 0xFF,
                    (self.packed >> 8) & 0xFF,
                    self.packed & 0xFF,
                ],
                axis=1,
            ).astype(np.float64)
            self._labs = np.asarray(_color.lab_from_rgb(rgb), dtype=np.float64)
        return self._labs

    def order(self, key: str = "rgb", target_rgb: Sequence[int] | None = None) -> np.ndarray:
        """按某种顺序返回颜色下标的数组。

        ``key`` 用 ``mixes.SORT_*`` 的同一批字符串；默认 ``rgb`` 就是存储顺序
        （建表时已经按 RGB 排过）。三色混色的颜色太多，排序用 numpy 做。
        """
        from .mixes import (
            SORT_HUE,
            SORT_LABEL,
            SORT_LIGHTNESS,
            SORT_PAIR,
            SORT_RGB,
            SORT_SIMILARITY,
        )

        count = self.colour_count
        if count == 0:
            return np.empty(0, dtype=np.int64)
        if key in (SORT_RGB, SORT_LABEL, SORT_PAIR):
            return np.arange(count, dtype=np.int64)
        lab = self.lab_table()
        if key == SORT_LIGHTNESS:
            lightness = -lab[:, 0]
            return np.lexsort((self.packed, lightness)).astype(np.int64)
        if key == SORT_HUE:
            hues = np.degrees(np.arctan2(lab[:, 2], lab[:, 1])) % 360.0
            return np.lexsort((self.packed, hues)).astype(np.int64)
        if key == SORT_SIMILARITY and target_rgb is not None:
            target = np.asarray(_color.lab_from_rgb(tuple(target_rgb)), dtype=np.float64)
            distance = np.asarray(_color.delta_e_2000(lab, target), dtype=np.float64)
            return np.argsort(distance, kind="stable").astype(np.int64)
        return np.arange(count, dtype=np.int64)

    def nearest(self, color_hex: str) -> TripleColour | None:
        """离给定颜色最近的三色混色。"""
        if self.colour_count == 0:
            return None
        target = np.asarray(_color.lab_from_rgb(_color.hex_to_rgb(color_hex)), dtype=np.float64)
        distance = np.asarray(_color.delta_e_2000(self.lab_table(), target), dtype=np.float64)
        return self.colour_at(int(np.argmin(distance)))


class TripleColours(Sequence):
    """A lazy, indexable view of every distinct three-filament colour.

    A 41-spool library merges down to millions of colours, so this never builds
    the whole list: it makes one cell at a time, only for the cell the grid is
    actually drawing.  ``index_of_key`` binary-searches the RGB-sorted table,
    which is why a selection can be restored without a scan.
    """

    __slots__ = ("_store", "_count")

    def __init__(self, store: TripleCatalog) -> None:
        self._store = store
        self._count = store.colour_count

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self._store.colour_at(i) for i in range(*index.indices(self._count))]
        index = int(index)
        if index < 0:
            index += self._count
        if not 0 <= index < self._count:
            raise IndexError("triple colour index out of range")
        return self._store.colour_at(index)

    def index_of_key(self, key: str) -> int:
        text = str(key)
        if text.startswith("triple|"):
            text = text[len("triple|"):]
        try:
            rgb = _color.hex_to_rgb(text)
        except ValueError:
            return -1
        packed = (int(rgb[0]) << 16) | (int(rgb[1]) << 8) | int(rgb[2])
        table = self._store.packed
        found = int(np.searchsorted(table, packed, side="left"))
        if found < self._count and int(table[found]) == packed:
            return found
        return -1


def _triples(count: int) -> np.ndarray:
    """``(C(n,3), 3)`` 的所有三卷组合，下标升序。"""
    if count < 3:
        return np.zeros((0, 3), dtype=np.int64)
    left, middle, right = np.meshgrid(
        np.arange(count), np.arange(count), np.arange(count), indexing="ij"
    )
    mask = (left < middle) & (middle < right)
    return np.stack([left[mask], middle[mask], right[mask]], axis=1).astype(np.int64)


class OrderedTripleColours(Sequence):
    """The same lazy view, walked in the order ``TripleCatalog.order()`` chose.

    Sorting four million colours by hand would build four million objects, so
    the order stays a numpy permutation and only the cell being drawn is made.
    """

    __slots__ = ("_view", "_order", "_rank")

    def __init__(self, view: TripleColours, order: np.ndarray) -> None:
        self._view = view
        self._order = np.asarray(order, dtype=np.int64)
        self._rank = None

    def __len__(self) -> int:
        return int(self._order.shape[0])

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("ordered triple colour index out of range")
        return self._view[int(self._order[index])]

    def index_of_key(self, key: str) -> int:
        """Where a colour sits in THIS order (the inverse permutation)."""
        found = self._view.index_of_key(key)
        if found < 0:
            return -1
        if self._rank is None:
            rank = np.empty(len(self), dtype=np.int64)
            rank[self._order] = np.arange(len(self), dtype=np.int64)
            self._rank = rank
        return int(self._rank[found])


def triple_cache_key(
    filaments: Sequence, engine_id: str, minimum: int = TRIPLE_MIN_PERCENT
) -> str:
    """一个耗材库 + 一个引擎对应一个短键，用来当缓存文件名。

    配方只跟「哪几卷、什么颜色、哪个引擎、最小比例」有关，所以这些一变，
    键就变，旧的缓存自然失效。
    """
    import hashlib

    digest = hashlib.sha256()
    digest.update(f"triples-1|{engine_id}|{minimum}".encode("utf-8"))
    for filament in sorted(filaments, key=lambda item: item.id):
        digest.update(f"|{filament.id}:{filament.color_hex.upper()}".encode("utf-8"))
    return digest.hexdigest()[:16]


def mix_sidecar_path(path) -> "Path":
    """导出档案旁边的混色缓存：``xxx.json`` → ``xxx.mixes.npz``。"""
    from pathlib import Path

    target = Path(path)
    return target.with_name(f"{target.stem}.mixes.npz")


def save_triple_cache(catalog: "TripleCatalog", path) -> None:
    """原子地写一份缓存（先写临时文件再改名）。"""
    import os
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    payload = dict(catalog.dumps())
    # 记下这套数组属于哪个耗材库，导入时才敢认。
    payload["key"] = np.array(
        triple_cache_key(catalog.filaments, catalog.engine.id)
    )
    with temporary.open("wb") as handle:
        np.savez(handle, **payload)
    os.replace(temporary, target)
    if target.name.startswith(TRIPLE_CACHE_PREFIX):
        prune_triple_caches(target.parent)


def prune_triple_caches(directory, keep: int = TRIPLE_CACHE_KEEP) -> "list[str]":
    """只留最近的 ``keep`` 份三色缓存，其余的删掉。

    41 卷一份缓存约 153 MB，而缓存名是「耗材库 + 引擎」的指纹：加一卷或
    改一个颜色就换一个键，旧文件再也没人读得到。不清理的话每编辑一次
    就多占 153 MB，所以每存一份新的就顺手扫一遍目录。
    """
    from pathlib import Path

    folder = Path(directory)
    if not folder.is_dir():
        return []
    removed: list[str] = []
    for stale in folder.glob(f"{TRIPLE_CACHE_PREFIX}*.tmp"):
        try:
            stale.unlink()
            removed.append(stale.name)
        except OSError:  # pragma: no cover - another process may hold it
            pass
    caches = sorted(
        (
            item
            for item in folder.glob(f"{TRIPLE_CACHE_PREFIX}*.npz")
            if item.is_file()
        ),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for old in caches[max(int(keep), 1):]:
        try:
            old.unlink()
            removed.append(old.name)
        except OSError:  # pragma: no cover - another process may hold it
            pass
    return removed


def load_triple_cache(path, filaments: Sequence, engine: str = DEFAULT_ENGINE):
    """读缓存；文件不存在、损坏或不属于这套耗材时安静地返回 ``None``。

    一定要核对指纹：加一卷耗材之后缓存文件名虽然变了，但旧文件还在，
    万一被搬到新名字上（比如边算边改耗材库），颜色就会张冠李戴。
    """
    from pathlib import Path

    target = Path(path)
    if not target.is_file():
        return None
    try:
        with np.load(target) as data:
            if "key" in data.files:
                if str(data["key"].item()) != triple_cache_key(filaments, engine):
                    return None
            return TripleCatalog.loads(filaments, data, engine=engine)
    except Exception:
        return None


def _tuple_lab(lab) -> tuple[float, float, float]:
    return (float(lab[0]), float(lab[1]), float(lab[2]))

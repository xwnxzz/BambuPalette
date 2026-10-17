"""把模型的颜色配成能打印的混色。

模型不是一个平面，颜色也不会像图片那样一个像素一个像素地排开；这里把模型里出现
的颜色排成一张很小的「色卡」——每种颜色一条竖带，宽度按它在模型里的面积分——然后
交给 :func:`app.core.image_matching.match_image`。合并相近色、耗材上限、方案评分
全是同一套代码，所以模型页与图像转换页给出的判据是同一个，不是两套近似算法。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from ..spectral import color as _color
from .glb import ModelColour
from .image_matching import (
    MatchResult,
    MatchSettings,
    PaletteEntry,
    match_fidelity,
    match_image,
    ProgressFn,
)

#: 色卡宽度。够宽让各色的占比分得开，又不至于让匹配变慢。
CHART_WIDTH = 1024
#: 色卡高度只是为了让「平滑半径」有上下邻居可看，模型本身不需要二维。
CHART_HEIGHT = 9


@dataclass(frozen=True)
class ModelColourMatch:
    """模型里的一种颜色和它的匹配结果。"""

    rgb: tuple[int, int, int]
    color_hex: str
    share: float
    delta_e: float
    entry: PaletteEntry | None

    @property
    def matched_hex(self) -> str:
        return self.entry.color_hex if self.entry is not None else self.color_hex

    @property
    def matched_rgb(self) -> tuple[int, int, int]:
        return (
            tuple(int(v) for v in self.entry.rgb) if self.entry is not None else self.rgb
        )

    @property
    def recipe(self) -> str:
        return self.entry.label if self.entry is not None else "没有匹配到"


@dataclass
class ModelMatch:
    """整只模型的配色结果。"""

    colours: list[ModelColourMatch] = field(default_factory=list)
    entries: list[PaletteEntry] = field(default_factory=list)
    mean_delta_e: float = 0.0
    worst_delta_e: float = 0.0
    spools: tuple[str, ...] = ()
    result: MatchResult | None = None

    @property
    def spool_count(self) -> int:
        return len(self.spools)


def chart_image(
    colours: list[ModelColour], *, width: int = CHART_WIDTH, height: int = CHART_HEIGHT
) -> tuple[Image.Image, list[tuple[int, int, int]]]:
    """把颜色按面积摊成一张色卡，返回（图，每种颜色的列区间）。

    宽度至少一列，保证「模型里只出现一两个像素角」的颜色也有自己的位置；最后一列
    的余数补给面积最大的那一种，免得四舍五入把宽度吃掉。
    """
    if not colours:
        raise ValueError("模型里没有颜色")
    areas = np.asarray([max(0.0, float(item.area)) for item in colours], dtype=np.float64)
    if areas.sum() <= 0:
        areas = np.ones_like(areas)
    exact = areas / areas.sum() * float(width)
    spans = np.maximum(1, np.round(exact).astype(np.int64))
    spans[int(np.argmax(areas))] += int(width) - int(spans.sum())
    spans = np.maximum(1, spans)
    data = np.zeros((height, int(spans.sum()), 3), dtype=np.uint8)
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for item, span in zip(colours, spans):
        stop = cursor + int(span)
        data[:, cursor:stop, :] = np.asarray(item.rgb, dtype=np.uint8)
        ranges.append((cursor, stop))
        cursor = stop
    return Image.fromarray(data, "RGB"), ranges


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*(int(max(0, min(255, c))) for c in rgb))


def match_model(
    colours: list[ModelColour],
    palette: list[PaletteEntry],
    settings: MatchSettings | None = None,
    *,
    progress: ProgressFn | None = None,
) -> ModelMatch:
    """给模型的每种颜色找一套能打印的混色。"""
    config = (settings or MatchSettings()).clamped()
    picture, ranges = chart_image(colours)
    config = MatchSettings(
        max_colours=config.max_colours,
        max_dimension=max(64, picture.width),
        alpha_threshold=config.alpha_threshold,
        dither=False,  # 色卡不需要抖动：每一列本来就是纯色
        merge_delta_e=config.merge_delta_e,
        smooth_radius=0,  # 带子之间有硬边界，平滑只会把颜色洗淡
        max_spools=config.max_spools,
    ).clamped()
    result = match_image(picture, palette, config, progress=progress)

    total_area = float(sum(max(0.0, item.area) for item in colours)) or 1.0
    matches: list[ModelColourMatch] = []
    row = picture.height // 2
    for item, (start, stop) in zip(colours, ranges):
        column = min(picture.width - 1, (start + stop - 1) // 2)
        index = int(result.indices[row, column])
        entry = result.palette[index] if 0 <= index < len(result.palette) else None
        distance = (
            float(_color.delta_e_2000(_color.lab_from_rgb(item.rgb), entry.lab))
            if entry is not None
            else 0.0
        )
        matches.append(
            ModelColourMatch(
                rgb=tuple(int(v) for v in item.rgb),
                color_hex=_hex(item.rgb),
                share=float(item.area) / total_area,
                delta_e=distance,
                entry=entry,
            )
        )

    average, worst = match_fidelity(result)
    used: list[PaletteEntry] = []
    seen: set[str] = set()
    for match in sorted(matches, key=lambda item: -item.share):
        if match.entry is None or match.entry.key in seen:
            continue
        seen.add(match.entry.key)
        used.append(match.entry)
    spools: list[str] = []
    for entry in used:
        for filament_id in entry.filament_ids:
            if filament_id not in spools:
                spools.append(filament_id)
    return ModelMatch(
        colours=matches,
        entries=used,
        mean_delta_e=float(average),
        worst_delta_e=float(worst),
        spools=tuple(spools),
        result=result,
    )


def summarise(match: ModelMatch, names: dict[str, str]) -> str:
    """一句话说明这次配色（给状态行用）。"""
    if not match.colours:
        return "模型里没有颜色。"
    spools = "、".join(names.get(value, value) for value in match.spools) or "（无）"
    return (
        f"{len(match.colours)} 种颜色，平均 ΔE00 {match.mean_delta_e:.2f}、"
        f"最差 {match.worst_delta_e:.2f}，用 {match.spool_count} 卷：{spools}"
    )

"""图片转矢量：把一张位图描成 SVG（1.6.64，m29374）。

这条流水线有四个刻意的顺序，缺一个效果就会垮：

1. **先放大再量化**。240 px 的图标上，眼珠只有三四个像素，量化时必然跟旁边的灰融成
   一片，描出来眼睛就没了。先按 LANCZOS 放大到长边 :data:`TARGET_LONG_EDGE`（最多
   :data:`MAX_UPSCALE` 倍）再量化，细小形状各自成块；SVG 是矢量，放大不损失精度。
2. **自己定调色板**。聚类用的是按感知加权的 k-means（不是「出现次数前 N 个色」——
   抗锯齿会把一条黑描边拆成几百个近黑像素，每个都只占几千分之一，按次数排序永远进不了
   前 N）。另外对暗部 / 亮部做兜底：源图有暗部而调色板没有，就把暗部均值塞进去。
3. **描摹时分层互不重叠**（vtracer 的 ``cutout``）。用 ``stacked`` 时后画的大色块会把
   先画的小细节盖掉，鼻子和眼珠会凭空消失。
4. **描完把填充色吸回调色板**。vtracer 会把同一片颜色按层次拆成 ``#F2F2F1``、
   ``#F3F3F2``…，一个 8 色图能描出 60 多种填充，看着像脏点；吸回去才是干净的 N 色。

依赖 ``vtracer``（MIT，Rust 的 visioncortex）。它是可选依赖：没装的时候这里抛
:class:`VectorizeUnavailable`，界面给一句人话，不要拖着整个程序崩。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

COLOUR_RANGE = (2, 32)
SPECKLE_RANGE = (0, 40)
SMOOTH_RANGE = (1.0, 12.0)

#: 长边放大到这个像素数再描（够细，又不至于慢）。
TARGET_LONG_EDGE = 1200
#: 小图放大倍数上限，避免 16 px 的小图标被拉成 1200 px 的糊块。
MAX_UPSCALE = 6.0

_FILL = re.compile(r'fill="(#[0-9A-Fa-f]{6})"')

DEFAULT_COLOURS = 12
DEFAULT_SPECKLE = 6
DEFAULT_SMOOTH = 6.0


class VectorizeUnavailable(RuntimeError):
    """没装 vtracer 之类的可选依赖时抛出，界面据此给一句人话。"""


@dataclass(frozen=True)
class TraceResult:
    """一次描摹的产出与统计（界面拿它写状态行）。"""

    path: Path
    colours: int
    paths: int
    bytes: int


def _require_vtracer():
    try:
        import vtracer  # noqa: PLC0415 - 可选依赖，用到才导入
    except ModuleNotFoundError as error:  # pragma: no cover - 只有缺依赖时走到
        raise VectorizeUnavailable(
            "导出 SVG 需要 vtracer（MIT 许可的描摹库），这个环境里没装。\n"
            "装了之后这个按钮就能用：pip install vtracer"
        ) from error
    return vtracer


def available() -> bool:
    """vtracer 装没装（界面据此把按钮标灰，而不是点下去才报错）。"""

    try:
        import vtracer  # noqa: F401,PLC0415 - 只是探一下依赖在不在
    except ModuleNotFoundError:  # pragma: no cover - 只有缺依赖时走到
        return False
    return True


def _prepared_image(source: Path):
    """读入、拍平透明底、必要时放大到长边 :data:`TARGET_LONG_EDGE`。"""

    from PIL import Image

    image = Image.open(source)
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGBA")
        flat = Image.new("RGBA", image.size, (255, 255, 255, 255))
        flat.alpha_composite(image)
        image = flat
    image = image.convert("RGB")

    longest = max(image.size)
    if longest < TARGET_LONG_EDGE:
        scale = min(MAX_UPSCALE, TARGET_LONG_EDGE / float(longest))
        image = image.resize(
            (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale)))),
            Image.LANCZOS,
        )
    return image


def _luminance(rgb) -> float:
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def _kmeans_palette(pixels, colours: int, *, iterations: int = 12, seed: int = 0):
    """按感知加权聚出 ``colours`` 个颜色（numpy 实现的 k-means++）。"""

    import numpy as np

    rng = np.random.default_rng(seed)
    flat = pixels.reshape(-1, 3).astype(np.float32)
    sample = flat
    if flat.shape[0] > 120_000:
        pick = rng.choice(flat.shape[0], 120_000, replace=False)
        sample = flat[pick]

    centres = [sample[rng.integers(sample.shape[0])]]
    for _ in range(1, colours):
        known = np.asarray(centres, dtype=np.float32)
        best = np.min(((sample[:, None, :] - known[None, :, :]) ** 2).sum(axis=2), axis=1)
        total = float(best.sum())
        if total <= 0:
            break
        centres.append(sample[int(rng.choice(sample.shape[0], p=best / total))])

    known = np.asarray(centres, dtype=np.float32)
    weight = np.asarray([2.0, 4.0, 3.0], dtype=np.float32)
    for _ in range(iterations):
        delta = sample[:, None, :] - known[None, :, :]
        assign = np.argmin((delta * delta * weight).sum(axis=2), axis=1)
        moved = False
        for index in range(known.shape[0]):
            members = sample[assign == index]
            if members.shape[0] == 0:
                continue
            centre = members.mean(axis=0)
            if float(np.abs(centre - known[index]).max()) > 0.5:
                moved = True
            known[index] = centre
        if not moved:
            break

    return [(int(round(c[0])), int(round(c[1])), int(round(c[2]))) for c in known]


def quantize(source: Path, colours: int, *, target: Path | None = None) -> Path:
    """把图压成恰好 ``colours`` 种颜色，返回那份临时 PNG 的路径。"""

    colours = max(COLOUR_RANGE[0], min(COLOUR_RANGE[1], int(colours)))
    image = _prepared_image(source)

    import numpy as np
    from PIL import Image

    pixels = np.asarray(image, dtype=np.uint8)
    palette = _kmeans_palette(pixels, colours)

    # 暗部 / 亮部兜底：聚类按方差分，暗部面积小的时候会被别的中心整个吃掉（实测猪信
    # 16 色时眼睛又消失）。源图有暗部、调色板里却没有，就把暗部均值补进去。
    flat_source = pixels.reshape(-1, 3)
    source_lum = (
        0.299 * flat_source[:, 0].astype(np.float32)
        + 0.587 * flat_source[:, 1].astype(np.float32)
        + 0.114 * flat_source[:, 2].astype(np.float32)
    )
    for low, high in ((0.0, 64.0), (200.0, 256.0)):
        band = flat_source[(source_lum >= low) & (source_lum < high)]
        if band.shape[0] < 16:
            continue
        if any(low <= _luminance(c) < high for c in palette):
            continue
        mean = band.mean(axis=0)
        slot = min(range(len(palette)), key=lambda i: abs(_luminance(palette[i]) - _luminance(mean)))
        palette[slot] = (int(round(mean[0])), int(round(mean[1])), int(round(mean[2])))

    centres = np.asarray(palette, dtype=np.int32)
    weight = np.asarray([2, 4, 3], dtype=np.int32)
    flat_pixels = pixels.reshape(-1, 3).astype(np.int32)
    out = np.empty_like(flat_pixels)
    step = 1 << 16
    for start in range(0, flat_pixels.shape[0], step):
        block = flat_pixels[start : start + step]
        delta = block[:, None, :] - centres[None, :, :]
        nearest = np.argmin((delta * delta * weight).sum(axis=2), axis=1)
        out[start : start + step] = centres[nearest]
    reduced = Image.fromarray(out.reshape(pixels.shape).astype("uint8"), "RGB")

    out_path = target or source.with_name(f"{source.stem}-q{colours}.png")
    reduced.save(out_path)
    return out_path


def palette_of(image_path: Path, colours: int) -> list[tuple[int, int, int]]:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    found = image.getcolors(maxcolors=1 << 24) or []
    found.sort(reverse=True)
    return [rgb for _count, rgb in found[: max(1, colours)]]


def snap_fills(svg_path: Path, palette: list[tuple[int, int, int]]) -> int:
    """把 SVG 的填充色吸到最近的调色板颜色上，返回改了几处。"""

    if not palette:
        return 0
    text = svg_path.read_text(encoding="utf-8", errors="replace")
    changed = 0

    def nearest(match: re.Match[str]) -> str:
        nonlocal changed
        value = match.group(1)
        rgb = (int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16))
        best = min(
            palette,
            key=lambda c: (c[0] - rgb[0]) ** 2 + (c[1] - rgb[1]) ** 2 + (c[2] - rgb[2]) ** 2,
        )
        if best == rgb:
            return match.group(0)
        changed += 1
        return f'fill="#{best[0]:02X}{best[1]:02X}{best[2]:02X}"'

    svg_path.write_text(_FILL.sub(nearest, text), encoding="utf-8")
    return changed


def image_to_svg(
    source: Path,
    target: Path,
    *,
    colours: int = DEFAULT_COLOURS,
    speckle: int = DEFAULT_SPECKLE,
    smooth: float = DEFAULT_SMOOTH,
    mode: str = "spline",
    stack: str = "cutout",
) -> TraceResult:
    """把 ``source`` 描成 ``target``（SVG），返回统计。

    参数就是界面上那三个滑杆：``colours`` 想要几种颜色、``speckle`` 小于这个面积的斑点
    直接并进邻块（像素，按放大后的图算）、``smooth`` 曲线简化强度（越大路径越少越硬）。
    """

    vtracer = _require_vtracer()
    colours = max(COLOUR_RANGE[0], min(COLOUR_RANGE[1], int(colours)))
    speckle = max(SPECKLE_RANGE[0], min(SPECKLE_RANGE[1], int(speckle)))
    smooth = max(SMOOTH_RANGE[0], min(SMOOTH_RANGE[1], float(smooth)))

    target.parent.mkdir(parents=True, exist_ok=True)
    prepared = quantize(source, colours)
    palette = palette_of(prepared, colours)
    try:
        vtracer.convert_image_to_svg_py(
            str(prepared),
            str(target),
            colormode="color",
            hierarchical=stack,
            mode=mode,
            filter_speckle=speckle,
            # 调色板已经定死，颜色精度拉满只要够区分它们；层次差给小一点，免得两层被并。
            color_precision=8,
            layer_difference=4,
            corner_threshold=60,
            length_threshold=smooth,
            max_iterations=10,
            splice_threshold=45,
            path_precision=3,
        )
    finally:
        prepared.unlink(missing_ok=True)

    snap_fills(target, palette)
    text = target.read_text(encoding="utf-8", errors="replace")
    return TraceResult(
        path=target,
        colours=len(set(_FILL.findall(text))),
        paths=text.count("<path"),
        bytes=target.stat().st_size,
    )

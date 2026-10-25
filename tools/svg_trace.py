"""图片转矢量（命令行版）：把 PNG/JPG 描成 SVG。

真正的实现在 :mod:`app.core.vectorize`（界面上「导出 SVG」那个按钮用的是同一份代码），
这里只是一层薄命令行，方便批量跑或者在没开界面的时候用。

    python tools/svg_trace.py 输入.png 输出.svg --colours 12
    python tools/svg_trace.py 输入.png 输出.svg --colours 16 --speckle 12 --smooth 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.vectorize import (  # noqa: E402
    COLOUR_RANGE,
    DEFAULT_COLOURS,
    DEFAULT_SMOOTH,
    DEFAULT_SPECKLE,
    SMOOTH_RANGE,
    SPECKLE_RANGE,
    VectorizeUnavailable,
    image_to_svg,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="图片转矢量（PNG/JPG → SVG）")
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument(
        "--colours", type=int, default=DEFAULT_COLOURS, help=f"想要几种颜色（{COLOUR_RANGE[0]}–{COLOUR_RANGE[1]}）"
    )
    parser.add_argument(
        "--speckle",
        type=int,
        default=DEFAULT_SPECKLE,
        help=f"去掉碎块：小于这个面积的斑点并进邻块（{SPECKLE_RANGE[0]}–{SPECKLE_RANGE[1]} px）",
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=DEFAULT_SMOOTH,
        help=f"曲线简化强度，越大路径越少（{SMOOTH_RANGE[0]}–{SMOOTH_RANGE[1]}）",
    )
    args = parser.parse_args(argv)

    if not args.source.is_file():
        parser.error(f"找不到 {args.source}")

    try:
        result = image_to_svg(
            args.source,
            args.target,
            colours=args.colours,
            speckle=args.speckle,
            smooth=args.smooth,
        )
    except VectorizeUnavailable as error:
        print(error, file=sys.stderr)
        return 2

    print(f"{result.path}  {result.bytes} 字节  {result.paths} 条路径  {result.colours} 种颜色")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

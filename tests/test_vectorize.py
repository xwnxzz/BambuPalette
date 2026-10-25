"""图片转矢量的单测（1.6.64，m29374）。

量的都是「描出来到底对不对」这件事：调色板够不够、暗部丢没丢、SVG 里是不是真有路径、
参数极值会不会炸。
"""

from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from app.core.vectorize import (
    COLOUR_RANGE,
    DEFAULT_COLOURS,
    VectorizeUnavailable,
    image_to_svg,
    palette_of,
    quantize,
)


def flat_icon(path: Path, size: int = 64) -> Path:
    """画一个「绿底 + 白猪身 + 黑眼睛」的小图：正好覆盖「大色块 + 小暗点」两种情况。"""

    image = Image.new("RGB", (size, size), (0, 200, 100))
    draw = ImageDraw.Draw(image)
    draw.ellipse((10, 14, size - 12, size - 10), fill=(245, 245, 245))
    draw.ellipse((22, 30, 27, 35), fill=(20, 20, 20))
    draw.ellipse((40, 30, 45, 35), fill=(20, 20, 20))
    image.save(path)
    return path


class QuantizeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._folder = tempfile.TemporaryDirectory()
        self.root = Path(self._folder.name)

    def tearDown(self) -> None:
        self._folder.cleanup()

    def test_it_keeps_exactly_the_colours_asked_for(self) -> None:
        source = flat_icon(self.root / "icon.png")
        for wanted in (3, 5, 8):
            reduced = quantize(source, wanted)
            colours = {rgb for _count, rgb in Image.open(reduced).convert("RGB").getcolors(1 << 24)}
            self.assertLessEqual(len(colours), wanted, f"要 {wanted} 色，出了 {len(colours)} 色")
            self.assertGreaterEqual(len(colours), 2)

    def test_small_dark_details_do_not_get_eaten(self) -> None:
        """眼睛只占千分之几：聚类可能把它整个吃掉，所以要看到调色板里留了暗色。"""

        source = flat_icon(self.root / "icon.png")
        reduced = quantize(source, 6)
        colours = [rgb for _count, rgb in Image.open(reduced).convert("RGB").getcolors(1 << 24)]
        darkest = min(0.299 * r + 0.587 * g + 0.114 * b for r, g, b in colours)
        self.assertLess(darkest, 90.0, f"调色板里没有暗色：{sorted(colours)[:4]}")

    def test_upscaling_happens_before_quantizing(self) -> None:
        """240 px 的小图会被拉到长边 1200 再量化（细节靠这一步保住）。"""

        source = flat_icon(self.root / "icon.png", size=240)
        reduced = quantize(source, 4)
        self.assertGreaterEqual(max(Image.open(reduced).size), 1000)

    def test_tiny_images_are_not_blown_up_without_limit(self) -> None:
        """太小的图最多放大 6 倍 —— 不然 16 px 的图标会被拉成一团糊。"""

        source = flat_icon(self.root / "tiny.png", size=60)
        reduced = quantize(source, 4)
        self.assertEqual(max(Image.open(reduced).size), 360)

    def test_it_flattens_transparency_onto_white(self) -> None:
        source = self.root / "alpha.png"
        image = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
        ImageDraw.Draw(image).rectangle((8, 8, 30, 30), fill=(200, 30, 30, 255))
        image.save(source)
        reduced = quantize(source, 3)
        corner = Image.open(reduced).convert("RGB").getpixel((1, 1))
        self.assertGreater(min(corner), 200, f"透明底没拍平到白：{corner}")


class TraceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._folder = tempfile.TemporaryDirectory()
        self.root = Path(self._folder.name)

    def tearDown(self) -> None:
        self._folder.cleanup()

    def test_a_flat_icon_traces_into_paths(self) -> None:
        source = flat_icon(self.root / "icon.png")
        target = self.root / "icon.svg"
        result = image_to_svg(source, target, colours=6, speckle=4, smooth=4.0)
        self.assertTrue(target.is_file())
        self.assertGreater(result.paths, 0)
        self.assertGreater(result.bytes, 200)
        text = target.read_text(encoding="utf-8")
        self.assertIn("<svg", text)
        self.assertIn("<path", text)

    def test_the_number_of_fills_matches_the_palette_size(self) -> None:
        source = flat_icon(self.root / "icon.png")
        target = self.root / "icon.svg"
        result = image_to_svg(source, target, colours=6, speckle=4, smooth=4.0)
        fills = set(re.findall(r'fill="(#[0-9A-Fa-f]{6})"', target.read_text(encoding="utf-8")))
        self.assertLessEqual(len(fills), 6)
        self.assertEqual(result.colours, len(fills))

    def test_the_cli_limits_are_enforced(self) -> None:
        source = flat_icon(self.root / "icon.png")
        target = self.root / "icon.svg"
        # 超范围的参数要被夹回范围内，而不是抛异常
        result = image_to_svg(source, target, colours=999, speckle=-5, smooth=99.0)
        self.assertGreater(result.paths, 0)
        self.assertLessEqual(result.colours, COLOUR_RANGE[1])

    def test_defaults_are_in_the_middle_of_the_useful_range(self) -> None:
        self.assertGreaterEqual(DEFAULT_COLOURS, 8)
        self.assertLessEqual(DEFAULT_COLOURS, COLOUR_RANGE[1])

    def test_missing_vtracer_gives_a_readable_error(self) -> None:
        """没装依赖时要抛一句人话（界面据此提示），而不是 ModuleNotFoundError。"""

        import builtins

        source = flat_icon(self.root / "icon.png")
        original = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "vtracer":
                raise ModuleNotFoundError("No module named 'vtracer'")
            return original(name, *args, **kwargs)

        builtins.__import__ = fake_import
        try:
            with self.assertRaises(VectorizeUnavailable):
                image_to_svg(source, self.root / "icon.svg")
        finally:
            builtins.__import__ = original


if __name__ == "__main__":
    unittest.main()

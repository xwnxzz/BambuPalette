"""Render the main window off-screen so the layout can be reviewed as a PNG.

Usage::

    .venv\\Scripts\\python.exe tools\\screenshot_ui.py

Writes ``samples/ui-main.png`` (browse mode) and ``samples/ui-pair.png``
(one pair filtered to its 81 ratios).  Nothing touches the user's real library:
the storage paths are redirected to a temporary folder.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app.core import paths  # noqa: E402

TEMP = Path(tempfile.mkdtemp(prefix="fcs-ui-"))
paths.library_path = lambda: TEMP / "filament-library.json"  # type: ignore[assignment]
paths.data_dir = lambda: TEMP  # type: ignore[assignment]

from app.core.library import Filament, FilamentLibrary  # noqa: E402
from app.main import build_application  # noqa: E402
from app.ui.main_window import MainWindow  # noqa: E402

SAMPLES = (
    ("大简 PETG HF 白", "大简", "PETG HF", "#F2F0EB"),
    ("大简 PETG HF 黑", "大简", "PETG HF", "#17181C"),
    ("大简 PETG HF 金", "大简", "PETG HF", "#D9A441"),
    ("大简 PETG HF 青", "大简", "PETG HF", "#2FA8B8"),
    ("大简 PETG HF 红", "大简", "PETG HF", "#C8342E"),
)

OUT = ROOT / "samples"


def main() -> int:
    app = build_application([])
    library = FilamentLibrary([
        Filament(name=name, brand=brand, material_type=material, color_hex=value)
        for name, brand, material, value in SAMPLES
    ])
    window = MainWindow(library=library)
    window.resize(1460, 900)
    window.show()
    app.processEvents()

    OUT.mkdir(parents=True, exist_ok=True)
    print("recipes:", len(window._recipes), "pairs:", window._catalog.pair_count)
    print("engine:", window._catalog.engine.id, window._catalog.engine.name)
    print("first:", window._recipes[0].color_hex, window._recipes[0].ratio_text)

    window._grid.selectRecipe(window._recipes[len(window._recipes) // 2])
    app.processEvents()
    window.grab().save(str(OUT / "ui-main.png"))

    window._show_pair(library[0].id, library[2].id)
    app.processEvents()
    window._grid.selectRecipe(window._recipes[40])
    app.processEvents()
    window.grab().save(str(OUT / "ui-pair.png"))

    print("selected:", window._detail.recipe().color_hex, window._detail.recipe().ratio_text)
    print("wrote", OUT / "ui-main.png")
    print("wrote", OUT / "ui-pair.png")

    # Picture page: import a generated sample, match it, and grab the result.
    sample = ROOT / "samples" / "sample-picture.png"
    _make_sample(sample)
    page = window._picture_page
    window._tabs.setCurrentIndex(1)
    page.loadImage(sample)
    app.processEvents()
    result = page.result
    assert result is not None, "the sample picture did not match"
    print("picture:", f"{result.width}x{result.height}", "colours:", len(result.palette))
    # Click a colour in the list, the way the user would.
    page._list.pin(len(result.palette) - 1)
    page._view.setSelected(len(result.palette) - 1)
    app.processEvents()
    window.grab().save(str(OUT / "ui-picture.png"))
    print("wrote", OUT / "ui-picture.png")

    page.buildPlate()
    print("plate:", page.plate.stats())
    print("notes:", page.plate.notes)
    return 0


def _make_sample(target: Path) -> None:
    """A small picture with flat colour fields and one soft gradient."""
    if target.exists():
        return
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (320, 240), "#F2F0EB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((16, 16, 150, 130), fill="#C8342E")
    draw.ellipse((170, 20, 300, 120), fill="#D9A441")
    draw.polygon([(40, 220), (160, 150), (280, 220)], fill="#2FA8B8")
    for x in range(16, 300):
        shade = int(32 + (x - 16) / 284 * 200)
        draw.line([(x, 132), (x, 145)], fill=(shade, shade, shade))
    image.save(target)


if __name__ == "__main__":
    raise SystemExit(main())

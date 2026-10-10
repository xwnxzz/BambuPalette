"""Turn the 大简 PETG HF colour list into a BambuPalette import file.

Run once; the generated JSON is what the app's 「导入耗材档案…」 reads.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.library import LIBRARY_FORMAT_VERSION, _now  # noqa: E402

BRAND = "大简"
MATERIAL = "PETG HF"

# (name, r, g, b, unsure)  -- the colour values are exactly as supplied; the two
# marked unsure carried a "?" in the source list and only their note records it.
SPOOLS = [
    ("白色", 247, 243, 243, False),
    ("黑色", 44, 42, 44, False),
    ("灰色", 164, 163, 163, False),
    ("天蓝色", 96, 178, 222, False),
    ("橙色", 255, 108, 43, False),
    ("透明", 150, 150, 150, False),
    ("透明紫", 136, 125, 149, True),
    ("透明绿", 152, 177, 89, False),
    ("透明蓝", 82, 120, 150, False),
    ("绿色", 109, 199, 43, False),
    ("松石绿", 10, 162, 164, False),
    ("苹果绿", 193, 204, 0, False),
    ("青色", 4, 126, 186, False),
    ("蓝色", 33, 113, 184, False),
    ("深蓝色", 42, 59, 128, False),
    ("樱花粉", 229, 183, 191, True),
    ("柠檬黄", 186, 177, 55, False),
    ("玫瑰红", 216, 75, 131, False),
    ("红色", 197, 42, 54, False),
    ("香芋紫", 207, 177, 219, False),
    ("紫罗兰", 84, 90, 165, False),
    ("绀紫色", 58, 43, 82, False),
    ("紫色", 107, 104, 183, False),
    ("肤色", 245, 210, 184, False),
    ("暗金色", 166, 130, 88, False),
    ("拿铁色", 176, 146, 117, False),
    ("粉红色", 128, 97, 128, False),
    ("棕色", 131, 75, 66, False),
    ("青铜色", 122, 103, 69, False),
    ("金色", 196, 150, 75, False),
    ("米白色", 231, 223, 209, False),
    ("玫紫色", 182, 60, 138, False),
    ("桃红色", 214, 81, 139, False),
    ("黄色", 242, 208, 46, False),
    ("杏色", 223, 200, 193, False),
    ("深灰色", 94, 93, 94, False),
    ("银色", 138, 136, 138, False),
    ("透明粉", 172, 147, 152, False),
    ("薄荷蓝", 188, 218, 219, False),
    ("蓝灰色", 97, 107, 120, False),
    ("草绿色", 150, 187, 42, False),
]

UNCERTAIN_NOTE = "色值待确认：原始清单里带问号"


def build() -> dict:
    stamp = _now()
    filaments = []
    for index, (name, r, g, b, unsure) in enumerate(SPOOLS, start=1):
        filaments.append(
            {
                "id": f"dajian-petg-hf-{index:02d}",
                "name": name,
                "brand": BRAND,
                "materialType": MATERIAL,
                "colorHex": f"#{r:02X}{g:02X}{b:02X}",
                "note": UNCERTAIN_NOTE if unsure else "",
                "createdAt": stamp,
                "updatedAt": stamp,
            }
        )
    return {
        "formatVersion": LIBRARY_FORMAT_VERSION,
        "savedAt": stamp,
        "note": f"{BRAND} {MATERIAL} 官方色卡，共 {len(filaments)} 色",
        "filaments": filaments,
    }


def main() -> int:
    payload = build()
    out_dir = ROOT / "presets"
    out_dir.mkdir(exist_ok=True)
    target = out_dir / "大简-PETG-HF-耗材档案.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("wrote", target, target.stat().st_size, "B")

    # Read it back through the real loader, so "the app can import it" is a
    # measured fact rather than a hope.
    from app.core.library import FilamentLibrary

    library = FilamentLibrary.load(target)
    print("loaded", len(library), "spools")
    seen = set()
    for filament in library.sort_by_color():
        seen.add(filament.color_hex)
    print("distinct colours", len(seen))
    bad = [f.name for f in library if len(f.color_hex) != 7]
    print("bad hex", bad)
    assert len(library) == len(SPOOLS), "a spool went missing in the round trip"
    assert not bad, bad
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""把耗材库导成 Blender 能直接拖进去的样板（m26980 第 1 条）。

脚本那条路（``material_script``）在 Blender 里要自己开文本编辑器、粘路径、按运行，
用户嫌麻烦；直接拖一个文件进去才是他要的。Blender 拖放能吃下 ``.obj``（连同同目录
的 ``.mtl``），导入后就有一个小方块代表这卷耗材，材质名字就是「编号 名称 颜色」，
换色、看色都直接用它。

几何很简单：每卷耗材一个小方片（10 × 10 个模型单位，彼此留 4 的间隔），按十列排开；
每个方片用自己的 ``usemtl``。``.mtl`` 的 ``Kd`` 写 0–1 的 sRGB 分量 —— 这是各家
``.mtl`` 的通行写法，Blender 导入后 base color 就是界面里看到的那个颜色。
"""

from __future__ import annotations

from pathlib import Path

#: 方片边长与间距（OBJ 单位）。
TILE = 10.0
GAP = 4.0
#: 一行放几块。
COLUMNS = 10


def _hex_to_rgb(color_hex: str) -> tuple[float, float, float]:
    value = (color_hex or "#000000").lstrip("#")
    if len(value) != 6:
        value = "000000"
    return tuple(int(value[index : index + 2], 16) / 255.0 for index in (0, 2, 4))


def _safe(name: str) -> str:
    """OBJ / MTL 里名字不能带空格（空格是分隔符）。"""
    cleaned = "".join(char if char.isalnum() or char in "-_.()#" else "_" for char in name)
    return cleaned or "material"


def _material_name(index: int, entry) -> str:
    """材质名：序号 + 耗材名（名字里已经带十六进制色号，不再重复一遍）。"""
    label = (getattr(entry, "short_label", "") or entry.color_hex).replace("#", "")
    label = label.replace("%", "pct")
    return _safe(f"{index:02d}_{label}").rstrip("_")


def swatch_mtl(entries, *, title: str = "BambuPalette") -> str:
    """``.mtl`` 文本：每卷耗材一条 ``newmtl``。"""
    lines = [
        f"# {title}：耗材库材质样板（BambuPalette 导出）",
        "# 每个材质名 = 序号 + 耗材名 + 颜色，用它给模型上色即可。",
        "",
    ]
    for index, entry in enumerate(entries, start=1):
        red, green, blue = _hex_to_rgb(entry.color_hex)
        name = _material_name(index, entry)
        lines.extend(
            [
                f"newmtl {name}",
                f"Kd {red:.6f} {green:.6f} {blue:.6f}",
                "Ka 0.000000 0.000000 0.000000",
                "Ks 0.000000 0.000000 0.000000",
                "d 1.0",
                "illum 1",
                "",
            ]
        )
    return "\n".join(lines)


def swatch_obj(entries, *, title: str = "BambuPalette") -> str:
    """``.obj`` 文本：每卷耗材一块小方片，各自 ``usemtl``。"""
    lines = [
        f"# {title}：耗材库材质样板（BambuPalette 导出）",
        "# 每块方片 = 一卷耗材；材质名字见同目录的 .mtl",
        "",
    ]
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int, int, str]] = []
    for index, entry in enumerate(entries):
        column = index % COLUMNS
        row = index // COLUMNS
        left = column * (TILE + GAP)
        top = -row * (TILE + GAP)
        base = len(vertices) + 1
        vertices.extend(
            [
                (left, top, 0.0),
                (left + TILE, top, 0.0),
                (left + TILE, top - TILE, 0.0),
                (left, top - TILE, 0.0),
            ]
        )
        name = _material_name(index + 1, entry)
        faces.append((base, base + 1, base + 2, base + 3, name))

    for x, y, z in vertices:
        lines.append(f"v {x:.4f} {y:.4f} {z:.4f}")
    lines.append("")
    for a, b, c, d, name in faces:
        lines.append(f"usemtl {name}")
        lines.append(f"f {a} {b} {c} {d}")
    lines.append("")
    return "\n".join(lines)


def write_swatch(path: str | Path, entries) -> tuple[Path, Path]:
    """写一对 ``.obj`` + ``.mtl``，返回两个路径（``.mtl`` 与 obj 同名）。"""
    target = Path(path)
    if target.suffix.lower() != ".obj":
        target = target.with_suffix(".obj")
    mtl = target.with_suffix(".mtl")
    rows = list(entries)
    target.write_text(swatch_obj(rows), encoding="utf-8")
    mtl.write_text(swatch_mtl(rows), encoding="utf-8")
    return target, mtl

"""把配色结果交给 Blender：能直接产出 ``.blend``，找不到 Blender 就退成脚本。

1.6.57（m26604 第 7 / 9 条）：用户希望「导出的东西 Blender 能直接用」。OBJ + MTL 与
GLB 本来就能导入，但真正要「一个 .blend 文件」就只能让 Blender 自己写 —— 这个格式是
Blender 的内存镜像（BHead + DNA1），外面拿 Python 拼不出可靠的版本。

所以这里做两件事：

* :func:`find_blender` 找本机的 ``blender.exe``（环境变量、PATH、常见安装目录、设置里
  手填的路径都看）；
* :func:`write_blend` 用 ``blender --background --python`` 跑一段脚本，把 OBJ 导进去、
  存成 ``.blend``；
* 找不到 Blender 时 :func:`write_blend_script` 写一份 ``.py``：在 Blender 的文本编辑器里
  跑一下，得到的是同一个场景。宁可给一条能走通的路，也不要假装写了 .blend。
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

#: 常见安装位置（Windows）。只按从新到旧排，不写死版本号。
_BLENDER_GLOBS = (
    r"C:\Program Files\Blender Foundation\Blender*\blender.exe",
    r"C:\Program Files (x86)\Blender Foundation\Blender*\blender.exe",
    "/Applications/Blender.app/Contents/MacOS/Blender",
    "/usr/bin/blender",
    "/usr/local/bin/blender",
)


def find_blender(configured: str | None = None) -> Path | None:
    """找到本机的 Blender 可执行文件；找不到返回 ``None``。

    ``configured`` 是「设置」里手填的路径（可以指向 blender.exe 或它所在的目录）。
    """
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_dir():
            for name in ("blender.exe", "Blender", "blender"):
                inside = candidate / name
                if inside.is_file():
                    return inside
        if candidate.is_file():
            return candidate

    for name in ("blender", "blender.exe"):
        found = shutil.which(name)
        if found:
            return Path(found)

    for pattern in _BLENDER_GLOBS:
        try:
            if any(char in pattern for char in "*"):
                root = Path(pattern).anchor or Path(pattern).drive or "/"
                relative = pattern[len(root) :] if root != "/" else pattern.lstrip("/\\")
                matches = sorted(Path(root).glob(relative), reverse=True)
                for match in matches:
                    if match.is_file():
                        return match
            else:
                candidate = Path(pattern)
                if candidate.is_file():
                    return candidate
        except (OSError, ValueError, NotImplementedError):
            continue
    return None


def write_blend(obj_path: Path, blend_path: Path, *, blender: Path | None = None) -> Path:
    """用 Blender 把 ``obj_path``（连同一个 .mtl）导入并存成 ``blend_path``。

    行不通时抛 ``RuntimeError``，由调用方决定怎么告诉用户。
    """
    executable = blender or find_blender()
    if executable is None:
        raise RuntimeError("没找到 Blender")
    blend_path = blend_path.with_suffix(".blend")
    script = (
        "import bpy, sys\n"
        f"bpy.ops.wm.read_factory_settings(use_empty=True)\n"
        f"bpy.ops.wm.obj_import(filepath={str(obj_path)!r})\n"
        f"bpy.ops.wm.save_as_mainfile(filepath={str(blend_path)!r})\n"
    )
    command = [str(executable), "--background", "--factory-startup",
               "--python-expr", script]
    finished = subprocess.run(command, capture_output=True, text=True, timeout=600)
    if finished.returncode != 0 or not blend_path.is_file():
        detail = (finished.stderr or finished.stdout or "").strip().splitlines()
        reason = detail[-1] if detail else f"退出码 {finished.returncode}"
        raise RuntimeError(f"Blender 没能写出 {blend_path.name}：{reason}")
    return blend_path


def write_blend_script(
    obj_path: Path, script_path: Path, *, note: str = ""
) -> Path:
    """找不到 Blender 时：写一份能把它变成 ``.blend`` 的脚本。

    用户装好 Blender 后在「文本编辑器」里打开它、按「运行脚本」，得到的就是同一个场景。
    """
    script_path = script_path.with_suffix(".py")
    text = (
        '"""在 Blender 里运行这个脚本：导入下面那个 OBJ（颜色在旁边的 MTL 里），'
        "再另存为 .blend。\n\n"
        f"OBJ：{obj_path}\n"
        + (f"{note}\n" if note else "")
        + '"""\n\n'
        "import bpy\n\n"
        f"bpy.ops.wm.obj_import(filepath={str(obj_path)!r})\n"
    )
    script_path.write_text(text, encoding="utf-8")
    return script_path


def material_script(entries, *, title: str = "BambuPalette 耗材") -> str:
    """一份在 Blender 里跑出这些耗材材质的脚本（m26604 第 7 条）。

    ``entries`` 是任何带 ``color_hex`` 与 ``name`` / ``short_label`` 的东西（耗材、混色条目
    都行）。颜色按 sRGB 写进去 —— Blender 的节点颜色是线性空间，所以脚本里做一次反 gamma，
    这样材质球在 Blender 里看到的颜色跟这里看到的一致。
    """
    lines = [
        '"""BambuPalette 耗材材质：在 Blender 的文本编辑器里运行这个脚本。',
        "",
        "会按名字建出这些材质的 Principled BSDF，颜色与这里一致（已转成线性空间）。",
        '"""',
        "",
        "import bpy",
        "",
        "",
        "def _linear(channel: int) -> float:",
        '    """sRGB 0-255 → 线性 0-1。"""',
        "    value = channel / 255.0",
        "    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4",
        "",
        "",
        "MATERIALS = [",
    ]
    for entry in entries:
        colour = str(getattr(entry, "color_hex", "#000000")).lstrip("#")[:6]
        name = str(getattr(entry, "short_label", None) or getattr(entry, "name", colour))
        try:
            red, green, blue = (int(colour[index : index + 2], 16) for index in (0, 2, 4))
        except ValueError:
            continue
        lines.append(f"    ({name!r}, ({red}, {green}, {blue})),")
    lines += [
        "]",
        "",
        "for name, (red, green, blue) in MATERIALS:",
        "    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)",
        "    material.use_nodes = True",
        "    shader = material.node_tree.nodes.get('Principled BSDF')",
        "    if shader is not None:",
        "        shader.inputs['Base Color'].default_value = (",
        "            _linear(red), _linear(green), _linear(blue), 1.0",
        "        )",
        "        shader.inputs['Roughness'].default_value = 0.55",
        "",
        "print('BambuPalette：建好 %d 个材质' % len(MATERIALS))",
    ]
    return "\n".join(lines) + "\n"


def describe_missing_blender() -> str:
    """找不到 Blender 时给用户看的一句话。"""
    return (
        "没找到 Blender。.blend 是 Blender 自己的存档格式，只有 Blender 能写。\n"
        "装一个 Blender（或在「设置 → 模型配色 → Blender 路径」里填上 blender.exe），"
        "再导出一次就能直接得到 .blend；这次已经给你写了一份等价的脚本，"
        "在 Blender 的文本编辑器里跑一下也一样。"
    )

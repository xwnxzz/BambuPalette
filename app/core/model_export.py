"""把「模型配色」的结果变成能打印的文件：3MF / OBJ + MTL，以及带配色的 GLB。

三条出口共用一件事：**模型本来没有像素网格**，所以这里把它的三角形原样搬进
:class:`app.mesh.plate.PlateModel`，每个三角面按「这块料配的是哪种颜色」归到一个零件
里。3MF 与 OBJ 那条路就此和图像转换页共用同一个导出器（``write_3mf`` / ``write_obj``），
混色耗材、耗材槽位、paint_color 编码都不用再写一遍。

坐标：glTF 的单位是米、Y 轴朝上；切片器和 3MF 都要毫米、Z 轴朝上、模型站在 z=0、底面
中心落在原点（不然 Bambu Studio 打开的件在盘外）。所以导出前一律做
``(x, -z, y) × 1000``、再把包围盒底面的中心挪到原点。
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from ..mesh.plate import PlateModel, PlatePart
from .glb import GlbModel

__all__ = [
    "MM_PER_UNIT",
    "display_points",
    "plate_from_model",
    "glb_bytes",
    "write_glb",
]

#: glTF 的长度单位是米，切片器用毫米。
MM_PER_UNIT = 1000.0


def display_points(points: np.ndarray) -> np.ndarray:
    """glTF 的 Y 朝上 → 模型的 Z 朝上（与 :mod:`app.ui.model_view` 同一套换算）。"""
    if points.size == 0:
        return points
    out = np.empty_like(points, dtype=np.float64)
    out[:, 0] = points[:, 0]
    out[:, 1] = -points[:, 2]
    out[:, 2] = points[:, 1]
    return out


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*(int(max(0, min(255, c))) for c in rgb))


def plate_from_model(
    model: GlbModel,
    colours: list[tuple[int, int, int]],
    *,
    slots: list[int] | None = None,
    palette_indices: list[int] | None = None,
    mm_per_unit: float = MM_PER_UNIT,
) -> PlateModel:
    """把模型摊成一块「板」：每个有三角形的 primitive 一个零件。

    ``colours[i]`` 是第 i 个 primitive 配到的颜色（``model.primitives`` 的顺序），
    ``slots[i]`` 是它占的耗材槽号（1 起），``palette_indices[i]`` 是它在调色板里的
    下标。三者缺省时按 1 / 0 处理。
    """
    parts: list[PlatePart] = []
    lows: list[np.ndarray] = []
    highs: list[np.ndarray] = []
    for order, primitive in enumerate(model.primitives):
        if primitive.positions.size == 0 or primitive.triangles.size == 0:
            continue
        points = display_points(np.asarray(primitive.positions, dtype=np.float64)) * mm_per_unit
        rgb = colours[order] if order < len(colours) else primitive.colour
        parts.append(
            PlatePart(
                name=primitive.name or f"第 {order + 1} 块",
                color_hex=_hex(rgb),
                palette_index=int(palette_indices[order]) if palette_indices else 0,
                extruder=int(slots[order]) if slots else 1,
                vertices=points,
                triangles=np.asarray(primitive.triangles, dtype=np.int64),
                region_pixels=int(len(primitive.triangles)),
                z_bottom_mm=float(points[:, 2].min()) if points.size else 0.0,
                z_top_mm=float(points[:, 2].max()) if points.size else 0.0,
            )
        )
        lows.append(points.min(axis=0))
        highs.append(points.max(axis=0))
    if not parts:
        raise ValueError("这个模型里没有三角形，导不出打印文件。")

    low = np.vstack(lows).min(axis=0)
    high = np.vstack(highs).max(axis=0)
    centre_x = float((low[0] + high[0]) * 0.5)
    centre_y = float((low[1] + high[1]) * 0.5)
    bottom = float(low[2])
    shift = np.array([centre_x, centre_y, bottom], dtype=np.float64)
    for part in parts:
        part.vertices = part.vertices - shift
        part.z_bottom_mm = float(part.vertices[:, 2].min())
        part.z_top_mm = float(part.vertices[:, 2].max())

    return PlateModel(
        parts=parts,
        width_mm=float(high[0] - low[0]),
        depth_mm=float(high[1] - low[1]),
        pixel_size_mm=0.0,
        base_thickness_mm=0.0,
        colour_thickness_mm=0.0,
        notes=[
            f"来自模型：{model.name}（{model.triangle_count:,} 个三角形）",
            f"尺寸：{high[0] - low[0]:.1f} × {high[1] - low[1]:.1f} × "
            f"{high[2] - low[2]:.1f} mm（glTF 按 1 单位 = 1 米换算）",
        ],
        source=model.name,
    )


def _linear(value: float) -> float:
    """sRGB 的 0..1 编码值 → 线性值（glTF 的 baseColorFactor 是线性的）。"""
    value = min(1.0, max(0.0, float(value)))
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def glb_bytes(model: GlbModel, colours: list[tuple[int, int, int]]) -> bytes:
    """把模型重新包成一个 GLB，材质换成配到的颜色。

    几何原样搬（glTF 坐标系，不挪不缩），所以再导入回来就是「带着配色的模型」，
    可以接着改。顶点色与贴图不写：这一版只认材质底色，导出的文件里一个颜色一块材质。
    """
    blobs = bytearray()
    views: list[dict] = []
    accessors: list[dict] = []
    primitives: list[dict] = []
    materials: list[dict] = []

    def push(payload: bytes, *, target: int) -> int:
        while len(blobs) % 4:
            blobs.append(0)
        offset = len(blobs)
        blobs.extend(payload)
        views.append(
            {
                "buffer": 0,
                "byteOffset": offset,
                "byteLength": len(payload),
                "target": target,
            }
        )
        return len(views) - 1

    for order, primitive in enumerate(model.primitives):
        if primitive.positions.size == 0 or primitive.triangles.size == 0:
            continue
        positions = np.ascontiguousarray(primitive.positions, dtype=np.float32)
        normals = np.ascontiguousarray(primitive.normals, dtype=np.float32)
        indices = np.ascontiguousarray(
            np.asarray(primitive.triangles, dtype=np.uint32).reshape(-1)
        )
        attributes = {"POSITION": len(accessors)}
        accessors.append(
            {
                "bufferView": push(positions.tobytes(), target=34962),
                "componentType": 5126,
                "count": int(positions.shape[0]),
                "type": "VEC3",
                "min": [float(v) for v in positions.min(axis=0)],
                "max": [float(v) for v in positions.max(axis=0)],
            }
        )
        if normals.size:
            attributes["NORMAL"] = len(accessors)
            accessors.append(
                {
                    "bufferView": push(normals.tobytes(), target=34962),
                    "componentType": 5126,
                    "count": int(normals.shape[0]),
                    "type": "VEC3",
                }
            )
        accessors.append(
            {
                "bufferView": push(indices.tobytes(), target=34963),
                "componentType": 5125,
                "count": int(indices.size),
                "type": "SCALAR",
            }
        )
        rgb = colours[order] if order < len(colours) else primitive.colour
        materials.append(
            {
                "name": primitive.material or f"第 {order + 1} 块",
                "doubleSided": True,
                "pbrMetallicRoughness": {
                    "baseColorFactor": [_linear(c / 255.0) for c in rgb] + [1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.65,
                },
            }
        )
        primitives.append(
            {
                "attributes": attributes,
                "indices": len(accessors) - 1,
                "material": len(materials) - 1,
                "mode": 4,
            }
        )

    if not primitives:
        raise ValueError("这个模型里没有三角形，导不出 GLB。")

    while len(blobs) % 4:
        blobs.append(0)
    document = {
        "asset": {"version": "2.0", "generator": "BambuPalette 模型配色"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": model.name or "model"}],
        "meshes": [{"name": model.name or "model", "primitives": primitives}],
        "materials": materials,
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(blobs)}],
    }
    text = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    while len(text) % 4:
        text += b" "

    total = 12 + 8 + len(text) + 8 + len(blobs)
    out = bytearray()
    out += struct.pack("<III", 0x46546C67, 2, total)
    out += struct.pack("<II", len(text), 0x4E4F534A) + text
    out += struct.pack("<II", len(blobs), 0x004E4942) + bytes(blobs)
    return bytes(out)


def write_glb(model: GlbModel, colours: list[tuple[int, int, int]], target: str | Path) -> Path:
    """写成一个 GLB 文件（同目录临时文件 + 改名，1.6.50 的规矩）。"""
    from .atomicio import atomic_path

    path = Path(target)
    if path.suffix.lower() != ".glb":
        path = path.with_suffix(".glb")
    payload = glb_bytes(model, colours)
    with atomic_path(path) as temporary:
        with open(temporary, "wb") as handle:
            handle.write(payload)
    return path

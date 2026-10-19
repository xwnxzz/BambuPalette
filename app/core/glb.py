"""读 GLB / glTF：三角形、法线与材质颜色。

只做「配色要用的那一部分」：世界坐标下的三角形、法线、每个 primitive 的主色。
蒙皮、动画、morph target、相机、灯光一律跳过 —— 本程序不打印模型，只给它配色。

颜色约定：glTF 里的 ``baseColorFactor`` 与 ``COLOR_0`` 都是**线性**值，界面和匹配
都在 sRGB 里做，所以出来之前统一转成 sRGB 0–255。``baseColorTexture`` 取的是纹理里
不透明像素的均值（按像素计），不是 UV 加权 —— 这一点写在 docs/算法说明.md 里。
"""

from __future__ import annotations

import base64
import io
import json
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote

import numpy as np

GLB_SUFFIXES = frozenset({".glb", ".gltf"})

#: glTF 的 accessor constituentType（3.6 表）。
_COMPONENT_TYPES: dict[int, tuple[str, int]] = {
    5120: ("b", 1),   # BYTE
    5121: ("B", 1),   # UNSIGNED_BYTE
    5122: ("h", 2),   # SHORT
    5123: ("H", 2),   # UNSIGNED_SHORT
    5125: ("I", 4),   # UNSIGNED_INT
    5126: ("f", 4),   # FLOAT
}
_TYPE_SIZE = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}
#: 只认三角形（glTF 里 4 = TRIANGLES）。点、线、三角扇一律跳过并记一笔。
_MODE_TRIANGLES = 4
#: 纹理取色时把颜色压到 5 bit 一格再数，最多留这么多种。
_TEXTURE_COLOURS = 3


@dataclass
class GlbPrimitive:
    """一个 primitive = 一次绘制，颜色按材质算。"""

    name: str
    material: str
    positions: np.ndarray
    normals: np.ndarray
    triangles: np.ndarray
    colour: tuple[int, int, int]
    extra_colours: tuple[tuple[int, int, int], ...] = ()
    area: float = 0.0

    @property
    def triangle_count(self) -> int:
        return int(self.triangles.shape[0])


@dataclass
class GlbModel:
    """一个 GLB 里所有能画的东西。"""

    name: str
    primitives: list[GlbPrimitive] = field(default_factory=list)
    low: np.ndarray = field(default_factory=lambda: np.zeros(3))
    high: np.ndarray = field(default_factory=lambda: np.zeros(3))
    skipped: tuple[str, ...] = ()
    #: 我们自己导出的 GLB 会把「当时怎么配的」写进 ``extras.bambupalette``，
    #: 读回来就放在这里，让模型配色页restore 回原样（1.6.55，m25568 第 3 条）。
    saved: dict | None = None

    @property
    def triangle_count(self) -> int:
        return sum(item.triangle_count for item in self.primitives)

    @property
    def vertex_count(self) -> int:
        return sum(int(item.positions.shape[0]) for item in self.primitives)

    @property
    def size(self) -> np.ndarray:
        return np.maximum(self.high - self.low, 1e-9)

    @property
    def is_empty(self) -> bool:
        return not self.primitives or self.triangle_count == 0


@dataclass(frozen=True)
class ModelColour:
    """模型里的一种颜色，以及它占了多少面积。"""

    rgb: tuple[int, int, int]
    area: float
    primitives: int

    @property
    def share(self) -> float:  # pragma: no cover - 由调用方除以总面积
        return self.area


# -- 容器 ------------------------------------------------------------------


def _read_container(path: Path) -> tuple[dict, bytes | None, Path]:
    """GLB 拆成 (JSON, BIN)；.gltf 只有 JSON。"""
    blob = path.read_bytes()
    if blob[:4] == b"glTF":
        magic, version, length = struct.unpack_from("<4sII", blob, 0)
        if magic != b"glTF":
            raise ValueError(f"这不是 GLB：{path.name}")
        if version != 2:
            raise ValueError(f"只认 glTF 2.0，这张是 {version}：{path.name}")
        offset = 12
        doc: dict | None = None
        binary: bytes | None = None
        while offset + 8 <= min(length, len(blob)):
            chunk_len, chunk_type = struct.unpack_from("<II", blob, offset)
            offset += 8
            payload = blob[offset : offset + chunk_len]
            offset += chunk_len
            if chunk_type == 0x4E4F534A:  # 'JSON'
                doc = json.loads(payload.decode("utf-8"))
            elif chunk_type == 0x004E4942:  # 'BIN\0'
                binary = payload
        if doc is None:
            raise ValueError(f"GLB 里没有 JSON 段：{path.name}")
        return doc, binary, path.parent
    try:
        doc = json.loads(blob.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ValueError(f"不是 GLB 也不是 glTF：{path.name}") from exc
    if not isinstance(doc, dict) or "meshes" not in doc:
        raise ValueError(f"这个 glTF 里没有 meshes：{path.name}")
    return doc, None, path.parent


def _buffers(doc: dict, binary: bytes | None, base: Path) -> list[bytes]:
    out: list[bytes] = []
    for index, entry in enumerate(doc.get("buffers", [])):
        uri = entry.get("uri")
        if uri is None:
            if binary is None:
                raise ValueError(f"buffer {index} 既没有 uri 也没有 BIN 段")
            out.append(binary)
            continue
        if uri.startswith("data:"):
            head, _, payload = uri.partition(",")
            if ";base64" in head:
                out.append(base64.b64decode(payload))
            else:
                out.append(unquote(payload).encode("latin-1"))
            continue
        out.append((base / unquote(uri)).read_bytes())
    if not out:
        out.append(binary or b"")
    return out


def _accessor(doc: dict, buffers: list[bytes], index: int) -> np.ndarray:
    """按 accessor 读出 (count, 分量) 的数组；稀疏访问器在这里不够用会报错。"""
    try:
        entry = doc["accessors"][index]
    except (KeyError, IndexError) as exc:
        raise ValueError(f"找不到 accessor {index}") from exc
    if "sparse" in entry:
        raise ValueError("带 sparse 的 accessor 暂不支持")
    count = int(entry.get("count", 0))
    kind = entry.get("type", "SCALAR")
    size = _TYPE_SIZE.get(kind)
    if size is None:
        raise ValueError(f"不认识的 accessor 类型：{kind}")
    fmt, width = _COMPONENT_TYPES.get(int(entry.get("componentType", 0)), ("", 0))
    if not fmt:
        raise ValueError(f"不认识的 componentType：{entry.get('componentType')}")
    if count == 0:
        return np.zeros((0, size), dtype=np.float64)
    view_index = entry.get("bufferView")
    if view_index is None:
        return np.zeros((count, size), dtype=np.float64)
    view = doc["bufferViews"][int(view_index)]
    buffer = buffers[int(view.get("buffer", 0))]
    base = int(view.get("byteOffset", 0)) + int(entry.get("byteOffset", 0))
    stride = int(view.get("byteStride", 0)) or width * size
    need = base + stride * (count - 1) + width * size
    if need > len(buffer):
        raise ValueError("accessor 越界：文件被截断了？")
    if stride == width * size:
        flat = np.frombuffer(buffer, dtype=np.dtype(fmt), count=count * size, offset=base)
        data = np.array(flat, dtype=np.float64).reshape(count, size)
    else:
        data = np.empty((count, size), dtype=np.float64)
        for row in range(count):
            start = base + row * stride
            chunk = np.frombuffer(buffer, dtype=np.dtype(fmt), count=size, offset=start)
            data[row] = np.asarray(chunk, dtype=np.float64)
    if entry.get("normalized") and fmt not in ("f",):
        data = data / float(np.iinfo(np.dtype(fmt)).max)
    return data


# -- 颜色 ------------------------------------------------------------------


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def linear_to_srgb(values: np.ndarray) -> np.ndarray:
    """线性 0–1 → sRGB 0–1（glTF 里的颜色都是线性的）。"""
    clipped = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    low = clipped * 12.92
    high = 1.055 * np.power(clipped, 1.0 / 2.4) - 0.055
    return np.where(clipped <= 0.0031308, low, high)


def _to_rgb255(linear: np.ndarray) -> tuple[int, int, int]:
    srgb = linear_to_srgb(np.asarray(linear, dtype=np.float64))
    return tuple(int(round(float(c) * 255.0)) for c in srgb[:3])


def _cluster(colours: np.ndarray, weights: np.ndarray, limit: int) -> list[tuple[int, int, int]]:
    """把一堆颜色按 5 bit 一格数出前 ``limit`` 名，返回 sRGB 0–255。"""
    if colours.size == 0:
        return []
    quant = np.clip(np.round(colours * 31.0), 0, 31).astype(np.int64)
    keys = (quant[:, 0] << 10) | (quant[:, 1] << 5) | quant[:, 2]
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    weights = np.asarray(weights, dtype=np.float64)[order]
    unique, start = np.unique(keys, return_index=True)
    totals = np.add.reduceat(weights, start)
    ranked = unique[np.argsort(-totals, kind="stable")]
    out: list[tuple[int, int, int]] = []
    for key in ranked[: max(1, limit)]:
        mask = keys == key
        mean = np.average(colours[order][mask], axis=0, weights=weights[mask])
        out.append(_to_rgb255(mean))
    return out


def _image_average(blob: bytes) -> tuple[int, int, int] | None:
    """纹理的均值色（只算不透明像素）。"""
    try:
        from PIL import Image

        with Image.open(io.BytesIO(blob)) as picture:
            frame = picture.convert("RGBA")
            data = np.asarray(frame, dtype=np.float64)
    except Exception:  # noqa: BLE001 - 图坏了就当没有纹理
        return None
    alpha = data[..., 3]
    keep = alpha > 127.0
    if not keep.any():
        return None
    rgb = data[keep][:, :3] / 255.0
    # 纹理是 sRGB 编码的图片；取均值之前在**线性**里平均才符合物理直觉，但配色
    # 是看肉眼效果，直接在 sRGB 里平均更接近用户看到的「这张图大概什么色」。
    mean = rgb.mean(axis=0) * 255.0
    return tuple(int(round(float(c))) for c in mean)


def _texture_colours(
    doc: dict, buffers: list[bytes], base: Path, texture_index: int
) -> list[tuple[int, int, int]]:
    try:
        texture = doc["textures"][texture_index]
        source = doc["images"][int(texture["source"])]
    except (KeyError, IndexError, TypeError):
        return []
    blob: bytes | None = None
    if "bufferView" in source:
        view = doc["bufferViews"][int(source["bufferView"])]
        buffer = buffers[int(view.get("buffer", 0))]
        start = int(view.get("byteOffset", 0))
        blob = buffer[start : start + int(view["byteLength"])]
    elif "uri" in source:
        uri = str(source["uri"])
        if uri.startswith("data:"):
            head, _, payload = uri.partition(",")
            if ";base64" in head:
                blob = base64.b64decode(payload)
            else:
                blob = unquote(payload).encode("latin-1")
        else:
            try:
                blob = (base / unquote(uri)).read_bytes()
            except OSError:
                return []
    if blob is None:
        return []
    average = _image_average(blob)
    return [average] if average is not None else []


def _material_colours(doc: dict, buffers: list[bytes], base: Path, material_index: int) -> list[tuple[int, int, int]]:
    """材质能给出的颜色：baseColorFactor + baseColorTexture 的主色。"""
    material = doc["materials"][material_index] if material_index < len(doc.get("materials", [])) else {}
    pbr = material.get("pbrMetallicRoughness", {}) if isinstance(material, dict) else {}
    out: list[tuple[int, int, int]] = []
    factor = pbr.get("baseColorFactor")
    if isinstance(factor, list) and len(factor) >= 3:
        out.append(_to_rgb255(np.array(factor[:3], dtype=np.float64)))
    texture = pbr.get("baseColorTexture")
    if isinstance(texture, dict) and "index" in texture:
        out.extend(_texture_colours(doc, buffers, base, int(texture["index"])))
    if not out:
        # 没有材质的 primitive 在 glTF 里就是默认材质：白，非金属，粗糙。
        out.append((255, 255, 255))
    return out


# -- 顶点颜色 --------------------------------------------------------------


def _vertex_colours(doc: dict, buffers: list[bytes], accessor_index: int) -> np.ndarray:
    data = _accessor(doc, buffers, accessor_index)
    if data.ndim != 2 or data.shape[1] < 3:
        return np.zeros((0, 3))
    return data[:, :3]


# -- 变换 ------------------------------------------------------------------


def _node_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        values = np.asarray(node["matrix"], dtype=np.float64)
        if values.size == 16:
            return values.reshape(4, 4).T
    matrix = np.eye(4)
    translation = node.get("translation")
    rotation = node.get("rotation")
    scale = node.get("scale")
    if rotation is not None and len(rotation) == 4:
        x, y, z, w = (float(v) for v in rotation)
        matrix[:3, :3] = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
    if scale is not None and len(scale) == 3:
        matrix[:3, :3] = matrix[:3, :3] @ np.diag([float(v) for v in scale])
    if translation is not None and len(translation) == 3:
        matrix[:3, 3] = [float(v) for v in translation]
    return matrix


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points
    homogeneous = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    return (homogeneous @ matrix.T)[:, :3]


def _apply_normals(matrix: np.ndarray, normals: np.ndarray) -> np.ndarray:
    if normals.size == 0:
        return normals
    linear = matrix[:3, :3]
    try:
        inverse = np.linalg.inv(linear).T
    except np.linalg.LinAlgError:
        inverse = linear
    out = normals @ inverse.T
    lengths = np.linalg.norm(out, axis=1, keepdims=True)
    return np.divide(out, np.where(lengths < 1e-12, 1.0, lengths))


# -- 主入口 ----------------------------------------------------------------


def load_model(source: str | Path) -> GlbModel:
    """读一个 .glb / .gltf，返回可以直接画、也可以直接配色的模型。"""
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"找不到文件：{path}")
    doc, binary, base = _read_container(path)
    buffers = _buffers(doc, binary, base)
    primitives: list[GlbPrimitive] = []
    skipped: list[str] = []

    def visit(node_index: int, parent: np.ndarray, depth: int) -> None:
        if depth > 64:
            skipped.append("节点层级太深，下面那截没读")
            return
        node = doc["nodes"][node_index]
        matrix = parent @ _node_matrix(node)
        mesh_index = node.get("mesh")
        if isinstance(mesh_index, int):
            for item in _mesh_primitives(doc, buffers, base, mesh_index, matrix, skipped):
                primitives.append(item)
        for child in node.get("children", []) or []:
            visit(int(child), matrix, depth + 1)

    for scene in doc.get("scenes", []) or []:
        for node_index in scene.get("nodes", []) or []:
            visit(int(node_index), np.eye(4), 0)
    if not primitives:
        for node_index, node in enumerate(doc.get("nodes", [])):
            if "mesh" in node:
                visit(node_index, np.eye(4), 0)
    if not primitives and doc.get("meshes"):
        skipped.append("这个文件里的网格都不是三角形，跳过了")

    if primitives:
        low = np.min([item.positions.min(axis=0) for item in primitives if item.positions.size], axis=0)
        high = np.max([item.positions.max(axis=0) for item in primitives if item.positions.size], axis=0)
    else:
        low = np.zeros(3)
        high = np.zeros(3)
    saved = (doc.get("extras") or {}).get("bambupalette") if isinstance(doc.get("extras"), dict) else None
    return GlbModel(
        name=path.stem,
        primitives=primitives,
        low=np.asarray(low, dtype=np.float64),
        high=np.asarray(high, dtype=np.float64),
        skipped=tuple(dict.fromkeys(skipped)),
        saved=saved if isinstance(saved, dict) else None,
    )


def _mesh_primitives(
    doc: dict,
    buffers: list[bytes],
    base: Path,
    mesh_index: int,
    matrix: np.ndarray,
    skipped: list[str],
) -> list[GlbPrimitive]:
    out: list[GlbPrimitive] = []
    mesh = doc["meshes"][mesh_index]
    mesh_name = str(mesh.get("name") or f"mesh{mesh_index}")
    for order, primitive in enumerate(mesh.get("primitives", []) or []):
        mode = int(primitive.get("mode", _MODE_TRIANGLES))
        attributes = primitive.get("attributes") or {}
        if "POSITION" not in attributes:
            skipped.append(f"{mesh_name} 第 {order + 1} 个 primitive 没有顶点，跳过了")
            continue
        positions = _apply(matrix, _accessor(doc, buffers, int(attributes["POSITION"])))
        normals = (
            _apply_normals(matrix, _accessor(doc, buffers, int(attributes["NORMAL"])))
            if "NORMAL" in attributes
            else np.zeros_like(positions)
        )
        if mode != _MODE_TRIANGLES:
            skipped.append(f"{mesh_name} 第 {order + 1} 个 primitive 不是三角形，跳过了")
            continue
        indices = primitive.get("indices")
        if isinstance(indices, int):
            raw = _accessor(doc, buffers, indices).astype(np.int64).ravel()
        else:
            raw = np.arange(positions.shape[0], dtype=np.int64)
        usable = (raw.size // 3) * 3
        triangles = raw[:usable].reshape(-1, 3)
        if triangles.size == 0:
            continue
        if triangles.max() >= positions.shape[0]:
            skipped.append(f"{mesh_name} 第 {order + 1} 个 primitive 的索引越界，跳过了")
            continue

        material_index = primitive.get("material")
        palette = (
            _material_colours(doc, buffers, base, int(material_index))
            if isinstance(material_index, int)
            else [(255, 255, 255)]
        )
        if "COLOR_0" in attributes:
            vertex = _vertex_colours(doc, buffers, int(attributes["COLOR_0"]))
            if vertex.shape[0] == positions.shape[0]:
                picks = _cluster(vertex, np.ones(vertex.shape[0]), len(palette) + 2)
                if picks:
                    palette = picks
        corner = positions[triangles]
        area = float(
            0.5 * np.linalg.norm(np.cross(corner[:, 1] - corner[:, 0], corner[:, 2] - corner[:, 0]), axis=1).sum()
        )
        out.append(
            GlbPrimitive(
                name=f"{mesh_name}#{order + 1}",
                material=str(
                    doc["materials"][int(material_index)].get("name", f"material{material_index}")
                    if isinstance(material_index, int) and int(material_index) < len(doc.get("materials", []))
                    else "默认材质"
                ),
                positions=positions,
                normals=normals,
                triangles=triangles.astype(np.int32),
                colour=palette[0],
                extra_colours=tuple(palette[1:]),
                area=area if math.isfinite(area) else 0.0,
            )
        )
    return out


def model_colours(model: GlbModel) -> list[ModelColour]:
    """模型里出现过哪些颜色、各占多少面积（按面积从大到小）。"""
    totals: dict[tuple[int, int, int], list] = {}
    for primitive in model.primitives:
        for index, colour in enumerate((primitive.colour, *primitive.extra_colours)):
            share = primitive.area if index == 0 else primitive.area * 0.35
            entry = totals.setdefault(tuple(int(c) for c in colour), [0.0, 0])
            entry[0] += share
            entry[1] += 1
    ordered = sorted(totals.items(), key=lambda item: (-item[1][0], item[0]))
    return [ModelColour(rgb=key, area=value[0], primitives=value[1]) for key, value in ordered]

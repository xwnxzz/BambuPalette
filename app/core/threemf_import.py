"""读 3MF —— 让模型配色页能直接打开切片/建模软件存出来的 3MF（1.6.55，m25568 第 4 条）。

出处与授权
----------
思路来自 https://github.com/SamiSalah221/3mf-to-glb（MIT，Printproof）。那个项目是
TypeScript / 浏览器端，把 3MF / G-code 转成 GLB / USDZ / OBJ；本程序是 Python + Qt，
不可能把一套 Node 工具链塞进安装包，所以这里只借用它的**解析思路**，用 Python 重新实现：

* 3MF 就是个 zip，几何在 ``3D/3dmodel.model``（XML）；
* 颜色有三种来源，按优先级取：
  1. 三角形自己的 ``pid`` + ``p1/p2/p3``（Bambu / Orca 的逐面上色）；
  2. 对象（或它引用的 base 材料）的 ``pid``；
  3. ``Metadata/model_settings.config`` 里的 ``filament colour``（Bambu 工程）。
* ``build/item`` 上的 ``transform`` 是一个 12 元组（3×4 行主序），逐层乘下去。

为什么读成 :class:`app.core.glb.GlbModel`
-----------------------------------------
模型配色页后面那一整条链（配色、三维视图、GLB / OBJ / 3MF 三个出口）吃的都是 GlbModel；
3MF 只是另一种「有颜色的一面面三角形」，读成同一个结构就不用复制第二套管线。

单位
----
3MF 用毫米，glTF 的 1 单位在 1.6.52 里定成 1 米（``model_export.MM_PER_UNIT = 1000``），
所以读进来时坐标乘 ``1/1000``，下游算板子尺寸时不必再分支。
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np

from .glb import GlbModel, GlbPrimitive

__all__ = ["THREEMF_SUFFIXES", "LOADABLE_SUFFIXES", "load_3mf_model", "threemf_to_glb_model"]

THREEMF_SUFFIXES = frozenset({".3mf"})
#: 模型配色页「打开」对话框认的扩展名（GLB / glTF / 3MF）。
LOADABLE_SUFFIXES = frozenset({".glb", ".gltf", ".3mf"})

#: 3MF 用毫米，GlbModel 用米（见模块说明）。
_MM_TO_UNIT = 0.001

_CORE_NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
_MATERIAL_NS = "http://schemas.microsoft.com/3dmanufacturing/material/2015/02"
_MODEL_ENTRY = "3D/3dmodel.model"
_SETTINGS_ENTRY = "Metadata/model_settings.config"

_UNIT_TO_MM = {
    "micron": 0.001,
    "millimeter": 1.0,
    "centimeter": 10.0,
    "inch": 25.4,
    "foot": 304.8,
    "meter": 1000.0,
}


def _local(tag: str) -> str:
    """``{ns}vertex`` → ``vertex``：3MF 各家的命名空间前缀不一样，只认本地名。"""
    return tag.rsplit("}", 1)[-1]


def _hex_to_rgb(text: str) -> tuple[int, int, int] | None:
    """``#RRGGBB`` / ``#RRGGBBAA`` → (r, g, b)。读不出来返回 None。"""
    value = (text or "").strip().lstrip("#")
    if len(value) < 6:
        return None
    try:
        return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))
    except ValueError:
        return None


def _parse_transform(text: str | None) -> np.ndarray:
    """3MF 的 ``transform`` 是 12 个数（3×4 行主序：m00 m01 m02 m03 m10 …）。"""
    matrix = np.eye(4, dtype=np.float64)
    if not text:
        return matrix
    try:
        values = [float(item) for item in re.split(r"[\s,]+", text.strip()) if item]
    except ValueError:
        return matrix
    if len(values) != 12:
        return matrix
    matrix[:3, :4] = np.asarray(values, dtype=np.float64).reshape(3, 4)
    return matrix


class _Object:
    """一个 ``<object>``：可能是网格，也可能是一组引用别人的 ``components``。"""

    def __init__(self, obj_id: str, name: str) -> None:
        self.id = obj_id
        self.name = name
        self.vertices = np.zeros((0, 3), dtype=np.float64)
        self.triangles = np.zeros((0, 3), dtype=np.int64)
        self.colour_of_face: list[tuple[int, int, int] | None] = []
        self.pid: str | None = None
        self.pindex: int | None = None
        self.components: list[tuple[str, np.ndarray]] = []
        self.parts: list[tuple[str, str | None]] = []


def _read_colour_lists(root: ET.Element) -> dict[str, list[tuple[int, int, int]]]:
    """``<basematerials>`` 与 ``<m:colorgroup>`` 里的颜色表（按 id 索引）。"""
    groups: dict[str, list[tuple[int, int, int]]] = {}
    for element in root.iter():
        name = _local(element.tag)
        if name not in {"basematerials", "colorgroup"}:
            continue
        group_id = element.get("id")
        if not group_id:
            continue
        colours: list[tuple[int, int, int]] = []
        for child in element:
            if _local(child.tag) == "base":
                rgb = _hex_to_rgb(child.get("displaycolor") or child.get("color") or "")
            else:
                rgb = _hex_to_rgb(child.get("color") or "")
            colours.append(rgb or (200, 200, 200))
        groups[group_id] = colours
    return groups


def _pick_colour(
    groups: dict[str, list[tuple[int, int, int]]],
    pid: str | None,
    index: int | None,
) -> tuple[int, int, int] | None:
    if pid is None:
        return None
    table = groups.get(pid)
    if not table:
        return None
    if index is None or not (0 <= index < len(table)):
        return table[0]
    return table[index]


def _read_objects(root: ET.Element) -> tuple[dict[str, _Object], dict[str, list[tuple[int, int, int]]]]:
    groups = _read_colour_lists(root)
    objects: dict[str, _Object] = {}
    for element in root.iter():
        if _local(element.tag) != "object":
            continue
        obj_id = element.get("id")
        if not obj_id:
            continue
        obj = _Object(obj_id, element.get("name") or f"object {obj_id}")
        obj.pid = element.get("pid")
        raw_pindex = element.get("pindex")
        obj.pindex = int(raw_pindex) if raw_pindex and raw_pindex.isdigit() else None
        for child in element:
            name = _local(child.tag)
            if name == "mesh":
                verts: list[list[float]] = []
                tris: list[list[int]] = []
                face_colours: list[tuple[int, int, int] | None] = []
                for part in child:
                    part_name = _local(part.tag)
                    if part_name == "vertices":
                        for vertex in part:
                            if _local(vertex.tag) != "vertex":
                                continue
                            try:
                                verts.append(
                                    [
                                        float(vertex.get("x", "0")),
                                        float(vertex.get("y", "0")),
                                        float(vertex.get("z", "0")),
                                    ]
                                )
                            except ValueError:
                                continue
                    elif part_name == "triangles":
                        for triangle in part:
                            if _local(triangle.tag) != "triangle":
                                continue
                            try:
                                tris.append(
                                    [
                                        int(triangle.get("v1", "0")),
                                        int(triangle.get("v2", "0")),
                                        int(triangle.get("v3", "0")),
                                    ]
                                )
                            except ValueError:
                                continue
                            colour = _pick_colour(
                                groups,
                                triangle.get("pid") or obj.pid,
                                int(triangle.get("p1", obj.pindex or 0) or 0),
                            )
                            face_colours.append(colour)
                obj.vertices = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
                obj.triangles = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
                obj.colour_of_face = face_colours
            elif name == "components":
                for component in child:
                    if _local(component.tag) != "component":
                        continue
                    target = component.get("objectid")
                    if target:
                        obj.components.append((target, _parse_transform(component.get("transform"))))
        objects[obj_id] = obj
    return objects, groups


def _read_settings_colours(archive: zipfile.ZipFile) -> list[tuple[int, int, int]]:
    """Bambu / Orca 把耗材颜色写在 ``Metadata/model_settings.config`` 里。"""
    try:
        text = archive.read(_SETTINGS_ENTRY).decode("utf-8", "replace")
    except KeyError:
        return []
    colours: list[tuple[int, int, int]] = []
    for match in re.finditer(r'metadata\s+key="filament_colour"\s+value="([^"]+)"', text):
        rgb = _hex_to_rgb(match.group(1))
        if rgb is not None:
            colours.append(rgb)
    return colours


def _flatten(
    objects: dict[str, _Object],
    obj_id: str,
    matrix: np.ndarray,
    out: list[tuple[np.ndarray, np.ndarray, list[tuple[int, int, int] | None], str]],
    depth: int = 0,
) -> None:
    """把一个对象（含 components）摊成「顶点 + 三角形 + 每面颜色」的几份。"""
    if depth > 12:
        return
    obj = objects.get(obj_id)
    if obj is None:
        return
    if obj.vertices.size and obj.triangles.size:
        verts = obj.vertices @ matrix[:3, :3].T + matrix[:3, 3]
        out.append((verts, obj.triangles, obj.colour_of_face, obj.name))
    for target, local in obj.components:
        _flatten(objects, target, matrix @ local, out, depth + 1)


def threemf_to_glb_model(path: Path, *, name: str | None = None) -> GlbModel:
    """把 3MF 读成 :class:`GlbModel`（坐标单位：米）。"""
    path = Path(path)
    if not zipfile.is_zipfile(path):
        raise ValueError(f"不是 3MF（它其实是个 zip）：{path.name}")
    skipped: list[str] = []
    with zipfile.ZipFile(path) as archive:
        entry = _MODEL_ENTRY
        try:
            payload = archive.read(entry)
        except KeyError:
            candidates = [item for item in archive.namelist() if item.lower().endswith(".model")]
            if not candidates:
                raise ValueError(f"这个 3MF 里没有 3D 模型：{path.name}") from None
            entry = candidates[0]
            payload = archive.read(entry)
        try:
            root = ET.fromstring(payload)
        except ET.ParseError as error:
            raise ValueError(f"3D 模型那段 XML 读不动：{error}") from None
        settings_colours = _read_settings_colours(archive)

    unit = (root.get("unit") or "millimeter").lower()
    scale = _UNIT_TO_MM.get(unit, 1.0) * _MM_TO_UNIT

    objects, _groups = _read_objects(root)
    build = None
    for element in root.iter():
        if _local(element.tag) == "build":
            build = element
            break
    items: list[tuple[str, np.ndarray]] = []
    if build is not None:
        for item in build:
            if _local(item.tag) != "item":
                continue
            target = item.get("objectid")
            if target:
                items.append((target, _parse_transform(item.get("transform"))))
    if not items:
        items = [(obj_id, np.eye(4)) for obj_id in objects]

    pieces: list[tuple[np.ndarray, np.ndarray, list[tuple[int, int, int] | None], str]] = []
    for target, matrix in items:
        _flatten(objects, target, matrix, pieces)
    if not pieces:
        skipped.append("这个 3MF 里没有三角形")

    flat: list[tuple[np.ndarray, np.ndarray, list[tuple[int, int, int] | None], str]] = []
    for verts, tris, colours, piece_name in pieces:
        if not verts.size or not tris.size:
            continue
        # 个别文件把单位写在 <model unit>，另一些干脆按毫米；这里统一乘 scale。
        flat.append((verts * scale, tris, colours, piece_name))
    if not flat and not skipped:
        skipped.append("这个 3MF 里没有三角形")

    model = GlbModel(name=name or path.stem, primitives=[], skipped=())
    low = np.full(3, np.inf)
    high = np.full(3, -np.inf)
    fallback = settings_colours[0] if settings_colours else (200, 200, 200)

    for verts, tris, colours, piece_name in flat:
        if tris.size == 0:
            continue
        valid = (
            (tris[:, 0] >= 0)
            & (tris[:, 1] >= 0)
            & (tris[:, 2] >= 0)
            & (tris[:, 0] < verts.shape[0])
            & (tris[:, 1] < verts.shape[0])
            & (tris[:, 2] < verts.shape[0])
        )
        if not valid.all():
            skipped.append(f"{piece_name}：有 {int((~valid).sum())} 个三角形的顶点下标越界，已跳过")
            tris = tris[valid]
            colours = [colour for colour, keep in zip(colours, valid) if keep] if colours else []
        if tris.size == 0:
            continue
        # 按颜色分组：一种颜色一个 GlbPrimitive，下游 model_colours 才有东西可聚。
        buckets: dict[tuple[int, int, int], np.ndarray] = {}
        if colours and len(colours) == tris.shape[0]:
            for index, colour in enumerate(colours):
                buckets.setdefault(colour or fallback, []).append(index)
        else:
            buckets[fallback] = list(range(tris.shape[0]))
        for colour, indices in buckets.items():
            subset = tris[np.asarray(indices, dtype=np.int64)]
            used = np.unique(subset)
            remap = {int(vertex): position for position, vertex in enumerate(used)}
            local_tris = np.asarray([[remap[int(v)] for v in row] for row in subset], dtype=np.int64)
            local_verts = verts[used]
            corner = local_verts[local_tris]
            area = float(
                np.linalg.norm(
                    np.cross(corner[:, 1] - corner[:, 0], corner[:, 2] - corner[:, 0]), axis=1
                ).sum()
                * 0.5
            )
            model.primitives.append(
                GlbPrimitive(
                    name=piece_name,
                    material=f"#{colour[0]:02X}{colour[1]:02X}{colour[2]:02X}",
                    positions=local_verts.astype(np.float32),
                    normals=np.zeros((0, 3), dtype=np.float32),
                    triangles=local_tris.astype(np.int64),
                    colour=(int(colour[0]), int(colour[1]), int(colour[2])),
                    area=area,
                )
            )
            low = np.minimum(low, local_verts.min(axis=0))
            high = np.maximum(high, local_verts.max(axis=0))

    if model.primitives:
        model.low = low
        model.high = high
    model.skipped = tuple(skipped)
    return model


def load_3mf_model(path: str | Path, *, name: str | None = None) -> GlbModel:
    """``load_3mf_model`` 是上面那个函数的好看名字（闸门与页面都用它）。"""
    return threemf_to_glb_model(Path(path), name=name)

"""GLB 读数：三角形、法线、材质颜色、顶点颜色、纹理均色。

这些测试用的都是现搭的 GLB（struct 拼出来的），所以不依赖任何外部资源。
"""

from __future__ import annotations

import io
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.core import glb


def build_glb(doc: dict, binary: bytes = b"") -> bytes:
    """把 dict 和一段 BIN 拼成合法的 GLB。"""
    payload = json.dumps(doc, ensure_ascii=False).encode("utf-8")
    payload += b" " * ((4 - len(payload) % 4) % 4)
    binary = binary + b"\0" * ((4 - len(binary) % 4) % 4)
    total = 12 + 8 + len(payload) + ((8 + len(binary)) if binary else 0)
    out = b"glTF" + struct.pack("<II", 2, total)
    out += struct.pack("<II", len(payload), 0x4E4F534A) + payload
    if binary:
        out += struct.pack("<II", len(binary), 0x004E4942) + binary
    return out


def triangle_doc(**extra) -> tuple[dict, bytes]:
    """一个三角形：POSITION + NORMAL，索引可选。"""
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    normals = np.array([[0, 0, 1]] * 3, dtype=np.float32)
    blob = positions.tobytes() + normals.tobytes()
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"name": "tri", "primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}}]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 36},
            {"buffer": 0, "byteOffset": 36, "byteLength": 36},
        ],
        "buffers": [{"byteLength": len(blob)}],
    }
    doc.update(extra)
    return doc, blob


class GlbTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._folder = tempfile.TemporaryDirectory()
        self.folder = Path(self._folder.name)

    def tearDown(self) -> None:
        self._folder.cleanup()

    def write(self, name: str, blob: bytes) -> Path:
        path = self.folder / name
        path.write_bytes(blob)
        return path


class LoadModelTests(GlbTestCase):
    def test_one_triangle_without_material_is_white(self) -> None:
        doc, blob = triangle_doc()
        model = glb.load_model(self.write("tri.glb", build_glb(doc, blob)))
        self.assertFalse(model.is_empty)
        self.assertEqual(model.triangle_count, 1)
        self.assertEqual(model.vertex_count, 3)
        primitive = model.primitives[0]
        self.assertEqual(primitive.colour, (255, 255, 255))
        self.assertEqual(primitive.material, "默认材质")
        self.assertAlmostEqual(primitive.area, 0.5, places=5)
        np.testing.assert_allclose(primitive.normals[0], [0, 0, 1], atol=1e-6)
        np.testing.assert_allclose(model.low, [0, 0, 0], atol=1e-6)
        np.testing.assert_allclose(model.high, [1, 1, 0], atol=1e-6)

    def test_material_factor_is_converted_from_linear_to_srgb(self) -> None:
        doc, blob = triangle_doc()
        # 线性 0.5 的灰，sRGB 里是 188（0.735 * 255）。
        doc["materials"] = [
            {"name": "灰", "pbrMetallicRoughness": {"baseColorFactor": [0.5, 0.5, 0.5, 1.0]}}
        ]
        doc["meshes"][0]["primitives"][0]["material"] = 0
        model = glb.load_model(self.write("grey.glb", build_glb(doc, blob)))
        self.assertEqual(model.primitives[0].colour, (188, 188, 188))
        self.assertEqual(model.primitives[0].material, "灰")

    def test_node_transform_moves_the_geometry(self) -> None:
        doc, blob = triangle_doc()
        doc["nodes"] = [{"mesh": 0, "translation": [10.0, 0.0, -2.0]}]
        model = glb.load_model(self.write("moved.glb", build_glb(doc, blob)))
        np.testing.assert_allclose(model.low, [10, 0, -2], atol=1e-6)
        np.testing.assert_allclose(model.high, [11, 1, -2], atol=1e-6)

    def test_vertex_colours_become_the_palette(self) -> None:
        doc, blob = triangle_doc()
        colours = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
        blob = blob + colours.tobytes()
        doc["accessors"].append({"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC3"})
        doc["bufferViews"].append({"buffer": 0, "byteOffset": 72, "byteLength": 36})
        doc["buffers"] = [{"byteLength": len(blob)}]
        doc["meshes"][0]["primitives"][0]["attributes"]["COLOR_0"] = 2
        model = glb.load_model(self.write("vcol.glb", build_glb(doc, blob)))
        primitive = model.primitives[0]
        self.assertEqual(primitive.colour, (255, 0, 0))
        self.assertIn((0, 0, 255), primitive.extra_colours)

    def test_indexed_primitive_and_two_materials(self) -> None:
        quad = np.array(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32
        )
        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint16)
        blob = quad.tobytes() + indices.tobytes()
        doc = {
            "asset": {"version": "2.0"},
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0}],
            "meshes": [
                {
                    "primitives": [
                        {"attributes": {"POSITION": 0}, "indices": 1, "material": 0},
                        {"attributes": {"POSITION": 0}, "indices": 1, "material": 1},
                    ]
                }
            ],
            "materials": [
                {"pbrMetallicRoughness": {"baseColorFactor": [1, 0, 0, 1]}},
                {"pbrMetallicRoughness": {"baseColorFactor": [0, 1, 0, 1]}},
            ],
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3"},
                {"bufferView": 1, "componentType": 5123, "count": 6, "type": "SCALAR"},
            ],
            "bufferViews": [
                {"buffer": 0, "byteOffset": 0, "byteLength": 48},
                {"buffer": 0, "byteOffset": 48, "byteLength": 12},
            ],
            "buffers": [{"byteLength": len(blob)}],
        }
        model = glb.load_model(self.write("quad.glb", build_glb(doc, blob)))
        self.assertEqual(model.triangle_count, 4)
        self.assertEqual([p.colour for p in model.primitives], [(255, 0, 0), (0, 255, 0)])
        self.assertAlmostEqual(sum(p.area for p in model.primitives), 2.0, places=5)
        colours = glb.model_colours(model)
        self.assertEqual(len(colours), 2)
        self.assertEqual(sum(item.primitives for item in colours), 2)
        self.assertAlmostEqual(colours[0].area, 1.0, places=5)

    def test_texture_in_buffer_view_gives_the_average_colour(self) -> None:
        from PIL import Image

        picture = Image.new("RGB", (4, 4), (0, 0, 255))
        buffer = io.BytesIO()
        picture.save(buffer, format="PNG")
        png = buffer.getvalue()
        doc, blob = triangle_doc()
        doc["images"] = [{"bufferView": 1, "mimeType": "image/png"}]
        doc["textures"] = [{"source": 0}]
        doc["materials"] = [
            {
                "pbrMetallicRoughness": {
                    "baseColorFactor": [1, 1, 1, 1],
                    "baseColorTexture": {"index": 0},
                }
            }
        ]
        doc["meshes"][0]["primitives"][0]["material"] = 0
        doc["bufferViews"] = [
            {"buffer": 0, "byteOffset": 0, "byteLength": 72},
            {"buffer": 0, "byteOffset": 72, "byteLength": len(png)},
        ]
        blob = blob + png
        doc["buffers"] = [{"byteLength": len(blob)}]
        model = glb.load_model(self.write("tex.glb", build_glb(doc, blob)))
        primitive = model.primitives[0]
        self.assertEqual(primitive.colour, (255, 255, 255))
        self.assertIn((0, 0, 255), primitive.extra_colours)

    def test_gltf_with_embedded_buffer(self) -> None:
        import base64

        doc, blob = triangle_doc()
        doc["buffers"] = [
            {
                "byteLength": len(blob),
                "uri": "data:application/octet-stream;base64," + base64.b64encode(blob).decode("ascii"),
            }
        ]
        path = self.folder / "tri.gltf"
        path.write_text(json.dumps(doc), encoding="utf-8")
        model = glb.load_model(path)
        self.assertEqual(model.triangle_count, 1)

    def test_non_triangle_primitive_is_skipped_with_a_reason(self) -> None:
        doc, blob = triangle_doc()
        doc["meshes"][0]["primitives"][0]["mode"] = 1  # LINES
        model = glb.load_model(self.write("lines.glb", build_glb(doc, blob)))
        self.assertTrue(model.is_empty)
        self.assertTrue(any("不是三角形" in note for note in model.skipped))

    def test_not_a_model_raises(self) -> None:
        path = self.write("broken.glb", b"not a model at all")
        with self.assertRaises(ValueError):
            glb.load_model(path)
        with self.assertRaises(FileNotFoundError):
            glb.load_model(self.folder / "missing.glb")

    def test_truncated_buffer_raises(self) -> None:
        doc, blob = triangle_doc()
        # 只留顶点那 36 字节：法线的 accessor 就落在文件外面了。
        with self.assertRaises(ValueError):
            glb.load_model(self.write("short.glb", build_glb(doc, blob[:36])))


class ColourHelpersTests(unittest.TestCase):
    def test_linear_to_srgb_ends(self) -> None:
        self.assertEqual(glb.linear_to_srgb(np.array(0.0)), 0.0)
        self.assertAlmostEqual(float(glb.linear_to_srgb(np.array(1.0))), 1.0, places=6)
        self.assertAlmostEqual(float(glb.linear_to_srgb(np.array(0.5))), 0.735, places=3)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

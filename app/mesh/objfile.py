"""Write the flat plate as OBJ + MTL.

OBJ is the universal fallback: no slicer-specific metadata, just geometry and
one material per colour region.  Coordinates use the same millimetre, Z-up
frame as the 3MF export, so both files describe the identical object.

Groups are named ``part_01_<hex>`` and the material library is written next to
the ``.obj`` with the same stem, which is what every viewer expects.
"""

from __future__ import annotations

from pathlib import Path

from .plate import PlateModel

__all__ = ["write_obj"]


def _rgb_floats(color_hex: str) -> tuple[float, float, float]:
    text = color_hex.lstrip("#")
    if len(text) != 6:
        return (1.0, 1.0, 1.0)
    return tuple(int(text[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def _fmt(value: float) -> str:
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-0") else "0"


def _safe(name: str) -> str:
    keep = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in name)
    return keep[:60] or "part"


def write_obj(plate: PlateModel, path: str | Path) -> Path:
    """Write ``plate`` as ``.obj`` plus a sibling ``.mtl`` and return the OBJ path."""
    if not plate.parts:
        raise ValueError("nothing to export: the plate has no parts")

    target = Path(path)
    if target.suffix.lower() != ".obj":
        target = target.with_suffix(".obj")
    target.parent.mkdir(parents=True, exist_ok=True)
    mtl_name = target.with_suffix(".mtl").name

    vertex_lines: list[str] = []
    face_lines: list[str] = []
    offset = 1  # OBJ indices are 1-based and shared across groups
    for part in plate.parts:
        vertex_lines.append(f"# {part.name}  {part.color_hex}  extruder {part.extruder}")
        for x, y, z in part.vertices:
            vertex_lines.append(f"v {_fmt(x)} {_fmt(y)} {_fmt(z)}")
        material = _safe(f"part_{part.palette_index + 1:02d}_{part.color_hex.lstrip('#')}")
        face_lines.append(f"g {_safe(part.name)}")
        face_lines.append(f"usemtl {material}")
        for a, b, c in part.triangles:
            face_lines.append(f"f {int(a) + offset} {int(b) + offset} {int(c) + offset}")
        offset += int(len(part.vertices))

    body = [
        "# BambuPalette — flat multi-material plate",
        f"# size {_fmt(plate.width_mm)} x {_fmt(plate.depth_mm)} mm, "
        f"thickness {_fmt(plate.total_thickness_mm)} mm, {len(plate.parts)} parts",
        f"mtllib {mtl_name}",
        *vertex_lines,
        *face_lines,
        "",
    ]
    target.write_text("\n".join(body), encoding="utf-8")

    mtl_lines = ["# BambuPalette — one material per colour region", ""]
    for part in plate.parts:
        red, green, blue = _rgb_floats(part.color_hex)
        mtl_lines.extend(
            [
                f"newmtl {_safe(f'part_{part.palette_index + 1:02d}_{part.color_hex.lstrip('#')}')}",
                f"# {part.name}  (extruder {part.extruder})",
                f"Ka {_fmt(red)} {_fmt(green)} {_fmt(blue)}",
                f"Kd {_fmt(red)} {_fmt(green)} {_fmt(blue)}",
                "Ks 0 0 0",
                "Ns 0",
                "d 1",
                "illum 1",
                "",
            ]
        )
    target.with_suffix(".mtl").write_text("\n".join(mtl_lines), encoding="utf-8")
    return target

"""Verify the extracted ICC polynomial coefficients against the binary .icc profile.

The 31x20 coefficient table used by the engine is parsed out of FullSpectrum's
generated C++ header.  This script independently locates the same table inside
the original ``xyz2PolyEstimateRefV2.icc`` profile and compares every value, so a
bug in either extraction path cannot go unnoticed.

Run:  .venv\\Scripts\\python.exe tools\\verify_icc_profile.py
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.spectral import icc_profile  # noqa: E402

ICC_PATH = ROOT / "tools" / "reference" / "xyz2PolyEstimateRefV2.icc"


def _f32(value: float) -> float:
    """Round a Python double through binary32, as a C++ ``float`` literal is stored."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def list_tags(data: bytes) -> list[tuple[str, int, int, int]]:
    tag_count = struct.unpack(">I", data[128:132])[0]
    tags = []
    for index in range(tag_count):
        base = 132 + index * 12
        signature, offset, size = struct.unpack(">4sII", data[base:base + 12])
        tags.append((signature.decode("latin-1"), offset, size, index))
    return tags


def read_tag_text(data: bytes, offset: int, size: int) -> str:
    """Decode an ICC ``textDescription``/``text`` tag payload into a Python string."""
    block = data[offset:offset + size]
    if len(block) < 12:
        return ""
    tag_type = block[:4]
    if tag_type == b"desc":
        length = struct.unpack(">I", block[8:12])[0]
        return block[12:12 + max(0, length - 1)].decode("latin-1")
    if tag_type == b"text":
        return block[8:].decode("latin-1").rstrip("\x00")
    if tag_type == b"mluc":
        count = struct.unpack(">I", block[8:12])[0]
        out = []
        for index in range(count):
            base = 16 + index * 12
            length, lang_offset = struct.unpack(">II", block[base:base + 8])
            start = base + lang_offset
            out.append(block[start:start + length].decode("utf-16-be", "replace"))
        return " / ".join(out)
    return f"<{tag_type.decode('latin-1')}>"


def main() -> int:
    if not ICC_PATH.exists():
        print(f"missing profile: {ICC_PATH}")
        return 1
    data = ICC_PATH.read_bytes()
    print(f"profile      : {ICC_PATH.name} ({len(data)} bytes)")

    tags = list_tags(data)
    print(f"tags         : {len(tags)}")
    for signature, offset, size, _ in tags:
        print(f"   {signature!r:>8}  offset={offset:<7} size={size}")
        if signature in ("desc", "cprt"):
            print(f"        text -> {read_tag_text(data, offset, size)!r}")

    table = getattr(icc_profile, "COEFFICIENTS")
    wavelengths, terms = len(table), len(table[0])
    flat = [value for row in table for value in row]
    print(f"header table : {wavelengths} x {terms} = {len(flat)} coefficients")

    # Locate the table by its first three contiguous values, trying both byte orders.
    head = [_f32(v) for v in flat[:3]]
    found = None
    for endian in ("<", ">"):
        pattern = b"".join(struct.pack(endian + "f", v) for v in head)
        offset = data.find(pattern)
        if offset >= 0:
            found = (offset, endian)
            break

    if found is None:
        print("\nRESULT: the first three header coefficients were NOT found verbatim.")
        return 2

    offset, endian = found
    order = "little" if endian == "<" else "big"
    print(f"\nlocated      : byte offset {offset}, {order}-endian float32")

    block = data[offset:offset + len(flat) * 4]
    values = struct.unpack(endian + f"{len(flat)}f", block)
    mismatches = [(i, flat[i], values[i]) for i in range(len(flat)) if _f32(flat[i]) != values[i]]

    print(f"compared     : {len(values)} coefficients, {len(mismatches)} differ")

    # How many coefficients can we read before the block stops looking like the table?
    run = 0
    for i in range(len(flat)):
        if _f32(flat[i]) != values[i]:
            break
        run += 1
    print(f"identical run: {run} / {len(flat)} from the located offset")

    if not mismatches:
        print("\nRESULT: OK - the header table is byte-identical to the profile's own table.")
        print("        Order is [wavelength][term] with terms")
        print("        1, X, Y, Z, X^2, Y^2, Z^2, XY, XZ, YZ, X^3, Y^3, Z^3,")
        print("        X^2Y, X^2Z, XY^2, XZ^2, Y^2Z, YZ^2, XYZ.")
        return 0

    for index, a, b in mismatches[:8]:
        print(f"   [{index:4d}] header={a!r} profile={b!r}")
    print("\nRESULT: MISMATCH - the header and the profile disagree.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

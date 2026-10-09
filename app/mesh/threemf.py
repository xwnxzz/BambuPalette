"""Write the flat plate as a Bambu Studio compatible 3MF.

Two files carry everything Bambu Studio needs:

``3D/3dmodel.model``
    One **core 3MF** document holding a single mesh object.  The base slab and
    every colour region are concatenated into that one mesh, and each triangle
    carries Bambu's ``paint_color`` code naming the filament it prints with.
    This is how a real multi-colour project stores its colours: not one part per
    colour, but one object whose triangles are painted.

``Metadata/project_settings.config``
    The project's filament list — ``filament_colour`` and friends.  Without it
    Bambu Studio paints the model with whatever spools happen to sit in the AMS,
    which is why an early build of this exporter showed a plate in the wrong
    colours even though the geometry was right.

An earlier version emitted one ``<part>`` per colour and relied purely on
``Metadata/model_settings.config``'s per-part ``extruder``.  That works for the
slicer but shows up in Bambu Studio as a stack of child objects (``底板``,
``01_…``, ``02_…``) rather than one painted plate, and — because the package had
no filament list at all — it rendered in the AMS slots' colours.

An even earlier version copied the layout of a project file Bambu Studio itself
had exported (``BambuStudio-02.06.00.51``): one ``3D/Objects/object_N.model`` per
part, pulled in through ``<component p:path=…>`` under the 3MF production
extension, plus ``p:UUID`` bookkeeping.  Round-tripping that through the
installed Bambu Studio (``02.08.02.61``) proved it does not work — the reader
kept the metadata and threw every component away, re-exporting a project whose
assembly object had an empty ``<components/>`` and ``face_count="0"``.  Core 3MF
with inline meshes round-trips correctly, which is what
``tools/verify_bambu_roundtrip.py`` checks.

Because a 3MF package is just a zip, everything is written with the standard
library.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from xml.sax.saxutils import escape

from .plate import PlateModel

__all__ = [
    "write_3mf",
    "FilamentSlot",
    "slots_from_palette",
    "THREEMF_APPLICATION",
    "DEFAULT_PLATE_MM",
    "PAINT_CODES",
    "MAX_PAINTED_FILAMENTS",
    "paint_code",
]

#: Recorded in ``<metadata name="Application">``.  Kept honest instead of
#: impersonating Bambu Studio; the importer keys off the standard parts below.
THREEMF_APPLICATION = "BambuPalette"

#: Bambu's printers all ship a 256 mm build plate, so that is where the object
#: is centred.  A larger plate simply leaves more room around it.
DEFAULT_PLATE_MM = 256.0

#: Bambu Studio's per-triangle multi-material code, transcribed verbatim from
#: ``CONST_FILAMENTS`` in ``src/libslic3r/Model.cpp`` (tag ``v02.08.02.61``,
#: lines 57-60).  Index == the 1-based filament slot.
#:
#: A triangle that belongs entirely to one filament is encoded by this string
#: alone: ``FacetsAnnotation::get_triangle_as_string`` (``Model.cpp:4605``)
#: emits straight from the table for its ``_3_SAME_COLOR`` case, and a real
#: Makerworld project that Bambu Studio itself exported carries exactly
#: ``paint_color="4"`` for its white filament, ``"0C"`` for black and ``"1C"``
#: for yellow — the 1st, 3rd and 4th entries of its ``filament_colour`` list.
#:
#: The table is irregular (``1 -> 4``, ``2 -> 8``, then ``3 -> 0C`` rising) and
#: stops at 32 filaments, so it is copied literally rather than computed.
PAINT_CODES = (
    "",
    "4", "8", "0C", "1C", "2C", "3C", "4C", "5C", "6C", "7C", "8C", "9C",
    "AC", "BC", "CC", "DC", "EC", "0FC", "1FC", "2FC", "3FC", "4FC", "5FC",
    "6FC", "7FC", "8FC", "9FC", "AFC", "BFC", "CFC", "DFC", "EFC",
)

#: Bambu Studio's own table has no entry past filament 32, and
#: ``get_real_filament_id`` logs "CONST_FILAMENTS out of array" and returns an
#: empty string there.  Both the matcher and this writer stay inside that range.
MAX_PAINTED_FILAMENTS = len(PAINT_CODES) - 1


def paint_code(extruder: int) -> str:
    """Bambu Studio's ``paint_color`` string for a 1-based filament slot."""
    slot = int(extruder)
    if 1 <= slot <= MAX_PAINTED_FILAMENTS:
        return PAINT_CODES[slot]
    raise ValueError(
        f"耗材槽 {extruder} 超出 1..{MAX_PAINTED_FILAMENTS}，"
        "Bambu Studio 的 paint_color 表里没有对应的编码。"
    )


@dataclass(frozen=True)
class FilamentSlot:
    """One entry of the project's filament list.

    ``color_hex`` is the colour the plate is painted with; the rest is what
    Bambu Studio shows in its 项目耗材列表 so the user recognises the spool.
    """

    color_hex: str
    material_type: str = "PLA"
    vendor: str = ""
    name: str = ""


CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml" />
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml" />
  <Default Extension="png" ContentType="image/png" />
  <Default Extension="gcode" ContentType="text/x.gcode" />
</Types>
"""

RELS = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Target="/3D/3dmodel.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel" />
</Relationships>
"""

#: The identity matrix Bambu Studio writes into a ``<part>`` that sits at the
#: origin.  It is column-major 4x4, which is what its reader expects.
IDENTITY_MATRIX = "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"


def _fmt(value: float) -> str:
    """Shortest decimal that still round-trips the millimetre coordinates."""
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-0") else "0"


def _attr(value: object) -> str:
    """Escape text for a double-quoted XML attribute.

    ``xml.sax.saxutils.escape`` handles ``&``, ``<`` and ``>`` but leaves the
    quote alone, and a filament name is free text the user can type a quote into.
    """
    return escape(str(value), {'"': "&quot;"})


# ---------------------------------------------------------------------------
# The mesh: every part concatenated, each triangle remembering its filament
# ---------------------------------------------------------------------------
def _merged_mesh(
    plate: PlateModel, translate: tuple[float, float, float]
) -> tuple[str, str, int]:
    """Return ``(vertex_rows, triangle_rows, vertex_count)`` for the whole plate.

    The base slab and every colour region end up in **one** object.  Their
    vertices are concatenated with an index offset and each triangle is tagged
    twice: with the core-3MF material it uses (``pid``/``p1`` into the
    ``<basematerials>`` table) and with Bambu Studio's own ``paint_color`` code.
    Carrying both means the plate is a single printable solid painted in several
    colours, and whichever colour mechanism the reader understands is present.
    """
    dx, dy, dz = translate
    vertex_rows: list[str] = []
    triangle_rows: list[str] = []
    offset = 0
    for part in plate.parts:
        code = paint_code(part.extruder)
        material = part.extruder - 1
        for x, y, z in part.vertices:
            vertex_rows.append(
                f'          <vertex x="{_fmt(x + dx)}" y="{_fmt(y + dy)}" z="{_fmt(z + dz)}" />\n'
            )
        for a, b, c in part.triangles:
            triangle_rows.append(
                f'          <triangle v1="{int(a) + offset}" v2="{int(b) + offset}" '
                f'v3="{int(c) + offset}" pid="{BASEMATERIALS_ID}" p1="{material}" '
                f'paint_color="{code}" />\n'
            )
        offset += int(len(part.vertices))
    return "".join(vertex_rows), "".join(triangle_rows), offset


#: The single object id used by both the model document and the settings file.
OBJECT_ID = 1

#: The id of the ``<basematerials>`` table every triangle points into.
BASEMATERIALS_ID = 99


def _model_document(
    plate: PlateModel, translate: tuple[float, float, float], slots: Sequence[FilamentSlot]
) -> str:
    """One core-3MF document holding the painted plate as a single object.

    Plain core 3MF — the mesh inline, no ``requiredextensions``, no ``p:path``,
    no ``p:UUID`` — is understood everywhere, which the production-extension
    layout demonstrably was not.  Colours travel twice: as standard
    ``<basematerials>`` entries with a ``pid``/``p1`` per triangle, which is how
    any 3MF reader is meant to colour a mesh, and as Bambu Studio's own
    ``paint_color`` on the triangles, exactly as its multi-colour projects store
    them.
    """
    vertex_rows, triangle_rows, vertex_count = _merged_mesh(plate, translate)
    materials = "".join(
        f'      <base name="{_attr(slot.name or f"耗材 {index + 1}")}" '
        f'displaycolor="{slot.color_hex.upper()}FF" />\n'
        for index, slot in enumerate(slots)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<model unit="millimeter" xml:lang="en-US" '
        'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">\n'
        f'  <metadata name="Application">{_attr(THREEMF_APPLICATION)}</metadata>\n'
        '  <metadata name="BambuStudio:3mfVersion">1</metadata>\n'
        "  <resources>\n"
        f'    <basematerials id="{BASEMATERIALS_ID}">\n'
        f"{materials}"
        "    </basematerials>\n"
        f'    <object id="{OBJECT_ID}" type="model">\n'
        "      <mesh>\n"
        "        <vertices>\n"
        f"{vertex_rows}"
        "        </vertices>\n"
        "        <triangles>\n"
        f"{triangle_rows}"
        "        </triangles>\n"
        "      </mesh>\n"
        "    </object>\n"
        "  </resources>\n"
        "  <build>\n"
        f'    <item objectid="{OBJECT_ID}" printable="1" />\n'
        "  </build>\n"
        "</model>\n"
    )


def _model_settings(plate: PlateModel, object_name: str) -> str:
    """``Metadata/model_settings.config`` for a single painted object.

    Modelled on a real Bambu Studio export: the object carries one
    ``<part subtype="normal_part">``, the object-level ``extruder`` is the
    fallback for triangles without a ``paint_color``, and the ``<plate>`` block
    records the filament map for the plate.
    """
    triangles = plate.triangle_count
    slots = plate.extruder_count or 1
    base_extruder = plate.parts[0].extruder if plate.parts else 1
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<config>\n"
        f'  <object id="{OBJECT_ID}">\n'
        f'    <metadata key="name" value="{_attr(object_name)}" />\n'
        f'    <metadata key="extruder" value="{base_extruder}" />\n'
        f'    <metadata face_count="{triangles}" />\n'
        f'    <part id="1" subtype="normal_part">\n'
        f'      <metadata key="name" value="{_attr(object_name)}" />\n'
        f'      <metadata key="matrix" value="{IDENTITY_MATRIX}" />\n'
        f'      <mesh_stat face_count="{triangles}" edges_fixed="0" '
        'degenerate_facets="0" facets_removed="0" facets_reversed="0" '
        'backwards_edges="0" />\n'
        "    </part>\n"
        "  </object>\n"
        "  <plate>\n"
        '    <metadata key="plater_id" value="1" />\n'
        '    <metadata key="plater_name" value="Plate 1" />\n'
        '    <metadata key="locked" value="false" />\n'
        '    <metadata key="filament_map_mode" value="Auto For Flush" />\n'
        f'    <metadata key="filament_maps" value="{" ".join(["1"] * slots)}" />\n'
        f'    <metadata key="filament_volume_maps" value="{" ".join(["0"] * slots)}" />\n'
        "    <model_instance>\n"
        f'      <metadata key="object_id" value="{OBJECT_ID}" />\n'
        '      <metadata key="instance_id" value="0" />\n'
        '      <metadata key="identify_id" value="76" />\n'
        "    </model_instance>\n"
        "  </plate>\n"
        "</config>\n"
    )


# ---------------------------------------------------------------------------
# The project's filament list
# ---------------------------------------------------------------------------
def slots_from_palette(palette: Sequence[object], library: object | None = None) -> list[FilamentSlot]:
    """Build the project's filament list from a match palette.

    ``library`` is only consulted to answer "which spool is this?" for a mix —
    a mix is two spools of the same 耗材种类, so its type and vendor are the
    parents'.  Everything is read through ``getattr`` so this module stays free
    of an import back into ``app.core``.
    """
    slots: list[FilamentSlot] = []
    for entry in palette:
        filament = getattr(entry, "filament", None)
        material_type = getattr(filament, "material_type", "") or ""
        vendor = getattr(filament, "brand", "") or ""
        if not material_type and library is not None:
            getter = getattr(library, "get", None)
            for filament_id in getattr(entry, "filament_ids", ()) or ():
                parent = getter(filament_id) if callable(getter) else None
                if parent is not None:
                    material_type = parent.material_type or ""
                    vendor = vendor or (parent.brand or "")
                    break
        slots.append(
            FilamentSlot(
                color_hex=getattr(entry, "color_hex", "#FFFFFF"),
                material_type=material_type or "PLA",
                vendor=vendor,
                name=getattr(entry, "label", "") or "",
            )
        )
    return slots


def _project_filaments(
    plate: PlateModel, filaments: Sequence[FilamentSlot] | None
) -> list[FilamentSlot]:
    """One :class:`FilamentSlot` per extruder, in extruder order.

    ``filaments`` is indexed the same way a ``MatchResult``'s palette is: entry
    ``i`` belongs to extruder ``i + 1``.  Anything missing falls back to the
    part's own colour so the plate is never painted with a wrong hex.
    """
    count = max(1, plate.extruder_count)
    slots: list[FilamentSlot | None] = [None] * count
    for part in plate.parts:
        index = part.extruder - 1
        if not 0 <= index < count:
            continue
        given = filaments[index] if filaments is not None and index < len(filaments) else None
        if given is None:
            slots[index] = FilamentSlot(color_hex=part.color_hex)
        else:
            slots[index] = FilamentSlot(
                color_hex=given.color_hex or part.color_hex,
                material_type=given.material_type or "PLA",
                vendor=given.vendor,
                name=given.name,
            )
    return [slot or FilamentSlot(color_hex="#FFFFFF") for slot in slots]


def _project_settings(slots: Sequence[FilamentSlot]) -> str:
    """``Metadata/project_settings.config`` — the project's filament list.

    Bambu Studio reads ``filament_colour`` from here for projects it considers
    its own — that is, files whose ``Application`` metadata starts with
    ``BambuStudio``.  A file imported from another program is read as a plain
    3MF, so Bambu Studio takes no colours from this file; the colours that do
    travel are the ``<basematerials>`` and ``paint_color`` attributes on the mesh
    itself.  This file is written anyway because it costs nothing, it documents
    the intended spool for every slot, and readers that do honour it then get the
    right answer.

    Only the keys this program can answer for are written — everything else
    (printer, process, g-code) falls back to Bambu Studio's own defaults.

    A warning to anyone tempted to "fix" this by renaming the Application
    metadata to ``BambuStudio-…``: Bambu Studio then treats the file as its own
    project, and with a filament list it did not write itself it aborts with an
    out-of-memory crash (reproduced three times against 02.08.02.61).  Shipping a
    file that crashes the slicer is far worse than a file whose colours have to
    be picked once in the import dialog.
    """
    count = len(slots)
    colours = [slot.color_hex for slot in slots]
    payload = {
        "filament_colour": colours,
        "filament_multi_colour": colours,
        "filament_colour_type": ["0"] * count,
        "filament_type": [slot.material_type or "PLA" for slot in slots],
        "filament_vendor": [slot.vendor for slot in slots],
        "filament_settings_id": [slot.name or f"BambuPalette {i + 1}" for i, slot in enumerate(slots)],
        "filament_ids": [""] * count,
        "filament_diameter": ["1.75"] * count,
        "filament_flow_ratio": ["0.98"] * count,
        "filament_map_mode": "Auto For Flush",
        "filament_map": ["1"] * count,
        "filament_volume_map": ["0"] * count,
        "filament_self_index": [str(i + 1) for i in range(count)],
        "filament_prime_volume": ["30"] * count,
        "default_filament_colour": [""] * count,
        "nozzle_diameter": ["0.4"],
        "extruder_colour": ["#018001"],
        "master_extruder_id": "1",
        "physical_extruder_map": ["0"],
        "single_extruder_multi_material": "1" if count > 1 else "0",
        "enable_prime_tower": "1" if count > 1 else "0",
    }
    return json.dumps(payload, indent=4, ensure_ascii=False) + "\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def write_3mf(
    plate: PlateModel,
    path: str | Path,
    *,
    object_name: str = "",
    plate_size_mm: float = DEFAULT_PLATE_MM,
    filaments: Sequence[FilamentSlot] | None = None,
) -> Path:
    """Write ``plate`` as a Bambu Studio project and return the written path."""
    if not plate.parts:
        raise ValueError("模型里没有任何零件，无法导出 3MF。")
    if plate.extruder_count > MAX_PAINTED_FILAMENTS:
        raise ValueError(
            f"这个模型用了 {plate.extruder_count} 种耗材，"
            f"超过 Bambu Studio 能识别的 {MAX_PAINTED_FILAMENTS} 种。"
        )

    target = Path(path)
    if target.suffix.lower() != ".3mf":
        target = target.with_suffix(".3mf")
    target.parent.mkdir(parents=True, exist_ok=True)

    # Centre the footprint on the bed.  The plate is built in millimetres with
    # its own origin at a corner, so the shift is half the leftover space.
    dx = (plate_size_mm - plate.width_mm) / 2.0
    dy = (plate_size_mm - plate.depth_mm) / 2.0
    translate = (dx, dy, 0.0)

    name = object_name or target.stem
    slots = _project_filaments(plate, filaments)

    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", CONTENT_TYPES)
        archive.writestr("_rels/.rels", RELS)
        archive.writestr("3D/3dmodel.model", _model_document(plate, translate, slots))
        archive.writestr(
            "Metadata/model_settings.config", _model_settings(plate, name)
        )
        archive.writestr("Metadata/project_settings.config", _project_settings(slots))
        archive.writestr(
            "Metadata/filament_sequence.json",
            json.dumps(
                {"plate_1": {"nozzle_sequence": [], "optimal_assignment": [], "sequence": []}}
            ),
        )
        archive.writestr(
            "Metadata/slice_info.config",
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<config><header>"
            f'<header_item key="X-BBL-Client-Type" value="{_attr(THREEMF_APPLICATION)}" />'
            # A Bambu Studio compatibility marker (the client version its own
            # parser expects), NOT BambuPalette's version — that lives in
            # app/__init__.py and is reported by --selftest.
            '<header_item key="X-BBL-Client-Version" value="1.0.0" />'
            "</header></config>\n",
        )
    return target

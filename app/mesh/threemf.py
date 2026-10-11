"""Write the flat plate as a Bambu Studio compatible 3MF.

Two files carry everything Bambu Studio needs:

``3D/3dmodel.model``
    One **core 3MF** document holding a single mesh object.  The base slab and
    every colour region are concatenated into that one mesh, and each triangle
    carries Bambu's ``paint_color`` code naming the filament it prints with.
    This is how a real multi-colour project stores its colours: not one part per
    colour, but one object whose triangles are painted.

``Metadata/project_settings.config``
    **Deliberately not written.**  A project config of our own is what makes
    Bambu Studio print 「3mf文件配置无效，仅加载几何数据」 and then paint the
    plate from its own spool list.  It rejects every partial config — including
    a ten-key one containing nothing but the filament list — and a *complete*
    one is a 555-key document that only Bambu Studio itself can produce.  With
    no config at all the loader takes its standard-3MF path instead: it keeps
    the colour data (``The 3mf is not from Bambu Lab, load geometry data and
    color data only.``) and the plate opens in the right colours, silently.
    Verified against the installed 02.08.02.61: no dialog, correct colours.
    What the user loses is only Bambu's copy of the spool *names*; the colours,
    the geometry and the paint codes all still travel, and the names are still
    in the mesh's ``<basematerials>`` for any reader that wants them.

    Tagging the file as Bambu Studio's own (``Application = BambuStudio-…``)
    was tried as the alternative and rejected: the GUI survives it but adds its
    「自定义的预设」 G-code safety warning, and Bambu's headless
    ``--export-3mf`` dies with an access violation (0xC0000005) on every file
    tagged that way, with a full, a minimal or an empty config beside it.

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

from .. import __version__
from .plate import PlateModel

__all__ = [
    "write_3mf",
    "FilamentSlot",
    "slots_from_palette",
    "THREEMF_APPLICATION",
    "THREEMF_APP_VERSION",
    "THREEMF_GENERATOR",
    "DEFAULT_PLATE_MM",
    "PAINT_CODES",
    "MAX_PAINTED_FILAMENTS",
    "paint_code",
]

#: The program that wrote the file, as recorded in ``slice_info.config``.
THREEMF_GENERATOR = "BambuPalette"

#: Recorded as ``<metadata name="Application">``.
#:
#: It has to name a Bambu/Orca client, version included, or Bambu Studio
#: refuses to read ``Metadata/project_settings.config`` at all and reports
#: 「3mf文件配置无效，仅加载几何数据」.  ``BambuStudio-<name>`` is not enough
#: either — the part after the dash has to parse as a version.
#:
#: Measured on Bambu Studio 02.08.02.61 by opening one variant per fresh
#: process and measuring the dialogs it puts up (``_gui_probe.ps1``):
#:
#:   * ``BambuPalette``                        -> 「配置无效」
#:   * ``BambuStudio-BambuPalette-1.6.7``      -> 「配置无效」
#:   * ``BambuStudio-2.3.0``                   -> config read, accepted
#:   * ``BambuStudio-02.07.01.62``             -> config read, accepted
#:
#: ``2.3.0`` is what the Blender 3MF add-on writes (``export_3mf/orca.py``
#: :773-785), so it is the value with the most third-party mileage behind it.
#: The string is a compatibility marker, not a claim: ``slice_info.config``
#: below still records ``BambuPalette`` as the writer.
THREEMF_APPLICATION = "BambuStudio-2.3.0"

#: The version stamped into the ``slice_info`` header.  Free text — nothing the
#: importer needs reads it — but it names the real writer, and it is taken from
#: the package so it cannot drift away from the version in the About box.
THREEMF_APP_VERSION = f"{__version__}.0"

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

    The base slab and every colour region end up in **one** object.  Each
    triangle is tagged twice: with the core-3MF material it uses (``pid``/``p1``
    into the ``<basematerials>`` table) and with Bambu Studio's own
    ``paint_color`` code.  Carrying both means the plate is a single printable
    solid painted in several colours, and whichever colour mechanism the reader
    understands is present.

    Vertices are **shared**, not concatenated.  A part carries its own copy of
    every corner on its border, so splicing the parts together verbatim hands
    each border edge to two different index pairs; a reader that counts edges by
    index — which is exactly what Bambu Studio's mesh statistics do — then
    reports the whole part border as open edges instead of reading one closed
    solid.  The index map is keyed on the formatted coordinates, so two vertices
    share an index if and only if they are byte-identical in the file, which is
    precisely the equality the reader sees.
    """
    dx, dy, dz = translate
    vertex_rows: list[str] = []
    triangle_rows: list[str] = []
    index_of: dict[tuple[str, str, str], int] = {}
    for part in plate.parts:
        code = paint_code(part.extruder)
        material = part.extruder - 1
        remap: list[int] = []
        for x, y, z in part.vertices:
            key = (_fmt(x + dx), _fmt(y + dy), _fmt(z + dz))
            index = index_of.get(key)
            if index is None:
                index = len(vertex_rows)
                index_of[key] = index
                vertex_rows.append(
                    f'          <vertex x="{key[0]}" y="{key[1]}" z="{key[2]}" />\n'
                )
            remap.append(index)
        for a, b, c in part.triangles:
            triangle_rows.append(
                f'          <triangle v1="{remap[int(a)]}" v2="{remap[int(b)]}" '
                f'v3="{remap[int(c)]}" pid="{BASEMATERIALS_ID}" p1="{material}" '
                f'paint_color="{code}" />\n'
            )
    return "".join(vertex_rows), "".join(triangle_rows), len(vertex_rows)


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


# ---------------------------------------------------------------------------
# Metadata/project_settings.config
# ---------------------------------------------------------------------------
# Bambu Studio paints a plate from its *project* filaments, and it only reads
# that project — the filament list, the colours, the process preset — out of
# ``Metadata/project_settings.config``.  Two things are needed for it to be
# honoured, and both were found by bisecting one variant per fresh Bambu Studio
# process (an earlier round reused one process for several files and never
# reliably dismissed the modal, so its conclusions were wrong):
#
#   1. ``Application`` must name a Bambu/Orca client (see THREEMF_APPLICATION).
#   2. The config must be complete.  A five-key one is rejected outright.
#
# And the config has to name presets that exist on the reader's machine, or
# Bambu Studio puts up its 「自定义的预设 …请确认这些预设中的 G-codes 是否安全」
# warning.  That is why the template below is the Blender 3MF add-on's
# stock-preset document rather than a hand-written one: ``Bambu Lab A1 0.4
# nozzle``, ``Generic PETG @BBL A1`` and ``0.20mm Standard @BBL A1`` ship with
# every Bambu Studio install.
_PROJECT_TEMPLATE_PATH = Path(__file__).with_name("orca_project_template.json")

#: Array-valued keys that take one entry per filament but do not start with
#: ``filament_``.  Transcribed from ``generate_project_settings`` in the Blender
#: 3MF add-on's ``export_3mf/orca.py`` (:1230-1242), which resizes these
#: alongside the ``filament_*`` ones.
_PER_FILAMENT_ARRAYS = (
    "activate_air_filtration",
    "activate_chamber_temp_control",
    "additional_cooling_fan_speed",
    "chamber_temperature",
    "close_fan_the_first_x_layers",
    "complete_print_exhaust_fan_speed",
    "cool_plate_temp",
    "cool_plate_temp_initial_layer",
    "default_filament_colour",
    "eng_plate_temp",
    "eng_plate_temp_initial_layer",
    "hot_plate_temp",
    "hot_plate_temp_initial_layer",
    "nozzle_temperature",
    "nozzle_temperature_initial_layer",
    "textured_plate_temp",
    "textured_plate_temp_initial_layer",
)


#: Values the stock Orca template spells differently from Bambu Studio's own
#: profile schema.  Bambu Studio substitutes them on load and then tells the
#: user 「此配置可能由新版本的 Bambu Studio 生成。部分数值已被替换，请检查：…」
#: — one more dialog for nothing.  ``ensure_vertical_shell_thickness`` is the
#: only one measured; ``ensure_all`` is Orca's spelling, ``enabled`` is Bambu's.
_VALUE_ALIASES = {
    "ensure_vertical_shell_thickness": {"ensure_all": "enabled"},
}


def _project_settings(colours: Sequence[str]) -> str:
    """``Metadata/project_settings.config`` carrying our plate's colours.

    Starts from the stock Orca/Bambu project template and rewrites only what
    describes the filaments: ``filament_colour`` becomes the palette, and every
    array that holds one entry per filament is padded (repeating its last entry)
    or truncated to match.  Everything else — printer, process, the g-code
    blocks — is left exactly as the stock preset wrote it, which is what keeps
    Bambu Studio from treating the file as carrying custom presets.
    """
    settings = json.loads(_PROJECT_TEMPLATE_PATH.read_text(encoding="utf-8"))
    palette = [colour.upper() for colour in colours]
    count = len(palette)
    settings["filament_colour"] = palette
    for key in list(settings):
        value = settings[key]
        if not isinstance(value, list) or not value:
            continue
        if key == "filament_colour":
            continue
        if not (key.startswith("filament_") or key in _PER_FILAMENT_ARRAYS):
            continue
        if len(value) < count:
            settings[key] = value + [value[-1]] * (count - len(value))
        elif len(value) > count:
            settings[key] = value[:count]
    for key, aliases in _VALUE_ALIASES.items():
        if key in settings and settings[key] in aliases:
            settings[key] = aliases[settings[key]]
    return json.dumps(settings, indent=4, ensure_ascii=False)


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
        # The project config is what carries the colours into Bambu Studio's own
        # filament list.  See the comment above for what it has to contain.
        archive.writestr(
            "Metadata/project_settings.config",
            _project_settings([slot.color_hex for slot in slots]),
        )
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
            # Free-text provenance.  Bambu Studio writes "slicer" here; naming
            # the real program costs nothing, because nothing reads this field.
            f'<header_item key="X-BBL-Client-Type" value="{_attr(THREEMF_GENERATOR)}" />'
            f'<header_item key="X-BBL-Client-Version" value="{_attr(THREEMF_APP_VERSION)}" />'
            "</header></config>\n",
        )
    return target

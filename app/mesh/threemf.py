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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence
from xml.sax.saxutils import escape

from .. import __version__
from ..core.atomicio import atomic_path
from .plate import PlateModel

__all__ = [
    "write_3mf",
    "FilamentComponent",
    "FilamentSlot",
    "ProjectFilaments",
    "project_filaments",
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
class FilamentComponent:
    """One physical spool a mixed colour is made of.

    ``percent`` is the share of that spool in the mix, 0-100, in the same order
    as the recipe.  Bambu Studio records it as ``filament_mixed_sublayer_ratios``
    (``0.3800,0.6200`` for 38% + 62%), so it is carried here as the number the
    rest of the program speaks in and formatted on the way out.
    """

    color_hex: str
    material_type: str = "PLA"
    vendor: str = ""
    name: str = ""
    percent: float = 0.0


@dataclass(frozen=True)
class FilamentSlot:
    """One entry of the project's filament list.

    ``color_hex`` is the colour the plate is painted with; the rest is what
    Bambu Studio shows in its 项目耗材列表 so the user recognises the spool.

    ``components`` is what turns the slot into a 混色耗材: the spools whose
    sublayers make this colour, in recipe order.  A plain spool leaves it empty.
    """

    color_hex: str
    material_type: str = "PLA"
    vendor: str = ""
    name: str = ""
    components: tuple[FilamentComponent, ...] = ()

    @property
    def is_mixed(self) -> bool:
        return bool(self.components)


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

#: ``3D/_rels/3dmodel.model.rels`` —— 模型文档指向这两份配置的关系表。
#: Bambu Studio 靠它把 ``Metadata/*.config`` 认成自己的工程文件；缺了它，整包只会被
#: 当成一个「只有几何的 3MF」，双击在 Bambu Studio 里打不开（m26604 第 4 条）。
MODEL_RELS = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Target="/Metadata/model_settings.config" Id="rel-1" Type="http://schemas.bambulab.com/package/2021/model_settings" />
  <Relationship Target="/Metadata/project_settings.config" Id="rel-2" Type="http://schemas.bambulab.com/package/2021/project_settings" />
</Relationships>
"""


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


def _model_settings(
    plate: PlateModel, object_name: str, filament_count: int | None = None
) -> str:
    """``Metadata/model_settings.config`` for a single painted object.

    Modelled on a real Bambu Studio export: the object carries one
    ``<part subtype="normal_part">``, the object-level ``extruder`` is the
    fallback for triangles without a ``paint_color``, and the ``<plate>`` block
    records the filament map for the plate — which is one entry per row of the
    project's filament list, so it counts ``filament_count`` (the mixed colours'
    parents included) and not just the painted extruders.
    """
    triangles = plate.triangle_count
    slots = int(filament_count) if filament_count else (plate.extruder_count or 1)
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
        # Bambu Studio 读 part 时按这几个 key 把「这一块」认成一次正常的导入
        # （1.6.57，m26604 第 4 条：缺少 source_* 时整个工程读不进来）。
        f'      <metadata key="source_file" value="{_attr(object_name)}.3mf" />\n'
        '      <metadata key="source_object_id" value="0" />\n'
        '      <metadata key="source_volume_id" value="0" />\n'
        '      <metadata key="source_offset_x" value="0" />\n'
        '      <metadata key="source_offset_y" value="0" />\n'
        '      <metadata key="source_offset_z" value="0" />\n'
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
def _mix_pairs(entry: object) -> tuple[tuple[str, int], ...]:
    """The recipe of a mix as ``((filament_id, percent), …)``.

    Read exactly the way 「更换颜色」 reads it: a two-spool mix carries a
    ``recipe`` (``percent_a``/``percent_b``), a three-spool one only ``parts``,
    and an entry restored from a project file carries whichever of the two it was
    saved with.  A plain spool — or a mix whose recipe is not recorded — answers
    with an empty tuple and is exported as the flat colour it is painted with.
    """
    parts = tuple(getattr(entry, "parts", ()) or ())
    if parts:
        return tuple((str(parent), int(percent)) for parent, percent in parts)
    recipe = getattr(entry, "recipe", None)
    if recipe is None:
        recipes = tuple(getattr(entry, "recipes", ()) or ())
        recipe = recipes[0] if recipes else None
    if recipe is None:
        return ()
    return (
        (str(recipe.a_id), int(recipe.percent_a)),
        (str(recipe.b_id), int(recipe.percent_b)),
    )


def slots_from_palette(palette: Sequence[object], library: object | None = None) -> list[FilamentSlot]:
    """Build the project's filament list from a match palette.

    ``library`` answers two questions: "which spool is this?" for a mix — a mix
    is two spools of the same 耗材种类, so its type and vendor are the parents'
    — and "what does each parent look like?" for the recipe itself.  Everything
    is read through ``getattr`` so this module stays free of an import back into
    ``app.core``.

    A mix the library can no longer resolve is exported as its flat colour
    rather than as a mix: a 混色耗材 that names a spool the reader does not have
    is worse than a plate painted the colour the user picked.
    """
    getter = getattr(library, "get", None)
    lookup = getter if callable(getter) else (lambda _filament_id: None)

    slots: list[FilamentSlot] = []
    for entry in palette:
        filament = getattr(entry, "filament", None)
        material_type = getattr(filament, "material_type", "") or ""
        vendor = getattr(filament, "brand", "") or ""
        components: list[FilamentComponent] = []
        for parent_id, percent in _mix_pairs(entry):
            parent = lookup(parent_id)
            if parent is None:
                components = []
                break
            components.append(
                FilamentComponent(
                    color_hex=str(getattr(parent, "color_hex", "") or ""),
                    material_type=str(getattr(parent, "material_type", "") or "") or "PLA",
                    vendor=str(getattr(parent, "brand", "") or ""),
                    name=str(getattr(parent, "display_name", "") or ""),
                    percent=float(percent),
                )
            )
        if len(components) < 2:
            components = []
        if not material_type and library is not None:
            for filament_id in getattr(entry, "filament_ids", ()) or ():
                parent = lookup(filament_id)
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
                components=tuple(components),
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
                components=given.components,
            )
    return [slot or FilamentSlot(color_hex="#FFFFFF") for slot in slots]


# ---------------------------------------------------------------------------
# The mixed filaments' parents
# ---------------------------------------------------------------------------
def _spool_key(color_hex: str, material_type: str, vendor: str) -> tuple[str, str, str]:
    """What makes two rows of the project list the same spool.

    The material falls back to ``PLA`` because that is what an appended row is
    written as, and two rows that share a colour have to agree on the spool.
    """
    return (color_hex.upper(), material_type or "PLA", vendor)


@dataclass(frozen=True)
class ProjectFilaments:
    """The project's filament list, with the mixed slots' parents resolved.

    ``mixed`` maps a 0-based slot index to ``(parent slots, ratios)``, both in
    recipe order — exactly what Bambu Studio's ``filament_mixed_components``
    (1-based) and ``filament_mixed_sublayer_ratios`` hold.  ``added`` is how many
    extra rows the recipes needed, which is what the export dialog reports, and
    ``order`` maps each painted extruder (1-based) to the 1-based row it now
    occupies, because the mixed rows moved to the end.
    """

    slots: list[FilamentSlot]
    mixed: dict[int, tuple[tuple[int, ...], tuple[float, ...]]]
    added: int
    order: dict[int, int]

    @property
    def count(self) -> int:
        return len(self.slots)

    @property
    def mixed_count(self) -> int:
        return len(self.mixed)


def project_filaments(slots: Sequence[FilamentSlot]) -> ProjectFilaments:
    """Add the spools the recipes need and put the mixed rows last.

    Bambu Studio keeps a mixed colour and the spools it mixes as separate rows of
    one project list, and ``filament_mixed_components`` points at the physical
    rows by 1-based index.  A recipe whose parent is not painted anywhere (a mix
    can be chosen for a block nothing else prints) therefore needs a row of its
    own — 「拆色完成后将自动创建混色耗材所需的项目耗材。」 is Bambu Studio doing
    the same thing.  Parents already in the list are reused, never duplicated.

    **Every mixed row comes after every other row.**  Bambu Studio 2.8.2.61 reads
    a project whose mixed rows are interleaved as 「Invalid parameter value(s)
    included in the 3mf file」 and exports nothing at all (``return_code`` -18) —
    measured, not guessed: the same package with the mixed rows moved to the end
    exports fine, and the files Bambu Studio writes itself list the spools first
    and append the mixes.  So the painted rows keep their relative order, the
    spools only a recipe needs follow them, and the mixes come last; ``order``
    carries where each painted extruder went, and :func:`write_3mf` hands it to
    :func:`_remapped_extruders` so the ``paint_color`` codes follow along.
    """
    final: list[FilamentSlot] = []
    order: dict[int, int] = {}
    recipes: list[tuple[int, list[int], list[float]]] = []
    index_of: dict[tuple[str, str, str], int] = {}

    def spool_row(component: FilamentComponent) -> int:
        """Return the 1-based row of ``component``, adding a row if it has none."""
        material = component.material_type or "PLA"
        key = _spool_key(component.color_hex, material, component.vendor)
        position = index_of.get(key)
        if position is None:
            final.append(
                FilamentSlot(
                    color_hex=component.color_hex,
                    material_type=material,
                    vendor=component.vendor,
                    name=component.name,
                )
            )
            position = len(final) - 1
            index_of[key] = position
        return position + 1

    # The painted spools keep their relative order at the front of the list.
    for index, slot in enumerate(slots, start=1):
        if sum(max(0.0, component.percent) for component in slot.components) <= 0.0:
            order[index] = len(final) + 1
            index_of.setdefault(
                _spool_key(slot.color_hex, slot.material_type, slot.vendor), len(final)
            )
            final.append(slot)

    # A recipe resolves against every painted spool, not only the ones above it in
    # the list, so a parent painted further down is reused instead of duplicated.
    # Spools that only a recipe uses join the spool block, ahead of the mixes.
    for index, slot in enumerate(slots, start=1):
        total = sum(max(0.0, component.percent) for component in slot.components)
        if total <= 0.0:
            continue
        parents = [spool_row(component) for component in slot.components]
        ratios = [max(0.0, component.percent) / total for component in slot.components]
        recipes.append((index, parents, ratios))

    mixed: dict[int, tuple[tuple[int, ...], tuple[float, ...]]] = {}
    for index, parents, ratios in recipes:
        order[index] = len(final) + 1
        mixed[len(final)] = (tuple(parents), tuple(ratios))
        final.append(slots[index - 1])
    return ProjectFilaments(final, mixed, len(final) - len(slots), order)


def _remapped_extruders(plate: PlateModel, order: Mapping[int, int]) -> PlateModel:
    """Give every part the slot the project list ended up using for it."""
    moved = [
        replace(part, extruder=int(order.get(part.extruder, part.extruder)))
        for part in plate.parts
    ]
    if all(after.extruder == before.extruder for after, before in zip(moved, plate.parts)):
        return plate
    return replace(plate, parts=moved)


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


def _mixed_keys(
    palette: Sequence[str],
    mixed: Mapping[int, tuple[tuple[int, ...], tuple[float, ...]]],
) -> dict[str, object]:
    """The 混色耗材 half of ``project_settings.config``.

    Every mixed colour is one slot of the project filament list *plus* one row
    per spool it mixes (see :func:`project_filaments`), so the keys are arrays
    with one entry per slot: ``""``/``"0"`` for the plain ones, and for a mix the
    parents and their ratios.

    The shape is taken from a project Bambu Studio exported itself (「猪包」,
    2026-10-11) — parents 1-based into the same list and in recipe order, ratios
    four decimals summing to 1.  The ``filament_mixed_gradient*`` keys belong to
    the *gradient* mix (a ratio that changes with height); our plates print one
    colour per block, so they stay off.

    ``enable_mixed_color_sublayer`` is switched on whenever there is a mix: with
    it off Bambu Studio alternates the parents layer by layer, and a plate is
    only ``(0.8 + 0.6) / 0.2`` = 7 layers tall, so the two colours would show as
    stripes instead of one blended colour.  Turning it on splits each layer into
    sublayers, which is the 混色 the user picked.  It is a setting inside the
    exported project, so the user can turn it off in Bambu Studio like any other.
    """
    count = len(palette)
    components: list[str] = []
    ratios: list[str] = []
    for index in range(count):
        parents, shares = mixed.get(index, ((), ()))
        components.append(",".join(str(parent) for parent in parents))
        ratios.append(",".join(f"{share:.4f}" for share in shares))

    keys: dict[str, object] = {
        "filament_multi_colour": list(palette),
        "filament_is_mixed": ["1" if index in mixed else "0" for index in range(count)],
        "filament_mixed_components": components,
        "filament_mixed_sublayer_ratios": ratios,
        "filament_mixed_gradient": ["0"] * count,
        "filament_mixed_gradient_curve": [""] * count,
        "filament_mixed_gradient_range": [""] * count,
        "filament_mixed_gradient_per_part": ["0"] * count,
    }
    if mixed:
        keys["enable_mixed_color_sublayer"] = "1"
    return keys


def _as_slot(value: FilamentSlot | str) -> FilamentSlot:
    """A slot from either a :class:`FilamentSlot` or a bare colour string.

    The colour-only form is what the selftest's template check passes in; a
    plate always goes through :func:`_project_filaments` and brings real slots.
    """
    if isinstance(value, FilamentSlot):
        return value
    return FilamentSlot(color_hex=str(value))


def _project_settings(
    slots: Sequence[FilamentSlot | str],
    mixed: Mapping[int, tuple[tuple[int, ...], tuple[float, ...]]] | None = None,
) -> str:
    """``Metadata/project_settings.config`` carrying our plate's colours.

    Starts from the stock Orca/Bambu project template and rewrites only what
    describes the filaments: ``filament_colour`` becomes the palette, every array
    that holds one entry per filament is padded (repeating its last entry) or
    truncated to match, and the mixed colours — the ones the recipes made — are
    written the way Bambu Studio writes them itself (see
    :func:`_mixed_keys`).  Everything else — printer, process, the g-code blocks
    — is left exactly as the stock preset wrote it, which is what keeps Bambu
    Studio from treating the file as carrying custom presets.
    """
    settings = json.loads(_PROJECT_TEMPLATE_PATH.read_text(encoding="utf-8"))
    palette = [_as_slot(slot).color_hex.upper() for slot in slots]
    count = len(palette)
    settings["filament_colour"] = palette
    settings.update(_mixed_keys(palette, mixed or {}))
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

    # Centre the footprint on the bed.  The plate is built in millimetres with
    # its own origin at a corner, so the shift is half the leftover space.
    dx = (plate_size_mm - plate.width_mm) / 2.0
    dy = (plate_size_mm - plate.depth_mm) / 2.0
    translate = (dx, dy, 0.0)

    name = object_name or target.stem
    slots = _project_filaments(plate, filaments)
    project = project_filaments(slots)
    # The mixed rows sit at the end of the project list (Bambu Studio refuses
    # anything else), so the plate is repainted with the slots the list ended up
    # using before either document is written.
    painted = _remapped_extruders(plate, project.order)
    if painted.extruder_count > MAX_PAINTED_FILAMENTS:
        raise ValueError(
            f"这个模型的混色配方把耗材列表排到第 {painted.extruder_count} 行，"
            f"超过 Bambu Studio 能识别的 {MAX_PAINTED_FILAMENTS} 种。"
        )

    # 先写同目录的临时文件再改名（1.6.50）：导出常常是覆盖上一次那份，直接以写入
    # 模式打开目标文件，中途出错就只剩一个被截断的 3MF。
    with atomic_path(target) as temporary:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("[Content_Types].xml", CONTENT_TYPES)
            archive.writestr("_rels/.rels", RELS)
            # 1.6.57（m26604 第 4 条）：Bambu Studio 是从这份 rels 里找到工程设置与
            # 模型设置的，缺了它整包会被判为「不是 Bambu 工程」，双击打不开。
            archive.writestr("3D/_rels/3dmodel.model.rels", MODEL_RELS)
            archive.writestr(
                "3D/3dmodel.model", _model_document(painted, translate, project.slots)
            )
            archive.writestr(
                "Metadata/model_settings.config",
                _model_settings(painted, name, project.count),
            )
            # The project config is what carries the colours — and the recipes behind
            # the mixed ones — into Bambu Studio's own filament list.  See the
            # comment above for what it has to contain.
            archive.writestr(
                "Metadata/project_settings.config",
                _project_settings(project.slots, project.mixed),
            )
            archive.writestr(
                "Metadata/filament_sequence.json",
                json.dumps(
                    {
                        "plate_1": {
                            "nozzle_sequence": [],
                            "optimal_assignment": [],
                            "sequence": [],
                        }
                    }
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

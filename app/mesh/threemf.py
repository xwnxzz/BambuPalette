"""Write the flat plate as a Bambu Studio compatible 3MF.

The package is plain **core 3MF**: one ``3D/3dmodel.model`` holding every part's
mesh inline, an assembly object that references them through ``<components>``,
and the usual OPC plumbing.  On top of that sits the one file Bambu Studio
actually needs for multi-material printing — ``Metadata/model_settings.config``,
whose per-``<part>`` ``extruder`` entries carry the filament assignment.

An earlier version copied the layout of a project file Bambu Studio itself had
exported (``BambuStudio-02.06.00.51``): one ``3D/Objects/object_N.model`` per
part, pulled in through ``<component p:path=…>`` under the 3MF production
extension, plus ``p:UUID`` bookkeeping.  That looked faithful, but round-tripping
it through the installed Bambu Studio (``02.08.02.61``) proved it does not work —
the reader kept the metadata and threw every component away, re-exporting a
project whose assembly object had an empty ``<components/>`` and
``face_count="0"``.  Core 3MF with inline meshes round-trips correctly, which is
what ``tools/verify_bambu_roundtrip.py`` checks.

There is deliberately no ``<basematerials>``: Bambu Studio takes colours from
``model_settings.config`` plus the AMS slots, not from the model XML.  Because a
3MF package is just a zip, everything is written with the standard library.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

from .plate import PlateModel, PlatePart

__all__ = ["write_3mf", "THREEMF_APPLICATION", "DEFAULT_PLATE_MM"]

#: Recorded in ``<metadata name="Application">``.  Kept honest instead of
#: impersonating Bambu Studio; the importer keys off the standard parts below.
THREEMF_APPLICATION = "BambuPalette"

#: Bambu's printers all ship a 256 mm build plate, so that is where the object
#: is centred.  A larger plate simply leaves more room around it.
DEFAULT_PLATE_MM = 256.0

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


def _mesh_object(part: PlatePart, object_id: int, translate: tuple[float, float, float]) -> str:
    dx, dy, dz = translate
    vertex_rows = "".join(
        f'          <vertex x="{_fmt(x + dx)}" y="{_fmt(y + dy)}" z="{_fmt(z + dz)}" />\n'
        for x, y, z in part.vertices
    )
    triangle_rows = "".join(
        f'          <triangle v1="{int(a)}" v2="{int(b)}" v3="{int(c)}" />\n'
        for a, b, c in part.triangles
    )
    return (
        f'    <object id="{object_id}" type="model">\n'
        "      <mesh>\n"
        "        <vertices>\n"
        f"{vertex_rows}"
        "        </vertices>\n"
        "        <triangles>\n"
        f"{triangle_rows}"
        "        </triangles>\n"
        "      </mesh>\n"
        "    </object>\n"
    )


def _model_document(plate: PlateModel, translate: tuple[float, float, float], assembly_id: int) -> str:
    """One core-3MF document holding every part inline plus the assembly object.

    An earlier version copied Bambu Studio's *own* layout — one geometry file per
    part, pulled in through ``<component p:path=…>`` under the 3MF production
    extension.  Round-tripping that through the installed Bambu Studio
    (``bambu-studio.exe --export-3mf``, version 02.08.02.61) showed the reader
    accepting the metadata but dropping **all** components: the re-exported
    project had an assembly object with an empty ``<components/>`` and
    ``face_count="0"``, i.e. the model arrived with no geometry at all.

    Plain core 3MF — meshes inline, components referencing local object ids by
    ``objectid`` alone — is understood everywhere, so that is what we emit.  The
    per-part id still equals the ``<part id=…>`` in ``model_settings.config``,
    which is what carries the extruder assignment.
    """
    meshes = "".join(
        _mesh_object(part, index, translate)
        for index, part in enumerate(plate.parts, start=1)
    )
    components = "".join(
        f'      <component objectid="{index}" transform="1 0 0 0 1 0 0 0 1 0 0 0" />\n'
        for index in range(1, len(plate.parts) + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<model unit="millimeter" xml:lang="en-US" '
        'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">\n'
        f'  <metadata name="Application">{_attr(THREEMF_APPLICATION)}</metadata>\n'
        '  <metadata name="BambuStudio:3mfVersion">1</metadata>\n'
        "  <resources>\n"
        f"{meshes}"
        f'    <object id="{assembly_id}" type="model">\n'
        "      <components>\n"
        f"{components}"
        "      </components>\n"
        "    </object>\n"
        "  </resources>\n"
        "  <build>\n"
        f'    <item objectid="{assembly_id}" printable="1" />\n'
        "  </build>\n"
        "</model>\n"
    )


def _model_settings(plate: PlateModel, assembly_id: int, object_name: str) -> str:
    parts = []
    for index, part in enumerate(plate.parts, start=1):
        parts.append(
            f'    <part id="{index}" subtype="normal_part">\n'
            f'      <metadata key="name" value="{_attr(part.name)}" />\n'
            f'      <metadata key="extruder" value="{part.extruder}" />\n'
            f'      <metadata key="source_object_id" value="0" />\n'
            f'      <metadata key="matrix" value="1 0 0 0 1 0 0 0 1 0 0 0" />\n'
            f'      <mesh_stat face_count="{part.triangle_count}" '
            f'vertex_count="{int(len(part.vertices))}" />\n'
            "    </part>\n"
        )
    instances = (
        "      <model_instance>\n"
        f'        <metadata key="object_id" value="{assembly_id}" />\n'
        '        <metadata key="instance_id" value="0" />\n'
        '        <metadata key="identify_id" value="76" />\n'
        "      </model_instance>\n"
    )
    default_extruder = plate.parts[0].extruder if plate.parts else 1
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<config>\n"
        f'  <object id="{assembly_id}">\n'
        f'    <metadata key="name" value="{_attr(object_name)}" />\n'
        f'    <metadata key="extruder" value="{default_extruder}" />\n'
        f'    <metadata key="face_count" value="{plate.triangle_count}" />\n'
        f"{''.join(parts)}"
        "  </object>\n"
        "  <plate>\n"
        '    <metadata key="plater_id" value="1" />\n'
        '    <metadata key="plater_name" value="Plate 1" />\n'
        '    <metadata key="locked" value="false" />\n'
        '    <metadata key="filament_map_mode" value="Auto For Flush" />\n'
        f"{instances}"
        "  </plate>\n"
        "</config>\n"
    )


def write_3mf(
    plate: PlateModel,
    path: str | Path,
    *,
    object_name: str = "",
    plate_size_mm: float = DEFAULT_PLATE_MM,
) -> Path:
    """Write ``plate`` as a Bambu-Studio-readable ``.3mf`` and return the path.

    Every part becomes its own inline object plus its own ``<part>`` entry, so
    each colour region arrives in Bambu Studio already assigned to its extruder.
    """
    if not plate.parts:
        raise ValueError("nothing to export: the plate has no parts")

    target = Path(path)
    if target.suffix.lower() != ".3mf":
        target = target.with_suffix(".3mf")
    target.parent.mkdir(parents=True, exist_ok=True)

    assembly_id = len(plate.parts) + 1
    name = object_name or plate.source or "混色底板"
    # Centre the footprint on the bed: the mesh is built with its corner at the
    # origin, while Bambu Studio positions an imported object by its transform.
    translate = (
        max(0.0, (plate_size_mm - plate.width_mm) / 2.0),
        max(0.0, (plate_size_mm - plate.depth_mm) / 2.0),
        0.0,
    )

    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", CONTENT_TYPES)
        archive.writestr("_rels/.rels", RELS)
        archive.writestr("3D/3dmodel.model", _model_document(plate, translate, assembly_id))
        archive.writestr(
            "Metadata/model_settings.config", _model_settings(plate, assembly_id, name)
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
            f'<header_item key="X-BBL-Client-Type" value="{_attr(THREEMF_APPLICATION)}" />'
            # A Bambu Studio compatibility marker (the client version its own
            # parser expects), NOT BambuPalette's version — that lives in
            # app/__init__.py and is reported by --selftest.
            '<header_item key="X-BBL-Client-Version" value="1.0.0" />'
            "</header></config>\n",
        )
    return target

"""The picture page: import an image, match it, export a flat plate.

Layout follows the user's description exactly — the picture on the left, the
list of colours the print needs on the right, and the two are linked both ways:

* clicking a colour in the list highlights that colour's pixels in the picture;
* clicking a pixel in the picture jumps to its colour in the list, and that
  colour is pinned to the top **without disturbing the order of the others**.

The page never talks to the network and never writes anywhere the user did not
choose; all Qt file dialogs default to the app's own exports directory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..core import paths
from ..core.image_matching import (
    MatchResult,
    MatchSettings,
    PaletteEntry,
    build_palette,
    match_image,
)
from ..core.library import FilamentLibrary
from ..core.mixes import MixCatalog
from ..mesh.objfile import write_obj
from ..mesh.plate import PlateModel, PlateSettings, build_plate
from ..mesh.threemf import write_3mf
from . import theme
from .swatch import swatch_pixmap

__all__ = ["PicturePage", "PictureView", "ColourList"]

IMAGE_FILTER = "图片 (*.png *.jpg *.jpeg *.bmp *.webp *.gif *.tif *.tiff);;所有文件 (*)"


def _to_qimage(image: Image.Image) -> QImage:
    """PIL image → QImage, copying so the buffer outlives the PIL object."""
    rgba = image.convert("RGBA")
    data = rgba.tobytes("raw", "RGBA")
    qimage = QImage(data, rgba.width, rgba.height, QImage.Format.Format_RGBA8888)
    return qimage.copy()


class PictureView(QWidget):
    """Shows the imported picture and reports which colour region was clicked."""

    regionClicked = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setMouseTracking(False)
        self._result: MatchResult | None = None
        self._source: QImage | None = None
        self._matched: QImage | None = None
        self._highlight: QPixmap | None = None
        self._selected = -1
        self._show_matched = True
        self._scale = 1.0
        self._origin = QPointF(0.0, 0.0)

    # -- contents ---------------------------------------------------------
    def setResult(self, result: MatchResult | None, source: QImage | None) -> None:
        self._result = result
        self._source = source
        self._matched = self._build_matched(result) if result is not None else None
        self._selected = -1
        self._highlight = None
        self.update()

    def setSelected(self, index: int) -> None:
        if index == self._selected:
            return
        self._selected = int(index)
        self._highlight = None
        self.update()

    def setShowMatched(self, matched: bool) -> None:
        self._show_matched = bool(matched)
        self.update()

    def result(self) -> MatchResult | None:
        return self._result

    def _build_matched(self, result: MatchResult) -> QImage:
        height, width = result.indices.shape
        canvas = np.zeros((height, width, 4), dtype=np.uint8)
        for index, entry in enumerate(result.palette):
            mask = result.indices == index
            if not mask.any():
                continue
            canvas[mask] = (*entry.rgb, 255)
        return _to_qimage(Image.fromarray(canvas, "RGBA"))

    # -- geometry ---------------------------------------------------------
    def _target_rect(self) -> QRectF:
        if self._result is None:
            return QRectF()
        width, height = self._result.width, self._result.height
        available_w = max(1, self.width() - 8)
        available_h = max(1, self.height() - 8)
        self._scale = min(available_w / width, available_h / height)
        draw_w = width * self._scale
        draw_h = height * self._scale
        self._origin = QPointF((self.width() - draw_w) / 2.0, (self.height() - draw_h) / 2.0)
        return QRectF(self._origin.x(), self._origin.y(), draw_w, draw_h)

    def _base_image(self) -> QImage | None:
        if self._show_matched:
            return self._matched
        return self._source

    def _dim_pixmap(self) -> QPixmap | None:
        """The picture with everything but the selected colour faded out."""
        result = self._result
        if result is None or self._selected < 0 or self._highlight is not None:
            return self._highlight
        height, width = result.indices.shape
        canvas = np.zeros((height, width, 4), dtype=np.uint8)
        canvas[..., 3] = 255
        keep = result.indices == self._selected
        # A wash rather than a solid fill: the rest of the picture stays legible
        # as context, so a small highlighted region is not floating in a void.
        canvas[~keep] = (246, 247, 249, 168)
        canvas[keep] = (0, 0, 0, 0)
        overlay = _to_qimage(Image.fromarray(canvas, "RGBA"))
        pixmap = QPixmap.fromImage(overlay)
        # A crisp outline makes the region readable even when it is small.
        outline = self._outline(result, keep)
        if outline is not None:
            painter = QPainter(pixmap)
            painter.drawImage(0, 0, outline)
            painter.end()
        self._highlight = pixmap
        return pixmap

    def _outline(self, result: MatchResult, keep: np.ndarray) -> QImage | None:
        edge = np.zeros_like(keep)
        edge[1:, :] |= keep[1:, :] & ~keep[:-1, :]
        edge[:-1, :] |= keep[:-1, :] & ~keep[1:, :]
        edge[:, 1:] |= keep[:, 1:] & ~keep[:, :-1]
        edge[:, :-1] |= keep[:, :-1] & ~keep[:, 1:]
        if not edge.any():
            return None
        canvas = np.zeros((*keep.shape, 4), dtype=np.uint8)
        canvas[edge] = (17, 24, 39, 255)
        return _to_qimage(Image.fromarray(canvas, "RGBA"))

    # -- painting and input ----------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(theme.PANEL_ALT))
        result = self._result
        base = self._base_image()
        if result is None or base is None:
            painter.setPen(QColor(theme.TEXT_FAINT))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "还没有导入图片")
            painter.end()
            return
        rect = self._target_rect()
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        painter.drawImage(rect, base)
        overlay = self._dim_pixmap()
        if overlay is not None:
            painter.drawPixmap(rect, overlay, QRectF(overlay.rect()))
        painter.setPen(QColor(theme.BORDER_STRONG))
        painter.drawRect(rect)
        painter.end()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        result = self._result
        if result is None:
            return
        rect = self._target_rect()
        if not rect.contains(event.position()):
            return
        x = int((event.position().x() - rect.x()) / self._scale)
        y = int((event.position().y() - rect.y()) / self._scale)
        if not (0 <= x < result.width and 0 <= y < result.height):
            return
        index = int(result.indices[y, x])
        if index < 0:
            return
        self.setSelected(index)
        self.regionClicked.emit(index)


class ColourList(QWidget):
    """The colours this print needs, with the clicked one pinned to the top."""

    colourSelected = Signal(int)
    colourActivated = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._result: MatchResult | None = None
        self._order: list[int] = []
        self._pinned = -1
        self._suppress = False

        self._list = QListWidget()
        self._list.setIconSize(swatch_pixmap("#FFFFFF").size())
        self._list.setUniformItemSizes(True)
        self._list.setAlternatingRowColors(False)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self._list.currentRowChanged.connect(self._on_row_changed)
        self._list.itemDoubleClicked.connect(self._on_row_double_clicked)

        self._summary = QLabel("")
        self._summary.setProperty("role", "hint")
        self._summary.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._list, 1)
        layout.addWidget(self._summary)

    def setResult(self, result: MatchResult | None) -> None:
        self._result = result
        self._pinned = -1
        self._order = list(range(len(result.palette))) if result is not None else []
        self._refresh()
        self._update_summary()

    def current(self) -> int:
        return self._pinned

    def entries(self) -> list[PaletteEntry]:
        return list(self._result.palette) if self._result is not None else []

    def pin(self, index: int) -> None:
        """Put ``index`` first; every other entry keeps its relative order."""
        result = self._result
        if result is None or not (0 <= index < len(result.palette)):
            return
        self._pinned = int(index)
        self._order = [self._pinned] + [i for i in range(len(result.palette)) if i != self._pinned]
        self._refresh()

    def _refresh(self) -> None:
        result = self._result
        # Everything that touches the view happens while suppressed: selecting
        # row 0 emits ``currentRowChanged``, which would otherwise come straight
        # back in here and recurse until the stack runs out.
        self._suppress = True
        self._list.clear()
        if result is not None:
            total = max(1, result.printed_pixels)
            for index in self._order:
                entry = result.palette[index]
                count = int(result.counts[index])
                item = QListWidgetItem(
                    swatch_pixmap(entry.color_hex),
                    f"{entry.color_hex}   {count} 像素 · {count / total * 100:.1f}%",
                )
                item.setData(Qt.ItemDataRole.UserRole, int(index))
                item.setToolTip(f"{entry.label}\n{entry.color_hex}")
                self._list.addItem(item)
            if self._pinned >= 0 and self._list.count():
                self._list.setCurrentRow(0)
        self._suppress = False

    def _update_summary(self) -> None:
        result = self._result
        if result is None:
            self._summary.setText("导入图片后，这里列出打印需要的颜色。")
            return
        self._summary.setText(
            f"{result.width}×{result.height} 像素 · 实心 {result.printed_pixels} 像素"
            f" · 共 {len(result.palette)} 种颜色"
        )

    def _on_row_changed(self, row: int) -> None:
        if self._suppress or row < 0:
            return
        item = self._list.item(row)
        if item is None:
            return
        index = int(item.data(Qt.ItemDataRole.UserRole))
        self.colourSelected.emit(index)

    def _on_row_double_clicked(self, item: QListWidgetItem) -> None:
        self.colourActivated.emit(int(item.data(Qt.ItemDataRole.UserRole)))


class PicturePage(QWidget):
    """Import a picture, match it against the filament library, export a plate."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._library: FilamentLibrary | None = None
        self._catalog: MixCatalog | None = None
        self._image: Image.Image | None = None
        self._source_qimage: QImage | None = None
        self._result: MatchResult | None = None
        self._plate: PlateModel | None = None
        self._build_ui()
        self._set_controls_enabled(False)

    # -- construction -----------------------------------------------------
    def _build_ui(self) -> None:
        self._import_button = QPushButton("导入图片…")
        self._import_button.setProperty("accent", "true")
        self._import_button.clicked.connect(self._on_import)

        self._rematch_button = QPushButton("重新匹配")
        self._rematch_button.clicked.connect(self._on_rematch)

        self._include_mixes = QCheckBox("混色也算候选")
        self._include_mixes.setChecked(True)
        self._include_mixes.setToolTip("勾选后，候选颜色包含全部两两混色的 81 个配比")

        self._colour_count = QSpinBox()
        self._colour_count.setRange(2, 48)
        self._colour_count.setValue(12)
        self._colour_count.setToolTip("图片最多用多少种颜色；越少越省换料")

        self._max_dimension = QSpinBox()
        self._max_dimension.setRange(32, 4096)
        self._max_dimension.setSingleStep(64)
        self._max_dimension.setValue(512)
        self._max_dimension.setSuffix(" px")

        self._dither = QCheckBox("抖动")
        self._dither.setToolTip("用误差扩散让渐变更平滑，但会牺牲色块的纯净度")

        self._show_matched = QCheckBox("显示匹配结果")
        self._show_matched.setChecked(True)
        self._show_matched.toggled.connect(self._on_show_matched)

        top = QHBoxLayout()
        top.setSpacing(8)
        top.addWidget(self._import_button)
        top.addWidget(QLabel("颜色数量:"))
        top.addWidget(self._colour_count)
        top.addWidget(QLabel("最大边长:"))
        top.addWidget(self._max_dimension)
        top.addWidget(self._include_mixes)
        top.addWidget(self._dither)
        top.addWidget(self._rematch_button)
        top.addWidget(self._show_matched)
        top.addStretch(1)

        self._width_spin = QDoubleSpinBox()
        self._width_spin.setRange(10.0, 400.0)
        self._width_spin.setValue(150.0)
        self._width_spin.setSuffix(" mm")
        self._width_spin.setDecimals(1)

        self._base_spin = QDoubleSpinBox()
        self._base_spin.setRange(0.2, 20.0)
        self._base_spin.setValue(0.8)
        self._base_spin.setSingleStep(0.2)
        self._base_spin.setSuffix(" mm")

        self._colour_spin = QDoubleSpinBox()
        self._colour_spin.setRange(0.2, 20.0)
        self._colour_spin.setValue(0.6)
        self._colour_spin.setSingleStep(0.2)
        self._colour_spin.setSuffix(" mm")

        self._base_combo = QComboBox()
        self._base_combo.setToolTip("底板用哪种颜色的耗材打印；默认用占比最大的颜色")

        self._export_3mf = QPushButton("生成 3MF（Bambu Studio）")
        self._export_3mf.setProperty("accent", "true")
        self._export_3mf.clicked.connect(lambda: self._on_export("3mf"))

        self._export_obj = QPushButton("生成 OBJ + MTL")
        self._export_obj.clicked.connect(lambda: self._on_export("obj"))

        bottom = QHBoxLayout()
        bottom.setSpacing(8)
        bottom.addWidget(QLabel("打印宽度:"))
        bottom.addWidget(self._width_spin)
        bottom.addWidget(QLabel("底板厚度:"))
        bottom.addWidget(self._base_spin)
        bottom.addWidget(QLabel("颜色层厚度:"))
        bottom.addWidget(self._colour_spin)
        bottom.addWidget(QLabel("底板颜色:"))
        bottom.addWidget(self._base_combo)
        bottom.addStretch(1)
        bottom.addWidget(self._export_obj)
        bottom.addWidget(self._export_3mf)

        self._view = PictureView()
        self._view.regionClicked.connect(self._on_region_clicked)

        self._list = ColourList()
        self._list.colourSelected.connect(self._on_colour_selected)
        self._list.colourActivated.connect(self._on_colour_activated)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._wrap(self._view, "图片"))
        splitter.addWidget(self._wrap(self._list, "这个模型需要的颜色"))
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([760, 320])

        self._status = QLabel("")
        self._status.setProperty("role", "hint")
        self._status.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addLayout(top)
        layout.addWidget(splitter, 1)
        layout.addWidget(self._status)
        layout.addLayout(bottom)

    def _wrap(self, widget: QWidget, title: str) -> QWidget:
        frame = QFrame()
        frame.setObjectName("panel")
        header = QLabel(title)
        header.setProperty("role", "sectionTitle")
        box = QVBoxLayout(frame)
        box.setContentsMargins(10, 8, 10, 10)
        box.setSpacing(8)
        box.addWidget(header)
        box.addWidget(widget, 1)
        return frame

    def _set_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self._rematch_button,
            self._width_spin,
            self._base_spin,
            self._colour_spin,
            self._base_combo,
            self._export_3mf,
            self._export_obj,
            self._show_matched,
        ):
            widget.setEnabled(enabled)

    # -- library ----------------------------------------------------------
    def setLibrary(self, library: FilamentLibrary, catalog: MixCatalog | None) -> None:
        """Called whenever the filament library or the mix catalogue changes."""
        self._library = library
        self._catalog = catalog
        if self._image is not None:
            self._on_rematch()

    @property
    def result(self) -> MatchResult | None:
        return self._result

    @property
    def plate(self) -> PlateModel | None:
        return self._plate

    # -- matching ---------------------------------------------------------
    def loadImage(self, source) -> None:
        """Load ``source`` (path or PIL image) and match it straight away."""
        image = Image.open(source) if isinstance(source, (str, Path)) else source
        self._image = image.convert("RGBA")
        self._source_qimage = _to_qimage(self._image)
        self._on_rematch()

    def _on_import(self) -> None:
        start = str(paths.projects_dir())
        name, _ = QFileDialog.getOpenFileName(self, "选择图片", start, IMAGE_FILTER)
        if not name:
            return
        try:
            self.loadImage(Path(name))
        except Exception as error:  # pragma: no cover - the dialog is interactive
            QMessageBox.warning(self, "无法读取图片", str(error))

    def _settings(self) -> MatchSettings:
        return MatchSettings(
            max_colours=int(self._colour_count.value()),
            max_dimension=int(self._max_dimension.value()),
            dither=bool(self._dither.isChecked()),
        ).clamped()

    def _on_rematch(self) -> None:
        if self._image is None:
            return
        if self._library is None or len(self._library) == 0:
            self._status.setText("请先在「混色配方」页添加耗材，再匹配图片。")
            return
        palette = build_palette(
            self._library, self._catalog, include_mixes=bool(self._include_mixes.isChecked())
        )
        if not palette:
            self._status.setText("调色板是空的，请先添加耗材。")
            return
        try:
            result = match_image(self._image, palette, self._settings())
        except Exception as error:  # pragma: no cover - defensive
            QMessageBox.warning(self, "匹配失败", str(error))
            return
        self._result = result
        self._plate = None
        self._view.setResult(result, self._source_qimage)
        self._list.setResult(result)
        self._base_combo.blockSignals(True)
        self._base_combo.clear()
        for index, entry in enumerate(result.palette):
            count = int(result.counts[index])
            self._base_combo.addItem(f"{entry.color_hex}  {entry.short_label}", index)
        self._base_combo.blockSignals(False)
        self._set_controls_enabled(True)
        self._status.setText(
            "匹配完成：{0}×{1} 像素，{2} 种颜色。点击右侧颜色看图片高光，"
            "点击图片看它属于哪种颜色。".format(result.width, result.height, len(result.palette))
        )

    def _on_show_matched(self, checked: bool) -> None:
        self._view.setShowMatched(bool(checked))

    # -- two-way selection ------------------------------------------------
    def _on_colour_selected(self, index: int) -> None:
        self._view.setSelected(index)
        self._list.pin(index)

    def _on_region_clicked(self, index: int) -> None:
        self._list.pin(index)

    def _on_colour_activated(self, index: int) -> None:
        self._list.pin(index)

    def _on_actions_changed(self) -> None:
        if self._result is not None:
            self._plate = None

    # -- export -----------------------------------------------------------
    def _plate_settings(self) -> PlateSettings:
        base_index = self._base_combo.currentData()
        return PlateSettings(
            target_width_mm=float(self._width_spin.value()),
            base_thickness_mm=float(self._base_spin.value()),
            colour_thickness_mm=float(self._colour_spin.value()),
            base_index=int(base_index) if base_index is not None else -1,
        ).clamped()

    def buildPlate(self) -> PlateModel:
        if self._result is None:
            raise ValueError("请先导入并匹配一张图片")
        self._plate = build_plate(
            self._result, self._plate_settings(), base_label="底板"
        )
        return self._plate

    def _on_export(self, kind: str) -> None:
        try:
            plate = self.buildPlate()
        except Exception as error:
            QMessageBox.warning(self, "无法生成模型", str(error))
            return
        suffix = ".3mf" if kind == "3mf" else ".obj"
        label = "3MF 模型" if kind == "3mf" else "OBJ 模型"
        start = str(paths.exports_dir() / f"混色底板{suffix}")
        name, _ = QFileDialog.getSaveFileName(self, f"保存{label}", start, f"{label} (*{suffix})")
        if not name:
            return
        target = Path(name)
        try:
            if kind == "3mf":
                written = write_3mf(plate, target)
            else:
                written = write_obj(plate, target)
        except Exception as error:
            QMessageBox.warning(self, "保存失败", str(error))
            return
        self._status.setText(self._describe(plate, written))
        self._announce(plate, written)

    def _describe(self, plate: PlateModel, written: Path) -> str:
        stats = plate.stats()
        lines = [
            f"已生成 {written.name}",
            f"{stats['width_mm']} × {stats['depth_mm']} mm，总厚度 {stats['thickness_mm']} mm，"
            f"像素 {stats['pixel_mm']} mm",
            f"{stats['parts']} 个部件 / {stats['extruders']} 种耗材 / {stats['triangles']} 个三角面",
        ]
        lines.extend(plate.notes)
        return "  ·  ".join(lines)

    def _announce(self, plate: PlateModel, written: Path) -> None:
        stats = plate.stats()
        box = QMessageBox(self)
        box.setWindowTitle("已生成模型")
        box.setIcon(QMessageBox.Icon.Information)
        rows = [
            f"文件：{written}",
            f"尺寸：{stats['width_mm']} × {stats['depth_mm']} mm，"
            f"底板 {plate.base_thickness_mm:g} mm + 颜色层 {plate.colour_thickness_mm:g} mm",
            f"部件：{stats['parts']} 个，共 {stats['extruders']} 种耗材",
            "",
            "在 Bambu Studio 里：",
            "1. 新建项目，把下面列出的耗材按顺序放进 AMS 的 1、2、3… 号槽；",
            "2. 打开这个文件，各颜色区域已经分别指定了挤出机；",
            "3. 切片前确认「耗材映射」与你的 AMS 槽位一致。",
            "",
            "需要的耗材：",
        ]
        for part in plate.parts:
            rows.append(f"  挤出机 {part.extruder}  {part.color_hex}  {part.name}")
        if plate.notes:
            rows.append("")
            rows.extend(f"提示：{note}" for note in plate.notes)
        box.setText("\n".join(rows))
        box.exec()

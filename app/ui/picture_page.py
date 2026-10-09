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

from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
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
from ..mesh.threemf import MAX_PAINTED_FILAMENTS, slots_from_palette, write_3mf
from . import theme
from .colour_picker import (
    ColourDetail,
    ColourPickerDialog,
    compare_pixmap,
    entry_recipe_text,
    hex_of,
)
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
    """The colours this print needs, with the clicked one pinned to the top.

    Each row compares the picture's own colour for that region with the spool or
    mix we matched it to, and names the recipe underneath, so the user can see at
    a glance which regions are a good match and which need a manual replacement.
    """

    colourSelected = Signal(int)
    colourActivated = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._result: MatchResult | None = None
        self._order: list[int] = []
        self._pinned = -1
        self._suppress = False
        self._library: FilamentLibrary | None = None

        self._list = QListWidget()
        self._list.setIconSize(compare_pixmap("#FFFFFF", "#000000").size())
        self._list.setUniformItemSizes(False)
        self._list.setAlternatingRowColors(False)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self._list.setWordWrap(False)
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
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the rows, keeping the pinned selection and the order."""
        self._refresh()
        self._update_summary()

    def setLibrary(self, library: FilamentLibrary | None) -> None:
        """The recipe text names the parent spools, so the list needs the library."""
        self._library = library
        self.refresh()

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
                image_hex = hex_of(result.region_colour(index))
                share = count / total * 100.0
                row = QListWidgetItem(
                    compare_pixmap(image_hex, entry.color_hex, 44, 20),
                    f"{image_hex} → {entry.color_hex}    {share:.1f}%\n"
                    f"{entry_recipe_text(entry, self._library, compact=True)}",
                )
                row.setData(Qt.ItemDataRole.UserRole, int(index))
                row.setToolTip(
                    f"图片里的颜色：{image_hex}（{count} 像素 · {share:.1f}%）\n"
                    f"匹配到的颜色：{entry.color_hex}\n"
                    f"{entry_recipe_text(entry, self._library)}"
                )
                self._list.addItem(row)
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
        self._image_name: str = "混色底板"
        self._source_qimage: QImage | None = None
        self._result: MatchResult | None = None
        self._plate: PlateModel | None = None
        # What the matcher decided, and what the USER decided on top of it. The
        # two are kept apart so 「恢复自动匹配」 can undo a replacement without
        # re-running the match, and so the export follows the user's choice.
        self._auto_palette: list[PaletteEntry] = []
        self._overrides: dict[int, PaletteEntry] = {}
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
        self._colour_count.setRange(2, MAX_PAINTED_FILAMENTS)
        self._colour_count.setValue(12)
        self._colour_count.setToolTip(
            f"图片最多用多少种颜色；越少越省换料。Bambu Studio 最多认 {MAX_PAINTED_FILAMENTS} 种"
        )

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

        self._detail = ColourDetail()
        self._detail.replaceRequested.connect(self._on_replace)
        self._detail.restoreRequested.connect(self._on_restore)

        # The list tells the user which colours the print needs; the detail
        # panel explains the one they clicked. Stacked, because both matter and
        # the right column is narrow.
        #
        # The detail panel is taller than its share of a short window (two
        # cards, a recipe, a hint and two buttons), and a 「更换颜色…」 button
        # that is off the bottom of the screen is the same as no button, so it
        # gets a scroll area of its own.
        detail_scroll = QScrollArea()
        detail_scroll.setWidgetResizable(True)
        detail_scroll.setFrameShape(QFrame.Shape.NoFrame)
        detail_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        detail_scroll.setWidget(self._detail)

        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(self._wrap(self._list, "这个模型需要的颜色"))
        right.addWidget(self._wrap(detail_scroll, "选中的颜色"))
        right.setStretchFactor(0, 3)
        right.setStretchFactor(1, 2)
        right.setCollapsible(1, False)
        right.setSizes([420, 340])

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._wrap(self._view, "图片"))
        splitter.addWidget(right)
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

    @property
    def library(self) -> FilamentLibrary | None:
        return self._library

    @property
    def image_name(self) -> str:
        """The imported picture's file name, used for the exported model."""
        return self._image_name

    # -- matching ---------------------------------------------------------
    def loadImage(self, source) -> None:
        """Load ``source`` (path or PIL image) and match it straight away."""
        if isinstance(source, (str, Path)):
            self._image_name = Path(source).stem or "混色底板"
            image = Image.open(source)
        else:
            image = source
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
        self._auto_palette = list(result.palette)
        self._overrides = {}
        self._list.setLibrary(self._library)
        self._view.setResult(result, self._source_qimage)
        self._list.setResult(result)
        self._refresh_base_combo()
        self._detail.clear()
        self._set_controls_enabled(True)
        self._status.setText(
            "匹配完成：{0}×{1} 像素，{2} 种颜色。点击右侧颜色看图片高光，"
            "点击图片看它属于哪种颜色；点「更换颜色…」可以手动换掉不合适的颜色。".format(
                result.width, result.height, len(result.palette)
            )
        )

    def _refresh_base_combo(self) -> None:
        result = self._result
        if result is None:
            return
        previous = self._base_combo.currentData()
        self._base_combo.blockSignals(True)
        self._base_combo.clear()
        for index, entry in enumerate(result.palette):
            self._base_combo.addItem(f"{entry.color_hex}  {entry.short_label}", index)
        if isinstance(previous, int) and 0 <= previous < self._base_combo.count():
            self._base_combo.setCurrentIndex(previous)
        self._base_combo.blockSignals(False)

    def _on_show_matched(self, checked: bool) -> None:
        self._view.setShowMatched(bool(checked))

    # -- two-way selection ------------------------------------------------
    def _select(self, index: int) -> None:
        """Show one colour everywhere it appears: picture, list and detail."""
        self._view.setSelected(index)
        self._list.pin(index)
        if self._result is None:
            return
        if not 0 <= index < len(self._result.palette):
            return
        original = self._auto_palette[index] if index < len(self._auto_palette) else None
        self._detail.setSelection(
            self._result,
            index,
            library=self._library,
            changed=index in self._overrides,
            original=original,
        )

    def _on_colour_selected(self, index: int) -> None:
        self._select(index)

    def _on_region_clicked(self, index: int) -> None:
        self._select(index)

    def _on_colour_activated(self, index: int) -> None:
        self._select(index)

    # -- manual replacement ------------------------------------------------
    def _current_index(self) -> int:
        index = self._list.current()
        if index < 0:
            return -1
        if self._result is None or not 0 <= index < len(self._result.palette):
            return -1
        return index

    def _apply_overrides(self) -> None:
        """Fold the user's replacements into the palette the print uses.

        Always rebuilt from ``_auto_palette`` rather than from the live palette:
        the live one already carries the previous replacements, so folding a
        removal into it would leave the removed colour stuck in place.
        """
        if self._result is None:
            return
        palette = list(self._auto_palette) if self._auto_palette else list(self._result.palette)
        for index, entry in self._overrides.items():
            if 0 <= index < len(palette):
                palette[index] = entry
        self._result = replace(self._result, palette=palette)
        self._plate = None
        self._view.setResult(self._result, self._source_qimage)
        self._list.refresh()
        self._refresh_base_combo()
        index = self._current_index()
        if index >= 0:
            self._select(index)
        self._announce_overrides()

    def _announce_overrides(self) -> None:
        if not self._overrides:
            self._status.setText(
                "匹配完成：{0}×{1} 像素，{2} 种颜色。点击右侧颜色看图片高光，"
                "点击图片看它属于哪种颜色；点「更换颜色…」可以手动换掉不合适的颜色。".format(
                    self._result.width, self._result.height, len(self._result.palette)
                )
                if self._result is not None
                else ""
            )
            return
        self._status.setText(
            f"已手动更换 {len(self._overrides)} 种颜色，导出的 3MF / OBJ 会用你选的颜色。"
        )

    def _on_replace(self) -> None:
        index = self._current_index()
        if index < 0 or self._result is None or self._library is None:
            return
        entries = build_palette(
            self._library,
            self._catalog,
            include_mixes=bool(self._include_mixes.isChecked()),
        )
        if not entries:
            self._status.setText("调色板是空的，请先添加耗材。")
            return
        dialog = ColourPickerDialog(
            entries,
            self._result.region_colour(index),
            library=self._library,
            current_key=self._result.palette[index].key,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        chosen = dialog.chosen()
        if chosen is None:
            return
        self._overrides[index] = chosen
        self._apply_overrides()

    def _on_restore(self) -> None:
        index = self._current_index()
        if index < 0 or index not in self._overrides:
            return
        self._overrides.pop(index, None)
        self._apply_overrides()

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
        suggested = f"{self._image_name}_{datetime.now():%Y%m%d-%H%M%S}{suffix}"
        start = str(paths.exports_dir() / suggested)
        name, _ = QFileDialog.getSaveFileName(self, f"保存{label}", start, f"{label} (*{suffix})")
        if not name:
            return
        target = Path(name)
        try:
            if kind == "3mf":
                # The filament list travels inside the project so Bambu Studio
                # opens it with *our* colours instead of whatever is in the AMS.
                filaments = (
                    slots_from_palette(self._result.palette, self._library)
                    if self._result is not None
                    else None
                )
                written = write_3mf(
                    plate, target, object_name=self._image_name, filaments=filaments
                )
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
            f"整板 1 个对象 / {stats['extruders']} 种耗材 / {stats['triangles']} 个三角面",
        ]
        lines.extend(plate.notes)
        return "  ·  ".join(lines)

    def _announce(self, plate: PlateModel, written: Path) -> None:
        stats = plate.stats()
        box = QMessageBox(self)
        box.setWindowTitle("已生成模型")
        box.setIcon(QMessageBox.Icon.Information)
        filaments = (
            slots_from_palette(self._result.palette, self._library)
            if self._result is not None
            else None
        )
        rows = [
            f"文件：{written}",
            f"尺寸：{stats['width_mm']} × {stats['depth_mm']} mm，"
            f"底板 {plate.base_thickness_mm:g} mm + 颜色层 {plate.colour_thickness_mm:g} mm",
            f"整块底板是一个对象，{stats['extruders']} 种颜色涂在不同的三角面上。",
            "",
            "在 Bambu Studio 里直接打开这个文件就行：",
            "1. 耗材清单（颜色 / 种类 / 品牌）已经写进 3MF，打开时会自动带进来，",
            "   不再取决于你 AMS 里插的是什么颜色的料；",
            "2. 整块底板是一个对象，不用再拼零件；",
            "3. 切片前确认「耗材映射」与你的 AMS 槽位一致。",
            "",
            "这个模型需要的耗材：",
        ]
        for index in range(1, plate.extruder_count + 1):
            slot = filaments[index - 1] if filaments and index <= len(filaments) else None
            colour = slot.color_hex if slot is not None else "?"
            kind = slot.material_type if slot is not None else ""
            vendor = slot.vendor if slot is not None else ""
            who = " ".join(part for part in (vendor, kind) if part)
            rows.append(f"  {index}. {colour}  {who}".rstrip())
        if plate.notes:
            rows.append("")
            rows.extend(f"提示：{note}" for note in plate.notes)
        box.setText("\n".join(rows))
        box.exec()

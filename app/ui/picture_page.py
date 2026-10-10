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
from PySide6.QtGui import QColor, QImage, QKeySequence, QPainter, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
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


def _scrollable(widget: QWidget) -> QScrollArea:
    """Put a widget in a frameless, resizable scroll area — the house style."""
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.Shape.NoFrame)
    area.setWidget(widget)
    return area


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
        keep = result.indices == self._selected
        printed = result.indices >= 0
        # A wash rather than a solid fill: the rest of the picture stays legible
        # as context, so a small highlighted region is not floating in a void.
        canvas[printed & ~keep] = (246, 247, 249, 168)
        # Deleted pixels stay fully transparent even while a colour is
        # highlighted: nothing is printed there, and the preview has to say so.
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
    #: The SET of selected rows changed — the page re-reads ``selected_indices``.
    selectionChanged = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._result: MatchResult | None = None
        self._order: list[int] = []
        self._pinned = -1
        self._suppress = False
        self._library: FilamentLibrary | None = None
        #: The rows the user has selected.  Kept here rather than read back from
        #: the widget because ``_refresh`` clears the list on every pin, and the
        #: selection has to survive that.
        self._picked: set[int] = set()

        self._list = QListWidget()
        self._list.setIconSize(compare_pixmap("#FFFFFF", "#000000").size())
        self._list.setUniformItemSizes(False)
        self._list.setAlternatingRowColors(False)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self._list.setWordWrap(False)
        # 「删除」 and 「合并」 both work on several colours at once, so the list
        # has to allow a multi-selection; a plain click still selects just one.
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._list.currentRowChanged.connect(self._on_row_changed)
        self._list.itemSelectionChanged.connect(self._on_selection_changed)
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
        self._picked = set()
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

    def selected_indices(self) -> list[int]:
        """Every selected colour, in list order — the input to 删除 / 合并."""
        result = self._result
        if result is None:
            return []
        live = {int(self._list.item(row).data(Qt.ItemDataRole.UserRole)) for row in range(self._list.count())}
        return sorted(index for index in self._picked if index in live)

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
                # No hover tooltip here: the row already prints the picture's
                # own colour, the matched colour, the share and the recipe, so
                # the black pop-up only repeated it.
                self._list.addItem(row)
            # Deleting or merging rebuilds the rows, and the user's selection has
            # to come back with them; a colour that is gone simply drops out.
            live = {
                int(self._list.item(row).data(Qt.ItemDataRole.UserRole))
                for row in range(self._list.count())
            }
            self._picked &= live
            for row in range(self._list.count()):
                item = self._list.item(row)
                if int(item.data(Qt.ItemDataRole.UserRole)) in self._picked:
                    item.setSelected(True)
            if self._pinned >= 0 and self._list.count():
                self._list.setCurrentRow(0)
        else:
            self._picked = set()
        self._suppress = False

    def _update_summary(self) -> None:
        result = self._result
        if result is None:
            self._summary.setText("导入图片后，这里列出打印需要的颜色。")
            return
        selected = len(self.selected_indices())
        extra = f" · 选中 {selected} 种" if selected else ""
        self._summary.setText(
            f"{result.width}×{result.height} 像素 · 实心 {result.printed_pixels} 像素"
            f" · 共 {len(result.palette)} 种颜色{extra}"
        )

    def _on_row_changed(self, row: int) -> None:
        if self._suppress or row < 0:
            return
        item = self._list.item(row)
        if item is None:
            return
        index = int(item.data(Qt.ItemDataRole.UserRole))
        self.colourSelected.emit(index)

    def _on_selection_changed(self) -> None:
        if self._suppress:
            return
        self._picked = set(self.selected_indices())
        self._update_summary()
        self.selectionChanged.emit()

    def _on_row_double_clicked(self, item: QListWidgetItem) -> None:
        self.colourActivated.emit(int(item.data(Qt.ItemDataRole.UserRole)))


class MergeDialog(QDialog):
    """Ask which of the selected colours the merged region should become.

    The user's requirement is explicit: 「合并后弹出窗口，从多选的颜色里选择一个
    颜色作为合并后的整体颜色」 — so the dialog lists exactly the selected colours
    and nothing else.
    """

    def __init__(self, entries, *, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("合并颜色")
        self._list = QListWidget()
        self._list.setIconSize(swatch_pixmap("#FFFFFF", 44, 20).size())
        for index, entry in enumerate(entries):
            row = QListWidgetItem(
                swatch_pixmap(entry.color_hex, 44, 20),
                f"{entry.color_hex}    {entry.short_label}",
            )
            row.setData(Qt.ItemDataRole.UserRole, index)
            self._list.addItem(row)
        self._list.setCurrentRow(0)
        self._list.itemDoubleClicked.connect(lambda _item: self.accept())

        hint = QLabel("这些颜色会并成一种，整块区域都用你选中的这个颜色打印。")
        hint.setProperty("role", "hint")
        hint.setWordWrap(True)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        ok.setText("合并")
        ok.setProperty("accent", "true")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        box = QVBoxLayout(self)
        box.setContentsMargins(14, 14, 14, 14)
        box.setSpacing(10)
        box.addWidget(hint)
        box.addWidget(self._list, 1)
        box.addWidget(buttons)
        self.resize(430, 340)

    def chosen(self) -> int:
        """Position inside the list this dialog was opened with, or -1."""
        item = self._list.currentItem()
        if item is None:
            return -1
        return int(item.data(Qt.ItemDataRole.UserRole))


class PicturePage(QWidget):
    """Import a picture, match it against the filament library, export a plate."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._library: FilamentLibrary | None = None
        self._catalog: MixCatalog | None = None
        self._image: Image.Image | None = None
        self._image_name: str = "混色底板"
        self._source_qimage: QImage | None = None
        # ``_base`` is the untouched match; ``_result`` is the base plus whatever
        # the user did to it.  Everything the user does is stored as an edit
        # against the ORIGINAL palette indices, never by mutating the pixels, so
        # undo and redo are a snapshot of three small dictionaries rather than a
        # megabyte of image data.
        self._base: MatchResult | None = None
        self._result: MatchResult | None = None
        self._plate: PlateModel | None = None
        #: Original indices the user deleted — those pixels become transparent,
        #: which means no base plate and nothing printed there.
        self._deleted: set[int] = set()
        #: ``source -> target``, original indices: the source's pixels are
        #: printed with the target's colour.
        self._merged: dict[int, int] = {}
        #: ``original index -> PaletteEntry``, the 「更换颜色…」 replacements.
        self._replaced: dict[int, PaletteEntry] = {}
        self._undo_stack: list[tuple] = []
        self._redo_stack: list[tuple] = []
        #: Live palette index -> original index, rebuilt by ``_compose``.
        self._live_map: list[int] = []
        # Set when the library changed while this page was off screen; the
        # re-match is then done in showEvent instead of in setLibrary.
        self._stale = False
        # Cache for _candidate_palette: the 全部颜色 list and the checkbox flag
        # it was built for.
        self._candidates: list[PaletteEntry] | None = None
        self._candidates_for: bool | None = None
        self._build_ui()
        self._set_controls_enabled(False)
        self._update_actions()

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

        # 原图 and 预览 sit side by side so the user can compare what they
        # imported with what the printer will actually lay down.  The original
        # never shows the matched render, so it stays a true reference.
        self._original_view = PictureView()
        self._original_view.setShowMatched(False)
        self._original_view.regionClicked.connect(self._on_region_clicked)

        self._list = ColourList()
        self._list.colourSelected.connect(self._on_colour_selected)
        self._list.colourActivated.connect(self._on_colour_activated)
        self._list.selectionChanged.connect(self._update_actions)

        # 「删除 / 合并 / 撤销 / 恢复」 sit right under the colour list, because
        # they are all about the selection in it.
        self._delete_button = QPushButton("删除")
        self._delete_button.setToolTip(
            "删掉选中的颜色：这些地方留空、变成透明，不打底板也不打印。可以多选。"
        )
        self._delete_button.clicked.connect(self._on_delete_colours)

        self._merge_button = QPushButton("合并")
        self._merge_button.setToolTip(
            "把选中的几种颜色并成一种，并成哪一种由你在弹窗里挑。至少要选两种。"
        )
        self._merge_button.clicked.connect(self._on_merge_colours)

        self._undo_button = QPushButton("撤销")
        self._undo_button.setToolTip("撤销上一步：删除、合并或更换颜色（Ctrl+Z）")
        self._undo_button.clicked.connect(self._undo)

        self._redo_button = QPushButton("恢复")
        self._redo_button.setToolTip("把撤销掉的一步再做回来（Ctrl+Y）")
        self._redo_button.clicked.connect(self._redo)

        actions = QHBoxLayout()
        actions.setSpacing(6)
        actions.addWidget(self._delete_button)
        actions.addWidget(self._merge_button)
        actions.addStretch(1)
        actions.addWidget(self._undo_button)
        actions.addWidget(self._redo_button)

        # The shortcuts are on the page, so Ctrl+Z works whichever of the two
        # panels has the focus.
        self._undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self._undo_shortcut.activated.connect(self._undo)
        self._redo_shortcut = QShortcut(QKeySequence("Ctrl+Y"), self)
        self._redo_shortcut.activated.connect(self._redo)

        list_box = QWidget()
        list_layout = QVBoxLayout(list_box)
        list_layout.setContentsMargins(0, 0, 0, 0)
        list_layout.setSpacing(6)
        list_layout.addWidget(self._list, 1)
        list_layout.addLayout(actions)

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
        right.addWidget(self._wrap(list_box, "这个模型需要的颜色"))
        right.addWidget(self._wrap(detail_scroll, "选中的颜色"))
        right.setStretchFactor(0, 3)
        right.setStretchFactor(1, 2)
        right.setCollapsible(1, False)
        right.setSizes([420, 340])

        pictures = QSplitter(Qt.Orientation.Horizontal)
        pictures.addWidget(self._wrap(_scrollable(self._original_view), "原图"))
        pictures.addWidget(self._wrap(_scrollable(self._view), "预览"))
        # 预览 is the one the user works in, so it gets the extra room.
        pictures.setStretchFactor(0, 1)
        pictures.setStretchFactor(1, 1)
        pictures.setSizes([340, 420])

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(pictures)
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
        ):
            widget.setEnabled(enabled)

    # -- library ----------------------------------------------------------
    def setLibrary(self, library: FilamentLibrary, catalog: MixCatalog | None) -> None:
        """Called whenever the filament library or the mix catalogue changes."""
        self._library = library
        self._catalog = catalog
        self._candidates = None
        if self._image is None:
            return
        if not self.isVisible():
            # The 混色配方 tab is showing, so nobody can see the picture.  A
            # re-match against 66,461 candidates costs ~0.7 s and would be spent
            # on a result that is not on screen; remember that it is owed and
            # do it when the tab comes back.
            self._stale = True
            return
        self._on_rematch()

    def showEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        super().showEvent(event)
        if self._stale:
            self._stale = False
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

    def _candidate_palette(self) -> list[PaletteEntry]:
        """The full 全部颜色 list, built once per library / checkbox state.

        At 41 spools this is 66,461 ``PaletteEntry`` objects and ~0.5 s to
        build.  It is needed both by the match and by the 「更换颜色…」 menu, and
        that menu is opened once per colour the user swaps — so rebuilding it
        every time was pure waste.  ``setLibrary`` drops the cache, and the
        checkbox flag is part of the key, so it can never go stale.
        """
        wanted = bool(self._include_mixes.isChecked())
        if self._candidates is None or self._candidates_for != wanted:
            self._candidates = build_palette(self._library, self._catalog, include_mixes=wanted)
            self._candidates_for = wanted
        return self._candidates

    def _on_rematch(self) -> None:
        if self._image is None:
            return
        if self._library is None or len(self._library) == 0:
            self._status.setText("请先在「混色配方」页添加耗材，再匹配图片。")
            return
        palette = self._candidate_palette()
        if not palette:
            self._status.setText("调色板是空的，请先添加耗材。")
            return
        try:
            result = match_image(self._image, palette, self._settings())
        except Exception as error:  # pragma: no cover - defensive
            QMessageBox.warning(self, "匹配失败", str(error))
            return
        self._base = result
        self._deleted = set()
        self._merged = {}
        self._replaced = {}
        self._undo_stack = []
        self._redo_stack = []
        self._plate = None
        self._list.setLibrary(self._library)
        self._apply_edits()
        self._set_controls_enabled(True)
        self._announce_edits()

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

    # -- two-way selection ------------------------------------------------
    def _select(self, index: int) -> None:
        """Show one colour everywhere it appears: picture, list and detail."""
        self._view.setSelected(index)
        self._original_view.setSelected(index)
        self._list.pin(index)
        if self._result is None:
            return
        if not 0 <= index < len(self._result.palette):
            return
        original = None
        changed = False
        if index < len(self._live_map):
            source = self._live_map[index]
            if self._base is not None and source < len(self._base.palette):
                original = self._base.palette[source]
            changed = source in self._replaced
        self._detail.setSelection(
            self._result,
            index,
            library=self._library,
            changed=changed,
            original=original,
        )

    def _on_colour_selected(self, index: int) -> None:
        self._select(index)

    def _on_region_clicked(self, index: int) -> None:
        self._select(index)

    def _on_colour_activated(self, index: int) -> None:
        self._select(index)

    # -- the user's edits: 删除 / 合并 / 更换颜色 / 撤销 / 恢复 ---------------
    def _current_index(self) -> int:
        index = self._list.current()
        if index < 0:
            return -1
        if self._result is None or not 0 <= index < len(self._result.palette):
            return -1
        return index

    def _snapshot(self) -> tuple:
        return (set(self._deleted), dict(self._merged), dict(self._replaced))

    def _restore(self, snapshot: tuple) -> None:
        deleted, merged, replaced = snapshot
        self._deleted = set(deleted)
        self._merged = dict(merged)
        self._replaced = dict(replaced)

    def _push(self) -> None:
        """Remember the state before an edit, and drop the redo history."""
        self._undo_stack.append(self._snapshot())
        # A long session should not grow without bound; 64 steps is far more
        # than anyone retraces by hand.
        del self._undo_stack[:-64]
        self._redo_stack.clear()

    def _resolve(self) -> tuple[dict[int, int], set[int]]:
        """``(index remap, indices that vanish)`` after the merges and deletions.

        Chains are followed — merging A into B and then B into C prints both as
        C — and a colour that was deleted takes everything merged into it with
        it, because those pixels are already printed as the deleted colour.
        """
        merged = dict(self._merged)
        for source in list(merged):
            target = merged[source]
            seen = {source}
            while target in merged and target not in seen:
                seen.add(target)
                target = merged[target]
            merged[source] = target
        gone = set(self._deleted)
        for source, target in merged.items():
            if target in gone:
                gone.add(source)
        return merged, gone

    def _compose(self) -> MatchResult | None:
        """Rebuild the live result from ``_base`` and the edit state.

        Always from the pristine match, never from the live result: the live one
        already carries the previous edits, so folding a new deletion into it
        would leave the deleted pixels behind.  Deleted pixels become ``-1``,
        which is exactly the transparent index the matcher already uses — so
        ``build_plate`` gives them no base plate and no colour prism.
        """
        base = self._base
        if base is None:
            return None
        merged, gone = self._resolve()
        indices = np.array(base.indices, dtype=np.int32, copy=True)
        for source, target in merged.items():
            if source in gone or source == target:
                continue
            mask = indices == source
            if mask.any():
                indices[mask] = target
        for index in gone:
            indices[indices == index] = -1

        # A colour survives when it was not deleted and nothing was merged into
        # it... rather: when it was not deleted and is not itself a merge source.
        alive = [
            index
            for index in range(len(base.palette))
            if index not in gone and index not in merged
        ]
        palette = [self._replaced.get(index, base.palette[index]) for index in alive]
        if alive:
            totals = np.bincount(
                indices[indices >= 0].ravel(), minlength=len(base.palette)
            )
            counts = np.asarray([totals[index] for index in alive], dtype=base.counts.dtype)
        else:
            counts = np.zeros(0, dtype=base.counts.dtype)
        means = None
        if base.means is not None and len(base.means) == len(base.palette):
            means = np.asarray([base.means[index] for index in alive], dtype=base.means.dtype)

        live = np.full(indices.shape, -1, dtype=np.int32)
        for position, original in enumerate(alive):
            mask = indices == original
            if mask.any():
                live[mask] = position
        self._live_map = alive
        return replace(
            base,
            palette=palette,
            indices=live,
            counts=counts,
            means=means,
        )

    def _apply_edits(self, keep: int | None = None) -> None:
        """Recompose the live result and show it everywhere.

        ``keep`` is an ORIGINAL palette index to keep selected across the
        rebuild; without it the colour that is currently selected is followed
        through the edit, so 「更换颜色…」 and 撤销 both leave the user where they
        were instead of jumping back to the top of the list.
        """
        anchor = keep
        if anchor is None:
            index = self._list.current()
            if 0 <= index < len(self._live_map):
                anchor = self._live_map[index]
        self._result = self._compose()
        self._plate = None
        result = self._result
        if result is None:
            self._view.setResult(None, self._source_qimage)
            self._original_view.setResult(None, self._source_qimage)
            self._list.setResult(None)
            self._refresh_base_combo()
            self._detail.clear()
            self._update_actions()
            self._announce_edits()
            return
        self._view.setResult(result, self._source_qimage)
        self._original_view.setResult(result, self._source_qimage)
        self._list.setResult(result)
        self._refresh_base_combo()
        target = self._live_map.index(anchor) if anchor in self._live_map else -1
        if target < 0 and len(result.palette):
            target = 0
        if 0 <= target < len(result.palette):
            self._select(target)
        else:
            self._detail.clear()
        self._update_actions()
        self._announce_edits()

    def _update_actions(self) -> None:
        """ 删除 needs one colour, 合并 needs two, and the history drives the rest."""
        picked = self._list.selected_indices() if self._result is not None else []
        live = bool(self._result is not None and len(self._result.palette))
        self._delete_button.setEnabled(live and len(picked) >= 1)
        self._merge_button.setEnabled(live and len(picked) >= 2)
        self._undo_button.setEnabled(bool(self._undo_stack))
        self._redo_button.setEnabled(bool(self._redo_stack))

    def _announce_edits(self) -> None:
        result = self._result
        if result is None:
            return
        parts = []
        if self._deleted:
            parts.append(f"删除 {len(self._deleted)} 种颜色（留空不打印）")
        if self._merged:
            parts.append(f"合并 {len(self._merged)} 种颜色")
        if self._replaced:
            parts.append(f"手动更换 {len(self._replaced)} 种颜色")
        if not parts:
            self._status.setText(
                "匹配完成：{0}×{1} 像素，{2} 种颜色。点击右侧颜色看图片高光，"
                "点击图片看它属于哪种颜色；点「更换颜色…」可以手动换掉不合适的颜色，"
                "多选颜色后可以「删除」（留空不打印）或「合并」。".format(
                    result.width, result.height, len(result.palette)
                )
            )
            return
        printed = result.printed_pixels
        total = max(1, result.total_pixels)
        self._status.setText(
            "、".join(parts)
            + f"，现在打 {len(result.palette)} 种颜色，实心 {printed} 像素"
            f"（占 {printed / total * 100:.1f}%）。导出的 3MF / OBJ 就照这样切，"
            "Ctrl+Z 撤销、Ctrl+Y 恢复。"
        )

    def _on_delete_colours(self) -> None:
        picked = self._list.selected_indices()
        if not picked:
            return
        originals = [self._live_map[i] for i in picked if 0 <= i < len(self._live_map)]
        if not originals:
            return
        self._push()
        self._deleted.update(originals)
        self._apply_edits()
        self._status.setText(
            f"已删除 {len(originals)} 种颜色：这些地方留空、变成透明，"
            "不打印也没有底板。Ctrl+Z 可以撤销。"
        )

    def _on_merge_colours(self) -> None:
        picked = self._list.selected_indices()
        if len(picked) < 2 or self._result is None:
            return
        entries = [self._result.palette[index] for index in picked]
        dialog = MergeDialog(entries, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        choice = dialog.chosen()
        if not 0 <= choice < len(picked):
            return
        target = self._live_map[picked[choice]]
        originals = [self._live_map[index] for index in picked]
        self._push()
        for original in originals:
            if original != target:
                self._merged[original] = target
        self._apply_edits(keep=target)
        self._status.setText(
            f"已把 {len(originals)} 种颜色合并成 {entries[choice].color_hex}，"
            "整块区域都用它打印。Ctrl+Z 可以撤销。"
        )

    def _undo(self) -> None:
        if not self._undo_stack:
            return
        self._redo_stack.append(self._snapshot())
        self._restore(self._undo_stack.pop())
        self._apply_edits()
        self._status.setText("已撤销上一步。Ctrl+Y 可以再做回来。")

    def _redo(self) -> None:
        if not self._redo_stack:
            return
        self._undo_stack.append(self._snapshot())
        self._restore(self._redo_stack.pop())
        self._apply_edits()
        self._status.setText("已恢复刚才撤销的那一步。")

    def _on_replace(self) -> None:
        index = self._current_index()
        if index < 0 or self._result is None or self._library is None:
            return
        entries = self._candidate_palette()
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
        original = self._live_map[index]
        self._push()
        self._replaced[original] = chosen
        self._apply_edits(keep=original)

    def _on_restore(self) -> None:
        index = self._current_index()
        if index < 0 or index >= len(self._live_map):
            return
        original = self._live_map[index]
        if original not in self._replaced:
            return
        self._push()
        self._replaced.pop(original, None)
        self._apply_edits(keep=original)

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
        # Building the plate for a 512×512 picture with 12 colours takes ~2 s and
        # writing the 3MF another ~0.4 s.  Without a wait cursor and a line of
        # text the window simply stops answering, which is the whole 「未响应」
        # complaint; the cursor is popped again on every exit path.
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self._status.setText("正在生成模型…")
        QApplication.processEvents()
        try:
            plate = self.buildPlate()
        except Exception as error:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "无法生成模型", str(error))
            return
        QApplication.restoreOverrideCursor()
        suffix = ".3mf" if kind == "3mf" else ".obj"
        label = "3MF 模型" if kind == "3mf" else "OBJ 模型"
        suggested = f"{self._image_name}_{datetime.now():%Y%m%d-%H%M%S}{suffix}"
        start = str(paths.exports_dir() / suggested)
        name, _ = QFileDialog.getSaveFileName(self, f"保存{label}", start, f"{label} (*{suffix})")
        if not name:
            return
        target = Path(name)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self._status.setText(f"正在写入 {target.name}…")
        QApplication.processEvents()
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
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "保存失败", str(error))
            return
        QApplication.restoreOverrideCursor()
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

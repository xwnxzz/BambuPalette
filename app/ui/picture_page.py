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

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import QEvent, QEventLoop, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QKeySequence, QPainter, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
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
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .. import __version__
from ..core import paths
from ..core.image_matching import (
    BudgetOption,
    MatchResult,
    MatchSettings,
    PaletteEntry,
    TriplePalette,
    build_palette,
    match_options,
    plan_caption,
    rematch_options,
    spool_usage,
)
from ..core.library import FilamentLibrary
from ..core.mixes import DEFAULT_ENGINE, MixCatalog
from ..core.project import (
    PROJECT_FILTER,
    PROJECT_SUFFIX,
    ProjectError,
    ProjectState,
    load_project,
    render_preview,
    save_project,
)
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
from .plan_filter import SpoolFilterDialog
from .selectable import selectable_text
from .swatch import swatch_pixmap

__all__ = ["PicturePage", "PictureView", "ColourList"]

IMAGE_FILTER = "图片 (*.png *.jpg *.jpeg *.bmp *.webp *.gif *.tif *.tiff);;所有文件 (*)"

#: Wheel zoom runs between 「整张铺满面板」 (1.0) and this many times the fit.
ZOOM_MIN = 1.0
ZOOM_MAX = 12.0
#: One wheel notch multiplies the zoom by this much.
ZOOM_STEP = 1.25
#: A press that moves fewer than this many pixels counts as a click, not a drag.
DRAG_SLOP = 3.0
#: The two picture panels are panned and zoomed the same way.
ZOOM_HINT = "滚轮缩放，按住左键拖动平移，双击还原成整张。"

#: 图片匹配默认值。颜色数量直接顶到 Bambu Studio 的天花板：混色就是为了用少数
#: 几卷料凑出一整张图，用户既然在用混色，就没有理由再自己往回缩。
DEFAULT_MAX_COLOURS = MAX_PAINTED_FILAMENTS
#: 一次打印最常用的卷数（AMS 四槽）。0 才是「不限」，默认值必须是真数字。
DEFAULT_SPOOL_CAP = 4

#: 「重新匹配」 on a real library takes 1.3–3.0 s (measured: 41 卷 + 三色池 =
#: 3.03 s，其中 1.38 s 是 CIEDE2000 搜索、1.47 s 是「耗材上限」合并）。进度条只在
#: 慢到值得看的时候才出现：小图匹配几十毫秒就完了，点一下闪一下比不显示更烦人。
BUSY_DELAY_MS = 200
#: The bar's internal resolution.  Percentages land on it without rounding to
#: 「0% 卡住」, and a plain 0–100 range is too coarse while the search walks.
PROGRESS_STEPS = 1000

#: 图片页面原来那行常驻说明。1.6.14 把它整段搬进「帮助」窗口（m01885 第 6 条）：
#: 页面上只留随图片变化的东西，要读的使用说明去帮助里读，不再压着图片。
PAGE_HELP = (
    "图像转换这一步：\n"
    f"· {ZOOM_HINT}\n"
    f"· 颜色数量最多 {MAX_PAINTED_FILAMENTS} 是 Bambu Studio 自己的限制：一个工程里只认 "
    f"{MAX_PAINTED_FILAMENTS} 个耗材位，第 {MAX_PAINTED_FILAMENTS + 1} 种颜色在切片时会被"
    "丢掉（我们已经就这条限制向拓竹提了工单）。耗材上限 0 表示不限制。\n"
    "· 改到一半可以存成工程接着改：Ctrl+S 保存（第一次会问存哪儿），Ctrl+Shift+S 换一个文件名；"
    "工程里装着原图、这张预览、用到的颜色和耗材，下次「打开工程」就回到现在这一步。\n"
    "· 双击颜色列表里的一行，看这个颜色的详情（图片里的颜色、匹配到的颜色和它的配方）。\n"
    "· 「筛选耗材」先说好哪几卷必须用、哪几卷不要用，之后每套搭配都照这个来。\n"
    "· 「按当前色块重新匹配」拿现在这些色块重新挑卷，删除、合并和手改过的颜色都留着。"
)


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
        self._plain = False
        self._scale = 1.0
        self._origin = QPointF(0.0, 0.0)
        # 缩放和平移叠在「铺满面板」之上：_fit_scale 由面板尺寸算出来，_zoom 是
        # 用户用滚轮乘上去的倍数，_pan 是像素位移。三者分开以后，改窗口大小仍然
        # 是以「铺满」为基准，不会把之前放大的倍数算丢。
        self._fit_scale = 1.0
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._press_pos: QPointF | None = None
        self._last_pos: QPointF | None = None
        self._dragged = False

    # -- contents ---------------------------------------------------------
    def setResult(self, result: MatchResult | None, source: QImage | None) -> None:
        previous = self._source
        self._result = result
        self._source = source
        self._matched = self._build_matched(result) if result is not None else None
        self._selected = -1
        self._highlight = None
        same_picture = (
            previous is not None
            and source is not None
            and previous.width() == source.width()
            and previous.height() == source.height()
        )
        if not same_picture:
            # 换了图片（或者清空了）就回到「整张铺满」，否则用户一打开新图看到的
            # 是上一张留下来的某个角落。
            self.reset_zoom(update=False)
        self.update()

    def setSelected(self, index: int) -> None:
        if self._plain:
            # 原图 must never react to the selection: the whole point of the
            # panel is to show what the user uploaded, next to what the match
            # made of it.  Dimming it made the upload look like a washed-out
            # copy of the preview.
            return
        if index == self._selected:
            return
        self._selected = int(index)
        self._highlight = None
        self.update()

    def setShowMatched(self, matched: bool) -> None:
        self._show_matched = bool(matched)
        self.update()

    def setPlain(self, plain: bool) -> None:
        """Show the uploaded picture untouched, ignoring selection and preview.

        Used by the 原图 panel: it always draws ``source``, so it stays the
        user's own file no matter which colour is selected on the right.
        """
        self._plain = bool(plain)
        if self._plain:
            self._selected = -1
            self._highlight = None
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
        self._fit_scale = min(available_w / width, available_h / height)
        self._scale = self._fit_scale * self._zoom
        draw_w = width * self._scale
        draw_h = height * self._scale
        if self._zoom <= ZOOM_MIN:
            # 没有放大就没有可平移的余量：永远居中，免得图片被拖到面板外。
            self._pan = QPointF(0.0, 0.0)
        self._pan = self._clamped_pan(self._pan, draw_w, draw_h)
        self._origin = QPointF(
            (self.width() - draw_w) / 2.0 + self._pan.x(),
            (self.height() - draw_h) / 2.0 + self._pan.y(),
        )
        return QRectF(self._origin.x(), self._origin.y(), draw_w, draw_h)

    def _draw_size(self) -> tuple[float, float]:
        """How big the picture is drawn right now, in panel pixels."""
        result = self._result
        if result is None:
            return 0.0, 0.0
        scale = self._fit_scale * self._zoom
        return result.width * scale, result.height * scale

    def _clamped_pan(self, pan: QPointF, draw_w: float, draw_h: float) -> QPointF:
        """Hold the pan inside the picture, so panning never opens a gap."""
        limit_x = max(0.0, (draw_w - self.width()) / 2.0)
        limit_y = max(0.0, (draw_h - self.height()) / 2.0)
        return QPointF(
            min(max(pan.x(), -limit_x), limit_x),
            min(max(pan.y(), -limit_y), limit_y),
        )

    def zoom(self) -> float:
        return self._zoom

    def reset_zoom(self, *, update: bool = True) -> None:
        """Back to 「整张铺满面板」, centred."""
        self._zoom = ZOOM_MIN
        self._pan = QPointF(0.0, 0.0)
        if update:
            self.update()

    def zoom_by(self, steps: float, focus: QPointF | None = None) -> None:
        """Zoom by ``steps`` wheel notches around ``focus`` (panel pixels).

        Public so the self-check can drive it without a real wheel event.
        """
        if self._result is None or steps == 0.0:
            return
        if focus is None:
            focus = QPointF(self.width() / 2.0, self.height() / 2.0)
        self._zoom_to(self._zoom * (ZOOM_STEP**steps), focus)

    def _zoom_to(self, zoom: float, focus: QPointF) -> None:
        """Zoom so the picture point under ``focus`` stays under ``focus``."""
        result = self._result
        if result is None:
            return
        rect = self._target_rect()  # also refreshes _fit_scale for this panel size
        if self._scale <= 0.0:
            return
        image_x = (focus.x() - rect.x()) / self._scale
        image_y = (focus.y() - rect.y()) / self._scale
        zoom = min(max(zoom, ZOOM_MIN), ZOOM_MAX)
        if abs(zoom - self._zoom) < 1e-9:
            return
        self._zoom = zoom
        scale = self._fit_scale * zoom
        draw_w = result.width * scale
        draw_h = result.height * scale
        pan = QPointF(
            focus.x() - (self.width() - draw_w) / 2.0 - image_x * scale,
            focus.y() - (self.height() - draw_h) / 2.0 - image_y * scale,
        )
        self._pan = self._clamped_pan(pan, draw_w, draw_h)
        self.update()

    def _base_image(self) -> QImage | None:
        if self._plain or not self._show_matched:
            return self._source
        return self._matched

    def _dim_pixmap(self) -> QPixmap | None:
        """The picture with everything but the selected colour faded out.

        Built once per selection and cached: it walks the whole class map, and a
        repaint happens for every wheel notch, every resize and every time the
        window is activated again — including when the user clicks another
        program.  The cache therefore has to be *handed back*, not merely
        remembered.  It used to return ``None`` whenever it was already built,
        so the block showed for exactly one frame and disappeared at the next
        repaint: zooming in and out or clicking another window wiped it.
        """
        result = self._result
        if self._plain or result is None or self._selected < 0:
            return None
        if self._highlight is None:
            self._highlight = self._build_highlight(result, self._selected)
        return self._highlight

    def _build_highlight(self, result: MatchResult, selected: int) -> QPixmap:
        """Fade everything but ``selected`` and outline the region that keeps it."""
        height, width = result.indices.shape
        canvas = np.zeros((height, width, 4), dtype=np.uint8)
        keep = result.indices == selected
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

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._result is None:
            event.ignore()
            return
        steps = event.angleDelta().y() / 120.0
        if steps == 0.0:
            event.ignore()
            return
        # A high-resolution wheel reports fractions of a notch; the exponent form
        # handles those and whole notches with the same line.
        self.zoom_by(steps, event.position())
        event.accept()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.button() != Qt.MouseButton.LeftButton or self._result is None:
            return
        # 「点一下挑颜色」和「按住拖动平移」共用左键，所以按下时只记录起点，
        # 究竟算哪一种，等松手时按移动距离判断。
        position = event.position()
        self._press_pos = QPointF(position)
        self._last_pos = QPointF(position)
        self._dragged = False
        if self._zoom > ZOOM_MIN:
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._press_pos is None or self._last_pos is None:
            return
        position = event.position()
        if not self._dragged:
            moved = position - self._press_pos
            if abs(moved.x()) + abs(moved.y()) < DRAG_SLOP:
                return
            self._dragged = True
        if self._zoom <= ZOOM_MIN:
            return
        self._target_rect()  # refreshes _fit_scale before the pan is clamped
        draw_w, draw_h = self._draw_size()
        self._pan = self._clamped_pan(self._pan + (position - self._last_pos), draw_w, draw_h)
        self._last_pos = QPointF(position)
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.button() != Qt.MouseButton.LeftButton:
            return
        press = self._press_pos
        dragged = self._dragged
        self._press_pos = None
        self._last_pos = None
        self._dragged = False
        self.unsetCursor()
        if press is None:
            return
        if dragged:
            # 这是一次平移：画面动过了，鼠标下已经不是用户当初点的那个像素。
            return
        self._pick_at(event.position())

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.button() != Qt.MouseButton.LeftButton or self._zoom <= ZOOM_MIN:
            return
        # 双击 = 「我看够了，还我整张」。顺手丢掉这次按下留下的状态，免得松手时
        # 又当成一次点选。
        self._press_pos = None
        self._last_pos = None
        self._dragged = False
        self.unsetCursor()
        self.reset_zoom()

    def _pick_at(self, position: QPointF) -> None:
        """Report the colour of the pixel under ``position``, if there is one."""
        result = self._result
        if self._plain or result is None:
            return
        rect = self._target_rect()
        if not rect.contains(position):
            return
        x = int((position.x() - rect.x()) / self._scale)
        y = int((position.y() - rect.y()) / self._scale)
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
    #: The user clicked the row of the colour that is already on screen — put it
    #: away again (「再点一下取消选中」).
    colourDeselected = Signal(int)
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
        #: What the last press landed on, and which colour was on screen then.
        self._press_index = -1
        self._press_shown = -1

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
        # 「再点一下同一个颜色就取消选中」: Qt moves the current row during the
        # press and only then emits itemPressed, so by the time the click signal
        # arrives the row has already changed and 「the same colour again」 is no
        # longer recognisable.  The event filter notes what was on screen BEFORE
        # Qt touches anything.
        self._list.viewport().installEventFilter(self)
        self._list.itemClicked.connect(self._on_row_clicked)

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
        """Every selected colour, in list order — the input to 删除 / 合并.

        Read straight off the widget.  It used to filter the cached ``_picked``
        set instead, and ``_on_selection_changed`` filled that cache FROM this
        method — a closed loop in which a selection could never appear at all,
        so 删除 / 合并 / 撤销 / 恢复 stayed greyed out no matter what was clicked.
        """
        if self._result is None:
            return []
        picked = []
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item is not None and item.isSelected():
                picked.append(int(item.data(Qt.ItemDataRole.UserRole)))
        return picked

    def entries(self) -> list[PaletteEntry]:
        return list(self._result.palette) if self._result is not None else []

    def pin(self, index: int, *, reorder: bool = False) -> None:
        """Make ``index`` the current colour and reveal its row.

        ``reorder=True`` also moves it to the top of the list.  A plain click
        must NOT reorder: the list used to shuffle on every click, so the row
        that appeared under the cursor was a DIFFERENT colour than the one just
        clicked (the user reported exactly that).  The order is now the sort order
        and nothing else; only the sample screenshots ask for a colour to be
        shown first.
        """
        result = self._result
        if result is None or not (0 <= index < len(result.palette)):
            return
        self._pinned = int(index)
        if not reorder:
            self._reveal()
            return
        self._order = [self._pinned] + [i for i in range(len(result.palette)) if i != self._pinned]
        self._refresh()

    def _row_of(self, index: int) -> int:
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item is not None and int(item.data(Qt.ItemDataRole.UserRole)) == index:
                return row
        return -1

    def _reveal(self) -> None:
        """Highlight the current colour's row and scroll it into view.

        What happens to the selection depends on the mode, and getting this wrong
        greys 删除 / 合并 / 更换颜色 out again:

        * 「一次只选一种」 (``ExtendedSelection``): ``setCurrentRow`` selects this
          row and drops the others — that IS the new selection, so nothing is put
          back.  It runs suppressed, so the cache and the buttons are refreshed by
          hand afterwards.
        * 「多选」 (``MultiSelection``): the user's set must survive, because
          revealing a row is not a selection act.  ``setCurrentRow`` is free to
          drop the rest, so the set is read first and written back.
        """
        if self._pinned < 0:
            return
        row = self._row_of(self._pinned)
        if row < 0:
            return
        multi = self._list.selectionMode() == QAbstractItemView.SelectionMode.MultiSelection
        keep = self.selected_indices() if multi else []
        self._suppress = True
        self._list.setCurrentRow(row)
        self._list.scrollToItem(self._list.item(row), QAbstractItemView.ScrollHint.EnsureVisible)
        self._suppress = False
        if multi:
            self.select(keep)
        else:
            self._on_selection_changed()

    def toggle(self, index: int) -> None:
        """Add or remove one colour from the selection.

        This is what a click on a block in 「预览」 does while 「多选」 is on: the
        picture and the list are two views of the same regions, so pointing at a
        region in the picture has to work like pointing at its row.
        """
        row = self._row_of(int(index))
        if row < 0:
            return
        item = self._list.item(row)
        item.setSelected(not item.isSelected())
        self._reveal()

    def select(self, indices) -> None:
        """Select these palette indices, exactly as clicking them would.

        Used by the automatic checks to reach 删除 / 合并 without a mouse.  It
        must go THROUGH the widget (not the ``_picked`` cache) so the same
        ``itemSelectionChanged`` path runs as for a real selection.
        """
        wanted = {int(index) for index in indices}
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item is not None:
                item.setSelected(int(item.data(Qt.ItemDataRole.UserRole)) in wanted)

    def setMultiSelect(self, on: bool) -> None:
        """Make a plain click ADD to the selection instead of replacing it.

        Ctrl+click did this already, but a modifier nobody was told about is not
        a feature — the page puts a 「多选」 toggle in front of the buttons
        instead.  The mode is changed on the widget (we never fake a selection),
        so the same ``itemSelectionChanged`` path keeps 删除 / 合并 in sync.
        """
        mode = (
            QAbstractItemView.SelectionMode.MultiSelection
            if on
            else QAbstractItemView.SelectionMode.ExtendedSelection
        )
        if self._list.selectionMode() == mode:
            return
        keep = self.selected_indices()
        self._list.setSelectionMode(mode)
        # Qt is free to drop the selection while the mode changes; put it back
        # through the widget so the buttons still agree with what is highlighted.
        self.select(keep)

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
        else:
            self._picked = set()
        self._suppress = False
        # The rows moved: put the highlight back on the colour it was on, not on
        # whatever happens to be row 0 now.
        self._reveal()

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

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt spelling
        """Remember the row and the shown colour from before the press landed.

        ``QAbstractItemView.mousePressEvent`` sets the current index first and
        emits ``pressed``/``clicked`` afterwards; a filter on the viewport runs
        before both, which is the only place the previous state is still intact.
        """
        if watched is self._list.viewport() and event.type() == QEvent.Type.MouseButtonPress:
            position = event.position().toPoint()
            row = self._list.row(self._list.itemAt(position))
            item = self._list.item(row) if row >= 0 else None
            self._press_index = (
                int(item.data(Qt.ItemDataRole.UserRole)) if item is not None else -1
            )
            self._press_shown = self._pinned
        return super().eventFilter(watched, event)

    def _on_row_clicked(self, item: QListWidgetItem) -> None:
        """A click on the colour that was ALREADY on screen means 「取消选中」.

        「多选」 is left out: there a second click toggles the row off by itself,
        which already drops it out of the selection the buttons work on.
        """
        if self._suppress:
            return
        if self._list.selectionMode() == QAbstractItemView.SelectionMode.MultiSelection:
            return
        index = int(item.data(Qt.ItemDataRole.UserRole))
        if index == self._press_index == self._press_shown:
            self.colourDeselected.emit(index)

    def clear_selection(self) -> None:
        """Drop the highlight from every row, without touching ``_pinned``'s order."""
        self._pinned = -1
        self._list.clearSelection()
        self._list.setCurrentRow(-1)


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
        selectable_text(self)

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
        #: The ``.bpproj`` this picture came from, or the one it was last saved
        #: to.  Ctrl+S overwrites it without asking; ``None`` means the next save
        #: has to ask where to put the file.
        self._project_path: Path | None = None
        #: True while there are edits that are not in ``_project_path`` yet, so
        #: 「打开工程」 can warn before throwing them away.
        self._dirty = False
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
        #: ``original index -> PaletteEntry``, the 「更换颜色」 replacements.
        self._replaced: dict[int, PaletteEntry] = {}
        self._undo_stack: list[tuple] = []
        self._redo_stack: list[tuple] = []
        #: Live palette index -> original index, rebuilt by ``_compose``.
        self._live_map: list[int] = []
        #: The colour on screen right now (-1 = none), and the ones it replaced.
        #: Clicking the same row again steps back through this list, so
        #: 「取消选中」 returns to the preview the user came from.
        self._shown = -1
        self._preview_history: list[int] = []
        # Set when the library changed while this page was off screen; the
        # re-match is then done in showEvent instead of in setLibrary.
        self._stale = False
        # Cache for _candidate_palette: the 全部颜色 list, dropped by setLibrary.
        self._candidates: list[PaletteEntry] | None = None
        #: The three-filament catalogue, pushed in by the main window whenever it
        #: is (re)built.  ``None`` means 「not ready」: matching then runs on the
        #: two-filament palette alone, and picks up the triples when they land.
        self._triples = None
        self._triple_palette = None
        #: The head list ``_triple_palette`` was built from, so the cache can be
        #: invalidated by identity instead of by rebuilding the tail.
        self._triple_head: list[PaletteEntry] | None = None
        self._build_ui()
        self._set_controls_enabled(False)
        self._update_actions()

    # -- construction -----------------------------------------------------
    def _build_ui(self) -> None:
        self._import_button = QPushButton("导入图片")
        self._import_button.setProperty("accent", "true")
        self._import_button.clicked.connect(self._on_import)

        self._rematch_button = QPushButton("重新匹配")
        self._rematch_button.clicked.connect(self._on_rematch)

        # 「按当前色块重新匹配」 is 重新匹配's neighbour and its opposite: 重新匹配
        # reads the picture again (and drops the edits, because the blocks it was
        # made of are gone), while this one takes the blocks on screen as the
        # question and only asks which rolls print them.  Two buttons rather than
        # one with a mode, because the user knows which question they are asking.
        self._rematch_blocks_button = QPushButton("按当前色块重新匹配")
        self._rematch_blocks_button.clicked.connect(self._on_rematch_blocks)

        # 打开 / 保存工程 sit next to 导入图片: all three are file actions on the
        # picture, and a session that is put down and picked up again is the
        # whole point of the project file.  Neither is accented — the blue
        # button on this page is 生成 3MF, the one that finishes the job.
        self._open_project_button = QPushButton("打开工程")
        self._open_project_button.clicked.connect(self._on_open_project)

        self._save_project_button = QPushButton("保存工程")
        self._save_project_button.clicked.connect(lambda: self._on_save_project())

        self._colour_count = QSpinBox()
        self._colour_count.setRange(2, MAX_PAINTED_FILAMENTS)
        self._colour_count.setValue(DEFAULT_MAX_COLOURS)

        self._spool_cap = QSpinBox()
        self._spool_cap.setRange(0, MAX_PAINTED_FILAMENTS)
        self._spool_cap.setValue(DEFAULT_SPOOL_CAP)

        self._max_dimension = QSpinBox()
        self._max_dimension.setRange(32, 4096)
        self._max_dimension.setSingleStep(64)
        self._max_dimension.setValue(512)
        self._max_dimension.setSuffix(" px")

        self._dither = QCheckBox("抖动")

        top = QHBoxLayout()
        top.setSpacing(8)
        top.addWidget(self._import_button)
        top.addWidget(self._open_project_button)
        top.addWidget(self._save_project_button)
        top.addWidget(QLabel("颜色数量:"))
        top.addWidget(self._colour_count)
        top.addWidget(QLabel("耗材上限:"))
        top.addWidget(self._spool_cap)
        top.addWidget(QLabel("最大边长:"))
        top.addWidget(self._max_dimension)
        top.addWidget(self._dither)
        top.addWidget(self._rematch_button)
        top.addWidget(self._rematch_blocks_button)
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
        # panel is in PLAIN mode: it draws the uploaded file itself and ignores
        # the selection entirely, so it stays a true reference instead of a
        # washed-out copy of the preview.
        self._original_view = PictureView()
        self._original_view.setPlain(True)

        self._list = ColourList()
        self._list.colourSelected.connect(self._on_colour_selected)
        self._list.colourDeselected.connect(self._on_colour_deselected)
        self._list.colourActivated.connect(self._on_colour_activated)
        self._list.selectionChanged.connect(self._update_actions)

        # 「删除 / 合并 / 更换颜色 / 撤销 / 恢复」 sit in ONE row right under the
        # colour list, in that order: the first three all act on the selection
        # in the list, and the last two act on whatever those three did.
        #
        # None of them carries a hover tooltip any more: the black pop-up they
        # raised covered the very picture the user was working on.  What each
        # button does is said in the bottom line the moment it matters (a delete
        # announces that those pixels are left blank, and so on).
        self._delete_button = QPushButton("删除")
        self._delete_button.clicked.connect(self._on_delete_colours)

        self._merge_button = QPushButton("合并")
        self._merge_button.clicked.connect(self._on_merge_colours)

        self._replace_button = QPushButton("更换颜色")
        # Deliberately NOT accented, exactly like 删除 / 合并 / 撤销 / 恢复 next to
        # it: these five are all plain actions on the selection, and the only
        # blue button on this page is 生成 3MF, which is the one that finishes
        # the job.  A blue 更换颜色 read as "this is the important one".
        self._replace_button.clicked.connect(self._on_replace)

        self._multi_button = QPushButton("多选")
        self._multi_button.setCheckable(True)
        self._multi_button.toggled.connect(self._on_multi_toggled)

        self._undo_button = QPushButton("撤销")
        self._undo_button.clicked.connect(self._undo)

        self._redo_button = QPushButton("恢复")
        self._redo_button.clicked.connect(self._redo)

        actions = QHBoxLayout()
        actions.setSpacing(6)
        actions.addWidget(self._multi_button)
        actions.addWidget(self._delete_button)
        actions.addWidget(self._merge_button)
        actions.addWidget(self._replace_button)
        actions.addStretch(1)
        actions.addWidget(self._undo_button)
        actions.addWidget(self._redo_button)

        # The shortcuts are on the page, so Ctrl+Z works whichever of the two
        # panels has the focus.
        self._undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self._undo_shortcut.activated.connect(self._undo)
        self._redo_shortcut = QShortcut(QKeySequence("Ctrl+Y"), self)
        self._redo_shortcut.activated.connect(self._redo)
        # Ctrl+S 保存到现有工程（没有就问存哪儿），Ctrl+Shift+S 另存为一份。
        self._save_shortcut = QShortcut(QKeySequence.StandardKey.Save, self)
        self._save_shortcut.activated.connect(lambda: self._on_save_project())
        self._save_as_shortcut = QShortcut(QKeySequence("Ctrl+Shift+S"), self)
        self._save_as_shortcut.activated.connect(lambda: self._on_save_project(ask=True))

        list_box = QWidget()
        list_layout = QVBoxLayout(list_box)
        list_layout.setContentsMargins(0, 0, 0, 0)
        list_layout.setSpacing(6)
        list_layout.addWidget(self._list, 1)
        list_layout.addLayout(actions)

        self._detail = ColourDetail()
        # 「选中的颜色」 used to be a panel of its own, permanently showing whichever
        # colour was last clicked.  The colour list above is where colours get
        # picked, so the recipe and the ΔE00 of one colour became a dialog the user
        # asks for (双击一行) instead of a third of the column that sits empty.
        self._detail_dialog: QDialog | None = None

        # 「耗材方案」: a match is not one answer, it is a few, and the user is the
        # one who knows whether a different set of rolls suits this print.  Each
        # row carries its own numbers — 平均 / 最差 ΔE00 and how many rolls — so the
        # comparison happens on screen instead of inside the program's head, which
        # is exactly what the old single-answer 「重新匹配」 hid.
        self._plans: list[BudgetOption] = []
        self._plan_key = ""
        self._plan_group = QButtonGroup(self)
        self._plan_group.setExclusive(True)
        self._plan_group.idClicked.connect(self._on_plan_clicked)

        # 筛选: 「必须有这几卷」 and 「这几卷不要用」 are facts about the printer (what is
        # already in the AMS, what ran out), not about the picture, so they are set
        # once and every match from then on obeys them.
        self._require: frozenset[str] = frozenset()
        self._avoid: frozenset[str] = frozenset()
        self._filter_button = QPushButton("筛选耗材")
        self._filter_button.clicked.connect(self._on_filter)
        self._filter_hint = QLabel("")
        self._filter_hint.setProperty("role", "hint")
        self._filter_hint.setWordWrap(True)
        self._update_filter_hint()

        self._plan_box = QWidget()
        self._plan_layout = QVBoxLayout(self._plan_box)
        self._plan_layout.setContentsMargins(0, 0, 0, 0)
        self._plan_layout.setSpacing(6)
        filter_row = QHBoxLayout()
        filter_row.setSpacing(6)
        filter_row.addWidget(self._filter_button)
        filter_row.addWidget(self._filter_hint, 1)
        self._plan_layout.addLayout(filter_row)

        # The 方案 rows are rebuilt on every match, so they live in a widget of
        # their own; the filter line, the hint and the roll list around them are
        # reused as they are.
        self._plan_rows = QWidget()
        self._plan_rows_layout = QVBoxLayout(self._plan_rows)
        self._plan_rows_layout.setContentsMargins(0, 0, 0, 0)
        self._plan_rows_layout.setSpacing(2)
        self._plan_layout.addWidget(self._plan_rows)

        self._plan_hint = QLabel("")
        self._plan_hint.setProperty("role", "hint")
        self._plan_hint.setWordWrap(True)
        self._plan_layout.addWidget(self._plan_hint)

        # 「用到了哪些耗材」 is not a panel of its own any more: it answers the same
        # question the 方案 rows disagree about — 用哪几卷 — so it sits under them,
        # for the 方案 that is picked.  It no longer scrolls on its own: the whole
        # 耗材方案 panel does (see below), because seven 方案 plus their rolls do
        # not fit in the panel anyway.
        self._usage = QLabel("")
        self._usage.setProperty("role", "hint")
        self._usage.setTextFormat(Qt.TextFormat.RichText)
        self._usage.setWordWrap(True)
        self._plan_layout.addWidget(self._usage)
        self._refresh_plans()

        # m01885 item 3: 「打印图片需要的颜色的窗口要能自由向上拖拽上滑」.  A panel
        # whose height is its content cannot be shrunk — with seven 方案 rows its
        # minimum was most of the column, so the splitter handle would not move —
        # so the whole panel scrolls and its vertical size policy ignores the
        # content's minimum.  That is what lets the handle go all the way up (and
        # 耗材方案 all the way away) while both panels stay usable.
        self._plan_scroll = _scrollable(self._plan_box)
        self._plan_scroll.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Ignored
        )
        self._plan_scroll.setMinimumHeight(48)

        right = QSplitter(Qt.Orientation.Vertical)
        self._right_splitter = right
        # 耗材方案 comes first: it decides what the panel under it says, and it now
        # answers the printer's question (which rolls to load) as well.
        right.addWidget(self._wrap(self._plan_scroll, "耗材方案"))
        right.addWidget(self._wrap(list_box, "打印图片需要的颜色"))
        right.setStretchFactor(0, 2)
        right.setStretchFactor(1, 3)
        # 耗材方案 may be collapsed away entirely so the colour list gets the whole
        # column; the colour list never collapses, so it cannot be lost.
        right.setCollapsible(0, True)
        right.setCollapsible(1, False)
        right.setSizes([430, 350])

        pictures = QSplitter(Qt.Orientation.Horizontal)
        # 原图 and 预览 both zoom and pan with the same gesture; the hint is a
        # permanent line under the toolbar now, not a tooltip on the canvas
        # (which covered the picture the user was looking at).
        original_area = _scrollable(self._original_view)
        preview_area = _scrollable(self._view)
        pictures.addWidget(self._wrap(original_area, "原图"))
        pictures.addWidget(self._wrap(preview_area, "预览"))
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

        # The progress bar lives beside the status line it is described by.  It
        # is hidden until a match has been running for BUSY_DELAY_MS, and the
        # step message goes into the status label rather than into the bar, so
        # there is one text surface and nothing jumps when the bar appears.
        self._progress = QProgressBar()
        self._progress.setRange(0, PROGRESS_STEPS)
        self._progress.setTextVisible(False)
        self._progress.setFixedHeight(10)
        self._progress.setVisible(False)
        self._busy_timer = QTimer(self)
        self._busy_timer.setSingleShot(True)
        self._busy_timer.setInterval(BUSY_DELAY_MS)
        self._busy_timer.timeout.connect(self._show_progress)

        # 使用说明不在页面上：1.6.14 把它搬进「帮助」窗口（m01885 第 6 条），
        # 从这里读一份给帮助窗口和自动检查用。
        self._help = PAGE_HELP

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addLayout(top)
        layout.addWidget(splitter, 1)
        layout.addWidget(self._progress)
        layout.addWidget(self._status)
        layout.addLayout(bottom)

    def help_text(self) -> str:
        """这一页怎么用，「帮助」窗口里的头一节（m01885 第 6 条）。"""
        return self._help

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
            self._rematch_blocks_button,
            self._filter_button,
            self._save_project_button,
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
        self._triple_palette = None
        self._triple_head = None
        if self._image is None:
            return
        if not self.isVisible():
            # The 颜色配方 tab is showing, so nobody can see the picture.  A
            # re-match against 66,461 candidates costs ~0.7 s and would be spent
            # on a result that is not on screen; remember that it is owed and
            # do it when the tab comes back.
            self._stale = True
            return
        self._on_rematch()

    def setTriples(self, triples, library: FilamentLibrary | None = None) -> None:
        """Hand over the three-filament catalogue once it exists.

        ``None`` means 「not ready yet」 (an unfinished catalogue has no ``packed``
        array), and the page then matches against 耗材本色 + 两色混色 only.  The
        old palette is dropped either way, because a three-filament colour is
        only valid for the spool set it was computed from.
        """
        self._triples = triples
        self._triple_palette = None
        self._triple_head = None
        if library is not None:
            self._library = library

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
        # A freshly imported picture is not the project that was open before it,
        # so Ctrl+S has to ask where this one goes.
        self._project_path = None
        self._dirty = False
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

    # -- 工程：存下来，下次接着改 -----------------------------------------
    def _on_open_project(self) -> None:
        """Open a ``.bpproj`` and put the page back the way it was saved."""
        if self._dirty and not self._confirm_discard("打开另一个工程"):
            return
        start = str(self._project_path or paths.exports_dir())
        name, _ = QFileDialog.getOpenFileName(self, "打开工程", start, PROJECT_FILTER)
        if not name:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self._status.setText("正在打开工程…")
        QApplication.processEvents()
        try:
            state = load_project(Path(name), library=self._library)
        except ProjectError as error:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "打开工程失败", str(error))
            return
        except Exception as error:  # pragma: no cover - defensive
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "打开工程失败", str(error))
            return
        QApplication.restoreOverrideCursor()
        self._adopt_project(state, Path(name))

    def _adopt_project(self, state: ProjectState, path: Path) -> None:
        """Make ``state`` the page's picture, match and edit state."""
        self._image = state.image.convert("RGBA")
        self._image_name = state.image_name or "混色底板"
        self._source_qimage = _to_qimage(self._image)
        self._base = state.base
        self._deleted = set(state.deleted)
        self._merged = dict(state.merged)
        self._replaced = dict(state.replaced)
        self._undo_stack = []
        self._redo_stack = []
        self._plate = None
        self._project_path = path
        # The controls go back to the settings the file was matched with, so
        # 「重新匹配」 reproduces the picture the user saved rather than quietly
        # re-running it against whatever this window happened to be set to.
        settings = state.base.settings
        self._colour_count.setValue(int(settings.max_colours))
        self._max_dimension.setValue(int(settings.max_dimension))
        self._dither.setChecked(bool(settings.dither))
        self._spool_cap.setValue(int(settings.max_spools))
        self._width_spin.setValue(float(state.plate.target_width_mm))
        self._base_spin.setValue(float(state.plate.base_thickness_mm))
        self._colour_spin.setValue(float(state.plate.colour_thickness_mm))
        # The 方案 list belongs to a match, and 「打开工程」 deliberately does not run
        # one: the colour table in the file is what the user saved, edits and all,
        # and re-matching could answer something slightly different.  So the panel
        # shows the single row this file came with — named for the 方案 it was
        # saved under — and 「重新匹配」 is what fills in the other choices.
        self._plans = [
            BudgetOption(
                key=state.plan,
                title=plan_caption(state.plan)[0],
                note=plan_caption(state.plan)[1],
                result=state.base,
            )
        ]
        self._plan_key = state.plan
        self._refresh_plans()
        self._stale = False
        self._apply_edits()
        if 0 <= int(state.plate.base_index) < self._base_combo.count():
            self._base_combo.setCurrentIndex(int(state.plate.base_index))
        self._set_controls_enabled(True)
        edits = []
        if self._deleted:
            edits.append(f"删除 {len(self._deleted)} 种")
        if self._merged:
            edits.append(f"合并 {len(self._merged)} 种")
        if self._replaced:
            edits.append(f"更换 {len(self._replaced)} 种")
        tail = f"；里面记着{'、'.join(edits)}颜色" if edits else ""
        self._status.setText(
            f"已打开工程 {path.name}"
            + (f"（保存于 {state.saved_at}）" if state.saved_at else "")
            + f"：{self._image_name} {state.image.width}×{state.image.height} 像素，"
            f"现在打 {len(self._result.palette) if self._result else 0} 种颜色{tail}。"
            "接着改就行，Ctrl+S 保存。"
        )
        self._dirty = False

    def _on_save_project(self, ask: bool = False) -> None:
        """Write the project file — the original picture plus every edit so far.

        ``ask`` (Ctrl+Shift+S) always opens the dialog; otherwise the file the
        project came from is overwritten, which is what Ctrl+S means everywhere
        else, and the dialog appears only when there is no such file yet.
        """
        if self._base is None or self._result is None or self._image is None:
            self._status.setText("先导入并匹配一张图片，再保存工程。")
            return
        target = None if ask else self._project_path
        if target is None:
            suggested = f"{self._image_name}{PROJECT_SUFFIX}"
            # 存在「下载」里，和导出的模型一样：存进程序自己的数据目录等于让人找
            # 不到——工程文件是要拿去接着改、发给别人的东西。
            start = str(
                (self._project_path.with_name(suggested))
                if self._project_path
                else (paths.exports_dir() / suggested)
            )
            name, _ = QFileDialog.getSaveFileName(self, "保存工程", start, PROJECT_FILTER)
            if not name:
                return
            target = Path(name)
        engine = self._catalog.engine.id if self._catalog is not None else DEFAULT_ENGINE
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self._status.setText("正在保存工程…")
        QApplication.processEvents()
        try:
            written = save_project(
                target,
                image=self._image,
                image_name=self._image_name,
                base=self._base,
                live=self._result,
                deleted=self._deleted,
                merged=self._merged,
                replaced=self._replaced,
                engine=engine,
                plate=self._plate_settings(),
                library=self._library,
                version=__version__,
                preview=render_preview(self._result),
                plan=self._plan_key,
            )
        except Exception as error:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "保存工程失败", str(error))
            return
        QApplication.restoreOverrideCursor()
        self._project_path = written
        self._dirty = False
        spools = len(spool_usage(self._result, self._library))
        self._status.setText(
            f"已保存工程 {written.name}（{written.parent}）："
            f"原图 + 预览 + {len(self._result.palette)} 种颜色 + {spools} 卷耗材。"
            "下次「打开工程」就回到这一步。"
        )

    def _confirm_discard(self, action: str) -> bool:
        """Ask before edits that are not in the project file yet are lost."""
        box = QMessageBox(self)
        box.setWindowTitle("还有没保存的改动")
        box.setIcon(QMessageBox.Icon.Warning)
        box.setText(f"这张图改过、还没存进工程。要直接{action}吗？")
        box.setInformativeText("没保存的删除 / 合并 / 更换颜色都会丢掉。想留着就先 Ctrl+S。")
        keep = box.addButton("先保存", QMessageBox.ButtonRole.AcceptRole)
        drop = box.addButton("不保存", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(keep)
        box.exec()
        clicked = box.clickedButton()
        if clicked is keep:
            self._on_save_project()
            return not self._dirty
        return clicked is drop

    def _settings(self) -> MatchSettings:
        return MatchSettings(
            max_colours=int(self._colour_count.value()),
            max_dimension=int(self._max_dimension.value()),
            dither=bool(self._dither.isChecked()),
            max_spools=int(self._spool_cap.value()),
        ).clamped()

    def _candidate_palette(self) -> list[PaletteEntry]:
        """The full 全部颜色 list, built once per library.

        At 41 spools this is 66,461 ``PaletteEntry`` objects and ~0.5 s to
        build.  It is needed both by the match and by the 「更换颜色」 menu, and
        that menu is opened once per colour the user swaps — so rebuilding it
        every time was pure waste.  ``setLibrary`` drops the cache.

        Mixes are always in it.  There used to be a 「混色也算候选」 box; a spool
        library without its 81-ratio mixes is not a smaller answer, it is half
        the palette, so that choice was never a real one.
        """
        if self._candidates is None:
            self._candidates = build_palette(self._library, self._catalog, include_mixes=True)
        return self._candidates

    def _match_palette(self) -> Sequence[PaletteEntry]:
        """What the matcher picks from: 耗材本色 + 两色 + 三色混色.

        Three-filament colours are handed in by the main window once the table
        behind them is finished, and they always take part — the box that used
        to gate them is gone.  Four million colours cannot be built as objects,
        so the triple catalogue contributes a lazy tail: only the few the
        picture actually keeps ever become one.  The head still wins its ties,
        which is why the tail skips colours the head already offers.

        The cache is keyed on the head *list* we handed over, not on
        ``pool.head``: ``TriplePalette`` copies what it is given, so comparing
        against its own copy would rebuild the tail — a four-million-row
        membership test — on every single match.
        """
        head = self._candidate_palette()
        if self._triples is None:
            return head
        if self._triple_palette is None or self._triple_head is not head:
            self._triple_palette = TriplePalette(head, self._triples, self._library)
            self._triple_head = head
        return self._triple_palette

    def _begin_match(self) -> None:
        """Arm the progress bar; it appears only if the match is slow enough."""
        self._progress.setValue(0)
        self._busy_timer.start()
        self._set_controls_enabled(False)

    def _end_match(self) -> None:
        """Put the bar away.  The controls come back with the new result."""
        self._busy_timer.stop()
        self._progress.setVisible(False)
        self._progress.setValue(0)

    def _show_progress(self) -> None:
        self._progress.setVisible(True)

    def _report_match(self, message: str, fraction: float) -> None:
        """One step of :func:`match_image`, drawn while the match is running.

        ``processEvents`` is what makes the bar move at all — the match is a
        plain synchronous call — and it excludes user input so a click cannot
        start a second match inside the first one, which is the only way this
        can go wrong.
        """
        self._progress.setValue(int(max(0.0, min(1.0, fraction)) * PROGRESS_STEPS))
        self._status.setText(f"正在匹配：{message}…")
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

    def _on_rematch(self) -> None:
        if self._image is None:
            return
        if self._library is None or len(self._library) == 0:
            self._status.setText("请先在「颜色配方」页添加耗材，再匹配图片。")
            return
        palette = self._match_palette()
        if not palette:
            self._status.setText("调色板是空的，请先添加耗材。")
            return
        # The rows on screen describe the match that is about to be replaced:
        # clearing them first is what keeps 「耗材方案」 from showing a choice that
        # is no longer the one being made.
        self._plans = []
        self._refresh_plans("正在重新匹配，马上列出几套耗材搭配。")
        self._begin_match()
        try:
            options = match_options(
                self._image,
                palette,
                self._settings(),
                require=self._require,
                avoid=self._avoid,
                progress=self._report_match,
            )
        except Exception as error:  # pragma: no cover - defensive
            QMessageBox.warning(self, "匹配失败", str(error))
            # The controls were switched off for the match, and this path has no
            # new result to switch them back on with.
            self._set_controls_enabled(self._result is not None)
            return
        finally:
            self._end_match()
        if not options:
            QMessageBox.warning(
                self,
                "匹配失败",
                "按现在的筛选条件没有找到能用的耗材搭配，先放宽筛选或者提高耗材上限。"
                if (self._require or self._avoid)
                else "这一版没有找到能用的耗材搭配。",
            )
            self._set_controls_enabled(self._result is not None)
            return
        # The first 方案 is 「总体最接近」, the answer a single-answer matcher would
        # have given, so 「重新匹配」 still lands where it always did and the other
        # rows are the alternatives the user can now see instead of never being
        # told about.
        self._adopt_options(options)

    def _on_rematch_blocks(self) -> None:
        """「按当前色块重新匹配」: keep the colours on screen, re-choose the rolls.

        Not the same button as 重新匹配.  That one reads the picture again and
        starts from the colours it finds; this one takes the blocks as settled —
        a deleted colour stays deleted, a merge prints the colour it was merged
        into, a hand-picked colour stays picked — and only asks 「哪几卷、怎么混」
        again.  See :func:`app.core.image_matching.rematch_options`.
        """
        if self._result is None or self._library is None:
            return
        palette = self._match_palette()
        if not palette:
            self._status.setText("调色板是空的，请先添加耗材。")
            return
        # 手改过的颜色是用户定下来的，重新匹配不能悄悄换回去。`pinned` 按当前色块
        # 的下标给出它们；编辑簿记用的是基准下标，`_live_map` 负责翻过来。
        pinned = {
            index: self._replaced[source]
            for index, source in enumerate(self._live_map)
            if source in self._replaced
        }
        self._begin_match()
        try:
            options = rematch_options(
                self._result,
                palette,
                self._settings(),
                pinned=pinned,
                require=self._require,
                avoid=self._avoid,
                progress=self._report_match,
            )
        except Exception as error:  # pragma: no cover - defensive
            QMessageBox.warning(self, "匹配失败", str(error))
            return
        finally:
            self._end_match()
        if not options:
            QMessageBox.warning(
                self, "匹配失败", "按现在的色块和筛选，没有找到能用的耗材搭配。"
            )
            return
        self._adopt_options(options)

    def _adopt_options(self, options: list[BudgetOption]) -> None:
        """Make ``options[0]`` the match on screen, with the rest listed under it.

        Every edit so far was made against the previous 方案 and the rolls it spent,
        so the edit history starts over.  A hand-picked colour is not lost — it is
        carried in as one of the blocks (「按当前色块重新匹配」 pins it) — but
        undoing across that boundary would undo a colour that is still on screen.
        """
        self._plans = options
        self._plan_key = options[0].key
        self._base = options[0].result
        self._deleted = set()
        self._merged = {}
        self._replaced = {}
        self._undo_stack = []
        self._redo_stack = []
        # A new match is a new picture: whatever the project file holds is now
        # one step behind.
        self._dirty = True
        self._plate = None
        self._list.setLibrary(self._library)
        self._refresh_plans()
        # m01885 item 7: 新的一次匹配从「什么都没选」开始。上一次的高亮指的是上一份
        # 结果里的色块，留着它等于替用户选了一种颜色；编辑（删除 / 合并 / 更换颜色 /
        # 撤销）才继续跟着用户点过的那一块走。`_show_colour(-1)` 把预览的压暗和描边、
        # 列表的选中行、右侧详情一次清干净（m00830 item 2）。
        self._show_colour(-1)
        self._apply_edits()
        self._set_controls_enabled(True)
        self._announce_edits()

    # -- 耗材方案：同一张图，几套搭配，用户自己挑 ---------------------------
    def _refresh_plans(self, placeholder: str = "") -> None:
        """Draw one radio row per 方案 and the line that explains the picked one.

        Rebuilt rather than updated: there are a handful of rows at most and every
        number on them changes with the match, so clearing and refilling is
        shorter than any diff.  ``placeholder`` replaces the rows with one sentence
        (used while a match is running, when the old rows are already wrong).
        """
        while self._plan_rows_layout.count():
            item = self._plan_rows_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        for button in self._plan_group.buttons():
            self._plan_group.removeButton(button)
        if not self._plans:
            self._plan_hint.setText(
                placeholder or "匹配一张图片后，这里会列出几套耗材搭配，自己挑一套。"
            )
            self._plan_hint.setVisible(True)
            return
        for index, option in enumerate(self._plans):
            average, worst = option.fidelity
            row = QWidget()
            line = QHBoxLayout(row)
            line.setContentsMargins(0, 0, 0, 0)
            line.setSpacing(6)
            button = QRadioButton(option.title)
            button.setChecked(option.key == self._plan_key)
            button.setToolTip(option.note)
            self._plan_group.addButton(button, index)
            numbers = QLabel(
                f"平均 ΔE00 {average:.2f} · 最差 {worst:.2f} · 差块 {option.defects} · "
                f"用 {len(option.spools)} 卷"
            )
            numbers.setProperty("role", "hint")
            numbers.setToolTip(
                "「差块」= 图片里有几种颜色打出来偏差超过 ΔE00 5（按图片自己的颜色数算），"
                "越少越不容易一眼看出偏差。"
            )
            line.addWidget(button)
            line.addWidget(numbers)
            line.addWidget(self._plan_swatches(option))
            line.addStretch(1)
            self._plan_rows_layout.addWidget(row)
        # 每一行按什么标准挑的、差块是什么，都在帮助窗口里（m01885 第 8 条）：
        # 面板上留着的那一句只讲现在是什么状态。
        self._plan_hint.clear()
        self._plan_hint.setVisible(False)

    def _plan_swatches(self, option: BudgetOption) -> QWidget:
        """One 方案 的卷，按它们自己的颜色画出来（m01885 第 1 条）。

        The rows used to be told apart by「（备选 N）」, which names a position
        instead of the thing the user is choosing.  Showing the colours means a
        方案 is named by what it prints, and the row and the 用到的耗材 list under
        it agree on which rolls they are talking about.
        """
        box = QWidget()
        line = QHBoxLayout(box)
        line.setContentsMargins(0, 0, 0, 0)
        line.setSpacing(3)
        shares = {
            use.filament.id: use for use in spool_usage(option.result, self._library)
        }
        for filament_id in option.spools:
            filament = self._library.get(filament_id) if self._library is not None else None
            chip = QLabel()
            chip.setFixedSize(14, 14)
            colour = filament.color_hex if filament is not None else "#000000"
            chip.setStyleSheet(
                f"background-color: {colour};"
                f" border: 1px solid {theme.BORDER}; border-radius: 3px;"
            )
            use = shares.get(filament_id)
            name = filament.display_name if filament is not None else filament_id
            chip.setToolTip(f"{name} {use.share:.1%}" if use is not None else name)
            line.addWidget(chip)
        return box

    # -- 筛选：哪几卷必须用、哪几卷不要用 ---------------------------------
    def _ordered_ids(self, ids: frozenset[str]) -> list[str]:
        """``ids`` in library order, so the line reads like the 耗材库 page."""
        if self._library is None:
            return sorted(ids)
        return [filament.id for filament in self._library if filament.id in ids]

    def _spool_name(self, filament_id: str) -> str:
        if self._library is None:
            return filament_id
        filament = self._library.get(filament_id)
        return filament.display_name if filament is not None else filament_id

    def _update_filter_hint(self) -> None:
        """One line saying what the filter demands (「没有筛选」 when it demands nothing)."""
        parts = []
        if self._require:
            parts.append(
                "必须有：" + "、".join(self._spool_name(fid) for fid in self._ordered_ids(self._require))
            )
        if self._avoid:
            parts.append(
                "不要用：" + "、".join(self._spool_name(fid) for fid in self._ordered_ids(self._avoid))
            )
        self._filter_hint.setText("；".join(parts) + "。" if parts else "每一卷都可以用。")

    def _on_filter(self) -> None:
        """Ask which rolls may be spent, then match again with that demand in hand."""
        if self._image is None or self._library is None:
            return
        dialog = SpoolFilterDialog(
            list(self._library.visible),
            self._require,
            self._avoid,
            cap=self._spool_cap.value(),
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        require = frozenset(dialog.required())
        avoid = frozenset(dialog.forbidden())
        if require == self._require and avoid == self._avoid:
            return
        if not self._confirm_filter():
            return
        self._require = require
        self._avoid = avoid
        self._update_filter_hint()
        # The filter is a rule about the rolls, so it invalidates the match that is
        # on screen: the rows above it were chosen without knowing it.
        self._on_rematch()

    def _confirm_filter(self) -> bool:
        """Warn before a new filter throws the edits away (it re-matches)."""
        if not (self._deleted or self._merged or self._replaced):
            return True
        ask = QMessageBox(self)
        ask.setWindowTitle("换筛选要重新匹配")
        ask.setIcon(QMessageBox.Icon.Warning)
        ask.setText("换个筛选会把这张图重新匹配一遍，现在这些改动会丢。")
        ask.setInformativeText("删除 / 合并 / 更换过的颜色只对当前这一套搭配有效。")
        change = ask.addButton("换筛选", QMessageBox.ButtonRole.AcceptRole)
        ask.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        ask.setDefaultButton(change)
        ask.exec()
        return ask.clickedButton() is change


    def _selected_plan(self) -> BudgetOption | None:
        for option in self._plans:
            if option.key == self._plan_key:
                return option
        return None

    def _check_plan(self, key: str) -> None:
        """Put the radios back in step with ``self._plan_key``.  No rebuild."""
        for index, option in enumerate(self._plans):
            button = self._plan_group.button(index)
            if button is not None:
                button.setChecked(option.key == key)

    def _on_plan_clicked(self, index: int) -> None:
        if not 0 <= index < len(self._plans):
            return
        option = self._plans[index]
        if option.key == self._plan_key:
            return
        if not self._confirm_plan_switch(option):
            self._check_plan(self._plan_key)
            return
        self._apply_plan(option)

    def _apply_plan(self, option: BudgetOption) -> None:
        """Make ``option`` the picture, dropping the edits made for the old one."""
        self._plan_key = option.key
        self._base = option.result
        self._deleted = set()
        self._merged = {}
        self._replaced = {}
        self._undo_stack = []
        self._redo_stack = []
        # A different 方案 is a different picture: the project file needs saving
        # again, and the mesh has to be rebuilt from the new colours.
        self._dirty = True
        self._plate = None
        self._list.setLibrary(self._library)
        # m00830 item 2: a different 方案 is a different picture.  Keeping the old
        # selection drew the dim wash and the outline of a colour in a picture the
        # user had just replaced; a switch now starts from 「什么都没选」.
        self._show_colour(-1)
        self._apply_edits()
        self._set_controls_enabled(True)
        self._announce_edits()

    def _confirm_plan_switch(self, option: BudgetOption) -> bool:
        """Ask before switching, because a 方案 switch repaints the picture.

        The per-colour edits are indices into the *old* palette, so there is no
        honest way to carry them across; saying so is the whole job here.
        """
        if not (self._deleted or self._merged or self._replaced):
            return True
        box = QMessageBox(self)
        box.setWindowTitle("换一套耗材要重新分配颜色")
        box.setIcon(QMessageBox.Icon.Warning)
        box.setText(f"换成「{option.title}」会重画这张图，现在这些改动会丢。")
        box.setInformativeText("删除 / 合并 / 更换过的颜色只对当前这一套搭配有效。")
        switch = box.addButton("换一套", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(switch)
        box.exec()
        return box.clickedButton() is switch

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
        """Show one colour everywhere it appears: picture, list and detail.

        The colour that was on screen goes on ``_preview_history`` first, which
        is what 「再点一下取消选中」 steps back through.
        """
        index = int(index)
        if index != self._shown and self._shown >= 0:
            self._preview_history.append(self._shown)
        self._show_colour(index)

    def _show_colour(self, index: int) -> None:
        """Put ``index`` on screen without touching the preview history."""
        index = int(index)
        self._shown = index
        self._view.setSelected(index)
        self._original_view.setSelected(index)
        if index < 0:
            # 「取消选中」: no colour, so no block, no detail and no row highlight.
            self._list.clear_selection()
            self._detail.clear()
            return
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

    def _on_colour_deselected(self, index: int) -> None:
        """Clicking the colour that is already on screen puts it away.

        The preview steps back to whatever it showed before that colour, which
        is the whole point of keeping the history: the user got here to compare,
        and losing the comparison would be worse than losing the colour.
        """
        if self._shown < 0 or int(index) != self._shown:
            return
        previous = self._preview_history.pop() if self._preview_history else -1
        self._show_colour(previous)

    def _on_region_clicked(self, index: int) -> None:
        """A click in 「预览」 means the same thing as a click on that row.

        With 「多选」 on it toggles the region in and out of the selection, so the
        user can pick colours by pointing at the picture instead of hunting for
        the row in the list.
        """
        if self._multi_button.isChecked():
            self._list.toggle(index)
        self._select(index)

    def _on_colour_activated(self, index: int) -> None:
        """双击一行 = 「这个颜色到底是什么」: 图片里的颜色、匹配到的颜色、配方.

        The details used to sit in a panel of their own, permanently showing
        whichever row was last clicked.  They are something the user asks about
        (double-click) rather than something that costs a third of the column,
        and the dialog is not modal so the next row can be asked about straight
        away — :meth:`_show_colour` keeps it in step with the selection.
        """
        self._select(index)
        if self._result is None or not 0 <= index < len(self._result.palette):
            return
        entry = self._result.palette[index]
        dialog = self._detail_window()
        dialog.setWindowTitle(f"颜色详情 · {entry.color_hex.upper()} · {entry.short_label}")
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _detail_window(self) -> QDialog:
        """The 颜色详情 dialog, built on first use.

        It is a shell around ``self._detail`` (the same :class:`ColourDetail` the
        smoke test and the self-test read), so 「更换颜色」 stays on the action row
        where it always was instead of moving into a pop-up.
        """
        if self._detail_dialog is None:
            dialog = QDialog(self)
            dialog.setObjectName("colourDetailDialog")
            dialog.setMinimumWidth(380)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll.setWidget(self._detail)
            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
            buttons.rejected.connect(dialog.close)
            close = buttons.button(QDialogButtonBox.StandardButton.Close)
            if close is not None:
                close.setText("关闭")
            box = QVBoxLayout(dialog)
            box.setContentsMargins(12, 12, 12, 12)
            box.setSpacing(8)
            box.addWidget(scroll, 1)
            box.addWidget(buttons)
            self._detail_dialog = dialog
        return self._detail_dialog

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
        # Every user edit comes through here, which is what makes this the one
        # place that has to remember 「不是工程文件里那一版了」.
        self._dirty = True

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
        through the edit, so 「更换颜色」 and 撤销 both leave the user where they
        were instead of jumping back to the top of the list.
        """
        anchor = keep
        if anchor is None:
            index = self._list.current()
            if 0 <= index < len(self._live_map):
                anchor = self._live_map[index]
        self._result = self._compose()
        self._plate = None
        # The rows were rebuilt, so the colours the preview history pointed at no
        # longer mean what they meant; 「取消选中」 starts over from here.
        self._shown = -1
        self._preview_history.clear()
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
        # m01885 item 7: a fresh match selects nothing.  The old code fell back to
        # row 0 here, which drew a highlight on a colour the user never picked and
        # made 「取消选中」 impossible to reach from a clean match.
        if 0 <= target < len(result.palette):
            self._select(target)
        else:
            self._show_colour(-1)
        self._update_actions()
        self._announce_edits()

    def _on_multi_toggled(self, on: bool) -> None:
        """「多选」：click to add a colour, click again to drop it — no Ctrl."""
        self._list.setMultiSelect(on)
        # Accent while it is ON, so "clicks add up" is never a guess.
        self._multi_button.setProperty("accent", "true" if on else "false")
        style = self._multi_button.style()
        style.unpolish(self._multi_button)
        style.polish(self._multi_button)

    def _update_actions(self) -> None:
        """ 删除 needs one colour, 合并 needs two, and the history drives the rest."""
        picked = self._list.selected_indices() if self._result is not None else []
        live = bool(self._result is not None and len(self._result.palette))
        self._delete_button.setEnabled(live and len(picked) >= 1)
        self._merge_button.setEnabled(live and len(picked) >= 2)
        self._replace_button.setEnabled(live and len(picked) >= 1)
        self._undo_button.setEnabled(bool(self._undo_stack))
        self._redo_button.setEnabled(bool(self._redo_stack))

    def _announce_edits(self) -> None:
        self._update_usage()
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
            # When the match found more than one way to spend the rolls, say so:
            # the panel is on the right, and a user who does not scroll that far
            # would never learn the choice exists.
            choices = (
                f"给出 {len(self._plans)} 套耗材搭配，右边「耗材方案」里自己挑一套；"
                if len(self._plans) > 1
                else ""
            )
            self._status.setText(
                "匹配完成：{0}×{1} 像素，{2} 种颜色。{3}点击右侧颜色看图片高光，"
                "点击图片看它属于哪种颜色；点「更换颜色」可以手动换掉不合适的颜色，"
                "多选颜色后可以「删除」（留空不打印）或「合并」。".format(
                    result.width, result.height, len(result.palette), choices
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

    def _update_usage(self) -> None:
        """Show which rolls this version of the picture spends, largest first.

        The percentages are shares of material, not of pixels: 「白 10% + 金 90%」
        uses nine times as much gold as white, so this is the line that tells
        the user which spools to actually load.
        """
        uses = spool_usage(self._result, self._library)
        # Back to the top: after an edit the biggest roll is the answer the user
        # came for, and it must not stay scrolled out of sight.
        bar = self._plan_scroll.verticalScrollBar()
        bar.setValue(bar.minimum())
        if not uses:
            self._usage.setText(
                "匹配完成后，这里列出这一版用到了哪几卷耗材，以及各卷占的色料比例。"
            )
            return
        # 一排（m00830 第 3 条）：以前是「共 3 卷」加一行一卷，三卷就吃掉三行；现在
        # 卷名和占比横着排，宽了才换行，而且每一项自己不断行（nowrap）。
        parts = []
        for use in uses:
            chip = (
                f'<span style="background-color:{use.color_hex};">&nbsp;&nbsp;&nbsp;</span> '
                if use.color_hex
                else ""
            )
            missing = "" if use.filament is not None else "（耗材库里已经没有这一卷）"
            parts.append(
                f'<span style="white-space:nowrap;">{chip}{use.display_name}'
                f"{missing} <b>{use.share:.1%}</b></span>"
            )
        self._usage.setText(f"共 {len(uses)} 卷：" + " · ".join(parts))

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
        self._dirty = True
        self._apply_edits()
        self._status.setText("已撤销上一步。Ctrl+Y 可以再做回来。")

    def _redo(self) -> None:
        if not self._redo_stack:
            return
        self._undo_stack.append(self._snapshot())
        self._restore(self._redo_stack.pop())
        self._dirty = True
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
        written = None
        try:
            written = self._write_model(plate, kind, target)
        except OSError as error:
            # The dialog opens where the user asked for it, which is 下载.  If
            # the shell or a policy refuses that folder, losing the model is not
            # an option: write it next to our own data and name the file that
            # really exists.  (1.6.5 probed Downloads up front instead, so the
            # dialog silently opened in a hidden folder.)
            for alternative in self._alternative_targets(target):
                try:
                    written = self._write_model(plate, kind, alternative)
                except OSError:
                    continue
                QApplication.restoreOverrideCursor()
                QMessageBox.information(
                    self,
                    "已存到别的位置",
                    f"{target.parent} 写不进去：{error.strerror or error}\n\n"
                    f"模型已经存到：\n{written}",
                )
                break
            if written is None:
                QApplication.restoreOverrideCursor()
                QMessageBox.warning(self, "保存失败", str(error))
                return
        except Exception as error:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "保存失败", str(error))
            return
        QApplication.restoreOverrideCursor()
        self._status.setText(self._describe(plate, written))
        self._announce(plate, written)

    def _write_model(self, plate: PlateModel, kind: str, target: Path) -> Path:
        """Write *plate* to *target* in the requested format and return the path."""
        if kind == "3mf":
            # The filament list travels inside the project so Bambu Studio opens
            # it with *our* colours instead of whatever is in the AMS.
            filaments = (
                slots_from_palette(self._result.palette, self._library)
                if self._result is not None
                else None
            )
            return write_3mf(plate, target, object_name=self._image_name, filaments=filaments)
        return write_obj(plate, target)

    @staticmethod
    def _alternative_targets(target: Path) -> list[Path]:
        """Where to retry a save the chosen folder refused, best first."""
        options: list[Path] = []
        homes: list[Path] = []
        try:
            homes.append(paths.data_dir() / "exports")
        except OSError:  # pragma: no cover - defensive
            pass
        try:
            homes.append(Path.home())
        except (OSError, RuntimeError):  # pragma: no cover - defensive
            pass
        for directory in homes:
            candidate = directory / target.name
            if candidate.parent == target.parent or candidate in options:
                continue
            try:
                candidate.parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                continue
            options.append(candidate)
        return options

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

"""A virtualised grid that can paint tens of thousands of mix swatches.

A 40-spool library produces 63 180 mixes, so the grid never creates one widget
per colour: it keeps a row layout (plus an extra gap between parent-pair groups)
and paints only the cells currently inside the viewport.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QAbstractScrollArea, QToolTip

from . import theme

#: Size of one grid cell in device-independent pixels.
CELL = 34
#: Swatch inset inside a cell.
INSET = 4.0
#: Extra vertical space inserted between two parent-pair groups.
GROUP_GAP = 10


@dataclass(frozen=True)
class SpoolCell:
    """A raw filament shown beside the mixes when 「全部颜色」 is on.

    It borrows ``MixRecipe``'s read-only shape — ``key``, ``color_hex``, ``rgb``,
    ``pair_index``, ``percent_a``/``percent_b`` — so ``MixGrid`` can paint it
    without caring which kind of cell it is. ``pair_index`` is ``-1``, which is
    never a real parent-pair number, so every spool lands in one leading group
    of its own and the mixes follow unchanged.
    """

    filament: object
    color_hex: str
    rgb: tuple[int, int, int]
    label: str
    key: str
    pair_index: int = -1
    percent_a: int = 0
    percent_b: int = 0

    @property
    def is_spool(self) -> bool:
        return True


def spool_cell(filament) -> SpoolCell:
    """Wrap a ``Filament`` as a grid cell."""
    return SpoolCell(
        filament=filament,
        color_hex=filament.color_hex,
        rgb=filament.rgb,
        label=filament.display_name,
        key=f"spool|{filament.id}",
    )


class MixGrid(QAbstractScrollArea):
    """Every predicted mix as one small swatch, in the catalogue's order."""

    recipeSelected = Signal(object)
    recipeActivated = Signal(object)
    selectionCleared = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._recipes: list = []
        self._rows: list[list[int]] = []
        self._row_y: list[float] = []
        self._content_h = 0.0
        self._columns = 1
        self._selected = -1
        self._grouped = True
        self._hover = -1

        self.setFrameShape(QAbstractScrollArea.Shape.NoFrame)
        self.setBackgroundRole(self.backgroundRole())
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor(theme.PANEL))
        self.setPalette(palette)

    # -- content -----------------------------------------------------------------
    def setRecipes(self, recipes, grouped: bool = True) -> None:
        self._recipes = list(recipes)
        self._grouped = grouped
        self._selected = -1
        self._hover = -1
        self._relayout()
        self.viewport().update()

    def recipes(self) -> list:
        return list(self._recipes)

    def selectedRecipe(self):
        if 0 <= self._selected < len(self._recipes):
            return self._recipes[self._selected]
        return None

    def clearSelection(self) -> None:
        """Drop the selection — clicking empty space in the grid does this."""
        if self._selected == -1:
            return
        self._selected = -1
        self.viewport().update()
        self.selectionCleared.emit()

    def setSelectedIndex(self, index: int, scroll: bool = True) -> None:
        if not self._recipes:
            self._selected = -1
            return
        index = max(0, min(index, len(self._recipes) - 1))
        self._selected = index
        if scroll:
            self._scrollToIndex(index)
        self.viewport().update()

    def selectRecipe(self, recipe) -> None:
        """Select by identity, then scroll so it is visible."""
        if recipe is None:
            return
        for index, candidate in enumerate(self._recipes):
            if candidate is recipe or candidate.key == recipe.key:
                self.setSelectedIndex(index)
                self.recipeSelected.emit(candidate)
                return

    def indexOfKey(self, key: str) -> int:
        for index, candidate in enumerate(self._recipes):
            if candidate.key == key:
                return index
        return -1

    # -- layout ------------------------------------------------------------------
    def _relayout(self) -> None:
        width = max(CELL, self.viewport().width())
        columns = max(1, width // CELL)
        self._columns = columns

        rows: list[list[int]] = []
        row_y: list[float] = []
        current: list[int] = []
        y = 0.0
        previous_pair: int | None = None

        for index, recipe in enumerate(self._recipes):
            if self._grouped and recipe.pair_index != previous_pair:
                if current:
                    rows.append(current)
                    row_y.append(y)
                    y += CELL
                    current = []
                if rows:
                    y += GROUP_GAP
                previous_pair = recipe.pair_index
            current.append(index)
            if len(current) == columns:
                rows.append(current)
                row_y.append(y)
                y += CELL
                current = []

        if current:
            rows.append(current)
            row_y.append(y)
            y += CELL

        self._rows = rows
        self._row_y = row_y
        self._content_h = y
        self._sync_scrollbars()

    def _sync_scrollbars(self) -> None:
        vbar = self.verticalScrollBar()
        vbar.setRange(0, max(0, int(self._content_h) - self.viewport().height()))
        vbar.setSingleStep(CELL)
        vbar.setPageStep(max(CELL, self.viewport().height()))

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._relayout()

    # -- painting ----------------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self.viewport())
        painter.fillRect(event.rect(), QColor(theme.PANEL))
        if not self._recipes:
            painter.setPen(QColor(theme.TEXT_FAINT))
            painter.drawText(
                self.viewport().rect(),
                Qt.AlignmentFlag.AlignCenter,
                "还没有混色。请先添加至少两种耗材。",
            )
            painter.end()
            return

        offset = self.verticalScrollBar().value()
        top = offset
        bottom = offset + self.viewport().height()
        first = max(0, bisect.bisect_right(self._row_y, top) - 1)

        for row_index in range(first, len(self._rows)):
            y = self._row_y[row_index]
            if y > bottom:
                break
            if y + CELL < top:
                continue
            for column, item_index in enumerate(self._rows[row_index]):
                recipe = self._recipes[item_index]
                left = column * CELL
                cell_top = y - offset
                rect = QRectF(left + INSET, cell_top + INSET,
                              CELL - 2 * INSET, CELL - 2 * INSET)
                # Nothing is ever dimmed. A selected mix is marked by a ring in its
                # OWN colour (drawn in the cell's margin, around the swatch), so the
                # neighbouring mixes keep showing the colour they really mix to.
                painter.fillRect(rect, QColor(*recipe.rgb))
                painter.setPen(QPen(_outline(recipe.rgb), 1))
                painter.drawRect(rect.adjusted(0.5, 0.5, -0.5, -0.5))
                if item_index == self._selected:
                    painter.setPen(QPen(_selection_colour(recipe.rgb), 2))
                    painter.drawRect(QRectF(left + 2.0, cell_top + 2.0, CELL - 4.0, CELL - 4.0))
                elif item_index == self._hover:
                    painter.setPen(QPen(QColor(theme.BORDER_STRONG), 1))
                    painter.drawRect(QRectF(left + 1.5, cell_top + 1.5, CELL - 3.0, CELL - 3.0))
        painter.end()

    # -- interaction -------------------------------------------------------------
    def _indexAt(self, point) -> int:
        if not self._rows:
            return -1
        y = point.y() + self.verticalScrollBar().value()
        row_index = bisect.bisect_right(self._row_y, y) - 1
        if row_index < 0 or row_index >= len(self._rows):
            return -1
        row = self._rows[row_index]
        column = int(point.x()) // CELL
        if 0 <= column < len(row):
            return row[column]
        return -1

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        index = self._indexAt(event.position().toPoint())
        if index < 0:
            # Blank space (a group gap, the area past the last cell, the empty
            # strip under the last row) clears the selection.
            self.clearSelection()
            return
        self._selected = index
        self.viewport().update()
        self.recipeSelected.emit(self._recipes[index])

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        index = self._indexAt(event.position().toPoint())
        if index >= 0:
            self.recipeActivated.emit(self._recipes[index])

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        index = self._indexAt(event.position().toPoint())
        if index != self._hover:
            self._hover = index
            self.viewport().update()
            if index >= 0:
                recipe = self._recipes[index]
                QToolTip.showText(event.globalPosition().toPoint(), self._tooltip(recipe))
            else:
                QToolTip.hideText()

    def leaveEvent(self, event) -> None:  # noqa: N802
        if self._hover != -1:
            self._hover = -1
            self.viewport().update()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if not self._recipes:
            return super().keyPressEvent(event)
        step = {Qt.Key.Key_Left: -1, Qt.Key.Key_Right: 1,
                Qt.Key.Key_Up: -self._columns, Qt.Key.Key_Down: self._columns}
        key = event.key()
        if key == Qt.Key.Key_Escape:
            self.clearSelection()
            return
        if key in step:
            start = self._selected if self._selected >= 0 else 0
            self.setSelectedIndex(start + step[key])
            self.recipeSelected.emit(self.selectedRecipe())
            return
        if key == Qt.Key.Key_Home:
            self.setSelectedIndex(0)
            self.recipeSelected.emit(self.selectedRecipe())
            return
        if key == Qt.Key.Key_End:
            self.setSelectedIndex(len(self._recipes) - 1)
            self.recipeSelected.emit(self.selectedRecipe())
            return
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            recipe = self.selectedRecipe()
            if recipe is not None:
                self.recipeActivated.emit(recipe)
            return
        if key == Qt.Key.Key_PageDown:
            self.setSelectedIndex(self._selected + self._columns * 10)
            self.recipeSelected.emit(self.selectedRecipe())
            return
        if key == Qt.Key.Key_PageUp:
            self.setSelectedIndex(self._selected - self._columns * 10)
            self.recipeSelected.emit(self.selectedRecipe())
            return
        super().keyPressEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802
        super().wheelEvent(event)
        self._hover = -1

    # -- helpers -----------------------------------------------------------------
    def _tooltip(self, recipe) -> str:
        if getattr(recipe, "pair_index", 0) < 0:
            return (
                f"{recipe.label}\n"
                f"{recipe.color_hex}\n"
                "这是耗材丝本身的颜色（不是混色）"
            )
        return (
            f"{recipe.color_hex}\n"
            f"{recipe.percent_a}% + {recipe.percent_b}%\n"
            "单击查看是哪两种耗材丝"
        )

    def _scrollToIndex(self, index: int) -> None:
        for row_index, row in enumerate(self._rows):
            if index in row:
                y = self._row_y[row_index]
                viewport_h = self.viewport().height()
                value = self.verticalScrollBar().value()
                if y < value:
                    self.verticalScrollBar().setValue(int(y))
                elif y + CELL > value + viewport_h:
                    self.verticalScrollBar().setValue(int(y + CELL - viewport_h))
                return

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(520, 360)


def _outline(rgb) -> QColor:
    """A near-invisible separator: dark on light swatches, light on dark ones."""
    r, g, b = rgb
    luminance = theme.relative_luminance(r, g, b)
    return QColor(0, 0, 0, 46) if luminance > 0.18 else QColor(255, 255, 255, 46)


def _selection_colour(rgb) -> QColor:
    """The selected swatch's own colour, at reduced opacity, for its ring.

    The ring sits in the cell margin, i.e. on the white grid, so a pale
    selection (a white filament, a near-white mix) would be invisible. Those
    colours are first pulled down their own scale — the hue is kept, only the
    level changes — and only then given the alpha. A saturated or dark colour is
    already legible and is used as-is.
    """
    r, g, b = (int(channel) for channel in rgb)
    level = theme.relative_luminance(r, g, b)
    if level > 0.28:
        factor = 0.30 + 0.42 * (1.0 - min(1.0, (level - 0.28) / 0.72))
        r, g, b = (int(channel * factor) for channel in (r, g, b))
    colour = QColor(r, g, b)
    colour.setAlpha(190)
    return colour

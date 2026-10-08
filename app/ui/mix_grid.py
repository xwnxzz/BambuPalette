"""A virtualised grid that can paint tens of thousands of mix swatches.

A 40-spool library produces 63 180 mixes, so the grid never creates one widget
per colour: it keeps a row layout (plus an extra gap between parent-pair groups)
and paints only the cells currently inside the viewport.
"""

from __future__ import annotations

import bisect

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


class MixGrid(QAbstractScrollArea):
    """Every predicted mix as one small swatch, in the catalogue's order."""

    recipeSelected = Signal(object)
    recipeActivated = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._recipes: list = []
        self._rows: list[list[int]] = []
        self._row_y: list[float] = []
        self._content_h = 0.0
        self._columns = 1
        self._selected = -1
        self._grouped = True
        self._focus_pair: int | None = None
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
        self._focus_pair = None
        self._relayout()
        self.viewport().update()

    def recipes(self) -> list:
        return list(self._recipes)

    def setFocusPair(self, pair_index: int | None) -> None:
        """Dim every cell that does not belong to ``pair_index``."""
        if self._focus_pair == pair_index:
            return
        self._focus_pair = pair_index
        self.viewport().update()

    def selectedRecipe(self):
        if 0 <= self._selected < len(self._recipes):
            return self._recipes[self._selected]
        return None

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
                rect = QRectF(column * CELL + INSET, y - offset + INSET,
                              CELL - 2 * INSET, CELL - 2 * INSET)
                dim = self._focus_pair is not None and recipe.pair_index != self._focus_pair
                painter.setOpacity(0.22 if dim else 1.0)
                painter.fillRect(rect, QColor(*recipe.rgb))
                if not dim:
                    painter.setPen(QPen(_outline(recipe.rgb), 1))
                    painter.drawRect(rect.adjusted(0.5, 0.5, -0.5, -0.5))
                if item_index == self._selected:
                    painter.setOpacity(1.0)
                    painter.setPen(QPen(QColor(theme.SELECTION_BORDER), 2))
                    painter.drawRect(rect.adjusted(1.0, 1.0, -1.0, -1.0))
                elif item_index == self._hover:
                    painter.setOpacity(1.0)
                    painter.setPen(QPen(QColor(theme.BORDER_STRONG), 1))
                    painter.drawRect(rect.adjusted(-1.0, -1.0, 1.0, 1.0))
        painter.setOpacity(1.0)
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
        index = self._indexAt(event.position().toPoint())
        if index < 0:
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

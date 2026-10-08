"""Reusable colour swatch widgets and painting helpers."""

from __future__ import annotations

from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QColorDialog, QPushButton, QSizePolicy, QWidget

from . import theme


def qcolor(value) -> QColor:
    """Accept ``#RRGGBB``, ``(r, g, b)`` or ``QColor`` and return a :class:`QColor`."""
    if isinstance(value, QColor):
        return QColor(value)
    if isinstance(value, (tuple, list)):
        return QColor(int(value[0]), int(value[1]), int(value[2]))
    return QColor(str(value))


def text_colour(value) -> QColor:
    colour = qcolor(value)
    return QColor(theme.contrasting_text(colour.red(), colour.green(), colour.blue()))


def border_colour(value) -> QColor:
    """Light swatches get a grey outline, dark ones a near-black one."""
    colour = qcolor(value)
    luminance = theme.relative_luminance(colour.red(), colour.green(), colour.blue())
    return QColor(theme.SWATCH_BORDER_LIGHT if luminance > 0.2 else theme.SWATCH_BORDER_DARK)


def draw_swatch(painter: QPainter, rect: QRectF, value, radius: float = 3.0) -> None:
    """Fill ``rect`` with the colour, plus an adaptive 1px outline."""
    colour = qcolor(value)
    path = QPainterPath()
    if radius > 0:
        path.addRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), radius, radius)
    else:
        path.addRect(rect.adjusted(0.5, 0.5, -0.5, -0.5))
    painter.save()
    painter.setPen(Qt.PenStyle.NoPen)
    painter.fillPath(path, colour)
    painter.setPen(QPen(border_colour(colour), 1))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawPath(path)
    painter.restore()


def swatch_pixmap(value, width: int = 18, height: int | None = None, radius: float = 3.0) -> QPixmap:
    """A standalone swatch image, handy for list icons."""
    height = width if height is None else height
    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    draw_swatch(painter, QRectF(0, 0, width, height), value, radius)
    painter.end()
    return pixmap


def swatch_icon(value, width: int = 18, height: int | None = None) -> QIcon:
    return QIcon(swatch_pixmap(value, width, height))


class SwatchLabel(QWidget):
    """A fixed-size preview card, optionally captioned with its hex value."""

    def __init__(self, value="#FFFFFF", size: int = 84, caption: bool = True, parent=None):
        super().__init__(parent)
        self._value = qcolor(value)
        self._caption = caption
        self._size = size
        self.setMinimumSize(size, size + (18 if caption else 0))
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def setValue(self, value) -> None:  # noqa: N802 - Qt naming
        colour = qcolor(value)
        if colour != self._value:
            self._value = colour
            self.update()

    def value(self) -> QColor:
        return QColor(self._value)

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(self._size, self._size + (18 if self._caption else 0))

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(0, 0, self._size, self._size)
        draw_swatch(painter, rect, self._value, radius=6.0)
        if self._caption:
            painter.setPen(QColor(theme.TEXT_MUTED))
            font = painter.font()
            font.setPointSizeF(max(7.5, font.pointSizeF() - 1.5))
            painter.setFont(font)
            painter.drawText(
                QRectF(0, self._size + 1, self._size, 17),
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
                self._value.name().upper(),
            )
        painter.end()


class ColorButton(QPushButton):
    """A button showing the current colour that opens the system colour picker.

    ``value=None`` starts the button **unset**: it paints an empty dashed slot
    instead of a colour, so a picker that is meant to be filled in by the user
    never shows a made-up default.  :meth:`isSet` tells the caller whether a
    colour has actually been chosen; :meth:`hex` returns ``""`` until then.
    """

    colorChanged = Signal(str)

    def __init__(self, value=None, title: str = "选择颜色", parent=None):
        super().__init__(parent)
        self._value = None if value is None or value == "" else qcolor(value)
        self._title = title
        self.setFixedSize(56, 30)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clicked.connect(self._choose)

    def _choose(self) -> None:
        start = self._value if self._value is not None else QColor("#FFFFFF")
        chosen = QColorDialog.getColor(start, self, self._title)
        if chosen.isValid():
            self.setValue(chosen.name().upper())

    def setValue(self, value) -> None:  # noqa: N802
        colour = qcolor(value)
        if colour == self._value:
            return
        self._value = colour
        self.update()
        self.colorChanged.emit(colour.name().upper())

    def clear(self) -> None:
        """Return the button to its unset state."""
        if self._value is None:
            return
        self._value = None
        self.update()
        self.colorChanged.emit("")

    def isSet(self) -> bool:  # noqa: N802
        return self._value is not None

    def value(self) -> QColor:
        return QColor() if self._value is None else QColor(self._value)

    def hex(self) -> str:
        return "" if self._value is None else self._value.name().upper()

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(5, 4, self.width() - 10, self.height() - 8)
        if self._value is None:
            painter.setPen(QPen(QColor(theme.BORDER_STRONG), 1.0, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), 4.0, 4.0)
        else:
            draw_swatch(painter, rect, self._value, 4.0)
        painter.end()


class GradientBar(QWidget):
    """The 比例 gradient strip, labelled 90% … 10% at either end."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stops: list[QColor] = []
        self._height = 22
        self.setMinimumHeight(self._height + 16)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def setColors(self, colors) -> None:  # noqa: N802
        self._stops = [qcolor(c) for c in colors]
        self.update()

    def setGradient(self, left, right, steps: int = 32) -> None:  # noqa: N802
        """Convenience: interpolate ``left``→``right`` in sRGB for a quick bar."""
        start, end = qcolor(left), qcolor(right)
        self._stops = []
        for index in range(steps):
            t = index / max(1, steps - 1)
            self._stops.append(
                QColor(
                    round(start.red() + (end.red() - start.red()) * t),
                    round(start.green() + (end.green() - start.green()) * t),
                    round(start.blue() + (end.blue() - start.blue()) * t),
                )
            )
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        bar = QRectF(0, 0, self.width(), self._height)
        if not self._stops:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.fillRect(bar, QColor(theme.PANEL_ALT))
        else:
            span = max(1, len(self._stops) - 1)
            width = self.width()
            for index, colour in enumerate(self._stops):
                x = width * index / span
                painter.fillRect(QRectF(x, 0, width / span + 1.0, self._height), colour)
        painter.setPen(QPen(border_colour("#FFFFFF"), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(bar.adjusted(0.5, 0.5, -0.5, -0.5))

        painter.setPen(QColor(theme.TEXT_MUTED))
        font = painter.font()
        font.setPointSizeF(max(7.5, font.pointSizeF() - 1.5))
        painter.setFont(font)
        label_rect = QRectF(0, self._height + 1, self.width(), 14)
        painter.drawText(label_rect, Qt.AlignmentFlag.AlignLeft, "90%")
        painter.drawText(label_rect, Qt.AlignmentFlag.AlignRight, "10%")
        painter.end()

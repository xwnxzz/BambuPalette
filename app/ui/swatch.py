"""Reusable colour swatch widgets and painting helpers."""

from __future__ import annotations

from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

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


def is_unset(value) -> bool:
    """Is this slot still waiting for the user to pick a colour?

    ``None`` and ``""`` both mean 「not chosen yet」.  They are kept distinct from
    a real colour because a made-up default is a lie the user only discovers in
    Bambu Studio, after the print.
    """
    return value is None or value == "" or value == Qt.GlobalColor.transparent


def draw_swatch(painter: QPainter, rect: QRectF, value, radius: float = 3.0) -> None:
    """Fill ``rect`` with the colour, plus an adaptive 1px outline.

    An unset ``value`` paints an empty dashed slot instead of a black square, so
    「没有颜色」 is visibly different from 「黑色」.
    """
    path = QPainterPath()
    if radius > 0:
        path.addRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), radius, radius)
    else:
        path.addRect(rect.adjusted(0.5, 0.5, -0.5, -0.5))
    painter.save()
    if is_unset(value):
        painter.fillPath(path, QColor(theme.PANEL_ALT))
        painter.setPen(QPen(QColor(theme.BORDER_STRONG), 1, Qt.PenStyle.DashLine))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)
        painter.restore()
        return
    colour = qcolor(value)
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
    """A fixed-size preview card, optionally captioned with its hex value.

    ``value=None`` starts the card **empty**: it paints a dashed slot captioned
    「未选择」 rather than pretending to be white.
    """

    def __init__(self, value=None, size: int = 84, caption: bool = True, parent=None):
        super().__init__(parent)
        self._value = None if is_unset(value) else qcolor(value)
        self._caption = caption
        self._size = size
        self.setMinimumSize(size, size + (18 if caption else 0))
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def setValue(self, value) -> None:  # noqa: N802 - Qt naming
        colour = None if is_unset(value) else qcolor(value)
        if colour != self._value:
            self._value = colour
            self.update()

    def isSet(self) -> bool:  # noqa: N802
        return self._value is not None

    def hex(self) -> str:
        return "" if self._value is None else self._value.name().upper()

    def clear(self) -> None:
        self.setValue(None)

    def value(self) -> QColor:
        return QColor() if self._value is None else QColor(self._value)

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
                "未选择" if self._value is None else self._value.name().upper(),
            )
        painter.end()


class ColorField(QWidget):
    """Type a colour in directly: swatch + ``#RRGGBB`` + R / G / B spin boxes.

    This replaces the system colour-picker dialog.  The user asked for it by
    name: the pop-up palette they were shown (「选择耗材颜色」) is a full HSV
    wheel with a screen dropper, which is the wrong tool for "my filament prints
    as 0, 0, 255" — they want to *type the numbers*.

    ``value=None`` starts unset (an empty dashed swatch, blank hex, zeroed spin
    boxes) so nothing pretends a colour has been chosen.  Every edit path funnels
    through :meth:`_set`, and a re-entrancy guard keeps the hex box and the three
    spin boxes from bouncing signals off each other.
    """

    colorChanged = Signal(str)

    def __init__(self, value=None, parent=None, *, spin_width: int = 78, stacked: bool = False):
        super().__init__(parent)
        self._value: QColor | None = None
        self._busy = False

        self._swatch = SwatchLabel(None, size=26, caption=False, parent=self)
        self._swatch.setFixedSize(30, 26)

        self._hex = QLineEdit(self)
        self._hex.setPlaceholderText("#RRGGBB")
        self._hex.setMaxLength(7)
        self._hex.setMinimumWidth(76)
        self._hex.setMaximumWidth(110)
        self._hex.setProperty("role", "mono")

        self._spins: list[QSpinBox] = []
        self._spin_labels: list[QLabel] = []
        # 78 is MEASURED, not guessed.  Under QWindows11Style the up/down buttons
        # take 21 px and the frame another ~25 px, so a 62 px box leaves a 16 px
        # edit field — less than two digits, which is why the panel used to show
        # "R 2" for 200.  74 px (28 px of text) still clipped the third digit;
        # 78 px gives a 32 px edit field and "255" needs 21 px, so everything fits.
        for name in ("R", "G", "B"):
            label = QLabel(name, self)
            label.setProperty("role", "hint")
            spin = QSpinBox(self)
            spin.setRange(0, 255)
            spin.setMinimumWidth(spin_width)
            spin.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            spin.valueChanged.connect(self._on_spin)
            self._spin_labels.append(label)
            self._spins.append(spin)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        # ``stacked`` exists because the same widget has to survive inside a
        # narrow side panel: one row of [swatch][hex][R][G][B] needs ~380 px and
        # the 我的耗材 panel is only ~300 px of usable width, which squeezed the
        # spin boxes down to a single visible digit.
        first = QHBoxLayout()
        first.setContentsMargins(0, 0, 0, 0)
        first.setSpacing(6)
        first.addWidget(self._swatch)
        first.addWidget(self._hex, 1)
        if not stacked:
            for label, spin in zip(self._spin_labels, self._spins):
                first.addWidget(label)
                first.addWidget(spin)
        first.addStretch(0 if stacked else 1)
        outer.addLayout(first)

        if stacked:
            # Tight spacing on the numbers row: 5 gaps at 6 px each cost 30 px
            # the three spin boxes need more than the panel has.
            second = QHBoxLayout()
            second.setContentsMargins(0, 0, 0, 0)
            second.setSpacing(2)
            for label, spin in zip(self._spin_labels, self._spins):
                second.addWidget(label)
                second.addWidget(spin, 1)
            second.addStretch(0)
            outer.addLayout(second)

        self._hex.editingFinished.connect(self._on_hex_edited)
        self.setValue(value)

    # -- reading -----------------------------------------------------------------
    def isSet(self) -> bool:  # noqa: N802
        return self._value is not None

    def hex(self) -> str:
        return "" if self._value is None else self._value.name().upper()

    def value(self) -> QColor:
        return QColor() if self._value is None else QColor(self._value)

    def rgb(self) -> tuple[int, int, int]:
        if self._value is None:
            return (0, 0, 0)
        return (self._value.red(), self._value.green(), self._value.blue())

    # -- writing -----------------------------------------------------------------
    def setValue(self, value) -> None:  # noqa: N802
        self._set(value, announce=True)

    def clear(self) -> None:
        self._set(None, announce=True)

    def _set(self, value, *, announce: bool) -> None:
        """The single funnel every edit goes through."""
        colour = None if is_unset(value) else qcolor(value)
        if colour is not None and not colour.isValid():
            return
        self._busy = True
        try:
            self._value = colour
            if colour is None:
                self._hex.clear()
                for spin in self._spins:
                    spin.setValue(0)
            else:
                self._hex.setText(colour.name().upper())
                for spin, channel in zip(self._spins, (colour.red(), colour.green(), colour.blue())):
                    spin.setValue(channel)
            self._swatch.setValue(None if colour is None else colour)
        finally:
            self._busy = False
        if announce:
            self.colorChanged.emit(self.hex())

    # -- edits -------------------------------------------------------------------
    def _on_hex_edited(self) -> None:
        if self._busy:
            return
        text = self._hex.text().strip()
        if not text:
            self._set(None, announce=True)
            return
        try:
            normalised = _normalise_hex(text)
        except ValueError:
            # Put back whatever is actually set — possibly nothing at all.
            self._set(self._value, announce=False)
            self._hex.setText(self.hex())
            return
        self._set(normalised, announce=True)

    def _on_spin(self, *_) -> None:
        if self._busy:
            return
        r, g, b = (spin.value() for spin in self._spins)
        self._set(QColor(r, g, b), announce=True)


def _normalise_hex(text: str) -> str:
    """``#abc`` / ``abc`` / ``#AABBCC`` → ``#AABBCC``; anything else raises."""
    cleaned = text.strip().lstrip("#")
    if len(cleaned) == 3 and all(c in "0123456789abcdefABCDEF" for c in cleaned):
        cleaned = "".join(c * 2 for c in cleaned)
    if len(cleaned) != 6 or any(c not in "0123456789abcdefABCDEF" for c in cleaned):
        raise ValueError(f"not a hex colour: {text!r}")
    return "#" + cleaned.upper()


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

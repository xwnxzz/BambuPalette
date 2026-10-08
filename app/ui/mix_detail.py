"""The answer panel: *which two spools, and in what ratio?*"""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from . import theme
from .swatch import SwatchLabel

#: How many stops the pair bar samples from the active mixing engine.
BAR_STOPS = 129


class PairBar(QWidget):
    """The whole 0 %→100 % sweep of one pair, with the recipe marked.

    The left end is the pure first filament, the right end the pure second one;
    the white divider sits at the selected weight of the first filament.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stops: list[QColor] = []
        self._marker = 0.5
        self._height = 26
        self.setMinimumHeight(self._height)
        self.setSizePolicy(self.sizePolicy().horizontalPolicy(), self.sizePolicy().verticalPolicy())
        self.setFixedHeight(self._height)

    def setPair(self, hex_a: str, hex_b: str, percent_a: float, engine) -> None:
        weights = [(i / (BAR_STOPS - 1)) for i in range(BAR_STOPS)]
        self._stops = [QColor(engine.mix_hex(hex_a, hex_b, weight * 100.0)) for weight in weights]
        self._marker = max(0.0, min(1.0, percent_a / 100.0))
        self.update()

    def clear(self) -> None:
        self._stops = []
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        bar = QRectF(0, 0, self.width(), self._height)
        if not self._stops:
            painter.fillRect(bar, QColor(theme.PANEL_ALT))
        else:
            span = len(self._stops) - 1
            width = self.width()
            for index, colour in enumerate(self._stops):
                x = width * index / span
                painter.fillRect(QRectF(x, 0, width / span + 1.0, self._height), colour)
            painter.setPen(QPen(QColor(theme.BORDER_STRONG), 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(bar.adjusted(0.5, 0.5, -0.5, -0.5))
            x = self._marker * width
            painter.setPen(QPen(QColor("#FFFFFF"), 3))
            painter.drawLine(int(x), 0, int(x), self._height)
            painter.setPen(QPen(QColor(theme.TEXT), 1))
            painter.drawLine(int(x), 0, int(x), self._height)
        painter.end()


class MixDetail(QWidget):
    """Shows the two parent spools, their percentages and the steps to copy."""

    engineChanged = Signal()
    pairRequested = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._library = None
        self._engine = None
        self._recipe = None

        self.setMinimumWidth(320)
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor(theme.PANEL))
        self.setPalette(palette)

        title = QLabel("混色配方", self)
        title.setProperty("role", "sectionTitle")

        self._copy = QPushButton("复制配方", self)
        self._copy.setToolTip("把「两种耗材 + 比例 + 预测颜色」复制到剪贴板，方便在 Bambu Studio 里配色")
        self._copy.clicked.connect(self._on_copy)
        self._copy.setEnabled(False)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self._copy)

        self._preview = SwatchLabel("#FFFFFF", size=92, caption=False, parent=self)
        self._hex = QLabel("—", self)
        self._hex.setProperty("role", "mono")
        self._ratio = QLabel("—", self)
        self._ratio.setProperty("role", "sectionTitle")
        self._engine_label = QLabel("", self)
        self._engine_label.setProperty("role", "hint")
        self._engine_label.setWordWrap(True)

        summary = QVBoxLayout()
        summary.setContentsMargins(0, 0, 0, 0)
        summary.setSpacing(3)
        summary.addWidget(self._ratio)
        summary.addWidget(self._hex)
        summary.addWidget(self._engine_label)
        summary.addStretch(1)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(14)
        top.addWidget(self._preview, 0, Qt.AlignmentFlag.AlignTop)
        top.addLayout(summary, 1)

        self._rows: list[dict] = []
        rows_box = QVBoxLayout()
        rows_box.setContentsMargins(0, 0, 0, 0)
        rows_box.setSpacing(8)
        for slot in (1, 2):
            swatch = SwatchLabel("#FFFFFF", size=34, caption=False, parent=self)
            badge = QLabel(f"耗材丝{slot}", self)
            badge.setProperty("role", "hint")
            name = QLabel("—", self)
            name.setWordWrap(True)
            detail_label = QLabel("—", self)
            detail_label.setProperty("role", "hint")
            detail_label.setWordWrap(True)
            percent = QLabel("—", self)
            percent.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)

            text = QVBoxLayout()
            text.setContentsMargins(0, 0, 0, 0)
            text.setSpacing(0)
            text.addWidget(badge)
            text.addWidget(name)
            text.addWidget(detail_label)

            percent_box = QVBoxLayout()
            percent_box.setContentsMargins(0, 0, 0, 0)
            percent_box.setSpacing(0)
            percent_box.addWidget(percent)
            percent_box.addStretch(1)

            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(10)
            row.addWidget(swatch, 0, Qt.AlignmentFlag.AlignTop)
            row.addLayout(text, 1)
            row.addLayout(percent_box, 0)

            holder = QWidget(self)
            holder.setLayout(row)
            holder.setMinimumHeight(64)
            rows_box.addWidget(holder)
            self._rows.append(
                {
                    "frame": holder,
                    "swatch": swatch,
                    "name": name,
                    "detail": detail_label,
                    "percent": percent,
                }
            )

        self._bar = PairBar(self)
        self._bar.setToolTip("左端 = 第一种耗材 100%，右端 = 第二种耗材 100%；白线是当前比例")

        self._pair_button = QPushButton("只看这一对耗材的全部混色", self)
        self._pair_button.setToolTip("把中间网格缩小到这两种耗材的 81 个配比")
        self._pair_button.clicked.connect(self._on_pair)

        self._steps = QLabel("", self)
        self._steps.setProperty("role", "hint")
        self._steps.setWordWrap(True)
        self._steps.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self._empty = QLabel(
            "在左侧网格里点选任意一个混色，这里会显示它由哪两种耗材丝、以什么比例混成。",
            self,
        )
        self._empty.setProperty("role", "hint")
        self._empty.setWordWrap(True)

        separator = QFrame(self)
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet(f"color: {theme.BORDER};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(12)
        layout.addLayout(header)
        layout.addWidget(self._empty)
        layout.addWidget(separator)
        layout.addLayout(top)
        layout.addLayout(rows_box)
        layout.addWidget(self._bar)
        layout.addWidget(self._pair_button)
        layout.addWidget(self._steps)
        layout.addStretch(1)

        self._content = (separator, self._preview)
        self.clear()

    # -- context -----------------------------------------------------------------
    def setContext(self, library, engine) -> None:
        self._library = library
        self._engine = engine
        if self._recipe is not None:
            self.showRecipe(self._recipe)

    def recipe(self):
        return self._recipe

    # -- content -----------------------------------------------------------------
    def clear(self) -> None:
        self._recipe = None
        self._copy.setEnabled(False)
        self._empty.setVisible(True)
        for key in ("_hex", "_ratio", "_engine_label"):
            getattr(self, key).setVisible(False)
        self._preview.setVisible(False)
        self._bar.setVisible(False)
        self._pair_button.setVisible(False)
        self._steps.setVisible(False)
        for row in self._rows:
            row["frame"].setVisible(False)

    def showRecipe(self, recipe) -> None:
        if recipe is None or self._library is None or self._engine is None:
            self.clear()
            return
        self._recipe = recipe
        self._empty.setVisible(False)
        filament_a = self._library.get(recipe.a_id)
        filament_b = self._library.get(recipe.b_id)
        hex_a = filament_a.color_hex if filament_a else "?"
        hex_b = filament_b.color_hex if filament_b else "?"

        self._preview.setValue(recipe.color_hex)
        self._preview.setVisible(True)
        self._hex.setText(f"预测颜色 {recipe.color_hex}   RGB {recipe.rgb[0]}, {recipe.rgb[1]}, {recipe.rgb[2]}")
        self._hex.setVisible(True)
        self._ratio.setText(f"{recipe.percent_a}% + {recipe.percent_b}%")
        self._ratio.setVisible(True)
        self._engine_label.setText(f"混色模型：{self._engine.name}")
        self._engine_label.setVisible(True)

        for row, filament, percent, slot in (
            (self._rows[0], filament_a, recipe.percent_a, 1),
            (self._rows[1], filament_b, recipe.percent_b, 2),
        ):
            row["frame"].setVisible(True)
            if filament is None:
                row["swatch"].setValue("#FFFFFF")
                row["name"].setText("（耗材已删除）")
                row["detail"].setText("")
                row["percent"].setText(f"{percent}%")
                continue
            row["swatch"].setValue(filament.color_hex)
            detail = " · ".join(p for p in (filament.brand, filament.material_type) if p)
            row["name"].setText(filament.display_name)
            row["detail"].setText(detail if detail and detail != filament.display_name else "")
            row["percent"].setText(f"{percent}%\n{filament.color_hex}")
            row["frame"].setToolTip(
                f"{filament.display_name}\n{filament.color_hex}"
                + (f"\n备注：{filament.note}" if filament.note else "")
            )

        self._bar.setVisible(True)
        self._bar.setPair(hex_a, hex_b, recipe.percent_a, self._engine)
        self._pair_button.setVisible(True)

        self._steps.setVisible(True)
        self._steps.setText(
            "在 Bambu Studio 里这样复现：\n"
            f"  1. 打开「添加混色耗材」（Add Mixed Filament）\n"
            f"  2. 耗材丝1 选 {_short(filament_a)}\n"
            f"  3. 耗材丝2 选 {_short(filament_b)}\n"
            f"  4. 比例拖到 {recipe.percent_a}% : {recipe.percent_b}%\n"
            f"  5. 效果预览应显示 {recipe.color_hex}\n"
            "注意：两种耗材的「耗材种类」必须相同，Bambu Studio 才允许混色。"
        )
        self._copy.setEnabled(True)

    # -- actions -----------------------------------------------------------------
    def _on_pair(self) -> None:
        if self._recipe is not None:
            self.pairRequested.emit(self._recipe.a_id, self._recipe.b_id)

    def _on_copy(self) -> None:
        recipe = self._recipe
        if recipe is None or self._library is None:
            return
        filament_a = self._library.get(recipe.a_id)
        filament_b = self._library.get(recipe.b_id)
        text = (
            f"{_short(filament_a)} {recipe.percent_a}%  +  "
            f"{_short(filament_b)} {recipe.percent_b}%  →  {recipe.color_hex}"
        )
        QApplication.clipboard().setText(text)
        self._copy.setText("已复制")
        from PySide6.QtCore import QTimer

        QTimer.singleShot(1200, lambda: self._copy.setText("复制配方"))


def _short(filament) -> str:
    if filament is None:
        return "（耗材已删除）"
    return f"{filament.display_name} ({filament.color_hex})"

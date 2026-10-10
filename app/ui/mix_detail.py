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

#: How many recipes of one merged colour the detail panel is willing to draw.
#: Two colours rarely give more than a handful, but three colours can give the
#: same hex dozens of times and a list nobody can read helps nobody.
MAX_RECIPE_ROWS = 50


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
    #: Emitted when the grid is already narrowed to one pair and the user wants
    #: every mix back. The button below the ratio bar becomes the way out.
    showAllRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._library = None
        self._engine = None
        self._recipe = None
        self._filament = None
        self._colour = None
        self._pair_filtered = False

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
        for slot in (1, 2, 3):
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

        # 「颜色详情里有合成这个颜色的所有配方」: one distinct colour can be reached by
        # several spool pairs and several ratios, so the panel lists them all
        # rather than pretending the first one is the only one.
        self._formula_title = QLabel("", self)
        self._formula_title.setProperty("role", "sectionTitle")
        self._formula_title.setWordWrap(True)
        self._formula_note = QLabel("", self)
        self._formula_note.setProperty("role", "hint")
        self._formula_note.setWordWrap(True)
        self._formula_box = QWidget(self)
        self._formula_layout = QVBoxLayout(self._formula_box)
        self._formula_layout.setContentsMargins(0, 0, 0, 0)
        self._formula_layout.setSpacing(4)
        self._formula_rows: list[dict] = []

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
        layout.addWidget(self._formula_title)
        layout.addWidget(self._formula_note)
        layout.addWidget(self._formula_box)
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
        elif self._filament is not None:
            self.showFilament(self._library.get(self._filament.id) or self._filament)

    def recipe(self):
        return self._recipe

    def colour(self):
        """The merged colour being shown, or None when a plain recipe is shown."""
        return self._colour

    def filament(self):
        return self._filament

    def selectionKey(self) -> str:
        """Whatever is currently shown, as one comparable key."""
        if self._colour is not None:
            return self._colour.key
        if self._recipe is not None:
            return self._recipe.key
        if self._filament is not None:
            return f"spool|{self._filament.id}"
        return ""

    # -- content -----------------------------------------------------------------
    def clear(self) -> None:
        self._recipe = None
        self._filament = None
        self._colour = None
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
        self._clear_formulas()

    def _clear_formulas(self) -> None:
        self._formula_title.setVisible(False)
        self._formula_note.setVisible(False)
        self._formula_box.setVisible(False)
        for row in self._formula_rows:
            row["frame"].setVisible(False)

    def showColour(self, colour) -> None:
        """A merged colour: the usual panel for its first recipe, then all of them.

        The user asked for the detail panel to name every recipe behind a colour
        (「颜色详情里有合成这个颜色的所有配方」).  The preview, the pair bar and the
        Bambu Studio steps keep describing the FIRST recipe, because that is the
        one the panel is already built around; the list underneath is the full
        answer, and clicking a row there re-points the panel at that recipe.
        """
        if colour is None or self._library is None or self._engine is None:
            self.clear()
            return
        recipes = list(getattr(colour, "recipes", ()))
        if not recipes:
            # A spool's own colour is produced by no mixture at all.  There is no
            # recipe to explain, so show it the way a plain spool is shown — but
            # keep the COLOUR, so the grid can still restore the selection.
            spools = tuple(getattr(colour, "filaments", ()))
            if not spools:
                self.clear()
                return
            self.showFilament(spools[0])
            self._colour = colour
            return
        self.showRecipe(recipes[0])
        self._fill_formulas(colour, recipes)
        # showRecipe() clears this, so it is restored last on purpose: the panel
        # remembers the merged colour the user actually clicked.
        self._colour = colour

    def _fill_formulas(self, colour, recipes) -> None:
        total = len(recipes)
        self._formula_title.setText(f"合成 {colour.color_hex} 的所有配方（{total} 条）")
        self._formula_title.setVisible(True)

        shown = recipes[:MAX_RECIPE_ROWS]
        while len(self._formula_rows) < len(shown):
            self._formula_rows.append(self._make_formula_row())
        for row, recipe in zip(self._formula_rows, shown):
            row["frame"].setVisible(True)
            row["swatch"].setValue(recipe.color_hex)
            line = _recipe_line(self._library, recipe)
            row["text"].setText(line)
            row["key"].setText(recipe.key)
            row["frame"].setToolTip(f"{line}\n{recipe.color_hex}")
        for row in self._formula_rows[len(shown):]:
            row["frame"].setVisible(False)

        if total > MAX_RECIPE_ROWS:
            self._formula_note.setText(
                f"只列出前 {MAX_RECIPE_ROWS} 条，还有 {total - MAX_RECIPE_ROWS} 条没有显示。"
            )
            self._formula_note.setVisible(True)
        else:
            self._formula_note.setVisible(False)
        self._formula_box.setVisible(True)

    def _make_formula_row(self) -> dict:
        swatch = SwatchLabel("#FFFFFF", size=20, caption=False, parent=self._formula_box)
        text = QLabel("—", self._formula_box)
        text.setProperty("role", "hint")
        # Deliberately NOT word-wrapped: a wrapping label reports a tiny minimum
        # height, so a colour reachable by eighteen recipes would squeeze all
        # eighteen rows into a few pixels and print them on top of each other.
        # Without wrapping the column has a real minimum and the page scrolls.
        text.setWordWrap(False)
        key = QLabel("—", self._formula_box)
        key.setProperty("role", "mono")
        key.setVisible(False)

        line = QHBoxLayout()
        line.setContentsMargins(0, 0, 0, 0)
        line.setSpacing(8)
        line.addWidget(swatch, 0, Qt.AlignmentFlag.AlignTop)
        line.addWidget(text, 1)

        frame = QWidget(self._formula_box)
        frame.setLayout(line)
        self._formula_layout.addWidget(frame)
        return {"frame": frame, "swatch": swatch, "text": text, "key": key}

    def showRecipe(self, recipe) -> None:
        if recipe is None or self._library is None or self._engine is None:
            self.clear()
            return
        # A plain recipe has exactly one formula, so the merged-colour list must
        # not linger from a previous selection.
        self._clear_formulas()
        self._recipe = recipe
        self._colour = None
        self._filament = None
        self._empty.setVisible(False)
        # Two- and three-filament recipes both arrive here: read the parents off
        # the recipe instead of assuming a pair.
        parent_ids = tuple(getattr(recipe, "parent_ids", ()) or (recipe.a_id, recipe.b_id))
        percents = tuple(getattr(recipe, "percents", ()) or (recipe.percent_a, recipe.percent_b))
        parents = [self._library.get(filament_id) for filament_id in parent_ids]
        hexes = [filament.color_hex if filament else "?" for filament in parents]

        self._preview.setValue(recipe.color_hex)
        self._preview.setVisible(True)
        self._hex.setText(f"预测颜色 {recipe.color_hex}   RGB {recipe.rgb[0]}, {recipe.rgb[1]}, {recipe.rgb[2]}")
        self._hex.setVisible(True)
        self._ratio.setText(" + ".join(f"{percent}%" for percent in percents))
        self._ratio.setVisible(True)
        self._engine_label.setText(f"混色模型：{self._engine.name}")
        self._engine_label.setVisible(True)

        for index, row in enumerate(self._rows):
            if index >= len(parents):
                row["frame"].setVisible(False)
                continue
            filament = parents[index]
            percent = percents[index]
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

        # The 0→100 % bar only means something for a pair of spools.
        three = len(parents) > 2
        self._bar.setVisible(not three)
        if not three:
            self._bar.setPair(hexes[0], hexes[1], percents[0], self._engine)
        self._pair_button.setVisible(not three)

        steps = [
            "在 Bambu Studio 里这样复现：",
            "  1. 打开「添加混色耗材」（Add Mixed Filament）",
        ]
        for slot, filament in enumerate(parents, start=1):
            steps.append(f"  {slot + 1}. 耗材丝{slot} 选 {_short(filament)}")
        steps.append(f"  {len(parents) + 2}. 比例拖到 " + " : ".join(f"{p}%" for p in percents))
        steps.append(f"  {len(parents) + 3}. 效果预览应显示 {recipe.color_hex}")
        steps.append(
            "注意：所有耗材的「耗材种类」必须相同，Bambu Studio 才允许混色。"
            if three
            else "注意：两种耗材的「耗材种类」必须相同，Bambu Studio 才允许混色。"
        )
        self._steps.setVisible(True)
        self._steps.setText("\n".join(steps))
        self._copy.setEnabled(True)

    def showFilament(self, filament) -> None:
        """Show a raw spool colour — the extra cells 「全部颜色」 adds to the grid.

        A spool is not a mix, so the pair bar and the second parent row are
        hidden rather than filled with a fake 100% : 0% recipe. What is shown is
        exactly what the user needs when they pick this cell: the colour, the
        RGB, and the fact that using it means loading that one spool.
        """
        if filament is None:
            self.clear()
            return
        self._clear_formulas()
        self._recipe = None
        self._colour = None
        self._filament = filament
        self._empty.setVisible(False)

        self._preview.setValue(filament.color_hex)
        self._preview.setVisible(True)
        self._hex.setText(
            f"耗材本色 {filament.color_hex}   "
            f"RGB {filament.rgb[0]}, {filament.rgb[1]}, {filament.rgb[2]}"
        )
        self._hex.setVisible(True)
        self._ratio.setText("单色 · 100%")
        self._ratio.setVisible(True)
        self._engine_label.setText(
            "这是耗材丝本身的颜色，不是混色；「全部颜色」把每种耗材丝也当成一个可选项，"
            "所以两种耗材时会看到 81 个混色 + 2 个本色 = 83 个颜色。"
        )
        self._engine_label.setVisible(True)

        row = self._rows[0]
        row["frame"].setVisible(True)
        row["swatch"].setValue(filament.color_hex)
        detail = " · ".join(p for p in (filament.brand, filament.material_type) if p)
        row["name"].setText(filament.display_name)
        row["detail"].setText(detail if detail and detail != filament.display_name else "")
        row["percent"].setText(f"100%\n{filament.color_hex}")
        row["frame"].setToolTip(
            f"{filament.display_name}\n{filament.color_hex}"
            + (f"\n备注：{filament.note}" if filament.note else "")
        )
        self._rows[1]["frame"].setVisible(False)

        # No pair, so no ratio bar and no "只看这一对耗材" jump.
        self._bar.setVisible(False)
        self._pair_button.setVisible(False)

        self._steps.setVisible(True)
        self._steps.setText(
            "在 Bambu Studio 里这样用：\n"
            f"  1. 把 {_short(filament)} 装进 AMS\n"
            f"  2. 颜色选 {filament.color_hex}\n"
            "单色耗材不需要配置混色比例。"
        )
        self._copy.setEnabled(True)

    # -- actions -----------------------------------------------------------------
    def setPairFilter(self, active: bool) -> None:
        """Turn the pair button into the way back out once a filter is on."""
        active = bool(active)
        if active == self._pair_filtered:
            return
        self._pair_filtered = active
        self._pair_button.setText(
            "显示全部颜色" if active else "只看这一对耗材的全部混色"
        )
        self._pair_button.setToolTip(
            "取消筛选，回到全部颜色"
            if active
            else "把中间网格缩小到这两种耗材的 81 个配比"
        )
        self._pair_button.setProperty("accent", "true" if active else "false")
        self._pair_button.style().unpolish(self._pair_button)
        self._pair_button.style().polish(self._pair_button)

    def _on_pair(self) -> None:
        if self._pair_filtered:
            self.showAllRequested.emit()
            return
        if self._recipe is not None:
            self.pairRequested.emit(self._recipe.a_id, self._recipe.b_id)

    def _on_copy(self) -> None:
        recipe = self._recipe
        if recipe is None:
            if self._filament is not None:
                self._flash_copy(
                    f"{_short(self._filament)} 100%  →  {self._filament.color_hex}"
                )
            return
        if self._library is None:
            return
        filament_a = self._library.get(recipe.a_id)
        filament_b = self._library.get(recipe.b_id)
        self._flash_copy(
            f"{_short(filament_a)} {recipe.percent_a}%  +  "
            f"{_short(filament_b)} {recipe.percent_b}%  →  {recipe.color_hex}"
        )

    def _flash_copy(self, text: str) -> None:
        QApplication.clipboard().setText(text)
        self._copy.setText("已复制")
        from PySide6.QtCore import QTimer

        QTimer.singleShot(1200, lambda: self._copy.setText("复制配方"))


def _recipe_line(library, recipe) -> str:
    """「白 30%  +  黑 40%  +  灰 30%」——两色和三色配方都走这里。"""
    ids = tuple(getattr(recipe, "parent_ids", ()) or (recipe.a_id, recipe.b_id))
    percents = tuple(getattr(recipe, "percents", ()) or (recipe.percent_a, recipe.percent_b))
    # 三色配方再带 #RRGGBB 就比面板还宽，会被硬裁掉，所以只在两色时带色号；
    # 完整文字仍然放在 tooltip 里。
    with_hex = len(ids) <= 2
    parts = []
    for filament_id, percent in zip(ids, percents):
        filament = library.get(filament_id)
        if filament is None:
            name = "（耗材已删除）"
        else:
            name = _short(filament) if with_hex else filament.display_name
        parts.append(f"{name} {percent}%")
    return "  +  ".join(parts)


def _short(filament) -> str:
    if filament is None:
        return "（耗材已删除）"
    return f"{filament.display_name} ({filament.color_hex})"

"""The 「选中的颜色」 panel and its 更换颜色 picker.

The picture page matches every colour region of a picture to a spool or a mix.
This module answers the two questions the user then asks about one such match:

* which rgb did the picture have there, which rgb did we match it to, and what
  is that match made of (the recipe)?
* that match is not what I wanted — let me pick another one out of 「全部颜色」.

The picker offers the very same ordering controls as the 混色配方 grid, plus
「按跟图片目标颜色最相似排序」, so the user can either scan the familiar RGB order
or let the program put the closest candidates first.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core.image_matching import PaletteEntry, sort_palette
from ..core.mixes import SORT_CHOICES_WITH_SIMILARITY, SORT_SIMILARITY
from ..spectral import color as _color
from .swatch import swatch_pixmap

RECIPE_PREFIX = "配方"
SPOOL_PREFIX = "耗材本色"


def hex_of(rgb) -> str:
    return _color.rgb_to_hex((int(rgb[0]), int(rgb[1]), int(rgb[2])))


def compare_pixmap(image_hex: str, match_hex: str, width: int = 40, height: int = 18):
    """A two-tone chip: the picture's colour on the left, the match on the right.

    The list has one icon slot per row, and the whole point of the row is the
    comparison, so both colours share it.
    """
    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QPainter, QPixmap

    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(Qt.PenStyle.NoPen)
    half = width / 2.0
    painter.setBrush(_color_qcolor(image_hex))
    painter.drawRoundedRect(QRectF(0, 0, half + 2.0, height), 3.0, 3.0)
    painter.setBrush(_color_qcolor(match_hex))
    painter.drawRoundedRect(QRectF(half - 2.0, 0, half + 2.0, height), 3.0, 3.0)
    painter.end()
    return pixmap


def _color_qcolor(value):
    from PySide6.QtGui import QColor

    return QColor(value)


def _parent_short_name(filament) -> str:
    """The shortest label that still tells two spools apart.

    A list row has roughly sixty characters of room in the picture page, and the
    percentages at the end of a recipe are the part that must not be clipped, so
    the compact form prefers the user's own name or note over brand + type.
    """
    for value in (filament.name, filament.note, filament.material_type, filament.brand):
        if value:
            return str(value)
    return filament.color_hex


def entry_recipe_text(entry: PaletteEntry, library=None, *, compact: bool = False) -> str:
    """「配方：白 (#FCF4F0) 82% + 红 (#C8342E) 18%」 or the spool itself.

    ``compact`` drops the brand/type and the hex, so the row can show the two
    percentages without eliding them; the full recipe stays in the tooltip and
    in the 「选中的颜色」 panel.
    """
    if entry.recipe is None:
        filament = entry.filament
        if filament is None:
            return f"{SPOOL_PREFIX}：{entry.color_hex}"
        name = _parent_short_name(filament) if compact else filament.display_name
        if compact:
            return f"{SPOOL_PREFIX}：{name} · 单色 100%"
        return f"{SPOOL_PREFIX}：{name} ({filament.color_hex}) · 单色 100%"
    recipe = entry.recipe
    parts = []
    for filament_id, percent in ((recipe.a_id, recipe.percent_a), (recipe.b_id, recipe.percent_b)):
        filament = library.get(filament_id) if library is not None else None
        if filament is None:
            parts.append(f"{filament_id} {percent}%")
        elif compact:
            parts.append(f"{_parent_short_name(filament)} {percent}%")
        else:
            parts.append(f"{filament.display_name} ({filament.color_hex}) {percent}%")
    return f"{RECIPE_PREFIX}：{parts[0]}  +  {parts[1]}"


def entry_short_label(entry: PaletteEntry, library=None) -> str:
    """A one-line description for a list row."""
    if entry.recipe is None:
        filament = entry.filament
        return filament.display_name if filament is not None else entry.color_hex
    return entry.label


class _SwatchCard(QFrame):
    """One labelled colour chip with a hex caption and a note."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("swatchCard")
        self._title = QLabel(title)
        self._title.setProperty("role", "sectionTitle")
        self._chip = QLabel()
        self._chip.setFixedSize(72, 72)
        self._hex = QLabel("—")
        self._hex.setProperty("role", "mono")
        self._note = QLabel("")
        self._note.setProperty("role", "hint")
        self._note.setWordWrap(True)

        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(4)
        box.addWidget(self._title)
        box.addWidget(self._chip)
        box.addWidget(self._hex)
        box.addWidget(self._note)
        box.addStretch(1)

    def setColour(self, color_hex: str, note: str) -> None:
        self._chip.setPixmap(swatch_pixmap(color_hex, 72, 72, 8))
        self._hex.setText(color_hex.upper())
        self._note.setText(note)

    def hex(self) -> str:
        return self._hex.text()

    def note(self) -> str:
        return self._note.text()


class ColourDetail(QWidget):
    """Everything about the colour region the user clicked."""

    replaceRequested = Signal()
    restoreRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image_card = _SwatchCard("图片里的颜色")
        self._match_card = _SwatchCard("匹配到的颜色")

        cards = QHBoxLayout()
        cards.setSpacing(14)
        cards.addWidget(self._image_card)
        cards.addWidget(self._match_card)
        cards.addStretch(1)

        self._recipe = QLabel("")
        self._recipe.setWordWrap(True)
        self._difference = QLabel("")
        self._difference.setProperty("role", "hint")
        self._difference.setWordWrap(True)
        self._hint = QLabel("点一个颜色，这里会显示图片里的颜色、匹配到的颜色和它的配方。")
        self._hint.setProperty("role", "hint")
        self._hint.setWordWrap(True)

        self._replace = QPushButton("更换颜色…")
        self._replace.setProperty("accent", "true")
        self._replace.setToolTip("从「全部颜色」里另选一个颜色替换这个色块")
        self._replace.clicked.connect(self.replaceRequested.emit)

        self._restore = QPushButton("恢复自动匹配")
        self._restore.setToolTip("撤销手动更换，用程序匹配到的颜色")
        self._restore.clicked.connect(self.restoreRequested.emit)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        buttons.addWidget(self._replace)
        buttons.addWidget(self._restore)
        buttons.addStretch(1)

        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(8)
        box.addLayout(cards)
        box.addWidget(self._recipe)
        box.addWidget(self._difference)
        box.addWidget(self._hint)
        box.addLayout(buttons)

        self.clear()

    def clear(self) -> None:
        self._image_card.setColour("#FFFFFF", "")
        self._match_card.setColour("#FFFFFF", "")
        self._image_card.setVisible(False)
        self._match_card.setVisible(False)
        self._recipe.setText("")
        self._difference.setText("")
        self._hint.setVisible(True)
        self._replace.setEnabled(False)
        self._restore.setEnabled(False)
        self._restore.setVisible(False)

    def setSelection(
        self,
        result,
        index: int,
        *,
        library=None,
        changed: bool = False,
        original: PaletteEntry | None = None,
    ) -> None:
        if result is None or not (0 <= index < len(result.palette)):
            self.clear()
            return
        entry = result.palette[index]
        image_rgb = result.region_colour(index)
        image_hex = hex_of(image_rgb)
        share = float(result.shares()[index]) * 100.0
        pixels = int(result.counts[index])

        self._image_card.setVisible(True)
        self._match_card.setVisible(True)
        self._image_card.setColour(
            image_hex, f"RGB {image_rgb[0]}, {image_rgb[1]}, {image_rgb[2]}\n{pixels} 像素 · {share:.1f}%"
        )

        difference = float(
            _color.delta_e_2000(
                _color.lab_from_rgb(image_rgb), _color.lab_from_rgb(entry.rgb)
            )
        )
        note = f"RGB {entry.rgb[0]}, {entry.rgb[1]}, {entry.rgb[2]}\nΔE00 {difference:.2f}"
        self._match_card.setColour(entry.color_hex, note)

        self._recipe.setText(entry_recipe_text(entry, library))
        self._hint.setVisible(False)

        if changed:
            original_hex = original.color_hex.upper() if original is not None else "自动匹配色"
            self._difference.setText(
                f"已手动更换：程序原本匹配到 {original_hex}（ΔE00 "
                f"{float(_color.delta_e_2000(_color.lab_from_rgb(image_rgb), _color.lab_from_rgb(original.rgb))):.2f}）。"
                "点「恢复自动匹配」可以撤销。"
            )
        else:
            level = "很接近" if difference < 2.0 else "接近" if difference < 6.0 else "偏差较大"
            self._difference.setText(
                f"图片颜色与匹配颜色的差距 ΔE00 {difference:.2f}（{level}；ΔE00 越小越像）。"
            )
        self._replace.setEnabled(True)
        self._restore.setVisible(True)
        self._restore.setEnabled(bool(changed))

    # -- read-back, for the smoke checks and the tests ---------------------
    def image_hex(self) -> str:
        """The hex of the picture region's own colour."""
        return self._image_card.hex()

    def match_hex(self) -> str:
        """The hex of the colour we matched (or the one the user chose)."""
        return self._match_card.hex()

    def recipe_text(self) -> str:
        return self._recipe.text()

    def difference_text(self) -> str:
        return self._difference.text()

    def can_restore(self) -> bool:
        # ``isVisible()`` is False whenever the page itself is not on screen, so
        # ask about the widget's own hidden flag instead of the window state.
        return not self._restore.isHidden() and self._restore.isEnabled()

    def can_replace(self) -> bool:
        return self._replace.isEnabled()


class ColourPickerDialog(QDialog):
    """Pick a replacement out of 「全部颜色」."""

    def __init__(
        self,
        entries,
        target_rgb,
        *,
        library=None,
        current_key: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._entries = list(entries)
        self._target_rgb = tuple(int(v) for v in target_rgb)
        self._library = library
        self._chosen: PaletteEntry | None = None
        self.setWindowTitle("更换颜色")
        self.resize(560, 620)

        self._target = _SwatchCard("要替换的图片颜色")
        self._target.setColour(
            hex_of(self._target_rgb),
            f"RGB {self._target_rgb[0]}, {self._target_rgb[1]}, {self._target_rgb[2]}",
        )

        self._sort = QComboBox()
        for key, label in SORT_CHOICES_WITH_SIMILARITY:
            self._sort.addItem(label, key)
        self._sort.setCurrentIndex(
            self._sort.findData(SORT_SIMILARITY) if self._sort.findData(SORT_SIMILARITY) >= 0 else 0
        )
        self._sort.setToolTip("和「混色配方」页一样的排序方式，外加按跟图片目标颜色最相似排序")
        self._sort.currentIndexChanged.connect(lambda *_: self._rebuild())

        self._search = QLineEdit()
        self._search.setPlaceholderText("搜索：RGB 片段如 3D7B，或耗材名称 / 品牌 / 种类")
        self._search.textChanged.connect(lambda *_: self._rebuild())

        self._list = QListWidget()
        self._list.setIconSize(swatch_pixmap("#FFFFFF", 22).size())
        self._list.setUniformItemSizes(True)
        self._list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self._list.currentRowChanged.connect(self._on_row_changed)
        self._list.itemDoubleClicked.connect(self._on_double_clicked)

        self._info = QLabel("")
        self._info.setProperty("role", "hint")
        self._info.setWordWrap(True)

        controls = QHBoxLayout()
        controls.setSpacing(8)
        controls.addWidget(QLabel("排序:"))
        controls.addWidget(self._sort)
        controls.addWidget(self._search, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        # Qt's own translations are not shipped in the bundle, so the standard
        # buttons come out as "OK"/"Cancel" in an otherwise Chinese window.
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("用这个颜色")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self._buttons = buttons

        box = QVBoxLayout(self)
        box.setContentsMargins(12, 12, 12, 12)
        box.setSpacing(10)
        box.addWidget(self._target)
        box.addLayout(controls)
        box.addWidget(self._list, 1)
        box.addWidget(self._info)
        box.addWidget(buttons)

        self._rebuild()

    # -- data -------------------------------------------------------------
    def chosen(self) -> PaletteEntry | None:
        return self._chosen

    def ordered(self) -> list[PaletteEntry]:
        return sort_palette(
            self._entries, self._sort.currentData(), target_rgb=self._target_rgb
        )

    def _matches(self, entry: PaletteEntry, query: str) -> bool:
        needle = query.strip().casefold()
        if not needle:
            return True
        if needle.lstrip("#") in entry.color_hex.casefold():
            return True
        if needle in entry.label.casefold():
            return True
        filament = entry.filament
        if filament is not None:
            fields = (filament.display_name, filament.brand, filament.material_type, filament.note)
            if any(needle in str(field).casefold() for field in fields if field):
                return True
        return False

    def _rebuild(self) -> None:
        query = self._search.text()
        rows = [entry for entry in self.ordered() if self._matches(entry, query)]
        self._list.blockSignals(True)
        self._list.clear()
        target_lab = _color.lab_from_rgb(self._target_rgb)
        for entry in rows:
            difference = float(
                _color.delta_e_2000(target_lab, entry.lab)
            )
            item = QListWidgetItem(
                swatch_pixmap(entry.color_hex, 22),
                f"{entry.color_hex}   ΔE00 {difference:5.2f}   {entry_short_label(entry, self._library)}",
            )
            item.setData(Qt.ItemDataRole.UserRole, entry.key)
            item.setToolTip(f"{entry_short_label(entry, self._library)}\n{entry_recipe_text(entry, self._library)}")
            self._list.addItem(item)
        self._list.blockSignals(False)
        self._info.setText(
            f"「全部颜色」共 {len(self._entries)} 个：{len(self._entries) - sum(1 for e in self._entries if e.recipe is None)} 个混色 "
            f"+ {sum(1 for e in self._entries if e.recipe is None)} 种耗材本色 · 当前显示 {len(rows)} 个"
        )
        self._chosen = None
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        if rows:
            self._list.setCurrentRow(0)

    def _on_row_changed(self, row: int) -> None:
        if row < 0:
            self._chosen = None
            self._buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
            return
        key = self._list.item(row).data(Qt.ItemDataRole.UserRole)
        self._chosen = next((entry for entry in self._entries if entry.key == key), None)
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(self._chosen is not None)

    def _on_double_clicked(self, item: QListWidgetItem) -> None:
        self._on_row_changed(self._list.row(item))
        if self._chosen is not None:
            self.accept()

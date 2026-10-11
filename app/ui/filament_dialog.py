"""Add or edit one spool: 品牌, 耗材种类, 颜色 (RGB), 备注."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)

from ..core.library import COMMON_BRANDS, COMMON_MATERIAL_TYPES, Filament
from ..spectral import color as _color
from . import theme
from .selectable import selectable_text
from .swatch import ColorField, SwatchLabel, is_unset


class FilamentDialog(QDialog):
    """Collect the five fields the user types for a spool."""

    def __init__(self, filament: Filament | None = None, brands=(), types=(), parent=None):
        super().__init__(parent)
        self._filament = filament
        self.setWindowTitle("编辑耗材" if filament is not None else "添加耗材")
        self.setMinimumWidth(440)
        self.setModal(True)

        self._name = QLineEdit(self)
        self._name.setPlaceholderText("可选，例如「大简 PETG HF 金色」")

        self._brand = QComboBox(self)
        self._brand.setEditable(True)
        for value in _merged(COMMON_BRANDS, brands):
            self._brand.addItem(value)
        self._brand.setCurrentText("")

        self._type = QComboBox(self)
        self._type.setEditable(True)
        for value in _merged(COMMON_MATERIAL_TYPES, types):
            self._type.addItem(value)
        self._type.setCurrentText("")

        # No preset colour, and no pop-up palette either: the colour is typed in
        # right here as a hex value or as three numbers.
        self._color = ColorField(None, self, stacked=True)

        self._note = QLineEdit(self)
        self._note.setPlaceholderText("可选，例如「2026-07 批次 / 实测色卡」")

        self._preview = SwatchLabel(None, size=64, caption=True)
        self._hint = QLabel("请填写颜色：可以直接输入 #RRGGBB，也可以填 R / G / B 三个数字。", self)
        self._hint.setProperty("role", "hint")
        self._hint.setWordWrap(True)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(10)
        form.addRow("名称", self._name)
        form.addRow("品牌", self._brand)
        form.addRow("耗材种类", self._type)
        form.addRow("颜色 (RGB)", self._color)
        form.addRow("备注", self._note)

        preview_row = QHBoxLayout()
        preview_row.setContentsMargins(0, 0, 0, 0)
        preview_row.setSpacing(12)
        preview_row.addWidget(self._preview, 0, Qt.AlignmentFlag.AlignTop)
        preview_row.addWidget(self._hint, 1)
        preview_row.addStretch(0)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("确认")
        buttons.button(QDialogButtonBox.StandardButton.Ok).setProperty("accent", True)
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        self._ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 14)
        layout.setSpacing(14)
        layout.addLayout(form)
        layout.addLayout(preview_row)
        layout.addStretch(1)
        layout.addWidget(buttons)

        self._color.colorChanged.connect(self._on_color_changed)

        # The hint spells out the RGB / CIELAB numbers the user may want to copy.
        selectable_text(self)

        if filament is not None:
            self._name.setText(filament.name)
            self._brand.setCurrentText(filament.brand)
            self._type.setCurrentText(filament.material_type)
            self._note.setText(filament.note)
            self._apply_color(filament.color_hex)
        else:
            self._apply_color(None)

    # -- colour plumbing ---------------------------------------------------------
    def _apply_color(self, value) -> None:
        self._color.setValue(value)
        self._preview.setValue(None if is_unset(value) else value)
        self._update_hint()

    def _on_color_changed(self, value: str) -> None:
        self._preview.setValue(value or None)
        self._update_hint()

    def _resolved_hex(self) -> str:
        """The colour this dialog means right now.

        The three spin boxes start at 0 / 0 / 0 and the user reasonably reads
        that as the colour they are about to add, so typing nothing at all and
        pressing 「确认」 has to work and has to mean black.  The hex box stays
        authoritative whenever it holds something.
        """
        typed = self._color.hex()
        if not is_unset(typed):
            return typed
        return _color.rgb_to_hex(self._color.rgb())

    def _update_hint(self) -> None:
        self._ok_button.setEnabled(True)
        if not self._color.isSet():
            r, g, b = self._color.rgb()
            self._hint.setText(
                "请填写颜色：可以直接输入 #RRGGBB，也可以填 R / G / B 三个数字。\n"
                f"什么都没填就按三个数字框里的值算，现在是 RGB {r}, {g}, {b}。"
            )
            return
        r, g, b = _color.hex_to_rgb(self._color.hex())
        lab = _color.lab_from_rgb((r, g, b))
        self._hint.setText(
            f"RGB {r}, {g}, {b}\n"
            f"CIELAB  L* {lab[0]:.1f}   a* {lab[1]:.1f}   b* {lab[2]:.1f}\n"
            "请填写该耗材实际打印出来的颜色，而不是线材本身的颜色。"
        )

    # -- results -----------------------------------------------------------------
    def values(self) -> dict:
        return {
            "name": self._name.text().strip(),
            "brand": self._brand.currentText().strip(),
            "material_type": self._type.currentText().strip(),
            "color_hex": self._resolved_hex(),
            "note": self._note.text().strip(),
        }

    def build_filament(self) -> Filament:
        value = self._resolved_hex()
        if is_unset(value):
            raise ValueError("请先选择颜色")
        if self._filament is None:
            return Filament(**self.values())
        return self._filament.copy(**self.values())


def _merged(common, extra) -> list[str]:
    """Common suggestions first, then whatever the user's library already uses."""
    seen: list[str] = []
    for value in list(common) + list(extra):
        text = (value or "").strip()
        if text and text not in seen:
            seen.append(text)
    return seen

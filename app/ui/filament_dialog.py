"""Add or edit one spool: 品牌, 耗材种类, 颜色 (RGB), 备注."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
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
        self._another = False
        self.setWindowTitle("编辑耗材" if filament is not None else "添加耗材")
        self.setMinimumWidth(440)
        self.setModal(True)

        self._name = QLineEdit(self)

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

        # The row is built by hand instead of with QDialogButtonBox so 「下一个」
        # sits IMMEDIATELY left of 「确认」, which is where the user asked for it —
        # a button box parks an ActionRole button at the far left edge instead.
        self._ok_button = QPushButton("确认", self)
        self._ok_button.setProperty("accent", True)
        self._ok_button.setDefault(True)
        self._cancel_button = QPushButton("取消", self)
        self._next_button = QPushButton("下一个", self)
        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.setSpacing(8)
        button_row.addStretch(1)
        button_row.addWidget(self._next_button)
        button_row.addWidget(self._ok_button)
        button_row.addWidget(self._cancel_button)
        self._ok_button.clicked.connect(self.accept)
        self._cancel_button.clicked.connect(self.reject)
        self._next_button.clicked.connect(self._on_next)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 14)
        layout.setSpacing(14)
        layout.addLayout(form)
        layout.addLayout(preview_row)
        layout.addStretch(1)
        layout.addLayout(button_row)

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
            # 新耗材默认 0, 0, 0（m09456 item 1）：用户要的就是这一条。原来的
            # 「一个颜色都不预设」让三个数字框显示空白、确认按钮发灰，而想填黑色的
            # 人第一件事就是打 0 —— 那时值没变、信号不响，看着像「输不进去」。
            # 现在对话框一打开就是纯黑，三个框显示 0，确认按钮直接可用。
            self._apply_color("#000000")

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

        新耗材一打开就是 ``#000000``（m09456 item 1），编辑时是这卷料原来的颜色，
        所以这里读到的永远是用户看得见的那个值。曾经有过一段「什么都不预设」的
        时间：hex 框空白、三个数字框空白，用户想填黑色打 0 却因为数字没变而没有
        任何反应；1.6.27 的 ``commitPending`` 又补上了「打完 hex 直接点确认」那条
        路（焦点还留在输入框里、值还没提交，确认按钮是灰的，灰按钮不接焦点，点击
        也不会让输入框失焦）。

        ``commitPending`` 先跑：读之前先把输入框里的完整颜色收进来。
        """
        self._color.commitPending()
        return self._color.hex()

    def _update_hint(self) -> None:
        """Say what the colour is, and keep 确认 shut until there is one.

        新耗材默认 ``#000000``（m09456 item 1），所以两个保存按钮一打开就是可用的；
        编辑时也一样。灰按钮只留给「用户把颜色清空了」这一种情况——那时确实没有
        颜色可存。
        """
        if not self._color.isSet():
            self._hint.setText("请填写颜色：可以直接输入 #RRGGBB，也可以填 R / G / B 三个数字。")
            self._ok_button.setEnabled(False)
            self._next_button.setEnabled(False)
            return
        self._ok_button.setEnabled(True)
        self._next_button.setEnabled(True)
        r, g, b = _color.hex_to_rgb(self._color.hex())
        lab = _color.lab_from_rgb((r, g, b))
        self._hint.setText(
            f"RGB {r}, {g}, {b}\n"
            f"CIELAB  L* {lab[0]:.1f}   a* {lab[1]:.1f}   b* {lab[2]:.1f}\n"
            "请填写该耗材实际打印出来的颜色，而不是线材本身的颜色。"
        )

    # -- results -----------------------------------------------------------------
    def _on_next(self) -> None:
        """Save this spool but stay open for the next one.

        The dialog reports it through :meth:`wants_another` rather than by
        avoiding ``accept()``, so the caller keeps ONE code path for reading
        ``values()`` and the window can open a fresh dialog for the next spool
        (a reset in place would have to un-remember every widget by hand).
        """
        self._another = True
        self.accept()

    def wants_another(self) -> bool:
        """True when the dialog was closed with 「下一个」 instead of 「确认」."""
        return self._another

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

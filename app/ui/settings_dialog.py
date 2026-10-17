"""「设置」：三页的默认参数，以及能被改掉的快捷键。

1.6.53（m21616 第 8 条）加的这一页。它只做一件事：把「每次打开都想先这么摆」的那些值
收进偏好文件（:mod:`app.core.settings`），打开新图 / 新模型时按它们来。

写在这儿的三条规矩：

* **只放默认值，不放开关状态**。比如「只匹配耗材」进设置的是「打开时就勾上」，
  而不是「替用户勾上」——页面上那个勾随时可以再改，改了也不回写偏好。
* **快捷键只给真的绑了的那些**。目前只有模型配色的「导出 GLB」（Ctrl+S）是可改的，
  其余按键（小键盘 1/3/7、Home、中键拖动）是三维视图自己的手感，不在这儿重绑。
* **取消就是不变**。点「取消」什么都不写；「恢复默认」只把这一页的控件拨回出厂值，
  仍然要再按「保存」才落盘，这样它也可以当「我只想看看出厂是多少」。
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QKeySequenceEdit,
    QLabel,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..core.mixes import SORT_CHOICES
from ..core.settings import DEFAULT_SETTINGS, load_settings
from .selectable import selectable_text

__all__ = ["SettingsDialog"]

#: 颜色数量的上下限：和页面上的数字框一致（2 种到 41 种）。
_COLOURS_MIN, _COLOURS_MAX = 2, 41
#: 耗材上限：0 表示不限。
_SPOOLS_MIN, _SPOOLS_MAX = 0, 41
#: 底板厚度：页面上的范围是 0.8～20 mm（至少要比一层颜色厚）。
_BASE_MIN, _BASE_MAX = 0.8, 20.0


def _spin(low: int, high: int, value: int, suffix: str = "") -> QSpinBox:
    box = QSpinBox()
    box.setRange(low, high)
    box.setValue(max(low, min(high, int(value))))
    if suffix:
        box.setSuffix(suffix)
    box.setAccelerated(True)
    return box


class SettingsDialog(QDialog):
    """默认参数 + 快捷键。``values()`` 交出要写进偏好的那一份。"""

    def __init__(self, values: dict | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("设置：默认参数与快捷键")
        self.setMinimumWidth(520)
        self._current = dict(DEFAULT_SETTINGS)
        self._current.update(values or load_settings())

        hint = QLabel(
            "这里改的是「下次打开时先按这一套来」。已经在页面上的图 / 模型不会因为保存设置而重新算，"
            "要它生效就按一次「重新匹配」（或者重新导入）。"
        )
        hint.setProperty("role", "hint")
        hint.setWordWrap(True)

        # -- 颜色配方 ---------------------------------------------------
        self._sort = QComboBox()
        for sort_id, label in SORT_CHOICES:
            self._sort.addItem(label, sort_id)
        index = self._sort.findData(str(self._current.get("sort") or "rgb"))
        self._sort.setCurrentIndex(max(0, index))
        mix_page = self._form_page(
            (
                (
                    "默认排序",
                    self._sort,
                    "打开时「全部颜色」按什么次序排。和这一页顶上那个「排序」是同一份列表。",
                ),
            )
        )

        # -- 图像转换 ---------------------------------------------------
        self._picture_colours = _spin(
            _COLOURS_MIN, _COLOURS_MAX, int(self._current.get("picture_colours", 32))
        )
        self._picture_spools = _spin(
            _SPOOLS_MIN, _SPOOLS_MAX, int(self._current.get("picture_spools", 4))
        )
        self._picture_dither = QCheckBox("打开时就勾上「抖动」")
        self._picture_dither.setChecked(bool(self._current.get("picture_dither", False)))
        self._picture_only = QCheckBox("打开时就勾上「只匹配耗材」")
        self._picture_only.setChecked(bool(self._current.get("picture_only_spools", False)))
        self._picture_base = QDoubleSpinBox()
        self._picture_base.setRange(_BASE_MIN, _BASE_MAX)
        self._picture_base.setSingleStep(0.2)
        self._picture_base.setDecimals(2)
        self._picture_base.setSuffix(" mm")
        self._picture_base.setValue(
            max(_BASE_MIN, min(_BASE_MAX, float(self._current.get("picture_base_mm", 1.4))))
        )
        picture_page = self._form_page(
            (
                ("颜色数量", self._picture_colours, "导入图片时「颜色数量」的起始值。"),
                ("耗材上限", self._picture_spools, "0 表示不限；超过上限的色块会被合成更便宜的色。"),
                ("", self._picture_dither, "抖动会把近似色打散成点，近距离看更细腻、远处更糊。"),
                ("", self._picture_only, "只用你自己这些耗材的本色，不拿混色去凑。"),
                ("底板厚度", self._picture_base, "生成 3MF / OBJ 时底板那一层的厚度。"),
            )
        )

        # -- 模型配色 ---------------------------------------------------
        self._model_colours = _spin(
            _COLOURS_MIN, _COLOURS_MAX, int(self._current.get("model_colours", 32))
        )
        self._model_spools = _spin(
            _SPOOLS_MIN, _SPOOLS_MAX, int(self._current.get("model_spools", 4))
        )
        self._model_only = QCheckBox("打开时就勾上「只匹配耗材」")
        self._model_only.setChecked(bool(self._current.get("model_only_spools", False)))
        self._model_flip = QCheckBox("旋转左右翻转（默认勾上）")
        self._model_flip.setChecked(bool(self._current.get("model_flip_rotate", True)))
        self._shortcut_glb = QKeySequenceEdit(
            QKeySequence(str(self._current.get("shortcut_export_glb") or "Ctrl+S"))
        )
        model_page = self._form_page(
            (
                ("颜色数量", self._model_colours, "打开模型时「颜色数量」的起始值。"),
                ("耗材上限", self._model_spools, "0 表示不限。"),
                ("", self._model_only, "只用你自己这些耗材的本色。"),
                (
                    "",
                    self._model_flip,
                    "三维视图里按住中键左右拖，模型跟着往哪边转。勾上是转盘式（拖左边，模型的左边转过来），"
                    "取消就是反过来。",
                ),
                ("「导出 GLB」快捷键", self._shortcut_glb, "在模型配色页按下它等于按「导出 GLB」。"),
            )
        )

        tabs = QTabWidget()
        tabs.addTab(mix_page, "颜色配方")
        tabs.addTab(picture_page, "图像转换")
        tabs.addTab(model_page, "模型配色")

        reset = QPushButton("恢复默认")
        reset.clicked.connect(self._on_reset)
        buttons = QDialogButtonBox()
        buttons.addButton("保存", QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton("取消", QDialogButtonBox.ButtonRole.RejectRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        footer = QHBoxLayout()
        footer.setSpacing(6)
        footer.addWidget(reset)
        footer.addStretch(1)
        footer.addWidget(buttons)

        box = QVBoxLayout(self)
        box.setContentsMargins(12, 12, 12, 12)
        box.setSpacing(8)
        box.addWidget(hint)
        box.addWidget(tabs, 1)
        box.addLayout(footer)
        selectable_text(self)

    @staticmethod
    def _form_page(rows) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        form.setContentsMargins(12, 12, 12, 12)
        form.setSpacing(8)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        for label, widget, tip in rows:
            if tip:
                widget.setToolTip(tip)
            form.addRow(QLabel(label) if label else QLabel(""), widget)
        return page

    def _on_reset(self) -> None:
        index = self._sort.findData(str(DEFAULT_SETTINGS.get("sort") or "rgb"))
        self._sort.setCurrentIndex(max(0, index))
        self._picture_colours.setValue(int(DEFAULT_SETTINGS["picture_colours"]))
        self._picture_spools.setValue(int(DEFAULT_SETTINGS["picture_spools"]))
        self._picture_dither.setChecked(bool(DEFAULT_SETTINGS["picture_dither"]))
        self._picture_only.setChecked(bool(DEFAULT_SETTINGS["picture_only_spools"]))
        self._picture_base.setValue(float(DEFAULT_SETTINGS["picture_base_mm"]))
        self._model_colours.setValue(int(DEFAULT_SETTINGS["model_colours"]))
        self._model_spools.setValue(int(DEFAULT_SETTINGS["model_spools"]))
        self._model_only.setChecked(bool(DEFAULT_SETTINGS["model_only_spools"]))
        self._model_flip.setChecked(bool(DEFAULT_SETTINGS["model_flip_rotate"]))
        self._shortcut_glb.setKeySequence(
            QKeySequence(str(DEFAULT_SETTINGS.get("shortcut_export_glb") or "Ctrl+S"))
        )

    def values(self) -> dict:
        """The preferences this dialog would write."""
        sequence = self._shortcut_glb.keySequence().toString()
        return {
            "sort": str(self._sort.currentData() or "rgb"),
            "picture_colours": int(self._picture_colours.value()),
            "picture_spools": int(self._picture_spools.value()),
            "picture_dither": bool(self._picture_dither.isChecked()),
            "picture_only_spools": bool(self._picture_only.isChecked()),
            "picture_base_mm": float(self._picture_base.value()),
            "model_colours": int(self._model_colours.value()),
            "model_spools": int(self._model_spools.value()),
            "model_only_spools": bool(self._model_only.isChecked()),
            "model_flip_rotate": bool(self._model_flip.isChecked()),
            "shortcut_export_glb": sequence or "Ctrl+S",
        }

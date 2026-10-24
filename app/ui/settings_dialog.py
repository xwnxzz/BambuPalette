"""「设置」：三页的默认参数，以及能被改掉的快捷键。

1.6.53（m21616 第 8 条）加的这一页；1.6.54（m22211 第 7 条）把它铺开：每页都有一组
自己的快捷键，还给模型配色加了三维视图的底色。

写在这儿的四条规矩：

* **只放默认值，不放开关状态**。比如「只匹配耗材」进设置的是「打开时就勾上」，
  而不是「替用户勾上」——页面上那个勾随时可以再改，改了也不回写偏好。
* **快捷键按页分开，页内不许撞车，跨页可以重复**。同一时刻只有一页在屏幕上，所以
  「图像转换的 Ctrl+O」和「模型配色的 Ctrl+O」互不干扰；但同一页里两个动作抢一个
  按键就没法用了，保存时会被拦下来（:meth:`SettingsDialog.accept`）。
* **取消就是不变**。点「取消」什么都不写；「恢复默认」只把这一页的控件拨回出厂值，
  仍然要再按「保存」才落盘，这样它也可以当「我只想看看出厂是多少」。
* **三维视图自己的手感不进这儿**：中键拖动、小键盘 1/3/7、Home 那些是视图内部的
  操作，重绑了反而不像 Blender。
"""

from __future__ import annotations

import re

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QKeySequenceEdit,
    QLabel,
    QLineEdit,
    QMessageBox,
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
        self.setMinimumWidth(560)
        self._current = dict(DEFAULT_SETTINGS)
        self._current.update(values or load_settings())

        hint = QLabel(
            "这里改的是「下次打开时先按这一套来」。已经在页面上的图 / 模型不会因为保存设置而重新算，"
            "要它生效就按一次「重新匹配」（或者重新导入）。快捷键按页分开：同一页里不许重复，"
            "不同页之间可以一样。"
        )
        hint.setProperty("role", "hint")
        hint.setWordWrap(True)

        # -- 颜色配方 ---------------------------------------------------
        self._sort = QComboBox()
        for sort_id, label in SORT_CHOICES:
            self._sort.addItem(label, sort_id)
        index = self._sort.findData(str(self._current.get("sort") or "rgb"))
        self._sort.setCurrentIndex(max(0, index))
        self._shortcut_find = self._shortcut("shortcut_recipe_find", "Ctrl+F")
        self._shortcut_add = self._shortcut("shortcut_recipe_add", "Ctrl+N")
        mix_page = self._form_page(
            (
                (
                    "默认排序",
                    self._sort,
                    "打开时「全部颜色」按什么次序排。和这一页顶上那个「排序」是同一份列表。",
                ),
                ("「找最接近的颜色」快捷键", self._shortcut_find, "在颜色配方页按下它等于按那颗按钮。"),
                ("「添加耗材」快捷键", self._shortcut_add, "在颜色配方页按下它等于按右上角那颗按钮。"),
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
        self._shortcut_pic_import = self._shortcut("shortcut_picture_import", "Ctrl+O")
        self._shortcut_pic_rematch = self._shortcut("shortcut_picture_rematch", "Ctrl+R")
        self._shortcut_pic_save = self._shortcut("shortcut_picture_save_project", "Ctrl+Shift+S")
        self._shortcut_pic_export = self._shortcut("shortcut_picture_export_png", "Ctrl+E")
        picture_page = self._form_page(
            (
                ("颜色数量", self._picture_colours, "导入图片时「颜色数量」的起始值。"),
                ("耗材上限", self._picture_spools, "0 表示不限；超过上限的色块会被合成更便宜的色。"),
                ("", self._picture_dither, "抖动会把近似色打散成点，近距离看更细腻、远处更糊。"),
                ("", self._picture_only, "只用你自己这些耗材的本色，不拿混色去凑。"),
                ("底板厚度", self._picture_base, "生成 3MF / OBJ 时底板那一层的厚度。"),
                ("「导入图片」快捷键", self._shortcut_pic_import, "等于按「导入图片」。"),
                ("「按当前色块重新匹配」快捷键", self._shortcut_pic_rematch, "等于按那颗按钮。"),
                ("「保存工程」快捷键", self._shortcut_pic_save, "等于按「保存工程」。"),
                ("「导出原图」快捷键", self._shortcut_pic_export, "等于按「导出原图」。"),
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

        # 1.6.60（m27359 第 3 条）：实验性开关。勾上以后两页都不再按「颜色数量」停手。
        self._no_limit = QCheckBox("实验性：没有颜色上限")
        self._no_limit.setChecked(bool(self._current.get("experimental_no_colour_limit", False)))

        # 三维视图底色：一个能点的色块 + 十六进制，两边同步（1.6.54，m22211 第 6 条）。
        self._background = str(self._current.get("model_background") or "#B4C6D2")
        self._background_button = QPushButton()
        self._background_button.clicked.connect(self._on_pick_background)
        self._background_edit = QLineEdit(self._background)
        self._background_edit.setMaximumWidth(110)
        self._background_edit.editingFinished.connect(self._on_background_text)
        background_row = QWidget()
        background_layout = QHBoxLayout(background_row)
        background_layout.setContentsMargins(0, 0, 0, 0)
        background_layout.setSpacing(6)
        background_layout.addWidget(self._background_button)
        background_layout.addWidget(self._background_edit)
        background_layout.addStretch(1)
        self._paint_background_button()

        self._shortcut_model_open = self._shortcut("shortcut_model_open", "Ctrl+O")
        self._shortcut_model_rematch = self._shortcut("shortcut_model_rematch", "Ctrl+R")
        self._shortcut_model_obj = self._shortcut("shortcut_model_export_obj", "Ctrl+Shift+E")
        self._shortcut_model_3mf = self._shortcut("shortcut_model_export_3mf", "Ctrl+Shift+M")
        self._shortcut_glb = self._shortcut("shortcut_export_glb", "Ctrl+S")
        model_page = self._form_page(
            (
                ("颜色数量", self._model_colours, "打开模型时「颜色数量」的起始值。"),
                ("耗材上限", self._model_spools, "0 表示不限。"),
                ("", self._model_only, "只用你自己这些耗材的本色。"),
                (
                    "",
                    self._no_limit,
                    "实验性：勾上后两个页都不看「颜色数量」，能分出多少种颜色就分多少种。"
                    "颜色一多，每种颜色分到的像素就少，打印出来更碎，先小图试。",
                ),
                (
                    "",
                    self._model_flip,
                    "三维视图里按住中键左右拖，模型跟着往哪边转。勾上是转盘式（拖左边，模型的左边转过来），"
                    "取消就是反过来。",
                ),
                ("三维视图底色", background_row, "两个视图的清屏色。默认照 SolidWorks 的浅蓝灰。"),
                ("「打开 GLB」快捷键", self._shortcut_model_open, "等于按「打开 GLB」。"),
                ("「重新匹配」快捷键", self._shortcut_model_rematch, "等于按「重新匹配」。"),
                ("「导出 GLB」快捷键", self._shortcut_glb, "在模型配色页按下它等于按「导出 GLB」。"),
                ("「生成 OBJ + MTL」快捷键", self._shortcut_model_obj, "等于按那颗按钮。"),
                ("「生成 3MF」快捷键", self._shortcut_model_3mf, "等于按那颗按钮。"),
            )
        )

        tabs = QTabWidget()
        tabs.addTab(mix_page, "颜色配方")
        tabs.addTab(picture_page, "图像转换")
        tabs.addTab(model_page, "模型配色")
        self._tabs = tabs

        #: 每一页自己的快捷键：``页名 → ((字段键, 标题, 编辑器), …)``。撞车检查按这一份走。
        self._page_shortcuts: dict[str, tuple[tuple[str, str, QKeySequenceEdit], ...]] = {
            "颜色配方": (
                ("shortcut_recipe_find", "「找最接近的颜色」", self._shortcut_find),
                ("shortcut_recipe_add", "「添加耗材」", self._shortcut_add),
            ),
            "图像转换": (
                ("shortcut_picture_import", "「导入图片」", self._shortcut_pic_import),
                ("shortcut_picture_rematch", "「按当前色块重新匹配」", self._shortcut_pic_rematch),
                ("shortcut_picture_save_project", "「保存工程」", self._shortcut_pic_save),
                ("shortcut_picture_export_png", "「导出原图」", self._shortcut_pic_export),
            ),
            "模型配色": (
                ("shortcut_model_open", "「打开 GLB」", self._shortcut_model_open),
                ("shortcut_model_rematch", "「重新匹配」", self._shortcut_model_rematch),
                ("shortcut_export_glb", "「导出 GLB」", self._shortcut_glb),
                ("shortcut_model_export_obj", "「生成 OBJ + MTL」", self._shortcut_model_obj),
                ("shortcut_model_export_3mf", "「生成 3MF」", self._shortcut_model_3mf),
            ),
        }

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

    # -- 小工具 -----------------------------------------------------------
    def _shortcut(self, key: str, fallback: str) -> QKeySequenceEdit:
        sequence = str(self._current.get(key) or fallback)
        return QKeySequenceEdit(QKeySequence(sequence))

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

    def _paint_background_button(self) -> None:
        colour = QColor(self._background)
        if not colour.isValid():
            colour = QColor("#B4C6D2")
        ink = "#1A1A1A" if colour.lightness() > 140 else "#FFFFFF"
        self._background_button.setText(self._background)
        self._background_button.setStyleSheet(
            f"background: {self._background}; color: {ink}; padding: 4px 12px;"
        )

    def _on_pick_background(self) -> None:
        chosen = QColorDialog.getColor(QColor(self._background), self, "三维视图底色")
        if not chosen.isValid():
            return
        self._background = chosen.name().upper()
        self._background_edit.setText(self._background)
        self._paint_background_button()

    def _on_background_text(self) -> None:
        text = self._background_edit.text().strip()
        if not re.fullmatch(r"#?[0-9a-fA-F]{6}", text):
            self._background_edit.setText(self._background)
            return
        self._background = "#" + text.lstrip("#").upper()
        self._background_edit.setText(self._background)
        self._paint_background_button()

    # -- 校验 / 复位 / 取值 ------------------------------------------------
    def conflicts(self) -> str:
        """同一页里两个动作抢同一个按键时的说明；没有冲突返回空串。"""
        problems: list[str] = []
        for page, entries in self._page_shortcuts.items():
            seen: dict[str, str] = {}
            for _key, title, editor in entries:
                sequence = editor.keySequence().toString()
                if not sequence:
                    continue
                if sequence in seen:
                    problems.append(f"{page}：{seen[sequence]} 与 {title} 都是 {sequence}")
                else:
                    seen[sequence] = title
        return "；".join(problems)

    def accept(self) -> None:  # noqa: D102 - QDialog
        problem = self.conflicts()
        if problem:
            QMessageBox.warning(
                self,
                "快捷键重复",
                f"{problem}。\n同一页里两个动作不能共用一个按键（不同页之间可以）。",
            )
            return
        super().accept()

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
        self._no_limit.setChecked(bool(DEFAULT_SETTINGS["experimental_no_colour_limit"]))
        self._background = str(DEFAULT_SETTINGS.get("model_background") or "#B4C6D2")
        self._background_edit.setText(self._background)
        self._paint_background_button()
        for _page, entries in self._page_shortcuts.items():
            for key, _title, editor in entries:
                editor.setKeySequence(QKeySequence(str(DEFAULT_SETTINGS.get(key) or "")))

    def values(self) -> dict:
        """The preferences this dialog would write."""
        out: dict = {
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
            "experimental_no_colour_limit": bool(self._no_limit.isChecked()),
            "model_background": self._background,
        }
        for _page, entries in self._page_shortcuts.items():
            for key, _title, editor in entries:
                text = editor.keySequence().toString()
                out[key] = text or str(DEFAULT_SETTINGS.get(key) or "")
        return out

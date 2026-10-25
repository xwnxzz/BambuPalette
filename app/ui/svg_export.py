"""导出 SVG 时先问那三个滑杆（1.6.64，m29374）。

sliders instead of a bare file dialog：描摹的质量基本由「几色」决定（实测 8 色时猪信的
鼻子和眼珠会糊成灰块，12 色就都在），其余两个是「精细度 ↔ 文件大小」的取舍。默认值取
的是实测最好那一组，用户直接按回车即可。
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)

from ..core.vectorize import (
    COLOUR_RANGE,
    DEFAULT_COLOURS,
    DEFAULT_SMOOTH,
    DEFAULT_SPECKLE,
    SMOOTH_RANGE,
    SPECKLE_RANGE,
)


class SvgExportDialog(QDialog):
    """「导出 SVG」的参数框：颜色数量 / 去碎块 / 曲线简化。"""

    def __init__(self, parent=None, *, name: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("导出 SVG")
        self.setModal(True)

        intro = QLabel(
            ("把「{0}」描成矢量图。\n" if name else "把这张图描成矢量图。\n")
            + "要的是矢量版（不是图片版）：放大不糊、能进 Blender 或 Illustrator 继续改。"
            "颜色数量决定细节 —— 调得太少，眼珠和描边会跟旁边的灰色糊在一起。"
        )
        if name:
            intro.setText(intro.text().format(name))
        intro.setWordWrap(True)

        self._colours = QSpinBox()
        self._colours.setRange(*COLOUR_RANGE)
        self._colours.setValue(DEFAULT_COLOURS)

        self._speckle = QSpinBox()
        self._speckle.setRange(*SPECKLE_RANGE)
        self._speckle.setValue(DEFAULT_SPECKLE)
        self._speckle.setSuffix(" px")

        self._smooth = QDoubleSpinBox()
        self._smooth.setRange(*SMOOTH_RANGE)
        self._smooth.setValue(DEFAULT_SMOOTH)
        self._smooth.setSingleStep(1.0)
        self._smooth.setDecimals(1)

        form = QFormLayout()
        form.addRow("颜色数量", self._colours)
        form.addRow("去掉碎块", self._speckle)
        form.addRow("曲线简化", self._smooth)

        note = QLabel(
            "去掉碎块：小于这个面积的斑点并进旁边的块，调大更干净、但会把小细节一起抹掉。\n"
            "曲线简化：越大路径越少、文件越小，形状也更硬。"
        )
        note.setProperty("role", "hint")
        note.setWordWrap(True)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("开始转换")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.addWidget(intro)
        layout.addLayout(form)
        layout.addWidget(note)
        layout.addWidget(buttons)

        self.setMinimumWidth(460)
        self._colours.setFocus(Qt.FocusReason.OtherFocusReason)

    def values(self) -> dict[str, object]:
        return {
            "colours": self._colours.value(),
            "speckle": self._speckle.value(),
            "smooth": self._smooth.value(),
        }

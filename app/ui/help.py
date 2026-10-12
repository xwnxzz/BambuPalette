"""「帮助」窗口。

1.6.11 删掉 文件 / 视图 / 帮助 三个菜单时，把「关于与算法来源」留在了顶栏上，
因为它装着版本号、耗材档案路径和混色算法的出处。1.6.14（用户 m01885 第 5 条）
把这个按钮改名叫「帮助」，并让它收下原来压在图片上的使用说明（第 6 条）和
「耗材方案」每一行的解释（第 8 条）——要读的东西放在用户主动打开的窗口里，
而不是压在正在看的图片上面。
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


class HelpDialog(QDialog):
    """A scrollable window holding the whole manual.

    Not modal, like the 颜色详情 dialog: the help explains the panel the user is
    looking at, so it must be possible to read one and use the other.
    """

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("helpDialog")
        self.setWindowTitle("帮助")
        self.setMinimumSize(620, 500)
        self._body = QLabel(text)
        self._body.setObjectName("helpBody")
        self._body.setTextFormat(Qt.TextFormat.PlainText)
        self._body.setWordWrap(True)
        self._body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._body.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._body.setContentsMargins(2, 2, 2, 2)
        scroll = QScrollArea()
        self._scroll = scroll
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(self._body)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.close)
        close = buttons.button(QDialogButtonBox.StandardButton.Close)
        if close is not None:
            close.setText("关闭")
        box = QVBoxLayout(self)
        box.setContentsMargins(14, 14, 14, 14)
        box.setSpacing(10)
        box.addWidget(scroll, 1)
        box.addWidget(buttons)

    def text(self) -> str:
        """The text on screen.

        The automatic checks read this instead of a screenshot: what matters is
        that every explanation left the picture page and arrived here.
        """
        return self._body.text()

    def set_text(self, text: str) -> None:
        """Replace the manual (the version and the archive path can change)."""
        self._body.setText(text)

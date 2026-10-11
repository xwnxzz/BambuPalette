"""让鼠标可以选中界面上的文字，方便复制。

Qt 的 ``QLabel`` 默认只负责显示：文字选不中，也就复制不走。本程序的提示、
配方、状态栏全是要粘进 Bambu Studio 或者发给别人的文本，所以它们都该可选中。

:func:`install` 在 ``QApplication`` 上挂一个事件过滤器，任何 ``QLabel`` 一被
创建就自动变成可选中——包括选中某个颜色之后才现做出来的配方行。比「每建一个
标签手动设一遍」可靠：以后新加的标签不会再漏掉一个。
:func:`selectable_text` 是给过滤器装好之前就已经存在的标签用的补漏。
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import QApplication, QLabel, QWidget

#: 让标签的文字可以用鼠标拖选（于是 Ctrl+C 能复制），但不会变成可编辑输入框。
SELECTABLE = Qt.TextInteractionFlag.TextSelectableByMouse


def selectable(label: QLabel) -> QLabel:
    """把 ``label`` 的文字设成可用鼠标选中，并把它返回以便链式书写。"""
    label.setTextInteractionFlags(SELECTABLE)
    return label


def selectable_text(root: QWidget) -> None:
    """把 ``root`` 底下所有 ``QLabel`` 的文字都设成可用鼠标选中。

    按钮、输入框、列表都不受影响：它们本来就能选能复制，改了反而会抢走
    单击、双击这些交互。
    """
    for label in root.findChildren(QLabel):
        label.setTextInteractionFlags(SELECTABLE)


class _LabelFilter(QObject):
    """``ChildAdded`` 一来就把新标签设成可选中。"""

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 - Qt naming
        if event.type() == QEvent.Type.ChildAdded:
            child = event.child()
            if isinstance(child, QLabel):
                child.setTextInteractionFlags(SELECTABLE)
        # 从不吞掉事件：这里只做顺手的设置，界面的正常行为一点都不改。
        return False


def install(app: QApplication) -> None:
    """让之后创建的每一个 ``QLabel`` 都自动可以选中文字。

    配方行是选中颜色时才现做的，一个面板只走一次 :func:`selectable_text`
    扫不到它们；挂在应用上的过滤器能看到所有标签的出生。
    """
    if getattr(app, "_selectable_filter", None) is None:
        app._selectable_filter = _LabelFilter(app)  # type: ignore[attr-defined]
        app.installEventFilter(app._selectable_filter)  # type: ignore[attr-defined]


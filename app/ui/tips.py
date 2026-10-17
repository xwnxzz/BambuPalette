"""把鼠标悬停时那层黄色小方块关掉（1.6.53，m21616 第 3 条）。

Qt 的 `QToolTip` 是全局的：只要控件带 `setToolTip()`，鼠标停够 `SH_ToolTip_WakeUpDelay`
就会弹一块黄底黑字的小方框，压在界面上、几秒才走。用户的原话是「把图三这种黄色悬停框全部删除」。

做法是拦事件而不是删调用：`QEvent.Type.ToolTip` 一到就吃掉。理由有三条：

* 全程序有多少个 `setToolTip()` 没人说得清（控件、布局提示、按钮、列表行……），漏一个就
  还会弹一块，用户看到的就是「你说删了怎么还有」；
* 删掉调用会把那些写在代码里的解释一起丢掉 —— 它们是给看代码的人读的注释，留着有用；
* 拦事件只在一个地方，闸门里一条 `disable_tooltips()` 就能证明整套界面不再弹框。

代价：`QToolTip.showText(...)` 这种手动弹出的框也一起没了（本程序没有用到）；键盘焦点切换时
系统自己画的提示也一并禁掉 —— 反正它们本来也该去帮助里读。
"""
from __future__ import annotations

from PySide6.QtCore import QEvent, QObject

__all__ = ["ToolTipBlocker", "disable_tooltips", "tooltips_disabled"]


class ToolTipBlocker(QObject):
    """App-wide event filter that swallows every tooltip request."""

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 (Qt)
        if event.type() == QEvent.Type.ToolTip:
            return True
        return super().eventFilter(watched, event)


_BLOCKER: ToolTipBlocker | None = None


def disable_tooltips() -> ToolTipBlocker | None:
    """Install the filter once.  Safe to call from every entry point."""
    global _BLOCKER
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        return None
    if _BLOCKER is None:
        _BLOCKER = ToolTipBlocker(app)
        app.installEventFilter(_BLOCKER)
    return _BLOCKER


def tooltips_disabled() -> bool:
    """True when the filter is installed (the gates check this)."""
    return _BLOCKER is not None

"""带倒计时的进度框，以及「把重活挪到后台线程」的壳子（1.6.66，m29882）。

为什么要有它（四个用户报的现象，一句话一条）：

* 导入 SVG 时没有进度条 —— 四千多万像素的矢量光栅化要好几秒，界面上什么都不动。
* 进度条不真实 —— 匹配引擎只在相位入口报一次（0.30 量化、0.35 找色），在大图上
  这两段各要好几秒，条子就停在原地。
* 导出时点一下窗口就「未响应」 —— 导出在主线程里跑，Windows 过了五秒没收到消息
  就给标题栏加「(未响应)」，用户以为程序死了。
* 想要倒计时 —— 「还要多久」比「已经多久」有用得多。

这里的做法不复杂：重活交给 :class:`QThread`，主线程只负责把它的 `report` 信号画到
进度条上；进度条自己按最近几秒的推进速度估剩下多久，估不出来就说「还在算」。
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from typing import Any, Callable

from PySide6.QtCore import QEventLoop, QObject, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

#: 估算剩余时间时看多长一段历史（秒）。太短会被一次卡顿带偏，太长反应不过来。
ETA_WINDOW = 4.0
#: 多久没推进就把「还要 N 秒」换成「还在算…」（秒）。
STALL_SECONDS = 1.5


def _format_seconds(value: float) -> str:
    """把秒数写成「8 秒」「1 分 20 秒」这种。"""
    total = max(0.0, float(value))
    if total < 1.0:
        return "不到 1 秒"
    if total < 60.0:
        return f"{total:.0f} 秒"
    minutes, seconds = divmod(int(round(total)), 60)
    return f"{minutes} 分 {seconds} 秒"


class _Eta:
    """按最近几秒的推进速度估剩余时间（1.6.66，m29882 第 4 条）。"""

    def __init__(self, window: float = ETA_WINDOW) -> None:
        self._window = float(window)
        self._samples: deque[tuple[float, float]] = deque()
        self._start = time.monotonic()
        self._last_move = self._start
        self._last_value = 0.0

    def reset(self) -> None:
        self._samples.clear()
        self._start = time.monotonic()
        self._last_move = self._start
        self._last_value = 0.0

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._start

    def push(self, value: float) -> float | None:
        """喂一个新的进度值（0～1），返回推算的剩余秒数；推不出来给 ``None``。"""
        now = time.monotonic()
        fraction = min(1.0, max(0.0, float(value)))
        if fraction > self._last_value + 1e-9:
            self._last_value = fraction
            self._last_move = now
        self._samples.append((now, fraction))
        while len(self._samples) > 2 and now - self._samples[0][0] > self._window:
            self._samples.popleft()
        if len(self._samples) < 2:
            return None
        first_time, first_value = self._samples[0]
        span = now - first_time
        gain = fraction - first_value
        if span <= 0.2 or gain <= 1e-9:
            return None
        rate = gain / span
        remaining = (1.0 - fraction) / rate
        # 明显不靠谱的估算（几分钟以上）不如不说。
        if remaining > 3600.0:
            return None
        return remaining

    @property
    def stalled(self) -> bool:
        return time.monotonic() - self._last_move > STALL_SECONDS


class ProgressDialog(QDialog):
    """一个会说话的进度框：在做什么、做到哪、还要多久、能不能停（1.6.66）。"""

    cancelled = Signal()

    def __init__(
        self,
        parent=None,
        title: str = "正在处理",
        *,
        note: str = "",
        modal: bool = True,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(bool(modal))
        self.setMinimumWidth(420)
        self._eta = _Eta()
        self._cancellable = False
        self._finished = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 12)
        layout.setSpacing(10)

        self._note = QLabel(note or "准备中…")
        self._note.setWordWrap(True)
        layout.addWidget(self._note)

        self._bar = QProgressBar(self)
        self._bar.setRange(0, 1000)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(10)
        layout.addWidget(self._bar)

        row = QHBoxLayout()
        row.setSpacing(8)
        self._time_label = QLabel("已用 0 秒")
        self._time_label.setProperty("role", "hint")
        row.addWidget(self._time_label)
        row.addStretch(1)
        self._cancel_button = QPushButton("取消")
        self._cancel_button.setVisible(False)
        self._cancel_button.clicked.connect(self._on_cancel)
        row.addWidget(self._cancel_button)
        layout.addLayout(row)

        self._ticker = QTimer(self)
        self._ticker.setInterval(250)
        self._ticker.timeout.connect(self._on_tick)
        self._ticker.start()

    # --- 外面用的接口 -------------------------------------------------------
    def set_cancellable(self, on: bool) -> None:
        self._cancellable = bool(on)
        self._cancel_button.setVisible(self._cancellable)

    def set_note(self, text: str) -> None:
        if self._finished:
            return
        self._note.setText(str(text))

    def set_fraction(self, value: float) -> None:
        """推进到 ``value``（0～1），顺带刷新倒计时。"""
        if self._finished:
            return
        fraction = min(1.0, max(0.0, float(value)))
        self._bar.setValue(int(round(fraction * 1000)))
        remaining = self._eta.push(fraction)
        self._update_time_label(remaining)

    def set_busy(self, hint: str = "") -> None:
        """不知道总量的时候：条子来回跑，只报已用时间。"""
        if self._finished:
            return
        self._bar.setRange(0, 0)
        if hint:
            self._note.setText(hint)
        self._update_time_label(None)

    def _update_time_label(self, remaining: float | None) -> None:
        elapsed = _format_seconds(self._eta.elapsed)
        if remaining is None:
            tail = "还在算…" if self._eta.stalled else ""
        else:
            tail = f"预计还要 {_format_seconds(remaining)}"
        self._time_label.setText(f"已用 {elapsed}" + (f"，{tail}" if tail else ""))

    def _on_tick(self) -> None:
        if self._finished:
            return
        # 停住的时候也要让「还在算…」出现，所以定时重算一次。
        self._update_time_label(None if self._eta.stalled else self._estimate_once())

    def _estimate_once(self) -> float | None:
        try:
            value = self._bar.value() / 1000.0
        except Exception:  # pragma: no cover - 极端情况下控件已销毁
            return None
        if value <= 0.0:
            return None
        remaining = self._eta.push(value)
        return remaining

    def _on_cancel(self) -> None:
        self._note.setText("正在收尾…")
        self._cancel_button.setEnabled(False)
        self.cancelled.emit()

    def finish(self) -> None:
        """活干完了：把条子推满，关掉窗口。"""
        self._finished = True
        self._ticker.stop()
        self._bar.setRange(0, 1000)
        self._bar.setValue(1000)
        self.accept()

    def restart(self, note: str = "") -> None:
        """复用这个框：把条子、计时和「还在算」全部归零（1.6.66）。

        匹配一次就新建一个 QDialog 会把还没销毁的窗口堆起来，Windows 收尾时直接
        0xC0000005；所以页面只造一个，每次开工调这个。
        """
        self._finished = False
        self._eta.reset()
        self._bar.setRange(0, 1000)
        self._bar.setValue(0)
        self._note.setText(str(note) if note else "")
        self._time_label.setText("已用 不到 1 秒")
        self._ticker.start()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        self._finished = True
        self._ticker.stop()
        super().closeEvent(event)


class _Worker(QObject):
    """把 ``work(report)`` 搬到后台线程里去跑（1.6.66，m29882 第 3 条）。"""

    reported = Signal(float, str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, work: Callable[[Callable[[float, str], None]], Any]) -> None:
        super().__init__()
        self._work = work

    def run(self) -> None:
        def report(value: float, note: str = "") -> None:
            self.reported.emit(float(value), str(note))

        try:
            result = self._work(report)
        except Exception as error:  # noqa: BLE001 - 线程里的异常只能这样带回来
            self.failed.emit(f"{type(error).__name__}: {error}")
        else:
            self.done.emit(result)


class TaskRunner:
    """一次「后台干活 + 进度框」的生命周期。

    1.6.66（m29882 第 3 条，第二版）：干活用**普通 Python 线程**，主线程只拿一个
    50 ms 的 QTimer 去队列里取「进度 / 完成 / 出错」。第一版用 QThread +
    ``moveToThread`` + 信号，在 offscreen 下会把主线程卡死（自检一跑就回不来），
    收尾时还会撞 0xC0000005 —— 那条路上有太多「QObject 归哪条线程」的规矩要守。
    跨线程只传值（float / str / 结果对象），Qt 的对象一个都不跨，稳得多。
    """

    def __init__(
        self,
        parent,
        title: str,
        work: Callable[[Callable[[float, str], None]], Any],
        *,
        note: str = "",
        expected_seconds: float | None = None,
    ) -> None:
        self.dialog = ProgressDialog(parent, title, note=note)
        self._work = work
        self._result: Any = None
        self._error: str | None = None
        self._messages: list[tuple[float, str]] = []
        self._finished = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._body, daemon=True)
        self._expected = expected_seconds
        self._started = time.monotonic()
        self._poll = QTimer(self.dialog)
        self._poll.setInterval(50)
        self._poll.timeout.connect(self._drain)

    # --- 后台 -------------------------------------------------------------
    def _body(self) -> None:
        def report(value: float, note: str = "") -> None:
            with self._lock:
                self._messages.append((float(value), str(note)))

        try:
            result = self._work(report)
        except Exception as error:  # noqa: BLE001 - 线程里的异常只能这样带回来
            with self._lock:
                self._error = f"{type(error).__name__}: {error}"
        else:
            with self._lock:
                self._result = result
        finally:
            with self._lock:
                self._finished = True

    # --- 主线程 -----------------------------------------------------------
    def _drain(self) -> None:
        with self._lock:
            messages = self._messages[:]
            self._messages.clear()
        for value, note in messages:
            if note:
                self.dialog.set_note(note)
            if value < 0:
                self.dialog.set_busy()
            else:
                self.dialog.set_fraction(value)

    # --- 外面 -------------------------------------------------------------
    def run(self) -> tuple[Any, str | None]:
        """跑起来并等它结束，返回 ``(结果, 错误文字)``。

        等的时候跑的是自己的 QEventLoop：主线程照样转、界面照样刷，活干完就退出来。
        不用 ``dialog.exec_()`` —— 模态的嵌套事件循环会挡住调用方剩下的代码。
        """
        if self._expected is not None and self._expected > 0.6:
            self.dialog.set_busy(f"预计 {_format_seconds(self._expected)}，请稍等…")
        self.dialog.show()
        self._poll.start()
        self._thread.start()
        loop = QEventLoop(self.dialog)
        self._watcher = QTimer(self.dialog)
        self._watcher.setInterval(50)
        self._watcher.timeout.connect(lambda: self._drain() or (self._is_done() and loop.quit()))
        self._watcher.start()
        loop.exec()
        self._watcher.stop()
        self._poll.stop()
        self._thread.join(5.0)
        self.dialog.finish()
        return self._result, self._error

    def _is_done(self) -> bool:
        with self._lock:
            return self._finished


def run_task(
    parent,
    title: str,
    work: Callable[[Callable[[float, str], None]], Any],
    *,
    note: str = "",
    expected_seconds: float | None = None,
) -> tuple[Any, str | None]:
    """简写：开一个进度框，在后台跑 ``work``，等它结束（1.6.66）。"""
    if os.environ.get("BAMBUPALETTE_SYNC_TASKS"):
        # 排障/闸门用：不起线程、也不开框，直接同步跑一遍（只在显式设了环境变量时生效）。
        def _report(value: float, text: str = "") -> None:
            pump_events()

        try:
            return work(_report), None
        except Exception as error:  # noqa: BLE001 - 和后台路径保持同一种返回形状
            return None, f"{type(error).__name__}: {error}"
    return TaskRunner(parent, title, work, note=note, expected_seconds=expected_seconds).run()


def pump_events() -> None:
    """主线程里干重活时，把事件队列放一口气（进度框才不会冻住）。"""
    QApplication.processEvents()


def export_seconds(kind: str) -> float | None:
    """上一次同类导出花了几秒 —— 给进度框的倒计时当起点（1.6.66，m29882 第 4 条）。"""
    try:
        from ..core.settings import load_settings

        raw = load_settings().get("export_seconds")
    except Exception:  # noqa: BLE001 - 设置读不出来就不倒计时
        return None
    if not isinstance(raw, dict):
        return None
    value = raw.get(str(kind))
    if isinstance(value, (int, float)) and float(value) > 0:
        return float(value)
    return None


def remember_export_seconds(kind: str, seconds: float) -> None:
    """把这次实测用时记进设置（指数滑动平均），下次的倒计时更准（1.6.66）。"""
    try:
        from ..core.settings import load_settings, save_settings

        values = load_settings()
        table = values.get("export_seconds")
        table = dict(table) if isinstance(table, dict) else {}
        previous = table.get(str(kind))
        fresh = max(0.1, float(seconds))
        table[str(kind)] = (
            fresh
            if not isinstance(previous, (int, float)) or float(previous) <= 0
            else (float(previous) * 2.0 + fresh) / 3.0
        )
        values["export_seconds"] = table
        save_settings(values)
    except Exception:  # noqa: BLE001 - 记不下来不影响导出
        pass


class InlineProgress:
    """主线程里跑的活（匹配引擎）用这个：每次引擎报进度就顺手更新界面。"""

    def __init__(self, dialog: ProgressDialog) -> None:
        self.dialog = dialog

    def __call__(self, message: str, fraction: float) -> None:
        if message:
            self.dialog.set_note(message)
        self.dialog.set_fraction(fraction)
        QApplication.processEvents()


__all__ = [
    "ETA_WINDOW",
    "STALL_SECONDS",
    "InlineProgress",
    "ProgressDialog",
    "TaskRunner",
    "export_seconds",
    "pump_events",
    "remember_export_seconds",
    "run_task",
]

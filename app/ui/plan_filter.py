"""Which rolls a 耗材方案 may spend — the 「筛选」 按钮 of the 耗材方案 面板.

The search in :mod:`app.core.image_matching` honours two hard demands: rolls every
方案 has to use (「必须包含」) and rolls no 方案 may touch (「不要用」).  This dialog
is the surface for them.  One row per spool, and a click walks that row through
不要求 → 必须有 → 不要用, so both demands are set on one list instead of in two
places that could contradict each other.

Whether the demand is reasonable is the user's business — 「我必须用这几卷，因为
它们已经在 AMS 里」 is a fact about the print, not about the picture.  The one thing
the dialog does check is arithmetic it can see: more 「必须有」 rolls than 「耗材上限」
can never be satisfied, and it says so before the match runs.

The dialog only collects the answer; the page runs the match.  Closing it with
「取消」 leaves the filter exactly as it was.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

from ..core.library import Filament
from .swatch import swatch_icon

__all__ = ["SpoolFilterDialog"]

#: Row states, in the order a click walks them.
NEUTRAL = 0
REQUIRED = 1
FORBIDDEN = 2

#: Where the state lives on a row.  ``UserRole`` holds the filament id.
_STATE_ROLE = Qt.ItemDataRole.UserRole + 1
#: The label without the state prefix, so repainting never stacks prefixes.
_TEXT_ROLE = Qt.ItemDataRole.UserRole + 2


def _state_of(item: QListWidgetItem) -> int:
    value = item.data(_STATE_ROLE)
    return NEUTRAL if value is None else int(value)


class SpoolFilterDialog(QDialog):
    """「必须包含」 / 「不要用」, per spool."""

    def __init__(
        self,
        filaments: Sequence[Filament],
        require: Iterable[str] = (),
        avoid: Iterable[str] = (),
        *,
        cap: int = 0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("筛选耗材：哪几卷必须用、哪几卷不要用")
        # A row carries a name, a hex and a state word; the default width clips
        # 「不要用 · 大简 PETG HF 白」 on a 41-spool library.
        self.setMinimumWidth(460)
        self._cap = cap
        self._names: dict[str, str] = {}
        wanted = frozenset(require)
        banned = frozenset(avoid)

        hint = QLabel(
            "点一行切换：不要求 → 必须有 → 不要用。\n"
            "「必须有」的卷，每一套搭配都会用上；「不要用」的卷，哪一套都不碰。"
        )
        hint.setProperty("role", "hint")
        hint.setWordWrap(True)

        self._list = QListWidget()
        self._list.setUniformItemSizes(True)
        for filament in filaments:
            self._names[filament.id] = filament.display_name
            label = f"{filament.display_name}　{filament.color_hex.upper()}"
            item = QListWidgetItem(swatch_icon(filament.color_hex), "")
            item.setData(Qt.ItemDataRole.UserRole, filament.id)
            item.setData(_TEXT_ROLE, label)
            if filament.id in wanted:
                item.setData(_STATE_ROLE, REQUIRED)
            elif filament.id in banned:
                item.setData(_STATE_ROLE, FORBIDDEN)
            else:
                item.setData(_STATE_ROLE, NEUTRAL)
            self._paint(item)
            self._list.addItem(item)
        self._list.itemClicked.connect(self._on_row_clicked)

        self._summary = QLabel("")
        self._summary.setProperty("role", "hint")
        self._summary.setWordWrap(True)

        clear = QPushButton("全部不要求")
        clear.clicked.connect(self._on_clear)

        buttons = QDialogButtonBox()
        buttons.addButton("应用筛选", QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton("取消", QDialogButtonBox.ButtonRole.RejectRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        footer = QHBoxLayout()
        footer.setSpacing(6)
        footer.addWidget(clear)
        footer.addStretch(1)
        footer.addWidget(buttons)

        box = QVBoxLayout(self)
        box.setContentsMargins(12, 12, 12, 12)
        box.setSpacing(8)
        box.addWidget(hint)
        box.addWidget(self._list, 1)
        box.addWidget(self._summary)
        box.addLayout(footer)

        self._refresh_summary()

    # -- rows -------------------------------------------------------------
    def _paint(self, item: QListWidgetItem) -> None:
        """Show the state three ways: a word, bold, and a strike-through."""
        state = _state_of(item)
        label = str(item.data(_TEXT_ROLE))
        if state == REQUIRED:
            item.setText(f"必须有 · {label}")
        elif state == FORBIDDEN:
            item.setText(f"不要用 · {label}")
        else:
            item.setText(label)
        font = item.font()
        font.setBold(state == REQUIRED)
        font.setStrikeOut(state == FORBIDDEN)
        item.setFont(font)

    def _on_row_clicked(self, item: QListWidgetItem) -> None:
        item.setData(_STATE_ROLE, (_state_of(item) + 1) % 3)
        self._paint(item)
        self._refresh_summary()

    def _on_clear(self) -> None:
        for row in range(self._list.count()):
            item = self._list.item(row)
            item.setData(_STATE_ROLE, NEUTRAL)
            self._paint(item)
        self._refresh_summary()

    # -- answer -----------------------------------------------------------
    def _ids(self, state: int) -> tuple[str, ...]:
        """The filament ids in ``state``, in library order."""
        wanted: list[str] = []
        for row in range(self._list.count()):
            item = self._list.item(row)
            if _state_of(item) == state:
                wanted.append(str(item.data(Qt.ItemDataRole.UserRole)))
        return tuple(wanted)

    def required(self) -> tuple[str, ...]:
        """The rolls every 方案 must use."""
        return self._ids(REQUIRED)

    def forbidden(self) -> tuple[str, ...]:
        """The rolls no 方案 may use."""
        return self._ids(FORBIDDEN)

    def _refresh_summary(self) -> None:
        required = self.required()
        forbidden = self.forbidden()
        if not required and not forbidden:
            self._summary.setText("没有筛选：每一卷都可能被用上。")
            return
        parts = []
        if required:
            parts.append("必须有：" + "、".join(self._names[fid] for fid in required))
        if forbidden:
            parts.append("不要用：" + "、".join(self._names[fid] for fid in forbidden))
        text = "；".join(parts) + "。"
        if self._cap and len(required) > self._cap:
            text += (
                f"\n「耗材上限」是 {self._cap} 卷，「必须有」已经比上限多了，"
                "不会有搭配能满足——先改上限，或者少要求几卷。"
            )
        self._summary.setText(text)

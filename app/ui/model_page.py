"""模型配色页：打开一个 GLB，把它的颜色在全部颜色里配成能打印的混色。

界面照「图像转换」页的样子：左边两块三维视图（原模型 / 配色后），右边是配色结果，
底下一条状态行。三维视图的鼠标与按键照 Blender（见 :mod:`app.ui.model_view`）。

匹配本身没有另写一套：模型里出现的颜色按面积摊成一张色卡，交给
:func:`app.core.image_matching.match_image`，所以「相近色合并」「耗材上限」「方案
评分」和图像转换页是同一套判据。
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEventLoop, QSize, Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..core.atomicio import atomic_write_text
from ..core.glb import GlbModel, load_model, model_colours
from ..core.image_matching import MatchSettings, PaletteEntry, TriplePalette, build_palette
from ..core.library import FilamentLibrary
from ..core.mixes import MixCatalog
from ..core.model_matching import ModelMatch, match_model, summarise
from .model_view import ModelView
from .selectable import selectable_text
from .swatch import swatch_pixmap

__all__ = ["ModelPage", "MODEL_FILTER"]

MODEL_FILTER = "三维模型 (*.glb *.gltf);;所有文件 (*)"

#: 色卡的宽度（与 model_matching.CHART_WIDTH 一致），显示在工具栏上给用户一个参照。
CHART_WIDTH = 1024

PAGE_HELP = (
    "打开一个 GLB / glTF：程序读出它每一块的颜色，把这些颜色在「全部颜色」里配成能打印的"
    "混色。左边是原模型，右边是按配色结果换过色的预览，两个视图都能转。"
    "转视角：按住中键拖动（没有中键就用左键拖）；平移：Shift + 中键；缩放：滚轮；"
    "正 / 右 / 顶视图：小键盘 1 / 3 / 7（按住 Ctrl 是反面那三个），5 切换透视与正交，"
    "Home 自适应；右上角那几个按钮就是这几件事，不用记键。"
    "配色只管颜色，不会改动模型本身，也不会导出打印文件。"
)


class ModelPage(QWidget):
    """「模型配色」页。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._library: FilamentLibrary | None = None
        self._catalog: MixCatalog | None = None
        self._triples = None
        self._candidates: list[PaletteEntry] | None = None
        self._triple_palette: TriplePalette | None = None
        self._triple_head: list[PaletteEntry] | None = None
        self._model: GlbModel | None = None
        self._match: ModelMatch | None = None
        self._path: Path | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        layout.addLayout(self._build_toolbar())

        blurb = QLabel(PAGE_HELP)
        blurb.setProperty("role", "hint")
        blurb.setWordWrap(True)
        blurb.setMaximumHeight(56)
        layout.addWidget(blurb)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        views = QSplitter(Qt.Orientation.Horizontal)
        views.setChildrenCollapsible(False)
        self._source_view = ModelView("原模型")
        self._preview_view = ModelView("预览（配色后）")
        views.addWidget(self._source_view)
        views.addWidget(self._preview_view)
        views.setSizes([420, 420])
        splitter.addWidget(views)
        splitter.addWidget(self._build_side_panel())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([900, 380])
        layout.addWidget(splitter, 1)

        self._status = QLabel("打开一个 GLB 之后，这里会说明这次配色的结果。")
        self._status.setProperty("role", "hint")
        self._status.setWordWrap(True)
        selectable_text(self._status)
        layout.addWidget(self._status)

        self._set_enabled(False)

    # -- 界面 -------------------------------------------------------------
    def _build_toolbar(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        from PySide6.QtWidgets import QPushButton

        self._open_button = QPushButton("打开 GLB")
        self._open_button.setProperty("accent", True)
        self._open_button.setToolTip("打开一个 .glb / .gltf 文件（Ctrl+O）")
        self._open_button.clicked.connect(self._on_open_model)
        row.addWidget(self._open_button)

        self._rematch_button = QPushButton("重新匹配")
        self._rematch_button.setToolTip("用当前的颜色数量与耗材上限重新配一遍")
        self._rematch_button.clicked.connect(self._on_rematch)
        row.addWidget(self._rematch_button)

        row.addSpacing(8)
        row.addWidget(QLabel("颜色数量:"))
        self._colour_count = QSpinBox()
        self._colour_count.setRange(2, 64)
        self._colour_count.setValue(12)
        self._colour_count.setToolTip("模型最多配成多少种颜色")
        self._colour_count.valueChanged.connect(self._on_setting_changed)
        row.addWidget(self._colour_count)

        row.addWidget(QLabel("耗材上限:"))
        self._spool_cap = QSpinBox()
        self._spool_cap.setRange(0, 16)
        self._spool_cap.setValue(4)
        self._spool_cap.setToolTip("最多用几卷耗材；0 表示不限")
        self._spool_cap.valueChanged.connect(self._on_setting_changed)
        row.addWidget(self._spool_cap)

        self._export_button = QPushButton("导出配色表")
        self._export_button.setToolTip("把「哪种颜色配哪套混色」存成一个 Markdown 表")
        self._export_button.clicked.connect(self._on_export_table)
        row.addWidget(self._export_button)

        row.addStretch(1)
        self._model_label = QLabel("还没有打开模型")
        self._model_label.setProperty("role", "hint")
        row.addWidget(self._model_label)
        return row

    def _build_side_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(320)
        panel.setMaximumWidth(460)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        title = QLabel("颜色与配方")
        title.setProperty("role", "sectionTitle")
        layout.addWidget(title)

        self._summary = QLabel("还没有配色。")
        self._summary.setProperty("role", "hint")
        self._summary.setWordWrap(True)
        selectable_text(self._summary)
        layout.addWidget(self._summary)

        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._list.setIconSize(QSize(44, 20))
        self._list.setWordWrap(True)
        layout.addWidget(self._list, 1)
        return panel

    def _set_enabled(self, enabled: bool) -> None:
        for widget in (
            self._rematch_button,
            self._colour_count,
            self._spool_cap,
            self._export_button,
        ):
            widget.setEnabled(enabled)

    # -- 耗材库 -----------------------------------------------------------
    def setLibrary(self, library: FilamentLibrary | None, catalog: MixCatalog | None) -> None:
        renamed = library is not self._library
        self._library = library
        self._catalog = catalog
        self._candidates = None
        self._triple_palette = None
        self._triple_head = None
        if renamed and self._model is not None:
            self._on_rematch()

    def setTriples(self, triples, library: FilamentLibrary | None = None) -> None:
        self._triples = triples
        self._triple_palette = None
        self._triple_head = None
        if library is not None and library is not self._library:
            self._library = library
            self._candidates = None
        if self._model is not None:
            self._on_rematch()

    def _candidate_palette(self) -> list[PaletteEntry]:
        if self._candidates is None:
            self._candidates = build_palette(self._library, self._catalog, include_mixes=True)
        return self._candidates

    def _match_palette(self):
        head = self._candidate_palette()
        if self._triples is None:
            return head
        if self._triple_palette is None or self._triple_head is not head:
            self._triple_palette = TriplePalette(head, self._triples, self._library)
            self._triple_head = head
        return self._triple_palette

    def _settings(self) -> MatchSettings:
        return MatchSettings(
            max_colours=int(self._colour_count.value()),
            max_dimension=CHART_WIDTH,
            dither=False,
            max_spools=int(self._spool_cap.value()),
        ).clamped()

    # -- 打开与匹配 -------------------------------------------------------
    def _on_open_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "打开三维模型", "", MODEL_FILTER)
        if not path:
            return
        self.loadModel(path)

    def loadModel(self, source: str | Path) -> bool:
        """读一个模型。闸门脚本直接调这个方法，绕开文件对话框。"""
        path = Path(source)
        try:
            model = load_model(path)
        except (OSError, ValueError) as error:
            self._status.setText(f"这个文件读不出来：{error}")
            return False
        self._path = path
        self._model = model
        self._source_view.setModel(model)
        self._preview_view.setModel(model)
        self._model_label.setText(
            f"{model.name}：{model.triangle_count:,} 个三角形、{len(model.primitives)} 块"
        )
        self._set_enabled(not model.is_empty)
        if model.is_empty:
            self._status.setText("这个模型里没有三角形（点、线、面片之外的东西都会被跳过）。")
            self._list.clear()
            return False
        if model.skipped:
            self._status.setText("；".join(model.skipped)[:400])
        unavailable = self._source_view.available()
        if unavailable:
            self._status.setText(f"{unavailable}；配色照常可用，只是画不出三维预览。")
        return self._on_rematch()

    def _on_setting_changed(self) -> None:
        if self._model is not None:
            self._on_rematch()

    def _report(self, message: str, fraction: float) -> None:
        self._status.setText(f"正在配色：{message}…")
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

    def _on_rematch(self) -> bool:
        if self._model is None or self._model.is_empty:
            return False
        if self._library is None or len(self._library) == 0:
            self._status.setText("请先在「颜色配方」页添加耗材，再给模型配色。")
            return False
        colours = model_colours(self._model)
        if not colours:
            self._status.setText("这个模型里没有读得出颜色的面。")
            return False
        self._match = match_model(colours, self._match_palette(), self._settings(), progress=self._report)
        self._preview_view.setColours(self._preview_colours())
        self._fill_list()
        self._summary.setText(
            f"平均 ΔE00 {self._match.mean_delta_e:.2f}、最差 {self._match.worst_delta_e:.2f}，"
            f"用 {self._match.spool_count} 卷耗材。"
        )
        self._status.setText(
            summarise(self._match, self._spool_names())
            + "。左边的原模型是按模型自己的颜色画的，右边是配色后的预览。"
        )
        return True

    def _preview_colours(self) -> list[tuple[int, int, int]]:
        if self._model is None or self._match is None:
            return []
        by_rgb = {tuple(item.rgb): tuple(item.matched_rgb) for item in self._match.colours}
        out: list[tuple[int, int, int]] = []
        for primitive in self._model.primitives:
            pick = by_rgb.get(tuple(primitive.colour))
            if pick is None:
                for extra in primitive.extra_colours:
                    pick = by_rgb.get(tuple(extra))
                    if pick is not None:
                        break
            out.append(tuple(pick) if pick is not None else tuple(primitive.colour))
        return out

    def _spool_names(self) -> dict[str, str]:
        if self._library is None:
            return {}
        return {item.id: item.name for item in self._library.filaments}

    def _fill_list(self) -> None:
        self._list.clear()
        if self._match is None:
            return
        for item in self._match.colours:
            entry = item.entry
            swatch = swatch_pixmap(item.matched_hex, 44, 20)
            text = (
                f"{item.color_hex}  {item.share * 100:.1f}%  →  {item.matched_hex}"
                f"   ΔE00 {item.delta_e:.2f}"
            )
            row = QListWidgetItem(swatch, text)
            recipe = item.recipe
            row.setToolTip(f"{recipe}\n原色 {item.color_hex}，匹配色 {item.matched_hex}")
            row.setData(Qt.ItemDataRole.UserRole, item.color_hex)
            self._list.addItem(row)
            detail = QListWidgetItem(recipe)
            detail.setFlags(Qt.ItemFlag.NoItemFlags)
            detail.setForeground(Qt.GlobalColor.gray)
            self._list.addItem(detail)

    # -- 导出 -------------------------------------------------------------
    def _on_export_table(self) -> None:
        if self._match is None or self._model is None:
            return
        default = f"{self._model.name}-配色表.md"
        path, _ = QFileDialog.getSaveFileName(self, "导出配色表", default, "Markdown (*.md);;所有文件 (*)")
        if not path:
            return
        text = self.table_text()
        try:
            atomic_write_text(Path(path), text)
        except OSError as error:
            self._status.setText(f"写不进 {path}：{error}")
            return
        self._status.setText(f"配色表已写到 {path}。")

    def table_text(self) -> str:
        """导出的那份 Markdown（闸门脚本也读它）。"""
        if self._match is None or self._model is None:
            return ""
        lines = [
            f"# {self._model.name} 的配色表",
            "",
            f"- 三角形：{self._model.triangle_count:,} 个（{len(self._model.primitives)} 块）",
            f"- 颜色：{len(self._match.colours)} 种，平均 ΔE00 {self._match.mean_delta_e:.2f}、"
            f"最差 {self._match.worst_delta_e:.2f}",
            f"- 耗材：{self._match.spool_count} 卷",
            "",
            "| 原色 | 占比 | 匹配色 | 配方 | ΔE00 |",
            "| --- | --- | --- | --- | --- |",
        ]
        for item in self._match.colours:
            lines.append(
                f"| {item.color_hex} | {item.share * 100:.1f}% | {item.matched_hex} | "
                f"{item.recipe.replace('|', '/')} | {item.delta_e:.2f} |"
            )
        lines.append("")
        return "\n".join(lines)

    # -- 给主窗口 / 闸门用 ------------------------------------------------
    @property
    def model(self) -> GlbModel | None:
        return self._model

    @property
    def match(self) -> ModelMatch | None:
        return self._match

    def statusText(self) -> str:  # noqa: N802 - Qt 的命名习惯
        return self._status.text()

    def previewHexes(self) -> list[str]:
        """预览里每种原色被换成了哪个色（闸门用）。"""
        if self._match is None:
            return []
        return [item.matched_hex for item in self._match.colours]

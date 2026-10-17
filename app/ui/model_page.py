"""模型配色页：打开一个 GLB，把它的颜色在全部颜色里配成能打印的混色。

界面照「图像转换」页的样子：左边两块三维视图（原模型 / 配色后），右边是配色结果，
底下一条状态行。三维视图的鼠标与按键照 Blender（见 :mod:`app.ui.model_view`）。

匹配本身没有另写一套：模型里出现的颜色按面积摊成一张色卡，交给
:func:`app.core.image_matching.match_image`，所以「相近色合并」「耗材上限」「方案
评分」「筛选耗材」和图像转换页是同一套判据。

出口有三个（1.6.52）：3MF / OBJ + MTL 走 :mod:`app.core.model_export` 摊成的
「板」，GLB 则把几何原样搬、只把材质底色换成配到的颜色 —— 那个文件能再导入回来接着改。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QEventLoop, QSize, Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..core import paths
from ..core.glb import GlbModel, load_model, model_colours
from ..core.image_matching import MatchSettings, PaletteEntry, TriplePalette, build_palette
from ..core.library import FilamentLibrary
from ..core.mixes import MixCatalog
from ..core.model_export import plate_from_model, write_glb
from ..core.model_matching import ModelMatch, match_model, summarise
from ..mesh.objfile import write_obj
from ..mesh.threemf import slots_from_palette, write_3mf
from .colour_picker import ColourPickerDialog, entry_recipe_text
from .model_view import ModelView
from .plan_filter import SpoolFilterDialog
from .selectable import selectable_text
from .swatch import swatch_pixmap

__all__ = ["ModelPage", "MODEL_FILTER"]

MODEL_FILTER = (
    "三维模型 (*.glb *.gltf);;GLB (*.glb);;glTF (*.gltf);;所有文件 (*)"
)

#: 色卡的宽度（与 model_matching.CHART_WIDTH 一致），显示在工具栏上给用户一个参照。
CHART_WIDTH = 1024

#: 模型最多配成多少种颜色 —— 与图像转换页同一个默认值（m21172 item 7）。
DEFAULT_COLOURS = 32

#: 最多用几卷耗材（0 = 不限）。
DEFAULT_SPOOL_CAP = 4

PAGE_HELP = (
    "打开一个 GLB / glTF：程序读出它每一块的颜色，把这些颜色在「全部颜色」里配成能打印的"
    "混色。左边是原模型，右边是按配色结果换过色的预览，两个视图都能转。"
    "转视角：按住中键拖动（没有中键就用左键拖）；平移：Shift + 中键；缩放：滚轮；"
    "正 / 右 / 顶视图：小键盘 1 / 3 / 7（按住 Ctrl 是反面那三个），5 切换透视与正交，"
    "Home 自适应；右上角那几个按钮就是这几件事，不用记键。"
    "勾「只匹配耗材」就只用你自己的耗材本色，不拿混色去凑（可以跨材质）。"
    "右侧颜色列表里点一行，再按「更换颜色」就能手动换；「撤销 / 恢复」管这一步。"
    "底下三个出口：生成 3MF（Bambu Studio）、生成 OBJ + MTL、导出 GLB（Ctrl+S，能再导入回来接着改）。"
)


class ModelPage(QWidget):
    """「模型配色」页。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._library: FilamentLibrary | None = None
        self._catalog: MixCatalog | None = None
        self._triples = None
        self._candidates: list[PaletteEntry] | None = None
        self._candidates_plain: list[PaletteEntry] | None = None
        self._triple_palette: TriplePalette | None = None
        self._triple_head: list[PaletteEntry] | None = None
        self._require: frozenset[str] = frozenset()
        self._avoid: frozenset[str] = frozenset()
        self._model: GlbModel | None = None
        self._match: ModelMatch | None = None
        self._path: Path | None = None
        #: 手动换过的颜色：原色 hex → 换成的那个 :class:`PaletteEntry`（1.6.52）。
        self._overrides: dict[str, PaletteEntry] = {}
        self._undo: list[dict[str, PaletteEntry]] = []
        self._redo: list[dict[str, PaletteEntry]] = []

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

        layout.addLayout(self._build_bottom_row())

        self._status = QLabel("打开一个 GLB 之后，这里会说明这次配色的结果。")
        self._status.setProperty("role", "hint")
        self._status.setWordWrap(True)
        selectable_text(self._status)
        layout.addWidget(self._status)

        self._save_shortcut = QShortcut(QKeySequence.StandardKey.Save, self)
        self._save_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._save_shortcut.activated.connect(self._on_export_glb)

        self._set_enabled(False)

    # -- 界面 -------------------------------------------------------------
    def _build_toolbar(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        self._open_button = QPushButton("打开 GLB")
        self._open_button.setProperty("accent", True)
        self._open_button.setToolTip("打开一个 .glb / .gltf 文件（Ctrl+O）")
        self._open_button.clicked.connect(self._on_open_model)
        row.addWidget(self._open_button)

        self._rematch_button = QPushButton("重新匹配")
        self._rematch_button.setToolTip("用当前的颜色数量、耗材上限与筛选重新配一遍")
        self._rematch_button.clicked.connect(lambda: self._on_rematch())
        row.addWidget(self._rematch_button)

        row.addSpacing(8)
        row.addWidget(QLabel("颜色数量:"))
        self._colour_count = QSpinBox()
        self._colour_count.setRange(2, 64)
        self._colour_count.setValue(DEFAULT_COLOURS)
        self._colour_count.setToolTip("模型最多配成多少种颜色（默认 32，与图像转换页一致）")
        self._colour_count.valueChanged.connect(self._on_setting_changed)
        row.addWidget(self._colour_count)

        row.addWidget(QLabel("耗材上限:"))
        self._spool_cap = QSpinBox()
        self._spool_cap.setRange(0, 16)
        self._spool_cap.setValue(DEFAULT_SPOOL_CAP)
        self._spool_cap.setToolTip("最多用几卷耗材；0 表示不限")
        self._spool_cap.valueChanged.connect(self._on_setting_changed)
        row.addWidget(self._spool_cap)

        self._only_spools = QCheckBox("只匹配耗材")
        self._only_spools.setToolTip(
            "勾上就只用你自己这些耗材，不再拿两色 / 三色混色去凑（可以跨材质）。\n"
            "颜色会差一些，但每一块就是一卷料，Bambu Studio 里不用配比。"
        )
        self._only_spools.toggled.connect(self._on_only_spools)
        row.addWidget(self._only_spools)

        self._filter_button = QPushButton("筛选耗材")
        self._filter_button.setToolTip("指定必须有哪几卷、不许用哪几卷")
        self._filter_button.clicked.connect(self._on_filter)
        row.addWidget(self._filter_button)

        row.addStretch(1)
        self._model_label = QLabel("还没有打开模型")
        self._model_label.setProperty("role", "hint")
        row.addWidget(self._model_label)
        return row

    def _build_bottom_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        self._filter_hint = QLabel("")
        self._filter_hint.setProperty("role", "hint")
        self._filter_hint.setWordWrap(True)
        self._update_filter_hint()
        row.addWidget(self._filter_hint, 1)

        self._export_obj = QPushButton("生成 OBJ + MTL")
        self._export_obj.setToolTip("导出成 OBJ，颜色分在 MTL 里")
        self._export_obj.clicked.connect(lambda: self._on_export_plate("obj"))
        row.addWidget(self._export_obj)

        self._export_3mf = QPushButton("生成 3MF（Bambu Studio）")
        self._export_3mf.setProperty("accent", "true")
        self._export_3mf.setToolTip("导出成 Bambu Studio 工程，耗材按配色结果配好")
        self._export_3mf.clicked.connect(lambda: self._on_export_plate("3mf"))
        row.addWidget(self._export_3mf)

        self._export_glb_button = QPushButton("导出 GLB")
        self._export_glb_button.setToolTip("存成带配色的 GLB（Ctrl+S），下次导入回来接着改")
        self._export_glb_button.clicked.connect(self._on_export_glb)
        row.addWidget(self._export_glb_button)
        return row

    def _build_side_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(340)
        panel.setMaximumWidth(480)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        title = QLabel("模型上色块的颜色")
        title.setProperty("role", "sectionTitle")
        layout.addWidget(title)

        self._summary = QLabel("还没有配色。")
        self._summary.setProperty("role", "hint")
        self._summary.setWordWrap(True)
        selectable_text(self._summary)
        layout.addWidget(self._summary)

        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._list.setIconSize(QSize(44, 20))
        self._list.setWordWrap(True)
        self._list.currentRowChanged.connect(self._on_row_changed)
        layout.addWidget(self._list, 1)

        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        self._replace_button = QPushButton("更换颜色")
        self._replace_button.setToolTip("把选中的这一种颜色换成一瓶更合适的混色")
        self._replace_button.clicked.connect(self._on_replace)
        buttons.addWidget(self._replace_button)
        self._undo_button = QPushButton("撤销")
        self._undo_button.clicked.connect(self._on_undo)
        buttons.addWidget(self._undo_button)
        self._redo_button = QPushButton("恢复")
        self._redo_button.clicked.connect(self._on_redo)
        buttons.addWidget(self._redo_button)
        layout.addLayout(buttons)
        return panel

    def _set_enabled(self, enabled: bool) -> None:
        for widget in (
            self._rematch_button,
            self._colour_count,
            self._spool_cap,
            self._only_spools,
            self._filter_button,
            self._export_obj,
            self._export_3mf,
            self._export_glb_button,
            self._replace_button,
        ):
            widget.setEnabled(enabled)
        self._update_history_buttons()

    # -- 耗材库 -----------------------------------------------------------
    def setLibrary(self, library: FilamentLibrary | None, catalog: MixCatalog | None) -> None:
        renamed = library is not self._library
        self._library = library
        self._catalog = catalog
        self._candidates = None
        self._candidates_plain = None
        self._triple_palette = None
        self._triple_head = None
        self._update_filter_hint()
        if renamed and self._model is not None:
            self._on_rematch()

    def setTriples(self, triples, library: FilamentLibrary | None = None) -> None:
        self._triples = triples
        self._triple_palette = None
        self._triple_head = None
        if library is not None and library is not self._library:
            self._library = library
            self._candidates = None
            self._candidates_plain = None
        if self._model is not None:
            self._on_rematch()

    def _candidate_palette(self) -> list[PaletteEntry]:
        if self._only_spools.isChecked():
            if self._candidates_plain is None:
                self._candidates_plain = build_palette(
                    self._library, self._catalog, include_mixes=False
                )
            return self._candidates_plain
        if self._candidates is None:
            self._candidates = build_palette(self._library, self._catalog, include_mixes=True)
        return self._candidates

    def _match_palette(self):
        head = self._candidate_palette()
        if self._only_spools.isChecked() or self._triples is None:
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

    # -- 筛选 -------------------------------------------------------------
    def _spool_name(self, filament_id: str) -> str:
        if self._library is None:
            return filament_id
        filament = self._library.get(filament_id)
        return filament.display_name if filament is not None else filament_id

    def _update_filter_hint(self) -> None:
        parts = []
        if self._require:
            parts.append("必须有：" + "、".join(self._spool_name(fid) for fid in sorted(self._require)))
        if self._avoid:
            parts.append("不要用：" + "、".join(self._spool_name(fid) for fid in sorted(self._avoid)))
        if self._only_spools.isChecked():
            parts.append("只用耗材，不用混色")
        self._filter_hint.setText("；".join(parts) + "。" if parts else "每一卷都可以用。")

    def _on_filter(self) -> None:
        if self._model is None or self._library is None:
            return
        dialog = SpoolFilterDialog(
            list(self._library.visible),
            self._require,
            self._avoid,
            cap=self._spool_cap.value(),
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        require = frozenset(dialog.required())
        avoid = frozenset(dialog.forbidden())
        if require == self._require and avoid == self._avoid:
            return
        self._require = require
        self._avoid = avoid
        self._update_filter_hint()
        self._on_rematch(drop_edits=True)

    def _on_only_spools(self, checked: bool) -> None:
        self._candidates_plain = None if checked else self._candidates_plain
        self._triple_palette = None
        self._triple_head = None
        self._update_filter_hint()
        if self._model is not None:
            self._on_rematch(drop_edits=True)

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
        self._overrides = {}
        self._undo.clear()
        self._redo.clear()
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
            self._on_rematch(drop_edits=True)

    def _report(self, message: str, fraction: float) -> None:
        self._status.setText(f"正在配色：{message}…")
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

    def _on_rematch(self, *, drop_edits: bool = False) -> bool:
        if self._model is None or self._model.is_empty:
            return False
        if self._library is None or len(self._library) == 0:
            self._status.setText("请先在「颜色配方」页添加耗材，再给模型配色。")
            return False
        if drop_edits:
            self._overrides = {}
            self._undo.clear()
            self._redo.clear()
        colours = model_colours(self._model)
        if not colours:
            self._status.setText("这个模型里没有读得出颜色的面。")
            return False
        try:
            self._match = match_model(
                colours,
                self._match_palette(),
                self._settings(),
                require=self._require,
                avoid=self._avoid,
                progress=self._report,
            )
        except Exception as error:  # pragma: no cover - 防守
            self._status.setText(f"配不出来：{error}")
            return False
        self._refresh_preview()
        self._fill_list()
        self._summary.setText(
            f"平均 ΔE00 {self._match.mean_delta_e:.2f}、最差 {self._match.worst_delta_e:.2f}，"
            f"用 {self._match.spool_count} 卷耗材。"
        )
        self._status.setText(
            summarise(self._match, self._spool_names())
            + "。左边的原模型是按模型自己的颜色画的，右边是配色后的预览。"
        )
        self._update_history_buttons()
        return True

    # -- 换色 / 撤销 ---------------------------------------------------------
    def _shown_entry(self, colour_hex: str) -> PaletteEntry | None:
        pick = self._overrides.get(colour_hex)
        if pick is not None:
            return pick
        if self._match is None:
            return None
        for item in self._match.colours:
            if item.color_hex == colour_hex:
                return item.entry
        return None

    def _display_colours(self) -> dict[str, tuple[int, int, int]]:
        """原色 hex → 现在要画的那个 RGB（换过色的用换过的）。"""
        out: dict[str, tuple[int, int, int]] = {}
        if self._match is None:
            return out
        for item in self._match.colours:
            entry = self._overrides.get(item.color_hex) or item.entry
            rgb = tuple(int(value) for value in entry.rgb) if entry is not None else item.matched_rgb
            out[item.color_hex] = rgb
        return out

    def _preview_colours(self) -> list[tuple[int, int, int]]:
        if self._model is None or self._match is None:
            return []
        by_rgb = self._display_colours()
        out: list[tuple[int, int, int]] = []
        for primitive in self._model.primitives:
            pick = by_rgb.get(_hex(primitive.colour))
            if pick is None:
                for extra in primitive.extra_colours:
                    pick = by_rgb.get(_hex(extra))
                    if pick is not None:
                        break
            out.append(tuple(pick) if pick is not None else tuple(primitive.colour))
        return out

    def _refresh_preview(self) -> None:
        self._preview_view.setColours(self._preview_colours())

    def _selected_hex(self) -> str | None:
        row = self._list.currentItem()
        if row is None:
            return None
        return row.data(Qt.ItemDataRole.UserRole) or None

    def _on_row_changed(self) -> None:
        self._update_history_buttons()

    def _on_replace(self) -> None:
        colour_hex = self._selected_hex()
        if self._match is None or colour_hex is None:
            self._status.setText("先在右边点一行颜色，再按「更换颜色」。")
            return
        current = self._shown_entry(colour_hex)
        target_rgb = current.rgb if current is not None else (255, 255, 255)
        dialog = ColourPickerDialog(
            self._candidate_palette(),
            target_rgb,
            library=self._library,
            current_key=current.key if current is not None else None,
            parent=self,
            title="换成哪种颜色",
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        entry = dialog.chosen()
        if entry is None:
            return
        self._undo.append(dict(self._overrides))
        self._redo.clear()
        self._overrides[colour_hex] = entry
        self._refresh_preview()
        self._fill_list()
        self._status.setText(f"{colour_hex} 换成了 {entry.color_hex}（{entry_recipe_text(entry, self._library)}）。")
        self._update_history_buttons()

    def _on_undo(self) -> None:
        if not self._undo:
            return
        self._redo.append(dict(self._overrides))
        self._overrides = self._undo.pop()
        self._refresh_preview()
        self._fill_list()
        self._status.setText("撤销上一次换色。")
        self._update_history_buttons()

    def _on_redo(self) -> None:
        if not self._redo:
            return
        self._undo.append(dict(self._overrides))
        self._overrides = self._redo.pop()
        self._refresh_preview()
        self._fill_list()
        self._status.setText("恢复刚才撤销的换色。")
        self._update_history_buttons()

    def _update_history_buttons(self) -> None:
        self._undo_button.setEnabled(bool(self._undo))
        self._redo_button.setEnabled(bool(self._redo))
        self._replace_button.setEnabled(self._match is not None and bool(self._match.colours))

    def _spool_names(self) -> dict[str, str]:
        if self._library is None:
            return {}
        return {item.id: item.name for item in self._library.filaments}

    def _fill_list(self) -> None:
        keep = self._selected_hex()
        self._list.blockSignals(True)
        self._list.clear()
        if self._match is None:
            self._list.blockSignals(False)
            return
        shown = self._display_colours()
        for item in self._match.colours:
            entry = self._overrides.get(item.color_hex) or item.entry
            matched_hex = entry.color_hex if entry is not None else item.matched_hex
            rgb = shown.get(item.color_hex, item.matched_rgb)
            recipe = entry_recipe_text(entry, self._library) if entry is not None else item.recipe
            delta = _delta(item.rgb, rgb)
            text = (
                f"{item.color_hex}  {item.share * 100:.1f}%  →  {_hex(rgb)}   ΔE00 {delta:.2f}"
            )
            row = QListWidgetItem(swatch_pixmap(_hex(rgb), 44, 20), text)
            row.setToolTip(f"{recipe}\n原色 {item.color_hex}，现在画成 {_hex(rgb)}")
            row.setData(Qt.ItemDataRole.UserRole, item.color_hex)
            self._list.addItem(row)
            detail = QListWidgetItem(recipe)
            detail.setFlags(Qt.ItemFlag.NoItemFlags)
            detail.setForeground(Qt.GlobalColor.gray)
            detail.setData(Qt.ItemDataRole.UserRole, None)
            self._list.addItem(detail)
        self._list.blockSignals(False)
        if keep is not None:
            for index in range(self._list.count()):
                row = self._list.item(index)
                if row.data(Qt.ItemDataRole.UserRole) == keep:
                    self._list.setCurrentRow(index)
                    break
        self._update_history_buttons()

    # -- 导出 -------------------------------------------------------------
    # 「导出配色表」在 1.6.52 里按用户要求删了（m21496 第 9 条）：那份 Markdown 说的
    # 事，右侧列表和底下的 3MF / OBJ / GLB 三个出口已经能表达。

    def _plate_payload(self):
        """摊成 PlateModel，同时给出它引用的调色板（顺序与 palette_index 对应）。"""
        if self._model is None or self._match is None:
            raise ValueError("还没有配色。")
        shown = self._display_colours()
        entries: list[PaletteEntry] = []
        slots: dict[str, int] = {}
        colours: list[tuple[int, int, int]] = []
        indices: list[int] = []
        slot_numbers: list[int] = []
        for primitive in self._model.primitives:
            key = _hex(primitive.colour)
            for extra in primitive.extra_colours:
                if key in shown:
                    break
                if _hex(extra) in shown:
                    key = _hex(extra)
                    break
            rgb = shown.get(key, tuple(primitive.colour))
            entry = self._shown_entry(key)
            colours.append(rgb)
            if entry is None:
                indices.append(0)
                slot_numbers.append(1)
                continue
            if entry.color_hex not in slots:
                slots[entry.color_hex] = len(entries)
                entries.append(entry)
            indices.append(slots[entry.color_hex])
            slot_numbers.append(slots[entry.color_hex] + 1)
        if not entries:
            raise ValueError("这个模型里没有能落到耗材上的颜色。")
        plate = plate_from_model(
            self._model,
            colours,
            slots=slot_numbers,
            palette_indices=indices,
        )
        return plate, entries

    def _on_export_plate(self, kind: str) -> None:
        if self._model is None or self._match is None:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            plate, entries = self._plate_payload()
        except Exception as error:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "无法生成模型", str(error))
            return
        QApplication.restoreOverrideCursor()
        suffix = ".3mf" if kind == "3mf" else ".obj"
        label = "3MF 模型" if kind == "3mf" else "OBJ 模型"
        suggested = f"{self._model.name}_{datetime.now():%Y%m%d-%H%M%S}{suffix}"
        try:
            start = str(paths.exports_dir() / suggested)
        except OSError:  # pragma: no cover - 防守
            start = suggested
        name, _ = QFileDialog.getSaveFileName(self, f"保存{label}", start, f"{label} (*{suffix})")
        if not name:
            return
        target = Path(name)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self._status.setText(f"正在写入 {target.name}…")
        QApplication.processEvents()
        try:
            if kind == "3mf":
                written = write_3mf(
                    plate,
                    target,
                    object_name=self._model.name,
                    filaments=slots_from_palette(entries, self._library),
                )
            else:
                written = write_obj(plate, target)
        except Exception as error:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "保存失败", str(error))
            self._status.setText(f"写不进 {target.name}：{error}")
            return
        QApplication.restoreOverrideCursor()
        stats = plate.stats()
        self._status.setText(
            f"已生成 {written.name}：{stats['width_mm']} × {stats['depth_mm']} mm，"
            f"{stats['triangles']:,} 个三角面。"
        )

    def _on_export_glb(self) -> None:
        """导出带配色的 GLB（Ctrl+S）。"""
        if self._model is None or self._match is None:
            return
        suggested = f"{self._model.name}-配色.glb"
        try:
            start = str(paths.exports_dir() / suggested)
        except OSError:  # pragma: no cover - 防守
            start = suggested
        name, _ = QFileDialog.getSaveFileName(self, "导出 GLB", start, "GLB (*.glb)")
        if not name:
            return
        colours = self._preview_colours()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            written = write_glb(self._model, colours, name)
        except Exception as error:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "保存失败", str(error))
            self._status.setText(f"写不进 {name}：{error}")
            return
        QApplication.restoreOverrideCursor()
        self._status.setText(
            f"已导出 {written.name}：{len(colours)} 块材质换成了配色结果，再导入回来就能接着改。"
        )

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
        shown = self._display_colours()
        if self._match is None:
            return []
        return [_hex(shown.get(item.color_hex, item.matched_rgb)) for item in self._match.colours]

    def override_of(self, colour_hex: str) -> str | None:
        """这一种颜色手动换成了什么（闸门用；没换过给 None）。"""
        entry = self._overrides.get(colour_hex)
        return entry.color_hex if entry is not None else None

    def replace_colour(self, colour_hex: str, entry: PaletteEntry) -> None:
        """手动换色（闸门用，省掉对话框）。"""
        self._undo.append(dict(self._overrides))
        self._redo.clear()
        self._overrides[colour_hex] = entry
        self._refresh_preview()
        self._fill_list()
        self._update_history_buttons()

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._on_undo()
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._on_redo()
        return True

    def plate(self):
        """摊好的板（闸门用；没配色时给 None）。"""
        if self._model is None or self._match is None:
            return None
        plate, _entries = self._plate_payload()
        return plate

    def glb_payload(self) -> bytes:
        """导出会写出的那些字节（闸门用，不碰磁盘）。"""
        from ..core.model_export import glb_bytes

        if self._model is None:
            raise ValueError("还没有打开模型。")
        return glb_bytes(self._model, self._preview_colours())


def _hex(rgb) -> str:
    return "#{:02X}{:02X}{:02X}".format(*(int(max(0, min(255, round(c)))) for c in rgb))


def _delta(a, b) -> float:
    from ..spectral.color import delta_e_2000, lab_from_rgb

    return float(delta_e_2000(lab_from_rgb(tuple(a)), lab_from_rgb(tuple(b))))

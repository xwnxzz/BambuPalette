"""The application window.

Layout follows the user's workflow: their spools on the left, every predicted
mix in the middle (81 per pair), and — once a mix is clicked — the recipe that
produced it on the right, ready to be re-created inside Bambu Studio.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QAction, QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..core import paths, settings
from ..core.engines import DEFAULT_ENGINE, ENGINE_CHOICES, get_engine
from ..core.library import (
    COMMON_BRANDS,
    COMMON_MATERIAL_TYPES,
    Filament,
    FilamentLibrary,
    LibraryError,
)
from ..core.mixes import (
    MIX_RATIOS,
    SORT_CHOICES,
    SORT_PAIR,
    SORT_RGB,
    MixCatalog,
    sorted_cells,
)
from ..spectral import color as _color
from . import theme
from .filament_dialog import FilamentDialog
from .mix_detail import MixDetail
from .mix_grid import MixGrid, spool_cell
from .picture_page import PicturePage
from .swatch import ColorButton, swatch_icon

_HEX_QUERY = re.compile(r"^[0-9a-fA-F]{1,6}$")


class MainWindow(QMainWindow):
    def __init__(self, library: FilamentLibrary | None = None, parent=None) -> None:
        super().__init__(parent)
        self.library = library if library is not None else self._load_library()
        self._catalog: MixCatalog | None = None
        self._recipes: list = []
        self._pair_filter: tuple[str, str] | None = None
        self._build_seconds = 0.0
        self._suppress_list_signal = False

        self.setWindowTitle("BambuPalette — 混色耗材色彩管理器")
        self.resize(1320, 840)
        self.setMinimumSize(1020, 640)

        self._build_ui()
        self._build_menus()
        self._reload_library()

    # -- construction ------------------------------------------------------------
    def _load_library(self) -> FilamentLibrary:
        try:
            return FilamentLibrary.load()
        except LibraryError as exc:
            QTimer.singleShot(0, lambda: QMessageBox.warning(self, "读取耗材档案失败", str(exc)))
            return FilamentLibrary()

    def _build_ui(self) -> None:
        title = QLabel("BambuPalette")
        title.setStyleSheet(f"font-size: 14px; font-weight: 650; color: {theme.TEXT};")
        subtitle = QLabel("混色耗材色彩管理器 · 每两种耗材 81 个配比")
        subtitle.setProperty("role", "hint")

        head = QVBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(1)
        head.addWidget(title)
        head.addWidget(subtitle)

        self._add_button = QPushButton("添加耗材")
        self._add_button.setProperty("accent", "true")
        self._add_button.clicked.connect(self._on_add)

        # Lumina Studio's topbar: the brand block on the left, the primary action
        # on the right, on a white bar closed by a hairline.
        topbar = QFrame()
        topbar.setObjectName("topbar")
        head_row = QHBoxLayout(topbar)
        head_row.setContentsMargins(16, 10, 16, 10)
        head_row.setSpacing(10)
        head_row.addLayout(head, 1)
        head_row.addWidget(self._add_button, 0, Qt.AlignmentFlag.AlignVCenter)

        # engine / sort / search controls
        self._engine_combo = QComboBox()
        for engine_id, label in ENGINE_CHOICES:
            self._engine_combo.addItem(label, engine_id)
        remembered = settings.load_settings().get("engine", DEFAULT_ENGINE)
        index = self._engine_combo.findData(remembered)
        if index < 0:
            index = self._engine_combo.findData(DEFAULT_ENGINE)
        self._engine_combo.setCurrentIndex(max(0, index))
        self._engine_combo.setToolTip(
            "Bambu 混色预览（默认）：与 Bambu Studio 2.8（02.08.x）「添加混色耗材」"
            "对话框一致，用的是 Bambu 内置的 filament_mixer 颜料混色多项式。\n"
            "Bambu 2.5 旧版预览：2.5.x 及更早用的是 8 位 sRGB 加权平均，"
            "如果你的 Studio 还是旧版，选这一项才对得上。\n"
            "光谱 KM 模型：用 Kubelka-Munk 光谱混合预测，更接近真实打印，但与 Studio 的预览值不同。\n"
            "选择会被记住，下次启动保持不变。"
        )
        self._engine_combo.currentIndexChanged.connect(self._on_engine_changed)

        self._sort_combo = QComboBox()
        for sort_id, label in SORT_CHOICES:
            self._sort_combo.addItem(label, sort_id)
        self._sort_combo.currentIndexChanged.connect(lambda *_: self._refresh_grid())

        self._search = QLineEdit()
        self._search.setPlaceholderText("搜索：RGB 片段如 3D7B，或耗材名称 / 品牌 / 种类")
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(lambda *_: self._refresh_grid())

        self._show_all_button = QPushButton("显示全部混色")
        self._show_all_button.clicked.connect(self._clear_filters)
        self._show_all_button.setEnabled(False)

        # Deliberately UNSET: the target is whatever the user wants to print, so
        # showing a made-up colour here would be a lie until they pick one.
        self._target_color = ColorButton()
        self._target_color.setToolTip("想打印出来的目标颜色；点击后选择，程序会找出最接近的混色配方")
        self._target_color.colorChanged.connect(self._on_target_color_changed)
        self._find_button = QPushButton("找最接近的混色")
        self._find_button.setToolTip("先选一个目标颜色；再按 CIEDE2000 在当前显示的全部混色里找最接近的配方")
        self._find_button.setEnabled(False)
        self._find_button.clicked.connect(self._on_find_nearest)

        controls = QHBoxLayout()
        controls.setSpacing(8)
        controls.addWidget(QLabel("混色模型:"))
        controls.addWidget(self._engine_combo)
        controls.addWidget(QLabel("排序:"))
        controls.addWidget(self._sort_combo)
        controls.addWidget(self._search, 1)
        controls.addWidget(self._show_all_button)

        target_row = QHBoxLayout()
        target_row.setSpacing(8)
        target_row.addWidget(QLabel("目标颜色:"))
        target_row.addWidget(self._target_color)
        target_row.addWidget(self._find_button)
        target_row.addStretch(1)
        target_row.addWidget(QLabel("双击某个混色 = 只看这一对耗材"))

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_library_panel())
        splitter.addWidget(self._build_grid_panel())
        self._detail = MixDetail()
        self._detail.pairRequested.connect(self._show_pair)
        splitter.addWidget(self._wrap(self._detail, ""))
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([300, 640, 400])

        blurb = QLabel(
            "输入你自己的耗材颜色，程序自动算出所有两两混色的 81 个配比；"
            "点击任意一个混色，就知道它由哪两种耗材丝、按什么比例混成，方便直接在 Bambu Studio 里配置。"
        )
        blurb.setProperty("role", "hint")
        blurb.setWordWrap(True)

        mix_page = QWidget()
        mix_layout = QVBoxLayout(mix_page)
        mix_layout.setContentsMargins(0, 0, 0, 0)
        mix_layout.setSpacing(10)
        mix_layout.addWidget(blurb)
        mix_layout.addLayout(controls)
        mix_layout.addLayout(target_row)
        mix_layout.addWidget(splitter, 1)

        self._picture_page = PicturePage()

        tabs = QTabWidget()
        tabs.addTab(mix_page, "混色配方")
        tabs.addTab(self._picture_page, "图片转模型")
        self._tabs = tabs

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(16, 12, 16, 10)
        layout.setSpacing(10)
        layout.addWidget(tabs, 1)

        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(topbar)
        outer.addWidget(body, 1)
        self.setCentralWidget(central)

        self._status = QLabel("")
        self._status.setProperty("role", "hint")
        self.statusBar().addWidget(self._status)

    def _wrap(self, widget: QWidget, title: str, header_extra: QWidget | None = None) -> QWidget:
        frame = QFrame()
        frame.setObjectName("panel")
        header = QLabel(title)
        header.setProperty("role", "sectionTitle")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(8)
        if title:
            title_row = QHBoxLayout()
            title_row.setContentsMargins(0, 0, 0, 0)
            title_row.setSpacing(8)
            title_row.addWidget(header)
            title_row.addStretch(1)
            if header_extra is not None:
                title_row.addWidget(header_extra)
            layout.addLayout(title_row)
        layout.addWidget(widget, 1)
        return frame

    def _build_library_panel(self) -> QWidget:
        self._list = QListWidget()
        self._list.setIconSize(QSize(30, 30))
        self._list.setUniformItemSizes(True)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self._list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._on_list_menu)
        self._list.itemDoubleClicked.connect(lambda *_: self._on_edit())
        self._list.currentItemChanged.connect(self._on_list_selection)

        self._edit_button = QPushButton("编辑")
        self._edit_button.clicked.connect(self._on_edit)
        self._delete_button = QPushButton("删除")
        self._delete_button.clicked.connect(self._on_delete)
        self._import_button = QPushButton("导入")
        self._import_button.setToolTip("从 JSON 文件导入耗材档案（可与他人共享）")
        self._import_button.clicked.connect(self._on_import)
        self._export_button = QPushButton("导出")
        self._export_button.setToolTip("把耗材档案导出成 JSON 文件")
        self._export_button.clicked.connect(self._on_export)

        row_one = QHBoxLayout()
        row_one.setSpacing(6)
        row_one.addWidget(self._edit_button)
        row_one.addWidget(self._delete_button)
        row_two = QHBoxLayout()
        row_two.setSpacing(6)
        row_two.addWidget(self._import_button)
        row_two.addWidget(self._export_button)

        self._library_hint = QLabel(
            "还没有耗材。点「添加耗材」输入颜色、种类和品牌；"
            "至少两种同种类的耗材就能开始混色。"
        )
        self._library_hint.setProperty("role", "hint")
        self._library_hint.setWordWrap(True)

        self._sample_button = QPushButton("载入示例耗材")
        self._sample_button.clicked.connect(self._on_sample)

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._list, 1)
        layout.addWidget(self._library_hint)
        layout.addWidget(self._sample_button)
        layout.addLayout(row_one)
        layout.addLayout(row_two)

        # Delete reaches the spool list from the keyboard too. The shortcut is
        # scoped to the list (WidgetWithChildrenShortcut) rather than the window,
        # so it cannot fire while the user is editing a hex field or typing a
        # search query elsewhere on the page.
        self._delete_shortcut = QShortcut(QKeySequence.StandardKey.Delete, self._list)
        self._delete_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._delete_shortcut.activated.connect(self._on_delete)
        self._backspace_shortcut = QShortcut(QKeySequence.StandardKey.Backspace, self._list)
        self._backspace_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._backspace_shortcut.activated.connect(self._on_delete)

        return self._wrap(panel, "我的耗材")

    def _build_grid_panel(self) -> QWidget:
        self._grid = MixGrid()
        self._grid.recipeSelected.connect(self._on_grid_selected)
        self._grid.recipeActivated.connect(self._on_grid_activated)
        self._grid.selectionCleared.connect(self._on_grid_cleared)
        self._grid_info = QLabel("")
        self._grid_info.setProperty("role", "hint")

        # The spool colours are a separate, optional half of the same list: in
        # Bambu Studio a mixed filament and the plain spool it is made from sit
        # side by side in the same picker, so a user choosing what to load needs
        # to see both. Off by default keeps the page about the mixes.
        self._all_colours = QCheckBox("全部颜色")
        self._all_colours.setToolTip(
            "把每种耗材丝本身的颜色也列进来。\n"
            "两种耗材时：81 个混色 + 2 个耗材本色 = 83 个颜色。\n"
            "列表顺序与左边的「排序」一致。"
        )
        self._all_colours.toggled.connect(lambda *_: self._refresh_grid())

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._grid, 1)
        layout.addWidget(self._grid_info)
        return self._wrap(panel, "全部混色", header_extra=self._all_colours)

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("文件(&F)")
        for text, slot, shortcut in (
            ("添加耗材…", self._on_add, "Ctrl+N"),
            ("导入耗材档案…", self._on_import, None),
            ("导出耗材档案…", self._on_export, None),
        ):
            action = QAction(text, self)
            if shortcut:
                action.setShortcut(QKeySequence(shortcut))
            action.triggered.connect(slot)
            file_menu.addAction(action)
        file_menu.addSeparator()
        quit_action = QAction("退出", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        view_menu = self.menuBar().addMenu("视图(&V)")
        clear_action = QAction("显示全部混色", self)
        clear_action.setShortcut(QKeySequence("Ctrl+0"))
        clear_action.triggered.connect(self._clear_filters)
        view_menu.addAction(clear_action)

        help_menu = self.menuBar().addMenu("帮助(&H)")
        about_action = QAction("关于与算法来源", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    # -- library -----------------------------------------------------------------
    def _reload_library(self) -> None:
        self._suppress_list_signal = True
        self._list.clear()
        for filament in self.library:
            # An unnamed spool's display_name IS its hex, so printing both would
            # render "#000000   #000000". The one other thing the user needs is
            # the numbers, so the duplicate becomes "RGB 0, 0, 0".
            detail = filament.color_hex
            if filament.display_name.strip().casefold() == filament.color_hex.casefold():
                red, green, blue = filament.rgb
                detail = f"RGB {red}, {green}, {blue}"
            item = QListWidgetItem(
                swatch_icon(filament.color_hex, 28, 28),
                f"{filament.display_name}   {detail}",
            )
            item.setData(Qt.ItemDataRole.UserRole, filament.id)
            item.setToolTip(
                f"名称：{filament.display_name}\n"
                f"颜色：{filament.color_hex}\n"
                f"品牌：{filament.brand or '—'}\n"
                f"种类：{filament.material_type or '—'}\n"
                f"备注：{filament.note or '—'}"
            )
            if self._pair_filter and filament.id in self._pair_filter:
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            self._list.addItem(item)
        self._suppress_list_signal = False

        has_any = len(self.library) > 0
        self._library_hint.setVisible(not has_any)
        self._sample_button.setVisible(not has_any)
        self._list.setVisible(has_any)
        self._rebuild_catalog()
        self._save_library()

    def _save_library(self) -> None:
        try:
            self.library.save(paths.library_path())
        except (LibraryError, OSError) as exc:
            # OSError covers DataDirectoryError: no usable location at all.
            self._status.setText(f"保存失败：{exc}")

    def _selected_filament(self) -> Filament | None:
        item = self._list.currentItem()
        if item is None:
            return None
        return self.library.get(item.data(Qt.ItemDataRole.UserRole))

    def _on_add(self) -> None:
        dialog = FilamentDialog(
            None,
            brands=[f.brand for f in self.library if f.brand],
            types=[f.material_type for f in self.library if f.material_type],
            parent=self,
        )
        if dialog.exec() != FilamentDialog.DialogCode.Accepted:
            return
        filament = dialog.build_filament()
        duplicate = self.library.find_duplicate(filament.color_hex, filament.material_type, filament.brand)
        if duplicate is not None:
            answer = QMessageBox.question(
                self,
                "已经存在同样的耗材",
                f"{duplicate.display_name}（{duplicate.color_hex}）看起来和这条一样，仍然要添加吗？",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.library.add(filament)
        self._reload_library()
        self._select_filament(filament.id)
        self._status.setText(f"已添加 {filament.display_name}（{filament.color_hex}）")

    def _on_edit(self) -> None:
        filament = self._selected_filament()
        if filament is None:
            return
        dialog = FilamentDialog(
            filament,
            brands=[f.brand for f in self.library if f.brand],
            types=[f.material_type for f in self.library if f.material_type],
            parent=self,
        )
        if dialog.exec() != FilamentDialog.DialogCode.Accepted:
            return
        self.library.update(filament.id, **dialog.values())
        self._reload_library()
        self._select_filament(filament.id)

    def _on_delete(self) -> None:
        """Delete the selected spool immediately, with no confirmation.

        The user asked for this explicitly: the recipe library is a scratchpad,
        a spool is two keystrokes to re-add, and a Yes/No box whose buttons are
        English in a Chinese window is worse than no box at all. The duplicate
        warning in :meth:`_on_add` stays, because that one prevents a mistake
        rather than confirming an intention.
        """
        filament = self._selected_filament()
        if filament is None:
            return
        self.library.remove(filament.id)
        if self._pair_filter and filament.id in self._pair_filter:
            self._pair_filter = None
        self._detail.clear()
        self._status.setText(f"已删除 {filament.display_name}（{filament.color_hex}）")
        self._reload_library()

    def _select_filament(self, filament_id: str) -> None:
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == filament_id:
                self._list.setCurrentItem(item)
                return

    def _on_list_selection(self, current, previous) -> None:
        if self._suppress_list_signal or current is None:
            return
        filament = self.library.get(current.data(Qt.ItemDataRole.UserRole))
        if filament is not None:
            self._status.setText(f"{filament.display_name} · {filament.color_hex} · "
                                 f"{filament.brand or '—'} · {filament.material_type or '—'}")

    def _on_list_menu(self, position) -> None:
        from PySide6.QtWidgets import QMenu

        filament = self._selected_filament()
        if filament is None:
            return
        menu = QMenu(self)
        menu.addAction("编辑…", self._on_edit)
        menu.addAction("复制一条", lambda: self._on_duplicate(filament))
        menu.addSeparator()
        menu.addAction("删除", self._on_delete)
        menu.exec(self._list.mapToGlobal(position))

    def _on_duplicate(self, filament: Filament) -> None:
        copy = filament.copy(name=(filament.name or filament.display_name) + " 副本")
        self.library.add(copy)
        self._reload_library()
        self._select_filament(copy.id)

    def _on_sample(self) -> None:
        for name, brand, material, hex_value in (
            ("示例白", "示例", "PLA Basic", "#FFFFFF"),
            ("示例黑", "示例", "PLA Basic", "#16161A"),
            ("示例红", "示例", "PLA Basic", "#D8342C"),
            ("示例蓝", "示例", "PLA Basic", "#1D64C8"),
        ):
            if self.library.find_duplicate(hex_value, material, brand) is None:
                self.library.add(Filament(name=name, brand=brand, material_type=material, color_hex=hex_value))
        self._reload_library()

    def _on_import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "导入耗材档案", str(paths.default_start_dir()), "JSON 文件 (*.json)")
        if not path:
            return
        try:
            incoming = FilamentLibrary.load(Path(path))
        except LibraryError as exc:
            QMessageBox.warning(self, "导入失败", str(exc))
            return
        added = 0
        for filament in incoming:
            if self.library.get(filament.id) is not None:
                filament = filament.copy()
            self.library.add(filament)
            added += 1
        self._reload_library()
        QMessageBox.information(self, "导入完成", f"已导入 {added} 条耗材。")

    def _on_export(self) -> None:
        default = paths.default_start_dir() / "filament-library-export.json"
        path, _ = QFileDialog.getSaveFileName(self, "导出耗材档案", str(default), "JSON 文件 (*.json)")
        if not path:
            return
        try:
            self.library.save(Path(path))
        except LibraryError as exc:
            QMessageBox.warning(self, "导出失败", str(exc))
            return
        QMessageBox.information(self, "导出完成", f"已写入\n{path}")

    # -- catalogue ---------------------------------------------------------------
    def _rebuild_catalog(self) -> None:
        started = time.perf_counter()
        self._catalog = MixCatalog(self.library.filaments, MIX_RATIOS, engine=self._engine_combo.currentData())
        self._catalog.build()
        self._build_seconds = time.perf_counter() - started
        self._detail.setContext(self.library, self._catalog.engine)
        if getattr(self, "_picture_page", None) is not None:
            self._picture_page.setLibrary(self.library, self._catalog)
        if self._pair_filter is not None:
            a_id, b_id = self._pair_filter
            if self.library.get(a_id) is None or self.library.get(b_id) is None:
                self._pair_filter = None
        self._refresh_grid()

    def _on_engine_changed(self, *args) -> None:
        try:
            settings.save_settings({"engine": self._engine_combo.currentData()})
        except OSError:
            pass  # a read-only data directory must not block a rebuild
        self._rebuild_catalog()

    def _label_of(self, filament_id: str) -> str:
        """Case-folded display name of a spool, for 「按名称」.

        Shared by the mixes and the raw spool rows so that a single sorted list
        compares like with like: an unknown id falls back to the id itself, which
        keeps the sort total instead of raising.
        """
        filament = self.library.get(filament_id)
        return filament.display_name.casefold() if filament is not None else filament_id

    def _matches(self, cell, query: str) -> bool:
        """Does one grid cell survive the search box?

        A spool cell is matched on its own name/brand/type/note; a mix cell also
        on both of its parents, so searching a spool name still finds every mix
        made from it.
        """
        if getattr(cell, "pair_index", 0) < 0:
            filament = cell.filament
            haystack = " ".join(
                (
                    filament.display_name,
                    filament.brand,
                    filament.material_type,
                    filament.note,
                )
            ).casefold()
            if _HEX_QUERY.match(query):
                return query.lower() in filament.color_hex[1:].lower()
            return query in haystack
        if _HEX_QUERY.match(query):
            return query.lower() in cell.color_hex[1:].lower()
        for filament_id in (cell.a_id, cell.b_id):
            filament = self.library.get(filament_id)
            if filament is None:
                continue
            haystack = " ".join(
                (filament.display_name, filament.brand, filament.material_type, filament.note)
            ).casefold()
            if query in haystack:
                return True
        return False

    def _refresh_grid(self) -> None:
        if self._catalog is None:
            return

        sort_key = self._sort_combo.currentData() or SORT_RGB
        if self._pair_filter is not None:
            recipes = self._catalog.pair_recipes(*self._pair_filter)
            grouped = False
        elif self._catalog.recipe_count:
            recipes = self._catalog.sorted_recipes(sort_key)
            grouped = sort_key == SORT_PAIR
        else:
            # Fewer than two spools: there are no mixes, but 「全部颜色」 can still
            # have something to show.  Bailing out here was why a single spool
            # left the middle panel empty while the footer counted its colour.
            recipes = []
            grouped = False

        # 「全部颜色」 adds the spools themselves, so a two-spool library offers
        # 81 mixes + 2 raw colours = 83 entries. The spools are NOT pinned on top:
        # they are folded into the very same order the 排序 control just gave the
        # mixes, because a colour list that breaks its own colour order the moment
        # the checkbox is ticked is not sorted at all.
        spools = []
        if self._all_colours.isChecked():
            spools = [spool_cell(f) for f in self.library.filaments]

        query = self._search.text().strip().lstrip("#").casefold()
        if query:
            recipes = [recipe for recipe in recipes if self._matches(recipe, query)]
            spools = [cell for cell in spools if self._matches(cell, query)]
            grouped = False

        if spools:
            cells = sorted_cells(spools + recipes, sort_key, label_of=self._label_of)
        else:
            cells = recipes

        # Truly nothing to show — an empty library, or a search that matched
        # nothing. The grid has to say so instead of rendering zero rows.
        if not cells:
            self._recipes = []
            self._grid.setRecipes([], grouped=False)
            self._show_all_button.setEnabled(False)
            self._update_status()
            return

        self._recipes = cells
        self._grid.setRecipes(cells, grouped=grouped)
        self._show_all_button.setEnabled(self._pair_filter is not None or bool(query))
        selected = self._detail.selectionKey()
        if selected:
            index = self._grid.indexOfKey(selected)
            if index >= 0:
                self._grid.setSelectedIndex(index, scroll=False)
        self._update_status()

    def _update_status(self) -> None:
        count = len(self.library)
        pairs = count * (count - 1) // 2
        shown = len(self._recipes)
        total = pairs * len(MIX_RATIOS)
        parts = [f"耗材 {count} 种", f"母材组合 {pairs} 对", f"混色 {total} 个"]
        if self._all_colours.isChecked():
            parts.append(f"加耗材本色后 {total + count} 个")
        if shown != total + (count if self._all_colours.isChecked() else 0):
            parts.append(f"当前显示 {shown} 个")
        if self._build_seconds:
            parts.append(f"计算用时 {self._build_seconds * 1000:.0f} ms")
        self._status.setText(" · ".join(parts))
        self._grid_info.setText(
            f"每两种耗材 81 个配比（10%–90%）"
            + (
                f" · 含 {count} 种耗材本色，共 {total + count} 个颜色"
                if self._all_colours.isChecked()
                else ""
            )
            + (f" · 当前只显示 1 对耗材的混色" if self._pair_filter else "")
            + (f" · 已按「{self._sort_combo.currentText()}」排列" if self._pair_filter is None else "")
        )

    def _clear_filters(self) -> None:
        self._pair_filter = None
        self._search.clear()
        self._reload_library()

    def _show_pair(self, a_id: str, b_id: str) -> None:
        self._pair_filter = (a_id, b_id)
        self._search.clear()
        self._reload_library()
        filament_a = self.library.get(a_id)
        filament_b = self.library.get(b_id)
        if filament_a and filament_b:
            self._grid_info.setText(
                f"{filament_a.display_name}（{filament_a.color_hex}）"
                f" × {filament_b.display_name}（{filament_b.color_hex}）的 81 个配比"
            )

    # -- selection ---------------------------------------------------------------
    def _on_grid_selected(self, cell) -> None:
        # A spool cell is not a mix, so it has no pair to explain: it gets the
        # single-colour view instead of a fabricated 100% : 0% recipe.
        if getattr(cell, "pair_index", 0) < 0:
            self._detail.showFilament(cell.filament)
            self._status.setText(
                f"{cell.color_hex}  =  {cell.filament.display_name}（耗材本色，不是混色）"
            )
            return
        self._detail.showRecipe(cell)
        filament_a = self.library.get(cell.a_id)
        filament_b = self.library.get(cell.b_id)
        if filament_a and filament_b:
            self._status.setText(
                f"{cell.color_hex}  =  {filament_a.display_name}（{filament_a.color_hex}）{cell.percent_a}%"
                f"  +  {filament_b.display_name}（{filament_b.color_hex}）{cell.percent_b}%"
            )

    def _on_grid_activated(self, cell) -> None:
        """Double-click: a mix jumps to its parent pair, a spool has no pair."""
        if getattr(cell, "pair_index", 0) < 0:
            self._detail.showFilament(cell.filament)
            return
        self._show_pair(cell.a_id, cell.b_id)

    def _on_grid_cleared(self) -> None:
        """The user clicked blank space (or pressed Esc): drop the recipe."""
        self._detail.clear()
        self._update_status()

    # -- nearest colour ----------------------------------------------------------
    def nearest_recipe(self, target_hex: str):
        """The catalogue entry whose predicted colour is closest to ``target_hex``."""
        if self._catalog is None or not self._recipes:
            return None
        # Spool cells carry no predicted Lab (they have no ratio), so they are
        # skipped: this button answers "which mix should I dial in".
        mixes = [cell for cell in self._recipes if getattr(cell, "pair_index", 0) >= 0]
        if not mixes:
            return None
        target = _color.lab_from_rgb(_color.hex_to_rgb(target_hex))
        labs = np.array([recipe.lab for recipe in mixes], dtype=np.float64)
        rough = np.linalg.norm(labs - target, axis=1)
        shortlist = np.argsort(rough)[:64]
        best = None
        best_distance = float("inf")
        for index in shortlist:
            recipe = mixes[int(index)]
            distance = _color.delta_e_2000(recipe.lab, tuple(target))
            if distance < best_distance:
                best_distance = distance
                best = recipe
        return best

    # -- misc --------------------------------------------------------------------
    def _on_target_color_changed(self, value: str) -> None:
        """The 找最接近的混色 button is only meaningful once a colour is chosen."""
        chosen = bool(value) and self._target_color.isSet()
        self._find_button.setEnabled(chosen)
        if not chosen:
            self._find_button.setToolTip(
                "先选一个目标颜色；再按 CIEDE2000 在当前显示的全部混色里找最接近的配方"
            )
        else:
            self._find_button.setToolTip(
                f"在当前显示的全部混色里，按 CIEDE2000 找出最接近 {value} 的配方"
            )

    def _on_find_nearest(self) -> None:
        target = self._target_color.hex()
        if not target:
            self._status.setText("请先点「目标颜色」选一个想打印出来的颜色。")
            return
        if not self._recipes:
            self._status.setText("还没有可搜索的混色，请先添加至少两种耗材。")
            return
        best = self.nearest_recipe(target)
        if best is None:
            self._status.setText("没有找到匹配的混色。")
            return
        distance = _color.delta_e_2000(best.lab, tuple(_color.lab_from_rgb(_color.hex_to_rgb(target))))
        self._grid.selectRecipe(best)
        self._status.setText(f"最接近 {target} 的混色是 {best.color_hex}（色差 ΔE00 = {distance:.2f}）")

    def _on_about(self) -> None:
        try:
            location = paths.library_path()
        except OSError as exc:
            location = f"（不可用：{exc}）"
        QMessageBox.information(
            self,
            "关于与算法来源",
            "BambuPalette（混色耗材色彩管理器）\n\n"
            "「Bambu 混色预览」（默认）复刻自 Bambu Studio 2.8.2\n"
            "（v02.08.02.61）src/slic3r/GUI/MixedFilamentDialog.cpp 中的\n"
            "blend_colors()/blend_n_colors()，它们调用 libslic3r 的\n"
            "Slic3r::filament_mixer_lerp()，即 Bambu 内置的 filament_mixer\n"
            "颜料混色库：330 项四次多项式回归（Mixbox 近似）。\n"
            "filament_mixer 为 MIT 许可，Copyright (c) 2026 Justin Hayes。\n"
            "系数由 tools/gen_filament_mixer_profile.py 从上游头文件转录。\n\n"
            "「Bambu 2.5 旧版预览」是 2.5.x 及更早版本的算法\n"
            "（v02.05.03.62 的 blend_colors()/blend_n_colors()）：\n"
            "按 8 位 sRGB 加权平均后截断取整。\n\n"
            "「光谱 KM 模型」使用 Kubelka-Munk 光谱混合，其反射率由\n"
            "ICC「Munsell Glossy D50 XYZ polynomial estimator」多项式\n"
            "（xyz2PolyEstimateRefV2.icc，Copyright 2022 International\n"
            "Color Consortium，profile ID 5436fbfce5f7414dc520bb6e5d9c1516，\n"
            "SHA-256 8291983e…2e4）估计，实现参考\n"
            "ratdoux/OrcaSlicer-FullSpectrum（AGPL-3.0）。\n\n"
            "耗材档案保存在：" + str(location),
        )

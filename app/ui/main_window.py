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
    QScrollArea,
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

from .. import __version__
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
    SORT_SIMILARITY,
    MixCatalog,
    merge_recipes,
)
from ..spectral import color as _color
from ..core.triples import (
    TRIPLE_CACHE_PREFIX,
    CombinedColours,
    OrderedCombinedColours,
    TripleCatalog,
    load_triple_cache,
    mix_sidecar_path,
    prune_triple_caches,
    save_triple_cache,
    triple_cache_key,
)
from . import theme
from .filament_dialog import FilamentDialog
from .mix_detail import MixDetail
from .mix_grid import MixGrid
from .picture_page import PicturePage
from .selectable import selectable_text
from .swatch import ColorField, swatch_icon

_HEX_QUERY = re.compile(r"^[0-9a-fA-F]{1,6}$")

# How long the launcher waits, after showing the window, before it starts the
# first mix-catalogue build.  Long enough for the window to paint (a zero timer
# loses the race against the paint event), short enough not to feel like a pause.
BUILD_DELAY_MS = 150


class MainWindow(QMainWindow):
    def __init__(
        self,
        library: FilamentLibrary | None = None,
        parent=None,
        *,
        defer_build: bool = False,
        auto_triples: bool = True,
    ) -> None:
        super().__init__(parent)
        self.library = library if library is not None else self._load_library()
        self._catalog: MixCatalog | None = None
        self._recipes: list = []
        self._pair_filter: tuple[str, str] | None = None
        self._build_seconds = 0.0
        self._suppress_list_signal = False
        # filament id -> the case-folded text the search box matches on; rebuilt
        # whenever the library changes.  See _filament_haystack.
        self._haystacks: dict[str, str] = {}
        # 「全部颜色」: one store holding the two-filament mixes, the three-filament
        # mixes and the spool colours, deduplicated across all three.  It is built
        # even while the triple table is still being computed, so the page stays
        # usable instead of blanking out for the minutes a big library takes.
        self._combined = None
        self._combined_view = None
        self._combined_key = None
        # Three-filament mixing: the built table and the QTimer that advances the
        # sliced build.  There is no checkbox — see _ensure_triples.
        self._triples = None
        self._triple_timer = None
        # The real launcher asks for the first catalogue build to happen after the
        # window is on screen (see _rebuild_catalog).  Tests and the headless tools
        # keep the synchronous default, so nothing has to wait for a timer.
        self._defer_build = defer_build
        # 三色混色在这台机器上要按百万条配方算，窗口自己会切成小片算并显示
        # 进度，所以默认就开着（用户要的就是「不用勾选」）。测试和工具要的是
        # 确定的网格内容，可以把它关掉。
        self._auto_triples = auto_triples
        self._build_notice = False

        # The version rides in the title bar so 「我装的是哪一版」 never needs a trip
        # through Explorer → 属性.  It is the same string the exe's file properties
        # carry: build/BambuPalette.spec reads it out of app/__init__.py.
        self.setWindowTitle(f"BambuPalette {__version__} — 混色耗材色彩管理器")
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

        self._show_all_button = QPushButton("显示全部颜色")
        self._show_all_button.clicked.connect(self._clear_filters)
        self._show_all_button.setEnabled(False)

        # Deliberately UNSET: the target is whatever the user wants to print, so
        # showing a made-up colour here would be a lie until they fill one in.
        self._target_color = ColorField()
        self._target_color.setToolTip(
            "想打印出来的目标颜色：直接输入 #RRGGBB，或者填 R / G / B 三个数字，"
            "程序会找出最接近的颜色配方"
        )
        self._target_color.colorChanged.connect(self._on_target_color_changed)
        self._find_button = QPushButton("找最接近的颜色")
        self._find_button.setToolTip(
            "不用先填目标颜色也能按：没填就拿颜色表里选中的那个颜色来比，"
            "表里也没选就拿 R / G / B 三个数字框里的值。"
            "程序按 CIEDE2000 在当前显示的全部颜色里找出最接近的一个"
        )
        # Deliberately always clickable — exactly like 确认 in the 添加耗材 dialog.
        # Pressing it before choosing a colour is not an error: it borrows the
        # colour the user is already looking at and says so in the bottom line.
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
        target_row.addWidget(QLabel("双击两色混色 = 只看这一对耗材"))

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_library_panel())
        splitter.addWidget(self._build_grid_panel())
        self._detail = MixDetail()
        self._detail.pairRequested.connect(self._show_pair)
        self._detail.showAllRequested.connect(self._clear_filters)
        # A colour reachable by a dozen recipes makes this panel taller than the
        # window, and a QSplitter squeezes its children rather than scrolling
        # them — the recipe rows then collapse to a few pixels and print on top
        # of each other.  The scroll area gives the panel its real height back.
        detail_scroll = QScrollArea()
        detail_scroll.setWidgetResizable(True)
        detail_scroll.setFrameShape(QFrame.Shape.NoFrame)
        detail_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        detail_scroll.setWidget(self._detail)
        splitter.addWidget(self._wrap(detail_scroll, ""))
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([288, 612, 440])

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
        tabs.addTab(mix_page, "颜色配方")
        tabs.addTab(self._picture_page, "图像转换")
        self._tabs = tabs
        # Connected only once _tabs exists: adding the first tab already emits
        # currentChanged, and the slot reads self._tabs.
        tabs.currentChanged.connect(self._on_tab_changed)

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(16, 12, 16, 10)
        layout.setSpacing(10)
        layout.addWidget(tabs, 1)

        # The bottom line carries the one long sentence that explains how the
        # whole table is built, plus the transient 「正在计算…」 / 「最接近…」
        # messages.  It is a single line of ~1400 px at a normal font size, so
        # two things matter:
        #
        # * it must be allowed to shrink (``Ignored`` horizontally), or its
        #   minimumSizeHint becomes the window's minimum width — before it moved
        #   here it lived in the grid panel and pinned that panel so hard the
        #   splitter handle could not be dragged at all;
        # * it must wrap, and live in a layout that honours the wrapped height,
        #   or a 1020 px window silently clips the tail of the sentence.
        #
        # That is why it is the last row of the page body rather than an item in
        # a QStatusBar: a status bar is one line high and would clip the second
        # line the moment the sentence wraps.  The tooltip still carries the
        # whole text for anyone who would rather hover than read.
        self._status = QLabel("")
        self._status.setProperty("role", "hint")
        self._status.setWordWrap(True)
        self._status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum)
        layout.addWidget(self._status)

        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(topbar)
        outer.addWidget(body, 1)
        self.setCentralWidget(central)

        # Every hint, recipe and status line is text worth pasting elsewhere.
        selectable_text(self)

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

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._list, 1)
        layout.addWidget(self._library_hint)
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

        # 「全部颜色」: the two-filament mixes, the three-filament mixes and the
        # spool colours themselves are one list, deduplicated across all of them.
        # There is deliberately no checkbox for any of the three parts — see
        # CombinedColours.
        self._filter_bar = QFrame()
        self._filter_bar.setObjectName("filterBar")
        self._filter_label = QLabel("")
        self._filter_label.setWordWrap(True)
        self._filter_back = QPushButton("显示全部颜色")
        self._filter_back.setProperty("accent", "true")
        self._filter_back.clicked.connect(self._clear_filters)
        bar_row = QHBoxLayout(self._filter_bar)
        bar_row.setContentsMargins(10, 6, 10, 6)
        bar_row.setSpacing(10)
        bar_row.addWidget(self._filter_label, 1)
        bar_row.addWidget(self._filter_back)
        self._filter_bar.setVisible(False)

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._filter_bar)
        layout.addWidget(self._grid, 1)
        return self._wrap(panel, "全部颜色")

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
        clear_action = QAction("显示全部颜色", self)
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
        # The spools are about to be replaced, so the search haystacks keyed by
        # filament id are stale.
        self._haystacks.clear()
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
        """Add a spool; 「下一个」 keeps the session going for the next one.

        The dialog is rebuilt for each spool rather than reset in place, so every
        field, the brand / type suggestions and the colour all start clean.
        """
        while True:
            dialog = FilamentDialog(
                None,
                brands=[f.brand for f in self.library if f.brand],
                types=[f.material_type for f in self.library if f.material_type],
                parent=self,
            )
            if dialog.exec() != FilamentDialog.DialogCode.Accepted:
                return
            filament = dialog.build_filament()
            duplicate = self.library.find_duplicate(
                filament.color_hex, filament.material_type, filament.brand
            )
            if duplicate is not None:
                answer = QMessageBox.question(
                    self,
                    "已经存在同样的耗材",
                    f"{duplicate.display_name}（{duplicate.color_hex}）看起来和这条一样，仍然要添加吗？",
                )
                if answer != QMessageBox.StandardButton.Yes:
                    if dialog.wants_another():
                        continue
                    return
            self.library.add(filament)
            self._reload_library()
            self._select_filament(filament.id)
            self._status.setText(f"已添加 {filament.display_name}（{filament.color_hex}）")
            if not dialog.wants_another():
                return

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
        sidecar = mix_sidecar_path(Path(path))
        note = ""
        if sidecar.is_file() and self._catalog is not None:
            if self._adopt_mix_sidecar(sidecar):
                note = "\n同时用上了档案里的三色混色缓存，不用重算。"
        QMessageBox.information(self, "导入完成", f"已导入 {added} 条耗材。{note}")

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
        note = ""
        written = self._write_mix_sidecar(Path(path))
        if written:
            note = (
                f"\n并把整套三色混色缓存一起导出到\n{written.name}"
                f"（{written.stat().st_size / 1024 / 1024:.1f} MB），"
                "下次导入这套耗材就不用重算了。"
            )
        QMessageBox.information(self, "导出完成", f"已写入\n{path}{note}")

    def _write_mix_sidecar(self, path: Path) -> Path | None:
        """把已经算好的三色混色表放在档案旁边，免得下次再等几分钟。"""
        catalog = self._triples
        if catalog is None or not getattr(catalog, "built", False):
            return None
        target = mix_sidecar_path(path)
        try:
            save_triple_cache(catalog, target)
        except Exception:  # a cache we cannot write is only a slow next launch
            return None
        return target

    def _adopt_mix_sidecar(self, sidecar: Path) -> bool:
        """档案旁边的混色表只有在确实属于这套耗材时才认。"""
        try:
            with np.load(sidecar, allow_pickle=False) as data:
                stored = str(data["key"].item()) if "key" in data else ""
                if stored != triple_cache_key(self.library.filaments, self._catalog.engine.id):
                    return False
                payload = {name: data[name] for name in data.files if name != "key"}
        except Exception:
            return False
        try:
            catalog = TripleCatalog.loads(
                self.library.filaments, payload, self._catalog.engine.id
            )
        except Exception:
            return False
        if not catalog.built:
            return False
        self._triples = catalog
        self._rebuild_combined()
        try:
            save_triple_cache(catalog, self._triples_path())
        except Exception:
            pass
        self._refresh_grid()
        return True

    # -- catalogue ---------------------------------------------------------------
    def _rebuild_catalog(self) -> None:
        started = time.perf_counter()
        engine_id = self._engine_combo.currentData()
        catalog = getattr(self, "_catalog", None)
        if catalog is None or catalog.engine.id != engine_id:
            if self._defer_build:
                # 41 spools is 66,420 recipes and ~3 s of solid computation. Doing
                # that inside __init__ means the window takes 4.4 s to appear and
                # looks hung; doing it here, on a zero timer, lets the window paint
                # a "calculating…" line and a wait cursor first.  The flag is
                # cleared before re-entering so the second call takes the normal
                # path below.
                self._defer_build = False
                self._announce_build()
                # A zero timer is NOT enough: it fires before the paint event
                # show() posted, so the window would still appear only after the
                # build.  A short delay lets the window paint its "calculating…"
                # state first, which is the whole point.
                QTimer.singleShot(BUILD_DELAY_MS, self._rebuild_catalog)
                return
            self._catalog = MixCatalog(self.library.filaments, MIX_RATIOS, engine=engine_id)
            self._catalog.build()
        else:
            # Adding or deleting one spool only needs that spool's pairs; a full
            # rebuild is 66,420 recipes and ~3 s at 41 spools.
            catalog.sync(self.library.filaments)
        self._build_seconds = time.perf_counter() - started
        self._invalidate_triples()
        self._end_build_notice()
        self._detail.setContext(self.library, self._catalog.engine)
        if getattr(self, "_picture_page", None) is not None:
            self._picture_page.setLibrary(self.library, self._catalog)
        if self._pair_filter is not None:
            a_id, b_id = self._pair_filter
            if self.library.get(a_id) is None or self.library.get(b_id) is None:
                self._pair_filter = None
        self._refresh_grid()
        # 三色混色是这一页的一部分，不是可选功能：先把两色和耗材本色画出来，
        # 再去拿缓存 / 开始算三色（见 _ensure_triples）。
        self._ensure_triples()

    def _announce_build(self) -> None:
        """Say what the pending catalogue build is about to do."""
        count = len(self.library)
        total = count * (count - 1) // 2 * len(MIX_RATIOS)
        # The grid's own empty state blames the library ("add two spools"); while
        # the calculation is running that is simply untrue.
        self._grid.setEmptyText("正在计算混色表…")
        self._set_status_text(f"正在计算 {total:,} 个混色，请稍候…")
        if not self._build_notice:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            self._build_notice = True
        QApplication.processEvents()

    def _end_build_notice(self) -> None:
        self._grid.setEmptyText("还没有混色。请先添加至少两种耗材。")
        if self._build_notice:
            self._build_notice = False
            QApplication.restoreOverrideCursor()

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

    def _filament_haystack(self, filament) -> str:
        """The case-folded text a spool is searched by, built once per spool.

        At 41 spools there are 66,420 mixes and every one of them asks for its
        two parents' names, so rebuilding these four strings inside the search
        loop cost ~0.1 s per keystroke.  The library is replaced wholesale on
        every edit, so keying on the id and clearing the cache in
        ``_reload_library`` is enough to stay correct.
        """
        cached = self._haystacks.get(filament.id)
        if cached is None:
            cached = " ".join(
                (
                    filament.display_name,
                    filament.brand,
                    filament.material_type,
                    filament.note,
                )
            ).casefold()
            self._haystacks[filament.id] = cached
        return cached

    def _matches(self, cell, query: str) -> bool:
        """Does one grid cell survive the search box?

        A spool cell is matched on its own name/brand/type/note; a colour cell
        also on the parents of every recipe that produces it, so searching a
        spool name still finds every colour that spool takes part in.
        """
        if getattr(cell, "pair_index", 0) < 0 and hasattr(cell, "filament"):
            filament = cell.filament
            if _HEX_QUERY.match(query):
                return query.lower() in filament.color_hex[1:].lower()
            return query in self._filament_haystack(filament)
        if _HEX_QUERY.match(query):
            return query.lower() in cell.color_hex[1:].lower()
        for recipe in getattr(cell, "recipes", (cell,)):
            # MixRecipe carries a_id/b_id; TripleRecipe carries parent_ids.
            parents = getattr(recipe, "parent_ids", None) or (
                recipe.a_id,
                recipe.b_id,
            )
            for filament_id in parents:
                filament = self.library.get(filament_id)
                if filament is not None and query in self._filament_haystack(filament):
                    return True
        return False

    # -- three-filament mixes ---------------------------------------------------

    def _triples_path(self):
        """三色混色缓存在哪：一个耗材库 + 一个引擎一个文件。"""
        engine_id = self._catalog.engine.id if self._catalog is not None else DEFAULT_ENGINE
        key = triple_cache_key(self.library.filaments, engine_id)
        return paths.data_dir() / "cache" / f"{TRIPLE_CACHE_PREFIX}{key}.npz"

    def _invalidate_triples(self) -> None:
        """耗材库或引擎变了，三色表就过期了：正在算的停掉，算好的丢掉。

        否则一边算一边改耗材，算完的那张表还是旧配方，却会被当成新耗材库的
        缓存写下去（文件名按新库取，内容按旧库算），下次打开就全是错颜色。
        """
        if self._triples is not None or self._triple_timer is not None:
            self._stop_triple_timer()
            self._triples = None
            self._end_build_notice()
        self._rebuild_combined()

    def _rebuild_combined(self) -> None:
        """「全部颜色」= 两色混色 + 三色混色 + 耗材本色，跨表去重成一张表。

        三色表还在算的时候也先建一份（只有两色和耗材本色）：正在算三色混色
        不该让这一页先空上几分钟。
        """
        if self._catalog is None:
            self._combined = None
        else:
            self._combined = CombinedColours(
                self._catalog, self._triples, self.library.filaments
            )
        self._combined_view = None
        self._combined_key = None

    def _ordered_combined(self, sort_key: str):
        """当前排序下的懒视图；排好的那一次留着，来回切设置不再重排。"""
        target = tuple(self._target_color.rgb())
        signature = (sort_key, target, len(self.library))
        if self._combined_key == signature and self._combined_view is not None:
            return self._combined_view
        self._combined_view = self._combined.ordered(sort_key, target)
        self._combined_key = signature
        return self._combined_view

    def _search_mask(self, query: str):
        """搜索框 → 命中的颜色下标（numpy 数组，可能是几百万个）。

        十六进制片段用 nibble 比较在所有颜色上一次性跑完；文字先找出名字匹配
        的耗材，再取它们参与过的每一种颜色。
        """
        text = query.strip().lstrip("#")
        if _HEX_QUERY.match(text):
            return self._combined.search_hex(text)
        wanted = [
            filament.id
            for filament in self.library.filaments
            if query.casefold() in self._filament_haystack(filament)
        ]
        if not wanted:
            return np.empty(0, dtype=np.int64)
        return self._combined.search_spools(wanted)

    # -- three-filament mixes ---------------------------------------------------

    def _triples_path(self):
        """三色混色缓存在哪：一个耗材库 + 一个引擎一个文件。"""
        engine_id = self._catalog.engine.id if self._catalog is not None else DEFAULT_ENGINE
        key = triple_cache_key(self.library.filaments, engine_id)
        return paths.data_dir() / "cache" / f"{TRIPLE_CACHE_PREFIX}{key}.npz"

    def _ensure_triples(self) -> None:
        """三色混色不用谁去勾选：有 3 卷料就该有它，先看缓存，没有就算。"""
        if not self._auto_triples:
            return
        if self._catalog is None or len(self.library) < 3:
            return
        if self._triples is not None or self._triple_timer is not None:
            return
        cached = load_triple_cache(
            self._triples_path(), self.library.filaments, self._catalog.engine.id
        )
        if cached is not None and cached.colour_count:
            self._triples = cached
            self._rebuild_combined()
            # 命中缓存就顺手清掉旧耗材库留下的那几份（一份约 153 MB）。
            prune_triple_caches(self._triples_path().parent)
            self._refresh_grid()
            return
        self._start_triple_build()

    def _start_triple_build(self) -> None:
        catalog = TripleCatalog(self.library.filaments, engine=self._catalog.engine.id)
        if not catalog.triple_count:
            return
        self._triples = catalog
        catalog.begin()
        self._announce_triples()
        self._triple_timer = QTimer(self)
        self._triple_timer.setInterval(0)
        self._triple_timer.timeout.connect(self._step_triple_build)
        self._triple_timer.start()

    def _announce_triples(self) -> None:
        self._set_status_text(f"正在计算三色混色：{self._triples.recipe_total:,} 条配方，请稍候…")
        self._grid.setEmptyText("正在计算三色混色…")
        if not self._build_notice:
            self._build_notice = True
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

    def _step_triple_build(self) -> None:
        catalog = self._triples
        if catalog is None:
            self._stop_triple_timer()
            return
        finished = catalog.step(0.18)
        done = catalog.mixed_done
        total = catalog.recipe_total
        percent = (done * 100 // total) if total else 100
        self._set_status_text(f"正在计算三色混色：{done:,} / {total:,}（{percent}%）")
        if not finished:
            return
        self._stop_triple_timer()
        self._set_status_text("正在排序去重…")
        QApplication.processEvents()
        catalog.finish()
        self._rebuild_combined()
        try:
            save_triple_cache(catalog, self._triples_path())
        except Exception:  # a cache we cannot write is only a slow next launch
            pass
        self._end_build_notice()
        self._refresh_grid()

    def _stop_triple_timer(self) -> None:
        timer = self._triple_timer
        if timer is not None:
            timer.stop()
            timer.deleteLater()
            self._triple_timer = None

    def _refresh_grid(self) -> None:
        if self._catalog is None:
            return

        sort_key = self._sort_combo.currentData() or SORT_RGB
        query = self._search.text().strip()
        if self._pair_filter is not None:
            # One pair still has to be merged: its 81 recipes can repeat a colour.
            cells_source = merge_recipes(self._catalog.pair_recipes(*self._pair_filter))
            if query:
                needle = query.lstrip("#").casefold()
                cells_source = [
                    colour for colour in cells_source if self._matches(colour, needle)
                ]
        elif self._combined is not None and len(self._combined):
            # 「全部颜色」 is one list: 两色混色 + 三色混色 + 耗材本色, deduplicated
            # across all three by CombinedColours.  Millions of entries means the
            # order stays a numpy permutation and a search is a mask over it; the
            # grid only ever materialises the cells it is drawing.
            view = self._ordered_combined(sort_key)
            if query:
                view = view.restrict(self._search_mask(query))
            cells_source = view
        else:
            # Fewer than two spools: there are no mixes, but a lone spool's own
            # colour is still something to show rather than an empty panel.
            cells_source = []

        # Truly nothing to show — an empty library, or a search that matched
        # nothing. The grid has to say so instead of rendering zero rows.
        if not cells_source:
            self._recipes = []
            self._grid.setRecipes([], grouped=False)
            # A search that matched nothing is exactly when the way back to the
            # full list matters most, so the button stays live.
            self._show_all_button.setEnabled(self._pair_filter is not None or bool(query))
            self._update_status()
            return

        self._recipes = cells_source
        self._grid.setRecipes(cells_source, grouped=False)
        self._show_all_button.setEnabled(self._pair_filter is not None or bool(query))
        selected = self._detail.selectionKey()
        if selected:
            index = self._grid.indexOfKey(selected)
            if index >= 0:
                self._grid.setSelectedIndex(index, scroll=False)
        self._update_status()

    def _on_tab_changed(self, index: int) -> None:
        """The colour-table note belongs to 颜色配方, so re-evaluate it."""
        self._update_status()

    def _set_status_text(self, text: str) -> None:
        """Write the bottom line, keeping the full sentence in the tooltip.

        The label wraps, so a long sentence folds onto a second line instead of
        forcing the window wider or being clipped — but its tail is still worth
        keeping in the tooltip for a single-glance read.
        """
        self._status.setText(text)
        self._status.setToolTip(text)

    def _update_status(self) -> None:
        count = len(self.library)
        pairs = count * (count - 1) // 2
        shown = len(self._recipes)
        double_recipes = pairs * len(MIX_RATIOS)
        colours = self._catalog.colour_count if self._catalog is not None else 0
        triples = self._triples if (self._triples is not None and self._triples.built) else None
        everything = self._combined.colour_count if self._combined is not None else colours

        # The bottom line explains how the table is built: how many ratios each
        # two- and three-spool combination contributes, and how many distinct
        # colours that leaves.  It replaced a 「#RRGGBB = 配料」 line that could
        # only ever name the FIRST recipe of the selected colour, which for a
        # three-colour mix silently dropped the third spool.
        note = "每两种耗材 81 个配比（10%–90%）"
        made = double_recipes
        if triples is not None:
            note += " · 三色混色：每三卷 2556 个配比（每种 10%–80%，按 1% 递增）"
            made += triples.recipe_count
        note += f" · 共 {made:,} 个配比"
        if count:
            note += f" · 含 {count} 种耗材本色"
        note += f" · 去重后共 {everything:,} 个颜色"
        if self._pair_filter is not None:
            note += " · 已筛选"
        else:
            note += f" · 已按「{self._sort_combo.currentText()}」排列"
        if shown != everything:
            note += f" · 当前显示 {shown:,} 个"
        if self._triple_timer is not None:
            note += " · 正在计算三色混色"
        elif self._build_seconds:
            note += f" · 计算用时 {self._build_seconds * 1000:.0f} ms"
        if self._tabs.currentIndex() != 0:
            # The sentence is about the colour table.  On 图像转换 it describes
            # nothing the reader can see, and the picture page has a status line
            # of its own right above this one.
            note = ""
        self._set_status_text(note)
        self._update_filter_bar(shown, everything, count)

    def _update_filter_bar(self, shown: int, everything: int, count: int) -> None:
        """Show why the grid is short, and offer the way back to everything."""
        reasons = []
        if self._pair_filter is not None:
            left = self.library.get(self._pair_filter[0])
            right = self.library.get(self._pair_filter[1])
            names = " × ".join(
                f"{f.display_name}（{f.color_hex}）" if f is not None else "（已删除）"
                for f in (left, right)
            )
            reasons.append(f"只看 {names} 的 81 个配比")
        query = self._search.text().strip()
        if query:
            reasons.append(f"搜索结果「{query}」")

        if not reasons:
            self._filter_bar.setVisible(False)
        else:
            hidden = max(everything - shown, 0)
            self._filter_label.setText(
                f"{'、'.join(reasons)}：现在显示 {shown} 个，另外 {hidden} 个没有显示。"
                "点右边的按钮可以回到全部颜色。"
            )
            self._filter_bar.setVisible(True)
        self._set_accent(self._show_all_button, bool(reasons))
        detail = getattr(self, "_detail", None)
        if detail is not None:
            detail.setPairFilter(self._pair_filter is not None)

    @staticmethod
    def _set_accent(button: QPushButton, on: bool) -> None:
        """Flip a button between the plain and the accent style.

        A dynamic property only repaints after Qt re-reads the stylesheet, which
        it does not do by itself when the property changes.
        """
        value = "true" if on else "false"
        if button.property("accent") == value:
            return
        button.setProperty("accent", value)
        button.style().unpolish(button)
        button.style().polish(button)
        button.update()

    def _clear_filters(self) -> None:
        self._pair_filter = None
        self._search.clear()
        self._reload_library()

    def _show_pair(self, a_id: str, b_id: str) -> None:
        self._pair_filter = (a_id, b_id)
        self._search.clear()
        # ``_reload_library`` reaches ``_update_status``, which names the pair in
        # the filter banner; overwriting ``_grid_info`` here as well would leave
        # two different explanations of the same filter on screen.
        self._reload_library()

    # -- selection ---------------------------------------------------------------
    def _on_grid_selected(self, cell) -> None:
        # A spool cell is not a mix, so it has no pair to explain: it gets the
        # single-colour view instead of a fabricated 100% : 0% recipe.
        #
        # The bottom line is deliberately left alone: it explains how the whole
        # table is built, and the detail panel already spells out the selected
        # colour — every one of its spools, which a one-line summary could not.
        if getattr(cell, "pair_index", 0) < 0 and hasattr(cell, "filament"):
            self._detail.showFilament(cell.filament)
            return
        self._detail.showColour(cell)

    def _on_grid_activated(self, cell) -> None:
        """Double-click: only a two-filament mix has one parent pair to jump to.

        A three-filament mix is made by three spools at once.  Filtering to its
        first two would show the 81 two-colour ratios of a DIFFERENT colour —
        which is exactly the 「配方是三色，双击后却变成两色」 report.  So a
        triple (and a spool) only refreshes the detail panel.
        """
        if getattr(cell, "pair_index", 0) < 0 and hasattr(cell, "filament"):
            self._detail.showFilament(cell.filament)
            return
        first = cell.recipes[0] if getattr(cell, "recipes", None) else None
        # ``TripleRecipe`` is the only recipe shape carrying a third spool.
        if first is None or hasattr(first, "percent_c"):
            self._detail.showColour(cell)
            return
        self._show_pair(first.a_id, first.b_id)

    def _on_grid_cleared(self) -> None:
        """The user clicked blank space (or pressed Esc): drop the recipe."""
        self._detail.clear()
        self._update_status()

    # -- nearest colour ----------------------------------------------------------
    def nearest_recipe(self, target_hex: str):
        """The catalogue entry whose predicted colour is closest to ``target_hex``.

        EVERY colour in the table is a candidate, the spool colours included.
        Skipping them is what made this button lie: asking for #FFFFFF answered
        #E5E5E5 (ΔE00 5.40) while pure #FFFFFF was sitting in the very same grid
        at ΔE00 0.
        """
        view = self._recipes
        if self._catalog is None or not len(view):
            return None
        target = _color.hex_to_rgb(target_hex)
        if self._combined is not None and isinstance(view, OrderedCombinedColours):
            # Millions of colours: the store's own Lab table plus one vectorised
            # CIEDE2000 argsort is the only affordable way to answer this.
            # ``order`` is a stable sort on the exact distance, so its first
            # entry is the nearest colour — no shortlist, no skipping.
            order = self._combined.order(SORT_SIMILARITY, target)
            if len(view) != len(self._combined):
                # A search narrowed the table: keep only what is on screen.  The
                # mask preserves the sort order, so the first survivor is still
                # the nearest visible colour.
                order = order[np.isin(order, view.indices)]
            if len(order):
                return self._combined[int(order[0])]
            return None
        # The short lists (one pair's 81 recipes, or a search inside it) have no
        # Lab table to sort, so measure every candidate against the target.
        lab_target = tuple(_color.lab_from_rgb(target))
        best = None
        best_distance = float("inf")
        for cell in view:
            lab = getattr(cell, "lab", None)
            if lab is None:
                continue
            distance = _color.delta_e_2000(lab, lab_target)
            if distance < best_distance:
                best_distance = distance
                best = cell
        return best

    # -- misc --------------------------------------------------------------------
    def _on_target_color_changed(self, value: str) -> None:
        """Only the tooltip follows the target colour; the button stays enabled."""
        chosen = bool(value) and self._target_color.isSet()
        if not chosen:
            self._find_button.setToolTip(
                "不用先填目标颜色也能按：没填就拿颜色表里选中的那个颜色来比，"
                "表里也没选就拿 R / G / B 三个数字框里的值。"
                "程序按 CIEDE2000 在当前显示的全部颜色里找出最接近的一个"
            )
        else:
            self._find_button.setToolTip(
                f"在当前显示的全部颜色里，按 CIEDE2000 找出最接近 {value} 的那一个"
            )

    def _on_find_nearest(self) -> None:
        """Find the closest colour, borrowing a target when the user typed none.

        「找最接近的颜色」 has to DO something every time it is pressed: with an
        empty 目标颜色 the old code only wrote a hint, which reads as a broken
        button.  So the target is taken from, in order: the colour the user is
        looking at in the table, then the R / G / B numbers as they stand.
        """
        target = self._target_color.hex()
        borrowed = ""
        if not target:
            current = self._grid.selectedRecipe()
            if current is not None:
                target = current.color_hex
                borrowed = f"你正看着的 {target}"
                # Show it in the field as well: a search must never aim at
                # something the window does not display.
                self._target_color.setValue(target)
            else:
                r, g, b = self._target_color.rgb()
                target = _color.rgb_to_hex((r, g, b))
                borrowed = f"数字框里的 RGB {r}, {g}, {b}"
        if not len(self._recipes):
            self._set_status_text("还没有可搜索的颜色，请先添加至少一种耗材。")
            return
        best = self.nearest_recipe(target)
        if best is None:
            self._set_status_text("没有找到匹配的颜色。")
            return
        distance = _color.delta_e_2000(best.lab, tuple(_color.lab_from_rgb(_color.hex_to_rgb(target))))
        self._grid.selectRecipe(best)
        # A spool's own colour is a legitimate answer — and the most accurate one
        # whenever the target IS a colour the user owns — so say so instead of
        # presenting it as a mixture.
        if getattr(best, "recipe_count", 1):
            found = f"最接近 {target} 的颜色是 {best.color_hex}"
        else:
            found = f"最接近 {target} 的颜色就是耗材本色 {best.color_hex}"
        self._set_status_text(
            f"{found}（色差 ΔE00 = {distance:.2f}）"
            + (f"；目标颜色取自{borrowed}" if borrowed else "")
        )

    def _about_text(self) -> str:
        """The 关于 text, version line included.

        Split out of ``_on_about`` so the automatic checks can read it without
        a modal dialog: the version shown here and the one in the exe's file
        properties both come from ``app.__version__``.
        """
        try:
            location = paths.library_path()
        except OSError as exc:
            location = f"（不可用：{exc}）"
        return (
            f"BambuPalette（混色耗材色彩管理器）  版本 {__version__}\n"
            f"BambuPalette.exe 的文件属性里也写着同一个版本号。\n\n"
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
            "耗材档案保存在：" + str(location)
        )

    def _on_about(self) -> None:
        QMessageBox.information(self, "关于与算法来源", self._about_text())

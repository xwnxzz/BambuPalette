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
from PySide6.QtGui import QBrush, QColor, QKeySequence, QShortcut
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
from ..core.image_matching import plan_help
from ..core.library import (
    ANY_MATERIAL,
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
from . import theme, tips
from .filament_dialog import FilamentDialog
from .help import HelpDialog
from .mix_detail import MixDetail
from .mix_grid import MixGrid
from .picture_page import PicturePage
from .model_page import ModelPage
from .selectable import selectable_text
from .settings_dialog import SettingsDialog
from .swatch import ColorField, swatch_icon

_HEX_QUERY = re.compile(r"^[0-9a-fA-F]{1,6}$")

#: 打开软件时「目标颜色」里预置的颜色（m02366 item 1：用户要求「打开软件后目标颜色
#: RGB 默认填写 000」）。以前它是「未选择」状态、三个数字框是空白，而空着的数字框读
#: 出来就是 (0, 0, 0)，于是点「找最接近的颜色」会拿纯黑去查、屏幕上却什么都没写，
#: 看起来像按钮查错了东西。现在预置成纯黑：数字框里明明白白写着 0 / 0 / 0，
#: 用户想查别的颜色就改它，清空成「未选择」也照旧可以。
DEFAULT_TARGET_HEX = "#000000"

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
        # 探针与闸门直接造 MainWindow（不走 app.main.build_application），所以在这里
        # 也装一次：悬停提示在 1.6.53 起全程序都不许再弹（m21616 第 3 条）。
        tips.disable_tooltips()
        loaded_from_disk = library is None
        self.library = library if library is not None else self._load_library()
        if loaded_from_disk:
            # 上次用的耗材种类 (m05557 item 1). 只有从磁盘读出来的档案才认这条
            # 记忆——调用方自己造一份库（测试、工具）时，它给的 material 就是
            # 它要的，不被磁盘上的偏好覆盖。
            self.library.material = self._remembered_material()
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
        # 导入一份「档案 + 混色表」时，档案旁边那份表的路径先记在这儿，让重建
        # 耗材库的那一步别急着开始算（见 _on_import / _ensure_triples）。
        self._pending_sidecar = None

        # The version rides in the title bar so 「我装的是哪一版」 never needs a trip
        # through Explorer → 属性.  It is the same string the exe's file properties
        # carry: build/BambuPalette.spec reads it out of app/__init__.py.
        self.setWindowTitle(f"BambuPalette {__version__} — 混色耗材色彩管理器")
        self.resize(1320, 840)
        self.setMinimumSize(1020, 640)

        self._build_ui()
        self._reload_library()

    # -- construction ------------------------------------------------------------
    def _load_library(self) -> FilamentLibrary:
        try:
            return FilamentLibrary.load()
        except LibraryError as exc:
            QTimer.singleShot(0, lambda: QMessageBox.warning(self, "读取耗材档案失败", str(exc)))
            return FilamentLibrary()

    def _remembered_material(self) -> str:
        """The 耗材种类 to start on: 上次选的那个，否则卷数最多的那一种.

        「切换材质后全部颜色跟着耗材颜色变化」只有在下拉显示的和实际用来配色
        的是同一种时才成立，所以记忆里的那种已经不在了（删了、藏了、改了种类）
        就落回库里最多的那一种，而不是留着一个查不到的字符串 (m05557 item 1).
        """
        names = self.library.materials()
        remembered = str(settings.load_settings().get("material") or "").strip()
        folded = {name.casefold(): name for name in names}
        if remembered and remembered.casefold() in folded:
            return folded[remembered.casefold()]
        return names[0] if names else ANY_MATERIAL

    def _build_ui(self) -> None:
        # 1.6.54（m22211 第 1 条）：整条顶栏（那个 「BambuPalette」 大标题）删了。
        # 1.6.53 起帮助 / 设置 / 添加耗材都已经搬到「目标颜色」那一行的右端，
        # 窗口标题栏里也写着名字与版本，这一条只剩一个重复的标题，白占一行高度。

        self._add_button = QPushButton("添加耗材")
        self._add_button.setProperty("accent", "true")
        self._add_button.clicked.connect(self._on_add)

        # 1.6.11 deleted the 文件 / 视图 / 帮助 menus: every action in them is
        # already a button on this page (添加耗材 here, 导入/导出 in 我的耗材,
        # 显示全部颜色 above the grid), so the only thing the menus still owned was
        # 关于与算法来源 — the algorithm credits, the version and the path of the
        # spool archive.  That one moved here instead of disappearing with them.
        # 1.6.14 (m01885 item 5) renamed it 帮助 and gave it the manual: the text
        # that used to sit under the picture toolbar and the 「耗材方案」 notes.
        self._help_button = QPushButton("帮助")
        self._help_button.clicked.connect(self._on_help)
        self._help_window: HelpDialog | None = None

        # 1.6.53（m21616 第 8 条）：设置。它和帮助、添加耗材一起放在「目标颜色」那一行的
        # 右端，顺序就是用户要的 帮助 → 设置 → 添加耗材。
        self._settings_button = QPushButton("设置")
        self._settings_button.clicked.connect(self._on_settings)

        # engine / sort / search controls
        self._engine_combo = QComboBox()
        for engine_id, label in ENGINE_CHOICES:
            self._engine_combo.addItem(label, engine_id)
        remembered = settings.load_settings().get("engine", DEFAULT_ENGINE)
        index = self._engine_combo.findData(remembered)
        if index < 0:
            index = self._engine_combo.findData(DEFAULT_ENGINE)
        self._engine_combo.setCurrentIndex(max(0, index))
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

        # 预置纯黑（见 DEFAULT_TARGET_HEX）：空的数字框读出来也是 (0,0,0)，但屏幕上
        # 是白的，点「找最接近的颜色」就像在查一个没人填过的目标。用户清空它仍然可以
        # 回到「未选择」，那时按原来的办法借用表格里选中的颜色。
        self._target_color = ColorField(DEFAULT_TARGET_HEX)
        self._find_button = QPushButton("找最接近的颜色")
        # Deliberately always clickable — exactly like 确认 in the 添加耗材 dialog.
        # Pressing it before choosing a colour is not an error: it borrows the
        # colour the user is already looking at and says so in the bottom line.
        # It is also the only place that answers 「用的是哪个颜色」, which is why the
        # answer goes into the status line instead of a hover pop-up.
        self._find_button.clicked.connect(self._on_find_nearest)

        # 1.6.54（m22211 第 7 条）：颜色配方页的两个动作绑上按键（键值在设置里改）。
        self._find_shortcut = QShortcut(QKeySequence("Ctrl+F"), self)
        self._find_shortcut.activated.connect(self._on_find_nearest)
        self._add_shortcut = QShortcut(QKeySequence("Ctrl+N"), self)
        self._add_shortcut.activated.connect(self._on_add)

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
        # 1.6.53（m21616 第 6/8 条）：帮助 → 设置 → 添加耗材，贴着「目标颜色」这一行的右端。
        target_row.addWidget(self._help_button, 0, Qt.AlignmentFlag.AlignVCenter)
        target_row.addWidget(self._settings_button, 0, Qt.AlignmentFlag.AlignVCenter)
        target_row.addWidget(self._add_button, 0, Qt.AlignmentFlag.AlignVCenter)

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
        self._model_page = ModelPage()

        tabs = QTabWidget()
        tabs.addTab(mix_page, "颜色配方")
        tabs.addTab(self._picture_page, "图像转换")
        tabs.addTab(self._model_page, "模型配色")
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
        self._import_button.clicked.connect(self._on_import)
        self._export_button = QPushButton("导出")
        self._export_button.clicked.connect(self._on_export)
        # Sitting in the panel header, next to the title, keeps the destructive
        # one away from 编辑 / 删除 / 导入 / 导出, where a mis-click would cost
        # one spool instead of all of them.  It is disabled while there is
        # nothing to clear.
        self._clear_button = QPushButton("清空耗材库")
        self._clear_button.clicked.connect(self._on_clear_library)
        self._clear_button.setEnabled(False)

        # 材质切换 (m05557 item 1): 两种塑料不能混进同一根喷嘴，所以配色只能认一
        # 个耗材种类。它和「清空耗材库」同在最上面一行，因为它管的是整库的颜色而
        # 不是某一卷：切换之后 混色 / 全部颜色 / 三色表 和「图像转换」页的匹配都
        # 只从这个种类的耗材里取色（见 FilamentLibrary.visible）。
        self._material_combo = QComboBox()
        self._material_combo.setToolTip(
            "按耗材种类配色：切换以后，全部颜色都只从这一种的耗材里取。"
        )
        self._material_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self._material_combo.currentIndexChanged.connect(self._on_material_changed)
        self._material_guard = False
        # 上一轮删空一种材质留下的那句话（m10789 item 1）。它挂在状态栏上直到下一次
        # 重载耗材库：三色混色算完还会再刷一次状态栏，写死在状态栏里的一句话会被刷掉。
        self._material_note = ""
        material_row = QWidget()
        material_layout = QHBoxLayout(material_row)
        material_layout.setContentsMargins(0, 0, 0, 0)
        material_layout.setSpacing(6)
        material_layout.addWidget(QLabel("材质"))
        material_layout.addWidget(self._material_combo)
        material_layout.addWidget(self._clear_button)

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

        return self._wrap(panel, "我的耗材", material_row)

    def _build_grid_panel(self) -> QWidget:
        self._grid = MixGrid()
        # Activating a cell (double-click, or Enter on the focused one) does the
        # same as selecting it.  It deliberately does NOT narrow the grid to the
        # recipe's pair any more: that surprise jump was reported as
        # 「双击混色以后颜色都不见了」, and the pair filter has an explicit button
        # of its own in the detail panel.
        self._grid.recipeSelected.connect(self._on_grid_selected)
        self._grid.recipeActivated.connect(self._on_grid_selected)
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

    # -- library -----------------------------------------------------------------
    def _reload_library(self) -> None:
        # 先定下「现在按哪种材质配色」，再画清单 (m10789 item 1)：一种材质下的卷被删光
        # 时 _refresh_material_combo 会把材质切回第一种，如果清单还按已经不在的那个名字
        # 过滤，左边会空着说「当前材质「PETG」下还没有耗材」，右边却在按新材质算颜色。
        dropped = self._refresh_material_combo()
        self._suppress_list_signal = True
        self._list.clear()
        # The spools are about to be replaced, so the search haystacks keyed by
        # filament id are stale.
        self._haystacks.clear()
        for filament in self.library:
            if not self.library.accepts(filament):
                # 「我的耗材」只列当前材质的颜色（m09456 item 2）。以前这里把别的
                # 材质的卷也灰着排一行（后缀「（PLA Basic，现在不配色）」），用户
                # 的要求是别显示：换上面的材质下拉，它们整份回来。
                continue
            # An unnamed spool's display_name IS its hex, so printing both would
            # render "#000000   #000000". The one other thing the user needs is
            # the numbers, so the duplicate becomes "RGB 0, 0, 0".
            detail = filament.color_hex
            if filament.display_name.strip().casefold() == filament.color_hex.casefold():
                red, green, blue = filament.rgb
                detail = f"RGB {red}, {green}, {blue}"
            # 行留着但灰掉：这卷料只是不参与混色，不是被删掉了。
            suffix = "（已隐藏）" if filament.hidden else ""
            item = QListWidgetItem(
                swatch_icon(filament.color_hex, 28, 28),
                f"{filament.display_name}   {detail}" + (f"   {suffix}" if suffix else ""),
            )
            item.setData(Qt.ItemDataRole.UserRole, filament.id)
            if filament.hidden:
                # 灰掉，让「这卷现在不算数」一眼看得出来。
                item.setForeground(QBrush(QColor(theme.TEXT_FAINT)))
            if not filament.hidden and self._pair_filter and filament.id in self._pair_filter:
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            self._list.addItem(item)
        self._suppress_list_signal = False

        has_any = self._list.count() > 0
        # 空的原因有两种，说错了会让人以为库里真的一卷都没有：这一份材质下没有，
        # 和整个库都空。别的材质的卷数照着「另有 N 卷」说出来（m09456 item 2）。
        others = self.library.other_material_count
        if not has_any and self.library.material and others:
            self._library_hint.setText(
                f"当前材质「{self.library.material}」下还没有耗材。"
                f"另有 {others} 卷是别的材质，换上面的材质下拉就能看到它们。"
            )
        else:
            self._library_hint.setText(
                "还没有耗材。点「添加耗材」输入颜色、种类和品牌；"
                "至少两种同种类的耗材就能开始混色。"
            )
        self._library_hint.setVisible(not has_any)
        self._list.setVisible(has_any)
        self._clear_button.setEnabled(has_any)
        self._rebuild_catalog()
        self._save_library()
        if dropped:
            # 材质被换掉的原因要说出来 (m10789 item 1)：删完最后一卷后如果只把下拉
            # 悄悄换成第一种，用户看到的是「材质下拉坏了，切不回原来那种」——而实际
            # 上是那种材质在库里已经一卷都不剩，下拉里没有它才是对的。
            if self.library.material:
                self._material_note = (
                    f"「{dropped}」下的耗材已经全部删除，"
                    f"已切回「{self.library.material}」（{len(self.library.visible)} 卷参与配色）"
                )
            else:
                self._material_note = f"「{dropped}」下的耗材已经全部删除，耗材库已经空了"
            self._update_status()
        else:
            self._material_note = ""

    def _refresh_material_combo(self) -> str | None:
        """Fill the 材质 picker from the library, keeping the current choice.

        The counts ride in the label because 「PETG HF（3）」 answers the question
        the switch is about — how many spools of that material will actually be
        spent — without opening anything (m05557 item 1).

        Returns the 耗材种类 we had to leave, or ``None``.  A material whose every
        spool is gone is not in the list any more (``materials()`` counts spools),
        so the picker cannot show it: the choice falls back to the first material
        and that fallback is written back to the settings.  Without the write-back
        the next launch would read the deleted material out of the settings again
        and the window would come up claiming 「当前材质「PETG」下还没有耗材」 while
        the list is full — the state the user read as 「材质无法切换了」 (m10789 item 1).
        """
        names = self.library.materials()
        dropped: str | None = None
        self._material_guard = True
        try:
            self._material_combo.clear()
            if not names:
                if self.library.material:
                    dropped = self.library.material
                    self.library.material = ANY_MATERIAL
                    settings.save_settings({"material": ANY_MATERIAL})
                self._material_combo.addItem("—", ANY_MATERIAL)
                self._material_combo.setEnabled(False)
                return dropped
            self._material_combo.setEnabled(True)
            counts: dict[str, int] = {}
            for filament in self.library:
                if not filament.hidden:
                    key = (filament.material_type or "").strip()
                    counts[key] = counts.get(key, 0) + 1
            for name in names:
                self._material_combo.addItem(f"{name}（{counts.get(name, 0)}）", name)
            index = self._material_combo.findData(self.library.material)
            if index < 0:
                index = 0
                dropped = self.library.material or None
                self.library.material = self._material_combo.itemData(0) or ANY_MATERIAL
                settings.save_settings({"material": self.library.material})
            self._material_combo.setCurrentIndex(index)
        finally:
            self._material_guard = False
        return dropped

    def _on_material_changed(self, index: int) -> None:
        """切换材质 = 换一份耗材清单，所有颜色跟着重来一遍 (m05557 item 1).

        Nothing else has to know: the catalogues, the triples table and the
        picture page are all built from ``library.visible``, so moving that one
        line moves every colour on every page.
        """
        if self._material_guard or index < 0:
            return
        material = self._material_combo.itemData(index) or ANY_MATERIAL
        if material == self.library.material:
            return
        self.library.material = material
        settings.save_settings({"material": material})
        self._refresh_material_combo()
        self._reload_library()

    def _save_library(self) -> None:
        try:
            self.library.save(paths.library_path())
        except (LibraryError, OSError) as exc:
            # OSError covers DataDirectoryError: no usable location at all.
            self._set_status_text(f"保存失败：{exc}")

    def _on_clear_library(self) -> None:
        """Throw the whole archive away — the only honest way to start empty.

        Spools are not shipped with the program and are not a preset: they live
        in one archive (``paths.library_path()``) that belongs to *this* copy —
        ``data\\filament-library.json`` beside the executable for a packaged
        build.  The confirmation shows the user that file, so starting over does
        not mean hunting for a path Explorer never showed them.
        """
        count = len(self.library)
        if count == 0:
            return
        try:
            location = paths.library_path()
        except OSError as exc:
            location = f"（路径不可用：{exc}）"
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("清空耗材库")
        box.setText(f"要删掉「我的耗材」里全部 {count} 条耗材吗？")
        # Named buttons instead of the stock Yes / No: the safe answer has to be
        # the one that says what it does, and it is the one Enter presses.
        box.setInformativeText(
            f"耗材档案：\n{location}\n\n"
            "它就在这一份程序自己的数据文件夹里——不跟着安装包发布，也不和别的"
            "副本共用。复制或解压一份新的程序，那一份从空库开始；想让它接着用，"
            "先把这里的耗材「导出」，再到那一份里「导入」。\n\n"
            "清空之后不能撤销；想留个备份就先点「导出」。"
        )
        clear = box.addButton("清空", QMessageBox.ButtonRole.DestructiveRole)
        cancel = box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(cancel)
        box.exec()
        if box.clickedButton() is not clear:
            return
        removed = len(self.library)
        self.library.clear()
        self._reload_library()
        self._set_status_text(
            f"已清空耗材库（{removed} 条）。接下来点「添加耗材」新建，或「导入」一份档案。"
        )

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
            self._set_status_text(f"已添加 {filament.display_name}（{filament.color_hex}）")
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
        self._set_status_text(f"已删除 {filament.display_name}（{filament.color_hex}）")
        self._reload_library()

    def _select_filament(self, filament_id: str) -> None:
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == filament_id:
                self._list.setCurrentItem(item)
                return

    def _on_list_selection(self, current, previous) -> None:
        """选中一卷料不再把它的名称/品牌/种类/备注念一遍 (m09456 item 3).

        这一行原来是悬停气泡的替身：行里只放得下名称和 #RRGGBB，品牌、种类、备注
        就搬到状态栏常驻。用户要求删掉——选中哪一卷不该让底部换成一串长句，那些
        事实在右侧详情面板里逐项列着，底部留给「全部颜色」的表格说明。
        """
        if self._suppress_list_signal:
            return
        self._update_status()

    def _on_list_menu(self, position) -> None:
        from PySide6.QtWidgets import QMenu

        filament = self._selected_filament()
        if filament is None:
            return
        menu = QMenu(self)
        if filament.hidden:
            menu.addAction("取消隐藏（重新参与混色）", lambda: self._set_hidden(filament, False))
        else:
            menu.addAction("隐藏（不参与混色）", lambda: self._set_hidden(filament, True))
        menu.addSeparator()
        menu.addAction("编辑…", self._on_edit)
        menu.addAction("复制一条", lambda: self._on_duplicate(filament))
        menu.addSeparator()
        menu.addAction("删除", self._on_delete)
        menu.exec(self._list.mapToGlobal(position))

    def _set_hidden(self, filament: Filament, hidden: bool) -> None:
        """藏起一卷料，或者把它放回来。

        隐藏不是删除：这一行还在表里，用户还得看得见它才能在想用的时候放回来。
        变的是它贡献的颜色 —— 它不再是任何配方的原料，自己那一色也不进「全部
        颜色」。 做法是让所有配方表都从 ``library.visible`` 建起来（而不是在
        匹配器、网格、全部颜色、导出里各过滤一遍，那四处迟早会走岔）。

        三色混色是跟着耗材清单算的，缓存键也来自那份清单，所以藏一卷料就相当于
        换了一个耗材库：``_rebuild_catalog`` 里的 ``_invalidate_triples`` 会把
        旧表丢掉（它里面还有这卷料），``_ensure_triples`` 再按新的键找缓存或重
        算。 藏一卷料后第一次打开要等的那几分钟，就是重算三色混色的时间。
        """
        if filament.hidden == hidden:
            return
        self.library.update(filament.id, hidden=hidden)
        if self._pair_filter and filament.id in self._pair_filter:
            # 「只看这一对」里挂着它，格子会全空；先放开筛选再说。
            self._pair_filter = None
        self._reload_library()
        self._select_filament(filament.id)
        if hidden:
            self._set_status_text(
                f"已隐藏 {filament.display_name}（{filament.color_hex}）："
                "它不再参与混色，颜色也不在「全部颜色」里。右键可以放回来。"
            )

    def _on_duplicate(self, filament: Filament) -> None:
        copy = filament.copy(name=(filament.name or filament.display_name) + " 副本")
        self.library.add(copy)
        self._reload_library()
        self._select_filament(copy.id)

    def _import_start_dir(self) -> Path:
        """Where 「导入」 opens: our own data folder, or the old shared one.

        Before 1.6.11 every copy read and wrote one library in the profile.  A
        copy that has not been given spools yet opens at that file's folder
        instead of a path Explorer never showed, because the spools in it are the
        user's — importing them is one click from there.
        """
        try:
            ours = paths.library_path()
        except OSError:
            return paths.default_start_dir()
        legacy = paths.legacy_shared_library()
        if legacy is not None and not ours.is_file():
            return legacy.parent
        return paths.default_start_dir()

    def _on_import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "导入耗材档案", str(self._import_start_dir()), "JSON 文件 (*.json)"
        )
        if not path:
            return
        try:
            incoming = FilamentLibrary.load(Path(path))
        except LibraryError as exc:
            QMessageBox.warning(self, "导入失败", str(exc))
            return
        added, skipped = self._merge_library(incoming)
        # 档案旁边那份三色混色缓存要在重建耗材库之前就登记好：重建会顺手开始算
        # 三色（见 _ensure_triples），而这份档案很可能自带现成的表——先算再丢掉，
        # 用户就要白等几分钟，鼠标还会一直转圈。
        sidecar = mix_sidecar_path(Path(path))
        self._pending_sidecar = sidecar if sidecar.is_file() else None
        self._reload_library()
        note = ""
        if self._pending_sidecar is not None and self._catalog is not None:
            adopted = self._adopt_mix_sidecar(self._pending_sidecar)
            self._pending_sidecar = None
            if adopted:
                note = "\n同时用上了档案里的三色混色缓存，不用重算。"
            else:
                # 表不是这套耗材的：丢掉这次登记，回到正常路径去算。
                self._ensure_triples()
        else:
            self._pending_sidecar = None
        skipped_note = f"跳过 {skipped} 条已存在的。" if skipped else ""
        QMessageBox.information(
            self, "导入完成", f"已导入 {added} 条耗材。{skipped_note}{note}"
        )

    def _merge_library(self, incoming) -> tuple[int, int]:
        """把一份档案并进当前耗材库，返回 (新增条数, 跳过条数)。

        1.6.59（m26980 第 2 条）：同一条耗材导第二次不该再长出一条来。以前只要 id
        撞了就给一份副本（副本带新 id），于是「再导入一次自己的档案」会让耗材表凭空
        多出几条一模一样的 —— 用户看到的「耗材 3、4、5 从哪来的」就是这么来的。
        现在先按「颜色 + 种类 + 品牌」认人：已经有了就跳过；真的只是 id 撞了（内容
        不同）才换 id 加进去。
        """
        added = 0
        skipped = 0
        for filament in incoming:
            same = self.library.find_duplicate(
                filament.color_hex, filament.material_type, filament.brand
            )
            if same is not None:
                skipped += 1
                continue
            if self.library.get(filament.id) is not None:
                filament = filament.copy()
            self.library.add(filament)
            added += 1
        return added, skipped

    def _on_export(self) -> None:
        default = paths.default_start_dir() / "filament-library-export.json"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出耗材档案",
            str(default),
            "耗材档案 (*.json);;Blender 材质样板 (*.obj);;Blender 材质脚本 (*.py)",
        )
        if not path:
            return
        target = Path(path)
        if target.suffix.lower() == ".obj":
            # 1.6.59（m26980 第 1 条）：拖进 Blender 就能用的一种 — OBJ + MTL，
            # 每卷耗材一块小方片、各自一个材质。比「开文本编辑器跑脚本」省事。
            from ..core.library_export import write_swatch  # noqa: PLC0415

            written, mtl = write_swatch(target, list(self.library.visible))
            QMessageBox.information(
                self,
                "导出完成",
                f"已写入\n{written}\n（材质表 {mtl.name}）\n\n"
                "把这个 .obj 直接拖进 Blender 窗口（它旁边的 .mtl 会被一起读进来），"
                "每卷耗材就是一块小方片，材质名是「序号 名称 颜色」，拿来给模型上色即可。",
            )
            return
        if target.suffix.lower() == ".py":
            # 1.6.57（m26604 第 7 条）：导出一份在 Blender 里跑一下就有这些材质的脚本。
            from ..core.blender_bridge import material_script  # noqa: PLC0415

            entries = list(self.library.visible)
            target.write_text(material_script(entries), encoding="utf-8")
            QMessageBox.information(
                self,
                "导出完成",
                f"已写入\n{target}\n\n在 Blender 的「文本编辑器」里打开它、按「运行脚本」，"
                f"就会出现这 {len(entries)} 个耗材材质（颜色已按线性空间换算，跟这里看到的一致）。",
            )
            return
        try:
            self.library.save(target)
        except LibraryError as exc:
            QMessageBox.warning(self, "导出失败", str(exc))
            return
        note = ""
        written = self._write_mix_sidecar(target)
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
                if stored != triple_cache_key(self.library.visible, self._catalog.engine.id):
                    return False
                payload = {name: data[name] for name in data.files if name != "key"}
        except Exception:
            return False
        try:
            catalog = TripleCatalog.loads(
                self.library.visible, payload, self._catalog.engine.id
            )
        except Exception:
            return False
        if not catalog.built:
            return False
        # 这份表就是答案：正在算的那份（如果刚被 _reload_library 起过）立刻停掉，
        # 不然那个计时器会继续往一张没有中间数组的表上写，进度文字和转圈鼠标也
        # 会一直挂在那儿下不来。
        self._stop_triple_timer()
        self._triples = catalog
        self._rebuild_combined()
        try:
            save_triple_cache(catalog, self._triples_path())
        except Exception:
            pass
        self._end_build_notice()
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
            self._catalog = MixCatalog(self.library.visible, MIX_RATIOS, engine=engine_id)
            self._catalog.build()
        else:
            # Adding or deleting one spool only needs that spool's pairs; a full
            # rebuild is 66,420 recipes and ~3 s at 41 spools.
            catalog.sync(self.library.visible)
        self._build_seconds = time.perf_counter() - started
        self._invalidate_triples()
        self._end_build_notice()
        self._detail.setContext(self.library, self._catalog.engine)
        if getattr(self, "_picture_page", None) is not None:
            self._picture_page.setLibrary(self.library, self._catalog)
        if getattr(self, "_model_page", None) is not None:
            self._model_page.setLibrary(self.library, self._catalog)
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
        count = len(self.library.visible)
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
        """三色混色缓存在哪：一份耗材清单 + 一个引擎一个文件。

        清单取 ``library.visible``：藏起一卷料就是换了一份清单，缓存键跟着变，
        旧缓存里那份还带着这卷料的表就自然对不上了。
        """
        engine_id = self._catalog.engine.id if self._catalog is not None else DEFAULT_ENGINE
        key = triple_cache_key(self.library.visible, engine_id)
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
                self._catalog, self._triples, self.library.visible
            )
        self._combined_view = None
        self._combined_key = None
        # The picture page may only offer three-filament colours once the table
        # behind them is really finished — an unfinished catalogue has no
        # ``packed`` array yet.  This is also the one place that knows both the
        # table and the library it was built from, so the hand-over happens here
        # rather than at each of the three call sites above.
        ready = self._triples if self._triples is not None and self._triples.built else None
        self._picture_page.setTriples(ready, self.library)
        if getattr(self, "_model_page", None) is not None:
            self._model_page.setTriples(ready, self.library)

    def _ordered_combined(self, sort_key: str):
        """当前排序下的懒视图；排好的那一次留着，来回切设置不再重排。"""
        target = tuple(self._target_color.rgb())
        signature = (sort_key, target, len(self.library.visible))
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
            for filament in self.library.visible
            if query.casefold() in self._filament_haystack(filament)
        ]
        if not wanted:
            return np.empty(0, dtype=np.int64)
        return self._combined.search_spools(wanted)

    def _ensure_triples(self) -> None:
        """三色混色不用谁去勾选：有 3 卷料就该有它，先看缓存，没有就算。"""
        if not self._auto_triples:
            return
        if self._catalog is None or len(self.library.visible) < 3:
            return
        if self._triples is not None or self._triple_timer is not None:
            return
        if self._pending_sidecar is not None:
            # 正在导入一份自带混色表的档案：几毫秒后就会换上它，现在开始算是白算。
            return
        cached = load_triple_cache(
            self._triples_path(), self.library.visible, self._catalog.engine.id
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
        catalog = TripleCatalog(self.library.visible, engine=self._catalog.engine.id)
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
        """Write the bottom line.

        The label wraps, so a long sentence folds onto a second line instead of
        forcing the window wider or being clipped.  It used to be duplicated into
        a hover pop-up as well; the pop-up is gone, so the wrapped label is the
        whole channel and the sentences here stay short enough to read in it.

        An empty line hides the label instead of holding an empty row open
        (m14748 第 3 条): the 「图像转换」页自己带一行状态，切过来时这条就是空的，
        那行高度全变成窗口底部的白边。藏起来之后省下的高度归上面的分栏 —— 图片
        和色块表都能多占一点。
        """
        self._status.setText(text)
        self._status.setVisible(bool(text.strip()))

    def _update_status(self) -> None:
        count = len(self.library.visible)
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
        hidden = self.library.hidden_count
        if hidden:
            # 藏起来的料不在这条统计里，就得说出来，否则「耗材本色只有 40 种」
            # 会像是丢了一卷。
            note += f" · 另有 {hidden} 卷已隐藏（不参与混色）"
        other_material = self.library.other_material_count
        if other_material:
            # 别的材质的卷同样不在这条统计里：不说出来，「耗材本色只有 4 种」
            # 会像是库里少了十几卷。切换材质它们就回来 (m05557 item 1)。
            note += (
                f" · 现在按 {self.library.material} 配色，"
                f"另有 {other_material} 卷是别的材质"
            )
        if self._material_note:
            note += f" · {self._material_note}"
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
    def _on_find_nearest(self) -> None:
        """Find the closest colour, borrowing a target when the user typed none.

        「找最接近的颜色」 has to DO something every time it is pressed: with an
        empty 目标颜色 the old code only wrote a hint, which reads as a broken
        button.  So the target is taken from, in order: the colour the user is
        looking at in the table, then the R / G / B numbers as they stand.
        软件打开时这个字段已经是纯黑（``DEFAULT_TARGET_HEX``），所以「借用」这一步
        只有在用户自己把目标颜色清空之后才会用到（m02366 item 1）。

        先 ``commitPending()``（1.6.27）：在目标颜色框里打完 hex 立刻按这个按钮时，
        焦点还在输入框里，值没提交——搜到的会是上一个颜色。
        """
        self._target_color.commitPending()
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
        """版本与算法出处，「帮助」窗口的最后一节。

        从 ``_on_help`` 里拆出来，自动检查才能不弹窗读到它：这里显示的版本号和
        exe 文件属性里的那个都取自 ``app.__version__``。
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

    def _help_text(self) -> str:
        """「帮助」窗口里的全部文字。

        三节：图片这一步怎么用（原来压在图片上的那行，m01885 第 6 条）、
        「耗材方案」每一行按什么标准挑的（原来是行下面那句，第 8 条）、
        以及本来就在这个窗口里的版本与算法出处。
        """
        return (
            f"{self._picture_page.help_text()}\n\n"
            f"{self._model_page.help_text()}\n\n"
            f"{plan_help()}\n\n"
            "算法与版本：\n"
            f"{self._about_text()}"
        )

    def _on_help(self) -> None:
        if self._help_window is None:
            self._help_window = HelpDialog(self._help_text(), self)
        else:
            self._help_window.set_text(self._help_text())
        self._help_window.show()
        self._help_window.raise_()
        self._help_window.activateWindow()

    def _on_settings(self) -> None:
        """「设置」：三页的默认参数与快捷键（1.6.53，m21616 第 8 条）。

        保存之后除了落盘，还当场灌进两个页面 —— 用户刚改完「颜色数量 24」，切回图像转换
        却看到还是 32，那是最容易让人以为设置没生效的一种错。已经在屏幕上的图不会被重算：
        页面只接值，不下命令（`apply_settings` 里写清了）。
        """
        dialog = SettingsDialog(settings.load_settings(), self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        values = dialog.values()
        try:
            settings.save_settings(values)
        except OSError as error:
            QMessageBox.warning(self, "设置没能保存", f"写不进偏好文件：{error}")
            return
        self._picture_page.apply_settings(values)
        self._model_page.apply_settings(values)
        # 1.6.54（m22211 第 7 条）：颜色配方页那两个快捷键也归设置管。
        for shortcut, key, fallback in (
            (self._find_shortcut, "shortcut_recipe_find", "Ctrl+F"),
            (self._add_shortcut, "shortcut_recipe_add", "Ctrl+N"),
        ):
            shortcut.setKey(QKeySequence(str(values.get(key) or fallback)))
        index = self._sort_combo.findData(str(values.get("sort") or "rgb"))
        if index >= 0 and index != self._sort_combo.currentIndex():
            self._sort_combo.setCurrentIndex(index)
        self._status.setText("设置已保存，下次打开按这一套来。")

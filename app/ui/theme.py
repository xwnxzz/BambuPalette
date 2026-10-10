"""Visual design tokens and the application stylesheet.

The palette, radii and type scale deliberately follow Lumina Studio's own
workshop-module design language (``src/style.css`` in the 创意工坊 project, whose
public theme tokens are ``--lumina-*``): a near-white surface set, a single blue
accent, 10 px cards / 7 px buttons / 6 px inputs, 13 px body type, and small
muted section labels.

Lumina exposes those tokens so a module can inherit the host theme. We cannot
read them from the host — this is a standalone desktop app — so the fallback
values Lumina itself ships are hard-coded here and every widget reads them from
this module rather than from a literal.
"""

from __future__ import annotations

# -- colours ---------------------------------------------------------------------------
# Surfaces
BG = "#F5F5F7"  # --lumina-surface-muted: the stage / page background
PANEL = "#FFFFFF"  # --lumina-surface: cards, bars, inputs at rest
PANEL_ALT = "#FAFAFC"  # a half-step between the two, for hover / input fills
# Lines
BORDER = "#D2D2D7"  # --lumina-border
BORDER_STRONG = "#C7C7CC"
# Type
TEXT = "#1D1D1F"  # --lumina-text
TEXT_MUTED = "#6E6E73"  # --lumina-text-muted
TEXT_FAINT = "#AEAEB2"
# Action
ACCENT = "#0071E3"  # --lumina-accent
ACCENT_DARK = "#0058B0"
ACCENT_SOFT = "#E8F1FD"
SELECTION = "#E8F1FD"
SELECTION_BORDER = "#0071E3"
WARNING = "#D9822B"
DANGER = "#D7392F"  # --ls-danger

#: Stroke colours used to outline very light swatches so they stay visible.
SWATCH_BORDER_LIGHT = "#D2D2D7"
SWATCH_BORDER_DARK = "#2A3038"

#: Corner radii, kept as strings so they can be dropped straight into the sheet.
RADIUS_CARD = "10px"
RADIUS_CONTROL = "7px"
RADIUS_INPUT = "6px"

MONO_FONT = "Consolas, 'Cascadia Mono', 'SFMono-Regular', 'Courier New', monospace"
UI_FONT = "'Segoe UI', 'Microsoft YaHei UI', 'PingFang SC', 'Hiragino Sans GB', sans-serif"

STYLESHEET = f"""
QWidget {{
    color: {TEXT};
    font-family: {UI_FONT};
    font-size: 13px;
}}
QMainWindow, QDialog {{
    background: {BG};
}}

/* ---------------------------------------------------------------- top bars */
QFrame#topbar {{
    background: {PANEL};
    border: none;
    border-bottom: 1px solid {BORDER};
}}
QFrame#stagebar {{
    background: {PANEL};
    border: none;
    border-top: 1px solid {BORDER};
    border-bottom: 1px solid {BORDER};
}}
QFrame#panel {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: {RADIUS_CARD};
}}
/* Says 「你正在只看一对耗材」 and carries the way back out. */
QFrame#filterBar {{
    background: {ACCENT_SOFT};
    border: 1px solid {ACCENT};
    border-radius: {RADIUS_INPUT};
}}
QFrame#filterBar QLabel {{
    color: {ACCENT_DARK};
    font-size: 12px;
}}
QStatusBar {{
    background: {PANEL};
    border-top: 1px solid {BORDER};
    color: {TEXT_MUTED};
    font-size: 12px;
}}
QStatusBar::item {{
    border: none;
}}

/* ------------------------------------------------------------------- menus */
QMenuBar {{
    background: transparent;
}}
QMenuBar::item {{
    background: transparent;
    padding: 4px 10px;
    border-radius: {RADIUS_INPUT};
}}
QMenuBar::item:selected {{
    background: {PANEL_ALT};
}}
QMenu {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 4px;
}}
QMenu::item {{
    padding: 6px 22px 6px 12px;
    border-radius: {RADIUS_INPUT};
}}
QMenu::item:selected {{
    background: {ACCENT_SOFT};
    color: {TEXT};
}}
QMenu::separator {{
    height: 1px;
    background: {BORDER};
    margin: 4px 8px;
}}
QToolBar {{
    background: {PANEL};
    border: none;
    border-bottom: 1px solid {BORDER};
    padding: 4px 6px;
    spacing: 4px;
}}
QToolBar QToolButton {{
    padding: 5px 10px;
    border-radius: {RADIUS_INPUT};
    border: 1px solid transparent;
}}
QToolBar QToolButton:hover {{
    background: {PANEL_ALT};
    border-color: {BORDER};
}}
QToolBar QToolButton:pressed {{
    background: {ACCENT_SOFT};
}}
QToolBar QToolButton:disabled {{
    color: {TEXT_FAINT};
}}

/* ------------------------------------------- tabs, drawn as a segmented control */
QTabWidget::pane {{
    background: transparent;
    border: none;
    top: -1px;
}}
QTabWidget > QWidget {{
    background: transparent;
}}
QTabBar {{
    background: transparent;
    qproperty-drawBase: 0;
}}
QTabBar::tab {{
    background: {PANEL};
    color: {TEXT_MUTED};
    border: 1px solid {BORDER};
    border-radius: 0;
    padding: 6px 18px;
    margin-right: -1px;
    font-size: 12px;
}}
QTabBar::tab:first {{
    border-top-left-radius: {RADIUS_CONTROL};
    border-bottom-left-radius: {RADIUS_CONTROL};
}}
QTabBar::tab:last {{
    border-top-right-radius: {RADIUS_CONTROL};
    border-bottom-right-radius: {RADIUS_CONTROL};
    margin-right: 0;
}}
QTabBar::tab:hover {{
    color: {TEXT};
}}
QTabBar::tab:selected {{
    background: {ACCENT};
    color: white;
    border-color: {ACCENT};
    font-weight: 600;
}}

/* ------------------------------------------------------------------ groups */
QGroupBox {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: {RADIUS_CARD};
    margin-top: 12px;
    padding-top: 10px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: {TEXT_MUTED};
    font-size: 12px;
}}

/* ------------------------------------------------------------------ inputs */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit {{
    background: {PANEL_ALT};
    border: 1px solid {BORDER};
    border-radius: {RADIUS_INPUT};
    padding: 5px 8px;
    font-size: 12px;
    selection-background-color: {ACCENT};
    selection-color: white;
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus,
QPlainTextEdit:focus, QTextEdit:focus {{
    background: {PANEL};
    border-color: {ACCENT};
}}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
    background: {BG};
    border-color: {BORDER};
    color: {TEXT_FAINT};
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QComboBox QAbstractItemView {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 4px;
    selection-background-color: {ACCENT_SOFT};
    selection-color: {TEXT};
    outline: none;
}}

/* ----------------------------------------------------------------- buttons */
QPushButton {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: {RADIUS_CONTROL};
    padding: 6px 12px;
    min-height: 18px;
    font-size: 12px;
}}
QPushButton:hover:enabled {{
    border-color: {TEXT_MUTED};
}}
QPushButton:pressed {{
    background: {PANEL_ALT};
}}
QPushButton:disabled {{
    background: {PANEL};
    border-color: {BORDER};
    color: {TEXT_FAINT};
}}
QPushButton[accent="true"] {{
    background: {ACCENT};
    border: 1px solid {ACCENT};
    color: white;
    font-weight: 600;
}}
QPushButton[accent="true"]:hover:enabled {{
    background: {ACCENT_DARK};
    border-color: {ACCENT_DARK};
}}
QPushButton[accent="true"]:disabled {{
    background: {BORDER};
    border-color: {BORDER};
    color: {PANEL};
}}
QPushButton[danger="true"] {{
    color: {DANGER};
}}
QPushButton[danger="true"]:hover:enabled {{
    border-color: {DANGER};
    color: {DANGER};
}}

/* ------------------------------------------------------------------- lists */
QListWidget, QListView, QTreeView, QTableView {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 8px;
    outline: none;
}}
QListWidget::item {{
    padding: 3px;
    border-radius: {RADIUS_INPUT};
    margin: 1px 2px;
}}
QListWidget::item:hover, QListView::item:hover {{
    background: {PANEL_ALT};
}}
QListWidget::item:selected, QListView::item:selected {{
    background: {ACCENT_SOFT};
    color: {TEXT};
}}
QHeaderView::section {{
    background: {PANEL_ALT};
    border: none;
    border-bottom: 1px solid {BORDER};
    border-right: 1px solid {BORDER};
    padding: 5px 8px;
    font-weight: 600;
    color: {TEXT_MUTED};
}}

/* -------------------------------------------------------------- scrollbars */
QScrollBar:vertical {{
    background: transparent;
    width: 11px;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: {BORDER_STRONG};
    border-radius: 5px;
    min-height: 28px;
}}
QScrollBar::handle:vertical:hover {{
    background: {TEXT_FAINT};
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 11px;
    margin: 2px;
}}
QScrollBar::handle:horizontal {{
    background: {BORDER_STRONG};
    border-radius: 5px;
    min-width: 28px;
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    height: 0;
    width: 0;
}}
QScrollBar::add-page, QScrollBar::sub-page {{
    background: transparent;
}}

/* ---------------------------------------------------------- splitters, tips */
QSplitter::handle {{
    background: transparent;
}}
QSplitter::handle:horizontal {{
    width: 10px;
}}
QSplitter::handle:hover {{
    background: {BORDER};
}}
QToolTip {{
    background: {TEXT};
    color: {PANEL};
    border: none;
    padding: 6px 9px;
    border-radius: {RADIUS_INPUT};
    font-size: 12px;
}}

/* ------------------------------------------------------------------ labels */
QLabel[role="sectionTitle"] {{
    font-size: 12px;
    font-weight: 600;
    color: {TEXT_MUTED};
}}
QLabel[role="hint"] {{
    font-size: 12px;
    color: {TEXT_MUTED};
}}
QLabel[role="mono"] {{
    font-family: {MONO_FONT};
}}
QCheckBox {{
    spacing: 6px;
    font-size: 12px;
    min-height: 18px;
}}
/* Windows 11's style paints a filled accent square for the checked state and
   then clips the tick into whatever room is left. At the 12px label size the
   auto-sized indicator collapses to ~16px, the tick vanishes, and a ticked box
   becomes indistinguishable from a colour chip. Pinning the indicator restores
   an unmistakable tick without giving up the small label. */
QCheckBox::indicator {{
    width: 15px;
    height: 15px;
}}
QCheckBox::indicator:disabled {{
    background: {PANEL_ALT};
    border: 1px solid {BORDER};
    border-radius: 3px;
}}
"""


def relative_luminance(red: int, green: int, blue: int) -> float:
    """WCAG relative luminance of an sRGB colour, 0..1."""
    channels = []
    for value in (red, green, blue):
        srgb = value / 255.0
        channels.append(srgb / 12.92 if srgb <= 0.03928 else ((srgb + 0.055) / 1.055) ** 2.4)
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def contrasting_text(red: int, green: int, blue: int) -> str:
    """Black or white text, whichever reads better on the given swatch colour."""
    return "#111418" if relative_luminance(red, green, blue) > 0.45 else "#FFFFFF"

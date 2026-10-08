"""Visual design tokens and the application stylesheet."""

from __future__ import annotations

# -- colours ---------------------------------------------------------------------------
BG = "#F4F6F8"
PANEL = "#FFFFFF"
PANEL_ALT = "#FAFBFC"
BORDER = "#E2E6EB"
BORDER_STRONG = "#CBD2DA"
TEXT = "#1F2429"
TEXT_MUTED = "#6B7480"
TEXT_FAINT = "#98A1AC"
ACCENT = "#00A86B"
ACCENT_DARK = "#008C58"
ACCENT_SOFT = "#E6F5EE"
SELECTION = "#DCEBFB"
SELECTION_BORDER = "#2C7BE5"
WARNING = "#D9822B"
DANGER = "#D64545"

#: Stroke colours used to outline very light swatches so they stay visible.
SWATCH_BORDER_LIGHT = "#C9D0D8"
SWATCH_BORDER_DARK = "#2A3038"

MONO_FONT = "Consolas, 'Cascadia Mono', 'Courier New', monospace"
UI_FONT = "'Microsoft YaHei UI', 'Segoe UI', 'PingFang SC', sans-serif"

STYLESHEET = f"""
QWidget {{
    color: {TEXT};
    font-family: {UI_FONT};
    font-size: 13px;
}}
QMainWindow, QDialog {{
    background: {BG};
}}
QTabWidget::pane {{
    background: {BG};
    border: 1px solid {BORDER};
    border-radius: 8px;
    top: -1px;
}}
QTabWidget > QWidget {{
    background: {BG};
}}
QTabBar::tab {{
    background: transparent;
    color: {TEXT_MUTED};
    border: 1px solid transparent;
    border-bottom: none;
    border-top-left-radius: 7px;
    border-top-right-radius: 7px;
    padding: 7px 18px;
    margin-right: 4px;
}}
QTabBar::tab:hover {{
    color: {TEXT};
    background: {PANEL_ALT};
}}
QTabBar::tab:selected {{
    background: {BG};
    color: {TEXT};
    border-color: {BORDER};
    font-weight: 600;
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
    border-radius: 6px;
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
QStatusBar {{
    background: {PANEL};
    border-top: 1px solid {BORDER};
    color: {TEXT_MUTED};
}}
QGroupBox {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 10px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: {TEXT_MUTED};
}}
QLineEdit, QComboBox, QSpinBox, QPlainTextEdit, QTextEdit {{
    background: {PANEL};
    border: 1px solid {BORDER_STRONG};
    border-radius: 6px;
    padding: 4px 8px;
    selection-background-color: {SELECTION};
    selection-color: {TEXT};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QPlainTextEdit:focus {{
    border-color: {ACCENT};
}}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled {{
    background: {PANEL_ALT};
    color: {TEXT_FAINT};
}}
QComboBox::drop-down {{
    border: none;
    width: 18px;
}}
QComboBox QAbstractItemView {{
    background: {PANEL};
    border: 1px solid {BORDER_STRONG};
    selection-background-color: {ACCENT_SOFT};
    selection-color: {TEXT};
    outline: none;
}}
QPushButton {{
    background: {PANEL};
    border: 1px solid {BORDER_STRONG};
    border-radius: 6px;
    padding: 6px 14px;
    min-height: 18px;
}}
QPushButton:hover {{
    background: {PANEL_ALT};
    border-color: {ACCENT};
}}
QPushButton:pressed {{
    background: {ACCENT_SOFT};
}}
QPushButton:disabled {{
    color: {TEXT_FAINT};
    border-color: {BORDER};
}}
QPushButton[accent="true"] {{
    background: {ACCENT};
    color: white;
    border: 1px solid {ACCENT_DARK};
    font-weight: 600;
}}
QPushButton[accent="true"]:hover {{
    background: {ACCENT_DARK};
}}
QPushButton[accent="true"]:disabled {{
    background: {BORDER_STRONG};
    border-color: {BORDER_STRONG};
    color: white;
}}
QPushButton[danger="true"]:hover {{
    border-color: {DANGER};
    color: {DANGER};
}}
QListWidget, QListView, QTreeView, QTableView {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 8px;
    outline: none;
}}
QListWidget::item {{
    padding: 2px;
    border-radius: 6px;
}}
QListWidget::item:selected, QListView::item:selected {{
    background: {SELECTION};
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
QSplitter::handle {{
    background: {BORDER};
}}
QSplitter::handle:horizontal {{
    width: 3px;
}}
QToolTip {{
    background: #2B3138;
    color: #F2F4F6;
    border: none;
    padding: 4px 8px;
    border-radius: 4px;
}}
QLabel[role="sectionTitle"] {{
    font-size: 13px;
    font-weight: 600;
    color: {TEXT};
}}
QLabel[role="hint"] {{
    color: {TEXT_MUTED};
}}
QLabel[role="mono"] {{
    font-family: {MONO_FONT};
}}
QCheckBox {{
    spacing: 6px;
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

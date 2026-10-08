"""Entry point: build the Qt application and show the main window."""

from __future__ import annotations

import sys


def build_application(argv=None):
    """Create the ``QApplication`` with the app-wide theme applied."""
    from PySide6.QtGui import QColor, QPalette
    from PySide6.QtWidgets import QApplication

    from .ui import theme

    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("BambuPalette")
    app.setOrganizationName("BambuPalette")
    app.setStyle("Fusion")
    # An explicit light palette, so parts the stylesheet does not reach never
    # inherit the operating system's dark palette.
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(theme.BG))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(theme.TEXT))
    palette.setColor(QPalette.ColorRole.Base, QColor(theme.PANEL))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(theme.PANEL_ALT))
    palette.setColor(QPalette.ColorRole.Text, QColor(theme.TEXT))
    palette.setColor(QPalette.ColorRole.Button, QColor(theme.PANEL))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(theme.TEXT))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(theme.PANEL))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(theme.TEXT))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(theme.SELECTION))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(theme.TEXT))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(theme.TEXT_FAINT))
    palette.setColor(QPalette.ColorRole.Mid, QColor(theme.BORDER))
    palette.setColor(QPalette.ColorRole.Dark, QColor(theme.BORDER_STRONG))
    app.setPalette(palette)
    app.setStyleSheet(theme.STYLESHEET)
    return app


def _report_startup_failure(message: str) -> None:
    """Show a readable failure instead of a bare traceback.

    The packaged build is a windowed executable with no console, so an exception
    raised before the window exists would otherwise surface as an opaque
    PyInstaller error box (or nothing at all).
    """
    try:
        from PySide6.QtWidgets import QMessageBox

        box = QMessageBox()
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle("BambuPalette 无法启动")
        box.setText("BambuPalette 启动失败")
        box.setInformativeText(message)
        box.setDetailedText(message)
        box.exec()
    except Exception:  # pragma: no cover - no Qt available at all
        print(message, file=sys.stderr)


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--selftest" in args:
        from .selftest import run_selftest

        return run_selftest()

    app = build_application(argv)

    from .core.paths import DataDirectoryError
    from .ui.main_window import MainWindow

    try:
        window = MainWindow()
    except DataDirectoryError as exc:
        _report_startup_failure(str(exc))
        return 1
    except Exception as exc:  # noqa: BLE001 - last line of defence before the GUI
        import traceback

        _report_startup_failure(f"启动时发生未预期的错误：\n\n{traceback.format_exc()}")
        del exc
        return 1

    window.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

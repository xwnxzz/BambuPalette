"""「导出 SVG」窗口：左边原图、右边描摹预览，三格参数 + 重新描摹 / 保存 / 直接导入。

为什么要有预览（m29614 第 2 条）：描摹是「猜」，不看一眼就写文件，用户只能先导出再拿
浏览器开、发现不行再回来调 —— 一步能做的事变成三步。所以这个窗口先在临时目录描一遍、
把 SVG 渲成位图铺在右边，用户对比着看，满意了才落盘。

「直接导入」是把描好的 SVG 当成一张新图交给图像转换页（会再问一次渲成多少像素）——
想要矢量版的耗材板子时，这条路比「保存到磁盘再导入」少两步。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..core import paths, vectorize

PREVIEW_SIZE = QSize(420, 420)


def _label(text: str, role: str = "hint") -> QLabel:
    widget = QLabel(text)
    widget.setProperty("role", role)
    widget.setWordWrap(True)
    return widget


def render_svg(svg_path: Path, size: QSize = PREVIEW_SIZE) -> QPixmap:
    """把 SVG 渲成一张位图，铺在预览框里。"""

    canvas = QImage(size, QImage.Format.Format_ARGB32_Premultiplied)
    canvas.fill(Qt.GlobalColor.white)
    renderer = QSvgRenderer(str(svg_path))
    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    if renderer.isValid():
        # 按比例塞满，别把长条图拉变形。
        bounds = renderer.viewBoxF()
        if bounds.width() <= 0 or bounds.height() <= 0:
            bounds = QRectF(0, 0, size.width(), size.height())
        scale = min(size.width() / bounds.width(), size.height() / bounds.height())
        width = bounds.width() * scale
        height = bounds.height() * scale
        target = QRectF((size.width() - width) / 2, (size.height() - height) / 2, width, height)
        renderer.render(painter, target)
    painter.end()
    return QPixmap.fromImage(canvas)


def render_image(image: QImage | QPixmap, size: QSize = PREVIEW_SIZE) -> QPixmap:
    pixmap = image if isinstance(image, QPixmap) else QPixmap.fromImage(image)
    return pixmap.scaled(
        size,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


class SvgExportDialog(QDialog):
    """描摹预览窗口（1.6.65，m29614 第 2 条）。"""

    def __init__(self, page, name: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("导出 SVG（矢量图）")
        self.setMinimumWidth(980)

        self._page = page
        self._name = name or "这张图"
        self._svg_path: Path | None = None
        self._imported = False
        self._saved_to: Path | None = None
        self._folder = Path(tempfile.mkdtemp(prefix="bp-svg-"))

        self._source_png = self._folder / "source.png"
        self._has_source = self._write_source()

        # 颜色数量：默认「自动」（按图自己有多少色来），勾掉就手填。
        self._auto = QCheckBox("自动识别图片本身的颜色数量")
        self._auto.setChecked(True)
        self._colours = QSpinBox()
        self._colours.setRange(*vectorize.COLOUR_RANGE)
        self._detected = self._detect()
        self._colours.setValue(self._detected)
        self._colours.setSuffix(" 色")
        self._colours.setEnabled(not self._auto.isChecked())
        self._auto.toggled.connect(lambda on: self._colours.setEnabled(not on))

        self._speckle = QSpinBox()
        self._speckle.setRange(*vectorize.SPECKLE_RANGE)
        self._speckle.setValue(vectorize.DEFAULT_SPECKLE)
        self._speckle.setSuffix(" px")

        self._smooth = QDoubleSpinBox()
        self._smooth.setRange(*vectorize.SMOOTH_RANGE)
        self._smooth.setSingleStep(0.5)
        self._smooth.setValue(vectorize.DEFAULT_SMOOTH)
        self._smooth.setDecimals(1)

        self._retrace_button = QPushButton("重新描摹")
        self._retrace_button.clicked.connect(self._retrace)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("颜色数量"))
        controls.addWidget(self._colours)
        controls.addWidget(self._auto)
        controls.addSpacing(12)
        controls.addWidget(QLabel("去掉碎块"))
        controls.addWidget(self._speckle)
        controls.addSpacing(12)
        controls.addWidget(QLabel("曲线简化"))
        controls.addWidget(self._smooth)
        controls.addStretch(1)
        controls.addWidget(self._retrace_button)

        self._source_view = QLabel()
        self._source_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._source_view.setMinimumSize(PREVIEW_SIZE)
        if self._has_source:
            self._source_view.setPixmap(render_image(self._source_image(), PREVIEW_SIZE))
        else:
            self._source_view.setText("没有图")
        self._preview_view = QLabel("正在描摹…")
        self._preview_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_view.setMinimumSize(PREVIEW_SIZE)

        views = QHBoxLayout()
        views.addWidget(self._panel("原图", self._source_view), 1)
        views.addWidget(self._panel("描摹预览", self._preview_view), 1)

        self._status = _label("", "hint")

        buttons = QDialogButtonBox(self)
        self._save_button = buttons.addButton("保存 SVG…", QDialogButtonBox.ButtonRole.AcceptRole)
        self._import_button = buttons.addButton(
            "直接导入这张 SVG", QDialogButtonBox.ButtonRole.ActionRole
        )
        self._close_button = buttons.addButton("取消", QDialogButtonBox.ButtonRole.RejectRole)
        self._save_button.clicked.connect(self._on_save)
        self._import_button.clicked.connect(self._on_import)
        self._close_button.clicked.connect(self.reject)
        self._save_button.setEnabled(False)
        self._import_button.setEnabled(False)

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(
            _label(
                "描摹是「猜」：先在这儿对比一眼，满意了再保存或直接导入。"
                "颜色数量越多越像原图、文件越大；去掉碎块调大会把细小细节一起抹掉。",
                "hint",
            )
        )
        layout.addLayout(views, 1)
        layout.addWidget(self._status)
        layout.addWidget(buttons)

        if self._has_source:
            self._retrace()
        else:
            self._preview_view.setText("没有图可描")
            self._status.setText("先导入一张图片。")

    # ---- 取图与描摹 -------------------------------------------------

    def _source_image(self) -> QImage:
        from PIL import Image

        image = Image.open(self._source_png)
        if image.mode != "RGB":
            image = image.convert("RGB")
        data = image.tobytes("raw", "RGB")
        return QImage(data, image.width, image.height, image.width * 3, QImage.Format.Format_RGB888).copy()

    def _write_source(self) -> bool:
        """把页面当前这张图落到临时 PNG。没有图就返回 False（闸门里会这么构造）。"""

        picture = getattr(self._page, "_image", None)
        if picture is None:
            return False
        picture.convert("RGB").save(self._source_png)
        return True

    def _detect(self) -> int:
        if not self._has_source:
            return int(vectorize.DEFAULT_COLOURS)
        try:
            return vectorize.detect_colours(self._source_png)
        except Exception:  # noqa: BLE001 - 识别不出来就用默认值，不要拦住导出
            return vectorize.DEFAULT_COLOURS

    def _settings_now(self) -> dict:
        colours = self._detected if self._auto.isChecked() else int(self._colours.value())
        return {
            "colours": max(vectorize.COLOUR_RANGE[0], min(vectorize.COLOUR_RANGE[1], colours)),
            "speckle": int(self._speckle.value()),
            "smooth": float(self._smooth.value()),
        }

    def values(self) -> dict:
        return self._settings_now()

    def _retrace(self) -> None:
        if not vectorize.available():
            self._status.setText("描摹不可用：这个环境里没装 vtracer。")
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            target = self._folder / "preview.svg"
            self._result = vectorize.image_to_svg(self._source_png, target, **self._settings_now())
            self._svg_path = target
            self._preview_view.setPixmap(render_svg(target, PREVIEW_SIZE))
            self._status.setText(
                f"{self._result.paths} 条路径、{self._result.colours} 种颜色、"
                f"{max(1, self._result.bytes // 1024)} KB"
            )
            self._save_button.setEnabled(True)
            self._import_button.setEnabled(True)
        except Exception as error:  # noqa: BLE001 - 描摹失败要给人话，不要甩栈
            self._svg_path = None
            self._save_button.setEnabled(False)
            self._import_button.setEnabled(False)
            self._preview_view.setText("描摹失败")
            self._status.setText(f"描摹失败：{error}")
        finally:
            QApplication.restoreOverrideCursor()

    # ---- 两个出口 ---------------------------------------------------

    def _on_save(self) -> None:
        if self._svg_path is None:
            return
        start = getattr(self._page, "_last_save_dir", None) or paths.exports_dir()
        suggested = f"{getattr(self._page, '_image_name', '') or 'picture'}.svg"
        name, _filter = QFileDialog.getSaveFileName(
            self, "保存 SVG", str(Path(start) / suggested), "矢量图 (*.svg)"
        )
        if not name:
            return
        target = Path(name)
        if target.suffix.lower() != ".svg":
            target = target.with_suffix(".svg")
        target.write_bytes(self._svg_path.read_bytes())
        self._saved_to = target
        self._status.setText(f"已保存 {target.name}")
        self.accept()

    def _on_import(self) -> None:
        if self._svg_path is None:
            return
        self._imported = True
        self.accept()

    def svg_path(self) -> Path | None:
        """描好的那份 SVG（临时文件，生命周期跟着这个窗口）。"""

        return self._svg_path

    def saved_to(self) -> Path | None:
        """用户点了「保存 SVG」时写到哪了；取消或直接导入时为 None。"""

        return self._saved_to

    def import_requested(self) -> bool:
        return self._imported

    def done(self, result: int) -> None:  # noqa: D102 - Qt 回调
        # 既没保存也没导入就把临时目录清掉；导入的情况要留着，页面会按路径重读它。
        if not self._imported and self._saved_to is None:
            import shutil

            shutil.rmtree(self._folder, ignore_errors=True)
        super().done(result)

    # ---- 小工具 -----------------------------------------------------

    def _panel(self, title: str, widget: QWidget) -> QWidget:
        frame = QWidget(self)
        frame.setObjectName("panel")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(6)
        heading = QLabel(title)
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)
        layout.addWidget(widget, 1)
        return frame

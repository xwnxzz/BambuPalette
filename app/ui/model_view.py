"""模型预览：Blender 式操作的三维视图。

按键与鼠标照 Blender 的习惯来（中键转视角、Shift+中键平移、Ctrl+中键缩放、滚轮
缩放、小键盘 1/3/7 正/右/顶、Ctrl+1/3/7 背面那三个、5 切换透视与正交、9 转到当前
视角的反面、4/6/8/2 每次转 15°、Home 自适应）；右上角把那几个最常用的做成按钮，
免得用户去背小键盘。

不用 GPU 也能跑：离屏平台（闸门脚本、无 OpenGL 的机器）自动退回一个只画提示的
占位控件，页面的其它逻辑照常工作。
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QMatrix4x4, QPainter, QVector3D
from PySide6.QtOpenGL import QOpenGLBuffer, QOpenGLShader, QOpenGLShaderProgram, QOpenGLVertexArrayObject
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from ..core.glb import GlbModel

#: Blender 默认相机是 50 mm 镜头，竖直视角约 39.6°。
CAMERA_FOV = 39.6
#: 每次 4/6/8/2 转多少度（Blender 是 15°）。
VIEW_STEP = 15.0
#: 正/右/顶这些基准视角，单位是（yaw, pitch，度，Z 轴朝上）。
#: yaw = 0 时相机在 -Y 看 +Y，就是 Blender 的「正视图」。
VIEW_ANGLES: dict[str, tuple[float, float]] = {
    "front": (0.0, 0.0),
    "back": (180.0, 0.0),
    "right": (90.0, 0.0),
    "left": (-90.0, 0.0),
    "top": (0.0, 90.0),
    "bottom": (0.0, -90.0),
}
#: 界面上的名字（右下角那一排按钮）。
VIEW_LABELS: tuple[tuple[str, str], ...] = (
    ("front", "前"),
    ("back", "后"),
    ("left", "左"),
    ("right", "右"),
    ("top", "顶"),
    ("bottom", "底"),
)

_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec3 in_colour;
uniform mat4 u_mvp;
uniform mat3 u_normal;
out vec3 v_normal;
out vec3 v_colour;
void main() {
    v_normal = normalize(u_normal * in_normal);
    v_colour = in_colour;
    gl_Position = u_mvp * vec4(in_position, 1.0);
}
"""

_FRAGMENT_SHADER = """
#version 330 core
in vec3 v_normal;
in vec3 v_colour;
out vec4 frag_colour;
void main() {
    vec3 n = normalize(v_normal);
    if (!gl_FrontFacing) { n = -n; }
    vec3 key = normalize(vec3(-0.45, 0.55, 0.72));
    vec3 fill = normalize(vec3(0.65, -0.25, 0.45));
    vec3 rim = normalize(vec3(0.05, 0.95, -0.3));
    // 三盏灯都用绝对值：GLB 里的法线方向不一定朝外，模型背面也不该黑成一片。
    float light = 0.30
        + 0.55 * abs(dot(n, key))
        + 0.18 * abs(dot(n, fill))
        + 0.10 * abs(dot(n, rim));
    frag_colour = vec4(clamp(v_colour * light, 0.0, 1.0), 1.0);
}
"""


def _to_display(points: np.ndarray) -> np.ndarray:
    """glTF 是 Y 轴朝上，Blender 是 Z 轴朝上：转到 Z 轴朝上再给用户看。"""
    if points.size == 0:
        return points
    out = np.empty_like(points)
    out[:, 0] = points[:, 0]
    out[:, 1] = -points[:, 2]
    out[:, 2] = points[:, 1]
    return out


def _smooth_normals(positions: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """没有法线的模型按面积加权的面法线凑一份，总比全黑强。"""
    normals = np.zeros_like(positions)
    if triangles.size == 0:
        normals[:, 2] = 1.0
        return normals
    corner = positions[triangles]
    face = np.cross(corner[:, 1] - corner[:, 0], corner[:, 2] - corner[:, 0])
    for corner_index in range(3):
        np.add.at(normals, triangles[:, corner_index], face)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    fallback = lengths[:, 0] < 1e-12
    normals = np.divide(normals, np.where(lengths < 1e-12, 1.0, lengths))
    normals[fallback] = (0.0, 0.0, 1.0)
    return normals


class _Camera:
    """轨道相机：绕着目标点转，正交与透视共用同一套（yaw, pitch, distance）。"""

    def __init__(self) -> None:
        self.yaw = 32.0
        self.pitch = 18.0
        self.distance = 5.0
        self.ortho = False
        self.ortho_scale = 3.0
        self.target = np.zeros(3)
        self.radius = 1.0

    def reset_for(self, model: GlbModel | None) -> None:
        if model is None or model.is_empty:
            self.target = np.zeros(3)
            self.radius = 1.0
        else:
            self.target = (model.low + model.high) * 0.5
            self.radius = float(np.linalg.norm(model.size) * 0.5) or 1.0
        self.yaw = 32.0
        self.pitch = 18.0
        self.frame()

    def frame(self) -> None:
        height = max(1e-6, self.radius * 2.2)
        self.ortho_scale = height
        self.distance = height / (2.0 * math.tan(math.radians(CAMERA_FOV) * 0.5)) * 1.15

    def eye(self) -> np.ndarray:
        yaw = math.radians(self.yaw)
        pitch = math.radians(self.pitch)
        direction = np.array(
            [
                math.cos(pitch) * math.sin(yaw),
                -math.cos(pitch) * math.cos(yaw),
                math.sin(pitch),
            ]
        )
        return self.target + direction * max(1e-4, self.distance)

    def up(self) -> np.ndarray:
        return np.array([0.0, 0.0, 1.0]) if abs(self.pitch) < 89.0 else np.array([0.0, 1.0, 0.0])

    def orbit(self, dx: float, dy: float) -> None:
        self.yaw = (self.yaw + dx) % 360.0
        self.pitch = max(-90.0, min(90.0, self.pitch + dy))

    def pan(self, dx: float, dy: float) -> None:
        right = np.cross(self.up(), np.array([0.0, 0.0, 1.0]))
        right = np.array([math.cos(math.radians(self.yaw)), math.sin(math.radians(self.yaw)), 0.0])
        up = self.up()
        scale = (self.ortho_scale if self.ortho else self.distance) / 900.0
        self.target = self.target - right * (dx * scale) + up * (dy * scale)

    def zoom(self, steps: float) -> None:
        factor = 0.88 ** steps
        self.distance = max(1e-3, min(1e6, self.distance * factor))
        self.ortho_scale = max(1e-3, min(1e6, self.ortho_scale * factor))


class _FallbackView(QWidget):
    """没有 OpenGL 时顶上的位置：说清楚怎么回事，不假装能画。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._text = "3D 视图需要 OpenGL；这台机器上没能创建。\n配色结果照常可用。"
        self.setMinimumSize(280, 220)

    def setMessage(self, text: str) -> None:
        self._text = text
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#1E1F22"))
        painter.setPen(QColor("#9AA0A6"))
        font = QFont(painter.font())
        font.setPointSize(10)
        painter.setFont(font)
        painter.drawText(self.rect().adjusted(24, 24, -24, -24), Qt.AlignmentFlag.AlignCenter, self._text)
        painter.end()


class _GLView(QOpenGLWidget):
    """真正画三角形的那一层。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.camera = _Camera()
        self._program: QOpenGLShaderProgram | None = None
        self._vertex_buffer: QOpenGLBuffer | None = None
        self._colour_buffer: QOpenGLBuffer | None = None
        self._vao: QOpenGLVertexArrayObject | None = None
        self._model: GlbModel | None = None
        self._vertices = np.zeros((0, 6), dtype=np.float32)
        self._colours = np.zeros((0, 3), dtype=np.float32)
        self._vertex_count = 0
        self._failed = ""
        self._last: QPointF | None = None
        self._mode = "orbit"
        self.setMinimumSize(280, 220)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(False)
        self._apply_format()

    # -- 数据 -------------------------------------------------------------
    def _apply_format(self) -> None:
        from PySide6.QtGui import QSurfaceFormat

        surface = QSurfaceFormat()
        surface.setVersion(3, 3)
        surface.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        surface.setDepthBufferSize(24)
        surface.setSamples(4)
        self.setFormat(surface)

    def setModel(self, model: GlbModel | None, colours: list[tuple[int, int, int]] | None = None) -> None:
        """顶点按三角形展开（不建索引缓冲）：``glDrawElements`` 在 PySide6 里要一个
        指针参数，传 0 与 None 都会被拒绝；展开成 draw-arrays 少一层坑，96 个顶点的
        小模型与 30 万个三角形的大模型都一样能画。"""
        self._model = model
        blocks: list[np.ndarray] = []
        colour_rows: list[np.ndarray] = []
        if model is not None:
            for order, primitive in enumerate(model.primitives):
                if primitive.positions.size == 0 or primitive.triangles.size == 0:
                    continue
                positions = _to_display(primitive.positions.astype(np.float64))
                normals = (
                    _smooth_normals(positions, primitive.triangles)
                    if primitive.normals.size == 0 or not np.any(primitive.normals)
                    else _to_display(primitive.normals.astype(np.float64))
                )
                corners = primitive.triangles.astype(np.int64).reshape(-1)
                blocks.append(np.concatenate([positions[corners], normals[corners]], axis=1))
                rgb = (
                    colours[order]
                    if colours is not None and order < len(colours)
                    else primitive.colour
                )
                row = np.array(rgb, dtype=np.float64) / 255.0
                colour_rows.append(np.tile(row, (corners.size, 1)))
        self._vertices = (
            np.concatenate(blocks, axis=0).astype(np.float32) if blocks else np.zeros((0, 6), dtype=np.float32)
        )
        self._colours = (
            np.concatenate(colour_rows, axis=0).astype(np.float32)
            if colour_rows
            else np.zeros((0, 3), dtype=np.float32)
        )
        self._vertex_count = int(self._vertices.shape[0])
        self.camera.reset_for(model)
        self._upload()
        self.update()

    def setColours(self, colours: list[tuple[int, int, int]], model: GlbModel | None = None) -> None:
        if model is not None:
            self._model = model
        if self._model is None or not self._colours.size:
            return
        rows: list[np.ndarray] = []
        for order, primitive in enumerate(self._model.primitives):
            if primitive.positions.size == 0 or primitive.triangles.size == 0:
                continue
            rgb = colours[order] if order < len(colours) else primitive.colour
            row = np.array(rgb, dtype=np.float64) / 255.0
            rows.append(np.tile(row, (int(primitive.triangles.size), 1)))
        if not rows:
            return
        self._colours = np.concatenate(rows, axis=0).astype(np.float32)
        if self._colour_buffer is not None and self._colour_buffer.isCreated():
            data = self._colours.tobytes()
            self._colour_buffer.bind()
            self._colour_buffer.write(0, data, len(data))
            self._colour_buffer.release()
        self.update()

    # -- OpenGL -----------------------------------------------------------
    def initializeGL(self) -> None:  # noqa: N802 - Qt naming
        program = QOpenGLShaderProgram(self)
        if not program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, _VERTEX_SHADER):
            self._failed = f"顶点着色器编译失败：{program.log()}"
            return
        if not program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, _FRAGMENT_SHADER):
            self._failed = f"片元着色器编译失败：{program.log()}"
            return
        if not program.link():
            self._failed = f"着色器链接失败：{program.log()}"
            return
        self._program = program
        # 3.3 core 档要求画之前有一个绑着的 VAO，否则 glDrawArrays 是无效操作（不报错，
        # 只画不出东西）——这就是先前只有清屏色的原因。
        self._vao = QOpenGLVertexArrayObject(self)
        self._vao.create()
        self._vertex_buffer = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self._vertex_buffer.create()
        self._colour_buffer = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self._colour_buffer.create()
        self._upload()

    def _upload(self) -> None:
        if self._vertex_buffer is None or not self._vertex_buffer.isCreated():
            return
        data = self._vertices.tobytes() if self._vertices.size else b""
        self._vertex_buffer.bind()
        self._vertex_buffer.allocate(data, len(data))
        self._vertex_buffer.release()
        if self._colour_buffer is not None and self._colour_buffer.isCreated():
            colours = self._colours.tobytes() if self._colours.size else b""
            self._colour_buffer.bind()
            self._colour_buffer.allocate(colours, len(colours))
            self._colour_buffer.release()

    def _matrix(self) -> QMatrix4x4:
        camera = self.camera
        eye = QVector3D(*(float(v) for v in camera.eye()))
        target = QVector3D(*(float(v) for v in camera.target))
        up = QVector3D(*(float(v) for v in camera.up()))
        projection = QMatrix4x4()
        aspect = max(0.05, self.width() / max(1.0, float(self.height())))
        near = max(1e-3, camera.distance / 500.0)
        far = max(10.0, camera.distance * 200.0)
        if camera.ortho:
            half = max(1e-4, camera.ortho_scale * 0.5)
            if aspect >= 1.0:
                projection.ortho(-half * aspect, half * aspect, -half, half, -far, far)
            else:
                projection.ortho(-half, half, -half / aspect, half / aspect, -far, far)
        else:
            projection.perspective(CAMERA_FOV, aspect, near, far)
        view = QMatrix4x4()
        view.lookAt(eye, target, up)
        return projection * view

    def paintGL(self) -> None:  # noqa: N802 - Qt naming
        functions = self.context().functions()
        functions.glClearColor(0.118, 0.122, 0.133, 1.0)
        functions.glClear(0x4000 | 0x0100)  # GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT
        if self._program is None or not self._vertex_count or self._vertex_buffer is None:
            return
        if self._failed:
            return
        try:
            self._draw(functions)
        except Exception as error:  # noqa: BLE001 - 画不出来就别每帧抛一次
            self._failed = f"画不出三维视图：{error}"

    def _draw(self, functions) -> None:
        functions.glEnable(0x0B71)  # GL_DEPTH_TEST
        functions.glDisable(0x0B44)  # GL_CULL_FACE：模型可能有单面片
        self._program.bind()
        matrix = self._matrix()
        self._program.setUniformValue("u_mvp", matrix)
        normal = QMatrix4x4(matrix)
        self._program.setUniformValue("u_normal", normal.normalMatrix())
        vao = self._vao
        if vao is not None and vao.isCreated():
            vao.bind()
        self._vertex_buffer.bind()
        self._program.enableAttributeArray(0)
        self._program.setAttributeBuffer(0, 0x1406, 0, 3, 24)  # GL_FLOAT
        self._program.enableAttributeArray(1)
        self._program.setAttributeBuffer(1, 0x1406, 12, 3, 24)
        self._colour_buffer.bind()
        self._program.enableAttributeArray(2)
        self._program.setAttributeBuffer(2, 0x1406, 0, 3, 12)
        functions.glDrawArrays(0x0004, 0, int(self._vertex_count))  # GL_TRIANGLES
        self._colour_buffer.release()
        self._vertex_buffer.release()
        if vao is not None and vao.isCreated():
            vao.release()
        self._program.release()

    # -- 交互 -------------------------------------------------------------
    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        button = event.button()
        modifiers = event.modifiers()
        if button == Qt.MouseButton.MiddleButton or button == Qt.MouseButton.LeftButton:
            if button == Qt.MouseButton.MiddleButton and modifiers & Qt.KeyboardModifier.ShiftModifier:
                self._mode = "pan"
            elif button == Qt.MouseButton.MiddleButton and modifiers & Qt.KeyboardModifier.ControlModifier:
                self._mode = "zoom"
            else:
                self._mode = "orbit"
            self._last = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._last is None:
            return
        delta = event.position() - self._last
        self._last = event.position()
        if self._mode == "pan":
            self.camera.pan(delta.x(), delta.y())
        elif self._mode == "zoom":
            self.camera.zoom(-delta.y() / 40.0)
        else:
            self.camera.orbit(delta.x() * 0.4, delta.y() * 0.4)
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton):
            self._last = None
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt naming
        steps = event.angleDelta().y() / 120.0
        if steps:
            self.camera.zoom(steps)
            self.update()
        event.accept()

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        key = event.key()
        control = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        mapping = {
            Qt.Key.Key_1: ("front", "back"),
            Qt.Key.Key_3: ("right", "left"),
            Qt.Key.Key_7: ("top", "bottom"),
        }
        if key in mapping:
            near, far = mapping[key]
            self.setView(far if control else near)
            event.accept()
            return
        if key == Qt.Key.Key_5:
            self.setOrtho(not self.camera.ortho)
            event.accept()
            return
        if key == Qt.Key.Key_9:
            self.flipView()
            event.accept()
            return
        if key in (Qt.Key.Key_4, Qt.Key.Key_6, Qt.Key.Key_8, Qt.Key.Key_2):
            step = {"4": (-VIEW_STEP, 0.0), "6": (VIEW_STEP, 0.0), "8": (0.0, VIEW_STEP), "2": (0.0, -VIEW_STEP)}[chr(key)]
            self.camera.orbit(*step)
            self.update()
            event.accept()
            return
        if key == Qt.Key.Key_Home:
            self.camera.frame()
            self.update()
            event.accept()
            return
        super().keyPressEvent(event)

    def setView(self, name: str) -> None:
        angles = VIEW_ANGLES.get(name)
        if angles is None:
            return
        self.camera.yaw, self.camera.pitch = angles
        self.camera.frame()
        self.update()

    def flipView(self) -> None:
        for name, (yaw, pitch) in VIEW_ANGLES.items():
            if (
                abs((self.camera.yaw % 360.0) - (yaw % 360.0)) < 1.0
                and abs(self.camera.pitch - pitch) < 1.0
            ):
                opposite = {
                    "front": "back",
                    "back": "front",
                    "left": "right",
                    "right": "left",
                    "top": "bottom",
                    "bottom": "top",
                }[name]
                self.setView(opposite)
                return
        self.camera.orbit(180.0, 0.0)
        self.update()

    def setOrtho(self, ortho: bool) -> None:
        self.camera.ortho = bool(ortho)
        self.update()

    def frameAll(self) -> None:
        self.camera.frame()
        self.update()


class ModelView(QWidget):
    """一个面板：右上角预设视角按钮 + 三维视图。"""

    def __init__(self, title: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model: GlbModel | None = None
        self._colours: list[tuple[int, int, int]] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.setSpacing(4)
        if title:
            from PySide6.QtWidgets import QLabel

            label = QLabel(title)
            label.setProperty("role", "hint")
            bar.addWidget(label)
        bar.addStretch(1)
        self._buttons: dict[str, QPushButton] = {}
        for name, text in VIEW_LABELS:
            button = self._make_button(text, f"Blender 的 {text}视图")
            button.clicked.connect(lambda _=False, key=name: self.setView(key))
            self._buttons[name] = button
            bar.addWidget(button)
        self._ortho_button = self._make_button("透视", "在透视投影与正交投影之间切换（小键盘 5）")
        self._ortho_button.clicked.connect(lambda: self.setOrtho(not self.isOrtho()))
        bar.addWidget(self._ortho_button)
        self._frame_button = self._make_button("自适应", "把整只模型放到画面里（Home）")
        self._frame_button.clicked.connect(lambda: self.frameAll())
        bar.addWidget(self._frame_button)
        layout.addLayout(bar)

        self._fallback: _FallbackView | None = None
        self._gl: _GLView | None = None
        if _open_gl_available():
            try:
                self._gl = _GLView(self)
                layout.addWidget(self._gl, 1)
            except Exception:  # noqa: BLE001 - 老机器上建不出来就退回占位
                self._gl = None
        if self._gl is None:
            self._fallback = _FallbackView(self)
            layout.addWidget(self._fallback, 1)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    @staticmethod
    def _make_button(text: str, tip: str) -> QPushButton:
        button = QPushButton(text)
        button.setToolTip(tip)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.setFixedHeight(24)
        button.setMinimumWidth(34)
        return button

    # -- 内容 -------------------------------------------------------------
    def setModel(self, model: GlbModel | None, colours: list[tuple[int, int, int]] | None = None) -> None:
        self._model = model
        self._colours = list(colours) if colours else []
        if self._gl is not None:
            self._gl.setModel(model, self._colours or None)
        elif self._fallback is not None:
            if model is None or model.is_empty:
                self._fallback.setMessage("还没有打开模型。\n「打开 GLB」之后这里显示原模型。")
            else:
                self._fallback.setMessage(
                    f"{model.name}：{model.triangle_count:,} 个三角形。\n3D 视图需要 OpenGL，这里只显示统计。"
                )

    def setColours(self, colours: list[tuple[int, int, int]]) -> None:
        self._colours = list(colours)
        if self._gl is not None:
            self._gl.setColours(self._colours, self._model)

    def setView(self, name: str) -> None:
        if self._gl is not None:
            self._gl.setView(name)
            self._gl.setFocus(Qt.FocusReason.OtherFocusReason)

    def setOrtho(self, ortho: bool) -> None:
        if self._gl is not None:
            self._gl.setOrtho(ortho)
        self._ortho_button.setText("正交" if ortho else "透视")

    def isOrtho(self) -> bool:
        return bool(self._gl is not None and self._gl.camera.ortho)

    def frameAll(self) -> None:
        if self._gl is not None:
            self._gl.frameAll()

    def cameraState(self) -> dict[str, float | bool]:
        if self._gl is None:
            return {"yaw": 0.0, "pitch": 0.0, "distance": 0.0, "ortho": False}
        camera = self._gl.camera
        return {
            "yaw": float(camera.yaw),
            "pitch": float(camera.pitch),
            "distance": float(camera.distance),
            "ortho": bool(camera.ortho),
        }

    def hasOpenGL(self) -> bool:
        return self._gl is not None

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        return QSize(420, 340)

    def available(self) -> str:
        """为什么没有 3D 视图（有的话返回空串），给页面写提示用。"""
        if self._gl is not None:
            return ""
        return "这台机器上没能创建 OpenGL 视图"


def _open_gl_available() -> bool:
    """离屏平台（闸门脚本、无显示的服务器）不试 OpenGL。"""
    from PySide6.QtGui import QGuiApplication

    application = QGuiApplication.instance()
    if application is None:
        return False
    name = QGuiApplication.platformName().lower()
    return not (name.startswith("offscreen") or name.startswith("minimal"))

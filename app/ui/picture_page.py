"""The picture page: import an image, match it, export a flat plate.

Layout follows the user's description exactly — the picture on the left, the
list of colours the print needs on the right, and the two are linked both ways:

* clicking a colour in the list highlights that colour's pixels in the picture;
* clicking a pixel in the picture jumps to its colour in the list, and that
  colour is pinned to the top **without disturbing the order of the others**.

The page never talks to the network and never writes anywhere the user did not
choose; all Qt file dialogs default to the app's own exports directory.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
import tempfile
import time
import traceback
import zlib

import numpy as np
from PIL import Image
from PySide6.QtCore import (
    QByteArray,
    QEvent,
    QEventLoop,
    QPoint,
    QPointF,
    QRectF,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import QColor, QImage, QKeySequence, QPainter, QPixmap, QShortcut

try:  # QtSvg 只有导入矢量图时用得到；缺了也不该让整个程序起不来。
    from PySide6.QtSvg import QSvgRenderer
except ImportError:  # pragma: no cover - 打包漏收 QtSvg 时才会走到这里
    QSvgRenderer = None  # type: ignore[assignment]
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .. import __version__
from ..core import paths
from ..core.atomicio import atomic_path
from ..core.blocks import (
    BlockMap,
    label_blocks,
    min_block_pixels,
    revalue_blocks,
    without_blocks,
)
from ..core.image_matching import (
    BudgetOption,
    MatchResult,
    MatchSettings,
    PaletteEntry,
    TriplePalette,
    build_palette,
    match_image,
    match_options,
    plan_caption,
    rematch_options,
    spool_usage,
)
from ..core.image_matching import HARD_COLOUR_LIMIT as NO_COLOUR_LIMIT  # noqa: F401
from ..core.library import FilamentLibrary
from ..core.mixes import DEFAULT_ENGINE, MixCatalog
from ..core.vectorize import VectorizeUnavailable, image_to_svg
from .svg_export import SvgExportDialog
from ..core.settings import (
    DEFAULT_PICK_MODE,
    PICK_COLOUR,
    PICK_EDGE,
    PICK_MODES,
    load_settings,
    resolve_pick_mode,
    save_settings,
)
from ..core.project import (
    PROJECT_FILTER,
    PROJECT_SUFFIX,
    ProjectError,
    ProjectState,
    load_project,
    render_preview,
    save_project,
)
from ..mesh.objfile import write_obj
from ..mesh.plate import CELL_MM, COLOUR_BAND_MM, PlateModel, PlateSettings, build_plate
from ..mesh.threemf import DEFAULT_PLATE_MM as PLATE_MM
from ..mesh.threemf import MAX_PAINTED_FILAMENTS, slots_from_palette, write_3mf
from . import theme
from .colour_picker import (
    ColourDetail,
    ColourPickerDialog,
    compare_pixmap,
    entry_recipe_text,
    hex_of,
)
from .plan_filter import SpoolFilterDialog
from .selectable import selectable_text
from .swatch import swatch_pixmap

__all__ = ["PicturePage", "PictureView", "ColourList"]

IMAGE_FILTER = (
    "图片 (*.png *.jpg *.jpeg *.bmp *.webp *.gif *.tif *.tiff *.svg);;"
    "矢量图 (*.svg);;所有文件 (*)"
)
#: 「导出原图」只写 PNG：原图里可能带着「删掉的地方是透明的」这种 alpha，换成 JPEG 就丢了
#: （m12426 item 3）。
PNG_FILTER = "PNG 图片 (*.png);;所有文件 (*)"

#: 矢量图（SVG）也能当原图 (m13907 item 2)。位图导入是「有什么像素就用什么像素」，
#: 而 SVG 是无限清晰的：渲染得比它自己声明的大，细笔画（猪尾巴那种一两像素的卷）才有
#: 像素可认。1.6.44 起渲染不再设上限 —— 拿来一个图标渲染成多大，由它自己的尺寸和栅格决定（m17910 第 2 条）。
SVG_SUFFIXES = frozenset({".svg"})

#: 导入 SVG 时那个弹窗里能写的像素数（1.6.49）。下界 16 已经和图标差不多大。
#: 1.6.62（m28297 第 1 条）：上界撤了 —— 用户要 8000 px 就给它 8000，不再悄悄夹到
#: 4096。夹着的坏处是明显的：填 6000 被改成 4096，导进来的图和想要的不一样，而且
#: 界面上没有任何提示。
VECTOR_MIN_PX = 16
#: Qt 的 QSpinBox 上限。写这么大等于「不设上限」，但控件本身还得有个数才合法。
VECTOR_MAX_PX = 100_000

#: Wheel zoom runs between 「整张铺满面板」 (1.0) and this many times the fit.
ZOOM_MIN = 1.0
ZOOM_MAX = 12.0
#: One wheel notch multiplies the zoom by this much.
ZOOM_STEP = 1.25
#: The two picture panels are panned and zoomed the same way.
ZOOM_HINT = "滚轮缩放，按住鼠标中键拖动平移，双击还原成整张。"

#: 图片匹配默认值。颜色数量直接顶到 Bambu Studio 的天花板：混色就是为了用少数
#: 几卷料凑出一整张图，用户既然在用混色，就没有理由再自己往回缩。
DEFAULT_MAX_COLOURS = MAX_PAINTED_FILAMENTS
#: 「实验性：没有颜色上限」时交给量化那一步的数（1.6.61，m27752 第 1 条）。它只是
#: 一个「比任何图都多」的哨兵：真正收手的是图自己的颜色数 —— 聚类簇数不可能超过
#: 「这张图里有几种不同的颜色」，所以这里写多大都不会凭空造出颜色。
UNLIMITED_COLOURS = 4096
#: 引擎报上来的分数是「第几步」，不是「过了多少时间」：``0.05/0.15/0.30`` 这三步
#: 在几千像素的图上眨眼就过，可是在 1680×2880 的图上，从 0.15 走到 0.30 之间的那次
#: 量化要一秒多 —— 进度条就定在那儿不动，用户看到的是「卡住了」（1.6.62，m28297 第 3 条）。
#: 这里按实测耗时重新排一遍：量化那一段（0.15→0.30）拿到 6%→45%，找色（0.35→0.60）
#: 拿 45%→72%，剩下的才分给收尾。表是 (引擎分数, 进度条分数)，中间线性插值。
_MATCH_PROGRESS_MAP = (
    (0.00, 0.00),
    (0.15, 0.06),
    (0.30, 0.45),
    (0.35, 0.48),
    (0.60, 0.72),
    (0.75, 0.88),
    (0.85, 0.94),
    (0.90, 0.96),
    (0.96, 0.99),
    (1.00, 1.00),
)


def _match_progress_fraction(engine_fraction: float) -> float:
    """引擎分数 → 进度条分数（见 ``_MATCH_PROGRESS_MAP``）。"""
    value = max(0.0, min(1.0, float(engine_fraction)))
    for (in_from, out_from), (in_to, out_to) in zip(
        _MATCH_PROGRESS_MAP, _MATCH_PROGRESS_MAP[1:]
    ):
        if value <= in_to:
            span = in_to - in_from
            if span <= 0:
                return out_to
            return out_from + (out_to - out_from) * (value - in_from) / span
    return 1.0


#: 一次打印最常用的卷数（AMS 四槽）。0 才是「不限」，默认值必须是真数字。
DEFAULT_SPOOL_CAP = 4

#: 「重新匹配」 on a real library takes 1.3–3.0 s (measured: 41 卷 + 三色池 =
#: 3.03 s，其中 1.38 s 是 CIEDE2000 搜索、1.47 s 是「耗材上限」合并）。进度条只在
#: 慢到值得看的时候才出现：小图匹配几十毫秒就完了，点一下闪一下比不显示更烦人。
BUSY_DELAY_MS = 200
#: The bar's internal resolution.  Percentages land on it without rounding to
#: 「0% 卡住」, and a plain 0–100 range is too coarse while the search walks.
PROGRESS_STEPS = 1000

#: 图片页面原来那行常驻说明。1.6.14 把它整段搬进「帮助」窗口（m01885 第 6 条）：
#: 页面上只留随图片变化的东西，要读的使用说明去帮助里读，不再压着图片。
#: 「点图片时」两种方式的说法（取值见 ``app.core.settings``）。
PICK_MODE_LABELS = {PICK_COLOUR: "按颜色选择", PICK_EDGE: "按色块边缘选择"}

#: 耗材方案面板空着时的两句话（m00940 第 1 条）。它们是一对——上一句说方案表会有
#: 什么，下一句说它下面的用量行会说什么——所以两句都在建面板时就写上，刚打开软件就
#: 一起看得见；以前第二句要等第一次匹配走完 `_update_usage` 才出现，用户只会看到
#: 上半句悬在面板中间。
PLAN_HINT_TEXT = "匹配一张图片后，这里会列出几套耗材搭配，自己挑一套。"
USAGE_HINT_TEXT = "匹配完成后，这里列出这一版用到了哪几卷耗材，以及各卷占的色料比例。"

#: 颜色列表面板的标题（m02316 item 5）。这一栏在两种「点图片时」下装的不是同一种东西：
#: 「按颜色选择」里一行是一种颜色（图里要印的色），「按色块边缘选择」里一行是一块（第 N
#: 块）。所以标题也跟着换，不然块模式下写着「打印图片需要的颜色」、列出来却是「第 1 块…」。
LIST_TITLE_COLOUR = "打印图片需要的颜色"
LIST_TITLE_BLOCK = "图片上色块的颜色"

#: 导出时圆滑色块的开关（m01647）。默认勾上，就在两个「生成」按钮左边。
SMOOTH_LABEL = "圆滑色块"
SMOOTH_TIP = (
    "勾上之后，预览和导出的 3MF/OBJ 都在细一档的栅格上重新分类色块，斜着走的边界不再是"
    "一格一格的台阶（默认细到图片像素的四分之一：150 mm 宽、512 像素的图约 0.07 mm 一格，"
    "颜色多时退到二分之一，都比喷嘴细），看起来、打出来都是圆的。\n"
    "不勾就按匹配栅格那一格一格的方块画，也按 1.6.21 的样子导出——用色块工具点格子时清楚一些。"
)

#: 图片印在哪一面（m02289 item 2）。默认底面：图片贴着打印板，顶面是一整片底板色。
IMAGE_SIDE_LABEL = "图片位置"
IMAGE_SIDE_BOTTOM = "底面"
IMAGE_SIDE_TOP = "顶面"
IMAGE_SIDE_TIP = (
    "图片印在模型的哪一面。\n"
    f"· {IMAGE_SIDE_BOTTOM}（默认）：图片朝下贴着打印板，顶面是一整片底板色，打出来顶面平整、"
    "不被色块的边线划过。看底面得把整块翻过来看，所以导出时图片会**左右翻一次**，翻过来看"
    "字和图才是正的（1.6.45 起）。\n"
    f"· {IMAGE_SIDE_TOP}：图片朝上，就是 1.6.22 及以前的样子。\n"
    "两面都是同一块闭合实体，只是上下对调；尺寸、厚度、体积都不变。"
)

#: 底板颜色（m16941 第 2 条）。从前只能从「打印图片需要的颜色」那几行里挑，现在跟
#: 「更换颜色」用同一个挑选框，整套混色都能搜。挑的那一种不在图上时，它会自己占一个
#: 耗材槽写进 3MF（底板的混色配方才带得过去）。
BASE_COLOUR_TIP = (
    "底板用哪一种颜色。\n"
    "· 点「更换底板颜色」从整套混色里挑（跟「更换颜色」是同一个框，可以按 ΔE00 排序、也能搜）；\n"
    "· 挑的颜色不在「打印图片需要的颜色」里也行 —— 导出时它自己占一个耗材槽；\n"
    "· 没挑就跟着图上占像素最多的那一块走。"
)


PAGE_HELP = (
    "图像转换这一步：\n"
    f"· {ZOOM_HINT}\n"    f"· 颜色数量最多 {MAX_PAINTED_FILAMENTS} 是 Bambu Studio 自己的限制：一个工程里只认 "
    f"{MAX_PAINTED_FILAMENTS} 个耗材位，第 {MAX_PAINTED_FILAMENTS + 1} 种颜色在切片时会被"
    "丢掉（我们已经就这条限制向拓竹提了工单）。耗材上限 0 表示不限制。\n"
    "· 改到一半可以存成工程接着改：Ctrl+S 保存（第一次会问存哪儿），Ctrl+Shift+S 换一个文件名；"
    "工程里装着原图、这张预览、用到的颜色和耗材，下次「打开工程」就回到现在这一步。\n"
    "· 双击颜色列表里的一行，看这个颜色的详情（图片里的颜色、匹配到的颜色和它的配方）。\n"
    "· 「筛选耗材」先说好哪几卷必须用、哪几卷不要用，之后每套搭配都照这个来。\n"
    "· 「按当前色块重新匹配」先从「原图」（含你做过的删除、合并）重新找一遍颜色、按「颜色数量」"
    "重算，再拿现在这些色块重新挑卷：色块的划分和编号都不动，你在色块模式里点过、改过的块还在原"
    "处；颜色表里那些「更换颜色」会清掉（行是按旧的颜色表排的，对不上了）。原图本身不改。\n"
    "· 一份匹配拿图片自己的像素数分色（1.6.43 起）：一格 0.4 mm，板的长宽就是像素数 × 0.4"
    "（1.6.48 起不再有「宽 × 长」两个框 —— 导入多少像素就按多少像素分）。\n"
    "· 「删除」也可以按键盘上的 Del：选中一种颜色或者一块之后按 Del，和点按钮一样。\n"
    "· 在「按色块边缘选择」下删除或合并，会同时改「原图」本身——删掉的那块在原图里变透明、"
    "并掉的那几块在原图里并进留下来的那一块，所以「按当前色块重新匹配」从改过的原图出发，删掉"
    "的地方不会又被匹配回来。原图被改过之后，它上方的标题旁边会出现「原图已修"
    "改」。按颜色删除或合并也一样改原图（1.6.20 起）：删掉一种颜色 = 这些像素在源图里变透明，"
    "合并几种颜色 = 这些像素在源图里并进留下来的那一种——改的是同一张图的同一批像素，就该是同"
    "一件事。只有「更换颜色」不动原图，它说的是「这块地方这一次用哪一卷来印」，换一套搭配还"
    "能再改。Ctrl+Z 会把原图一起退回去。\n"
    "· 合并之后原图的颜色不会变多（1.6.21 起）：并掉的那些像素在原图里用的是**留下来的那一块"
    "本来就在的颜色**，不是预览里那卷耗材印出来的颜色。所以原图始终只由它自己的颜色组成，变的"
    "只是色块大小；再点「按当前色块重新匹配」，就是拿原图里剩下的那几种颜色重新挑卷。\n"
    "· 「耗材方案」里一行是一套卷组（哪几卷料），同一批卷的另一种配方算同一行。颜色越多，"
    "能凑出不同卷组的余地越大；一张图只剩两三种颜色时，往往只有一套卷组能凑出来，那几行就会"
    "变少——想看别的搭配，得先让图里有更多颜色（颜色数量调大，再按一次「按当前色块重新匹配」）。\n"
    "· 匹配时状态栏右边有一条进度条，跟着「正在匹配：…」一路走到 100%。它要等匹配超过"
    f" {BUSY_DELAY_MS} 毫秒才出现，几毫秒就完成的匹配不会闪一下。\n"
    f"· 点图片时默认「{PICK_MODE_LABELS[PICK_COLOUR]}」：点在哪一块就选哪一块。描边这类细线"
    f"一条只有一两个像素宽，不好点，换成「{PICK_MODE_LABELS[PICK_EDGE]}」后会选离你点的位置"
    "最近的那一块，点歪一点也能选中。\n"
    f"· 选了「{PICK_MODE_LABELS[PICK_EDGE]}」，右边的清单就变成一行一块（第 N 块、它印出来的"
    "颜色、配方和占比），标题也跟着从「打印图片需要的颜色」改成「图片上色块的颜色」：点图选中"
    "的是整块，下面那排按钮也跟着按块来——「删除」删掉选中的那几块（留空、不打印），「合并」"
    "在你选的这些块里挑一种颜色、把选中的块全变成它，「更换颜色」把选中的块一起换掉。切回"
    "「按颜色选择」清单和按钮立刻回到一种颜色一行；换选择方式本身也进撤销历史（Ctrl+Z 能退回"
    "去）。\n"
    "· 单选时再点一下已经选中的那一行／那一块，就取消选中（预览里的高亮一起收掉），不用先点到"
    "别处去。列表里只要一个都没选中（点列表下面的空白处也算），预览里就不会留着任何高亮和描边"
    "（1.6.21 起）。\n"
    "· 「删除」要选一行、「合并」要选两行（1.6.34 起把这句话也挂在按钮的提示里）：没点「多选」"
    "的时候清单是单选的，按住 Ctrl 也选不中第二行——先点「多选」，再点第二行，「合并」就会亮。\n"
    "· 「打开工程」之后，「耗材方案」除了文件里存着的那一套，还会把同一批色块的其它卷组搭配"
    "列在下面（1.6.20 起）：文件里只记得住挑中的那一套，别的搭法本来要按「按当前色块重新匹配」"
    "才看得到。当前这一套仍然是文件里那一套，勾还在它上面。\n"
    "· 「圆滑色块」默认勾上（1.6.22 起）：模型里的色块边界本来落在匹配像素的格线上，斜着走的"
    "地方就是一格一格的台阶（150 mm 宽、512 像素的图约 0.29 mm 一级），在 Bambu Studio 里放大看"
    "是锯齿。勾上之后导出时会在细一档的栅格上重新分类，边界落在格子之间（默认细到图片像素的"
    "四分之一，约 0.07 mm，颜色多时退到二分之一，都比喷嘴细），打出来"
    "是圆的；同一个开关也写进 .bpproj，所以存过的工程下次打开还是上次那个样子。1.6.26 起这个勾"
    "也管预览：勾着的时候「预览」和存进工程的「预览.png」画的就是导出用的细格，边界是连续的；"
    "不勾才是一块块的原始格子（用色块工具点格子时清楚一些），导出也照 1.6.21 的样子。\n"
    "· 预览里的色块形状跟原图对不上，是上面两条叠出来的，不是画错了（m14360）：识别先把原图"
    "压到「颜色数量」那么多种颜色，每个像素归到最近的那一种——细到一两个像素的线（尾巴、描边）"
    "本来就会被两边抢走或者反过来抢两边，块的胖瘦就变了；「圆滑色块」勾着时预览画的又是细四倍"
    "的格子，边界从台阶变成圆滑曲线，小块面积还会再动几个百分点（实测大块 ±0.5% 以内、最小的"
    "几块 −5%～−10%）。想看原始格子就取消这个勾；想让细线更准，把「颜色数量」调大，或者导入 "
    "SVG——矢量放大不会先糊一层。\n"
    "· 预览里的格子是**匹配栅格**，所以格子大小和原图不是一回事（m14567）：匹配前整张图会先缩到"
    "最长边那么多像素，原图比它大时一个色块格就盖住好几个原像素，台阶看着比原图的笔触粗、"
    "一两个原像素宽的细线也落不进格子。实测同一张图（SVG 按 1024 画）：栅格 256（一格 4.0 像素）"
    "时有 7.87% 的格子跟原图归的色对不上，512（2.0 像素）降到 3.41%，1024（1.0 像素）是 1.97%"
    "——剩下那 1.97% 才是「颜色数量」和抗锯齿坡合并那一层。"
    "1.6.42 起导入时按图片自己的像素数定栅格，所以上面的数字只剩在「打开的是老工程」时才会出现；"
    "那种情况下状态行会写着原图尺寸、匹配栅格和缩小了几倍。\n"
    "· 导入图片时按图片自己的像素数建匹配栅格（1.6.43 起不再有「最大边长」数字框，1.6.44 起连上限"
    "也没有）：一格 0.4 mm，板的长宽就是像素数 × 0.4。 图片多少像素就按多少像素分（没有上限）。"
    "图越大匹配越慢，那是像素数本身的事。\n"
    "· 导入 SVG（矢量图）时会先弹一个框问「渲成多少像素」（1.6.49 起）：默认填的正是它在文件里声明"
    "的那一档，所以直接回车就是老行为；渲得大一点，细笔画（猪尾巴那种一两像素的卷）才有像素，颜色"
    "也分得更开。 框里能写 16～4096，勾着「锁定长宽比」时改一边另一边按原比例跟着走，取消它就能拉"
    "成不等比，下面实时写着这么渲出来是多大的板。 按「取消」这张图就不导入，原来那张原图不动。"
    "位图不弹这个框 —— 放大只是插值，多不出细节。\n"
    "· 「底板颜色」点「更换底板颜色」可以从**整套混色**里挑任意一种当底板（1.6.43 起，跟「更换颜色」用同"
    "一个框，能按 ΔE00 排序、也能搜）：挑的正好是图上某一卷时，底板跟那一卷共用一个耗材槽；"
    "不在图上时它自己占一个槽写进 3MF，那一片的混色配方才带得过去；不挑就跟着占像素最多的那一块"
    "走。 这一项写进 .bpproj，Ctrl+Z 也撤得回来。\n"
    f"· 「{IMAGE_SIDE_LABEL}」选图片印在模型的哪一面（1.6.23 起）：{IMAGE_SIDE_BOTTOM}（默认）"
    f"是图片朝下贴着打印板、顶面一整片底板色，打出来顶面平整（看底面要把整块翻过来，所以导出时"
    "图片左右翻一次，翻过来看是正的，1.6.45 起）；"
    f"{IMAGE_SIDE_TOP} 是图片朝上，"
    "也就是 1.6.22 及以前的样子。两面都是同一块闭合实体上下对调，尺寸、厚度、体积都不变；"
    "这一项也写进 .bpproj。"
)


#: 「按色块边缘选择」在光标周围找多远的像素。10 像素够得着描边，又不会在空白处
#: 一点就选中半张图外的一块。
EDGE_PICK_RADIUS = 10

#: 光标附近这么大一圈里，比脚下那一块更小的块先被选中（m13246 item 1）。深棕色笔画
#: 里嵌着的小斑点画出来的样子比真实像素大，只按「最近边界」判的话永远轮不到它们。
EDGE_SMALL_RADIUS = 3


def _nearest_edge(result: MatchResult, x: int, y: int, *, fallback: int) -> int:
    """The block with a pixel nearest to ``(x, y)``, within :data:`EDGE_PICK_RADIUS`.

    ``result.indices[y, x]`` is the honest answer and stays the default; this is for
    the strokes it cannot reach.  Inside a block the nearest pixel is the cursor's
    own, so the answer is that block; next to a one-pixel brow the answer is the
    brow.  ``fallback`` is used when the neighbourhood holds no block at all.
    """
    left = max(0, x - EDGE_PICK_RADIUS)
    right = min(result.width, x + EDGE_PICK_RADIUS + 1)
    top = max(0, y - EDGE_PICK_RADIUS)
    bottom = min(result.height, y + EDGE_PICK_RADIUS + 1)
    window = np.asarray(result.indices[top:bottom, left:right])
    ys, xs = np.nonzero(window >= 0)
    if not len(ys):
        return fallback
    # Squared pixel distance: the nearest boundary wins, ties going to the pixel
    # first in raster order so the same click always picks the same block.
    distances = (xs + left - x) ** 2 + (ys + top - y) ** 2
    closest = int(np.argmin(distances))
    return int(window[ys[closest], xs[closest]])


def _nearest_label(labels: np.ndarray, x: int, y: int, *, fallback: int) -> int:
    """The block whose pixel is nearest to ``(x, y)``, within :data:`EDGE_PICK_RADIUS`.

    The block-numbered twin of :func:`_nearest_edge`: same window, same
    distance rule, but the answer is a block id instead of a palette index, so
    「按色块边缘选择」 can say 「这一块」 rather than 「这一种颜色」 (m02366 item 2).
    """
    height, width = labels.shape
    left = max(0, x - EDGE_PICK_RADIUS)
    right = min(width, x + EDGE_PICK_RADIUS + 1)
    top = max(0, y - EDGE_PICK_RADIUS)
    bottom = min(height, y + EDGE_PICK_RADIUS + 1)
    window = np.asarray(labels[top:bottom, left:right])
    ys, xs = np.nonzero(window >= 0)
    if not len(ys):
        return fallback
    distances = (xs + left - x) ** 2 + (ys + top - y) ** 2
    closest = int(np.argmin(distances))
    return int(window[ys[closest], xs[closest]])


def _pick_smaller_near(
    labels: np.ndarray,
    sizes: np.ndarray | None,
    x: int,
    y: int,
    own: int,
    *,
    radius: int = EDGE_SMALL_RADIUS,
) -> int:
    """A smaller block near ``(x, y)``, if there is one — otherwise ``own``.

    「按色块边缘选择」本来是「最近边界」说了算：光标落在哪一块上就选哪一块，落在
    空白处就选最近的那一块。深棕色笔画里嵌着几处小斑点时，这条规则够不着它们：
    预览里的圆滑色块把小斑点画得比真实像素大，用户点在画出来的斑点上，脚下最近
    的真实像素还是周围那根大笔画。

    所以在 ``radius`` 像素以内先找一圈：里面有比 ``own`` 更小的块，就选离得最近的
    那一个（一样近就选更小的）；没有才认 ``own``。``own`` 是空白（``-1``）时圈里
    任何一块都算更小，于是退化成原来那条「最近边界」（m13246 item 1）。
    """
    if sizes is None:
        return own
    height, width = labels.shape
    left = max(0, x - radius)
    right = min(width, x + radius + 1)
    top = max(0, y - radius)
    bottom = min(height, y + radius + 1)
    window = np.asarray(labels[top:bottom, left:right])
    ys, xs = np.nonzero(window >= 0)
    if not len(ys):
        return own
    own_size = float(sizes[own]) if 0 <= own < len(sizes) else float("inf")
    distances = (xs + left - x) ** 2 + (ys + top - y) ** 2
    best = own
    best_key: tuple[float, float] | None = None
    for spot in range(len(ys)):
        label = int(window[ys[spot], xs[spot]])
        size = float(sizes[label]) if 0 <= label < len(sizes) else float("inf")
        if size >= own_size:
            continue
        key = (float(distances[spot]), size)
        if best_key is None or key < best_key:
            best, best_key = label, key
    return best


def _live_means(
    base_means: np.ndarray, indices: np.ndarray, live: np.ndarray, count: int
) -> np.ndarray:
    """每种 live 颜色的「图里本色」：它那些像素原来所在颜色的加权平均。

    图片本身没变，变的只是「这些像素打印成什么」，所以块级编辑改过的像素仍然沿用
    它原来所在颜色的本色——详情窗口左边那个 swatch 就是图上这一块的颜色。
    """
    base_flat = indices.ravel()
    live_flat = live.ravel()
    keep = (base_flat >= 0) & (live_flat >= 0)
    if not keep.any():
        return np.zeros((count, 3), dtype=base_means.dtype)
    slots = live_flat[keep].astype(np.int64)
    targets = np.asarray(base_means, dtype=np.float64)[base_flat[keep]]
    codes = (slots[:, None] * 3 + np.arange(3, dtype=np.int64)).ravel()
    sums = np.bincount(codes, weights=targets.ravel(), minlength=count * 3)[: count * 3]
    weights = np.bincount(slots, minlength=count).astype(np.float64)
    weights[weights == 0] = 1.0
    return (sums.reshape(count, 3) / weights[:, None]).astype(base_means.dtype)


def _to_image_grid(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """最近的邻放大：把匹配网格上的图放回原图尺寸（``size`` 是 ``(宽, 高)``）。

    匹配先把原图缩到最长边 ``max_dimension`` 再分色块，所以 ``_base.indices`` 和每个
    ``BlockPaint.mask`` 都长在那张小网格上（1875×1188 的样例配 512 的网格会长成
    324×512），而原图本身一直是原始分辨率。要改原图的像素，得先把蒙版／标签放回去。
    """
    width, height = int(size[0]), int(size[1])
    source_height, source_width = array.shape
    if (source_width, source_height) == (width, height):
        return array
    rows = np.minimum(
        (np.arange(height) * source_height) // height, source_height - 1
    )
    cols = np.minimum(
        (np.arange(width) * source_width) // width, source_width - 1
    )
    return array[rows][:, cols]


def _pixel_means(pixels: np.ndarray, indices: np.ndarray, count: int) -> np.ndarray:
    """每种 live 颜色在 ``pixels`` 上的平均色，按 ``indices`` 分组，洞（``-1``）不算。

    匹配时量下的 ``MatchResult.means`` 是「那张图」的本色。1.6.19 起 删除/合并 会改
    原图本身（m01566 的答复），所以「这一块本来是什么颜色」得从当前原图重新量，
    否则「按当前色块重新匹配」会把用户刚删掉的颜色又匹配回来。
    """
    flat = np.asarray(indices).ravel()
    values = np.asarray(pixels).reshape(-1, 3).astype(np.int64)
    means = np.zeros((count, 3), dtype=np.uint8)
    if flat.shape[0] != values.shape[0]:
        return means
    keep = flat >= 0
    if not keep.any():
        return means
    slots = flat[keep].astype(np.int64)
    weights = np.bincount(slots, minlength=count)[:count].astype(np.float64)
    sums = np.stack(
        [
            np.bincount(slots, weights=values[keep, channel], minlength=count)[:count]
            for channel in range(3)
        ],
        axis=1,
    )
    live = weights > 0
    means[live] = np.rint(sums[live] / weights[live, None]).astype(np.uint8)
    return means


@dataclass(frozen=True)
class BlockPaint:
    """One 色块级 edit: exactly these pixels now print ``entry`` (``None`` = 留空).

    「按色块边缘选择」 下删除 / 合并 / 更换颜色都不再是「这一种颜色全改」，而是
    「这几块改」——所以它们不能记在 ``_deleted`` / ``_merged`` / ``_replaced`` 里
    （那三本账是整张图按颜色算的），只能按像素记。``mask`` 指向 ``_base.indices``
    的像素坐标，那个坐标系在编辑之间永远不变，撤销/恢复只要把这一步从列表里拿掉
    或放回去就回到原样。
    """

    #: 与 ``_base.indices`` 同形状的 bool 蒙版：这一步改的是哪些像素。
    mask: np.ndarray
    #: 这些像素改成哪种颜色；``None`` 表示删除（留空、变透明）。
    entry: PaletteEntry | None
    #: 改之前这些像素属于哪些色块，用来在改动之后把选中的块找回来。
    labels: tuple[int, ...] = ()


def _result_fingerprint(result: "MatchResult") -> tuple:
    """这份匹配「画出来会是什么样」的身份（m26286 第 4 条）。

    预览只取决于是哪张图、每种颜色各是什么、有没有圆滑，所以指纹只看这几样：
    宽高 + 每个像素落在哪一色（indices）+ 每一色的平均色（means）。``MatchResult``
    的对象身份不能用 —— 每次 ``_apply_edits`` 都会新建一个，哪怕内容一模一样。
    """
    indices = np.ascontiguousarray(result.indices)
    means = np.ascontiguousarray(result.means) if result.means is not None else None
    return (
        result.width,
        result.height,
        zlib.crc32(indices.tobytes()),
        None if means is None else zlib.crc32(means.tobytes()),
    )


def _to_qimage(image: Image.Image) -> QImage:
    """PIL image → QImage, copying so the buffer outlives the PIL object."""
    rgba = image.convert("RGBA")
    data = rgba.tobytes("raw", "RGBA")
    qimage = QImage(data, rgba.width, rgba.height, QImage.Format.Format_RGBA8888)
    return qimage.copy()


def _vector_declared_size(path: Path) -> tuple[int, int]:
    """这张 SVG 在文件里声明的宽 × 长（弹窗的默认值，1.6.49）。

    只做「打开、问一句」这一件事，渲染留给 ``_load_vector``；两者用的是同一个
    ``QSvgRenderer``，所以弹窗里写下的数就是真渲出来的数。
    """
    if QSvgRenderer is None:  # pragma: no cover - 只有打包漏收 QtSvg 才会走到这里
        raise ValueError("这个版本没带上 SVG 支持（缺 QtSvg），请先导出成 PNG 再导入。")
    renderer = QSvgRenderer(QByteArray(path.read_bytes()))
    if not renderer.isValid():
        raise ValueError(f"这张 SVG 读不出来：{path.name}")
    declared = renderer.defaultSize()
    return (max(1, declared.width()), max(1, declared.height()))


def _load_vector(
    path: Path, target: int = 1, *, size: tuple[int, int] | None = None
) -> Image.Image:
    """SVG → 位图（m13907 item 2）。

    用 Qt 自己的 QSvgRenderer：跟屏幕上画的是同一个渲染器，颜色和透明度不用自己解释，
    也不多一个依赖。

    ``size`` 是用户在导入弹窗里写下的「宽 × 长」像素数（1.6.49，m20069），给了就照它渲
    —— 宽和长各自独立，用户锁了长宽比的话弹窗已经替他算好。没给就按 ``target``：渲染
    边长取「SVG 自己声明的尺寸」与 ``target`` 里较大的那个 —— 矢量图放大不掉细节，渲得
    大一点，细笔画才落在像素上。1.6.44 起不再设上限 —— 渲多大由图片自己的尺寸与调用方
    要的栅格决定（m17910 第 2 条）。
    """
    if QSvgRenderer is None:  # pragma: no cover - 只有打包漏收 QtSvg 才会走到这里
        raise ValueError("这个版本没带上 SVG 支持（缺 QtSvg），请先导出成 PNG 再导入。")
    renderer = QSvgRenderer(QByteArray(path.read_bytes()))
    if not renderer.isValid():
        raise ValueError(f"这张 SVG 读不出来：{path.name}")
    declared = renderer.defaultSize()
    if size is None:
        span = max(1, declared.width(), declared.height())
        scale = max(1.0, float(target) / span)
        width = max(1, round(max(1, declared.width()) * scale))
        height = max(1, round(max(1, declared.height()) * scale))
    else:
        width = max(1, int(size[0]))
        height = max(1, int(size[1]))
    canvas = QImage(width, height, QImage.Format.Format_RGBA8888)
    canvas.fill(0)
    painter = QPainter(canvas)
    try:
        renderer.render(painter)
    finally:
        painter.end()
    return Image.frombytes("RGBA", (width, height), bytes(canvas.constBits()))


def _scrollable(widget: QWidget) -> QScrollArea:
    """Put a widget in a frameless, resizable scroll area — the house style."""
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.Shape.NoFrame)
    area.setWidget(widget)
    return area


class PictureView(QWidget):
    """Shows the imported picture and reports which colour region was clicked."""

    regionClicked = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setMouseTracking(False)
        self._result: MatchResult | None = None
        self._source: QImage | None = None
        self._matched: QImage | None = None
        # 预览是按「圆滑色块」画还是按原始格子画（m05557 item 3）。默认圆滑，
        # 和导出侧 PlateSettings.smooth_edges 的默认值一致。
        self._smooth = True
        # 两种画法各留一份，来回勾「圆滑色块」就不必重算：细格那一版要一秒多
        # （512 像素、8 色约 1.2–1.5 s），勾一次卡一次是不能接受的。
        self._matched_cache: dict[tuple, QImage] = {}
        self._highlight: QPixmap | None = None
        self._selected = -1
        self._show_matched = True
        self._plain = False
        self._scale = 1.0
        self._origin = QPointF(0.0, 0.0)
        # 缩放和平移叠在「铺满面板」之上：_fit_scale 由面板尺寸算出来，_zoom 是
        # 用户用滚轮乘上去的倍数，_pan 是像素位移。三者分开以后，改窗口大小仍然
        # 是以「铺满」为基准，不会把之前放大的倍数算丢。
        self._fit_scale = 1.0
        self._zoom = 1.0
        # m26604 第 10 条：选中的是只有几像素的块时，描边加粗、蒙版压实的开关。
        self._tiny_highlight = False
        self._pan = QPointF(0.0, 0.0)
        self._press_pos: QPointF | None = None
        self._last_pos: QPointF | None = None
        self._panning = False
        #: 「按颜色选择」 or 「按色块边缘选择」 — owned by the page, saved in settings.
        self._pick_mode = DEFAULT_PICK_MODE
        #: 「按色块边缘选择」下的色块编号图（``-1`` = 没打印），点图片时用它找块。
        self._block_labels: np.ndarray | None = None
        #: 每块多少像素，和 ``_block_labels`` 配套：小块优先得靠它比大小（m13246 item 1）。
        self._block_sizes: np.ndarray | None = None
        #: 选中的那一堆色块的并集掩码；比单个颜色更宽，因为「多选」可以选好几块。
        self._selected_mask: np.ndarray | None = None

    # -- contents ---------------------------------------------------------
    def setResult(self, result: MatchResult | None, source: QImage | None) -> None:
        previous = self._source
        self._result = result
        self._source = source
        self._matched = self._build_matched(result) if result is not None else None
        self._selected = -1
        self._selected_mask = None
        self._highlight = None
        same_picture = (
            previous is not None
            and source is not None
            and previous.width() == source.width()
            and previous.height() == source.height()
        )
        if not same_picture:
            # 换了图片（或者清空了）就回到「整张铺满」，否则用户一打开新图看到的
            # 是上一张留下来的某个角落。
            self.reset_zoom(update=False)
        self.update()

    def setSelected(self, index: int) -> None:
        if self._plain:
            # 原图 must never react to the selection: the whole point of the
            # panel is to show what the user uploaded, next to what the match
            # made of it.  Dimming it made the upload look like a washed-out
            # copy of the preview.
            return
        if index == self._selected and self._selected_mask is None:
            return
        self._selected = int(index)
        self._selected_mask = None
        self._highlight = None
        self.update()

    def setBlockLabels(
        self, labels: np.ndarray | None, sizes: np.ndarray | None = None
    ) -> None:
        """The block numbering 「按色块边缘选择」 points with (``None`` = no blocks).

        ``sizes`` is how many pixels each block owns.  The edge picker gives a
        smaller block priority inside :data:`EDGE_SMALL_RADIUS`, which is what
        keeps a speck inside a dark stroke clickable (m13246 item 1).
        """
        self._block_labels = None if labels is None else np.asarray(labels)
        self._block_sizes = None if sizes is None else np.asarray(sizes)
        if self._selected_mask is not None:
            self._highlight = None
            self.update()

    def setSelectedMask(self, mask: np.ndarray | None) -> None:
        """Highlight a whole set of blocks at once.

        One colour is not one thing on screen — the eyebrows, the outline and a
        patch of the body can all print with the same spool — so 「按色块边缘选择」
        selects several blocks, and the preview has to draw exactly those
        (m02366 item 2/3).
        """
        if self._plain:
            return
        if mask is None:
            changed = self._selected_mask is not None
            self._selected_mask = None
        else:
            wanted = np.asarray(mask, dtype=bool)
            changed = self._selected_mask is None or not np.array_equal(
                self._selected_mask, wanted
            )
            self._selected_mask = wanted
        if not changed:
            return
        self._selected = -1
        self._highlight = None
        self.update()

    def setShowMatched(self, matched: bool) -> None:
        self._show_matched = bool(matched)
        self.update()

    def setPlain(self, plain: bool) -> None:
        """Show the uploaded picture untouched, ignoring selection and preview.

        Used by the 原图 panel: it always draws ``source``, so it stays the
        user's own file no matter which colour is selected on the right.
        """
        self._plain = bool(plain)
        if self._plain:
            self._selected = -1
            self._highlight = None
        self.update()

    def result(self) -> MatchResult | None:
        return self._result

    def setSmooth(self, smooth: bool) -> None:
        """Draw the picture in the block borders of the exported plate, or in blocks.

        1.6.26 (m05557 item 3): 「圆滑色块」 used to change nothing on screen —
        it only reached ``PlateSettings`` at export time — so the one place the
        user could see the difference was the slicer, after a round trip.  Now
        the box drives the preview too, and the picture is rebuilt from the same
        refined grid the plate is cut from.
        """
        smooth = bool(smooth)
        if smooth == self._smooth:
            return
        self._smooth = smooth
        if self._result is not None:
            self._matched = self._build_matched(self._result)
        self.update()

    #: 两个视图（「原图」与「预览」）以前各有一份缓存，同一张图要各自 render 一遍；
    #: 1680×2880 的工程上一遍 0.45 s，删一块等于白付一次（1.6.62，m28297 第 4 条）。
    #: 键是「画出来的东西」（指纹 + 圆滑开关），与对象 id 无关，所以两个视图能共用。
    _SHARED_PREVIEW: dict = {}

    def _build_matched(self, result: MatchResult) -> QImage:
        # 缓存按「画出来的东西」认，不按对象 id 认（m26286 第 4 条）：``_apply_edits``
        # 每次都重建一个 MatchResult，什么都没改也换了对象，于是每次切模式都要重画一遍
        # 预览（实测 2 × 110 ms，全在 plate.smooth_indices 里），这就是用户说的停顿。
        key = (_result_fingerprint(result), self._smooth, int(result.width), int(result.height))
        cached = self._SHARED_PREVIEW.get(key)
        if cached is not None:
            return cached
        image = _to_qimage(render_preview(result, smooth=self._smooth))
        if len(self._SHARED_PREVIEW) > 6:
            self._SHARED_PREVIEW.clear()
        self._SHARED_PREVIEW[key] = image
        self._matched_cache[key] = image
        return image

    # -- geometry ---------------------------------------------------------
    def _target_rect(self) -> QRectF:
        if self._result is None:
            return QRectF()
        width, height = self._result.width, self._result.height
        available_w = max(1, self.width() - 8)
        available_h = max(1, self.height() - 8)
        self._fit_scale = min(available_w / width, available_h / height)
        self._scale = self._fit_scale * self._zoom
        draw_w = width * self._scale
        draw_h = height * self._scale
        if self._zoom <= ZOOM_MIN:
            # 没有放大就没有可平移的余量：永远居中，免得图片被拖到面板外。
            self._pan = QPointF(0.0, 0.0)
        self._pan = self._clamped_pan(self._pan, draw_w, draw_h)
        self._origin = QPointF(
            (self.width() - draw_w) / 2.0 + self._pan.x(),
            (self.height() - draw_h) / 2.0 + self._pan.y(),
        )
        return QRectF(self._origin.x(), self._origin.y(), draw_w, draw_h)

    def _draw_size(self) -> tuple[float, float]:
        """How big the picture is drawn right now, in panel pixels."""
        result = self._result
        if result is None:
            return 0.0, 0.0
        scale = self._fit_scale * self._zoom
        return result.width * scale, result.height * scale

    def _clamped_pan(self, pan: QPointF, draw_w: float, draw_h: float) -> QPointF:
        """Hold the pan inside the picture, so panning never opens a gap."""
        limit_x = max(0.0, (draw_w - self.width()) / 2.0)
        limit_y = max(0.0, (draw_h - self.height()) / 2.0)
        return QPointF(
            min(max(pan.x(), -limit_x), limit_x),
            min(max(pan.y(), -limit_y), limit_y),
        )

    def zoom(self) -> float:
        return self._zoom

    def setTinyHighlight(self, on: bool) -> None:
        """要不要把选中的块画得更扎眼（m26604 第 10 条）。

        0.0% 的块可能只有几像素，正常描边一个像素宽，缩到整张的画面里等于看不见；
        这时把描边加粗、蒙版压得更实，用户才看得出「选中的是这里」。
        """
        on = bool(on)
        if on == self._tiny_highlight:
            return
        self._tiny_highlight = on
        self._highlight = None
        self.update()

    def focusOn(self, mask: np.ndarray) -> None:
        """把 ``mask`` 的重心挪到预览画面正中间，必要时放大（m26604 第 10 条）。

        只挪视角、不改匹配结果：用户点的是一个只有几像素的块，原来它可能贴在画面
        边角，高亮框在视野外，看着像「点了没反应」。
        """
        if self._result is None or mask is None:
            return
        wanted = np.asarray(mask, dtype=bool)
        if wanted.shape != self._result.indices.shape or not wanted.any():
            return
        rows, cols = np.nonzero(wanted)
        centre_x = float(cols.mean())
        centre_y = float(rows.mean())
        # 一小块在整张画面里只有几个像素宽，放大到能看清为止。
        span = max(int(cols.max() - cols.min()) + 1, int(rows.max() - rows.min()) + 1)
        wanted_zoom = ZOOM_MIN
        if span > 0:
            # 让这一块至少占画面短边的 1/6，免得一个孤零零的像素还是一片模糊。
            wanted_zoom = min(8.0, max(ZOOM_MIN, (min(self.width(), self.height()) / 6.0) / max(1.0, span * self._fit_scale)))
        self._zoom = max(self._zoom, wanted_zoom)
        self._target_rect()
        draw_w, draw_h = self._draw_size()
        if draw_w <= 0 or draw_h <= 0:
            self.update()
            return
        # 图片坐标 → 画布坐标，再解出让这个点落在面板中心的平移量。
        px = self._fit_scale * self._zoom * (centre_x + 0.5)
        py = self._fit_scale * self._zoom * (centre_y + 0.5)
        self._pan = self._clamped_pan(
            QPointF(draw_w / 2.0 - px, draw_h / 2.0 - py), draw_w, draw_h
        )
        self.update()

    def reset_zoom(self, *, update: bool = True) -> None:
        """Back to 「整张铺满面板」, centred."""
        self._zoom = ZOOM_MIN
        self._pan = QPointF(0.0, 0.0)
        if update:
            self.update()

    def zoom_by(self, steps: float, focus: QPointF | None = None) -> None:
        """Zoom by ``steps`` wheel notches around ``focus`` (panel pixels).

        Public so the self-check can drive it without a real wheel event.
        """
        if self._result is None or steps == 0.0:
            return
        if focus is None:
            focus = QPointF(self.width() / 2.0, self.height() / 2.0)
        self._zoom_to(self._zoom * (ZOOM_STEP**steps), focus)

    def _zoom_to(self, zoom: float, focus: QPointF) -> None:
        """Zoom so the picture point under ``focus`` stays under ``focus``."""
        result = self._result
        if result is None:
            return
        rect = self._target_rect()  # also refreshes _fit_scale for this panel size
        if self._scale <= 0.0:
            return
        image_x = (focus.x() - rect.x()) / self._scale
        image_y = (focus.y() - rect.y()) / self._scale
        zoom = min(max(zoom, ZOOM_MIN), ZOOM_MAX)
        if abs(zoom - self._zoom) < 1e-9:
            return
        self._zoom = zoom
        scale = self._fit_scale * zoom
        draw_w = result.width * scale
        draw_h = result.height * scale
        pan = QPointF(
            focus.x() - (self.width() - draw_w) / 2.0 - image_x * scale,
            focus.y() - (self.height() - draw_h) / 2.0 - image_y * scale,
        )
        self._pan = self._clamped_pan(pan, draw_w, draw_h)
        self.update()

    def _base_image(self) -> QImage | None:
        if self._plain or not self._show_matched:
            return self._source
        return self._matched

    def _dim_pixmap(self) -> QPixmap | None:
        """The picture with everything but the selected colour faded out.

        Built once per selection and cached: it walks the whole class map, and a
        repaint happens for every wheel notch, every resize and every time the
        window is activated again — including when the user clicks another
        program.  The cache therefore has to be *handed back*, not merely
        remembered.  It used to return ``None`` whenever it was already built,
        so the block showed for exactly one frame and disappeared at the next
        repaint: zooming in and out or clicking another window wiped it.
        """
        result = self._result
        if self._plain or result is None:
            return None
        keep = self._keep_mask(result)
        if keep is None:
            return None
        if self._highlight is None:
            self._highlight = self._build_highlight(result, keep)
        return self._highlight

    def _keep_mask(self, result: MatchResult) -> np.ndarray | None:
        """Which pixels stay lit: the selected blocks, or every pixel of one colour.

        ``None`` means nothing is selected, which has to mean nothing is drawn:
        列表说没选中，预览就不能还画着一块（m00830 item 2）。
        """
        if self._selected_mask is not None:
            if self._selected_mask.shape == result.indices.shape:
                return self._selected_mask
            return None
        if self._selected < 0:
            return None
        return result.indices == self._selected

    def _build_highlight(self, result: MatchResult, keep: np.ndarray) -> QPixmap:
        """Fade everything but ``keep`` and outline the region that keeps it."""
        height, width = result.indices.shape
        canvas = np.zeros((height, width, 4), dtype=np.uint8)
        printed = result.indices >= 0
        # A wash rather than a solid fill: the rest of the picture stays legible
        # as context, so a small highlighted region is not floating in a void.
        canvas[printed & ~keep] = (246, 247, 249, 200 if self._tiny_highlight else 168)
        # Deleted pixels stay fully transparent even while a colour is
        # highlighted: nothing is printed there, and the preview has to say so.
        canvas[keep] = (0, 0, 0, 0)
        overlay = _to_qimage(Image.fromarray(canvas, "RGBA"))
        pixmap = QPixmap.fromImage(overlay)
        # A crisp outline makes the region readable even when it is small.
        outline = self._outline(result, keep)
        if outline is not None:
            painter = QPainter(pixmap)
            painter.drawImage(0, 0, outline)
            painter.end()
        return pixmap

    def _outline(self, result: MatchResult, keep: np.ndarray) -> QImage | None:
        edge = np.zeros_like(keep)
        edge[1:, :] |= keep[1:, :] & ~keep[:-1, :]
        edge[:-1, :] |= keep[:-1, :] & ~keep[1:, :]
        edge[:, 1:] |= keep[:, 1:] & ~keep[:, :-1]
        edge[:, :-1] |= keep[:, :-1] & ~keep[:, 1:]
        if self._tiny_highlight:
            # m26604 第 10 条：一丁点大的块把描边往外扩两圈，否则缩在整张画面里
            # 根本看不出高亮画在哪。
            for _ in range(2):
                thick = np.zeros_like(edge)
                thick[1:, :] |= edge[:-1, :]
                thick[:-1, :] |= edge[1:, :]
                thick[:, 1:] |= edge[:, :-1]
                thick[:, :-1] |= edge[:, 1:]
                edge |= thick
        if not edge.any():
            return None
        canvas = np.zeros((*keep.shape, 4), dtype=np.uint8)
        canvas[edge] = (17, 24, 39, 255)
        return _to_qimage(Image.fromarray(canvas, "RGBA"))

    # -- painting and input ----------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(theme.PANEL_ALT))
        result = self._result
        base = self._base_image()
        if result is None or base is None:
            painter.setPen(QColor(theme.TEXT_FAINT))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "还没有导入图片")
            painter.end()
            return
        rect = self._target_rect()
        # 圆滑色块开着就平滑缩放：放大的边界是连续的，而不是一格一格的锯齿
        # （原始格子是给色块工具看的，那时保持锐利反而更好点）。
        painter.setRenderHint(
            QPainter.RenderHint.SmoothPixmapTransform, bool(self._smooth)
        )
        painter.drawImage(rect, base)
        overlay = self._dim_pixmap()
        if overlay is not None:
            painter.drawPixmap(rect, overlay, QRectF(overlay.rect()))
        painter.setPen(QColor(theme.BORDER_STRONG))
        painter.drawRect(rect)
        painter.end()

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._result is None:
            event.ignore()
            return
        steps = event.angleDelta().y() / 120.0
        if steps == 0.0:
            event.ignore()
            return
        # A high-resolution wheel reports fractions of a notch; the exponent form
        # handles those and whole notches with the same line.
        self.zoom_by(steps, event.position())
        event.accept()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._result is None:
            return
        button = event.button()
        if button == Qt.MouseButton.MiddleButton:
            # 平移专用中键（1.6.51，m20572 item 1）。在此之前左键既挑颜色又平移，
            # 只能靠移动距离猜用户想干哪一件；现在左键只挑颜色，中键只平移。
            self._panning = True
            self._last_pos = QPointF(event.position())
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if button != Qt.MouseButton.LeftButton:
            return
        self._press_pos = QPointF(event.position())

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if not self._panning or self._last_pos is None:
            return
        position = event.position()
        self._target_rect()  # refreshes _fit_scale before the pan is clamped
        draw_w, draw_h = self._draw_size()
        self._pan = self._clamped_pan(self._pan + (position - self._last_pos), draw_w, draw_h)
        self._last_pos = QPointF(position)
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.button() == Qt.MouseButton.MiddleButton:
            if self._panning:
                self._panning = False
                self._last_pos = None
                self.unsetCursor()
                event.accept()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        press = self._press_pos
        self._press_pos = None
        if press is None:
            return
        self._pick_at(event.position())

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.button() != Qt.MouseButton.LeftButton or self._zoom <= ZOOM_MIN:
            return
        # 双击 = 「我看够了，还我整张」。顺手丢掉这次按下留下的状态，免得松手时
        # 又当成一次点选。
        self._press_pos = None
        self.unsetCursor()
        self.reset_zoom()

    def _pick_at(self, position: QPointF) -> None:
        """Report the colour of the pixel under ``position``, if there is one."""
        result = self._result
        if self._plain or result is None:
            return
        rect = self._target_rect()
        if not rect.contains(position):
            return
        x = int((position.x() - rect.x()) / self._scale)
        y = int((position.y() - rect.y()) / self._scale)
        if not (0 <= x < result.width and 0 <= y < result.height):
            return
        index = self._pick_index(result, x, y)
        if index < 0:
            return
        self.setSelected(index)
        self.regionClicked.emit(index)

    def pickMode(self) -> str:
        """How a click chooses a colour: the block it is on, or the nearest edge."""
        return self._pick_mode

    def setPickMode(self, mode: str) -> None:
        """Switch between 「按颜色选择」 and 「按色块边缘选择」 (m01411 item 4)."""
        self._pick_mode = resolve_pick_mode(mode)

    def _pick_index(self, result: MatchResult, x: int, y: int) -> int:
        """What a click at ``(x, y)`` means in the current 「点图片时」 mode.

        「按颜色选择」 answers with a palette index; 「按色块边缘选择」 answers with a
        block id — and that is the whole difference between 「我要改这种颜色」 and
        「我要改这一块」 (m02366 item 2).
        """
        if self._pick_mode != PICK_EDGE:
            return int(result.indices[y, x])
        exact = int(result.indices[y, x])
        labels = self._block_labels
        if labels is not None and labels.shape == result.indices.shape:
            own = int(labels[y, x])
            # 小块优先：深棕色笔画里的小斑点也在这一条里被够着（m13246 item 1）。
            smaller = _pick_smaller_near(labels, self._block_sizes, x, y, own)
            if smaller >= 0:
                return smaller
            return _nearest_label(labels, x, y, fallback=-1)
        return _nearest_edge(result, x, y, fallback=exact)


class _StatusLabel(QLabel):
    """状态行的一句话。

    1.6.57（m26286 第 2 条）：这一行原来始终占着一行高度，没话说的时候就是一块
    空白。``owner`` 指回页面，字一改就把整行（含进度条）按需显隐。
    """

    owner: "PicturePage | None" = None

    def setText(self, text: str) -> None:  # noqa: N802 - Qt 的命名
        super().setText(text)
        if self.owner is not None:
            self.owner._sync_status_bar()

    def clear(self) -> None:  # noqa: D401 - Qt 的命名
        super().clear()
        if self.owner is not None:
            self.owner._sync_status_bar()


class ColourList(QWidget):
    """The colours this print needs, with the clicked one pinned to the top.

    Each row compares the picture's own colour for that region with the spool or
    mix we matched it to, and names the recipe underneath, so the user can see at
    a glance which regions are a good match and which need a manual replacement.
    """

    colourSelected = Signal(int)
    colourActivated = Signal(int)
    #: The user clicked the row of the colour that is already on screen — put it
    #: away again (「再点一下取消选中」).
    colourDeselected = Signal(int)
    #: The SET of selected rows changed — the page re-reads ``selected_indices``.
    selectionChanged = Signal()
    #: Del was pressed on the list: the page does exactly what 「删除」 does
    #: (m01566 item 1).  The key is NOT bound as a window-wide shortcut, because
    #: that would swallow Del inside the hex and number fields.
    deletePressed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._result: MatchResult | None = None
        #: 设上它，行就变成「一块一块」而不是「一种颜色一种颜色」（m02366 item 2/3）。
        self._blocks: BlockMap | None = None
        self._order: list[int] = []
        self._pinned = -1
        self._suppress = False
        self._library: FilamentLibrary | None = None
        #: The rows the user has selected.  Kept here rather than read back from
        #: the widget because ``_refresh`` clears the list on every pin, and the
        #: selection has to survive that.
        self._picked: set[int] = set()
        #: What the last press landed on, and which colour was on screen then.
        self._press_index = -1
        self._press_shown = -1

        self._list = QListWidget()
        self._list.setIconSize(compare_pixmap("#FFFFFF", "#000000").size())
        self._list.setUniformItemSizes(False)
        self._list.setAlternatingRowColors(False)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self._list.setWordWrap(False)
        # 「删除」 and 「合并」 both work on several colours at once, but only after
        # the user says so: 「多选」 starts off and off means exactly one row
        # (m02289 item 1 — 「没有多选却多选了色块」).
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._list.currentRowChanged.connect(self._on_row_changed)
        self._list.itemSelectionChanged.connect(self._on_selection_changed)
        self._list.itemDoubleClicked.connect(self._on_row_double_clicked)
        # 1.6.61（m27752 第 5 条）：颜色详情改成右键打开。三击太难按 —— 连点三下要
        # 每一下都落在大约 0.5 s 内，手指快一点就变成「选中 / 取消选中」。
        self._list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._on_row_context_menu)
        # 「再点一下同一个颜色就取消选中」: Qt moves the current row during the
        # press and only then emits itemPressed, so by the time the click signal
        # arrives the row has already changed and 「the same colour again」 is no
        # longer recognisable.  The event filter notes what was on screen BEFORE
        # Qt touches anything.
        self._list.viewport().installEventFilter(self)
        # Del has to work while the list owns the focus, and ``QListWidget``
        # swallows keys the viewport filter never sees — so the list itself is
        # watched too (m01566 item 1).
        self._list.installEventFilter(self)
        self._list.itemClicked.connect(self._on_row_clicked)

        self._summary = QLabel("")
        self._summary.setProperty("role", "hint")
        self._summary.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._list, 1)
        layout.addWidget(self._summary)

    def setContent(self, result: MatchResult | None, blocks: BlockMap | None = None) -> None:
        """Show ``result`` — its colours, or its blocks when ``blocks`` is given.

        The list is the one place the user points at what they want to change, so
        in 「按色块边缘选择」 it lists blocks (one row each, largest first) and the
        row ids are block labels; in 「按颜色选择」 it lists colours and the ids are
        palette indices.
        """
        self._result = result
        self._blocks = blocks
        self._pinned = -1
        self._picked = set()
        if result is None:
            self._order = []
        elif blocks is not None:
            self._order = blocks.order()
        else:
            self._order = list(range(len(result.palette)))
        self.refresh()

    def setResult(self, result: MatchResult | None) -> None:
        self.setContent(result)

    def setBlocks(self, blocks: BlockMap | None) -> None:
        self.setContent(self._result, blocks)

    def refresh(self) -> None:
        """Rebuild the rows, keeping the pinned selection and the order."""
        self._refresh()
        self._update_summary()

    def setLibrary(self, library: FilamentLibrary | None) -> None:
        """The recipe text names the parent spools, so the list needs the library."""
        self._library = library
        self.refresh()

    def current(self) -> int:
        return self._pinned

    def row_count(self) -> int:
        """How many rows are on screen: one per colour, or one per block (1.6.17)."""
        return self._list.count()

    def row_ids(self) -> list[int]:
        """What the rows carry, top to bottom.

        A palette index in 「按颜色选择」, a **block id** in 「按色块边缘选择」 — the
        read-back the tests and the self-checks use to see what the user is
        actually pointing at (``value_of`` answers with the colour instead).
        """
        ids = []
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item is not None:
                ids.append(int(item.data(Qt.ItemDataRole.UserRole)))
        return ids

    def selected_indices(self) -> list[int]:
        """Every selected colour, in list order — the input to 删除 / 合并.

        Read straight off the widget.  It used to filter the cached ``_picked``
        set instead, and ``_on_selection_changed`` filled that cache FROM this
        method — a closed loop in which a selection could never appear at all,
        so 删除 / 合并 / 撤销 / 恢复 stayed greyed out no matter what was clicked.
        """
        if self._result is None:
            return []
        picked = []
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item is not None and item.isSelected():
                picked.append(int(item.data(Qt.ItemDataRole.UserRole)))
        return picked

    def entries(self) -> list[PaletteEntry]:
        return list(self._result.palette) if self._result is not None else []

    def value_of(self, index: int) -> int:
        """The palette index a row stands for.

        A colour row IS a palette index; a block row answers with the colour that
        block prints with, which is what 详情 / 更换颜色 need (``-1`` if it is gone).
        """
        if self._blocks is not None:
            if 0 <= int(index) < self._blocks.count:
                return self._blocks.value(index)
            return -1
        return int(index)

    def pin(self, index: int, *, reorder: bool = False) -> None:
        """Make ``index`` the current row and reveal it.

        ``reorder=True`` also moves it to the top of the list.  A plain click
        must NOT reorder: the list used to shuffle on every click, so the row
        that appeared under the cursor was a DIFFERENT colour than the one just
        clicked (the user reported exactly that).  The order is now the sort order
        and nothing else; only the sample screenshots ask for a colour to be
        shown first.

        ``index`` is validated against the rows that are actually there, not
        against ``result.palette``: in 「按色块边缘选择」 the rows carry block
        labels, which are neither palette indices nor in palette order.
        """
        if self._result is None or self._row_of(int(index)) < 0:
            return
        self._pinned = int(index)
        if not reorder:
            self._reveal()
            return
        self._order = [self._pinned] + [one for one in self._order if one != self._pinned]
        self._refresh()

    def _row_of(self, index: int) -> int:
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item is not None and int(item.data(Qt.ItemDataRole.UserRole)) == index:
                return row
        return -1

    def _reveal(self) -> None:
        """Highlight the current colour's row and scroll it into view.

        What happens to the selection depends on the mode, and getting this wrong
        greys 删除 / 合并 / 更换颜色 out again:

        * 「一次只选一种」 (``SingleSelection``): ``setCurrentRow`` selects this
          row and drops the others — that IS the new selection, so nothing is put
          back.  It runs suppressed, so the cache and the buttons are refreshed by
          hand afterwards.
        * 「多选」 (``MultiSelection``): the user's set must survive, because
          revealing a row is not a selection act.  ``setCurrentRow`` is free to
          drop the rest, so the set is read first and written back.
        """
        if self._pinned < 0:
            return
        row = self._row_of(self._pinned)
        if row < 0:
            return
        multi = self._list.selectionMode() == QAbstractItemView.SelectionMode.MultiSelection
        # The pinned row belongs to the selection even while 「多选」 is on: revealing it
        # used to write the old set back on top of it, which left the list claiming
        # nothing was selected while 预览 drew that colour's block (m01411 item 3).
        keep = set(self.selected_indices()) | {self._pinned} if multi else set()
        self._suppress = True
        self._list.setCurrentRow(row)
        self._list.scrollToItem(self._list.item(row), QAbstractItemView.ScrollHint.EnsureVisible)
        self._suppress = False
        if multi:
            self.select(keep)
        else:
            self._on_selection_changed()

    def toggle(self, index: int) -> None:
        """Add or remove one colour from the selection.

        This is what a click on a block in 「预览」 does while 「多选」 is on: the
        picture and the list are two views of the same regions, so pointing at a
        region in the picture has to work like pointing at its row.
        """
        row = self._row_of(int(index))
        if row < 0:
            return
        item = self._list.item(row)
        item.setSelected(not item.isSelected())
        if not item.isSelected() and self._pinned == index:
            # 在预览里再点一次就是把这一块拿掉。它已经不在多选里了，高亮也不能
            # 还赖在它身上（m01411 item 3：列表说没选中的东西，预览不能还画着）。
            self._pinned = -1
        self._reveal()

    def select(self, indices) -> None:
        """Select these palette indices, exactly as clicking them would.

        Used by the automatic checks to reach 删除 / 合并 without a mouse.  It
        must go THROUGH the widget (not the ``_picked`` cache) so the same
        ``itemSelectionChanged`` path runs as for a real selection.
        """
        wanted = {int(index) for index in indices}
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item is not None:
                item.setSelected(int(item.data(Qt.ItemDataRole.UserRole)) in wanted)

    def setMultiSelect(self, on: bool) -> None:
        """Make a plain click ADD to the selection instead of replacing it.

        Ctrl+click used to do this, but a modifier nobody was told about is not
        a feature — the page puts a 「多选」 toggle in front of the buttons
        instead.  The mode is changed on the widget (we never fake a selection),
        so the same ``itemSelectionChanged`` path keeps 删除 / 合并 in sync.

        Off is not 「ExtendedSelection with Ctrl」 but a real single selection:
        多选 关着就是一次只认一样东西 (m02289 item 1).  用户报的正是「没有多选却多
        选了几块」——那条路上有三扇门：Ctrl+click 悄悄加行、关掉「多选」时把多选
        留在列表里、以及编辑之后按块回填选中集。三扇门一起关上：关掉时只留光标那一
        行，模式回到 SingleSelection，回填也只看第一块（``_apply_edits``）。
        """
        mode = (
            QAbstractItemView.SelectionMode.MultiSelection
            if on
            else QAbstractItemView.SelectionMode.SingleSelection
        )
        if self._list.selectionMode() == mode:
            return
        keep = self.selected_indices()
        # Qt is free to drop the selection while the mode changes; that is not the
        # user emptying the set, so nothing may react to it.  Read it first, change
        # the mode behind a suppress, then put the set back through the widget so
        # the buttons still agree with what is highlighted.
        self._suppress = True
        self._list.setSelectionMode(mode)
        self._suppress = False
        if on:
            self.select(keep)
            return
        # 关掉「多选」：光标那一行活下去（没有光标就留第一个），其余放掉。
        survivor = self._pinned if self._pinned in keep else (keep[0] if keep else -1)
        self.select([] if survivor < 0 else [survivor])

    def _refresh(self) -> None:
        result = self._result
        # Everything that touches the view happens while suppressed: selecting
        # row 0 emits ``currentRowChanged``, which would otherwise come straight
        # back in here and recurse until the stack runs out.
        self._suppress = True
        self._list.clear()
        if result is not None:
            total = max(1, result.printed_pixels)
            if self._blocks is not None:
                self._fill_block_rows(result, total)
            else:
                self._fill_colour_rows(result, total)
            # Deleting or merging rebuilds the rows, and the user's selection has
            # to come back with them; a colour that is gone simply drops out.
            live = {
                int(self._list.item(row).data(Qt.ItemDataRole.UserRole))
                for row in range(self._list.count())
            }
            self._picked &= live
            for row in range(self._list.count()):
                item = self._list.item(row)
                if int(item.data(Qt.ItemDataRole.UserRole)) in self._picked:
                    item.setSelected(True)
        else:
            self._picked = set()
        self._suppress = False
        # The rows moved: put the highlight back on the colour it was on, not on
        # whatever happens to be row 0 now.
        self._reveal()

    def _groups(self, result: MatchResult) -> list[tuple[int, list[int]]]:
        """一行一种**打印出来的颜色**：同一个配方被几块选中就并成一行（m12426 item 1）。

        标题写着「打印图片需要的颜色」，那这张表就不该有重复行 —— 用户看到同一种颜色
        出现三次（三个色块都印 ``#F5D2B8``），那是色块级的事实，属于「按色块边缘选择」
        那张表（一行一块，m11985），不该在这里冒出来。

        返回 ``(代表位子, 这一行的所有位子)``，代表取像素最多的那一个，这样高亮、详情
        和「图片里的颜色」都指着他。
        """
        order: list[tuple[int, list[int]]] = []
        seen: dict[str, int] = {}
        for index in self._order:
            key = result.palette[index].key
            if key in seen:
                order[seen[key]][1].append(index)
                continue
            seen[key] = len(order)
            order.append((index, [index]))
        grouped = []
        for index, members in order:
            survivor = max(members, key=lambda position: int(result.counts[position]))
            grouped.append((int(survivor), [int(position) for position in members]))
        return grouped

    def members_of(self, index: int) -> list[int]:
        """这一行代表的**所有**调色板位子（分组之后一行可能顶着好几个）。

        「删除」「合并」「更换颜色」说的是「这一种颜色」，所以它们要作用在这一行的
        全部位子上，而不是碰巧当代表的那个 (m12426 item 1)。
        """
        result = self._result
        if result is None or not 0 <= int(index) < len(result.palette):
            return [int(index)]
        key = result.palette[int(index)].key
        return [
            position
            for position in range(len(result.palette))
            if result.palette[position].key == key
        ]

    def expanded(self, indices: list[int]) -> list[int]:
        """把行号摊开成它们代表的全部位子，按出现的先后去重。"""
        out: list[int] = []
        for index in indices:
            for position in self.members_of(int(index)):
                if position not in out:
                    out.append(position)
        return out

    def _fill_colour_rows(self, result: MatchResult, total: int) -> None:
        """「按颜色选择」：一行一种打印出来的颜色，行号是它的代表位子。"""
        for index, members in self._groups(result):
            entry = result.palette[index]
            count = int(sum(int(result.counts[position]) for position in members))
            # 1.6.56（m25951 第 1 条）：一个像素都没有的颜色不列出来。它印不出来，删它
            # 也没有像素可删（删除是把那些像素改成透明），于是那行永远删不掉、永远挂着
            # 一个 0.0%。不列，问题就不存在。
            if count <= 0:
                continue
            image_hex = hex_of(result.region_colour(index))
            share = count / total * 100.0
            where = "" if len(members) == 1 else f" 等 {len(members)} 处"
            row = QListWidgetItem(
                compare_pixmap(image_hex, entry.color_hex, 44, 20),
                f"{image_hex}{where} → {entry.color_hex}    {share:.1f}%\n"
                f"{entry_recipe_text(entry, self._library, compact=True)}",
            )
            row.setData(Qt.ItemDataRole.UserRole, int(index))
            # No hover tooltip here: the row already prints the picture's
            # own colour, the matched colour, the share and the recipe, so
            # the black pop-up only repeated it.
            self._list.addItem(row)

    def _fill_block_rows(self, result: MatchResult, total: int) -> None:
        """「按色块边缘选择」：一行一块（m02366 item 2/3）。

        The row id is the block label, NOT a palette index — that is why
        ``value_of`` exists.  The text still names the colour the block prints
        with, because 更换颜色 / 合并 are about colours even here.
        """
        blocks = self._blocks
        assert blocks is not None
        for rank, label in enumerate(self._order, start=1):
            value = blocks.value(label)
            size = blocks.size(label)
            # 1.6.56（m25951 第 1 条）：空块不列（理由同颜色那一栏：删不掉，只会挂着）。
            if size <= 0:
                continue
            share = size / total * 100.0
            if 0 <= value < len(result.palette):
                entry = result.palette[value]
                image_hex = hex_of(result.region_colour(value))
                text = (
                    f"第 {rank} 块  {image_hex} → {entry.color_hex}    {share:.1f}%\n"
                    f"{entry_recipe_text(entry, self._library, compact=True)}"
                )
                icon = compare_pixmap(image_hex, entry.color_hex, 44, 20)
            else:
                text = f"第 {rank} 块    {share:.1f}%"
                icon = compare_pixmap("#FFFFFF", "#FFFFFF", 44, 20)
            row = QListWidgetItem(icon, text)
            row.setData(Qt.ItemDataRole.UserRole, int(label))
            self._list.addItem(row)

    def _update_summary(self) -> None:
        result = self._result
        if result is None:
            self._summary.setText("导入图片后，这里列出打印需要的颜色。")
            return
        selected = len(self.selected_indices())
        if self._blocks is not None:
            extra = f" · 选中 {selected} 块" if selected else ""
            self._summary.setText(
                f"{result.width}×{result.height} 像素 · 实心 {result.printed_pixels} 像素"
                f" · 共 {self._blocks.count} 个色块{extra}"
            )
            return
        extra = f" · 选中 {selected} 种" if selected else ""
        # 「共几种颜色」按**打印出来的颜色**数：同一个配方被三块选中是一种颜色，不是三种
        # （m12426 item 1）。要看色块数量是「按色块边缘选择」那张表的事。
        colours = len({entry.key for entry in result.palette})
        self._summary.setText(
            f"{result.width}×{result.height} 像素 · 实心 {result.printed_pixels} 像素"
            f" · 共 {colours} 种颜色{extra}"
        )

    def _on_row_changed(self, row: int) -> None:
        if self._suppress or row < 0:
            return
        item = self._list.item(row)
        if item is None:
            return
        index = int(item.data(Qt.ItemDataRole.UserRole))
        self.colourSelected.emit(index)

    def _on_selection_changed(self) -> None:
        if self._suppress:
            return
        picked = self.selected_indices()
        if len(picked) > 1 and self._list.selectionMode() != (
            QAbstractItemView.SelectionMode.MultiSelection
        ):
            # 最后一道门：「多选」关着就只认一样东西——不管这多出来的几行是
            # Ctrl+click、item.setSelected() 还是别的代码留下的 (m02289 item 1)。
            keeper = self._pinned if self._pinned in picked else picked[0]
            self._suppress = True
            self.select([keeper])
            self._suppress = False
            self._pinned = keeper
            picked = [keeper]
        self._picked = set(picked)
        if not self._picked:
            # 「一个都没选中」也要把光标那一行放掉。点列表下面的空白处（Qt 直接清选中）
            # 或者「多选」里点掉最后一行都会走到这里，而页面上的预览高亮只认选中集：
            # 光标记着的那一块必须跟着一起没有 (m01191 item 1, m00830 item 2)。
            self._pinned = -1
        self._update_summary()
        self.selectionChanged.emit()

    def _on_row_double_clicked(self, item: QListWidgetItem) -> None:
        """双击什么都不做（1.6.59，m26980 第 3 条；1.6.61 起详情在右键上）。

        双击在「多选」和「连点换选中」两条路上太容易误触 —— 用户本来只是想连点
        两下换一行，详情窗口就弹出来盖住半边。1.6.61 起详情改走右键，见
        :meth:`_on_row_context_menu`。
        """

    def _on_row_context_menu(self, position: QPoint) -> None:
        """右键一行就打开它的颜色详情（1.6.61，m27752 第 5 条）。

        原来要连点三下，间隔还得落在双击时间里，手一快就变成「选中 / 取消选中」。
        右键是一步到位的动作，而且不会误碰列表的选中状态。
        """
        item = self._list.itemAt(position)
        if item is None:
            return
        self.colourActivated.emit(int(item.data(Qt.ItemDataRole.UserRole)))

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt spelling
        """Remember the row and the shown colour from before the press landed.

        ``QAbstractItemView.mousePressEvent`` sets the current index first and
        emits ``pressed``/``clicked`` afterwards; a filter on the viewport runs
        before both, which is the only place the previous state is still intact.
        """
        if watched is self._list and event.type() == QEvent.Type.KeyPress:
            # Del on the list means 「删除」 (m01566 item 1).  A list is not a text
            # field, so nothing here is taken away from the user's typing.
            if event.key() == Qt.Key.Key_Delete:
                self.deletePressed.emit()
                return True
        if watched is self._list.viewport() and event.type() == QEvent.Type.MouseButtonPress:
            position = event.position().toPoint()
            row = self._list.row(self._list.itemAt(position))
            item = self._list.item(row) if row >= 0 else None
            self._press_index = (
                int(item.data(Qt.ItemDataRole.UserRole)) if item is not None else -1
            )
            self._press_shown = self._pinned
        return super().eventFilter(watched, event)

    def _on_row_clicked(self, item: QListWidgetItem) -> None:
        """A click on the colour that was ALREADY on screen means 「取消选中」.

        「多选」 is left out: there a second click toggles the row off by itself,
        which already drops it out of the selection the buttons work on.
        """
        if self._suppress:
            return
        if self._list.selectionMode() == QAbstractItemView.SelectionMode.MultiSelection:
            return
        index = int(item.data(Qt.ItemDataRole.UserRole))
        if index == self._press_index == self._press_shown:
            self.colourDeselected.emit(index)

    def clear_selection(self) -> None:
        """Drop the highlight from every row, without touching ``_pinned``'s order."""
        self._pinned = -1
        self._list.clearSelection()
        self._list.setCurrentRow(-1)


class MergeDialog(QDialog):
    """Ask which of the selected colours the merged region should become.

    The user's requirement is explicit: 「合并后弹出窗口，从多选的颜色里选择一个
    颜色作为合并后的整体颜色」 — so the dialog lists exactly the selected colours
    and nothing else.
    """

    def __init__(self, entries, *, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("合并颜色")
        self._list = QListWidget()
        self._list.setIconSize(swatch_pixmap("#FFFFFF", 44, 20).size())
        for index, entry in enumerate(entries):
            row = QListWidgetItem(
                swatch_pixmap(entry.color_hex, 44, 20),
                f"{entry.color_hex}    {entry.short_label}",
            )
            row.setData(Qt.ItemDataRole.UserRole, index)
            self._list.addItem(row)
        self._list.setCurrentRow(0)
        self._list.itemDoubleClicked.connect(lambda _item: self.accept())

        hint = QLabel("这些颜色会并成一种，整块区域都用你选中的这个颜色打印。")
        hint.setProperty("role", "hint")
        hint.setWordWrap(True)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        ok.setText("合并")
        ok.setProperty("accent", "true")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        box = QVBoxLayout(self)
        box.setContentsMargins(14, 14, 14, 14)
        box.setSpacing(10)
        box.addWidget(hint)
        box.addWidget(self._list, 1)
        box.addWidget(buttons)
        self.resize(430, 340)
        selectable_text(self)

    def chosen(self) -> int:
        """Position inside the list this dialog was opened with, or -1."""
        item = self._list.currentItem()
        if item is None:
            return -1
        return int(item.data(Qt.ItemDataRole.UserRole))


class SvgSizeDialog(QDialog):
    """导入 SVG 时问一句「渲成多少像素」（1.6.49，m20069）。

    为什么只对矢量图问：位图有多少像素就多少像素 —— 放大只是插值。m19452 量过：240 px
    的图插到 512，第一行平均 ΔE00 从 4.82 掉到 7.00（最差 7.46→13.35、差块 4→6、颜色
    7→9），插出来的过渡色只会让四卷耗材去追更多颜色。矢量图没有分辨率，渲得大出来的是
    **真**细节：同一张声明 240×240 的 SVG（m20031 那张猪），渲成 512 时第一行 21 种颜色、
    预览顺滑，渲成 240 时细笔画只剩 1 px 宽，灰色描边和背景渐变被并成 7 种颜色。

    默认值就是它在文件里声明的宽 × 长，所以直接回车等于 1.6.48 的行为；「锁定长宽比」默认
    勾着（改一边另一边按原始比例跟着走，取整到整数像素），想要非等比拉伸就取消它。
    """

    def __init__(
        self, declared: tuple[int, int], *, name: str = "", parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("这张 SVG 渲成多少像素")
        self._ratio = (declared[1] / declared[0]) if declared[0] else 1.0
        self._syncing = False

        self._width = QSpinBox()
        self._height = QSpinBox()
        for box, value in ((self._width, declared[0]), (self._height, declared[1])):
            box.setRange(VECTOR_MIN_PX, VECTOR_MAX_PX)
            box.setSuffix(" px")
            box.setAccelerated(True)
            box.setValue(min(VECTOR_MAX_PX, max(VECTOR_MIN_PX, int(value))))
        self._width.valueChanged.connect(self._on_width)
        self._height.valueChanged.connect(self._on_height)

        self._lock = QCheckBox("锁定长宽比")
        self._lock.setChecked(True)
        self._lock.toggled.connect(lambda _on: self._sync_note())

        hint = QLabel(
            f"{name} 是矢量图，没有分辨率：渲得大一点，细笔画才有像素。"
            "位图不弹这个框 —— 放大只是插值，多不出细节。"
        )
        hint.setProperty("role", "hint")
        hint.setWordWrap(True)

        self._note = QLabel()
        self._note.setProperty("role", "hint")
        self._note.setWordWrap(True)
        self._sync_note()

        row_width = QHBoxLayout()
        row_width.addWidget(QLabel("宽度"))
        row_width.addWidget(self._width)
        row_width.addStretch(1)
        row_height = QHBoxLayout()
        row_height.addWidget(QLabel("长度"))
        row_height.addWidget(self._height)
        row_height.addStretch(1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        ok.setText("用这个尺寸导入")
        ok.setProperty("accent", "true")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        box = QVBoxLayout(self)
        box.setContentsMargins(14, 14, 14, 14)
        box.setSpacing(10)
        box.addWidget(hint)
        box.addLayout(row_width)
        box.addLayout(row_height)
        box.addWidget(self._lock)
        box.addWidget(self._note)
        box.addWidget(buttons)
        self.resize(430, 230)
        selectable_text(self)

    def chosen(self) -> tuple[int, int]:
        """用户写下的宽 × 长（px）。"""
        return (int(self._width.value()), int(self._height.value()))

    def _on_width(self, value: int) -> None:
        if self._syncing:
            return
        if self._lock.isChecked():
            self._syncing = True
            self._height.setValue(self._clamp(round(value * self._ratio)))
            self._syncing = False
        self._sync_note()

    def _on_height(self, value: int) -> None:
        if self._syncing:
            return
        if self._lock.isChecked() and self._ratio:
            self._syncing = True
            self._width.setValue(self._clamp(round(value / self._ratio)))
            self._syncing = False
        self._sync_note()

    @staticmethod
    def _clamp(value: int) -> int:
        return max(VECTOR_MIN_PX, min(VECTOR_MAX_PX, int(value)))

    def _sync_note(self) -> None:
        width, height = self.chosen()
        note = (
            f"匹配 {width * height:,} 个像素：板 {width * CELL_MM:.1f} × {height * CELL_MM:.1f} mm"
            f"（一格 {CELL_MM} mm），渲出来的就是 {width} × {height} 像素的位图。"
        )
        # 1.6.62（m28297 第 1 条）：像素数不再封顶，板就可能大过机器行程。这里照实说
        # 一句，别让用户导进来才发现导出时超框。
        longest = max(width, height) * CELL_MM
        if longest > PLATE_MM:
            note += (
                f"\n注意：最长一边 {longest:.0f} mm，超过 {PLATE_MM:.0f} mm 的机器行程 ——"
                "导出前得把图缩到这个范围里。"
            )
        self._note.setText(note)


def _ask_vector_size(
    parent: QWidget | None, path: Path, declared: tuple[int, int]
) -> tuple[int, int] | None:
    """导入 SVG 之前弹窗问一次（1.6.49）。用户按「取消」返回 ``None`` —— 这张图就不导入。

    单独抽成一个函数是为了让闸门脚本能把它换掉：冒烟与自检里没有人能点那个框。
    """
    dialog = SvgSizeDialog(declared, name=path.name, parent=parent)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    return dialog.chosen()


class PicturePage(QWidget):
    """Import a picture, match it against the filament library, export a plate."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._library: FilamentLibrary | None = None
        self._catalog: MixCatalog | None = None
        self._image: Image.Image | None = None
        self._image_name: str = "混色底板"
        #: 磁盘上那张图本来的像素数（m19402）：状态行要说的是这张原图，不是算过的那份。
        self._source_size: tuple[int, int] | None = None
        self._source_qimage: QImage | None = None
        #: True once 删除/合并 have changed the picture itself (1.6.19, m01566 item 2:
        #: 「预览的色块删除合并之后，原图也一并修改…原图修改后，标注原图已修改」).
        #: 「更换颜色」 never sets it — the user was explicit that a colour swap is a
        #: print decision and must leave the original alone.
        self._original_edited: bool = False
        #: The 「原图已修改」 badge in the 原图 panel header (built by ``_wrap``).
        self._original_badge: QLabel | None = None
        #: The ``.bpproj`` this picture came from, or the one it was last saved
        #: to.  Ctrl+S overwrites it without asking; ``None`` means the next save
        #: has to ask where to put the file.
        self._project_path: Path | None = None
        #: True while there are edits that are not in ``_project_path`` yet, so
        #: 「打开工程」 can warn before throwing them away.
        self._dirty = False
        # ``_base`` is the untouched match; ``_result`` is the base plus whatever
        # the user did to it.  Everything the user does is stored as an edit
        # against the ORIGINAL palette indices, never by mutating the pixels, so
        # undo and redo are a snapshot of three small dictionaries rather than a
        # megabyte of image data.
        self._base: MatchResult | None = None
        self._result: MatchResult | None = None
        self._plate: PlateModel | None = None
        #: Original indices the user deleted — those pixels become transparent,
        #: which means no base plate and nothing printed there.
        self._deleted: set[int] = set()
        #: ``source -> target``, original indices: the source's pixels are
        #: printed with the target's colour.
        self._merged: dict[int, int] = {}
        #: ``original index -> PaletteEntry``, the 「更换颜色」 replacements.
        self._replaced: dict[int, PaletteEntry] = {}
        #: 「按色块边缘选择」 的块级编辑，按发生顺序叠加（m02366 item 3）。每一条都是
        #: 像素级的，因为「删除这一块」跟「删除这一种颜色」在图上不是一回事。
        self._block_edits: list[BlockPaint] = []
        #: 当前匹配上认出来的色块；只在块模式下建（顺序、掩码都由它出）。
        self._blocks: BlockMap | None = None
        #: 上一次算出来的色块表，连同它对应的那份结果。切回「按色块边缘选择」时同一份
        #: 结果（同一个对象）没必要再分一次块——分块要跑一遍全分辨率的连通域，正是用户
        #: 说的那一下停顿（m26286 第 4 条）。结果一换（重新匹配、改色块、撤销）对象就
        #: 不是同一个了，缓存自然失效。
        self._blocks_cache: tuple[tuple, BlockMap] | None = None
        #: 「这一份划分说了算」的标签图（m15721 第 1 条）。「按当前色块重新匹配」重新
        #: 分色时把旧划分留在这里，_:meth:`_build_blocks` 见到尺寸对得上就用它，块数和
        #: 编号都不变；导入新图、从头匹配、打开工程都会清掉它。
        self._block_template: np.ndarray | None = None
        #: 匹配栅格的上限（m16941 第 1 条）。 以前界面上有个「最大边长」数字框，默认 512，
        #: 用户得自己去调；现在没有这个框：导入图片时就取图片自己的最长边（SVG 取它在文件
        #: 里声明的那一档），一格就是一个原像素。 打开工程仍按工程里存的那个数走。
        self._span: int = 512
        #: 当前「点图片时」的方式。存在页面里（而不是只问 view），是因为历史、
        #: 列表和块表都要看它（m02439 的答复：切换方式本身算一步撤销）。
        self._mode: str = DEFAULT_PICK_MODE
        self._undo_stack: list[tuple] = []
        self._redo_stack: list[tuple] = []
        #: Live palette index -> original index, rebuilt by ``_compose``.
        self._live_map: list[int] = []
        #: The colour on screen right now (-1 = none).「取消选中」就是什么都不选，不再
        #: 退回「上一次看过的颜色」——那看起来像「点这一个却选中了上一次那个」
        #: （m12426 item 2）。
        self._shown = -1
        # Set when the library changed while this page was off screen; the
        # re-match is then done in showEvent instead of in setLibrary.
        self._stale = False
        # Cache for _candidate_palette: the 全部颜色 list, dropped by setLibrary.
        self._candidates: list[PaletteEntry] | None = None
        #: 反过来那一侧：勾了「只匹配耗材」时要的「只有耗材本色」那份（1.6.52）。
        self._candidates_plain: list[PaletteEntry] | None = None
        #: The three-filament catalogue, pushed in by the main window whenever it
        #: is (re)built.  ``None`` means 「not ready」: matching then runs on the
        #: two-filament palette alone, and picks up the triples when they land.
        self._triples = None
        self._triple_palette = None
        #: The head list ``_triple_palette`` was built from, so the cache can be
        #: invalidated by identity instead of by rebuilding the tail.
        self._triple_head: list[PaletteEntry] | None = None
        self._build_ui()
        self._set_controls_enabled(False)
        self._update_actions()

    # -- construction -----------------------------------------------------
    def _build_ui(self) -> None:
        self._import_button = QPushButton("导入图片")
        self._import_button.setProperty("accent", "true")
        self._import_button.clicked.connect(self._on_import)

        # 「按当前色块重新匹配」按一次做两件事（m15721 第 1 条）：先从改过的原图重新分色
        # （「颜色数量」说了算，所以颜色表可能变），再拿着**现有这些色块**去重新挑卷。
        # 色块的划分和编号不动 —— 用户在色块模式里点出来、改过的那些块还在原处，只用新
        # 颜色重算每块打哪一色；颜色表里行级的「更换颜色」会清掉（那几行按下标记的，新
        # 颜色表一换就对不上了）、色块级的编辑按像素掩码留着。
        # 从前那个「重新匹配」按钮已删：它和这个只差「要不要重新分块」，两个按钮并排时
        # 用户得先想清楚自己要问哪个问题，而分块是软件该自己认的事。
        self._rematch_blocks_button = QPushButton("按当前色块重新匹配")

        # 「点图片时」：how a click in 预览 chooses a colour (m01411 item 4).  No
        # tooltip — a black pop-up over the picture is the one thing this page does
        # not do; the 帮助 window explains both modes.
        self._pick_combo = QComboBox()
        for mode in PICK_MODES:
            self._pick_combo.addItem(PICK_MODE_LABELS[mode], mode)
        self._pick_combo.currentIndexChanged.connect(self._on_pick_mode_changed)
        self._rematch_blocks_button.clicked.connect(self._on_rematch_blocks)

        # 打开 / 保存工程 sit next to 导入图片: all three are file actions on the
        # picture, and a session that is put down and picked up again is the
        # whole point of the project file.  Neither is accented — the blue
        # button on this page is 生成 3MF, the one that finishes the job.
        self._open_project_button = QPushButton("打开工程")
        self._open_project_button.clicked.connect(self._on_open_project)

        self._save_project_button = QPushButton("保存工程")
        self._save_project_button.clicked.connect(lambda: self._on_save_project())

        # 「导出原图」：把改过的原图存成 PNG（m12426 item 3）。删除和合并会**同时改原图**
        # （m02316 item 2），可存下来的只有工程文件里的那份；用户要把这张改过的图拿到
        # 别处用（继续修图、发给别人），得有个直接存图的出口。它跟 导入/打开/保存 三个
        # 文件动作排在一起，也不上蓝色 —— 这一页的蓝按钮只有 生成 3MF。
        self._export_original_button = QPushButton("导出原图")
        self._export_original_button.clicked.connect(self._on_export_original)

        # 「导出 SVG」（1.6.64，m29374）：把这张图描成矢量。用户要求按钮排在「点图片时：
        # 按颜色选择」后面，所以它加在下拉框之后、弹簧之前 —— 那一排左边全是文件动作，
        # 最右这件也是文件动作，读起来是一条线。
        self._export_svg_button = QPushButton("导出 SVG")
        self._export_svg_button.clicked.connect(self._on_export_svg)

        self._colour_count = QSpinBox()
        self._colour_count.setRange(2, MAX_PAINTED_FILAMENTS)
        self._colour_count.setValue(DEFAULT_MAX_COLOURS)

        self._spool_cap = QSpinBox()
        self._spool_cap.setRange(0, MAX_PAINTED_FILAMENTS)
        self._spool_cap.setValue(DEFAULT_SPOOL_CAP)
        # m09456 item 4 (1.6.28): 上限改了以后，手上那几行可能已经超过新的上限 —— 用户
        # 截图里就出现过「耗材上限 4」而方案行写着「用 9 卷」。当场把超限的行标出来，不等
        # 「重新匹配」（那一次要 1.3–3.0 s，不适合挂在数字框上）。
        self._spool_cap.valueChanged.connect(self._on_spool_cap_changed)

        self._dither = QCheckBox("抖动")

        # 「只匹配耗材」（m21172 item 2）：勾上以后不拿预制混色去凑，只用我自己这几卷
        # 耗材的本色 —— 也就是 Bambu Studio 里「一种颜色一卷」的打法。跨材质：耗材列表本来
        # 就不分材质，PETG 的卷也在候选里，所以勾上照样能跨过去。比混色差多少，文档第 78
        # 章有实测数字。
        self._only_spools = QCheckBox("只匹配耗材")
        self._only_spools.setToolTip(
            "勾上就只用你自己这些耗材，不再拿两色 / 三色混色去凑（可以跨材质）。\n"
            "颜色会差一些，但每一块就是一卷料，Bambu Studio 里不用配比。"
        )
        self._only_spools.toggled.connect(self._on_only_spools)

        #: 设置里的实验性开关（1.6.60，m27359 第 3 条 / 1.6.61，m27752 第 6 条）。
        #: 它们不是控件，是「这一版按不按颜色数量 / 耗材上限停手」的旗标，由
        #: apply_settings 灌进来。
        self._no_colour_limit = False
        self._no_spool_limit = False

        top = QHBoxLayout()
        top.setSpacing(8)
        top.addWidget(self._import_button)
        top.addWidget(self._open_project_button)
        top.addWidget(self._save_project_button)
        top.addWidget(self._export_original_button)
        top.addWidget(QLabel("颜色数量:"))
        top.addWidget(self._colour_count)
        top.addWidget(QLabel("耗材上限:"))
        top.addWidget(self._spool_cap)
        top.addWidget(self._dither)
        top.addWidget(self._only_spools)
        top.addWidget(self._rematch_blocks_button)
        top.addWidget(QLabel("点图片时:"))
        top.addWidget(self._pick_combo)
        top.addWidget(self._export_svg_button)
        top.addStretch(1)
        # 1.6.56（m25951 第 4 条）把「点图片时改成…」那句搬到右上角；1.6.57（m26604
        # 第 2 条）用户要求把那句话删掉 —— 下拉框里写着现在是哪种模式，做法本身在
        # 「帮助」里。这里不再建标签，`_update_mode_hint()` 留成空实现给旧调用点。

        # 底板厚度 = **整块**的厚度，含图上那一层颜色（1.6.42）。 一格多宽由喷嘴定
        # （CELL_MM = 0.4 mm），颜色层厚度由 Bambu Studio 自己管，这两个都不用再问用户。
        self._base_spin = QDoubleSpinBox()
        self._base_spin.setRange(COLOUR_BAND_MM + 0.2, 20.0)
        self._base_spin.setValue(1.4)
        self._base_spin.setSingleStep(0.2)
        self._base_spin.setSuffix(" mm")
        self._base_spin.setToolTip(
            "整块底板的厚度，含图上那一层颜色（0.4 mm）。 除以你的层高就是板有几层。"
        )

        # 底板颜色 (m16941 第 2 条): 从前这里是一个只列「打印图片需要的颜色」那几行的
        # 下拉框 —— 想拿别的混色当底板就没路可走。 现在跟「更换颜色」用同一个挑选框
        # （整套混色都能搜），旁边留一个小色块显示当前挑的是哪一种；没挑就跟着图上最大
        # 的一块走。
        self._base_entry: PaletteEntry | None = None
        self._base_swatch = QLabel()
        self._base_swatch.setFixedSize(18, 18)
        self._base_swatch.setToolTip(BASE_COLOUR_TIP)
        self._base_button = QPushButton("更换底板颜色")
        self._base_button.setToolTip(BASE_COLOUR_TIP)
        self._base_button.clicked.connect(self._on_pick_base_colour)


        # 导出时圆滑色块 (1.6.22): 匹配栅格上的色块边界是一格一格的台阶，在 Bambu
        # Studio 里放大看就是锯齿。 默认勾上；不勾就照 1.6.21 的样子导出。
        # 1.6.26 (m05557 item 3): 这个勾现在也管预览——勾着的时候预览按导出用的
        # 细格画，边界是连续的；不勾还是原始格子，方便用色块工具点格子。
        self._smooth_box = QCheckBox(SMOOTH_LABEL)
        self._smooth_box.setChecked(True)
        self._smooth_box.setToolTip(SMOOTH_TIP)
        self._smooth_box.toggled.connect(self._on_smooth_toggled)

        # 图片印在哪一面 (1.6.23): 默认底面，和 1.6.22 及以前的导出上下对调。
        self._side_combo = QComboBox()
        self._side_combo.addItem(IMAGE_SIDE_BOTTOM, True)
        self._side_combo.addItem(IMAGE_SIDE_TOP, False)
        self._side_combo.setToolTip(IMAGE_SIDE_TIP)

        self._export_3mf = QPushButton("生成 3MF（Bambu Studio）")
        self._export_3mf.setProperty("accent", "true")
        self._export_3mf.clicked.connect(lambda: self._on_export("3mf"))

        self._export_obj = QPushButton("生成 OBJ + MTL")
        self._export_obj.clicked.connect(lambda: self._on_export("obj"))

        bottom = QHBoxLayout()
        bottom.setSpacing(8)
        bottom.addWidget(QLabel("底板厚度:"))
        bottom.addWidget(self._base_spin)
        bottom.addWidget(QLabel("底板颜色:"))
        bottom.addWidget(self._base_swatch)
        bottom.addWidget(self._base_button)
        bottom.addStretch(1)
        bottom.addWidget(QLabel(IMAGE_SIDE_LABEL + ":"))
        bottom.addWidget(self._side_combo)
        bottom.addWidget(self._smooth_box)
        bottom.addWidget(self._export_obj)
        bottom.addWidget(self._export_3mf)

        self._view = PictureView()
        self._view.regionClicked.connect(self._on_region_clicked)
        # Whichever mode the user used last, from the previous session.
        self._set_pick_mode(load_settings().get("pick_mode", DEFAULT_PICK_MODE), save=False)

        # 原图 and 预览 sit side by side so the user can compare what they
        # imported with what the printer will actually lay down.  The original
        # panel is in PLAIN mode: it draws the uploaded file itself and ignores
        # the selection entirely, so it stays a true reference instead of a
        # washed-out copy of the preview.
        self._original_view = PictureView()
        self._original_view.setPlain(True)

        self._list = ColourList()
        self._list.colourSelected.connect(self._on_colour_selected)
        self._list.colourDeselected.connect(self._on_colour_deselected)
        self._list.colourActivated.connect(self._on_colour_activated)
        self._list.selectionChanged.connect(self._update_actions)
        # 块模式下「多选」是一组色块，预览要画的是这一组的并集，不只是光标那一行。
        self._list.selectionChanged.connect(self._sync_block_highlight)
        # 列表里一个都没选中 → 预览不画高亮、不画描边 (m01191 item 1)。
        self._list.selectionChanged.connect(self._on_list_selection_changed)
        # Del 等于点「删除」(m01566 item 1)。
        self._list.deletePressed.connect(self._on_delete_colours)

        # 「删除 / 合并 / 更换颜色 / 撤销 / 恢复」 sit in ONE row right under the
        # colour list, in that order: the first three all act on the selection
        # in the list, and the last two act on whatever those three did.
        #
        # None of them carries a hover tooltip any more: the black pop-up they
        # raised covered the very picture the user was working on.  What each
        # button does is said in the bottom line the moment it matters (a delete
        # announces that those pixels are left blank, and so on).
        # The key is written on the button because it is the only place that can
        # say it: a tooltip would cover the picture (see above), and a keyboard
        # shortcut nobody can see is a shortcut nobody uses (m01566 item 1).
        self._delete_button = QPushButton("删除")
        self._delete_button.clicked.connect(self._on_delete_colours)

        self._merge_button = QPushButton("合并")
        self._merge_button.clicked.connect(self._on_merge_colours)
        # 「合并点不动」是能一句话说清的事：Qt 的单选列表里按住 Ctrl 也选不中第二行，
        # 必须先开「多选」。把这句话挂在按钮上，省得用户以为按钮坏了。
        self._merge_tip = "合并要把两行一起选中：先点左边的「多选」，再点第二行"
        self._merge_button.setToolTip(self._merge_tip)

        self._replace_button = QPushButton("更换颜色")
        # Deliberately NOT accented, exactly like 删除 / 合并 / 撤销 / 恢复 next to
        # it: these five are all plain actions on the selection, and the only
        # blue button on this page is 生成 3MF, which is the one that finishes
        # the job.  A blue 更换颜色 read as "this is the important one".
        self._replace_button.clicked.connect(self._on_replace)

        self._multi_button = QPushButton("多选")
        self._multi_button.setCheckable(True)
        self._multi_button.toggled.connect(self._on_multi_toggled)

        self._undo_button = QPushButton("撤销")
        self._undo_button.clicked.connect(self._undo)

        self._redo_button = QPushButton("恢复")
        self._redo_button.clicked.connect(self._redo)

        actions = QHBoxLayout()
        actions.setSpacing(6)
        actions.addWidget(self._multi_button)
        actions.addWidget(self._delete_button)
        actions.addWidget(self._merge_button)
        actions.addWidget(self._replace_button)
        actions.addStretch(1)
        actions.addWidget(self._undo_button)
        actions.addWidget(self._redo_button)

        # The shortcuts are on the page, so Ctrl+Z works whichever of the two
        # panels has the focus.
        self._undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self._undo_shortcut.activated.connect(self._undo)
        self._redo_shortcut = QShortcut(QKeySequence("Ctrl+Y"), self)
        self._redo_shortcut.activated.connect(self._redo)
        # Ctrl+S 保存到现有工程（没有就问存哪儿），Ctrl+Shift+S 另存为一份。
        self._save_shortcut = QShortcut(QKeySequence.StandardKey.Save, self)
        self._save_shortcut.activated.connect(lambda: self._on_save_project())
        self._save_as_shortcut = QShortcut(QKeySequence("Ctrl+Shift+S"), self)
        self._save_as_shortcut.activated.connect(lambda: self._on_save_project(ask=True))
        # 1.6.54（m22211 第 7 条）：另外四个动作也绑上按键，键值在「设置 → 图像转换」里改。
        self._import_shortcut = QShortcut(QKeySequence("Ctrl+O"), self)
        self._import_shortcut.activated.connect(self._on_import)
        self._rematch_blocks_shortcut = QShortcut(QKeySequence("Ctrl+R"), self)
        self._rematch_blocks_shortcut.activated.connect(self._on_rematch_blocks)
        self._export_original_shortcut = QShortcut(QKeySequence("Ctrl+E"), self)
        self._export_original_shortcut.activated.connect(self._on_export_original)

        list_box = QWidget()
        list_layout = QVBoxLayout(list_box)
        list_layout.setContentsMargins(0, 0, 0, 0)
        list_layout.setSpacing(6)
        list_layout.addWidget(self._list, 1)
        list_layout.addLayout(actions)

        self._detail = ColourDetail()
        # 「选中的颜色」 used to be a panel of its own, permanently showing whichever
        # colour was last clicked.  The colour list above is where colours get
        # picked, so the recipe and the ΔE00 of one colour became a dialog the user
        # asks for (双击一行) instead of a third of the column that sits empty.
        self._detail_dialog: QDialog | None = None

        # 「耗材方案」: a match is not one answer, it is a few, and the user is the
        # one who knows whether a different set of rolls suits this print.  Each
        # row carries its own numbers — 平均 / 最差 ΔE00 and how many rolls — so the
        # comparison happens on screen instead of inside the program's head, which
        # is exactly what the old single-answer 「重新匹配」 hid.
        self._plans: list[BudgetOption] = []
        # 方案表里的数字是「上一次匹配」算出来的；删掉 / 合并 / 换掉颜色之后，图上的
        # 颜色就不再是那一份，行上的 差块 甚至会超过现在的颜色数（m11180 第 5 条）。
        # 这里只记「改过了」，重算交给「按当前色块重新匹配」——不自动重挑卷，是刻意
        # 保守：同一套方案是用户挑出来的，不能让程序背着他换（m01411 第 1 条）。
        self._plans_stale = False
        self._plan_key = ""
        self._plan_group = QButtonGroup(self)
        self._plan_group.setExclusive(True)
        self._plan_group.idClicked.connect(self._on_plan_clicked)

        # 筛选: 「必须有这几卷」 and 「这几卷不要用」 are facts about the printer (what is
        # already in the AMS, what ran out), not about the picture, so they are set
        # once and every match from then on obeys them.
        self._require: frozenset[str] = frozenset()
        self._avoid: frozenset[str] = frozenset()
        self._filter_button = QPushButton("筛选耗材")
        self._filter_button.clicked.connect(self._on_filter)
        self._filter_hint = QLabel("")
        self._filter_hint.setProperty("role", "hint")
        self._filter_hint.setWordWrap(True)
        self._update_filter_hint()

        self._plan_box = QWidget()
        self._plan_layout = QVBoxLayout(self._plan_box)
        self._plan_layout.setContentsMargins(0, 0, 0, 0)
        self._plan_layout.setSpacing(6)
        filter_row = QHBoxLayout()
        filter_row.setSpacing(6)
        filter_row.addWidget(self._filter_button)
        filter_row.addWidget(self._filter_hint, 1)
        self._plan_layout.addLayout(filter_row)

        # The 方案 rows are rebuilt on every match, so they live in a widget of
        # their own; the filter line, the hint and the roll list around them are
        # reused as they are.
        self._plan_rows = QWidget()
        self._plan_rows_layout = QVBoxLayout(self._plan_rows)
        self._plan_rows_layout.setContentsMargins(0, 0, 0, 0)
        self._plan_rows_layout.setSpacing(2)
        self._plan_layout.addWidget(self._plan_rows)

        self._plan_hint = QLabel(PLAN_HINT_TEXT)
        self._plan_hint.setProperty("role", "hint")
        self._plan_hint.setWordWrap(True)
        self._plan_layout.addWidget(self._plan_hint)

        # 「用到了哪些耗材」 is not a panel of its own any more: it answers the same
        # question the 方案 rows disagree about — 用哪几卷 — so it sits under them,
        # for the 方案 that is picked.  It no longer scrolls on its own: the whole
        # 耗材方案 panel does (see below), because seven 方案 plus their rolls do
        # not fit in the panel anyway.  Its sentence is written here rather than
        # left to the first match, so the two hints appear together (m00940 第 1 条).
        self._usage = QLabel(USAGE_HINT_TEXT)
        self._usage.setProperty("role", "hint")
        self._usage.setTextFormat(Qt.TextFormat.RichText)
        self._usage.setWordWrap(True)
        self._plan_layout.addWidget(self._usage)

        # 两句话、方案行和用量行都按内容高，多余的高度落到这块空白上（m00940 第 1
        # 条）：没有 stretch 时 480 高的面板把多出来的高度平均分给三个子部件，每句话
        # 在自己的槽里垂直居中，看上去像分散在三个盒子里，而不是面板开头挨着的两行。
        self._plan_layout.addStretch(1)
        self._refresh_plans()

        # m01885 item 3: 「打印图片需要的颜色的窗口要能自由向上拖拽上滑」.  A panel
        # whose height is its content cannot be shrunk — with seven 方案 rows its
        # minimum was most of the column, so the splitter handle would not move —
        # so the whole panel scrolls and its vertical size policy ignores the
        # content's minimum.  That is what lets the handle go all the way up (and
        # 耗材方案 all the way away) while both panels stay usable.
        self._plan_scroll = _scrollable(self._plan_box)
        self._plan_scroll.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Ignored
        )
        self._plan_scroll.setMinimumHeight(48)

        right = QSplitter(Qt.Orientation.Vertical)
        self._right_splitter = right
        # 耗材方案 comes first: it decides what the panel under it says, and it now
        # answers the printer's question (which rolls to load) as well.
        right.addWidget(self._wrap(self._plan_scroll, "耗材方案"))
        self._list_panel = self._wrap(list_box, LIST_TITLE_COLOUR)
        right.addWidget(self._list_panel)
        right.setStretchFactor(0, 2)
        right.setStretchFactor(1, 3)
        # 耗材方案 may be collapsed away entirely so the colour list gets the whole
        # column; the colour list never collapses, so it cannot be lost.
        right.setCollapsible(0, True)
        right.setCollapsible(1, False)
        right.setSizes([430, 350])

        pictures = QSplitter(Qt.Orientation.Horizontal)
        # 原图 and 预览 both zoom and pan with the same gesture; the hint is a
        # permanent line under the toolbar now, not a tooltip on the canvas
        # (which covered the picture the user was looking at).
        original_area = _scrollable(self._original_view)
        preview_area = _scrollable(self._view)
        #: 原图 面板：标题里会带上原图的像素尺寸（``_sync_source_title``），因为预览画的
        #: 是缩下来的匹配栅格（m14567）。
        self._original_panel = self._wrap(original_area, "原图", badge="原图已修改")
        pictures.addWidget(self._original_panel)
        pictures.addWidget(self._wrap(preview_area, "预览"))
        # 预览 is the one the user works in, so it gets the extra room.
        pictures.setStretchFactor(0, 1)
        pictures.setStretchFactor(1, 1)
        pictures.setSizes([340, 420])

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(pictures)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([760, 320])

        self._status = _StatusLabel("")
        self._status.setProperty("role", "hint")
        self._status.setWordWrap(True)
        self._status.owner = self

        # The progress bar sits at the end of the status line it belongs to
        # (m00940 第 2 条: 「在文本『正在匹配：…』的地方添加匹配进度条」).  It is
        # hidden until a match has been running for BUSY_DELAY_MS — a 50 ms match
        # should not flash it — and the step message goes into the status label
        # rather than into the bar, so there is one text surface and nothing
        # jumps when the bar appears.
        self._progress = QProgressBar()
        self._progress.setRange(0, PROGRESS_STEPS)
        self._progress.setTextVisible(False)
        self._progress.setFixedSize(260, 12)
        self._progress.setVisible(False)
        self._busy_timer = QTimer(self)
        self._busy_timer.setSingleShot(True)
        # 两档之间还有很长的活（量化、找色）：这条计时器让进度条朝下一档慢慢爬，
        # 停在 45% 不动比慢慢爬更像「卡住」（1.6.62，m28297 第 3 条）。
        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(90)
        self._progress_timer.timeout.connect(self._creep_progress)
        self._progress_goal = 0
        self._busy_timer.setInterval(BUSY_DELAY_MS)
        self._busy_timer.timeout.connect(self._show_progress)

        # 使用说明不在页面上：1.6.14 把它搬进「帮助」窗口（m01885 第 6 条），
        # 从这里读一份给帮助窗口和自动检查用。
        self._help = PAGE_HELP

        self._status_bar = QWidget()
        status_row = QHBoxLayout(self._status_bar)
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.setSpacing(10)
        status_row.addWidget(self._status, 1)
        status_row.addWidget(self._progress)
        # 没有话要说的时候整行收起来（m26286 第 2 条）：原来它始终占着一行，
        # 上下留白加起来就是用户说的「空着的那块」。
        self._status_bar.setVisible(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addLayout(top)
        layout.addWidget(splitter, 1)
        layout.addWidget(self._status_bar)
        layout.addLayout(bottom)
        # 1.6.53（m21616 第 8 条）：开机就按「设置」里的默认值摆好。
        self.apply_settings(load_settings())

    def help_text(self) -> str:
        """这一页怎么用，「帮助」窗口里的头一节（m01885 第 6 条）。"""
        return self._help

    def apply_settings(self, values: dict) -> None:
        """把「设置」里图像转换页的默认值灌进来（1.6.53，m21616 第 8 条）。

        只改控件、不重算：正在看的那张图不该因为保存了一次设置就被重新配一遍。
        """
        for box, key in (
            (self._colour_count, "picture_colours"),
            (self._spool_cap, "picture_spools"),
        ):
            if key in values:
                box.blockSignals(True)
                box.setValue(int(values[key]))
                box.blockSignals(False)
        if "picture_dither" in values:
            self._dither.blockSignals(True)
            self._dither.setChecked(bool(values["picture_dither"]))
            self._dither.blockSignals(False)
        if "picture_base_mm" in values:
            self._base_spin.blockSignals(True)
            self._base_spin.setValue(float(values["picture_base_mm"]))
            self._base_spin.blockSignals(False)
        only = bool(values.get("picture_only_spools", self._only_spools.isChecked()))
        if only != self._only_spools.isChecked():
            self._only_spools.blockSignals(True)
            self._only_spools.setChecked(only)
            self._only_spools.blockSignals(False)
            self._on_only_spools(only)
        # 1.6.60（m27359 第 3 条）/ 1.6.61（m27752 第 1、2、6 条）：实验性开关按页分开记。
        # 勾上就把对应的数字框灰掉，它这一版不参与计算（值还留着，取消勾选就恢复）。
        self._no_colour_limit = bool(
            values.get("picture_no_colour_limit", self._no_colour_limit)
        )
        self._no_spool_limit = bool(
            values.get("picture_no_spool_limit", self._no_spool_limit)
        )
        self._colour_count.setEnabled(not self._no_colour_limit)
        self._spool_cap.setEnabled(not self._no_spool_limit)
        # 1.6.54（m22211 第 7 条）：这一页自己的四个快捷键。
        for editor, key, fallback in (
            (self._import_shortcut, "shortcut_picture_import", "Ctrl+O"),
            (self._rematch_blocks_shortcut, "shortcut_picture_rematch", "Ctrl+R"),
            (self._save_shortcut, "shortcut_picture_save_project", "Ctrl+S"),
            (self._export_original_shortcut, "shortcut_picture_export_png", "Ctrl+E"),
        ):
            sequence = str(values.get(key) or fallback)
            # QShortcut 没有 setKeySequence，q6 里换键要重新 setKey。
            editor.setKey(QKeySequence(sequence))

    def _wrap(self, widget: QWidget, title: str, badge: str | None = None) -> QWidget:
        frame = QFrame()
        frame.setObjectName("panel")
        header = QLabel(title)
        header.setProperty("role", "sectionTitle")
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(8)
        head.addWidget(header)
        if badge is not None:
            # 「原图已修改」 sits next to the 原图 title and only shows once the picture
            # itself has been changed by 删除/合并 (1.6.19, m01566 item 2).
            self._original_badge = QLabel(badge)
            self._original_badge.setProperty("role", "hint")
            self._original_badge.setVisible(False)
            head.addWidget(self._original_badge)
        head.addStretch(1)
        box = QVBoxLayout(frame)
        box.setContentsMargins(10, 8, 10, 10)
        box.setSpacing(8)
        box.addLayout(head)
        box.addWidget(widget, 1)
        # 标题要能换字（m02316 item 5）：颜色列的标题随「点图片时」在两种说法之间切。
        frame.title_label = header  # type: ignore[attr-defined]
        return frame

    def _set_controls_enabled(self, enabled: bool) -> None:
        # 「只匹配耗材」不在这个表里（1.6.53，m21616 第 1 条）：它是一句「这次用哪
        # 张候选表」的偏好，空图时也该点得动，勾上以后再导图片就按它算。
        for widget in (
            self._rematch_blocks_button,
            self._filter_button,
            self._save_project_button,
            self._base_spin,
            self._base_button,
            self._side_combo,
            self._smooth_box,
            self._export_3mf,
            self._export_obj,
        ):
            widget.setEnabled(enabled)

    # -- library ----------------------------------------------------------
    def setLibrary(self, library: FilamentLibrary, catalog: MixCatalog | None) -> None:
        """Called whenever the filament library or the mix catalogue changes."""
        self._library = library
        self._catalog = catalog
        self._candidates = None
        self._candidates_plain = None
        self._triple_palette = None
        self._triple_head = None
        if self._image is None:
            return
        if not self.isVisible():
            # The 颜色配方 tab is showing, so nobody can see the picture.  A
            # re-match against 66,461 candidates costs ~0.7 s and would be spent
            # on a result that is not on screen; remember that it is owed and
            # do it when the tab comes back.
            self._stale = True
            return
        self._on_rematch()

    def setTriples(self, triples, library: FilamentLibrary | None = None) -> None:
        """Hand over the three-filament catalogue once it exists.

        ``None`` means 「not ready yet」 (an unfinished catalogue has no ``packed``
        array), and the page then matches against 耗材本色 + 两色混色 only.  The
        old palette is dropped either way, because a three-filament colour is
        only valid for the spool set it was computed from.
        """
        self._triples = triples
        self._triple_palette = None
        self._triple_head = None
        if library is not None:
            self._library = library

    def showEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        super().showEvent(event)
        if self._stale:
            self._stale = False
            self._on_rematch()

    @property
    def result(self) -> MatchResult | None:
        return self._result

    @property
    def plate(self) -> PlateModel | None:
        return self._plate

    @property
    def library(self) -> FilamentLibrary | None:
        return self._library

    @property
    def image_name(self) -> str:
        """The imported picture's file name, used for the exported model."""
        return self._image_name

    # -- matching ---------------------------------------------------------
    def loadImage(self, source) -> None:
        """Load ``source`` (path or PIL image) and match it straight away.

        矢量图（SVG）先弹窗问「渲成多少像素」（1.6.49）；位图按它自己的像素导入。
        """
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.suffix.lower() in SVG_SUFFIXES:
                # 矢量图先渲染成位图（m13907 item 2）。渲成多少像素由用户在弹窗里定
                # （1.6.49，m20069）：矢量没有分辨率，渲得大出来的是真细节，渲得小细笔画
                # 就只剩一两像素宽（见 SvgSizeDialog 那张猪的实测）。弹窗的默认值是它在
                # 文件里声明的那一档，所以直接回车就是 1.6.48 的行为。之后整条流程
                # （匹配、删除/合并、导出）跟位图导入完全一样。
                declared = _vector_declared_size(path)
                chosen = _ask_vector_size(self, path, declared)
                if chosen is None:
                    return  # 用户取消：这张图不导入，原来那张原图一个像素都不动
                image = _load_vector(path, 1, size=chosen)
            else:
                image = Image.open(source)
            self._image_name = path.stem or "混色底板"
        else:
            image = source
        self._source_size = (int(image.width), int(image.height))
        # 位图一个像素都不插值（m19452 实测的结论）：240 px 的图插到 512，第一行从平均
        # ΔE00 4.82 掉到 7.00（最差 7.46→13.35、差块 4→6、颜色 7→9）—— 插出来的过渡色
        # 让四卷耗材去追更多颜色。所以位图按它自己的像素导入，界面上没有能写出更大数字的
        # 地方（m19523 撤掉了那两个像素框）；矢量图不一样，1.6.49 起在导入时弹窗问一次
        # 「渲成多少像素」（m20069），因为矢量放大是真细节，不是插值。
        # 位图按它自己的像素导入（m15721 第 2 条，m16941 第 1 条）：从前这里固定缩到最长边
        # 512，于是 2363 px 的照片一进来就只剩 512，细笔画全糊进格子里；后来改成一个要用户
        # 自己调的「最大边长」框，现在连框也没有了 —— 一格 0.4 mm，图多大板就多大。
        self._span = max(int(image.width), int(image.height))
        self._image = image.convert("RGBA")
        # A freshly imported picture has not been edited yet (1.6.19).
        self._original_edited = False
        self._refresh_source()
        # A freshly imported picture is not the project that was open before it,
        # so Ctrl+S has to ask where this one goes.
        self._project_path = None
        self._dirty = False
        # Nor is it the picture the match and the edits behind Ctrl+Z belonged to:
        # undoing past an import would paint the old match over this one.
        self._base = None
        self._plans = []
        self._plans_stale = False
        self._plan_key = None
        self._deleted = set()
        self._merged = {}
        self._replaced = {}
        self._block_edits = []
        self._block_template = None
        self._undo_stack = []
        self._redo_stack = []
        self._on_rematch()

    def _refresh_source(self) -> None:
        """Re-draw 原图/预览's backdrop from ``self._image`` and show the badge.

        Called whenever the picture itself changes (删除/合并, undo, 打开工程): the
        two views are handed ``self._source_qimage`` by ``_apply_edits``, so it has
        to be the current picture, holes and merged colours included.
        """
        self._source_qimage = _to_qimage(self._image) if self._image is not None else None
        if self._original_badge is not None:
            self._original_badge.setVisible(self._original_edited)
        self._sync_source_title()

    def _picture_colour(self, mask: np.ndarray) -> tuple[int, int, int] | None:
        """The colour this patch of 「原图」 already has, or ``None`` if it has none.

        合并 用得到它 (m01191 item 2)。用户把这条规矩讲得很清楚：原图色块 a 对应预览
        色块 b、原图色块 c 对应预览色块 d，b 和 d 合并之后 d 成了 b 的一部分，那么原图
        里 c 也并进 a —— 用的是 **a 自己本来的颜色**，不是预览里那卷耗材印出来的色。
        所以原图的颜色集合一点都不变，变的只有色块大小。

        取的是这一片像素里出现最多的那个不透明颜色（抗锯齿的边和已经删掉的透明像素都
        不算数）：大片平涂的色块，最多的那个就是它的本色，而按平均值取会在小数上挪动
        一位，并完之后原图里就会多出一种差一级的近色——那正是这条要求在防的事。
        """
        if self._image is None:
            return None
        box = _to_image_grid(np.asarray(mask, dtype=bool), self._image.size)
        if box.shape != (self._image.height, self._image.width) or not box.any():
            return None
        pixels = np.asarray(self._image.convert("RGBA"))[box]
        opaque = pixels[pixels[:, 3] > 0][:, :3]
        if not opaque.size:
            return None
        values, counts = np.unique(opaque, axis=0, return_counts=True)
        red, green, blue = values[int(np.argmax(counts))]
        return (int(red), int(green), int(blue))

    def _paint_original(
        self, mask: np.ndarray, colour: tuple[int, int, int] | None
    ) -> None:
        """Write a 删除/合并 edit into the picture itself (1.6.19, m01566 item 2).

        ``colour is None`` means 删除: those pixels become transparent, so they are
        blank in 原图 as well and the next match leaves them unprinted.
        Otherwise 合并: the pixels take ``colour``, so 原图 really shows one block of
        one colour and matching the modified picture finds that colour again.

        1.6.19 只给色块级的删除／合并用；1.6.20 起按颜色的删除／合并也走这里（m02316
        item 2）——两者都是「这几片地方以后按这个来印」，改的既然是同一张图、同一批
        像素，就应该是同一件事，区别只在于掩码是「选中的块」还是「选中的那几种颜色」。

        ``colour`` 由调用方用 ``_picture_colour`` 从存活下来的那一块上量出来（m01191
        item 2）；1.6.20 及以前这里收的是一张 ``PaletteEntry``，写进去的是**匹配出来的
        打印色**，于是原图会凭空多出一种它本来没有的颜色。

        This is deliberately separate from ``_block_edits``: that overlay is what
        预览 shows right now, this is what 匹配 starts from later.
        """
        if self._image is None or not mask.any():
            return
        # The mask lives on the match grid; the picture is full size (see
        # ``_to_image_grid``).  Painting the grid's pixels as-is would hit the
        # wrong part of a picture that was downscaled for matching.
        mask = _to_image_grid(np.asarray(mask, dtype=bool), self._image.size)
        if mask.shape != (self._image.height, self._image.width):
            return
        pixels = np.asarray(self._image.convert("RGBA")).copy()
        if colour is None:
            pixels[..., 3][mask] = 0
        else:
            red, green, blue = (int(value) for value in colour)
            pixels[mask] = (red, green, blue, 255)
        self._image = Image.fromarray(pixels)
        self._original_edited = True
        self._refresh_source()

    def _on_import(self) -> None:
        start = str(paths.projects_dir())
        name, _ = QFileDialog.getOpenFileName(self, "选择图片", start, IMAGE_FILTER)
        if not name:
            return
        try:
            self.loadImage(Path(name))
        except Exception as error:  # pragma: no cover - the dialog is interactive
            QMessageBox.warning(self, "无法读取图片", str(error))

    def _on_export_original(self) -> None:
        """把改过的原图存成 PNG（m12426 item 3）。

        删除 / 合并 / 更换颜色都会**同时改原图**（m02316 item 2），可改完的这张图一直
        只活在工程文件里 —— 用户要拿去继续修图、发给别人，就得有个直接存图的出口。
        存的就是 ``self._image``：和「原图」面板里画的是同一张，被删掉的地方是透明的
        （「留空不打印」），alpha 原样保留。
        """
        if self._image is None:
            self._status.setText("先导入一张图片，再导出原图。")
            return
        suggested = f"{self._image_name}-原图.png"
        start = str(
            (self._project_path.with_name(suggested))
            if self._project_path
            else (paths.exports_dir() / suggested)
        )
        name, _ = QFileDialog.getSaveFileName(self, "导出原图", start, PNG_FILTER)
        if not name:
            return
        target = Path(name)
        if target.suffix.lower() != ".png":
            target = target.with_suffix(".png")
        try:
            # 先写临时文件再改名（1.6.50）：用户很可能就选了上次导出的那一张，直接
            # 写会在中途出错时把原来那份弄成半张 PNG。
            with atomic_path(target) as temporary:
                self._image.save(temporary, format="PNG")
        except Exception as error:  # pragma: no cover - the dialog is interactive
            QMessageBox.warning(self, "导出原图失败", str(error))
            return
        self._status.setText(
            f"已导出原图 {target.name}（{target.parent}）：这张图里已经带着删除和合并的结果，"
            "删掉的地方是透明的。"
        )

    # -- 导出 SVG：把位图描成矢量 ------------------------------------------
    def _on_export_svg(self) -> None:
        """把这张图描成 SVG（1.6.64，m29374）。

        描的是 ``self._image`` —— 跟「导出原图」同一张，也就是**已经带着删除/合并结果**
        的那张。用户删掉的色块在矢量里就该是空的，不能绕回去描那张没改过的原图。
        """
        if self._image is None:
            self._status.setText("先导入一张图片，再导出 SVG。")
            return

        dialog = SvgExportDialog(self, name=self._image_name)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        options = dialog.values()

        suggested = f"{self._image_name}-矢量.svg"
        start = str(
            (self._project_path.with_name(suggested))
            if self._project_path
            else (paths.exports_dir() / suggested)
        )
        name, _ = QFileDialog.getSaveFileName(self, "导出 SVG", start, "SVG 矢量图 (*.svg);;所有文件 (*)")
        if not name:
            return
        target = Path(name)
        if target.suffix.lower() != ".svg":
            target = target.with_suffix(".svg")

        # 描摹库吃的是文件路径，所以先把手上这张 PIL 图落到临时文件；用完就删。
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self._status.setText("正在把这张图描成矢量…")
        QApplication.processEvents()
        try:
            with tempfile.TemporaryDirectory(prefix="fcs-svg-") as folder:
                staged = Path(folder) / "source.png"
                self._image.save(staged, format="PNG")
                result = image_to_svg(
                    staged,
                    target,
                    colours=int(options["colours"]),
                    speckle=int(options["speckle"]),
                    smooth=float(options["smooth"]),
                )
        except VectorizeUnavailable as error:
            QApplication.restoreOverrideCursor()
            self._status.setText("导出 SVG 不可用：这台机器没装 vtracer。")
            QMessageBox.information(self, "导出 SVG 不可用", str(error))
            return
        except Exception as error:  # pragma: no cover - the dialog is interactive
            QApplication.restoreOverrideCursor()
            self._status.setText("导出 SVG 失败。")
            QMessageBox.warning(self, "导出 SVG 失败", str(error))
            return
        QApplication.restoreOverrideCursor()

        self._status.setText(
            f"已导出矢量 {target.name}（{target.parent}）：{result.paths} 条路径、"
            f"{result.colours} 种颜色、{result.bytes // 1024} KB。放大不糊，能进 Blender 继续改。"
        )

    # -- 工程：存下来，下次接着改 -----------------------------------------
    def _on_open_project(self) -> None:
        """Open a ``.bpproj`` and put the page back the way it was saved."""
        if self._dirty and not self._confirm_discard("打开另一个工程"):
            return
        start = str(self._project_path or paths.exports_dir())
        name, _ = QFileDialog.getOpenFileName(self, "打开工程", start, PROJECT_FILTER)
        if not name:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self._status.setText("正在打开工程…")
        QApplication.processEvents()
        try:
            state = load_project(Path(name), library=self._library)
        except ProjectError as error:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "打开工程失败", str(error))
            return
        except Exception as error:  # pragma: no cover - defensive
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "打开工程失败", str(error))
            return
        QApplication.restoreOverrideCursor()
        self._adopt_project(state, Path(name))

    def _adopt_project(self, state: ProjectState, path: Path) -> None:
        """Make ``state`` the page's picture, match and edit state."""
        self._image = state.image.convert("RGBA")
        self._image_name = state.image_name or "混色底板"
        # The file's picture IS the one the user left, edits and all (1.6.19): the
        # flag only says whether it had been changed when it was saved.
        self._original_edited = bool(state.original_edited)
        self._refresh_source()
        self._base = state.base
        self._deleted = set(state.deleted)
        self._merged = dict(state.merged)
        self._replaced = dict(state.replaced)
        self._undo_stack = []
        self._redo_stack = []
        self._plate = None
        self._project_path = path
        # The controls go back to the settings the file was matched with, so
        # 「重新匹配」 reproduces the picture the user saved rather than quietly
        # re-running it against whatever this window happened to be set to.
        settings = state.base.settings
        self._colour_count.setValue(int(settings.max_colours))
        # 工程里记着当时用的栅格（m16941 第 1 条之后界面上没有这个输入框了），照它开，
        # 存过的图才算一模一样地回来。
        self._span = int(settings.max_dimension)
        self._dither.setChecked(bool(settings.dither))
        self._spool_cap.setValue(int(settings.max_spools))
        self._base_spin.setValue(float(state.plate.base_thickness_mm))
        self._smooth_box.setChecked(bool(state.plate.smooth_edges))
        self._side_combo.setCurrentIndex(0 if state.plate.image_on_bottom else 1)
        # The 方案 list belongs to a match, and 「打开工程」 deliberately does not run
        # one: the colour table in the file is what the user saved, edits and all,
        # and re-matching could answer something slightly different.  So the panel
        # starts with the single row this file came with — named for the 方案 it was
        # saved under — and 1.6.20 fills the other 搭法 in behind it below
        # (``_fill_open_plans``, m02316 item 7).
        saved = BudgetOption(
            key=state.plan,
            title=plan_caption(state.plan)[0],
            note=plan_caption(state.plan)[1],
            result=state.base,
            from_file=True,
        )
        self._plans = [saved]
        # 文件里的方案是照文件里那张图算的，两边一致，所以不算「改过了」。
        self._plans_stale = False
        self._plan_key = state.plan
        # 色块级的改动（哪几块被删掉 / 合并 / 换成了什么）跟删除、合并、更换一起存，
        # 打开工程要把它们摆回同样的顺序，预览才是保存时的那张 (m02366 item 3).
        #
        # 1.6.26：文件里每一步是一个字典 ``{"mask": …, "entry": …}``（见
        # ``project._paints_from_json``），内存里的 ``ProjectState.block_paints``
        # 也是这么放的。原先这里写的是 ``for mask, entry in state.block_paints``，
        # 解包的其实是字典自己的两个键名，于是 ``entry`` 变成字符串 ``"entry"``、
        # ``mask`` 变成字符串 ``"mask"``：打开带色块改动的工程时毫无反应（
        # ``np.asarray("mask", dtype=bool)`` 只是一格 True），一直撑到保存工程时
        # ``entry_to_json("entry")`` 才炸成「'str' object has no attribute 'key'」
        # ——用户看到的就是「打开工程好好的，一保存就失败」。两种形状都认。
        self._block_edits = []
        # 工程文件里的块来自保存时那一份划分；重新分色之后块要重新认，别拿旧标签图
        # 硬贴（m15721 第 1 条只对「按当前色块重新匹配」一路留划分）。
        self._block_template = None
        for paint in state.block_paints:
            mask, entry = (
                (paint.get("mask"), paint.get("entry"))
                if isinstance(paint, dict)
                else (paint[0], paint[1])
            )
            self._block_edits.append(
                BlockPaint(mask=np.asarray(mask, dtype=bool), entry=entry)
            )
        self._refresh_plans()
        self._stale = False
        self._apply_edits()
        # 底板颜色（m16941 第 2 条）：工程里记的是那一种的 key，从整套混色里找回来；找不到
        # （耗材库换过、混色配方没了）就退回跟着图上最大的一块走，而不是随便挑一个。
        self._restore_base_entry(str(getattr(state.plate, "base_key", "") or ""))
        self._set_controls_enabled(True)
        edits = []
        if self._deleted:
            edits.append(f"删除 {len(self._deleted)} 种")
        if self._merged:
            edits.append(f"合并 {len(self._merged)} 种")
        if self._replaced:
            edits.append(f"更换 {len(self._replaced)} 种")
        if self._block_edits:
            edits.append(f"色块改动 {len(self._block_edits)} 处")
        tail = f"；里面记着{'、'.join(edits)}" if edits else ""
        message = (
            f"已打开工程 {path.name}"
            + (f"（保存于 {state.saved_at}）" if state.saved_at else "")
            + f"：{self._image_name} {state.image.width}×{state.image.height} 像素，"
            f"现在打 {len(self._result.palette) if self._result else 0} 种颜色{tail}。"
            "接着改就行，Ctrl+S 保存。"
        )
        # 文件里只存了一套方案，别的搭法在这里补上（m02316 item 7）：用的就是「按当前
        # 色块重新匹配」那条路，所以列出来的和按那个按钮得到的一样。
        self._fill_open_plans(saved)
        if len(self._plans) > 1:
            self._status.setText(
                message + f"顺便把另外 {len(self._plans) - 1} 套耗材搭配也列出来了。"
            )
        else:
            # 补方案会一路改状态栏（正在匹配…），所以这句话最后再说一次。
            self._status.setText(message)
        self._dirty = False

    def _on_save_project(self, ask: bool = False) -> None:
        """Write the project file — the original picture plus every edit so far.

        ``ask`` (Ctrl+Shift+S) always opens the dialog; otherwise the file the
        project came from is overwritten, which is what Ctrl+S means everywhere
        else, and the dialog appears only when there is no such file yet.
        """
        if self._base is None or self._result is None or self._image is None:
            self._status.setText("先导入并匹配一张图片，再保存工程。")
            return
        target = None if ask else self._project_path
        if target is None:
            suggested = f"{self._image_name}{PROJECT_SUFFIX}"
            # 存在「下载」里，和导出的模型一样：存进程序自己的数据目录等于让人找
            # 不到——工程文件是要拿去接着改、发给别人的东西。
            start = str(
                (self._project_path.with_name(suggested))
                if self._project_path
                else (paths.exports_dir() / suggested)
            )
            name, _ = QFileDialog.getSaveFileName(self, "保存工程", start, PROJECT_FILTER)
            if not name:
                return
            target = Path(name)
        engine = self._catalog.engine.id if self._catalog is not None else DEFAULT_ENGINE
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self._status.setText("正在保存工程…")
        QApplication.processEvents()
        try:
            written = save_project(
                target,
                image=self._image,
                image_name=self._image_name,
                base=self._base,
                live=self._result,
                deleted=self._deleted,
                merged=self._merged,
                replaced=self._replaced,
                engine=engine,
                plate=self._plate_settings(),
                library=self._library,
                version=__version__,
                preview=render_preview(
                self._result, smooth=bool(self._smooth_box.isChecked())
            ),
                plan=self._plan_key,
                # 色块级的删除 / 合并 / 更换颜色 (m02366 item 3)：整块一起改，改的是像素
                # 而不是「哪一种颜色」，所以文件里存的是掩码，见 save_project.
                block_paints=[(edit.mask, edit.entry) for edit in self._block_edits],
                # 这张原图已经被删除/合并改过（1.6.19），重开时照旧显示「原图已修改」。
                original_edited=self._original_edited,
            )
        except Exception as error:
            QApplication.restoreOverrideCursor()
            log = self._log_save_failure(error)
            detail = f"\n\n详细调用栈已写到 {log}" if log is not None else ""
            QMessageBox.warning(self, "保存工程失败", f"{error}{detail}")
            return
        QApplication.restoreOverrideCursor()
        self._project_path = written
        self._dirty = False
        spools = len(spool_usage(self._result, self._library))
        blocks = (
            f" + {len(self._block_edits)} 处色块改动" if self._block_edits else ""
        )
        self._status.setText(
            f"已保存工程 {written.name}（{written.parent}）："
            f"原图 + 预览 + {len(self._result.palette)} 种颜色 + {spools} 卷耗材"
            f"{blocks}。下次「打开工程」就回到这一步。"
        )

    def _log_save_failure(self, error: BaseException) -> Path | None:
        """Write the failure and its stack next to the user's data, and say where.

        The message box alone told nobody anything actionable: the traceback lived
        in a dialog that vanishes with the click, which is why 「还能保存失败？」
        was unanswerable the first time it happened (1.6.26).  The file is appended
        to, not overwritten, so a repeat keeps the earlier evidence.
        """
        try:
            folder = paths.data_dir() / "logs"
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / "保存失败.log"
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            detail = "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            )
            if error.__traceback__ is None:
                # 没有栈的异常（在别处造好再转手过来的）也要留下位置：日志里只剩
                # 一句话的话，下次还是查不出是在哪儿报的。
                detail += "".join(traceback.format_stack())
            with target.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"[{stamp}] 保存工程失败：{error}\n"
                    # 用传进来的这个异常对象，不用 traceback.format_exc()：后者只认
                    # 「正在处理的那一个」，从 except 外面调它就会写成 NoneType: None。
                    f"{detail}\n"
                )
            return target
        except Exception:  # pragma: no cover - 记日志再失败也不能盖掉原来的错
            return None

    def _confirm_discard(self, action: str) -> bool:
        """Ask before edits that are not in the project file yet are lost."""
        box = QMessageBox(self)
        box.setWindowTitle("还有没保存的改动")
        box.setIcon(QMessageBox.Icon.Warning)
        box.setText(f"这张图改过、还没存进工程。要直接{action}吗？")
        box.setInformativeText("没保存的删除 / 合并 / 更换颜色都会丢掉。想留着就先 Ctrl+S。")
        keep = box.addButton("先保存", QMessageBox.ButtonRole.AcceptRole)
        drop = box.addButton("不保存", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(keep)
        box.exec()
        clicked = box.clickedButton()
        if clicked is keep:
            self._on_save_project()
            return not self._dirty
        return clicked is drop

    def _settings(self) -> MatchSettings:
        return MatchSettings(
            max_colours=self._colour_limit(),
            max_dimension=int(self._span),
            dither=bool(self._dither.isChecked()),
            max_spools=self._spool_limit(),
            no_colour_limit=bool(self._no_colour_limit),
            no_spool_limit=bool(self._no_spool_limit),
        ).clamped()

    def _colour_limit(self) -> int:
        """这一版匹配最多分几种颜色（1.6.60，m27359 第 3 条 / 1.6.61，m27752 第 1 条）。

        勾了「实验性：没有颜色上限」就交一个「比任何图都多」的数，真正的上限由量化那一步
        按这张图**实际有几种颜色**决定（簇数不会超过它）；「颜色数量」那个框留着显示原来
        的值，只是灰掉不参与。
        """
        if self._no_colour_limit:
            return UNLIMITED_COLOURS
        return int(self._colour_count.value())

    def _spool_limit(self) -> int:
        """这一版匹配最多用几卷（1.6.61，m27752 第 6 条）。0 = 不限。"""
        if self._no_spool_limit:
            return 0
        return int(self._spool_cap.value())

    def _candidate_palette(self) -> list[PaletteEntry]:
        """The full 全部颜色 list, built once per library.

        At 41 spools this is 66,461 ``PaletteEntry`` objects and ~0.5 s to
        build.  It is needed both by the match and by the 「更换颜色」 menu, and
        that menu is opened once per colour the user swaps — so rebuilding it
        every time was pure waste.  ``setLibrary`` drops the cache.

        Mixes are always in it.  There used to be a 「混色也算候选」 box; a spool
        library without its 81-ratio mixes is not a smaller answer, it is half
        the palette, so that choice was never a real one.
        """
        if self._only_spools.isChecked():
            if self._candidates_plain is None:
                self._candidates_plain = build_palette(
                    self._library, self._catalog, include_mixes=False
                )
            return self._candidates_plain
        if self._candidates is None:
            self._candidates = build_palette(self._library, self._catalog, include_mixes=True)
        return self._candidates

    def _on_only_spools(self, checked: bool) -> None:
        """勾上 / 取消「只匹配耗材」：候选少了，手上那套搭配就得重算。"""
        # 两张缓存都作废（1.6.54，m22211 第 4 条）：勾着时只读了 `_candidates_plain`，
        # 取消勾选时那张混色表也可能是在别的库里建的，一律重来一次更省心。
        self._candidates = None
        self._candidates_plain = None
        self._triple_palette = None
        self._triple_head = None
        if self._image is not None:
            self._on_rematch()

    def _match_palette(self) -> Sequence[PaletteEntry]:
        """What the matcher picks from: 耗材本色 + 两色 + 三色混色.

        Three-filament colours are handed in by the main window once the table
        behind them is finished, and they always take part — the box that used
        to gate them is gone.  Four million colours cannot be built as objects,
        so the triple catalogue contributes a lazy tail: only the few the
        picture actually keeps ever become one.  The head still wins its ties,
        which is why the tail skips colours the head already offers.

        The cache is keyed on the head *list* we handed over, not on
        ``pool.head``: ``TriplePalette`` copies what it is given, so comparing
        against its own copy would rebuild the tail — a four-million-row
        membership test — on every single match.
        """
        head = self._candidate_palette()
        if self._only_spools.isChecked():
            # 1.6.54（m22211 第 4 条）：勾着就一个混色都不许进匹配器。正常情况下
            # `_candidate_palette()` 已经只给本色了，这里再滤一遍是兜底 —— 缓存或
            # 撤销栈里漏进来的旧候选，不能让它把「只匹配耗材」变成一个摆设。
            return [entry for entry in head if entry.kind == "filament"]
        if self._triples is None:
            return head
        if self._triple_palette is None or self._triple_head is not head:
            self._triple_palette = TriplePalette(head, self._triples, self._library)
            self._triple_head = head
        return self._triple_palette

    def _begin_match(self) -> None:
        """Arm the progress bar; it appears only if the match is slow enough."""
        self._progress.setValue(0)
        self._progress_goal = 0
        self._busy_timer.start()
        self._set_controls_enabled(False)

    def _end_match(self) -> None:
        """Put the bar away.  The controls come back with the new result."""
        self._busy_timer.stop()
        self._progress_timer.stop()
        self._progress_goal = 0
        self._progress.setVisible(False)
        self._progress.setValue(0)
        self._sync_status_bar()

    def _show_progress(self) -> None:
        self._progress.setVisible(True)
        self._sync_status_bar()

    def _sync_status_bar(self) -> None:
        """状态行只在有话要说、或者进度条正显示的时候占地方（m26286 第 2 条）。"""
        bar = getattr(self, "_status_bar", None)
        if bar is None:
            return
        bar.setVisible(bool(self._status.text().strip()) or self._progress.isVisible())

    def _creep_progress(self) -> None:
        """朝 ``_progress_goal`` 慢慢挪，永不越过它（1.6.62，m28297 第 3 条）。

        引擎的分数是「第几步」，两档之间的量化、找色各自要好几秒。原地不动看起来就是
        死机；按固定节奏往下一档挪，既不动摇「还没做完」这个事实，也让用户看见它在走。
        """
        goal = int(self._progress_goal)
        value = int(self._progress.value())
        if value >= goal or not self._progress.isVisible():
            return
        step = max(1, (goal - value) // 12)
        self._progress.setValue(min(goal, value + step))

    def _report_match(self, message: str, fraction: float) -> None:
        """One step of :func:`match_image`, drawn while the match is running.

        ``processEvents`` is what makes the bar move at all — the match is a
        plain synchronous call — and it excludes user input so a click cannot
        start a second match inside the first one, which is the only way this
        can go wrong.
        """
        shown = _match_progress_fraction(fraction)
        self._progress_goal = int(max(0.0, min(1.0, shown)) * PROGRESS_STEPS)
        self._progress.setValue(self._progress_goal)
        if self._progress_goal < PROGRESS_STEPS:
            self._progress_timer.start()
        else:
            self._progress_timer.stop()
        self._status.setText(f"正在匹配：{message}…")
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

    def _on_rematch(self, *, pushed: bool = False) -> None:
        if self._image is None:
            return
        if self._library is None or len(self._library) == 0:
            self._status.setText("请先在「颜色配方」页添加耗材，再匹配图片。")
            return
        palette = self._match_palette()
        if not palette:
            self._status.setText("调色板是空的，请先添加耗材。")
            return
        # The rows on screen describe the match that is about to be replaced:
        # clearing them first is what keeps 「耗材方案」 from showing a choice that
        # is no longer the one being made.  The snapshot has to be taken **first**
        # (m18338 第 1 条): 清空之后再压栈，Ctrl+Z 拿回来的就是一份没有方案的匹配。
        before = list(self._plans)
        if self._base is not None and not pushed:
            self._push()
            pushed = True
        self._plans = []
        self._plans_stale = False
        self._refresh_plans("正在重新匹配，马上列出几套耗材搭配。")
        self._begin_match()
        try:
            options = match_options(
                self._image,
                palette,
                self._settings(),
                require=self._require,
                avoid=self._avoid,
                progress=self._report_match,
            )
        except Exception as error:  # pragma: no cover - defensive
            QMessageBox.warning(self, "匹配失败", str(error))
            self._drop_failed_match(before, pushed)
            # The controls were switched off for the match, and this path has no
            # new result to switch them back on with.
            self._set_controls_enabled(self._result is not None)
            return
        finally:
            self._end_match()
        if not options:
            QMessageBox.warning(
                self,
                "匹配失败",
                "按现在的筛选条件没有找到能用的耗材搭配，先放宽筛选或者提高耗材上限。"
                if (self._require or self._avoid)
                else "这一版没有找到能用的耗材搭配。",
            )
            self._drop_failed_match(before, pushed)
            self._set_controls_enabled(self._result is not None)
            return
        # 1.6.27 起第一行是 「大色差色块优先」 (m06704 第 3 条), which is also what
        # 「重新匹配」 uses, so the plain button lands on the row the user asked to be
        # the answer and the other rows stay the alternatives.
        # 从头匹配的这份划分是新的：不许再拿「按当前色块重新匹配」留下的那张标签图。
        self._block_template = None
        self._adopt_options(options, pushed=pushed)

    def _drop_failed_match(self, before: list[BudgetOption], pushed: bool) -> None:
        """一次没成功的匹配不该留下痕迹：行摆回去，白压的那一步也撤掉。"""
        if pushed and self._undo_stack:
            snapshot = self._undo_stack.pop()
            # 快照末位就是 _span（m18338 第 1 条）：这一步没成功，栅格也得跟着退回按下
            # 之前，不然下一次匹配会按新像素数重划色块。
            self._span = int(snapshot[-1])
        self._plans = before
        self._refresh_plans()

    def _original_means(self, result: MatchResult) -> np.ndarray | None:
        """每种 live 颜色在当前「原图」上的平均色（``None`` = 量不了，仍用旧的）。

        只在原图被 删除/合并 改过之后才需要：那时 ``result.means`` 描述的是一张
        已经不存在的图。
        """
        if self._image is None:
            return None
        pixels = np.asarray(self._image.convert("RGB"))
        # ``result.indices`` is the match grid, the picture is full size.
        labels = _to_image_grid(np.asarray(result.indices), self._image.size)
        if pixels.shape[:2] != labels.shape:
            return None
        return _pixel_means(pixels, labels, len(result.palette))

    def _rematch_source(self) -> tuple[MatchResult, dict[int, PaletteEntry]]:
        """（拿着当前色块去重新挑卷的输入, 用户手改过、不许换掉的那几种颜色）。

        「按当前色块重新匹配」和「打开工程之后补全方案表」用的是同一份东西，所以抽出来
        (m02316 item 7)：手改过的颜色是用户定下来的，重新挑卷不能悄悄换回去；删除/合并
        改的是原图本身，所以「这一块本来是什么颜色」要从改过的原图上重新量。
        """
        # `pinned` 按当前色块的下标给出它们；编辑簿记用的是基准下标，`_live_map` 翻过来。
        pinned = {
            index: self._replaced[source]
            for index, source in enumerate(self._live_map)
            if source in self._replaced
        }
        source = self._result
        if source is not None and self._original_edited:
            means = self._original_means(source)
            if means is not None:
                source = replace(
                    source, means=means, source_values=None, source_counts=None
                )
        return source, pinned

    def _fill_open_plans(self, saved: BudgetOption) -> None:
        """List the other 耗材方案 right after 「打开工程」 (m02316 item 7).

        工程文件只记着保存时挑中的那一套方案，所以打开之后面板里就一行，用户想换一套
        得先按「按当前色块重新匹配」——但那个按钮的说明是「拿现在这些色块重新挑卷」，没人
        觉得「打开一个工程」要先按它。这里在打开工程时顺手把同一批色块的其它搭法算出来
        摆在下面：走的就是「按当前色块重新匹配」那条路（``_rematch_source`` +
        ``rematch_options``），所以挑中哪一套都和按那个按钮得到的一样，而当前这一套仍然
        是文件里那一套（``_plan_key`` 不动，勾还在它上面）。

        算不出来（没耗材、筛选太紧、匹配报错）就保持原样：打开工程的成功与否不该被
        这件事拖累，用户还有「重新匹配」和「按当前色块重新匹配」两个按钮。
        """
        if self._result is None or self._library is None:
            return
        palette = self._match_palette()
        if not palette:
            return
        source, pinned = self._rematch_source()
        if source is None:
            return
        self._begin_match()
        try:
            options = rematch_options(
                source,
                palette,
                self._settings(),
                pinned=pinned,
                require=self._require,
                avoid=self._avoid,
                progress=self._report_match,
            )
        except Exception:  # pragma: no cover - 打开工程不该因为补方案失败而失败
            return
        finally:
            self._end_match()
            # _end_match 只管把进度条收起来；控件是 _begin_match 关掉的，得还回来。
            self._set_controls_enabled(True)
        extra = [option for option in options if option.key != saved.key]
        if not extra:
            return
        self._plans = [saved, *extra]
        # 这一份 = 文件里那套 + 照现在这张图现算的几套，都对着现在的色块。
        self._plans_stale = False
        self._plan_key = saved.key
        self._refresh_plans()

    def _on_rematch_blocks(self) -> None:
        """「按当前色块重新匹配」(m15721 第 1 条): 颜色重找，色块不动，卷重挑。

        按一次走三步：

        1. 从**改过的原图**重新分色（``match_image`` 那一套，「颜色数量」说了算），
           所以颜色表和刚才可能不一样；
        2. 色块的划分与编号原样搬过来 —— 拿现在这份划分的标签图，只把每块的颜色按新的
           分色结果重算（:func:`app.core.blocks.revalue_blocks`）。 用户在色块模式里点过、
           改过的块因此还在原处，块级编辑（按像素掩码）也照样贴得回去；
        3. 再拿这些色块去重新挑卷（:func:`app.core.image_matching.rematch_options`）。

        行级的「更换颜色」会清掉：它按下标记的是旧颜色表里的第几行，新表一换就对不上了
        （旧表第 5 行是白，新表第 5 行可能是绿）。 从前那个「重新匹配」按钮已删：它和
        这个只差「要不要重新分块」，而分块是软件该自己认的事。
        """
        if self._image is None or self._result is None or self._library is None:
            return
        palette = self._match_palette()
        if not palette:
            self._status.setText("调色板是空的，请先添加耗材。")
            return
        # 现在这份划分：进过色块模式才有标签图，没有就照旧由 _build_blocks 自己分。
        labels = None if self._blocks is None else self._blocks.labels
        self._begin_match()
        try:
            fresh = match_image(self._image, palette, self._settings())
            options = rematch_options(
                fresh,
                palette,
                self._settings(),
                pinned={},
                require=self._require,
                avoid=self._avoid,
                progress=self._report_match,
            )
        except Exception as error:  # pragma: no cover - defensive
            QMessageBox.warning(self, "匹配失败", str(error))
            return
        finally:
            self._end_match()
        if not options:
            QMessageBox.warning(
                self, "匹配失败", "按现在的色块和筛选，没有找到能用的耗材搭配。"
            )
            return
        self._adopt_options(options, block_labels=labels, keep_block_edits=True)

    def _adopt_options(
        self,
        options: list[BudgetOption],
        *,
        block_labels: np.ndarray | None = None,
        keep_block_edits: bool = False,
        pushed: bool = False,
    ) -> None:
        """Make ``options[0]`` the match on screen, with the rest listed under it.

        Every edit so far was made against the previous 方案 and the rolls it spent,
        so the edit history starts over — but the state it had is pushed onto the
        undo stack first.  「按当前色块重新匹配」 is the one button that throws a whole
        match away, and the reporter pressed it, disliked the answer and could not
        get back (m01411 item 1): one Ctrl+Z now restores the match, its plans and
        the edits that were made on it.

        ``block_labels`` 是「新结果也用这份色块划分」的标签图（m15721 第 1 条）：
        :meth:`_build_blocks` 见到尺寸对得上就用 :func:`revalue_blocks` 把它贴到新
        分色上，块数和编号都不变。 ``keep_block_edits`` 则留着按像素掩码记的块级编辑
        —— 划分没变，那些掩码指的仍是用户当初点的那几块。
        """
        # The very first match of a picture has nothing behind it; pushing there
        # would arm Ctrl+Z against a page that had no match at all.  ``pushed``
        # says the caller already took it — 「重新匹配」 has to, because it clears
        # the rows before the match and the snapshot would otherwise miss them.
        if self._base is not None and not pushed:
            self._push()
        self._plans = options
        self._plans_stale = False
        self._plan_key = options[0].key
        self._base = options[0].result
        self._deleted = set()
        self._merged = {}
        self._replaced = {}
        self._block_template = None if block_labels is None else np.asarray(block_labels)
        if keep_block_edits:
            # 划分没动，掩码还落在同一批像素上：留着。
            pass
        else:
            # 块编辑记的是上一张匹配里某几块的像素；换了一张匹配，那些像素不再代表
            # 用户当时点的那一块，所以一起清掉（Ctrl+Z 能把它们和旧匹配一起拿回来）。
            self._block_edits = []
        # A new match is a new picture: whatever the project file holds is now
        # one step behind.
        self._dirty = True
        self._plate = None
        self._list.setLibrary(self._library)
        self._refresh_plans()
        # m01885 item 7: 新的一次匹配从「什么都没选」开始。上一次的高亮指的是上一份
        # 结果里的色块，留着它等于替用户选了一种颜色；编辑（删除 / 合并 / 更换颜色 /
        # 撤销）才继续跟着用户点过的那一块走。`_show_colour(-1)` 把预览的压暗和描边、
        # 列表的选中行、右侧详情一次清干净（m00830 item 2）。
        self._show_colour(-1)
        self._apply_edits()
        self._set_controls_enabled(True)
        self._announce_edits()

    # -- 耗材方案：同一张图，几套搭配，用户自己挑 ---------------------------
    def _on_spool_cap_changed(self, *_) -> None:
        """上限一改，旧的那几行就不一定是合法搭配了，当场重画一边记号。

        只重画，不重算：真正的搭配要按下 「重新匹配」 才换（m09456 item 4）。
        """
        if self._plans:
            self._refresh_plans()

    def _mark_plans_stale(self) -> None:
        """色块刚被改过：方案表里的数字不是这张图的了，标一下（不重算）。

        删除 / 合并 / 更换颜色 改的是图上有什么颜色，可是方案行还是上一次匹配算出来
        的那一份：用户删到只剩两种颜色，「差块 9」还挂在行上（差块是按图片的颜色数算
        的，两种颜色最多两个差块）——「颜色匹配还没改好」有很大一部分就是这个（m11180）。

        为什么只标记不重算：重算＝照现在的色块重新挑卷，等于替用户按了「按当前色块重
        新匹配」。用户按那个按钮、不满意、又回不去的那次投诉（m01411 第 1 条）之后，
        「换方案」只发生在用户自己按按钮的时候。这里和「超过现在的上限（N 卷）」一样，
        把话说清楚，把按钮指给他。
        """
        self._plans_stale = True
        if self._plans:
            self._refresh_plans()

    def _refresh_plans(self, placeholder: str = "") -> None:
        """Draw one radio row per 方案 and the line that explains the picked one.

        Rebuilt rather than updated: there are a handful of rows at most and every
        number on them changes with the match, so clearing and refilling is
        shorter than any diff.  ``placeholder`` replaces the rows with one sentence
        (used while a match is running, when the old rows are already wrong).
        """
        while self._plan_rows_layout.count():
            item = self._plan_rows_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        for button in self._plan_group.buttons():
            self._plan_group.removeButton(button)
        if not self._plans:
            self._plan_hint.setText(placeholder or PLAN_HINT_TEXT)
            self._plan_hint.setVisible(True)
            return
        for index, option in enumerate(self._plans):
            average, worst = option.fidelity
            cap = self._spool_cap.value()
            over = bool(cap) and len(option.spools) > cap
            row = QWidget()
            line = QHBoxLayout(row)
            line.setContentsMargins(0, 0, 0, 0)
            line.setSpacing(6)
            button = QRadioButton(option.title)
            button.setChecked(option.key == self._plan_key)
            button.setToolTip(option.note)
            self._plan_group.addButton(button, index)
            numbers = QLabel(
                f"平均 ΔE00 {average:.2f} · 最差 {worst:.2f} · 差块 {option.defects} · "
                f"用 {len(option.spools)} 卷"
            )
            numbers.setProperty("role", "hint")
            numbers.setToolTip(
                "「差块」= 图片里有几种颜色打出来偏差超过 ΔE00 5（按图片自己的颜色数算），"
                "越少越不容易一眼看出偏差。"
            )
            if option.from_file:
                # 这一行是打开工程时从文件里读回来的（m10318 第 2 条）：数字是保存
                # 那一刻的，不会因为升级了匹配规则而变——不写清楚，用户会以为
                # 「匹配还没改好」。
                numbers.setText(
                    numbers.text() + " · 从工程里读的那套，按「按当前色块重新匹配」重算"
                )
                button.setToolTip(
                    option.note
                    + "　这套方案是打开工程时从文件里读出来的，不是照现在这套规则和现在的"
                    "耗材库重新算的；按「按当前色块重新匹配」就会重算。"
                )
            if over:
                # 这一行是上一次匹配（上限还是别的数）算出来的，现在装了会超。
                numbers.setText(
                    numbers.text()
                    + f" · 超过现在的上限（{cap} 卷），按「按当前色块重新匹配」重算"
                )
                button.setEnabled(False)
            elif self._plans_stale:
                # m11180：这一行是改色块之前算的 —— 数字留着（那是当时那一套的事实），
                # 但要说清它没跟着改。右栏不宽，行尾只写最短的一句，怎么重算放在提示里。
                numbers.setText(numbers.text() + " · 色块改过了，数字是改之前的")
                button.setToolTip(
                    option.note
                    + "　这行是改色块之前算的：删掉 / 合并 / 更换颜色 之后，图上的颜色"
                    "换了，这几个数字（尤其「差块」）还是旧的。按工具条上的"
                    "「按当前色块重新匹配」，会把搭配照现在的色块重算一遍。"
                )
            line.addWidget(button)
            line.addWidget(numbers)
            line.addWidget(self._plan_swatches(option))
            line.addStretch(1)
            self._plan_rows_layout.addWidget(row)
        # 每一行按什么标准挑的、差块是什么，都在帮助窗口里（m01885 第 8 条）：
        # 面板上留着的那一句只讲现在是什么状态。
        self._plan_hint.clear()
        self._plan_hint.setVisible(False)

    def _plan_swatches(self, option: BudgetOption) -> QWidget:
        """One 方案 的卷，按它们自己的颜色画出来（m01885 第 1 条）。

        The rows used to be told apart by「（备选 N）」, which names a position
        instead of the thing the user is choosing.  Showing the colours means a
        方案 is named by what it prints, and the row and the 用到的耗材 list under
        it agree on which rolls they are talking about.
        """
        box = QWidget()
        line = QHBoxLayout(box)
        line.setContentsMargins(0, 0, 0, 0)
        line.setSpacing(3)
        # 1.6.62（m28297 第 2 条）：``use.filament`` 可能是 None —— 工程是从别人那里
        # 拿来的，里面的耗材本机库里没有，spool_usage 只能认出 id。以前这里直接读
        # ``use.filament.id``，于是「打开工程」当场 AttributeError。
        shares = {
            (use.filament.id if use.filament is not None else use.filament_id): use
            for use in spool_usage(option.result, self._library)
        }
        for filament_id in option.spools:
            filament = self._library.get(filament_id) if self._library is not None else None
            chip = QLabel()
            chip.setFixedSize(14, 14)
            colour = filament.color_hex if filament is not None else "#000000"
            chip.setStyleSheet(
                f"background-color: {colour};"
                f" border: 1px solid {theme.BORDER}; border-radius: 3px;"
            )
            use = shares.get(filament_id)
            name = filament.display_name if filament is not None else filament_id
            chip.setToolTip(f"{name} {use.share:.1%}" if use is not None else name)
            line.addWidget(chip)
        return box

    # -- 筛选：哪几卷必须用、哪几卷不要用 ---------------------------------
    def _ordered_ids(self, ids: frozenset[str]) -> list[str]:
        """``ids`` in library order, so the line reads like the 耗材库 page."""
        if self._library is None:
            return sorted(ids)
        return [filament.id for filament in self._library if filament.id in ids]

    def _spool_name(self, filament_id: str) -> str:
        if self._library is None:
            return filament_id
        filament = self._library.get(filament_id)
        return filament.display_name if filament is not None else filament_id

    def _update_filter_hint(self) -> None:
        """One line saying what the filter demands (「没有筛选」 when it demands nothing)."""
        parts = []
        if self._require:
            parts.append(
                "必须有：" + "、".join(self._spool_name(fid) for fid in self._ordered_ids(self._require))
            )
        if self._avoid:
            parts.append(
                "不要用：" + "、".join(self._spool_name(fid) for fid in self._ordered_ids(self._avoid))
            )
        self._filter_hint.setText("；".join(parts) + "。" if parts else "每一卷都可以用。")

    def _on_filter(self) -> None:
        """Ask which rolls may be spent, then match again with that demand in hand."""
        if self._image is None or self._library is None:
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
        if not self._confirm_filter():
            return
        self._require = require
        self._avoid = avoid
        self._update_filter_hint()
        # The filter is a rule about the rolls, so it invalidates the match that is
        # on screen: the rows above it were chosen without knowing it.
        self._on_rematch()

    def _confirm_filter(self) -> bool:
        """Warn before a new filter throws the edits away (it re-matches)."""
        if not (self._deleted or self._merged or self._replaced):
            return True
        ask = QMessageBox(self)
        ask.setWindowTitle("换筛选要重新匹配")
        ask.setIcon(QMessageBox.Icon.Warning)
        ask.setText("换个筛选会把这张图重新匹配一遍，现在这些改动会丢。")
        ask.setInformativeText("删除 / 合并 / 更换过的颜色只对当前这一套搭配有效。")
        change = ask.addButton("换筛选", QMessageBox.ButtonRole.AcceptRole)
        ask.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        ask.setDefaultButton(change)
        ask.exec()
        return ask.clickedButton() is change


    def _selected_plan(self) -> BudgetOption | None:
        for option in self._plans:
            if option.key == self._plan_key:
                return option
        return None

    def _check_plan(self, key: str) -> None:
        """Put the radios back in step with ``self._plan_key``.  No rebuild."""
        for index, option in enumerate(self._plans):
            button = self._plan_group.button(index)
            if button is not None:
                button.setChecked(option.key == key)

    def _on_plan_clicked(self, index: int) -> None:
        if not 0 <= index < len(self._plans):
            return
        option = self._plans[index]
        if option.key == self._plan_key:
            return
        if not self._confirm_plan_switch(option):
            self._check_plan(self._plan_key)
            return
        self._apply_plan(option)

    def _apply_plan(self, option: BudgetOption) -> None:
        """Make ``option`` the picture, dropping the edits made for the old one.

        The edits are dropped — the dialog said so — but the *state* that had them
        is pushed onto the undo stack first, so 「换错了」 costs one Ctrl+Z instead of
        the whole session's work.
        """
        self._push()
        self._plan_key = option.key
        self._base = option.result
        self._deleted = set()
        self._merged = {}
        self._replaced = {}
        # 方案换了就是换了一张图，色块级的改动跟着上一张走（Ctrl+Z 能一起拿回来）。
        self._block_edits = []
        # A different 方案 is a different picture: the project file needs saving
        # again, and the mesh has to be rebuilt from the new colours.
        self._dirty = True
        self._plate = None
        self._list.setLibrary(self._library)
        # m00830 item 2: a different 方案 is a different picture.  Keeping the old
        # selection drew the dim wash and the outline of a colour in a picture the
        # user had just replaced; a switch now starts from 「什么都没选」.
        self._show_colour(-1)
        self._apply_edits()
        self._set_controls_enabled(True)
        self._announce_edits()

    def _confirm_plan_switch(self, option: BudgetOption) -> bool:
        """Ask before switching, because a 方案 switch repaints the picture.

        The per-colour edits are indices into the *old* palette, so there is no
        honest way to carry them across; saying so is the whole job here.
        """
        if not (self._deleted or self._merged or self._replaced):
            return True
        box = QMessageBox(self)
        box.setWindowTitle("换一套耗材要重新分配颜色")
        box.setIcon(QMessageBox.Icon.Warning)
        box.setText(f"换成「{option.title}」会重画这张图，现在这些改动会丢。")
        box.setInformativeText("删除 / 合并 / 更换过的颜色只对当前这一套搭配有效。")
        switch = box.addButton("换一套", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(switch)
        box.exec()
        return box.clickedButton() is switch

    def _auto_base_entry(self) -> PaletteEntry | None:
        """没挑底板颜色时，跟着图上最大的一块走 —— 也就是老下拉框里的第一项。"""
        result = self._result
        if result is None or not result.palette:
            return None
        index = int(np.argmax(result.counts)) if len(result.counts) else 0
        if not 0 <= index < len(result.palette):
            index = 0
        return result.palette[index]

    def _shown_base_entry(self) -> PaletteEntry | None:
        """现在这块底板实际会用哪一种颜色：挑过就是挑的那一种，没挑就是自动那一种。"""
        return self._base_entry if self._base_entry is not None else self._auto_base_entry()

    def _base_pick_rgb(self) -> tuple[int, int, int]:
        """挑选框要的「现在这一种」——RGB 三元组，不是 ``"#RRGGBB"``。

        m17910 第 1 条咬过一次：之前这里把 ``color_hex`` 直接塞进
        :class:`ColourPickerDialog`，它在 ``int('#')`` 上抛 ``ValueError``，框还没建出来就
        死了，界面上看起来就是「更换…」按钮点不动。
        """
        current = self._shown_base_entry()
        if current is None:
            return (255, 255, 255)
        return tuple(int(value) for value in current.rgb)

    def _refresh_base_view(self) -> None:
        """把小色块和提示画成当前的底板颜色（m16941 第 2 条）。"""
        entry = self._shown_base_entry()
        colour = entry.color_hex if entry is not None else None
        if colour:
            self._base_swatch.setStyleSheet(
                f"background:{colour}; border:1px solid #9aa2ad; border-radius:3px;"
            )
        else:
            self._base_swatch.setStyleSheet(
                "background:transparent; border:1px dashed #9aa2ad; border-radius:3px;"
            )
        self._base_button.setEnabled(self._result is not None and bool(self._candidate_palette()))
        # 文案只说这个按钮干什么（m18149 第 2 条：原来的「更换…（自动）」把「自动」塞进按钮
        # 文字里，读起来像两个按钮拼在一起）；自动还是手挑，写进 tooltip。
        self._base_button.setText("更换底板颜色")
        if entry is None:
            self._base_button.setToolTip(BASE_COLOUR_TIP)
        else:
            where = (
                "现在这一种是自动挑的（跟着图上最大的一块）。"
                if self._base_entry is None
                else "现在这一种是手动挑的。"
            )
            self._base_button.setToolTip(
                f"{BASE_COLOUR_TIP}\n现在用的是 {colour}（{entry.short_label}）。{where}"
            )

    def _restore_base_entry(self, key: str) -> None:
        """按 ``PaletteEntry.key`` 认回工程里存的那一种底板颜色（m16941 第 2 条）。"""
        self._base_entry = None
        if key:
            for entry in self._candidate_palette():
                if str(getattr(entry, "key", "") or "") == key:
                    self._base_entry = entry
                    break
        self._refresh_base_view()

    def _on_pick_base_colour(self) -> None:
        """「底板颜色 → 更换…」：跟「更换颜色」用同一个挑选框，整套混色都能搜（m16941 第 2 条）。

        从前这里是一个下拉框，选项只有「打印图片需要的颜色」那几行 —— 想拿别的一卷混色
        当底板就没有路。 现在选中的是 ``PaletteEntry`` 本身（可能是图上没有的颜色），
        :func:`app.mesh.plate.build_plate` 会把它当成一个额外的耗材槽写进 3MF。
        """
        if self._result is None or self._library is None:
            return
        entries = self._candidate_palette()
        if not entries:
            self._status.setText("调色板是空的，请先添加耗材。")
            return
        current = self._shown_base_entry()
        dialog = ColourPickerDialog(
            entries,
            # 这个框要的是 RGB 三元组，不是 "#RRGGBB"（m17910 第 1 条：传 hex 会在 int('#')
            # 上抛异常，框建不出来，表现就是「点不动」）。
            self._base_pick_rgb(),
            library=self._library,
            current_key=current.key if current is not None else "",
            parent=self,
            title="更换底板颜色",
            target_label="现在的底板颜色",
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        chosen = dialog.chosen()
        if chosen is None:
            return
        self._push()
        self._base_entry = chosen
        self._plate = None
        self._refresh_base_view()
        self._dirty = True
        self._status.setText(
            f"底板颜色换成 {chosen.color_hex}（{chosen.short_label}）。导出时它自己占一个耗材槽；"
            "Ctrl+Z 可以撤销。"
        )

    # -- two-way selection ------------------------------------------------
    def _select(self, index: int) -> None:
        """Show one colour everywhere it appears: picture, list and detail.

        以前这里会把「上一个看过的颜色」记一笔，好让「再点一下取消选中」退回去；m12426
        item 2 之后取消选中就是**什么都不选**（``_on_colour_deselected``），这笔账不用
        记了 —— 用户点的是哪一个，屏幕上就只有哪一个。
        """
        self._show_colour(int(index))

    def _show_colour(self, index: int, *, keep_picked: bool = False) -> None:
        """Put ``index`` on screen without touching the preview history.

        ``index`` is whatever the list's rows carry: a palette index in
        「按颜色选择」, a **block id** in 「按色块边缘选择」 (m02366 item 2).  In 块模式
        the highlight is that block's pixels, not every pixel of its colour —
        which is the whole difference between 「这一块」 and 「这一种颜色」.

        ``keep_picked`` is for taking the highlight off one block while a
        multi-selection stays put: 「多选」 is a set of colours the user is about to
        删除 or 合并, so clearing the preview must not empty that set (m01411 item 3).
        """
        index = int(index)
        self._shown = index
        blocks = self._blocks
        # 块模式下 view 自己认不出块编号，一切高亮都由页面上算好的掩码说了算。
        self._view.setSelected(-1 if blocks is not None else index)
        self._original_view.setSelected(-1 if blocks is not None else index)
        if index < 0:
            # 「取消选中」: no colour, so no block, no detail and no row highlight.
            if not keep_picked:
                self._list.clear_selection()
            if blocks is not None:
                self._view.setSelectedMask(None)
            self._detail.clear()
            return
        self._list.pin(index)
        if blocks is not None:
            if not 0 <= index < blocks.count:
                self._view.setSelectedMask(None)
                self._detail.clear()
                return
            self._sync_block_highlight()
        elif self._result is not None and 0 <= index < len(self._result.palette):
            # 颜色模式这一侧也走同一套「太小的块把视角推过去」的逻辑（m26604 第 10 条）：
            # 0.0% 的颜色只有几像素，高亮画在原处等于没画。
            self._focus_tiny_block(self._result.indices == index)
        if self._result is None:
            return
        value = self._list.value_of(index)
        if not 0 <= value < len(self._result.palette):
            self._detail.clear()
            return
        original = None
        changed = False
        if value < len(self._live_map):
            source = self._live_map[value]
            if self._base is not None and 0 <= source < len(self._base.palette):
                original = self._base.palette[source]
            # ``-1`` = 块级编辑带进来的新颜色，本来就是「换过的」，没有原色可比。
            changed = source < 0 or source in self._replaced
        self._detail.setSelection(
            self._result,
            value,
            library=self._library,
            changed=changed,
            original=original,
        )

    def _on_colour_selected(self, index: int) -> None:
        self._select(index)

    def _on_colour_deselected(self, index: int) -> None:
        """Clicking the thing that is already on screen puts it away (m12426 item 2).

        单选时再点一下已经选中的颜色／色块，意思就是「取消选中」：**什么都不选**，清单里
        没有选中行，预览里也没有高亮。以前颜色模式下会退回「上一次看过的颜色」，于是看
        起来像「我点的是这一个，它却选中了上一次那个」—— 正是用户报的那一条。块模式早
        就是这个规矩（m02316 item 6），现在两种模式一致。
        """
        if self._shown < 0 or int(index) != self._shown:
            return
        self._list.clear_selection()
        self._show_colour(-1)

    def _on_region_clicked(self, index: int) -> None:
        """A click in 「预览」 means the same thing as a click on that row.

        With 「多选」 on it toggles the region in and out of the selection, so the
        user can pick colours by pointing at the picture instead of hunting for
        the row in the list.  Clicking a block that is already in the selection
        takes it back out — and then the block must not stay highlighted (m01411
        item 3: 列表说没选中的东西，预览不能还画着它).

        1.6.53（m21616 第 9/10 条）：单选时点「已经选中的那一种颜色」或「已经选中的
        那一块」，也是取消选中 —— 和列表里再点一行同一个动作，不再只是把高亮刷新一遍。
        """
        if self._multi_button.isChecked():
            self._list.toggle(index)
            if index not in self._list.selected_indices():
                self._show_colour(-1, keep_picked=True)
                return
        elif self._shown >= 0 and int(index) == self._shown:
            self._on_colour_deselected(int(index))
            return
        self._select(index)

    def _on_colour_activated(self, index: int) -> None:
        """双击一行 = 「这个颜色到底是什么」: 图片里的颜色、匹配到的颜色、配方.

        The details used to sit in a panel of their own, permanently showing
        whichever row was last clicked.  They are something the user asks about
        (double-click) rather than something that costs a third of the column,
        and the dialog is not modal so the next row can be asked about straight
        away — :meth:`_show_colour` keeps it in step with the selection.

        块模式下一行是一块，问的仍然是「这一块打印成什么颜色」，所以整段按
        ``value_of`` 折算后的调色板下标走 (m02366 item 3).
        """
        self._select(index)
        value = self._list.value_of(index)
        if self._result is None or not 0 <= value < len(self._result.palette):
            return
        entry = self._result.palette[value]
        dialog = self._detail_window()
        dialog.setWindowTitle(f"颜色详情 · {entry.color_hex.upper()} · {entry.short_label}")
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _detail_window(self) -> QDialog:
        """The 颜色详情 dialog, built on first use.

        It is a shell around ``self._detail`` (the same :class:`ColourDetail` the
        smoke test and the self-test read), so 「更换颜色」 stays on the action row
        where it always was instead of moving into a pop-up.
        """
        if self._detail_dialog is None:
            dialog = QDialog(self)
            dialog.setObjectName("colourDetailDialog")
            dialog.setMinimumWidth(380)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll.setWidget(self._detail)
            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
            buttons.rejected.connect(dialog.close)
            close = buttons.button(QDialogButtonBox.StandardButton.Close)
            if close is not None:
                close.setText("关闭")
            box = QVBoxLayout(dialog)
            box.setContentsMargins(12, 12, 12, 12)
            box.setSpacing(8)
            box.addWidget(scroll, 1)
            box.addWidget(buttons)
            self._detail_dialog = dialog
        return self._detail_dialog

    # -- the user's edits: 删除 / 合并 / 更换颜色 / 撤销 / 恢复 ---------------
    def _current_index(self) -> int:
        """The palette index the highlighted row stands for (``-1`` = none).

        行号在块模式下是块编号，所以这里统一过一遍 ``value_of``：调用方（更换颜色、
        详情）要的永远是「调色板里哪一种颜色」，不是「第几块」。
        """
        index = self._list.current()
        if index < 0:
            # 1.6.56（m25951 第 1 条连带）：`select()` 只改选中、不动「当前行」，切模式时
            # `clear_selection()` 又把当前行清成 -1 —— 于是「只选中、没有当前行」时按
            # 「更换颜色」会一声不响地什么都不做。没有当前行就拿选中的第一行。
            picked = self._list.selected_indices()
            if not picked:
                return -1
            return int(picked[0])
        value = self._list.value_of(index)
        if self._result is None or not 0 <= value < len(self._result.palette):
            return -1
        return value

    def _snapshot(self) -> tuple:
        """Everything one step of history has to put back.

        The edit bookkeeping alone used to be enough, because 删除 / 合并 /
        更换颜色 are the only acts that touch it.  「按当前色块重新匹配」 and a 方案
        switch replace the match itself — a new ``_base``, a new list of plans, a
        newly checked row — so those travel with the snapshot too, and Ctrl+Z after
        a re-match lands exactly where the user was (m01411 item 1).

        1.6.17 adds two more: the 色块级 edits (``_block_edits`` — the user asked for
        撤销/恢复 on them too) and「点图片时」 itself, because switching between
        「按颜色选择」 and 「按色块边缘选择」 changes what the list shows and therefore
        has to be one step the user can take back (m02439 的答复).

        1.6.19 adds the picture: 删除/合并 write into 原图 itself (m01566 的答复), so
        Ctrl+Z has to bring the old picture back as well.  The reference is enough —
        every path that changes ``_image`` (``_paint_original``, ``loadImage``,
        ``_adopt_project``) REPLACES the object and nothing ever edits one in place.

        m18338 第 1 条 adds ``_span``（匹配拿多少像素）: 导入/开工程时它等于图的长边，
        换掉它就是换了一套栅格 —— 那也是能撤的一步。
        """
        return (
            set(self._deleted),
            dict(self._merged),
            dict(self._replaced),
            self._base,
            list(self._plans),
            self._plan_key,
            list(self._block_edits),
            self._mode,
            self._image,
            self._original_edited,
            self._plans_stale,
            self._base_entry,
            self._span,
        )

    def _restore(self, snapshot: tuple) -> None:
        (
            deleted,
            merged,
            replaced,
            base,
            plans,
            plan_key,
            block_edits,
            mode,
            image,
            original_edited,
            plans_stale,
            base_entry,
            span,
        ) = snapshot
        self._deleted = set(deleted)
        self._merged = dict(merged)
        self._replaced = dict(replaced)
        self._base = base
        self._plans = list(plans)
        self._plan_key = plan_key
        self._block_edits = list(block_edits)
        self._mode = mode
        self._image = image
        self._original_edited = bool(original_edited)
        # 撤销回到「改色块之前」，那一行也就重新变回准的了（m11180）。
        self._plans_stale = bool(plans_stale)
        # 底板颜色也是一步能撤的编辑（m16941 第 2 条）。
        self._base_entry = base_entry
        # 匹配栅格也是（m18338 第 1 条）: 撤回到「改像素数之前」的那一套。
        self._span = int(span)
        self._plate = None
        self._refresh_base_view()
        # 原图 has to catch up before ``_apply_edits`` hands the views a backdrop.
        self._refresh_source()

    def _push(self) -> None:
        """Remember the state before an edit, and drop the redo history."""
        self._undo_stack.append(self._snapshot())
        # A long session should not grow without bound; 64 steps is far more
        # than anyone retraces by hand.
        del self._undo_stack[:-64]
        self._redo_stack.clear()
        # Every user edit comes through here, which is what makes this the one
        # place that has to remember 「不是工程文件里那一版了」.
        self._dirty = True

    def _resolve(self) -> tuple[dict[int, int], set[int]]:
        """``(index remap, indices that vanish)`` after the merges and deletions.

        Chains are followed — merging A into B and then B into C prints both as
        C — and a colour that was deleted takes everything merged into it with
        it, because those pixels are already printed as the deleted colour.
        """
        merged = dict(self._merged)
        for source in list(merged):
            target = merged[source]
            seen = {source}
            while target in merged and target not in seen:
                seen.add(target)
                target = merged[target]
            merged[source] = target
        gone = set(self._deleted)
        for source, target in merged.items():
            if target in gone:
                gone.add(source)
        return merged, gone

    def _compose(self) -> MatchResult | None:
        """Rebuild the live result from ``_base`` and the edit state.

        Always from the pristine match, never from the live result: the live one
        already carries the previous edits, so folding a new deletion into it
        would leave the deleted pixels behind.  Deleted pixels become ``-1``,
        which is exactly the transparent index the matcher already uses — so
        ``build_plate`` gives them no base plate and no colour prism.

        1.6.17 的色块级编辑（m02366 item 3）叠在最后一步：它们记的是「哪些像素改成
        什么」，跟颜色编号无关。所以先按颜色把 ``_base`` 折算成 live 编号，再把每一步
        的掩码刷上去；刷上去需要的颜色如果当前调色板里没有就追加一个位子——「把这几块
        换成这个颜色」当然可以带进来一种新颜色。
        """
        base = self._base
        if base is None:
            return None
        merged, gone = self._resolve()
        indices = np.array(base.indices, dtype=np.int32, copy=True)
        for source, target in merged.items():
            if source in gone or source == target:
                continue
            mask = indices == source
            if mask.any():
                indices[mask] = target
        for index in gone:
            indices[indices == index] = -1

        # A colour survives when it was not deleted and nothing was merged into
        # it... rather: when it was not deleted and is not itself a merge source.
        alive = [
            index
            for index in range(len(base.palette))
            if index not in gone and index not in merged
        ]
        palette = [self._replaced.get(index, base.palette[index]) for index in alive]
        live = np.full(indices.shape, -1, dtype=np.int32)
        for position, original in enumerate(alive):
            mask = indices == original
            if mask.any():
                live[mask] = position

        # live 位子 -> 原调色板下标（``-1`` = 块级编辑带进来的新颜色，图里没有它）。
        origins: list[int] = list(alive)
        if self._block_edits:
            slot_of: dict[str, int] = {}
            for position, entry in enumerate(palette):
                slot_of.setdefault(entry.key, position)
            for edit in self._block_edits:
                mask = edit.mask
                if mask.shape != live.shape or not mask.any():
                    continue
                if edit.entry is None:
                    live[mask] = -1
                    continue
                slot = slot_of.get(edit.entry.key)
                if slot is None:
                    slot = len(palette)
                    palette.append(edit.entry)
                    origins.append(-1)
                    slot_of[edit.entry.key] = slot
                live[mask] = slot

        if palette:
            totals = np.bincount(live[live >= 0].ravel(), minlength=len(palette))
            counts = np.asarray(totals[: len(palette)], dtype=base.counts.dtype)
        else:
            counts = np.zeros(0, dtype=base.counts.dtype)
        means = None
        if palette and base.means is not None and len(base.means) == len(base.palette):
            means = _live_means(base.means, indices, live, len(palette))

        self._live_map = origins
        return replace(
            base,
            palette=palette,
            indices=live,
            counts=counts,
            means=means,
            # 色块级编辑之后「每种颜色的色差」这本账不再对得上：它按原来的颜色记，
            # 而现在同一片地方改了打印色。宁可退回按块的均值算（``_fidelity_values``
            # 的第二条路），也不要拿旧数字冒充。
            source_values=None if self._block_edits else base.source_values,
            source_counts=None if self._block_edits else base.source_counts,
        )

    def _apply_edits(
        self,
        keep: int | None = None,
        *,
        reselect: Sequence[np.ndarray] | np.ndarray | None = None,
        drop_blocks: Sequence[int] | None = None,
    ) -> None:
        """Recompose the live result and show it everywhere.

        ``keep`` is the palette index the edit was about (the colour the user merged
        into, the one they swapped); the highlight follows it so the detail panel still
        shows what the edit produced.  If that colour is gone — the user deleted the very
        colour they had picked — nothing is selected instead.

        ``reselect`` is the pixels a 色块级 edit just touched, one mask per block it
        acted on: the block ids from before it are meaningless afterwards, so each
        of those blocks is found again by its own pixels and stays selected, which
        is what lets the user act twice in a row without hunting for the same block
        again (m02366 item 3).  一个一个地找而不是拿并集找，是为了「只改了一块就只选中
        那一块」(m02316 item 4).

        What must never happen is 预览 drawing a block while the list claims nothing is
        selected: that dim wash and outline over an unclaimed colour is 「没有选择颜色的时候
        不要显示色块」 (m00830 item 2, m01411 item 3).  ``_reveal`` keeps the list's selection
        and the highlight in step; the callers that replace the whole match (a 方案
        switch, 重新匹配) clear both with ``_show_colour(-1)`` before calling this.
        """
        self._result = self._compose()
        self._plate = None
        # 行都重建过了，屏幕上那个颜色不再有意义：一切从头开始。
        anchor = keep if keep is not None else self._list.current()
        self._shown = -1
        result = self._result
        if result is None:
            self._blocks = None
            self._view.setResult(None, self._source_qimage)
            self._original_view.setResult(None, self._source_qimage)
            self._view.setBlockLabels(None)
            self._list.setResult(None)
            self._refresh_base_view()
            self._detail.clear()
            self._update_actions()
            self._announce_edits()
            return
        # 「删除」是把整块像素变透明：留下的块一个像素都不会少，也不会被切成两半，
        # 所以照旧表去掉那几块就等价 —— 不必把整张图重新分一遍块（1.6.62，m28297
        # 第 4 条）。1680×2880 的工程上，那一次重新分块要 4 秒。
        if drop_blocks is not None and self._blocks is not None:
            blocks = without_blocks(self._blocks, drop_blocks)
        else:
            blocks = self._build_blocks(result)
        self._blocks = blocks
        self._view.setResult(result, self._source_qimage)
        self._original_view.setResult(result, self._source_qimage)
        self._view.setBlockLabels(
            blocks.labels if blocks is not None else None,
            blocks.sizes if blocks is not None else None,
        )
        self._list.setContent(result, blocks)
        self._refresh_base_view()
        if blocks is not None:
            # 块模式：选中的是块编号，跟调色板下标没有关系，所以整条高亮都走块表。
            wanted = self._blocks_after(reselect)
            if wanted and not self._multi_button.isChecked():
                # 「多选」关着的时候，编辑之后也只认一块：不然列表里会自己冒出两块
                # 高亮，而按钮明明在说「一次一样」(m02289 item 1)。
                wanted = wanted[:1]
            if wanted:
                self._list.select(wanted)
            if wanted:
                self._select(wanted[0])
            else:
                self._show_colour(-1)
        else:
            target = self._live_map.index(anchor) if anchor in self._live_map else -1
            if 0 <= target < len(result.palette):
                self._select(target)
            else:
                self._show_colour(-1)
        self._update_actions()
        self._announce_edits()

    # -- 色块：认出它们、画它们、改完之后把选中的找回来 --------------------
    def _build_blocks(self, result: MatchResult | None) -> BlockMap | None:
        """The block table of the current match — ``None`` outside 块模式.

        The threshold comes from the printed pixel count (:func:`min_block_pixels`):
        an anti-aliased edge is hundreds of single-pixel specks, and a list with
        one row per speck is unreadable and unclickable (measured on the reported
        pig: 953 blocks, 773 of them a single pixel).

        ``keep_colours`` 补上「去噪之后一个位子都不剩、可是颜色表里还写着」的颜色
        （m14748 第 2 条：那张工程里红色 474 px 全是 136 px 以下的碎片，色块表里
        一块红色都没有，而颜色表照样列着它）。

        ``self._block_template`` 是「这一份划分说了算」的标签图（m15721 第 1 条）：
        「按当前色块重新匹配」重新分色之后，块还得是同一批块 —— 尺寸对得上就用
        :func:`revalue_blocks` 把新颜色贴上去，尺寸对不上（用户改过「最大边长」）
        就退回自己分块。
        """
        if result is None or self._mode != PICK_EDGE:
            return None
        cache = getattr(self, "_blocks_cache", None)
        fingerprint = self._blocks_fingerprint(result)
        if cache is not None and cache[0] == fingerprint:
            # 同一份划分：块表没变，直接用上一次那份（m26286 第 4 条）。
            return cache[1]
        template = self._block_template
        if template is not None and np.shape(template) == np.shape(result.indices):
            # ``previous`` 传现在这一批块：某块在新分色里一格都不剩（全是「不打印」）时
            # 就留它本来那一种颜色，而不是从别人那里顺手取一个。
            blocks = revalue_blocks(template, result.indices, previous=self._blocks)
        else:
            blocks = label_blocks(
                result.indices,
                min_pixels=min_block_pixels(result.printed_pixels),
                keep_colours=True,
            )
        self._blocks_cache = (fingerprint, blocks)
        return blocks

    def _blocks_fingerprint(self, result: MatchResult) -> tuple:
        """块表的身份：同一份划分就认这一份缓存。

        不能拿 ``result is`` 当身份：``_apply_edits`` 每次都重建一个 ``MatchResult``，
        什么都没改也换了对象，于是每次切模式都重分一遍块（实测一次 350 ms，正是用户
        说的停顿）。拿像素本身做指纹，看得见的划分没变就复用（m26286 第 4 条）。
        """
        return (
            result.width,
            result.height,
            len(self._block_edits),
            zlib.crc32(np.ascontiguousarray(result.indices).tobytes()),
        )
    def _blocks_after(
        self, masks: Sequence[np.ndarray] | np.ndarray | None
    ) -> list[int]:
        """Which blocks to select after an edit: one block per block that was edited.

        ``masks`` 是刚被改动的那几块**各自**的像素（单个掩码也收）。一块一块地找：
        那块像素在新的一组块里落在哪个块上最多，就认哪一个 —— 合并三块得到的是一块，
        更换颜色得到的还是那一块，只是挨着边界的块不会被顺手拖进来。

        以前把它们的并集当成一个掩码、按「这个块有一半以上落在掩码里」来收，于是
        改一块的时候，相邻那块只要有一半压在掩码上就一起被选中（m02316 item 4：
        用户只换了一块，列表里却高亮了两块）。逐块取最大重叠既不会多收，也不会因为
        掩码被别的块分走一半而漏收。
        """
        blocks = self._blocks
        if blocks is None or masks is None:
            return []
        boxes = masks if isinstance(masks, (list, tuple)) else [masks]
        labels = blocks.labels
        picked: list[int] = []
        for mask in boxes:
            if mask is None:
                continue
            box = np.asarray(mask, dtype=bool)
            if not box.any() or box.shape != labels.shape:
                continue
            touched = labels[box]
            touched = touched[touched >= 0]
            if not touched.size:
                continue
            counts = np.bincount(touched.astype(np.int64))
            label = int(np.argmax(counts))
            if label not in picked:
                picked.append(label)
        if not picked:
            return []
        # 按块表自己的顺序返回（`order()` 是答案的顺序），列表选择集和详情面板才稳定。
        wanted = set(picked)
        return [int(label) for label in blocks.order() if int(label) in wanted]

    def _sync_block_highlight(self) -> None:
        """Draw the union of the selected blocks in 预览 (m02366 item 2/3).

        「多选」 in 块模式 is a set of blocks, so the picture has to show that set
        and not just the row the cursor is on.  In 颜色模式 nothing changes: the
        view keeps using ``_selected`` and its palette-index mask.
        """
        blocks = self._blocks
        if blocks is None or self._result is None:
            return
        mask = None
        for label in self._list.selected_indices():
            if not 0 <= label < blocks.count:
                continue
            one = blocks.mask(label)
            mask = one if mask is None else (mask | one)
        if mask is None:
            # 一块都没选：预览不许还画着东西（m00830 item 2）。
            if self._list.current() >= 0 and self._blocks is not None:
                label = self._list.current()
                if 0 <= label < blocks.count:
                    mask = blocks.mask(label)
                    self._view.setSelectedMask(mask)
                    self._focus_tiny_block(mask)
                    return
            self._view.setSelectedMask(None)
            return
        self._view.setSelectedMask(mask)
        self._focus_tiny_block(mask)

    def _focus_tiny_block(self, mask: np.ndarray) -> None:
        """选中的是 0.0% 那种一丁点大的块时，把它挪到画面中间并加强高光（m26604 第 10 条）。

        这种块在整张图里只有几像素，选完还留在原来的角落，用户根本看不见高亮在哪 ——
        列表里明明高亮了，预览里却像什么都没发生。所以这里把视角推过去：按块的重心居中，
        必要时放大到能看清的倍率；描边也交给视图加粗。
        """
        if mask is None:
            return
        size = int(mask.sum())
        if size <= 0:
            return
        grid = mask.shape[0] * mask.shape[1]
        if size > max(64, int(grid * 0.002)):
            # 够大就不动视角：用户自己看着整张图在选，突然跳一下反而添乱。
            self._view.setTinyHighlight(False)
            return
        self._view.setTinyHighlight(True)
        self._view.focusOn(mask)

    def _on_list_selection_changed(self) -> None:
        """No row selected means 预览 draws nothing (m00830 item 2, m01191 item 1).

        用户报的是「在未选中颜色的情况下，预览显示色块」，而且说不清自己是怎么把选中
        取消掉的——所以这里按选中集本身说话，而不是按某一条路说话。列表的选中变成空的
        来路有好几条：点列表下面的空白处（Qt 直接清选中，一次点击就够）、「多选」里再点
        一下已选中的最后一行、代码里 ``clear_selection()``、切换选择方式重建列表。以前
        只有 ``_reveal``／``_show_colour`` 里的那几条会顺手收掉高亮，空白处那一次没有，
        于是列表里空着、按钮灰着，而肤色那一块还亮着带描边。

        收高亮走 ``_show_colour(-1, keep_picked=True)``：选中集本来就是空的，不用再清；
        颜色模式放下 ``_selected``，块模式放下 ``setSelectedMask``，详情面板一起清空。
        """
        if self._list.selected_indices():
            self._sync_block_highlight()
            return
        self._show_colour(-1, keep_picked=True)

    def _on_pick_mode_changed(self, index: int) -> None:
        mode = self._pick_combo.itemData(index)
        # 切换选择方式本身算一步，Ctrl+Z 能把用户带回上一种模式（m02439 的答复）。
        self._set_pick_mode(
            mode if isinstance(mode, str) else DEFAULT_PICK_MODE, record=True
        )

    def _set_pick_mode(self, mode: str, *, save: bool = True, record: bool = False) -> None:
        """Point the view, the list and the combo at ``mode``, and remember it.

        m01411 item 4: 「按颜色选择」 is what this page always did — 点在哪一块就是
        哪一块; 「按色块边缘选择」 picks the block whose edge is nearest instead, so the
        one-pixel brows and outlines of a picture like the reporter's pig can be hit
        at all.  Whichever they used last is what they get next time.

        1.6.17: the mode now also decides what the list LISTS — colours or blocks —
        and switching it is a step of its own in the history, so Ctrl+Z takes the
        user back to the other mode (m02439 的答复).  ``record=False`` is for the
        undo/redo/load paths, which are restoring a mode that is already recorded.
        """
        mode = resolve_pick_mode(mode)
        if record and mode != self._mode:
            self._push()
        switched = mode != self._mode
        self._mode = mode
        self._view.setPickMode(mode)
        row = self._pick_combo.findData(mode)
        if row >= 0 and row != self._pick_combo.currentIndex():
            self._pick_combo.blockSignals(True)
            self._pick_combo.setCurrentIndex(row)
            self._pick_combo.blockSignals(False)
        if save:
            save_settings({"pick_mode": mode})
        # 标题跟着模式走（m02316 item 5）：块模式下这一栏装的是「第 N 块」，标题不能
        # 还写着「打印图片需要的颜色」。切换模式时才需要改，所以放在这里而不是每次刷新。
        self._sync_list_title()
        # ``_build_ui`` sets the saved mode before the list exists yet; there is
        # nothing to re-list at that point, only the view to point at the mode.
        if switched and getattr(self, "_list", None) is not None:
            # 列表整个换了内容，旧的选中集（颜色编号 vs 块编号）不能带过去。
            self._list.clear_selection()
            # 切到「按色块边缘选择」时第一次要跑一遍全分辨率的连通域，这是那一下停顿的
            # 来源（m26286 第 4 条）。已有缓存就不用等，也不必闪一下提示；真要重算就
            # 先说一声、把光标换成漏斗，别让界面看着像卡死。
            cache = getattr(self, "_blocks_cache", None)
            slow = (
                self._mode == PICK_EDGE
                and self._result is not None
                and (cache is None or cache[0] is not self._result)
            )
            if slow:
                self._status.setText("正在重新分色块…")
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                QApplication.processEvents(
                    QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents
                )
            try:
                self._apply_edits()
            finally:
                if slow:
                    QApplication.restoreOverrideCursor()
            # 1.6.56（m25951 第 4 条）：说明写到右上角那行，不再占用底部状态行。
            self._update_mode_hint()

    def _sync_source_title(self) -> None:
        """把原图的像素尺寸写进「原图」面板的标题（m14567）。

        预览画的是匹配栅格，原图比它大一截的时候色块形状自然对不上；标题里带上尺寸，
        用户一眼就知道自己被缩了多少，不用去猜是识别画错了还是图本身变了。
        """
        label = getattr(getattr(self, "_original_panel", None), "title_label", None)
        if label is None:
            return
        title = "原图" if self._image is None else f"原图 {self._image.width}×{self._image.height}"
        if label.text() != title:
            label.setText(title)

    def _grid_note(self) -> str:
        """「原图 2048×2048 → 匹配 512×512（缩小 4.0 倍）」这半句话（m14567）。

        状态行原来只说「匹配完成：512×512 像素」——用户看到预览里的台阶，会以为那
        512 就是原图的分辨率。把原图尺寸和倍数一起写出来，误解就没有了。

        1.6.47（m19050 第 2 条）：框里写的数比原图大时，明确说一句「不放大」——用户
        写了 512 而图只有 240 px，引擎不会插出 512 格来（插值插不出细节，只会把板撑
        1.6.47（m19050 第 2 条）：框里写的数比原图大时，明确说一句「不放大」——用户
        写 512 而图只有 240 px，屏幕上看不出差别，会以为是功能坏了。1.6.48（m19523）
        把「宽/长」两个框与那颗按钮撤了（用户要的），这一段话就只讲图自己多大、匹配用了
        多大；导入始终按图片自己的像素数跑。
        """
        source_w, source_h = self._source_size or self._image.size
        if self._result is None:
            return f"原图 {source_w}×{source_h}"
        tail_note = ""
        if int(self._span) > max(source_w, source_h):
            tail_note = (
                f"；框里写的 {int(self._span)} 比原图大，不放大"
                "（插值只会把格子撑大，多不出细节）"
            )
        grid_w, grid_h = self._result.width, self._result.height
        if (grid_w, grid_h) == (source_w, source_h):
            return f"原图 {source_w}×{source_h}（一格就是一个原像素）{tail_note}"
        factor = max(source_w / max(1, grid_w), source_h / max(1, grid_h))
        return (
            f"原图 {source_w}×{source_h} → 匹配 {grid_w}×{grid_h}"
            f"（缩小 {factor:.1f} 倍）{tail_note}"
        )
    def _sync_list_title(self) -> None:
        """Name the colour/block column after what it is currently listing.

        两种「点图片时」装的不是同一种东西，所以标题也不一样 (m02316 item 5)：
        「按颜色选择」里一行是一种颜色，「按色块边缘选择」里一行是一块。
        """
        panel = getattr(self, "_list_panel", None)
        if panel is None:
            return
        title = LIST_TITLE_BLOCK if self._mode == PICK_EDGE else LIST_TITLE_COLOUR
        label = getattr(panel, "title_label", None)
        if label is not None and label.text() != title:
            label.setText(title)

    def _update_mode_hint(self) -> None:
        """不再有那句说明（1.6.57，m26604 第 2 条）。

        1.6.56 把它搬到工具栏右端，1.6.57 用户要求直接删掉：下拉框里写着现在是哪种
        模式，做法本身在「帮助」里，页面上不必再占一行字。方法留着是因为切模式那几处
        调用点还在这里。
        """

    def _on_multi_toggled(self, on: bool) -> None:
        """「多选」：click to add a colour, click again to drop it — no Ctrl.

        关掉的时候把多出来的那几行放掉，并在状态行说明白——用户报过「没有多选却
        多选了色块」，那一行消失不能是悄悄发生的 (m02289 item 1)。
        """
        before = len(self._list.selected_indices())
        self._list.setMultiSelect(on)
        # Accent while it is ON, so "clicks add up" is never a guess.
        self._multi_button.setProperty("accent", "true" if on else "false")
        style = self._multi_button.style()
        style.unpolish(self._multi_button)
        style.polish(self._multi_button)
        if not on and before > 1:
            self._status.setText(
                f"已关掉「多选」：刚才选中的 {before} 项只留了当前这一项，"
                "要一次改几样就再点开「多选」。"
            )

    def _update_actions(self) -> None:
        """删除 needs one row, 合并 needs two, and the history drives the rest.

        一行在「按颜色选择」里是一种颜色、在「按色块边缘选择」里是一块，所以这条规则
        两种模式共用：块模式下它就是「删除要选一块、合并要选两块」(m02366 item 3).
        """
        picked = self._list.selected_indices() if self._result is not None else []
        live = bool(self._result is not None and len(self._result.palette))
        self._delete_button.setEnabled(live and len(picked) >= 1)
        self._merge_button.setEnabled(live and len(picked) >= 2)
        self._replace_button.setEnabled(live and len(picked) >= 1)
        self._undo_button.setEnabled(bool(self._undo_stack))
        self._redo_button.setEnabled(bool(self._redo_stack))

    def _announce_edits(self) -> None:
        self._update_usage()
        result = self._result
        if result is None:
            return
        parts = []
        if self._deleted:
            parts.append(f"删除 {len(self._deleted)} 种颜色（留空不打印）")
        if self._merged:
            parts.append(f"合并 {len(self._merged)} 种颜色")
        if self._replaced:
            parts.append(f"手动更换 {len(self._replaced)} 种颜色")
        if self._block_edits:
            parts.append(f"色块级改动 {len(self._block_edits)} 处")
        if not parts:
            # When the match found more than one way to spend the rolls, say so:
            # the panel is on the right, and a user who does not scroll that far
            # would never learn the choice exists.
            choices = (
                f"给出 {len(self._plans)} 套耗材搭配，右边「耗材方案」里自己挑一套；"
                if len(self._plans) > 1
                else ""
            )
            hint = (
                "点图片选一整块，多选可以攒成一组色块；点「更换颜色」把这些块一起换掉，"
                "也可以「删除」（留空不打印）或「合并」（都变成其中一种颜色）。"
                if self._blocks is not None
                else "点击右侧颜色看图片高光，点击图片看它属于哪种颜色；点「更换颜色」可以"
                "手动换掉不合适的颜色，多选颜色后可以「删除」（留空不打印）或「合并」。"
            )
            # 「N 种颜色」数的是**打印出来的颜色**（顶着同一种打印色的几个位子算一种），
            # 跟右侧小结、跟「颜色数量」上限同一个口径 (m13246 item 2)：位子数会多出来，
            # 用户对着两个数会以为上限又破了。
            self._status.setText(
                "匹配完成：{0}，{1} 种颜色。{2}".format(
                    self._grid_note() or f"{result.width}×{result.height} 像素",
                    len({entry.key for entry in result.palette}),
                    choices,
                )
                + hint
            )
            return
        printed = result.printed_pixels
        total = max(1, result.total_pixels)
        self._status.setText(
            "、".join(parts)
            + f"，现在打 {len({entry.key for entry in result.palette})} 种颜色，实心 {printed} 像素"
            f"（占 {printed / total * 100:.1f}%）。导出的 3MF / OBJ 就照这样切，"
            "Ctrl+Z 撤销、Ctrl+Y 恢复。"
        )

    def _update_usage(self) -> None:
        """Show which rolls this version of the picture spends, largest first.

        The percentages are shares of material, not of pixels: 「白 10% + 金 90%」
        uses nine times as much gold as white, so this is the line that tells
        the user which spools to actually load.
        """
        uses = spool_usage(self._result, self._library)
        # Back to the top: after an edit the biggest roll is the answer the user
        # came for, and it must not stay scrolled out of sight.
        bar = self._plan_scroll.verticalScrollBar()
        bar.setValue(bar.minimum())
        if not uses:
            self._usage.setText(USAGE_HINT_TEXT)
            return
        # 一排（m00830 第 3 条）：以前是「共 3 卷」加一行一卷，三卷就吃掉三行；现在
        # 卷名和占比横着排，宽了才换行，而且每一项自己不断行（nowrap）。
        parts = []
        for use in uses:
            chip = (
                f'<span style="background-color:{use.color_hex};">&nbsp;&nbsp;&nbsp;</span> '
                if use.color_hex
                else ""
            )
            missing = "" if use.filament is not None else "（耗材库里已经没有这一卷）"
            parts.append(
                f'<span style="white-space:nowrap;">{chip}{use.display_name}'
                f"{missing} <b>{use.share:.1%}</b></span>"
            )
        self._usage.setText(f"共 {len(uses)} 卷：" + " · ".join(parts))

    def _picked_blocks(self) -> tuple[list[int], np.ndarray | None]:
        """（选中的块编号, 它们合起来的掩码）—— 块模式下三个动作的输入。

        颜色模式下这三个动作拿的是「哪几种颜色」，块模式下拿的是「哪几块」；
        掩码是它们的像素，因为块级编辑记的就是像素 (m02366 item 3).
        """
        blocks = self._blocks
        if blocks is None:
            return [], None
        labels = [
            int(label)
            for label in self._list.selected_indices()
            if 0 <= int(label) < blocks.count
        ]
        if not labels:
            return [], None
        mask = None
        for label in labels:
            one = blocks.mask(label)
            mask = one if mask is None else (mask | one)
        return labels, mask

    def _delete_blocks(self, labels: list[int]) -> None:
        """「删除」 in 块模式：这几块留空，别的块照打 (m02366 item 3)."""
        blocks = self._blocks
        if blocks is None:
            return
        masks = [blocks.mask(label) for label in labels if 0 <= label < blocks.count]
        if not masks:
            return
        mask = masks[0]
        for one in masks[1:]:
            mask = mask | one
        self._push()
        # 原图 也跟着改：这些像素变透明，之后无论「重新匹配」还是「按当前色块重新
        # 匹配」都从改过的原图出发，删掉的地方就真的不会再被匹配回来 (m01566 答复).
        self._paint_original(mask, None)
        self._block_edits.append(BlockPaint(mask=mask, entry=None, labels=tuple(sorted(labels))))
        # 逐块交回去，删除之后选中的就是「删掉的那几块留下的那几块」(m02316 item 4).
        self._apply_edits(reselect=masks, drop_blocks=labels)
        self._mark_plans_stale()
        self._status.setText(
            f"已删除 {len(masks)} 块：这些地方留空、变成透明，不打印也没有底板；"
            "「原图」同样改掉了，所以重新匹配也不会再填回来。Ctrl+Z 可以撤销。"
        )

    # -- keyboard: Del means 「删除」 (m01566 item 1) ---------------------------

    #: Widgets that own the keyboard for typing.  Del inside one of these belongs
    #: to the field (「删掉一个字符」), never to 「删除」.
    _TEXT_WIDGETS = (QLineEdit, QAbstractSpinBox, QTextEdit, QPlainTextEdit)

    def _typing_in_field(self) -> bool:
        """True when the keyboard focus sits in a text field or a number box."""
        return isinstance(QApplication.focusWidget(), self._TEXT_WIDGETS)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt spelling
        """Del on the picture page does exactly what the 「删除」 button does.

        Qt hands the key to the focused child first, so a Del typed into a hex or
        number field is already consumed there; this only ever sees the key when
        no text widget wanted it (the list and the picture both ignore it), and
        ``_typing_in_field`` is a second lock on the same door.  There is no
        window-wide ``QShortcut`` on purpose: a shortcut outranks the focus
        widget and would break Del inside 颜色配方's search and hex fields.
        """
        if event.key() == Qt.Key.Key_Delete and not self._typing_in_field():
            self._on_delete_colours()
            event.accept()
            return
        super().keyPressEvent(event)

    def _on_delete_colours(self) -> None:
        picked = self._list.selected_indices()
        if not picked:
            return
        if self._blocks is not None:
            self._delete_blocks(picked)
            return
        # 一行一种颜色，可能顶着好几个位子（m12426 item 1）：删的是「这一种颜色」，
        # 所以它的每一处都要一起删。
        picked = self._list.expanded(picked)
        originals = [self._live_map[i] for i in picked if 0 <= i < len(self._live_map)]
        if not originals or self._result is None:
            return
        self._push()
        # 原图 一起改（m02316 item 2，用户选的就是「颜色也改图」）：删掉一种颜色 = 这些像素
        # 在源图里变透明，和按块删除做的是同一件事，所以「重新匹配」也不会再把它填回来。
        mask = np.isin(np.asarray(self._result.indices), np.asarray(picked, dtype=np.int32))
        self._paint_original(mask, None)
        self._deleted.update(originals)
        self._apply_edits()
        self._mark_plans_stale()
        self._status.setText(
            f"已删除 {len(originals)} 种颜色：这些地方留空、变成透明，"
            "不打印也没有底板；「原图」同样改掉了。Ctrl+Z 可以撤销。"
        )

    def _merge_blocks(self, labels: list[int]) -> None:
        """「合并」 in 块模式：在选中的块里挑一种颜色，所有选中的块都变成它.

        用户的原话是「用户在选中的色块里选一个颜色，把所有选中的色块变成选择的颜色」
        (m02366 item 3)：所以对话框里列的是这些块当前打印的颜色（去重，一种一行），
        选定之后每一块都被刷成那种颜色——它们会因此变成同色的块，本来就挨着的会连成
        一整块。颜色本身没有新增，所以调色板里的位子照旧。
        """
        blocks = self._blocks
        result = self._result
        if blocks is None or result is None:
            return
        entries: list[PaletteEntry] = []
        found: list[tuple[int, PaletteEntry]] = []
        seen: set[str] = set()
        for label in labels:
            value = blocks.value(label)
            if not 0 <= value < len(result.palette):
                continue
            entry = result.palette[value]
            if entry.key in seen:
                continue
            seen.add(entry.key)
            entries.append(entry)
            found.append((int(value), entry))
        if len(entries) < 2:
            return
        dialog = MergeDialog(entries, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        choice = dialog.chosen()
        if not 0 <= choice < len(entries):
            return
        target_slot, target = found[choice]
        masks = [blocks.mask(label) for label in labels if 0 <= label < blocks.count]
        if not masks:
            return
        mask = masks[0]
        for one in masks[1:]:
            mask = mask | one
        self._push()
        # 原图 一起改（m01566 答复）：并掉的那几块在原图里并进留下来的那一块，用的是
        # **那一块本来的颜色**，不是这里选中的打印色 (m01191 item 2)。所以原图的颜色
        # 集合不变、变的只是色块大小；重新匹配之后它们仍然是一种颜色、仍然连在一起。
        survivor = None
        for label in labels:
            if not 0 <= label < blocks.count or blocks.value(label) != target_slot:
                continue
            one = blocks.mask(label)
            survivor = one if survivor is None else (survivor | one)
        colour = None if survivor is None else self._picture_colour(survivor)
        if colour is not None:
            self._paint_original(mask, colour)
        for label in labels:
            if 0 <= label < blocks.count:
                self._block_edits.append(
                    BlockPaint(mask=blocks.mask(label), entry=target, labels=(int(label),))
                )
        # 逐块交回去：并掉的那几块现在是一块，选中的就是它 (m02316 item 4).
        self._apply_edits(reselect=masks)
        self._mark_plans_stale()
        self._status.setText(
            f"已把选中的 {len(masks)} 块都换成 {target.color_hex}，"
            "这些地方整块都用它打印；「原图」里它们也并成了留下来的那一块，"
            "颜色还是原图自己的。Ctrl+Z 可以撤销。"
        )

    def _on_merge_colours(self) -> None:
        picked = self._list.selected_indices()
        if len(picked) < 2 or self._result is None:
            return
        if self._blocks is not None:
            self._merge_blocks(picked)
            return
        # 合并说的是「这几行一起变成挑中的那一种」；一行可能顶着好几个位子
        # （m12426 item 1），它们跟着各自那一行走。
        entries = [self._result.palette[index] for index in picked]
        dialog = MergeDialog(entries, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        choice = dialog.chosen()
        if not 0 <= choice < len(picked):
            return
        chosen = int(picked[choice])
        picked = self._list.expanded(picked)
        target = self._live_map[chosen]
        originals = [self._live_map[index] for index in picked]
        self._push()
        # 原图 一起改（m02316 item 2）：合并几种颜色 = 这些像素在源图里并进留下来的那一种，
        # 用的是**它本来的颜色**，不是预览里那卷耗材的打印色 (m01191 item 2)。所以原图的
        # 颜色集合不变、变的只是色块大小，换成别的方案之后它们还是同一种颜色。
        mask = np.isin(np.asarray(self._result.indices), np.asarray(picked, dtype=np.int32))
        # 取色用的是「用户挑中的那一行」——picked 上面已经被展开成一行背后的好几个位子，
        # 不能拿展开后的随便一个下标当留下来的那一块 (m12426 item 1)。
        survivor = np.asarray(self._result.indices) == chosen
        colour = self._picture_colour(survivor)
        if colour is not None:
            self._paint_original(mask, colour)
        if target < 0:
            # 1.6.59（m27164）：合并的目标本身是「色块级编辑带进来的颜色」，它在基准
            # 调色板里没有位子，`_live_map` 给的是 -1。以前这条路上 `_merged[源] = -1`
            # 就只是把源颜色的像素写成透明 —— 用户看着像「源颜色被删掉了」，那块地方
            # 从预览里消失，正是这次报的 bug。
            # `_compose` 的顺序是「先按 merged/deleted 挪成 -1，最后才刷色块级编辑」，
            # 所以这里补一条把并过来的像素刷成目标颜色的编辑，透明那一步会被刷回来。
            self._block_edits.append(
                BlockPaint(mask=mask, entry=entries[choice], labels=())
            )
        for original in originals:
            if original >= 0 and original != target:
                self._merged[original] = target
        self._apply_edits(keep=target if target >= 0 else None)
        self._mark_plans_stale()
        self._status.setText(
            f"已把 {len(originals)} 种颜色合并成 {entries[choice].color_hex}，"
            "整块区域都用它打印；「原图」里这些地方也并进了留下来的那一块，"
            "颜色还是原图自己的。Ctrl+Z 可以撤销。"
        )

    def _undo(self) -> None:
        if not self._undo_stack:
            return
        self._redo_stack.append(self._snapshot())
        self._restore(self._undo_stack.pop())
        self._dirty = True
        # 撤销的那一步可能是「切换选择方式」，所以连模式一起摆回去（不记新的一步）。
        self._set_pick_mode(self._mode, save=False)
        self._apply_edits()
        # The step may have been a re-match or a 方案 switch, so the rows on the
        # right are re-drawn from the plans that came back with it.
        self._refresh_plans()
        self._status.setText("已撤销上一步。Ctrl+Y 可以再做回来。")

    def _redo(self) -> None:
        if not self._redo_stack:
            return
        self._undo_stack.append(self._snapshot())
        self._restore(self._redo_stack.pop())
        self._dirty = True
        self._set_pick_mode(self._mode, save=False)
        self._apply_edits()
        self._refresh_plans()
        self._status.setText("已恢复刚才撤销的那一步。")

    def _replace_blocks(self) -> None:
        """「更换颜色」 in 块模式：选中的块一起换成同一种颜色 (m02366 item 3)."""
        labels, mask = self._picked_blocks()
        result = self._result
        if not labels or mask is None or result is None or self._library is None:
            return
        index = self._current_index()
        value = index if index >= 0 else self._list.value_of(labels[0])
        if not 0 <= value < len(result.palette):
            return
        entry = result.palette[value]
        entries = self._candidate_palette()
        if not entries:
            self._status.setText("调色板是空的，请先添加耗材。")
            return
        # 「要替换的图片颜色」 is the colour of the block the user is looking at:
        # 块模式下列表里一行是一块，选中的那一块就是问题所在。
        dialog = ColourPickerDialog(
            entries,
            result.region_colour(value),
            library=self._library,
            current_key=entry.key,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        chosen = dialog.chosen()
        if chosen is None:
            return
        self._push()
        masks = []
        for label in labels:
            if 0 <= label < self._blocks.count:
                one = self._blocks.mask(label)
                masks.append(one)
                self._block_edits.append(
                    BlockPaint(mask=one, entry=chosen, labels=(int(label),))
                )
        # 逐块交回去：换了一块就只该选中那一块 (m02316 item 4).
        self._apply_edits(reselect=masks or mask)
        self._mark_plans_stale()
        self._status.setText(
            f"已把选中的 {len(labels)} 块换成 {chosen.color_hex}。Ctrl+Z 可以撤销。"
        )

    def _on_replace(self) -> None:
        if self._blocks is not None:
            self._replace_blocks()
            return
        index = self._current_index()
        if index < 0 or self._result is None or self._library is None:
            return
        entries = self._candidate_palette()
        if not entries:
            self._status.setText("调色板是空的，请先添加耗材。")
            return
        dialog = ColourPickerDialog(
            entries,
            self._result.region_colour(index),
            library=self._library,
            current_key=self._result.palette[index].key,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        chosen = dialog.chosen()
        if chosen is None:
            return
        # 「更换颜色」说的是这一行这一种颜色，它顶着的每一处都要换（m12426 item 1）。
        originals = [
            self._live_map[position]
            for position in self._list.expanded([index])
            if 0 <= position < len(self._live_map)
        ]
        original = self._live_map[index]
        self._push()
        if original < 0:
            # 1.6.60（m27359 第 1 条）：这一行是「色块级编辑带进来的颜色」，基准调色板里
            # 没有它的位子（`_live_map` 给 -1）。老代码把 `_replaced[-1] = 新颜色` 写进一本
            # 死账 —— 界面上什么都不换，用户看着就是「点了没反应」／「凭空多出一条 0%」。
            # 现在改成落一条色块级着色：这一行的像素直接刷成新颜色。
            mask = np.asarray(self._result.indices) == index
            self._block_edits.append(BlockPaint(mask=mask, entry=chosen, labels=()))
            self._apply_edits()
            self._mark_plans_stale()
            self._status.setText(
                f"已把整行 {chosen.color_hex} 换掉，"
                f"{int(mask.sum())} 个像素改用它打印。Ctrl+Z 可以撤销。"
            )
            return
        for slot in originals or [original]:
            self._replaced[slot] = chosen
        self._apply_edits(keep=original)
        self._mark_plans_stale()

    def _on_actions_changed(self) -> None:
        if self._result is not None:
            self._plate = None

    # -- export -----------------------------------------------------------
    def _plate_settings(self) -> PlateSettings:
        """板长宽按用户的图算（一格 :data:`CELL_MM` mm），厚度只有「底板厚度」一个输入。

        m15721 第 3 条：从前这里问三样 —— 打印宽度、底板厚度、颜色层厚度。 一格的
        实际尺寸本来就等于 打印宽度 ÷ 图的像素数，所以「打印宽度」和「一格多宽」是
        同一件事的两种说法；颜色层厚度则是 Bambu Studio 自己管的事（混色写在它的
        3MF 里），在这里填什么都不影响那边的结果。 现在只留底板厚度，而且是**整块**
        的厚度（含图上那一层 0.4 mm 的颜色）。

        m16941 第 2 条：底板颜色不再是「打印图片需要的颜色」那几行里的一个下标，用户可以从
        整套混色里挑任意一种 —— 挑的记成 ``base_key``（存进工程），那一种不在图片的调色板
        里时还会另占一个耗材槽（见 :func:`app.mesh.plate.build_plate` 的 ``base_entry``）。
        """
        base_index = -1
        shown = self._shown_base_entry()
        if shown is not None and self._result is not None:
            for position, entry in enumerate(self._result.palette):
                if str(getattr(entry, "key", "") or "") == str(getattr(shown, "key", "") or ""):
                    base_index = position
                    break
        return PlateSettings(
            cell_mm=CELL_MM,
            base_thickness_mm=float(self._base_spin.value()),
            colour_thickness_mm=COLOUR_BAND_MM,
            base_index=base_index,
            base_key="" if self._base_entry is None else str(self._base_entry.key),
            smooth_edges=bool(self._smooth_box.isChecked()),
            image_on_bottom=bool(self._side_combo.currentData()),
        ).clamped()

    def _on_smooth_toggled(self, checked: bool) -> None:
        """「圆滑色块」同时管预览 (1.6.26, m05557 item 3)。

        以前这个勾只进 ``PlateSettings``，也就是只在导出时才起作用：用户在界面上
        勾来勾去，看到的还是原始格子，只有在 Bambu Studio 里才看得出差别。 现在
        预览也照导出用的细格画，勾上就是连续边界、取消就是一块块。
        """
        self._view.setSmooth(bool(checked))

    def buildPlate(self) -> PlateModel:
        if self._result is None:
            raise ValueError("请先导入并匹配一张图片")
        self._plate = build_plate(
            self._result,
            self._plate_settings(),
            base_label="底板",
            base_entry=self._base_entry,
            progress=self._report_export,
        )
        return self._plate

    def _report_export(self, message: str, fraction: float) -> None:
        """One step of :func:`build_plate`, drawn while the export is running.

        圆滑色块要在细栅格上给每种颜色各做一趟取样，32 色的图能跑到好几秒；没有这根
        进度条用户看到的就是一个不动的窗口。 和匹配时一样用 ``ExcludeUserInputEvents``，
        免得在导出中间再点一次「生成」。
        """
        self._progress.setValue(int(max(0.0, min(1.0, fraction)) * PROGRESS_STEPS))
        self._status.setText(f"正在生成模型：{message}…")
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

    def _on_export(self, kind: str) -> None:
        # Building the plate for a 512×512 picture takes ~2 s, and with 「圆滑色块」
        # on (the default) the refinement adds a pass per colour on the fine grid.
        # Without a wait cursor, the progress bar and a line of text the window
        # simply stops answering, which is the whole 「未响应」 complaint; the
        # cursor is popped again on every exit path.
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self._status.setText("正在生成模型…")
        self._progress.setValue(0)
        self._progress.setVisible(True)
        QApplication.processEvents()
        try:
            plate = self.buildPlate()
        except Exception as error:
            QApplication.restoreOverrideCursor()
            self._progress.setVisible(False)
            QMessageBox.warning(self, "无法生成模型", str(error))
            return
        QApplication.restoreOverrideCursor()
        self._progress.setVisible(False)
        suffix = ".3mf" if kind == "3mf" else ".obj"
        label = "3MF 模型" if kind == "3mf" else "OBJ 模型"
        suggested = f"{self._image_name}_{datetime.now():%Y%m%d-%H%M%S}{suffix}"
        start = str(paths.exports_dir() / suggested)
        name, _ = QFileDialog.getSaveFileName(self, f"保存{label}", start, f"{label} (*{suffix})")
        if not name:
            return
        target = Path(name)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self._status.setText(f"正在写入 {target.name}…")
        QApplication.processEvents()
        written = None
        try:
            written = self._write_model(plate, kind, target)
        except OSError as error:
            # The dialog opens where the user asked for it, which is 下载.  If
            # the shell or a policy refuses that folder, losing the model is not
            # an option: write it next to our own data and name the file that
            # really exists.  (1.6.5 probed Downloads up front instead, so the
            # dialog silently opened in a hidden folder.)
            for alternative in self._alternative_targets(target):
                try:
                    written = self._write_model(plate, kind, alternative)
                except OSError:
                    continue
                QApplication.restoreOverrideCursor()
                QMessageBox.information(
                    self,
                    "已存到别的位置",
                    f"{target.parent} 写不进去：{error.strerror or error}\n\n"
                    f"模型已经存到：\n{written}",
                )
                break
            if written is None:
                QApplication.restoreOverrideCursor()
                QMessageBox.warning(self, "保存失败", str(error))
                return
        except Exception as error:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "保存失败", str(error))
            return
        QApplication.restoreOverrideCursor()
        self._status.setText(self._describe(plate, written))
        self._announce(plate, written)

    def _write_model(self, plate: PlateModel, kind: str, target: Path) -> Path:
        """Write *plate* to *target* in the requested format and return the path."""
        if kind == "3mf":
            # The filament list travels inside the project so Bambu Studio opens
            # it with *our* colours instead of whatever is in the AMS.
            filaments = (
                slots_from_palette(plate.filament_entries(self._result.palette), self._library)
                if self._result is not None
                else None
            )
            return write_3mf(plate, target, object_name=self._image_name, filaments=filaments)
        return write_obj(plate, target)

    @staticmethod
    def _alternative_targets(target: Path) -> list[Path]:
        """Where to retry a save the chosen folder refused, best first."""
        options: list[Path] = []
        homes: list[Path] = []
        try:
            homes.append(paths.data_dir() / "exports")
        except OSError:  # pragma: no cover - defensive
            pass
        try:
            homes.append(Path.home())
        except (OSError, RuntimeError):  # pragma: no cover - defensive
            pass
        for directory in homes:
            candidate = directory / target.name
            if candidate.parent == target.parent or candidate in options:
                continue
            try:
                candidate.parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                continue
            options.append(candidate)
        return options

    def _describe(self, plate: PlateModel, written: Path) -> str:
        stats = plate.stats()
        lines = [
            f"已生成 {written.name}",
            f"{stats['width_mm']} × {stats['depth_mm']} mm，总厚度 {stats['thickness_mm']} mm，"
            f"一格 {stats['pixel_mm']} mm",
            f"整板 1 个对象 / {stats['extruders']} 种耗材 / {stats['triangles']} 个三角面",
        ]
        lines.extend(plate.notes)
        return "  ·  ".join(lines)

    def _announce(self, plate: PlateModel, written: Path) -> None:
        stats = plate.stats()
        box = QMessageBox(self)
        box.setWindowTitle("已生成模型")
        box.setIcon(QMessageBox.Icon.Information)
        filaments = (
            slots_from_palette(plate.filament_entries(self._result.palette), self._library)
            if self._result is not None
            else None
        )
        rows = [
            f"文件：{written}",
            f"尺寸：{stats['width_mm']} × {stats['depth_mm']} mm，"
            f"厚 {plate.base_thickness_mm:g} mm（图上那层颜色 {plate.colour_thickness_mm:g} mm）",
            f"整块底板是一个对象，{stats['extruders']} 种颜色涂在不同的三角面上。",
            "",
            "在 Bambu Studio 里直接打开这个文件就行：",
            "1. 耗材清单（颜色 / 种类 / 品牌）已经写进 3MF，打开时会自动带进来，",
            "   不再取决于你 AMS 里插的是什么颜色的料；",
            "2. 整块底板是一个对象，不用再拼零件；",
            "3. 切片前确认「耗材映射」与你的 AMS 槽位一致。",
            "",
            "这个模型需要的耗材：",
        ]
        for index in range(1, plate.extruder_count + 1):
            slot = filaments[index - 1] if filaments and index <= len(filaments) else None
            colour = slot.color_hex if slot is not None else "?"
            kind = slot.material_type if slot is not None else ""
            vendor = slot.vendor if slot is not None else ""
            who = " ".join(part for part in (vendor, kind) if part)
            rows.append(f"  {index}. {colour}  {who}".rstrip())
        if plate.notes:
            rows.append("")
            rows.extend(f"提示：{note}" for note in plate.notes)
        box.setText("\n".join(rows))
        box.exec()

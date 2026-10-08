# 第三方组件、算法出处与授权说明

本程序（BambuPalette）的代码为 **GPL-3.0-or-later**。
下面列出它使用的第三方代码、算法、数据与参考实现，以及各自的出处和授权。

---

## 1. 运行时依赖

| 组件 | 版本 | 授权 | 用途 |
| --- | --- | --- | --- |
| **PySide6-Essentials**（Qt for Python） | 6.8–6.x | LGPL-3.0 / 商业双授权 | 图形界面（QtCore / QtGui / QtWidgets） |
| **NumPy** | ≥1.26 | BSD-3-Clause | 数值计算、矢量化混色与网格生成 |
| **Pillow** | ≥10.2 | MIT-CMU | 图片读取、缩放、颜色量化 |
| **PyInstaller** | 6.x（仅打包用） | GPL-2.0-or-later with bootloader exception | 打包成独立 exe |

---

## 2. ICC 光谱估计多项式（「光谱 KM 模型」引擎）

「光谱 KM 模型」的第一步——把 sRGB 颜色反推成 31 段可见光反射率——使用了 **ICC 的「Munsell Glossy D50 XYZ polynomial estimator」** 配置文件：

- 文件：`xyz2PolyEstimateRefV2.icc`
- 来源：<https://www.color.org/resources/spectral/xyz2PolyEstimateRefV2.icc>
- SHA-256：`8291983ea02ca7b7adf023a1f3ddd3fc618a853ef20ffd01eb145504a92ff2e4`
- Profile ID：`5436fbfce5f7414dc520bb6e5d9c1516`
- Copyright 2022 International Color Consortium
- 授权：见 ICC 的 profile library 条款 <https://registry.color.org/profile-library/>

本程序从该 ICC 文件中读出了 **31 × 20 = 620 个多项式系数**，固化在
`app/spectral/icc_profile.py` 里（由 `tools/gen_icc_profile.py` 生成），运行时不再需要 ICC 文件。
多项式项的顺序、波长采样（400–700 nm，每 10 nm，共 31 点）与常数完全按 ICC 规范。

**观测者与光源**用的是 CIE 1931 **10°** 标准观察者与 **D65** 光源，数据取自 CIE 公开标准表
（`app/spectral/color.py` 中的 `_CMF_10` / `_D65` 表）。CIE 数据表可自由使用，引用时请注明 CIE 出处。

---

## 3. Kubelka-Munk 与颜色数学

K/S 混合、反射率↔K/S 转换、XYZ↔Lab、sRGB 传递函数、**CIEDE2000** 色差公式都是公开的标准算法，
按 CIE / Kubelka-Munk 原始文献实现，没有引用第三方代码：

- Kubelka, P. & Munk, F. (1931). *Ein Beitrag zur Optik der Farbanstriche.*
- CIE 15:2004 *Colorimetry*；CIE 1931 标准观察者。
- Sharma, G., Wu, W., & Dalal, E. N. (2005). *The CIEDE2000 color-difference formula: Implementation notes, supplementary test data, and mathematical observations.*
  本程序的 CIEDE2000 实现通过了该论文给出的全部参考测试值（见 `tests/test_engine.py::ColourDifferenceTests`）。

---

## 4. 参考实现：OrcaSlicer-FullSpectrum

光谱引擎的算法路径参照了 **ratdoux/OrcaSlicer-FullSpectrum** 的公开源码，用于确认
「任意颜色 → 估计反射率 → K/S 混合 → Lab」这条链路的结构与参数：

- 仓库：<https://github.com/ratdoux/OrcaSlicer-FullSpectrum>
- 参考文件（下载后存放在 `tools/reference/`，仅作研究留档，不参与构建）：
  - `FullSpectrumICCPolynomialEstimator.cpp` / `.hpp`
  - `FullSpectrumICCPolynomialProfile.h`（620 个系数 + 常数）
  - `FullSpectrumKSPairResidual.cpp` / `.hpp`
  - `FullSpectrumMaterialDatabaseProfile.h`
  - `FullSpectrumMaterialHigherOrderProfile.h`
- 上游为 **AGPL-3.0**（OrcaSlicer 系）。本程序的**产品代码没有复制其任何代码**，只使用了其中的
  算法思路与从 ICC 文件独立提取的系数；对已知耗材材料的成对/三阶残余修正项
  （`FullSpectrumKSPairResidual` / `MaterialHigherOrderProfile`）**完全没有使用**，因为用户录入的是
  任意 RGB，无法对应到材料数据库中的已知材料。
- ⚠️ 上面那几个 `tools/reference/` 文件是**上游原文的留档**，不是本程序的一部分：它们不参与构建、
  不被导入、不进入 exe。它们各自的授权仍归上游（AGPL-3.0），把本仓库整体再分发时请一并遵守。
  `tools/reference/FilamentMixerModel.hpp` 同理（MIT）。如果你只想用本程序的代码，删掉整个
  `tools/reference/` 目录即可，程序与测试都不受影响。

---

## 5. 默认引擎：filament_mixer 颜料混合多项式

默认的「Bambu 混色预览」引擎，复刻的是 **Bambu Studio 2.8.2**（tag `v02.08.02.61`）
**当前**「添加混色耗材」对话框的算法。自 02.08 起该对话框不再做 sRGB 加权平均，
而是调用 `Slic3r::filament_mixer_lerp()`，也就是 Bambu 内置的 **filament_mixer** 库：

- `src/slic3r/GUI/MixedFilamentDialog.cpp`：`blend_colors()` 调用
  `Slic3r::filament_mixer_lerp(a.R, a.G, a.B, b.R, b.G, b.B, 1.0 - ratio_a, &r, &g, &bl)`；
  `blend_n_colors()` 生成十六进制颜色串与整数权重后调用 `Slic3r::blend_color_multi()`。
- `src/libslic3r/FilamentMixer.cpp` / `.hpp`：`filament_mixer_lerp()` 是对
  `::filament_mixer::lerp()` 的封装；`blend_color_multi()` 会**先**查 Bambu 私有的
  实测混色表 `lookup_measured_blend_color()`，查不到才退回多项式。
- `src/libslic3r/FilamentMixerModel.hpp`：**filament_mixer** 本体，纯头文件的
  四次多项式回归（330 项特征、7 个输入 `r1 g1 b1 r2 g2 b2 t`），用来近似 Mixbox 的
  颜料混色行为（作者给出的平均 ΔE ≈ 2.07）。

**filament_mixer 的授权是 MIT**，本程序**直接使用了它的系数数据**，因此附带其许可证：

```
MIT License

Copyright (c) 2026 Justin Hayes

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

处理方式：

- 上游头文件留档在 `tools/reference/FilamentMixerModel.hpp`（不参与构建）。
- `tools/gen_filament_mixer_profile.py` 解析该头文件，把 `POWERS[330][7]`、
  `COEF[330][3]`、`INTERCEPT[3]` 转录成 `app/spectral/filament_mixer_profile.py`；
  生成前会校验每行指数（0–4 且行和 ≤ 4），并用上游文档给出的样例
  `lerp(0,33,133, 252,211,0, 0.5) == (47,141,56)` 做自检，不通过就拒绝写文件。
- `app/spectral/filament_mixer.py` 是按该头文件重写的 NumPy 实现（不是复制 C++ 代码），
  包括 `t<=0` / `t>=1` 直接返回输入、以及每通道 `static_cast<int>()` 的**截断**行为。
- Bambu Studio 自身为 **AGPL-3.0**。本程序**没有复制它的任何代码**，只把公开源码作为
  「行为基准」；`blend_color_multi()` 里的实测混色表是 Bambu 私有数据，
  第三方无法复现，本程序走的是多项式分支。

---

## 6. 旧版引擎：Bambu Studio 2.5 的 sRGB 加权平均

「Bambu 2.5 旧版预览（sRGB 平均）」引擎照着 **Bambu Studio 2.5.3**（tag `v02.05.03.62`）的
`src/slic3r/GUI/MixedFilamentDialog.cpp` 复刻，供仍在用 2.5.x 的用户对上自己的预览色：

- 依据的代码：`blend_colors()`（两个 8 位 sRGB 颜色的加权平均，`unsigned char` 截断）
  与 `blend_n_colors()`（多色加权平均）。
- 另据该文件确认的界面规则：两色比例 `std::max(10, std::min(90, …))`、`MAX_COMPONENTS = 3`、
  推荐色块用 `0.5/0.5`（两色）与 `0.25/0.25/0.50`（三色）、以及**只允许同种耗材种类混色**。
- 该版本对话框里**没有**任何光谱 / Kubelka-Munk / `FilamentMixer` 代码，
  所以 2.5 的行为就是一份加权平均。

---

## 7. 参考文件：Bambu Studio 工程结构

3MF 写出器（`app/mesh/threemf.py`）的包结构是照着用户自己用 Bambu Studio `02.06.00.51`
导出的工程文件 `swatch_box.3mf` 复刻的（`[Content_Types].xml`、`_rels/.rels`、
`3D/3dmodel.model` 的 component 组装、`Metadata/model_settings.config` 里的
`<part extruder=…>` 与 `<plate>` 段）。这是 3MF 规范（3D Manufacturing Format，
<https://3mf.io/specification/>）加上 Bambu Studio 的私有扩展字段，属于公开格式。

---

## 8. 字体

界面使用系统自带字体：**Microsoft YaHei UI**、Segoe UI（Windows 内置），
等宽内容使用 Consolas / Cascadia Mono。本程序不打包、不重新分发任何字体文件。

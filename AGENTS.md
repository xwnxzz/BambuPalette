# 本仓库的工作约定

## 发版：用户已授权自动发布

用户 2026-10-14 明确说过「以后都自动发布」——**做完一轮改动、闸门全绿就自己发，不要再问**。
顺序：

1. `app/__init__.py` 的 `__version__` 升一档（第三段 +1：1.6.24 → 1.6.25）。exe 的版本资源由
   `build/BambuPalette.spec` 从这个文件读，所以只要改这一处。
2. 写 `dist/release-notes-<版本>.md`，照 `dist/release-notes-1.6.24.md` 的骨架：`# BambuPalette <版本>`、
   两行引言、`---`、编号小节（每条改动一节，带实测数字）、`## 文档`、`---`、`## 验证`、`## 升级`。
3. 写 `dist/commit-<版本>.txt`：第一行是提交标题（`<版本>：<一句话>`），空行，正文分段，
   末段是「闸门：…」与「docs/算法说明.md …；版本号升到 <版本>。」。
4. 四道闸门必须全绿（数字写进 release notes 与提交信息）：
   * `.venv\Scripts\python.exe -m unittest tests.test_blocks tests.test_engine tests.test_export tests.test_image tests.test_mesh tests.test_paths tests.test_project tests.test_triples`
     （`tests/` 里没有 `__init__.py`，所以不能 `-t .` discover）
   * `.venv\Scripts\python.exe tools\smoke_app.py` → `SMOKE OK`
   * `.venv\Scripts\python.exe -m app.main --selftest` → 退出码 0
   * `.venv\Scripts\python.exe -m PyInstaller --noconfirm --clean build\BambuPalette.spec`
     → `tools\package_release.py` → `tools\verify_release_exe.py`
     （打包版 exe 自检专抓 1.6.12 那种「只有冻结版才缺模块」的坑，不能省）
   * 动了导出 / 混色的版本再加跑 `tools\verify_bambu_roundtrip.py <导出的 3mf>`（本机装了 Bambu Studio）
5. 发布：
   ```
   git add -A
   git commit -F dist/commit-<版本>.txt
   git tag -a v<版本> -m "<提交标题>"
   git push origin main
   git push origin v<版本>
   gh release create v<版本> --title "BambuPalette <版本>" --notes-file dist/release-notes-<版本>.md dist/BambuPalette-<版本>-win64.zip
   ```
6. `dist/`、`selftest-report.txt`、`selftest-ui.png`、`build/BambuPalette/` 都在 `.gitignore` 里：
   release notes 只活在本地磁盘与 gh release 正文里，**不要**试图把它们提交进仓库。

## 写码与文档

* `docs/算法说明.md` 是唯一的文档：每一条行为改动都补一章，写清实测数字、取舍与**代价**——
  没量过的话不写，输了的地方照写（例：51.1 里「海报 48 色旧做法更好」）。
* 报出的数字要能复现：探针放 `%TEMP%` 下跑，脚本思路与关键行写进文档。
* 界面文案、注释、文档一律用中文；变量名与代码标识符保持英文。
* 不为了「看起来更好」改数字或改判据；闸门红了就修代码，不改闸门。

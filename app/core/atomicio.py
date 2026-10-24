"""原子落盘：先写同目录的临时文件，写完再改名顶上去。

1.6.49 之前，几处「保存」都是**以写入模式直接打开目标文件**：``project.save_project``
的 ``zipfile.ZipFile(target, "w")``、``settings.save_settings`` 的 ``target.write_text``、
以及 3MF / OBJ 导出。写入模式一打开就把目标文件截成 0 字节，于是中途出任何事——磁盘满、
U 盘被拔、断电、包里的某一步抛异常——留下的都是一个被截断的旧档：用户原来那份工程、
那份导出就没了，而报错信息还只是说「保存失败」。

改成先写 ``目标名.pid.attempt.tmp``（**同一个目录**，``os.replace`` 才保证是改名而不是
跨卷复制），写完 ``flush`` + ``fsync`` 再顶上去；出错就把临时文件删掉，目标文件一个字节
都不动。

临时文件名不用 :func:`tempfile.mkstemp`，理由与 ``library._stage`` 一样：Windows 上
``mkstemp`` 的重试循环会把「目录被强制完整性标签 / 「删除挂起」/ 杀软锁住」当成文件名
冲突，一直转到 ``TMP_MAX``；那里的注释有完整记录。这里用 ``进程号 + 序号`` 现算名字，
最多试 ``attempts`` 次，撞不到就报错走人。
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

#: 现算临时名时最多试几个（同目录下同一进程的并发写不会超过这个数）。
ATTEMPTS = 8


def _discard(path: Path) -> None:
    """Best-effort removal of a temporary file we are giving up on."""
    try:
        path.unlink()
    except OSError:  # pragma: no cover - defensive
        pass


def _flush_to_disk(path: Path) -> None:
    """把临时文件的内容压到盘上，再让它改名顶替。

    这一步做不到也不阻断保存：内容已经写进文件，``os.replace`` 本身仍然是原子的；
    fsync 只是把「改名已经在盘上、内容还在缓存里」这个窗口关掉。
    """
    try:
        with open(path, "rb+") as handle:
            handle.flush()
            os.fsync(handle.fileno())
    except OSError:  # pragma: no cover - defensive
        pass


@contextmanager
def atomic_path(target: str | Path, *, attempts: int = ATTEMPTS) -> Iterator[Path]:
    """Yield 一个同目录的临时路径；**正常退出**时才把它改名到 ``target``。

    用法就照普通文件写：::

        with atomic_path(目标) as 临时:
            with zipfile.ZipFile(临时, "w") as archive:
                ...

    正常退出 ⇒ ``fsync`` 之后 ``os.replace(临时, 目标)``；中途抛异常 ⇒ 删掉临时文件、
    异常照原样往外抛，``目标`` 保持原样（原来没有就还是没有）。
    """
    destination = Path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    for attempt in range(attempts):
        candidate = destination.with_name(
            f"{destination.name}.{os.getpid()}.{attempt}.tmp"
        )
        if not candidate.exists():
            temporary = candidate
            break
    if temporary is None:
        raise OSError(f"临时文件名重复，写不进 {destination}")
    try:
        yield temporary
    except BaseException:
        _discard(temporary)
        raise
    _flush_to_disk(temporary)
    # 1.6.62（m28297 第 2 条）：Windows 上杀软、索引器、OneDrive 同步会短暂攥住目标
    # 文件，``os.replace`` 于是报 WinError 5/32 —— 隔一百多毫秒再试一次通常就过去了。
    # 所有「保存」都走这条路径（工程、设置、3MF、OBJ），重试放在这里一次就够。
    last: OSError | None = None
    for pause in (0.0, 0.15, 0.35):
        if pause:
            time.sleep(pause)
        try:
            os.replace(temporary, destination)
            return
        except OSError as error:
            last = error
    # 三次都没顶上去（目标被别的程序锁着）：临时文件里是**这次**写好的内容，不能删
    # —— 报错里说清楚它在哪，比留一个没人认得的文件名有用。
    raise OSError(
        f"写不进 {destination}（{getattr(last, 'strerror', None) or last}）；"
        f"这次的内容留在 {temporary}"
    ) from last


def atomic_write_text(
    target: str | Path, text: str, *, encoding: str = "utf-8"
) -> Path:
    """Write ``text`` through a temporary file and swap it in."""
    destination = Path(target)
    with atomic_path(destination) as temporary:
        # ``newline`` 跟着 ``Path.write_text`` 的默认走（不指定 = 换行按平台翻译），
        # 这样这次改动不顺手改掉别人文件的换行风格。
        temporary.write_text(text, encoding=encoding)
    return destination


def atomic_write_bytes(target: str | Path, payload: bytes) -> Path:
    """Write ``payload`` through a temporary file and swap it in."""
    destination = Path(target)
    with atomic_path(destination) as temporary:
        temporary.write_bytes(payload)
    return destination

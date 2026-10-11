"""Filesystem locations for user data, caches and exports.

The application stores its filament library in the per-user application data
directory so the packaged ``.exe`` keeps working when it is installed somewhere
read-only.  Dropping a file named ``portable.txt`` next to the executable (or
next to this package when running from source) switches to portable mode, which
keeps everything in a ``data`` folder beside the executable instead.

Creating that directory can be refused for reasons that have nothing to do with
this program — an anti-virus lock, a roaming-profile redirection, a directory
left in Windows' "delete pending" state by an interrupted delete, or a process
running at a lower integrity level than the directory's mandatory label.  A
colour manager that cannot remember spools is annoying; one that refuses to
start at all is worse, so :func:`data_dir` walks a list of fallback locations and
only gives up after every one of them has failed.

Existence is not enough to decide that a directory is usable: Windows happily
lets a process *see* a directory it may not create files in, and ``mkdir`` on an
existing directory succeeds without needing write permission at all.  Every
candidate is therefore confirmed by actually creating and removing a probe file,
because that is the only question the rest of the program ever asks.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

APP_DIR_NAME = "BambuPalette"

#: Directory names this application used before it was renamed.  A user who
#: already had a library under one of them must not silently lose it.
LEGACY_APP_DIR_NAMES = ("FilamentColorStudio",)

#: Files worth carrying over when the preferred data directory is unusable.
_SEED_FILES = ("filament-library.json", "settings.json")


class DataDirectoryError(OSError):
    """Raised when no writable location could be found for user data.

    Deliberately an :class:`OSError`: every caller that already guards a
    filesystem write with ``except OSError`` therefore handles this too, which is
    what we want — "the disk refused" and "there is no usable disk" are the same
    problem from the caller's point of view.
    """


def is_frozen() -> bool:
    """True when running from a PyInstaller bundle."""
    return bool(getattr(sys, "frozen", False))


def executable_dir() -> Path:
    """Directory that holds the executable, or the project root when run from source."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent.parent


def resource_path(*parts: str) -> Path:
    """Locate a read-only file shipped with the program (the logo, and so on).

    Two different directories have to be tried, because PyInstaller unpacks
    bundled data into ``sys._MEIPASS`` while a source checkout keeps it in the
    project's ``assets`` folder.  ``parts`` are joined with the platform
    separator, e.g. ``resource_path("logo.ico")``.
    """
    bundle = getattr(sys, "_MEIPASS", None)
    roots = [Path(bundle) / "assets"] if bundle else []
    roots.append(Path(__file__).resolve().parent.parent.parent / "assets")
    for root in roots:
        candidate = root.joinpath(*parts)
        if candidate.is_file():
            return candidate
    return roots[-1].joinpath(*parts)


def portable_mode() -> bool:
    override = os.environ.get("BAMBU_PALETTE_PORTABLE", os.environ.get("FILAMENT_STUDIO_PORTABLE", ""))
    override = override.strip().lower()
    if override in {"1", "true", "yes", "on"}:
        return True
    if override in {"0", "false", "no", "off"}:
        return False
    return (executable_dir() / "portable.txt").exists()


def data_dir_override() -> Path | None:
    """Explicit data directory from ``BAMBU_PALETTE_DATA_DIR``, if set.

    This exists so that scripts — screenshot tools, experiments, a second copy
    started for comparison — can be pointed at a scratch directory.  Without it
    every ``MainWindow()`` writes its library straight into the user's real
    data directory, which is exactly how a screenshot script once replaced a
    user's spools with demo colours.
    """
    raw = os.environ.get("BAMBU_PALETTE_DATA_DIR", "").strip()
    if not raw:
        return None
    return Path(raw).expanduser()


def _copy_missing(source: Path, target: Path, names=None) -> None:
    """Copy tree ``source`` into ``target``, never overwriting what is there."""
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError:  # pragma: no cover - defensive
        return
    try:
        items = list(source.iterdir())
    except OSError:  # pragma: no cover - defensive
        return
    for item in items:
        if names is not None and item.name not in names:
            continue
        destination = target / item.name
        if destination.exists():
            continue
        try:
            if item.is_dir():
                shutil.copytree(item, destination)
            else:
                shutil.copy2(item, destination)
        except OSError:  # pragma: no cover - defensive
            continue


def _adopt_legacy_data_dir(base: Path) -> None:
    """Bring a pre-rename data directory into place, once.

    A plain rename is tried first because it is instant and keeps the old path
    from lingering.  Windows refuses to replace an existing directory, and the
    new directory may already exist but be empty (that is what an interrupted
    first run leaves behind), so when the rename is not possible the old files
    are *copied* across instead.  If even that fails the application starts with
    an empty library rather than crashing.
    """
    if base.is_dir():
        try:
            if any(base.iterdir()):
                return  # already holds data; never touch what is there
        except OSError:  # pragma: no cover - defensive
            return

    for legacy in LEGACY_APP_DIR_NAMES:
        candidate = base.parent / legacy
        if not candidate.is_dir():
            continue
        if not base.exists():
            try:
                os.replace(candidate, base)
                return
            except OSError:
                pass
            try:
                shutil.move(str(candidate), str(base))
                return
            except OSError:
                pass
        _copy_missing(candidate, base)
        return


def _candidate_roots() -> list[Path]:
    """Every place we are willing to keep data, best first."""
    # An explicit override is the whole list: a caller that said "use this
    # directory" must never end up quietly writing into the user's real one.
    forced = data_dir_override()
    if forced is not None:
        return [forced]

    roots: list[Path] = []
    portable = portable_mode()
    if portable:
        roots.append(executable_dir() / "data")
    local = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    if local:
        roots.append(Path(local) / APP_DIR_NAME)
    scratch = os.environ.get("TEMP") or os.environ.get("TMP")
    if scratch:
        roots.append(Path(scratch) / APP_DIR_NAME)
    roots.append(Path.home() / f".{APP_DIR_NAME.lower()}")
    if not portable and is_frozen():
        # Last resort for the packaged application: a ``data`` folder beside the
        # executable.  This is the one place that is still writable when the
        # process runs at a lower integrity level than the user profile
        # directories, which is exactly the case where every location above is
        # refused.  Only when frozen, so running from source cannot quietly
        # start writing into the project tree.
        roots.append(executable_dir() / "data")

    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root).casefold()
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def _write_probe(directory: Path, attempts: int = 4) -> str | None:
    """Return ``None`` when a new file can really be created in *directory*.

    Otherwise return a short human-readable reason.  ``open(..., "x")`` is used
    rather than :func:`tempfile.mkstemp` because a single ``open`` never retries:
    a refusal is reported at once instead of being mistaken for a name clash.
    """
    for attempt in range(attempts):
        probe = directory / f".{APP_DIR_NAME.lower()}-write-test-{os.getpid()}-{attempt}"
        try:
            with open(probe, "x", encoding="utf-8"):
                pass
        except FileExistsError:
            continue
        except OSError as exc:
            return exc.strerror or str(exc)
        try:
            probe.unlink()
        except OSError:  # pragma: no cover - defensive
            pass
        return None
    return "暂时无法创建探测文件"


def _holds_data(base: Path) -> bool:
    """True when *base* already contains one of the files we would seed."""
    for name in _SEED_FILES:
        try:
            if (base / name).exists():
                return True
        except OSError:  # pragma: no cover - defensive
            continue
    return False


def _data_stamp(base: Path) -> float:
    """Newest modification time among the files we would seed, or ``-1.0``."""
    newest = -1.0
    for name in _SEED_FILES:
        try:
            newest = max(newest, (base / name).stat().st_mtime)
        except OSError:
            continue
    return newest


def _usable(base: Path) -> str | None:
    """Prepare *base*; return ``None`` when it can really be written to.

    Otherwise return a ready-to-report reason that names the directory.  This is
    the only place that decides a candidate is acceptable, so "is it a
    directory", "does ``mkdir`` succeed" and "can a file actually be created"
    can never drift apart between callers.
    """
    # Adopt before creating the directory: a plain rename is the only path that
    # moves a pre-rename library rather than copying it, and Windows refuses to
    # replace a directory that already exists.
    _adopt_legacy_data_dir(base)
    try:
        base.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return f"{base} （{exc.strerror or exc}）"
    if not base.is_dir():
        return f"{base} （不是目录）"
    refusal = _write_probe(base)
    if refusal is not None:
        return f"{base} （无法写入：{refusal}）"
    return None


def data_dir() -> Path:
    """Root directory for all persisted application state.

    Falls back to the temporary directory, the home directory and finally the
    executable's own folder when the preferred location cannot be written,
    seeding the fallback with whatever library already exists so a transient
    lock never looks like data loss.
    """
    problems: list[str] = []
    roots = _candidate_roots()
    portable = portable_mode()

    # A library already written to a fallback location must not be orphaned when
    # the preferred location becomes writable again — a process that ran at a
    # lower integrity level, or an anti-virus lock that has since been lifted.
    # Whichever candidate already holds data wins over list order, and the newest
    # one wins over that, so the user's spools never appear to vanish.
    #
    # Portable mode is exempt: "keep the data beside the program" is explicit
    # intent and must beat a library that merely happens to sit in the profile.
    if not portable:
        populated: list[tuple[float, Path]] = []
        for base in roots:
            if not _holds_data(base):
                continue
            problem = _usable(base)
            if problem is None:
                populated.append((_data_stamp(base), base))
            else:
                problems.append(problem)
        if populated:
            populated.sort(key=lambda item: item[0], reverse=True)
            return populated[0][1]

    # Fresh start: the first writable candidate in priority order, seeded from
    # the preferred root.
    for position, base in enumerate(roots):
        problem = _usable(base)
        if problem is None:
            if position and not portable:
                _copy_missing(roots[0], base, names=_SEED_FILES)
            return base
        problems.append(problem)

    detail = "\n".join(f"  · {item}" for item in problems)
    raise DataDirectoryError(
        "找不到可写的数据目录，无法保存耗材档案。\n已尝试：\n"
        f"{detail}\n"
        "请检查磁盘空间和杀毒软件拦截；也可以在程序旁边放一个 portable.txt，"
        "让数据保存在程序目录下。"
    )


def scratch_dir(prefix: str = "") -> Path:
    """Create a fresh temporary directory, without an unbounded retry loop.

    :func:`tempfile.mkdtemp` has the same Windows retry behaviour as
    ``mkstemp``: a persistent refusal in a writable-looking parent is treated as
    a name collision and retried billions of times.  A bounded ``mkdir`` loop
    raises :class:`DataDirectoryError` instead.
    """
    stem = prefix or f"{APP_DIR_NAME.lower()}-"
    try:
        base = Path(tempfile.gettempdir())
    except (OSError, RuntimeError) as exc:  # pragma: no cover - defensive
        raise DataDirectoryError(f"找不到可用的临时目录：{exc}") from exc
    for attempt in range(8):
        candidate = base / f"{stem}{os.getpid()}-{attempt}"
        try:
            candidate.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            continue
        except OSError as exc:
            raise DataDirectoryError(
                f"无法在 {base} 建立临时目录：{exc.strerror or exc}"
            ) from exc
        return candidate
    raise DataDirectoryError(f"无法在 {base} 建立临时目录：文件名重复")


def library_path() -> Path:
    """JSON file that holds the filament library."""
    return data_dir() / "filament-library.json"


def default_start_dir() -> Path:
    """Directory for a file dialog to open at, even if our data dir is unusable.

    A save/open dialog is not worth an error: fall back to the home directory so
    the user can still pick a location by hand.
    """
    try:
        return data_dir()
    except OSError:
        return Path.home()


def projects_dir() -> Path:
    """Directory holding saved colour-matching projects."""
    path = data_dir() / "projects"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _downloads_candidates() -> list[Path]:
    """Places Windows is likely to keep the user's downloads, best first.

    ``%USERPROFILE%`` is asked before :func:`Path.home` because the latter can
    answer with a roaming or mapped profile while Explorer keeps using the local
    one.  Only the *display* name of the folder is translated on a Chinese
    Windows, so ``Downloads`` is the real name on every locale.
    """
    candidates: list[Path] = []
    profile = os.environ.get("USERPROFILE")
    if profile:
        candidates.append(Path(profile) / "Downloads")
    try:
        candidates.append(Path.home() / "Downloads")
    except RuntimeError:  # pragma: no cover - no home directory at all
        pass
    return candidates


def exports_dir() -> Path:
    """Default directory offered by the export dialogs.

    The exports have to be *findable*.  They used to land in
    ``%LOCALAPPDATA%\\BambuPalette\\exports`` — a hidden folder that Windows
    Explorer does not show by default and that no user would think to open, so
    the file they had just saved appeared to have vanished.  The user's own
    downloads folder is offered instead, which is where they asked for it and
    where the next step (dragging the file into Bambu Studio) happens anyway.

    A folder is only offered after a real write has been confirmed in it; a
    Downloads folder redirected onto a locked or disconnected drive falls back
    to the private folder so the dialog never opens somewhere the save will be
    refused.
    """
    for candidate in _downloads_candidates():
        if candidate.is_dir() and _write_probe(candidate) is None:
            return candidate
    path = data_dir() / "exports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_dir() -> Path:
    """Scratch space for cached spectra and thumbnails."""
    path = data_dir() / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path

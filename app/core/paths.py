"""Filesystem locations for user data, caches and exports.

A packaged copy keeps its spools in a ``data`` folder beside its own executable.
Unzipping a new version therefore starts from an empty library instead of
inheriting whatever the previous copy left behind, and two copies never share
one library.  When that folder cannot be written to — Program Files, a directory
carrying a low integrity label, an anti-virus lock — the copy falls back to
per-copy directories under the profile, the temporary directory and the home
directory, told apart by the name of the program folder (see
:func:`installation_tag`).

Running from source keeps the older profile-first order, so a checkout does not
scatter data next to its code.  Dropping a file named ``portable.txt`` next to
the executable (or next to this package when running from source) switches to
portable mode: the same ``data`` folder beside the program, and never a library
from the profile.

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

import hashlib
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


def installation_tag() -> str:
    """Short, filesystem-safe name for *this* copy of the program.

    Each extracted copy owns its library, so the fallback locations have to be
    per-copy too: two folders that both happen to be called ``BambuPalette``
    would otherwise share one library, which is exactly the sharing this release
    removes.  The folder name stays in front — with the program's own name
    stripped, so the result does not read ``BambuPalette-BambuPalette-1.6.11`` —
    because that is what a human sees when they go looking for the file; the
    digest keeps copies with the same folder name apart.
    """
    root = executable_dir()
    stem = "".join(ch if (ch.isalnum() or ch in "-_.") else "-" for ch in root.name)
    for prefix in (APP_DIR_NAME, *LEGACY_APP_DIR_NAMES):
        if stem.casefold().startswith(prefix.casefold()):
            stem = stem[len(prefix) :].lstrip("-_. ")
            break
    # Windows silently drops trailing dots/spaces, so never end the name with one.
    stem = stem.rstrip("-_. ")[:40].rstrip("-_. ") or "copy"
    digest = hashlib.sha1(str(root).casefold().encode("utf-8")).hexdigest()[:6]
    return f"{stem}-{digest}"


def _tagged(base: Path, tag: str) -> Path:
    """*base* itself, or a sibling named after the copy when *tag* is set."""
    return base if not tag else base.with_name(f"{base.name}-{tag}")


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

    portable = portable_mode()
    roots: list[Path] = []
    if is_frozen():
        # A packaged copy is a program of its own: its spools live in a ``data``
        # folder beside its own executable, and it never reads a library out of
        # the profile.  This is what makes "unzip the new version" start empty,
        # and what keeps two copies from sharing one library.
        roots.append(executable_dir() / "data")
        tag = installation_tag()
    else:
        if portable:
            roots.append(executable_dir() / "data")
        tag = ""

    # Fallbacks, in the same order as before.  A packaged copy gets per-copy
    # directories as well, so it stays independent of every other copy even when
    # its own folder cannot be written to.
    local = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    if local:
        roots.append(_tagged(Path(local) / APP_DIR_NAME, tag))
    scratch = os.environ.get("TEMP") or os.environ.get("TMP")
    if scratch:
        roots.append(_tagged(Path(scratch) / APP_DIR_NAME, tag))
    roots.append(_tagged(Path.home() / f".{APP_DIR_NAME.lower()}", tag))

    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root).casefold()
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def legacy_shared_library() -> Path | None:
    """The per-machine library an earlier version would have left behind.

    Copies used to read and write one file in the profile — ``BambuPalette``
    after the rename, ``FilamentColorStudio`` before it — so unzipping a fresh
    copy showed the previous copy's spools.  Neither file is read any more, but
    they hold *the user's* spools, so the 「导入」 dialog offers the folder of
    whichever is still there instead of making the user guess a path Explorer
    never showed them.
    """
    local = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    if not local:
        return None
    base = Path(local)
    for name in (APP_DIR_NAME, *LEGACY_APP_DIR_NAMES):
        candidate = base / name / "filament-library.json"
        try:
            if candidate.is_file():
                return candidate
        except OSError:  # pragma: no cover - defensive
            continue
    return None


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

    A packaged copy keeps everything in a ``data`` folder beside its own
    executable, so a freshly unzipped copy has an empty library and no copy
    shares another's spools.  That folder, the copy's own profile/temporary/home
    directories are tried in order, and a library already written to a fallback
    is preferred over an empty first choice so that a transient lock never looks
    like data loss.
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
        "请检查磁盘空间和杀毒软件拦截。（从源码运行时，也可以在程序旁边放一个 "
        "portable.txt，把数据固定到程序目录下。）"
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

    Downloads is offered **without testing it first**.  1.6.5 probed, and on the
    reporter's machine the probe failed while the folder was perfectly usable:
    the dialog then opened in the hidden private folder, which is the exact
    complaint 「默认保存地址为下载」.  A probe decides whether *our* process may
    create a file, and that is not the same question as where the user wants to
    look for it — so the folder they asked for is offered, and a write that is
    genuinely refused is handled where it happens (``picture_page._write_model``
    retries next to our own data and names the file that really exists).
    """
    for candidate in _downloads_candidates():
        if candidate.is_dir():
            return candidate
    path = data_dir() / "exports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_dir() -> Path:
    """Scratch space for cached spectra and thumbnails."""
    path = data_dir() / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path

"""Where user data lives, and what happens when that location is unusable.

These tests exist because of a real startup crash: creating the data directory
raised ``PermissionError: [WinError 5]`` and the whole application died before
the window appeared.  A colour manager that cannot find a place to write must
still start.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app.core import paths

ENV_KEYS = (
    "LOCALAPPDATA",
    "APPDATA",
    "TEMP",
    "TMP",
    "BAMBU_PALETTE_PORTABLE",
    "FILAMENT_STUDIO_PORTABLE",
    "BAMBU_PALETTE_DATA_DIR",
)


def _library(color: str = "#123456") -> dict:
    return {
        "version": 2,
        "filaments": [
            {
                "id": "f1",
                "brand": "大简",
                "material_type": "PETG HF",
                "color_hex": color,
                "note": "",
                "name": "测试",
            }
        ],
    }


class DataDirTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory(prefix="fcs-paths-")
        self.root = Path(self._temp.name)
        self._saved = {key: os.environ.get(key) for key in ENV_KEYS}

    def tearDown(self) -> None:
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._temp.cleanup()

    def env(self, **values) -> None:
        for key, value in values.items():
            os.environ[key] = str(value)


class LegacyAdoptionTests(DataDirTestCase):
    def test_legacy_directory_is_renamed_into_place(self) -> None:
        self.env(LOCALAPPDATA=self.root)
        legacy = self.root / "FilamentColorStudio"
        legacy.mkdir()
        (legacy / "filament-library.json").write_text(
            json.dumps(_library(), ensure_ascii=False), encoding="utf-8"
        )

        resolved = paths.data_dir()

        self.assertEqual(resolved, self.root / "BambuPalette")
        self.assertFalse(legacy.exists())
        payload = json.loads((resolved / "filament-library.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["filaments"][0]["color_hex"], "#123456")

    def test_legacy_data_survives_an_empty_new_directory(self) -> None:
        """The regression: an interrupted first run leaves an empty new folder.

        The old library used to be abandoned for good, because adoption returned
        early whenever the new directory merely existed.
        """
        self.env(LOCALAPPDATA=self.root)
        (self.root / "BambuPalette").mkdir()  # empty, but present
        legacy = self.root / "FilamentColorStudio"
        legacy.mkdir()
        (legacy / "filament-library.json").write_text(
            json.dumps(_library("#ABCDEF"), ensure_ascii=False), encoding="utf-8"
        )
        (legacy / "settings.json").write_text('{"engine": "spectral"}', encoding="utf-8")

        resolved = paths.data_dir()

        payload = json.loads((resolved / "filament-library.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["filaments"][0]["color_hex"], "#ABCDEF")
        self.assertEqual(
            json.loads((resolved / "settings.json").read_text(encoding="utf-8"))["engine"],
            "spectral",
        )

    def test_existing_new_data_is_never_overwritten_by_the_legacy_copy(self) -> None:
        self.env(LOCALAPPDATA=self.root)
        current = self.root / "BambuPalette"
        current.mkdir()
        (current / "filament-library.json").write_text(
            json.dumps(_library("#0000FF"), ensure_ascii=False), encoding="utf-8"
        )
        legacy = self.root / "FilamentColorStudio"
        legacy.mkdir()
        (legacy / "filament-library.json").write_text(
            json.dumps(_library("#FF0000"), ensure_ascii=False), encoding="utf-8"
        )

        resolved = paths.data_dir()

        payload = json.loads((resolved / "filament-library.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["filaments"][0]["color_hex"], "#0000FF")

    def test_a_renamed_directory_is_not_adopted_twice(self) -> None:
        self.env(LOCALAPPDATA=self.root)
        legacy = self.root / "FilamentColorStudio"
        legacy.mkdir()
        (legacy / "filament-library.json").write_text("{}", encoding="utf-8")

        first = paths.data_dir()
        (first / "filament-library.json").write_text('{"kept": true}', encoding="utf-8")
        second = paths.data_dir()

        self.assertEqual(first, second)
        self.assertEqual(
            json.loads((second / "filament-library.json").read_text(encoding="utf-8")),
            {"kept": True},
        )


class FallbackTests(DataDirTestCase):
    def _block(self, name: str) -> Path:
        """A path that cannot contain a directory because it is a file."""
        blocker = self.root / name
        blocker.write_text("not a directory", encoding="utf-8")
        return blocker

    def test_falls_back_to_the_temporary_directory(self) -> None:
        scratch = self.root / "scratch"
        scratch.mkdir()
        self.env(LOCALAPPDATA=self._block("blocked-local"), TEMP=scratch, TMP=scratch)

        resolved = paths.data_dir()

        self.assertEqual(resolved, scratch / "BambuPalette")
        self.assertTrue(resolved.is_dir())

    def test_falls_back_to_the_home_directory(self) -> None:
        home = self.root / "home"
        home.mkdir()
        self.env(LOCALAPPDATA=self._block("blocked-local"), TEMP=self._block("blocked-temp"), TMP=self._block("blocked-tmp"))
        with mock.patch.object(Path, "home", classmethod(lambda cls: home)):
            resolved = paths.data_dir()

        self.assertEqual(resolved, home / ".bambupalette")
        self.assertTrue(resolved.is_dir())

    def test_every_location_failing_raises_a_readable_error(self) -> None:
        self.env(
            LOCALAPPDATA=self._block("blocked-local"),
            TEMP=self._block("blocked-temp"),
            TMP=self._block("blocked-tmp"),
        )
        with mock.patch.object(Path, "home", classmethod(lambda cls: self._block("blocked-home"))):
            with self.assertRaises(paths.DataDirectoryError) as caught:
                paths.data_dir()

        message = str(caught.exception)
        self.assertIn("找不到可写的数据目录", message)
        self.assertIn("BambuPalette", message)

    def test_library_path_uses_the_fallback(self) -> None:
        scratch = self.root / "scratch"
        scratch.mkdir()
        self.env(LOCALAPPDATA=self._block("blocked-local"), TEMP=scratch, TMP=scratch)

        self.assertEqual(paths.library_path(), scratch / "BambuPalette" / "filament-library.json")

    def test_the_error_is_an_os_error_so_existing_guards_catch_it(self) -> None:
        """Callers already guard filesystem writes with ``except OSError``."""
        self.assertTrue(issubclass(paths.DataDirectoryError, OSError))

    def test_default_start_dir_never_raises_when_the_data_dir_is_unusable(self) -> None:
        home = self.root / "home"
        home.mkdir()
        self.env(
            LOCALAPPDATA=self._block("blocked-local"),
            TEMP=self._block("blocked-temp"),
            TMP=self._block("blocked-tmp"),
        )
        with mock.patch.object(Path, "home", classmethod(lambda cls: home)):
            resolved = paths.default_start_dir()

        self.assertTrue(resolved.is_dir())
        self.assertEqual(resolved.parent, home)

    def test_default_start_dir_returns_the_home_directory_when_nothing_works(self) -> None:
        blocked_home = self._block("blocked-home")
        self.env(
            LOCALAPPDATA=self._block("blocked-local"),
            TEMP=self._block("blocked-temp"),
            TMP=self._block("blocked-tmp"),
        )
        with mock.patch.object(Path, "home", classmethod(lambda cls: blocked_home)):
            self.assertEqual(paths.default_start_dir(), blocked_home)

    def test_default_start_dir_is_the_data_directory_when_it_works(self) -> None:
        self.env(LOCALAPPDATA=self.root)
        self.assertEqual(paths.default_start_dir(), self.root / "BambuPalette")


class SettingsFallbackTests(DataDirTestCase):
    """Preferences must not be able to stop the application from starting."""

    def setUp(self) -> None:
        super().setUp()
        self.home = self.root / "home"
        self.home.mkdir()
        self.blocked_home = self.root / "h-blocked"
        self.blocked_home.write_text("nope", encoding="utf-8")

    def _break_everything(self) -> None:
        def block(name: str) -> Path:
            blocker = self.root / name
            blocker.write_text("not a directory", encoding="utf-8")
            return blocker

        self.env(LOCALAPPDATA=block("b1"), TEMP=block("b2"), TMP=block("b3"))

    def test_load_settings_never_raises_without_a_usable_location(self) -> None:
        from app.core import settings

        self._break_everything()
        with mock.patch.object(Path, "home", classmethod(lambda cls: self.blocked_home)):
            self.assertEqual(settings.load_settings(), settings.DEFAULT_SETTINGS)

    def test_save_settings_raises_a_catchable_os_error(self) -> None:
        from app.core import settings

        self._break_everything()
        with mock.patch.object(Path, "home", classmethod(lambda cls: self.blocked_home)):
            with self.assertRaises(OSError):
                settings.save_settings({"engine": "spectral"})

    def test_settings_round_trip_through_the_data_directory(self) -> None:
        from app.core import settings

        self.env(LOCALAPPDATA=self.root)
        settings.save_settings({"engine": "spectral"})
        self.assertEqual(settings.load_settings()["engine"], "spectral")
        self.assertEqual(settings.settings_path(), self.root / "BambuPalette" / "settings.json")

    def test_an_unknown_engine_id_is_ignored(self) -> None:
        from app.core import settings

        self.env(LOCALAPPDATA=self.root)
        (self.root / "BambuPalette").mkdir()
        (self.root / "BambuPalette" / "settings.json").write_text(
            '{"engine": "telepathy"}', encoding="utf-8"
        )
        self.assertEqual(settings.load_settings()["engine"], settings.DEFAULT_SETTINGS["engine"])


class SettingsMigrationTests(DataDirTestCase):
    """A stored engine id has to survive the default moving underneath it.

    The id ``"bambu"`` used to name the sRGB average AND was the shipped default,
    so a file without a version cannot be told apart from "never touched".  Such
    a file is migrated forward; a file that records the current version is obeyed
    exactly, so an explicit choice of the legacy engine is never overridden.
    """

    def setUp(self) -> None:
        super().setUp()
        self.env(LOCALAPPDATA=self.root)
        (self.root / "BambuPalette").mkdir()

    def _write(self, payload: str) -> None:
        (self.root / "BambuPalette" / "settings.json").write_text(payload, encoding="utf-8")

    def _stored(self) -> str:
        return (self.root / "BambuPalette" / "settings.json").read_text(encoding="utf-8")

    def test_a_pre_version_default_is_upgraded_to_the_new_default(self) -> None:
        from app.core.engines import DEFAULT_ENGINE
        from app.core import settings

        self._write('{"engine": "bambu"}')
        self.assertEqual(settings.load_settings()["engine"], DEFAULT_ENGINE)

    def test_a_pre_version_explicit_choice_is_kept(self) -> None:
        from app.core import settings

        self._write('{"engine": "spectral"}')
        self.assertEqual(settings.load_settings()["engine"], "spectral")

    def test_a_versioned_bambu_choice_is_kept(self) -> None:
        from app.core import settings

        self._write('{"engine": "bambu", "version": 2}')
        self.assertEqual(settings.load_settings()["engine"], "bambu")

    def test_a_pre_version_default_written_before_a_real_choice_is_not_reapplied(self) -> None:
        from app.core import settings

        self._write('{"engine": "bambu"}')
        settings.save_settings({})  # a launch that just persists the migration
        self.assertEqual(settings.load_settings()["engine"], settings.DEFAULT_SETTINGS["engine"])
        self.assertIn('"version": 2', self._stored())

    def test_saving_records_the_current_version(self) -> None:
        from app.core import settings

        settings.save_settings({"engine": "spectral"})
        raw = json.loads(self._stored())
        self.assertEqual(raw["version"], settings.SETTINGS_VERSION)
        self.assertEqual(raw["engine"], "spectral")

    def test_a_legacy_engine_alias_still_resolves(self) -> None:
        from app.core import settings

        self._write('{"engine": "srgb", "version": 2}')
        self.assertEqual(settings.load_settings()["engine"], "bambu")

    def test_a_missing_engine_falls_back_to_the_default(self) -> None:
        from app.core import settings

        self._write('{"version": 1}')
        self.assertEqual(settings.load_settings()["engine"], settings.DEFAULT_SETTINGS["engine"])

    def test_a_corrupt_file_is_treated_as_no_preferences(self) -> None:
        from app.core import settings

        self._write("{ not json at all")
        self.assertEqual(settings.load_settings(), settings.DEFAULT_SETTINGS)

    def test_a_non_object_file_is_treated_as_no_preferences(self) -> None:
        from app.core import settings

        self._write('[1, 2, 3]')
        self.assertEqual(settings.load_settings(), settings.DEFAULT_SETTINGS)


class CopyMissingTests(DataDirTestCase):
    def test_copies_only_the_missing_named_files(self) -> None:
        source = self.root / "source"
        target = self.root / "target"
        source.mkdir()
        (source / "filament-library.json").write_text("source", encoding="utf-8")
        (source / "settings.json").write_text("source", encoding="utf-8")
        (source / "notes.txt").write_text("source", encoding="utf-8")
        target.mkdir()
        (target / "filament-library.json").write_text("target", encoding="utf-8")

        paths._copy_missing(source, target, names=paths._SEED_FILES)

        self.assertEqual((target / "filament-library.json").read_text(encoding="utf-8"), "target")
        self.assertEqual((target / "settings.json").read_text(encoding="utf-8"), "source")
        self.assertFalse((target / "notes.txt").exists())

    def test_copies_everything_when_no_names_are_given(self) -> None:
        source = self.root / "source"
        target = self.root / "target"
        (source / "nested").mkdir(parents=True)
        (source / "nested" / "deep.txt").write_text("deep", encoding="utf-8")

        paths._copy_missing(source, target)

        self.assertEqual((target / "nested" / "deep.txt").read_text(encoding="utf-8"), "deep")


class DataDirOverrideTests(DataDirTestCase):
    """``BAMBU_PALETTE_DATA_DIR`` is the seatbelt for every script and experiment.

    A screenshot script that builds a bare ``MainWindow()`` saves its library on
    construction.  Without an override it saves over the user's real spools —
    which is not hypothetical, it happened, and the user found demo colours in
    their library the next time they opened the program.
    """

    def test_the_override_is_the_only_root(self) -> None:
        scratch = self.root / "scratch"
        self.env(BAMBU_PALETTE_DATA_DIR=str(scratch), LOCALAPPDATA=self.root / "profile")

        self.assertEqual(paths.data_dir(), scratch)
        self.assertEqual(paths.library_path(), scratch / "filament-library.json")

    def test_the_override_beats_a_library_in_the_profile(self) -> None:
        """Even a populated profile directory must not win over explicit intent."""
        profile = self.root / "profile" / "BambuPalette"
        profile.mkdir(parents=True)
        (profile / "filament-library.json").write_text("{}", encoding="utf-8")

        scratch = self.root / "scratch"
        self.env(BAMBU_PALETTE_DATA_DIR=str(scratch), LOCALAPPDATA=self.root / "profile")

        self.assertEqual(paths.data_dir(), scratch)

    def test_an_empty_override_is_ignored(self) -> None:
        self.env(BAMBU_PALETTE_DATA_DIR="")
        self.assertIsNone(paths.data_dir_override())

    def test_the_override_is_expanded(self) -> None:
        self.env(BAMBU_PALETTE_DATA_DIR="~/bambupalette-scratch")
        override = paths.data_dir_override()
        self.assertIsNotNone(override)
        self.assertNotIn("~", str(override))


class PortableModeTests(DataDirTestCase):
    def test_portable_flag_keeps_data_beside_the_program(self) -> None:
        program = self.root / "program"
        program.mkdir()
        self.env(BAMBU_PALETTE_PORTABLE="1", TEMP=self.root / "scratch")

        with mock.patch.object(paths, "executable_dir", lambda: program):
            resolved = paths.data_dir()

        self.assertEqual(resolved, program / "data")

    def test_portable_file_beside_the_program_enables_portable_mode(self) -> None:
        program = self.root / "program"
        program.mkdir()
        (program / "portable.txt").write_text("", encoding="utf-8")
        os.environ.pop("BAMBU_PALETTE_PORTABLE", None)
        os.environ.pop("FILAMENT_STUDIO_PORTABLE", None)

        with mock.patch.object(paths, "executable_dir", lambda: program):
            self.assertTrue(paths.portable_mode())
            self.assertEqual(paths.data_dir(), program / "data")

    def test_the_old_environment_variable_is_still_honoured(self) -> None:
        self.env(FILAMENT_STUDIO_PORTABLE="yes")
        os.environ.pop("BAMBU_PALETTE_PORTABLE", None)
        self.assertTrue(paths.portable_mode())

    def test_an_explicit_no_wins_over_the_old_variable(self) -> None:
        self.env(BAMBU_PALETTE_PORTABLE="0", FILAMENT_STUDIO_PORTABLE="1")
        with mock.patch.object(paths, "executable_dir", lambda: self.root):
            self.assertFalse(paths.portable_mode())


class SteadyStateRootTests(DataDirTestCase):
    """A library in a fallback must not be orphaned when the preferred root recovers.

    This is the double-click scenario seen from the other side: the packaged
    application ran at a lower integrity level than the user profile, so it
    stored its spools beside the executable.  Once the preferred directory
    accepts writes again, silently switching back to it would look exactly like
    the user's library had disappeared.
    """

    def _library_at(self, directory: Path, stamp: float) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "filament-library.json"
        target.write_text('{"filaments": []}', encoding="utf-8")
        os.utime(target, (stamp, stamp))
        return target

    def test_a_library_in_the_fallback_beats_an_empty_preferred_root(self) -> None:
        local = self.root / "local"
        (local / "BambuPalette").mkdir(parents=True)
        scratch = self.root / "scratch"
        fallback = scratch / "BambuPalette"
        self._library_at(fallback, 1_000_000)
        self.env(LOCALAPPDATA=local, TEMP=scratch, TMP=scratch)

        self.assertEqual(paths.data_dir(), fallback)

    def test_the_newer_library_wins(self) -> None:
        local = self.root / "local"
        self._library_at(local / "BambuPalette", 1_000_000)
        scratch = self.root / "scratch"
        fallback = scratch / "BambuPalette"
        self._library_at(fallback, 2_000_000)
        self.env(LOCALAPPDATA=local, TEMP=scratch, TMP=scratch)

        self.assertEqual(paths.data_dir(), fallback)

    def test_the_preferred_root_wins_when_its_library_is_newer(self) -> None:
        local = self.root / "local"
        preferred = local / "BambuPalette"
        self._library_at(preferred, 2_000_000)
        scratch = self.root / "scratch"
        self._library_at(scratch / "BambuPalette", 1_000_000)
        self.env(LOCALAPPDATA=local, TEMP=scratch, TMP=scratch)

        self.assertEqual(paths.data_dir(), preferred)

    def test_portable_mode_ignores_a_library_in_the_profile(self) -> None:
        local = self.root / "local"
        self._library_at(local / "BambuPalette", 2_000_000)
        program = self.root / "program"
        program.mkdir()
        scratch = self.root / "scratch"
        scratch.mkdir()
        self.env(LOCALAPPDATA=local, TEMP=scratch, TMP=scratch, BAMBU_PALETTE_PORTABLE="1")

        with mock.patch.object(paths, "executable_dir", return_value=program):
            self.assertEqual(paths.data_dir(), program / "data")

    def test_a_populated_but_unwritable_root_is_not_used(self) -> None:
        local = self.root / "local"
        preferred = local / "BambuPalette"
        self._library_at(preferred, 2_000_000)
        scratch = self.root / "scratch"
        fallback = scratch / "BambuPalette"
        self._library_at(fallback, 1_000_000)
        self.env(LOCALAPPDATA=local, TEMP=scratch, TMP=scratch)

        def probe(directory: Path, attempts: int = 4) -> str | None:
            return "拒绝访问。" if directory.parent == local else None

        with mock.patch.object(paths, "_write_probe", probe):
            self.assertEqual(paths.data_dir(), fallback)


class WriteProbeTests(DataDirTestCase):
    """Existence is not writability — the check that stops the startup hang.

    ``mkdir(exist_ok=True)`` succeeds on a directory that already exists without
    needing write permission, so the old code accepted a directory it could not
    put a file into.  Windows then answers "access denied" to every create — a
    mandatory integrity label, a directory left "delete pending" by an
    interrupted delete, an anti-virus lock — and the library save retried that
    refusal up to ``TMP_MAX`` (2147483647) times.  That presents as the
    application never opening, which is exactly what a user reported.
    """

    def test_a_writable_directory_passes(self) -> None:
        target = self.root / "writable"
        target.mkdir()

        self.assertIsNone(paths._write_probe(target))

    def test_the_probe_leaves_no_file_behind(self) -> None:
        target = self.root / "writable"
        target.mkdir()

        paths._write_probe(target)

        self.assertEqual(list(target.iterdir()), [])

    def test_a_directory_that_refuses_new_files_reports_a_reason(self) -> None:
        target = self.root / "refusing"
        target.mkdir()

        with mock.patch.object(
            paths, "open", create=True, side_effect=PermissionError(13, "拒绝访问。")
        ):
            reason = paths._write_probe(target)

        self.assertIsNotNone(reason)
        self.assertIn("拒绝访问", reason)

    def test_a_missing_directory_reports_a_reason(self) -> None:
        self.assertIsNotNone(paths._write_probe(self.root / "absent"))


class UnwritableRootTests(DataDirTestCase):
    """A root that exists but cannot accept files must be skipped, not returned."""

    def _blocked_local(self) -> Path:
        local = self.root / "local"
        (local / "BambuPalette").mkdir(parents=True)
        return local

    def _refuse_only(self, local: Path):
        def probe(directory: Path, attempts: int = 4) -> str | None:
            return "拒绝访问。" if directory.parent == local else None

        return mock.patch.object(paths, "_write_probe", probe)

    def test_an_existing_but_unwritable_root_is_skipped(self) -> None:
        local = self._blocked_local()
        scratch = self.root / "scratch"
        scratch.mkdir()
        self.env(LOCALAPPDATA=local, TEMP=scratch, TMP=scratch)

        with self._refuse_only(local):
            resolved = paths.data_dir()

        self.assertEqual(resolved, scratch / "BambuPalette")

    def test_the_library_path_still_resolves_when_the_preferred_root_refuses(self) -> None:
        local = self._blocked_local()
        scratch = self.root / "scratch"
        scratch.mkdir()
        self.env(LOCALAPPDATA=local, TEMP=scratch, TMP=scratch)

        with self._refuse_only(local):
            self.assertEqual(
                paths.library_path(),
                scratch / "BambuPalette" / "filament-library.json",
            )

    def test_the_refusal_reason_is_reported_when_nothing_is_writable(self) -> None:
        local = self._blocked_local()
        scratch = self.root / "scratch"
        scratch.mkdir()
        home = self.root / "home"
        home.mkdir()
        self.env(LOCALAPPDATA=local, TEMP=scratch, TMP=scratch)

        with mock.patch.object(paths, "_write_probe", lambda d, attempts=4: "拒绝访问。"):
            with mock.patch.object(Path, "home", classmethod(lambda cls: home)):
                with self.assertRaises(paths.DataDirectoryError) as caught:
                    paths.data_dir()

        self.assertIn("无法写入", str(caught.exception))


class ScratchDirTests(DataDirTestCase):
    """``mkdtemp`` retries a refusal forever; :func:`paths.scratch_dir` does not."""

    def test_creates_a_fresh_directory_each_time(self) -> None:
        first = paths.scratch_dir("fcs-test-")
        second = paths.scratch_dir("fcs-test-")
        self.addCleanup(lambda: [path.rmdir() for path in (first, second)])

        self.assertTrue(first.is_dir())
        self.assertTrue(second.is_dir())
        self.assertNotEqual(first, second)

    def test_a_refused_creation_raises_instead_of_retrying(self) -> None:
        calls: list[str] = []

        def refuse(*args, **kwargs):
            calls.append(str(args[0]))
            raise PermissionError(13, "拒绝访问。")

        with mock.patch.object(Path, "mkdir", refuse):
            with self.assertRaises(paths.DataDirectoryError):
                paths.scratch_dir("fcs-test-")

        self.assertEqual(len(calls), 1)


class LibrarySaveTests(unittest.TestCase):
    """The atomic save must fail fast rather than spin."""

    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory(prefix="fcs-save-")
        self.root = Path(self._temp.name)
        self.addCleanup(self._temp.cleanup)

    def test_a_refused_save_raises_a_library_error(self) -> None:
        from app.core import library as lib

        calls: list[str] = []

        def refuse(*args, **kwargs):
            calls.append(str(args[0]))
            raise PermissionError(13, "拒绝访问。")

        with mock.patch.object(lib, "open", create=True, side_effect=refuse):
            with self.assertRaises(lib.LibraryError) as caught:
                lib.FilamentLibrary([]).save(self.root / "filament-library.json")

        # The refusal is reported, not retried: the old tempfile.mkstemp path
        # treated exactly this PermissionError as a name collision and tried
        # TMP_MAX = 2147483647 times, which is what hung a plain double-click.
        self.assertIn("拒绝访问", str(caught.exception))
        self.assertEqual(len(calls), 1)

    def test_a_name_collision_retries_a_bounded_number_of_times(self) -> None:
        from app.core import library as lib

        calls: list[str] = []

        def collide(*args, **kwargs):
            calls.append(str(args[0]))
            raise FileExistsError(17, "文件已存在。")

        with mock.patch.object(lib, "open", create=True, side_effect=collide):
            with self.assertRaises(lib.LibraryError) as caught:
                lib.FilamentLibrary([]).save(self.root / "filament-library.json")

        self.assertIn("临时文件名重复", str(caught.exception))
        self.assertEqual(len(calls), 8)

    def test_each_retry_uses_a_different_temporary_name(self) -> None:
        from app.core import library as lib

        names: list[str] = []

        def collide(args_0, *args, **kwargs):
            names.append(args_0)
            raise FileExistsError(17, "文件已存在。")

        with mock.patch.object(lib, "open", create=True, side_effect=collide):
            with self.assertRaises(lib.LibraryError):
                lib.FilamentLibrary([]).save(self.root / "filament-library.json")

        self.assertEqual(len(set(names)), 8)
        self.assertTrue(all(str(name).endswith(".tmp") for name in names))

    def test_a_successful_save_leaves_only_the_library_behind(self) -> None:
        from app.core import library as lib

        target = self.root / "filament-library.json"
        lib.FilamentLibrary([]).save(target)

        self.assertEqual([path.name for path in self.root.iterdir()], [target.name])
        self.assertTrue(target.read_text(encoding="utf-8").startswith("{"))

    def test_the_next_save_reuses_no_stale_temporary_name(self) -> None:
        from app.core import library as lib

        target = self.root / "filament-library.json"
        lib.FilamentLibrary([]).save(target)
        lib.FilamentLibrary([]).save(target)

        # The second save keeps one generation of undo beside the library; what
        # must not survive is a stray ``.tmp`` from either write.
        self.assertEqual(
            sorted(path.name for path in self.root.iterdir()),
            [target.name, target.name + ".bak"],
        )

    def test_a_save_keeps_the_previous_version_as_a_backup(self) -> None:
        """An accident must cost one generation, not the whole library."""
        from app.core import library as lib
        from app.core.library import Filament

        target = self.root / "filament-library.json"
        original = lib.FilamentLibrary(
            [Filament(color_hex="#123456", material_type="PLA")]
        )
        original.save(target)

        # Something replaces the library wholesale, the way a stray script once
        # did. The old file has to still be readable.
        lib.FilamentLibrary([Filament(color_hex="#FFFFFF")]).save(target)

        backup = target.with_name(target.name + ".bak")
        self.assertTrue(backup.exists())
        restored = lib.FilamentLibrary.load(backup)
        self.assertEqual([f.color_hex for f in restored], ["#123456"])

    def test_the_first_save_creates_no_backup(self) -> None:
        """There is nothing to back up yet, and an empty ``.bak`` would lie."""
        from app.core import library as lib

        target = self.root / "filament-library.json"
        lib.FilamentLibrary([]).save(target)

        self.assertFalse(target.with_name(target.name + ".bak").exists())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

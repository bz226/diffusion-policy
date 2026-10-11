"""Stage3 source checks retain their original history after metadata archival."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


CODE = Path(__file__).resolve().parents[1] / "results/stage3/code"
SPEC = importlib.util.spec_from_file_location("stage3_source_git", CODE / "source_git.py")
SOURCE_GIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SOURCE_GIT)
SPEC = importlib.util.spec_from_file_location("stage3_archive_monitor", CODE / "monitor.py")
MONITOR = importlib.util.module_from_spec(SPEC)
with mock.patch.dict(sys.modules, {"source_git": SOURCE_GIT}):
    SPEC.loader.exec_module(MONITOR)


class Stage3SourceGitTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="stage3-source-git-test-", dir="/tmp")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "project"
        self.repo = self.root / "results/stage3/dppo"
        self.repo.mkdir(parents=True)
        self.env = os.environ.copy()
        for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR",
                    "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_CONFIG_COUNT"):
            self.env.pop(key, None)
        self.env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1",
                        GIT_TERMINAL_PROMPT="0", LC_ALL="C")
        self.init_repo(self.root)
        self.git(self.root, "commit", "-qm", "Parent history", "--allow-empty")
        self.parent_head = self.git(self.root, "rev-parse", "HEAD").stdout.strip()
        self.init_repo(self.repo)
        (self.repo / "source.py").write_text("VALUE = 1\n")
        self.git(self.repo, "add", "source.py")
        self.git(self.repo, "commit", "-qm", "Original source")
        self.base = self.git(self.repo, "rev-parse", "HEAD").stdout.strip()
        (self.repo / "source.py").write_text("VALUE = 2\n")
        self.git(self.repo, "commit", "-qam", "Scientific source changes")
        self.head = self.git(self.repo, "rev-parse", "HEAD").stdout.strip()
        self.record_path = self.repo.parent / "provenance/dppo_git_archive.json"
        self.archive = self.root / ".codex-runtime/git-backups/stage3-dppo/.git"

    def run_command(self, command, check=True):
        return subprocess.run(command, env=self.env, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, check=check)

    def git(self, repo, *args):
        return self.run_command(["git", "-C", str(repo), *args])

    def init_repo(self, repo):
        self.git(repo, "init", "-q", "--initial-branch=main")
        self.git(repo, "config", "user.name", "Stage3 Source History Test")
        self.git(repo, "config", "user.email", "source-test@example.invalid")
        self.git(repo, "config", "core.autocrlf", "false")

    def move_metadata(self, write_record=True):
        self.archive.parent.mkdir(parents=True)
        (self.repo / ".git").rename(self.archive)
        if write_record:
            self.record_path.parent.mkdir(parents=True)
            self.record_path.write_text(json.dumps({
                "source_directory": self.repo.relative_to(self.root).as_posix(),
                "archived_git_directory": self.archive.relative_to(self.root).as_posix(),
                "commit": self.head,
                "original_base_commit": self.base,
            }))

    def source(self, *args, check=True):
        return self.run_command(SOURCE_GIT.source_git_command(self.repo, *args), check=check)

    def validate_source(self, commit):
        with mock.patch.object(MONITOR, "SOURCE", self.repo), mock.patch.dict(os.environ, self.env, clear=True):
            return MONITOR.validate_source(commit)

    def test_live_checkout_needs_no_archive_record(self):
        self.assertFalse(self.record_path.exists())
        self.assertEqual(self.source("rev-parse", "HEAD").stdout.strip(), self.head)
        self.assertNotEqual(self.head, self.parent_head)
        self.assertEqual(self.source("status", "--porcelain").stdout, "")

    def test_archived_history_preserves_head_show_and_historical_diff(self):
        self.move_metadata()
        self.assertFalse((self.repo / ".git").exists())
        self.assertEqual(self.source("rev-parse", "HEAD").stdout.strip(), self.head)
        self.assertEqual(self.source("show", self.base + ":source.py").stdout, "VALUE = 1\n")
        self.assertEqual(self.source("diff", "--name-only", self.base, "HEAD", "--").stdout,
                         "source.py\n")
        self.assertEqual(self.source("status", "--porcelain").stdout, "")

    def test_archived_history_detects_changed_source(self):
        self.move_metadata()
        (self.repo / "source.py").write_text("VALUE = 3\n")
        self.assertEqual(self.source("diff", "--exit-code", "HEAD", "--", check=False).returncode, 1)
        self.assertIn(" M source.py", self.source("status", "--porcelain").stdout)

    def test_missing_record_does_not_use_parent_repository(self):
        self.move_metadata(write_record=False)
        self.assertEqual(self.git(self.repo, "rev-parse", "HEAD").stdout.strip(), self.parent_head)
        with self.assertRaisesRegex(RuntimeError, "archive record missing"):
            SOURCE_GIT.source_git_command(self.repo, "rev-parse", "HEAD")

    def test_missing_backup_fails_clearly(self):
        self.move_metadata()
        self.archive.rename(self.archive.with_name("moved-backup"))
        with self.assertRaisesRegex(RuntimeError, "Archived DPPO Git metadata is unavailable"):
            SOURCE_GIT.source_git_command(self.repo, "rev-parse", "HEAD")

    def test_custom_repo_cannot_use_another_sources_archive(self):
        self.move_metadata()
        other_repo = self.repo.with_name("custom_dppo")
        other_repo.mkdir()
        with self.assertRaisesRegex(RuntimeError, "not requested source"):
            SOURCE_GIT.source_git_command(other_repo, "rev-parse", "HEAD")

    def test_live_git_pointer_file_is_supported(self):
        self.move_metadata(write_record=False)
        (self.repo / ".git").write_text("gitdir: {}\n".format(self.archive))
        self.assertEqual(self.source("rev-parse", "HEAD").stdout.strip(), self.head)
        self.assertEqual(self.source("status", "--porcelain").stdout, "")

    def test_archive_record_paths_must_stay_inside_project(self):
        self.move_metadata()
        record = json.loads(self.record_path.read_text())
        for path in (str(self.archive), "../outside/.git"):
            with self.subTest(path=path):
                record["archived_git_directory"] = path
                self.record_path.write_text(json.dumps(record))
                with self.assertRaisesRegex(RuntimeError, "Invalid DPPO Git archive record"):
                    SOURCE_GIT.source_git_command(self.repo, "rev-parse", "HEAD")

    def test_monitor_accepts_same_source_before_and_after_archive(self):
        expected = {"commit": self.head, "clean": True, "path": str(self.repo)}
        self.assertEqual(self.validate_source(self.head), expected)
        self.move_metadata()
        self.assertEqual(self.validate_source(self.head), expected)

    def test_monitor_rejects_wrong_commit_after_archive(self):
        self.move_metadata()
        with self.assertRaisesRegex(ValueError, "source differs"):
            self.validate_source(self.base)

    def test_monitor_rejects_modified_and_untracked_source_after_archive(self):
        self.move_metadata()
        for filename in ("source.py", "untracked.py"):
            with self.subTest(filename=filename):
                source = self.repo / filename
                original = source.read_text() if source.exists() else None
                source.write_text("VALUE = 3\n")
                with self.assertRaisesRegex(ValueError, "source differs"):
                    self.validate_source(self.head)
                if original is None:
                    source.unlink()
                else:
                    source.write_text(original)

    def test_monitor_fails_without_history_instead_of_using_parent(self):
        self.move_metadata(write_record=False)
        with self.assertRaisesRegex(RuntimeError, "archive record missing"):
            self.validate_source(self.parent_head)


if __name__ == "__main__":
    unittest.main()

"""A new study starts as ordinary, versionable source without nested Git state."""

import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest


HELPER = Path(__file__).resolve().parents[1] / "scripts/create_dppo_snapshot.py"


class CreateDppoSnapshotTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="dppo-snapshot-test-", dir="/tmp")
        self.addCleanup(temporary.cleanup)
        self.temporary = Path(temporary.name)
        self.root = self.temporary / "project"
        self.source = self.temporary / "upstream"
        self.root.mkdir()
        self.source.mkdir()
        self.env = os.environ.copy()
        for key in (
            "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR",
            "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES",
            "GIT_CONFIG_COUNT",
        ):
            self.env.pop(key, None)
        self.env.update(
            GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1",
            GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0", LC_ALL="C",
        )
        self.init_repo(self.root)
        self.init_repo(self.source)
        (self.root / "scripts").mkdir()
        shutil.copy2(HELPER, self.root / "scripts/create_dppo_snapshot.py")
        (self.root / ".gitignore").write_text(
            "**/checkpoint/\n**/checkpoints/\n*.pt\n*.pth\n*.ckpt\n"
            "**/data/\n**/datasets/\n**/cache/\n/.codex-runtime/\n",
            encoding="utf-8",
        )
        self.git(self.root, "add", ".gitignore", "scripts/create_dppo_snapshot.py")
        self.git(self.root, "commit", "-qm", "Parent repository")
        (self.source / "module").mkdir()
        self.source_files = {
            "source.py": b"VALUE = 1\n",
            ".hidden-source": b"hidden source\n",
            "space and\nnewline.py": b"VALUE = 2\n",
            "setup.sh": b"#!/bin/sh\ntrue\n",
            "module/important.py": b"VALUE = 3\n",
            ".gitignore": b"*.sh\n*.pkl\n*.log\nlog/\n",
            "module/.gitignore": b"important.py\n",
        }
        for name, contents in self.source_files.items():
            (self.source / name).write_bytes(contents)
        (self.source / "setup.sh").chmod(0o755)
        (self.source / "source-link").symlink_to("source.py")
        self.git(self.source, "add", "-f", "--", ".")
        self.git(self.source, "commit", "-qm", "Source revision")
        self.git(self.source, "remote", "add", "origin", "https://example.invalid/dppo.git")
        self.revision = self.git(self.source, "rev-parse", "HEAD").stdout.decode().strip()
        self.destination = self.root / "results/stage_next/dppo"

    def git(self, repo, *args):
        result = subprocess.run(
            ["git", "-C", str(repo), *args], env=self.env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if result.returncode:
            self.fail("git {} failed: {}".format(args, result.stderr.decode(errors="replace")))
        return result

    def init_repo(self, path):
        self.git(path, "init", "-q", "--initial-branch=main")
        self.git(path, "config", "user.name", "Snapshot Test")
        self.git(path, "config", "user.email", "snapshot@example.invalid")
        self.git(path, "config", "core.fileMode", "true")
        self.git(path, "config", "core.autocrlf", "false")

    def run_helper(self, source=None, study="stage_next", revision=None, subdir=None, check=True):
        command = [sys.executable, str(self.root / "scripts/create_dppo_snapshot.py"),
                   "--source", str(source or self.source),
                   "--revision", revision or self.revision, "--study", study]
        if subdir is not None:
            command.extend(["--subdir", subdir])
        result = subprocess.run(
            command,
            cwd=self.root / "scripts", env=self.env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if check and result.returncode:
            self.fail("snapshot helper failed: {}{}".format(
                result.stdout.decode(errors="replace"), result.stderr.decode(errors="replace")))
        return result

    def tree_snapshot(self, root):
        entries = {}
        for path in root.rglob("*"):
            name = path.relative_to(root).as_posix()
            mode = stat.S_IMODE(path.lstat().st_mode)
            if path.is_symlink():
                entries[name] = ("symlink", mode, os.readlink(path))
            elif path.is_file():
                entries[name] = ("file", mode, path.read_bytes())
            else:
                entries[name] = ("directory", mode)
        return entries

    def assert_rejected_without_mutation(self, **kwargs):
        before = self.tree_snapshot(self.temporary)
        result = self.run_helper(check=False, **kwargs)
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue((result.stdout + result.stderr).strip())
        self.assertEqual(self.tree_snapshot(self.temporary), before)
        return result

    def index_entries(self):
        entries = {}
        for entry in self.git(self.root, "ls-files", "--stage", "-z").stdout.split(b"\0"):
            if entry:
                metadata, name = entry.split(b"\t", 1)
                entries[os.fsdecode(name)] = metadata.decode().split()[0]
        return entries

    def test_exports_complete_source_and_preserves_source_and_parent_metadata(self):
        source_before = self.tree_snapshot(self.source)
        parent_git_before = self.tree_snapshot(self.root / ".git")

        self.run_helper()

        self.assertEqual(self.tree_snapshot(self.source), source_before)
        self.assertEqual(self.tree_snapshot(self.root / ".git"), parent_git_before)
        self.assertFalse(any(path.name == ".git" for path in self.destination.rglob("*")))
        for name, contents in self.source_files.items():
            if Path(name).name != ".gitignore":
                self.assertEqual((self.destination / name).read_bytes(), contents)
        self.assertTrue((self.destination / "setup.sh").stat().st_mode & stat.S_IXUSR)
        self.assertTrue((self.destination / "source-link").is_symlink())
        self.assertEqual(os.readlink(self.destination / "source-link"), "source.py")
        self.assertFalse((self.destination / "module/.gitignore").exists())
        provenance = self.destination.parent / "provenance"
        record = json.loads((provenance / "dppo_source_snapshot.json").read_text())
        self.assertEqual(record["format_version"], 1)
        self.assertEqual(record["commit"], self.revision)
        self.assertEqual(record["source_file_count"], 8)
        self.assertEqual(record["snapshot_file_count"], 7)
        self.assertIn("https://example.invalid/dppo.git", json.dumps(record))
        original_ignores = {entry["source_path"]: entry["saved_path"]
                            for entry in record["upstream_gitignores"]}
        self.assertEqual(set(original_ignores), {".gitignore", "module/.gitignore"})
        for name, saved_path in original_ignores.items():
            self.assertEqual(saved_path, "provenance/upstream_gitignores/" + name + ".txt")
            self.assertEqual((self.destination.parent / saved_path).read_bytes(),
                             self.source_files[name])

    def test_plain_staging_includes_source_and_selected_results_with_parent_exclusions(self):
        self.run_helper()
        run = self.destination / "log/gym-finetune/run1"
        (run / ".hydra").mkdir(parents=True)
        selected = ["result.pkl", "run.log", ".hydra/config.yaml",
                    ".hydra/hydra.yaml", ".hydra/overrides.yaml"]
        for name in selected:
            (run / name).write_text("small selected result\n", encoding="utf-8")
        excluded = ["checkpoint/model.bin", "checkpoints/model.bin", "data/input.bin",
                    "cache/cache.bin", "__pycache__/module.pyc", "weights.pt"]
        for name in excluded:
            path = self.destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"generated artifact\n")
        (self.destination / "parent-excluded.py").write_text("omitted\n", encoding="utf-8")
        with (self.root / ".gitignore").open("a", encoding="utf-8") as stream:
            stream.write("/results/stage_next/dppo/parent-excluded.py\n")

        result = self.git(self.root, "add", "-A")

        self.assertNotIn(b"embedded git repository", result.stderr)
        entries = self.index_entries()
        prefix = self.destination.relative_to(self.root).as_posix() + "/"
        for name in self.source_files:
            if Path(name).name != ".gitignore":
                self.assertIn(prefix + name, entries)
        self.assertEqual(entries[prefix + "setup.sh"], "100755")
        self.assertEqual(entries[prefix + "source-link"], "120000")
        for name in selected:
            self.assertIn(prefix + "log/gym-finetune/run1/" + name, entries)
        for name in excluded + ["parent-excluded.py"]:
            self.assertNotIn(prefix + name, entries)
        self.assertNotIn("160000", entries.values())
        self.assertFalse(any("/.git/" in name for name in entries))

    def test_accepts_archived_git_metadata_without_source_worktree(self):
        archive = self.temporary / "archived-history"
        (self.source / ".git").rename(archive)
        shutil.rmtree(self.source)
        before = self.tree_snapshot(archive)

        self.run_helper(source=archive)

        self.assertEqual((self.destination / "source.py").read_bytes(), b"VALUE = 1\n")
        self.assertFalse((self.destination / ".git").exists())
        self.assertEqual(self.tree_snapshot(archive), before)

    def test_exports_requested_revision_instead_of_current_head(self):
        (self.source / "source.py").write_text("VALUE = 99\n", encoding="utf-8")
        self.git(self.source, "commit", "-qam", "Later source")

        self.run_helper(revision=self.revision)

        self.assertEqual((self.destination / "source.py").read_bytes(), b"VALUE = 1\n")

    def test_keeps_tracked_source_even_when_upstream_marks_it_export_ignore(self):
        (self.source / ".gitattributes").write_text("source.py export-ignore\n", encoding="utf-8")
        self.git(self.source, "add", ".gitattributes")
        self.git(self.source, "commit", "-qm", "Upstream archive exclusions")
        revision = self.git(self.source, "rev-parse", "HEAD").stdout.decode().strip()

        self.run_helper(revision=revision)

        self.assertEqual((self.destination / "source.py").read_bytes(), b"VALUE = 1\n")

    def commit_ordinary_study(self):
        self.run_helper()
        self.git(self.root, "add", "-A")
        self.git(self.root, "commit", "-qm", "Ordinary study source")
        return self.git(self.root, "rev-parse", "HEAD").stdout.decode().strip()

    def test_reuses_committed_ordinary_study_from_parent_subtree(self):
        parent_revision = self.commit_ordinary_study()
        parent_git_before = self.tree_snapshot(self.root / ".git")

        self.run_helper(source=self.root / ".git", revision=parent_revision,
                        study="stage_after", subdir="results/stage_next/dppo")

        destination = self.root / "results/stage_after/dppo"
        self.assertEqual((destination / "source.py").read_bytes(), b"VALUE = 1\n")
        self.assertTrue((destination / "setup.sh").stat().st_mode & stat.S_IXUSR)
        self.assertTrue((destination / "source-link").is_symlink())
        self.assertFalse(any(path.name == ".git" for path in destination.rglob("*")))
        self.assertFalse((destination / "results").exists())
        record = json.loads((destination.parent / "provenance/dppo_source_snapshot.json").read_text())
        self.assertEqual(record["commit"], parent_revision)
        self.assertEqual(record["source_subdirectory"], "results/stage_next/dppo")
        self.assertEqual(self.tree_snapshot(self.root / ".git"), parent_git_before)

    def test_subtree_import_allows_unrelated_uncommitted_parent_changes(self):
        parent_revision = self.commit_ordinary_study()
        with (self.root / ".gitignore").open("a", encoding="utf-8") as stream:
            stream.write("# Unrelated uncommitted parent change\n")
        (self.root / "untracked-notes.txt").write_text("unrelated notes\n", encoding="utf-8")
        parent_git_before = self.tree_snapshot(self.root / ".git")

        self.run_helper(source=self.root, revision=parent_revision,
                        study="stage_after", subdir="results/stage_next/dppo")

        self.assertEqual((self.root / "results/stage_after/dppo/source.py").read_bytes(),
                         b"VALUE = 1\n")
        self.assertEqual(self.tree_snapshot(self.root / ".git"), parent_git_before)
        self.assertIn("Unrelated uncommitted", (self.root / ".gitignore").read_text())

    def test_subtree_import_refuses_relevant_uncommitted_source_changes(self):
        parent_revision = self.commit_ordinary_study()
        (self.destination / "source.py").write_text("VALUE = 99\n", encoding="utf-8")

        result = self.assert_rejected_without_mutation(
            source=self.root, revision=parent_revision, study="stage_after",
            subdir="results/stage_next/dppo")

        self.assertIn(b"uncommitted", result.stderr)

    def test_refuses_existing_destination_without_touching_it(self):
        self.destination.mkdir(parents=True)
        (self.destination / "existing.py").write_text("keep me\n", encoding="utf-8")
        self.assert_rejected_without_mutation()

    def test_refuses_existing_study_even_when_source_directory_is_absent(self):
        self.destination.parent.mkdir(parents=True)
        (self.destination.parent / "methods.md").write_text("existing study\n", encoding="utf-8")
        self.assert_rejected_without_mutation()

    def test_refuses_results_symlink_without_writing_outside_project(self):
        outside = self.temporary / "outside-results"
        outside.mkdir()
        (self.root / "results").symlink_to(outside, target_is_directory=True)
        self.assert_rejected_without_mutation()

    def test_refuses_dirty_source_without_dropping_uncommitted_changes(self):
        (self.source / "source.py").write_text("VALUE = 99\n", encoding="utf-8")
        self.assert_rejected_without_mutation()

    def test_refuses_submodules(self):
        self.git(self.source, "update-index", "--add", "--cacheinfo",
                 "160000,{},vendor/nested".format(self.revision))
        self.git(self.source, "commit", "-qm", "Nested dependency")
        revision = self.git(self.source, "rev-parse", "HEAD").stdout.decode().strip()
        # Read committed metadata directly so an uninitialized submodule's worktree
        # status cannot mask the unsupported tree-entry check.
        self.assert_rejected_without_mutation(source=self.source / ".git", revision=revision)

    def test_refuses_symlinks_escaping_snapshot(self):
        (self.source / "outside-link").symlink_to("../outside.py")
        self.git(self.source, "add", "outside-link")
        self.git(self.source, "commit", "-qm", "Escaping link")
        revision = self.git(self.source, "rev-parse", "HEAD").stdout.decode().strip()
        self.assert_rejected_without_mutation(revision=revision)

    def test_refuses_unsafe_study_names(self):
        for study in ("../outside", "/tmp/escape", "nested/study", ".", ".."):
            with self.subTest(study=study):
                self.assert_rejected_without_mutation(study=study)


if __name__ == "__main__":
    unittest.main()

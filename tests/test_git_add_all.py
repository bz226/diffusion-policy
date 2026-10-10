"""Integration tests for staging result checkouts without changing their Git state."""

import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest


HELPER = Path(__file__).resolve().parents[1] / "scripts" / "git_add_all.py"


class GitAddAllTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="git-add-all-test-", dir="/tmp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "parent"
        self.root.mkdir()
        self.env = os.environ.copy()
        for key in (
            "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR",
            "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES",
            "GIT_CONFIG_COUNT",
        ):
            self.env.pop(key, None)
        self.env.update(
            GIT_CONFIG_GLOBAL=os.devnull,
            GIT_CONFIG_NOSYSTEM="1",
            GIT_TERMINAL_PROMPT="0",
            LC_ALL="C",
        )
        self.init_repo(self.root)
        (self.root / "scripts").mkdir()
        shutil.copy2(HELPER, self.root / "scripts/git_add_all.py")
        (self.root / ".gitignore").write_text(
            "**/checkpoint/\n**/checkpoints/\n*.pt\n*.pth\n*.ckpt\n"
            "**/data/\n**/datasets/\n**/cache/\n",
            encoding="utf-8",
        )
        self.git(self.root, "add", ".gitignore", "scripts/git_add_all.py")
        self.git(self.root, "commit", "-qm", "Initial parent")

    def git(self, repo, *args, data=None, check=True):
        result = subprocess.run(
            ["git", *args], cwd=repo, env=self.env, input=data,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if check and result.returncode:
            self.fail(
                f"git {args!r} failed in {repo}:\n"
                f"{result.stdout.decode(errors='replace')}"
                f"{result.stderr.decode(errors='replace')}"
            )
        return result

    def init_repo(self, repo):
        self.git(repo, "init", "-q", "--initial-branch=main")
        self.git(repo, "config", "user.name", "Integration Test")
        self.git(repo, "config", "user.email", "test@example.invalid")
        self.git(repo, "config", "core.fileMode", "true")
        self.git(repo, "config", "core.autocrlf", "false")

    def make_child(self, name="stage1"):
        child = self.root / "results" / name / "dppo"
        child.mkdir(parents=True)
        self.init_repo(child)
        (child / ".gitignore").write_text(
            "log/\n*.pkl\n*.log\n*.sh\n*.pt\ndata/\ncache/\ncheckpoint/\n",
            encoding="utf-8",
        )
        (child / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
        (child / ".hidden-source").write_text("hidden\n", encoding="utf-8")
        (child / "setup.sh").write_text("#!/bin/sh\ntrue\n", encoding="utf-8")
        (child / "setup.sh").chmod(0o755)
        (child / "source-link").symlink_to("source.py")
        (child / "space and\nnewline.py").write_text("VALUE = 2\n", encoding="utf-8")
        self.git(child, "add", "-f", "--", ".gitignore", "source.py",
                 ".hidden-source", "setup.sh", "source-link", "space and\nnewline.py")
        self.git(child, "commit", "-qm", "Initial source")
        return child

    def make_run(self, child, name="run1"):
        run = child / "log/gym-finetune/halfcheetah-example" / name
        (run / ".hydra").mkdir(parents=True)
        (run / "result.pkl").write_bytes(b"small result fixture\n")
        (run / "run.log").write_text("Training completed\n", encoding="utf-8")
        for name in ("config.yaml", "hydra.yaml", "overrides.yaml"):
            (run / ".hydra" / name).write_text("seed: 1\n", encoding="utf-8")
        (run / "weights.pt").write_bytes(b"ignored weight fixture\n")
        (run / ".hydra/notes.txt").write_text("not allowlisted\n", encoding="utf-8")
        return run

    def rel(self, path):
        return path.relative_to(self.root).as_posix()

    def run_helper(self, *args, check=True):
        result = subprocess.run(
            [sys.executable, str(self.root / "scripts/git_add_all.py"), *args],
            # Deliberately do not invoke from the repository root.
            cwd=self.root / "scripts", env=self.env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if check and result.returncode:
            self.fail(
                f"helper failed ({result.returncode}):\n"
                f"{result.stdout.decode(errors='replace')}"
                f"{result.stderr.decode(errors='replace')}"
            )
        return result

    def index_entries(self):
        entries = {}
        for record in self.git(self.root, "ls-files", "--stage", "-z").stdout.split(b"\0"):
            if record:
                header, path = record.split(b"\t", 1)
                mode, oid, stage = header.decode().split()
                entries[os.fsdecode(path)] = (mode, oid, stage)
        return entries

    def child_state(self, child):
        # Read status before index bytes: status itself can refresh index stat data.
        head = self.git(child, "rev-parse", "HEAD").stdout
        status = self.git(child, "status", "--porcelain=v1", "-z").stdout
        return head, status, (child / ".git/index").read_bytes()

    def assert_child_unchanged(self, child, before):
        self.assertEqual((child / ".git/index").read_bytes(), before[2])
        self.assertEqual(self.child_state(child), before)

    def tree_snapshot(self):
        snapshot = {}
        for path in self.root.rglob("*"):
            relative = path.relative_to(self.root).as_posix()
            mode = stat.S_IMODE(path.lstat().st_mode)
            if path.is_symlink():
                snapshot[relative] = ("symlink", mode, os.readlink(path))
            elif path.is_file():
                snapshot[relative] = ("file", mode, path.read_bytes())
            elif path.is_dir():
                snapshot[relative] = ("directory", mode)
        return snapshot

    def assert_run_staged(self, entries, run):
        for name in ("result.pkl", "run.log", ".hydra/config.yaml",
                     ".hydra/hydra.yaml", ".hydra/overrides.yaml"):
            self.assertEqual(entries[self.rel(run / name)][0], "100644")
        self.assertNotIn(self.rel(run / "weights.pt"), entries)
        self.assertNotIn(self.rel(run / ".hydra/notes.txt"), entries)

    def test_converts_staged_gitlink_without_changing_inner_repository(self):
        child = self.make_child()
        run = self.make_run(child)
        for directory in ("data", "cache", "checkpoint"):
            (child / directory).mkdir()
            (child / directory / "generated.bin").write_bytes(b"excluded\n")
        self.git(self.root, "add", "--", self.rel(child))
        self.assertEqual(self.index_entries()[self.rel(child)][0], "160000")
        before = self.child_state(child)
        self.assertEqual(before[1], b"")

        result = self.run_helper()

        self.assertNotIn(b"adding embedded git repository", result.stderr)
        entries = self.index_entries()
        self.assertNotIn(self.rel(child), entries)
        for name in ("source.py", ".gitignore", ".hidden-source", "space and\nnewline.py"):
            self.assertEqual(entries[self.rel(child / name)][0], "100644")
        self.assertEqual(entries[self.rel(child / "setup.sh")][0], "100755")
        self.assertEqual(entries[self.rel(child / "source-link")][0], "120000")
        self.assert_run_staged(entries, run)
        self.assertFalse(any("/.git/" in path for path in entries))
        for directory in ("data", "cache", "checkpoint"):
            self.assertNotIn(self.rel(child / directory / "generated.bin"), entries)
        self.assert_child_unchanged(child, before)

    def test_reruns_and_discovers_new_runs_and_fresh_checkouts(self):
        child = self.make_child()
        self.make_run(child)
        self.run_helper()
        self.git(self.root, "commit", "-qm", "Snapshot experiment")
        before_entries = self.index_entries()
        self.run_helper()
        self.assertEqual(self.index_entries(), before_entries)
        self.assertEqual(self.git(self.root, "diff", "--cached", "--name-only").stdout, b"")

        new_run = self.make_run(child, "run2")
        (child / "module").mkdir()
        (child / "module/new.py").write_text("NEW = True\n", encoding="utf-8")
        (child / "source.py").write_text("VALUE = 3\n", encoding="utf-8")
        (child / ".hidden-source").unlink()
        fresh = self.make_child("stage2")
        fresh_run = self.make_run(fresh)
        before, fresh_before = self.child_state(child), self.child_state(fresh)

        self.run_helper()

        entries = self.index_entries()
        self.assert_run_staged(entries, new_run)
        self.assert_run_staged(entries, fresh_run)
        self.assertEqual(entries[self.rel(fresh / "setup.sh")][0], "100755")
        self.assertIn(self.rel(child / "module/new.py"), entries)
        self.assertNotIn(self.rel(child / ".hidden-source"), entries)
        self.assertNotEqual(entries[self.rel(child / "source.py")],
                            before_entries[self.rel(child / "source.py")])
        self.assertFalse(any(entry[0] == "160000" for entry in entries.values()))
        self.assert_child_unchanged(child, before)
        self.assert_child_unchanged(fresh, fresh_before)

    def test_dry_run_leaves_indexes_and_all_files_unchanged(self):
        child = self.make_child()
        self.make_run(child)
        self.git(self.root, "add", "--", self.rel(child))
        self.make_run(self.make_child("stage2"))
        before = self.tree_snapshot()

        result = self.run_helper("--dry-run")

        self.assertTrue(result.stdout.strip(), "dry-run should print its plan")
        self.assertEqual(self.tree_snapshot(), before)

    def assert_rejected_without_mutation(self):
        before = self.tree_snapshot()
        result = self.run_helper(check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue((result.stdout + result.stderr).strip())
        self.assertEqual(self.tree_snapshot(), before)

    def prepare_valid_child_before_invalid(self):
        valid = self.make_child("a_valid")
        self.make_run(valid)
        self.git(self.root, "add", "--", self.rel(valid))

    def test_rejects_child_gitlinks_before_any_mutation(self):
        self.prepare_valid_child_before_invalid()
        child = self.make_child("z_invalid")
        commit = self.git(child, "rev-parse", "HEAD").stdout.decode().strip()
        self.git(child, "update-index", "--add", "--cacheinfo",
                 f"160000,{commit},vendor/nested")

        self.assert_rejected_without_mutation()

    def test_rejects_conflicted_child_before_any_mutation(self):
        self.prepare_valid_child_before_invalid()
        child = self.make_child("z_invalid")
        blob = self.git(child, "rev-parse", "HEAD:source.py").stdout.decode().strip()
        self.git(child, "update-index", "--force-remove", "--", "source.py")
        records = "".join(f"100644 {blob} {stage}\tsource.py\n" for stage in (1, 2, 3))
        self.git(child, "update-index", "--index-info", data=records.encode())

        self.assert_rejected_without_mutation()

    def test_rejects_empty_child_before_any_mutation(self):
        self.prepare_valid_child_before_invalid()
        child = self.root / "results/z_invalid/dppo"
        child.mkdir(parents=True)
        self.init_repo(child)

        self.assert_rejected_without_mutation()

    def test_preserves_declared_submodule(self):
        declared = self.make_child("declared")
        (self.root / ".gitmodules").write_text(
            '[submodule "declared"]\n'
            f'\tpath = {self.rel(declared)}\n'
            '\turl = https://example.invalid/dppo.git\n',
            encoding="utf-8",
        )
        self.git(self.root, "add", "--", ".gitmodules", self.rel(declared))
        gitlink = self.index_entries()[self.rel(declared)]
        ordinary = self.make_child("ordinary")
        before = self.child_state(declared)

        self.run_helper()

        entries = self.index_entries()
        self.assertEqual(entries[self.rel(declared)], gitlink)
        self.assertFalse(any(path.startswith(self.rel(declared) + "/") for path in entries))
        self.assertIn(self.rel(ordinary / "source.py"), entries)
        self.assert_child_unchanged(declared, before)

    def test_parent_ignores_override_inner_tracked_files(self):
        child = self.make_child()
        with (self.root / ".gitignore").open("a", encoding="utf-8") as stream:
            stream.write(f"/{self.rel(child)}/omit.py\n")
        for name in ("omit.py", "tracked_weight.pt", "data/tracked_fixture.txt"):
            path = child / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("excluded by parent\n", encoding="utf-8")
            self.git(child, "add", "-f", "--", name)
        self.git(child, "commit", "-qm", "Files excluded by parent policy")
        before = self.child_state(child)

        self.run_helper()

        entries = self.index_entries()
        for name in ("omit.py", "tracked_weight.pt", "data/tracked_fixture.txt"):
            self.assertNotIn(self.rel(child / name), entries)
        self.assertIn(self.rel(child / "source.py"), entries)
        self.assertIn(self.rel(child / "setup.sh"), entries)
        self.assert_child_unchanged(child, before)


if __name__ == "__main__":
    unittest.main()

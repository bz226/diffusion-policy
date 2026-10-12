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
MAX_FILE_BYTES = 100 * 1024 * 1024


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
                info = path.stat()
                if info.st_size > 1024 * 1024:
                    # Sparse oversized fixtures should not consume their apparent
                    # size in memory; size and timestamps still detect writes.
                    snapshot[relative] = ("large-file", mode, info.st_size,
                                          info.st_mtime_ns, info.st_ctime_ns)
                else:
                    snapshot[relative] = ("file", mode, path.read_bytes())
            elif path.is_dir():
                snapshot[relative] = ("directory", mode)
        return snapshot

    def assert_run_staged(self, entries, run):
        for name in ("result.pkl", ".hydra/config.yaml",
                     ".hydra/hydra.yaml", ".hydra/overrides.yaml"):
            self.assertEqual(entries[self.rel(run / name)][0], "100644")
        self.assertNotIn(self.rel(run / "run.log"), entries)
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

    def assert_rejected_without_mutation(self, *args):
        before = self.tree_snapshot()
        result = self.run_helper(*args, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue((result.stdout + result.stderr).strip())
        self.assertEqual(self.tree_snapshot(), before)
        return result

    def make_large_file(self, path, size=MAX_FILE_BYTES + 1):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            stream.truncate(size)
        return path

    def assert_size_rejected(self, path, *args):
        result = self.assert_rejected_without_mutation(*args)
        self.assertIn(b"100 MiB per-file limit", result.stderr)
        self.assertIn(repr(self.rel(path)).encode(), result.stderr)
        return result

    def test_rejects_untracked_large_file_before_any_mutation(self):
        self.prepare_valid_child_before_invalid()
        path = self.make_large_file(self.root / "results/replicates.jsonl")

        self.assert_size_rejected(path)

    def test_dry_run_rejects_large_file_without_any_mutation(self):
        self.prepare_valid_child_before_invalid()
        path = self.make_large_file(self.root / "results/replicates.jsonl")

        self.assert_size_rejected(path, "--dry-run")

    def test_rejects_tracked_large_file_with_and_without_ignore_rule(self):
        path = self.root / "commands.jsonl"
        path.write_text("small original log\n", encoding="utf-8")
        self.git(self.root, "add", "--", self.rel(path))
        self.git(self.root, "commit", "-qm", "Track log")
        self.make_large_file(path)

        self.assert_size_rejected(path)

        with (self.root / ".gitignore").open("a", encoding="utf-8") as stream:
            stream.write("/commands.jsonl\n")

        self.assert_size_rejected(path)

    def test_rejects_large_nested_source_before_replacing_gitlink(self):
        child = self.make_child()
        self.git(self.root, "add", "--", self.rel(child))
        # setup.sh is tracked in the child but covered by its ignore rules.
        path = self.make_large_file(child / "setup.sh")

        self.assert_size_rejected(path)

    def test_rejects_large_selected_nested_result(self):
        child = self.make_child()
        run = self.make_run(child)
        path = self.make_large_file(run / "result.pkl")

        self.assert_size_rejected(path)

    def test_ignores_untracked_generated_large_files_and_symlink_targets(self):
        child = self.make_child()
        generated = self.make_large_file(child / "checkpoint/weights.pt")
        external_generated = self.make_large_file(self.root / "data/generated.bin")
        link = self.root / "large-target-link"
        link.symlink_to(external_generated.relative_to(self.root))

        self.run_helper()

        entries = self.index_entries()
        self.assertNotIn(self.rel(generated), entries)
        self.assertNotIn(self.rel(external_generated), entries)
        self.assertEqual(entries[self.rel(link)][0], "120000")

    def test_large_child_source_excluded_by_parent_is_not_a_candidate(self):
        child = self.make_child()
        path = self.make_large_file(child / "source.py")
        with (self.root / ".gitignore").open("a", encoding="utf-8") as stream:
            stream.write(f"/{self.rel(path)}\n")

        self.run_helper()

        self.assertNotIn(self.rel(path), self.index_entries())

    def test_exact_size_limit_is_allowed_in_dry_run(self):
        self.make_large_file(self.root / "at-limit.bin", MAX_FILE_BYTES)
        before = self.tree_snapshot()

        self.run_helper("--dry-run")

        self.assertEqual(self.tree_snapshot(), before)

    def test_untracks_all_runtime_log_patterns_without_deleting_local_files(self):
        logs = [
            "run.log", "run.log.1", "scheduler.out", "scheduler.err", "commands.jsonl",
            "bootstrap_commands.jsonl", "monitor_history.jsonl",
            "campaign_monitor_history.jsonl", "campaign_revised_monitor.jsonl", "events.jsonl",
        ]
        directory = self.root / "results/study [1]*/runs/space and\nnewline"
        directory.mkdir(parents=True)
        paths = [directory / name for name in logs]
        # Cover the zero-directory case in results/**/<pattern> as well.
        paths.append(self.root / "results/commands.jsonl")
        for path in paths:
            path.write_bytes(b"original local log\n")
        self.git(self.root, "add", "--", "results")
        self.git(self.root, "commit", "-qm", "Legacy logs")
        paths[0].write_bytes(b"current local log, including unstaged changes\n")
        before = {path: path.read_bytes() for path in paths}

        result = self.run_helper()

        self.assertIn(f"Untracking {len(paths)} runtime log files".encode(), result.stdout)
        entries = self.index_entries()
        for path, contents in before.items():
            self.assertNotIn(self.rel(path), entries)
            self.assertEqual(path.read_bytes(), contents)
        deletions = self.git(self.root, "diff", "--cached", "--name-only", "--diff-filter=D", "-z")
        self.assertEqual(set(deletions.stdout.rstrip(b"\0").split(b"\0")),
                         {os.fsencode(self.rel(path)) for path in paths})
        self.run_helper()
        self.assertEqual(self.index_entries(), entries)

    def test_untracks_large_runtime_log_and_leaves_contents_and_timestamps_unchanged(self):
        path = self.root / "results/study/runs/commands.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("initial log\n", encoding="utf-8")
        self.git(self.root, "add", "--", self.rel(path))
        self.git(self.root, "commit", "-qm", "Track initial log")
        self.make_large_file(path)
        before = path.stat()

        self.run_helper()

        self.assertNotIn(self.rel(path), self.index_entries())
        after = path.stat()
        self.assertEqual((after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                         (before.st_size, before.st_mtime_ns, before.st_ctime_ns))

    def test_dry_run_reports_runtime_log_removal_without_mutation(self):
        path = self.root / "results/commands.jsonl"
        path.parent.mkdir()
        path.write_text("initial log\n", encoding="utf-8")
        self.git(self.root, "add", "--", self.rel(path))
        self.git(self.root, "commit", "-qm", "Track initial log")
        self.make_large_file(path)
        before = self.tree_snapshot()

        result = self.run_helper("--dry-run")

        self.assertIn(b"Would untrack 1 runtime log files", result.stdout)
        self.assertEqual(self.tree_snapshot(), before)

    def test_child_ignore_negations_cannot_reinclude_runtime_logs(self):
        directory = self.root / "results/study [1]*/runs"
        directory.mkdir(parents=True)
        with (self.root / ".gitignore").open("a", encoding="utf-8") as stream:
            stream.write("*.log\n*.out\n*.err\ncommands.jsonl\n")
        (directory / ".gitignore").write_text("!*.log\n!*.out\n!*.err\n!commands.jsonl\n")
        logs = [directory / name for name in ("console.log", "a.out", "a.err", "commands.jsonl")]
        for path in logs:
            self.make_large_file(path)
        measurements = [directory / name for name in (
            "replicates.jsonl", "runtime_estimates.jsonl", "metrics.csv", "result.pkl",
            "config.yaml", "analyze.py",
        )]
        for path in measurements:
            path.write_bytes(b"selected measurement or source\n")
        # Scope is results/: fixtures with the same basenames elsewhere survive.
        fixture = self.root / "tests/fixtures/commands.jsonl"
        fixture.parent.mkdir(parents=True)
        fixture.write_bytes(b"test fixture\n")
        self.git(self.root, "add", "-f", "--", self.rel(fixture))

        self.run_helper()

        entries = self.index_entries()
        for path in logs:
            self.assertNotIn(self.rel(path), entries)
            self.assertEqual(path.stat().st_size, MAX_FILE_BYTES + 1)
        for path in measurements + [fixture]:
            self.assertIn(self.rel(path), entries)

    def test_filters_nested_tracked_and_untracked_logs_without_changing_child(self):
        child = self.make_child("study [1]*")
        tracked_log = child / "old.log"
        tracked_log.write_bytes(b"nested log\n")
        self.git(child, "add", "-f", "--", tracked_log.name)
        self.git(child, "commit", "-qm", "Legacy nested log")
        self.make_large_file(tracked_log)
        new_log = self.make_large_file(child / "commands.jsonl")
        run = self.make_run(child)
        before = self.child_state(child)

        self.run_helper()

        entries = self.index_entries()
        self.assertNotIn(self.rel(tracked_log), entries)
        self.assertNotIn(self.rel(new_log), entries)
        self.assertIn(self.rel(child / "source.py"), entries)
        self.assert_run_staged(entries, run)
        self.assert_child_unchanged(child, before)

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

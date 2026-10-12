"""Synthetic Git integration tests; no real histories or experiments are changed."""

from contextlib import redirect_stdout
import importlib.util
import io
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest
from unittest.mock import patch


HELPER = Path(__file__).resolve().parents[1] / "scripts/repair_large_log_history.py"
SPEC = importlib.util.spec_from_file_location("repair_large_log_history", HELPER)
REPAIR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPAIR)


class RepairLargeLogHistoryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="repair-large-log-test-", dir="/tmp")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "repo"
        self.root.mkdir()
        env = os.environ.copy()
        for key in tuple(env):
            if key.startswith("GIT_"):
                env.pop(key)
        env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1",
                   GIT_AUTHOR_NAME="Original Author", GIT_AUTHOR_EMAIL="author@example.invalid",
                   GIT_COMMITTER_NAME="Original Committer", GIT_COMMITTER_EMAIL="committer@example.invalid",
                   GIT_AUTHOR_DATE="2026-01-02T03:04:05+03:00",
                   GIT_COMMITTER_DATE="2026-01-03T04:05:06-07:00")
        environment = patch.dict(os.environ, env, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        limit = patch.object(REPAIR, "MAX_BLOB_BYTES", 1024)
        limit.start()
        self.addCleanup(limit.stop)
        self.git("init", "-q", "--initial-branch=main")
        self.git("config", "core.fileMode", "true")
        self.git("config", "core.autocrlf", "false")
        self.source = self.root / "source.py"
        self.source.write_bytes(b"original source\n")
        self.script = self.root / "launch.sh"
        self.script.write_bytes(b"#!/bin/sh\ntrue\n")
        self.script.chmod(0o755)
        (self.root / "source-link").symlink_to("source.py")
        (self.root / ".gitignore").write_bytes(b"*.cache\n")
        self.base = self.commit("Published base")
        self.git("update-ref", "refs/remotes/origin/main", self.base)
        self.source.write_bytes(b"scientific gate source\n")
        self.gate = self.commit("Source gate revision")
        self.log = self.root / REPAIR.LOG_PATH
        self.log.parent.mkdir(parents=True)
        self.log.write_bytes(b"small historical command log\n")
        self.small = self.commit("Small command journal")
        self.log.write_bytes(b"a" * 1200)
        (self.root / "result.txt").write_text("first result\n")
        self.big1 = self.commit("First oversized snapshot")
        self.git("branch", "stage3-gradmean", self.big1)
        self.log.write_bytes(b"b" * 1400)
        (self.root / "result.txt").write_text("second result\n")
        self.script.chmod(0o644)
        self.big2 = self.commit("Second oversized snapshot\n\nPreserve this message exactly.")

    def git(self, *args, data=None):
        return subprocess.run(["git", "-C", str(self.root), *args], input=data,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout

    def commit(self, message):
        self.git("add", "-A")
        self.git("commit", "-qm", message)
        return self.git("rev-parse", "HEAD").decode().strip()

    def invoke(self, apply=False):
        stream = io.StringIO()
        with redirect_stdout(stream):
            REPAIR.repair(self.root, apply=apply)
        return stream.getvalue()

    def snapshot(self, include_git=True):
        result = {}
        for path in self.root.rglob("*"):
            relative = path.relative_to(self.root)
            if not include_git and relative.parts[0] == ".git":
                continue
            mode = stat.S_IMODE(path.lstat().st_mode)
            if path.is_symlink():
                result[str(relative)] = ("symlink", mode, os.readlink(path))
            elif path.is_file():
                result[str(relative)] = ("file", mode, path.read_bytes())
            else:
                result[str(relative)] = ("dir", mode)
        return result

    def metadata(self, commit):
        lines, message, _parent = REPAIR.header_and_message(self.git("cat-file", "commit", commit))
        return [line for line in lines if not line.startswith((b"tree ", b"parent "))], message

    def entries(self, commit):
        return {record.split(b"\t", 1)[1]: record.split(b"\t", 1)[0]
                for record in self.git("ls-tree", "-rz", commit).split(b"\0") if record}

    def expect_rejection_without_mutation(self, pattern):
        before = self.snapshot()
        with self.assertRaisesRegex(RuntimeError, pattern):
            self.invoke(apply=True)
        self.assertEqual(self.snapshot(), before)

    def test_dry_run_is_completely_read_only(self):
        before = self.snapshot()
        output = self.invoke()
        self.assertIn("Would rewrite 2 local commits", output)
        self.assertIn("Preserve 2 earlier unpublished commits", output)
        self.assertEqual(self.snapshot(), before)

    def test_repairs_only_bad_commits_preserving_pending_files_and_metadata(self):
        self.source.write_bytes(b"pending local edit\n")
        self.log.write_bytes(b"currently growing local log\n" * 100)
        (self.root / ".gitignore").write_text("*.cache\n/" + REPAIR.LOG_PATH + "\n")
        (self.root / "pending result.txt").write_text("untracked pending result\n")
        before = self.snapshot(include_git=False)
        before_index = REPAIR.git(self.root, "ls-files", "--stage", "-z").stdout

        output = self.invoke(apply=True)

        self.assertEqual(self.snapshot(include_git=False), before)
        self.assertIn("Repair complete", output)
        self.assertEqual(self.git("diff", "--cached", "--name-only"), b"")
        self.assertEqual(self.git("ls-files", "--stage", "-z"), REPAIR.without_log(before_index))
        new2 = self.git("rev-parse", "main").decode().strip()
        new1 = self.git("rev-parse", "main^").decode().strip()
        self.assertEqual(self.git("rev-parse", "main^^").decode().strip(), self.small)
        self.assertEqual(self.git("rev-parse", "main^^^").decode().strip(), self.gate)
        self.assertEqual(self.git("rev-parse", "stage3-gradmean").decode().strip(), new1)
        for old, new in ((self.big1, new1), (self.big2, new2)):
            expected = self.entries(old)
            expected.pop(REPAIR.LOG_PATH.encode())
            self.assertEqual(self.entries(new), expected)
            self.assertEqual(self.metadata(old), self.metadata(new))
        self.assertEqual(REPAIR.oversized_blobs(self.root, self.base, new2), {})
        backups = self.git("for-each-ref", "--format=%(objectname) %(refname)", "refs/backups/")
        self.assertIn(self.big1.encode(), backups)
        self.assertIn(self.big2.encode(), backups)
        self.assertEqual(self.git("rev-parse", "origin/main").decode().strip(), self.base)
        self.assertEqual(self.git("cat-file", "-s", new1 + "^:" + REPAIR.LOG_PATH).strip(), b"29")
        after = self.snapshot()
        self.assertIn("No changes needed", self.invoke(apply=True))
        self.assertEqual(self.snapshot(), after)

    def test_rejects_staged_changes(self):
        self.source.write_text("staged change\n")
        self.git("add", "source.py")
        self.expect_rejection_without_mutation("Staged changes")

    def test_rejects_intent_to_add(self):
        (self.root / "new-source.py").write_text("pending\n")
        self.git("add", "-N", "new-source.py")
        self.expect_rejection_without_mutation("Staged changes")

    def test_rejects_conflicted_index(self):
        blob = self.git("rev-parse", "HEAD:source.py").strip()
        self.git("update-index", "--force-remove", "source.py")
        self.git("update-index", "--index-info", data=b"100644 " + blob + b" 1\tsource.py\n")
        self.expect_rejection_without_mutation("index conflicts")

    def test_rejects_wrong_branch(self):
        self.git("symbolic-ref", "HEAD", "refs/heads/stage3-gradmean")
        self.expect_rejection_without_mutation("current branch must be main")

    def test_rejects_nonancestor_remote_ref(self):
        tree = self.git("rev-parse", self.base + "^{tree}").decode().strip()
        other = self.git("commit-tree", tree, "-m", "independent remote commit").decode().strip()
        self.git("update-ref", "refs/remotes/origin/main", other)
        self.expect_rejection_without_mutation("not an ancestor")

    def test_rejects_published_log(self):
        self.git("update-ref", "refs/remotes/origin/main", self.small)
        self.expect_rejection_without_mutation("already exists on origin/main")

    def test_rejects_merge(self):
        tree = self.git("rev-parse", "HEAD^{tree}").decode().strip()
        merged = self.git("commit-tree", tree, "-p", self.big2, "-p", self.base,
                          "-m", "merge fixture").decode().strip()
        self.git("update-ref", "refs/heads/main", merged)
        self.expect_rejection_without_mutation("merge repair refused")

    def test_rejects_signed_commit(self):
        header, message = self.git("cat-file", "commit", self.big2).split(b"\n\n", 1)
        raw = header + b"\ngpgsig -----BEGIN PGP SIGNATURE-----\n fake-test-signature\n -----END PGP SIGNATURE-----\n\n" + message
        signed = self.git("hash-object", "-t", "commit", "-w", "--stdin", data=raw).decode().strip()
        self.git("update-ref", "refs/heads/main", signed)
        self.expect_rejection_without_mutation("Signed commits")

    def test_rejects_other_oversized_files(self):
        (self.root / "other.bin").write_bytes(b"other" * 300)
        self.commit("Another big file")
        self.expect_rejection_without_mutation("Other oversized")

    def test_rejects_study_branch_checked_out_elsewhere(self):
        self.git("worktree", "add", "-q", str(self.root.parent / "other"), "stage3-gradmean")
        self.expect_rejection_without_mutation("checked out in another worktree")

    def test_does_not_update_unrelated_study_tip(self):
        self.git("update-ref", "refs/heads/stage3-gradmean", self.gate)
        self.invoke(apply=True)
        self.assertEqual(self.git("rev-parse", "stage3-gradmean").decode().strip(), self.gate)
        self.assertNotIn(b"/stage3-gradmean", self.git("for-each-ref", "--format=%(refname)", "refs/backups/"))

    def test_handles_absent_log_at_tip_and_becomes_idempotent(self):
        self.log.unlink()
        self.commit("Remove command log later")
        before = self.snapshot(include_git=False)
        self.invoke(apply=True)
        self.assertEqual(self.snapshot(include_git=False), before)
        self.assertEqual(REPAIR.oversized_blobs(self.root, self.base, "main"), {})
        self.assertIn("No changes needed", self.invoke(apply=True))

    def test_ref_failure_preserves_original_index_refs_and_worktree(self):
        before_worktree = self.snapshot(include_git=False)
        before_index = (self.root / ".git/index").read_bytes()
        original = REPAIR.git

        def fail_publication(root, *args, **kwargs):
            if args[0] == "update-ref":
                raise RuntimeError("injected ref publication failure")
            return original(root, *args, **kwargs)

        with patch.object(REPAIR, "git", fail_publication):
            with self.assertRaisesRegex(RuntimeError, "injected ref publication failure"):
                self.invoke(apply=True)
        self.assertEqual((self.root / ".git/index").read_bytes(), before_index)
        self.assertEqual(self.git("rev-parse", "main").decode().strip(), self.big2)
        self.assertEqual(self.git("rev-parse", "stage3-gradmean").decode().strip(), self.big1)
        self.assertEqual(self.snapshot(include_git=False), before_worktree)
        self.assertFalse((self.root / ".git/index.lock").exists())

    def test_index_install_failure_rolls_back_refs(self):
        before_index = (self.root / ".git/index").read_bytes()
        with patch.object(REPAIR.os, "replace", side_effect=OSError("injected index installation failure")):
            with self.assertRaisesRegex(OSError, "injected index installation failure"):
                self.invoke(apply=True)
        self.assertEqual((self.root / ".git/index").read_bytes(), before_index)
        self.assertEqual(self.git("rev-parse", "main").decode().strip(), self.big2)
        self.assertEqual(self.git("rev-parse", "stage3-gradmean").decode().strip(), self.big1)
        self.assertFalse((self.root / ".git/index.lock").exists())

    def test_interrupt_after_ref_transaction_restores_original_refs_and_index(self):
        before_index = (self.root / ".git/index").read_bytes()
        original = REPAIR.git

        def interrupt_after_publication(root, *args, **kwargs):
            result = original(root, *args, **kwargs)
            if args[0] == "update-ref" and "remove oversized unpublished command log" in args:
                raise KeyboardInterrupt("after successful ref transaction")
            return result

        with patch.object(REPAIR, "git", side_effect=interrupt_after_publication):
            with self.assertRaises(KeyboardInterrupt):
                self.invoke(apply=True)
        self.assertEqual((self.root / ".git/index").read_bytes(), before_index)
        self.assertEqual(self.git("rev-parse", "main").decode().strip(), self.big2)
        self.assertEqual(self.git("rev-parse", "stage3-gradmean").decode().strip(), self.big1)
        self.assertFalse((self.root / ".git/index.lock").exists())

    def test_interrupt_after_index_install_restores_original_refs_and_index(self):
        before_index = (self.root / ".git/index").read_bytes()
        original = os.replace
        calls = []

        def interrupt_once(source, destination):
            original(source, destination)
            calls.append(destination)
            if len(calls) == 1:
                raise KeyboardInterrupt("after successful rename")

        with patch.object(REPAIR.os, "replace", side_effect=interrupt_once):
            with self.assertRaises(KeyboardInterrupt):
                self.invoke(apply=True)
        self.assertEqual((self.root / ".git/index").read_bytes(), before_index)
        self.assertEqual(self.git("rev-parse", "main").decode().strip(), self.big2)
        self.assertEqual(self.git("rev-parse", "stage3-gradmean").decode().strip(), self.big1)
        self.assertFalse((self.root / ".git/index.lock").exists())

    def test_preserves_index_file_permissions(self):
        index = self.root / ".git/index"
        index.chmod(0o640)
        self.invoke(apply=True)
        self.assertEqual(stat.S_IMODE(index.stat().st_mode), 0o640)

    def test_does_not_delete_another_process_lock_after_success(self):
        original = os.replace
        lock = self.root / ".git/index.lock"

        def install_then_new_lock(source, destination):
            original(source, destination)
            lock.write_bytes(b"new lock owned by another Git operation\n")

        with patch.object(REPAIR.os, "replace", side_effect=install_then_new_lock):
            self.invoke(apply=True)
        self.assertEqual(lock.read_bytes(), b"new lock owned by another Git operation\n")
        self.assertNotEqual(self.git("rev-parse", "main").decode().strip(), self.big2)

    def test_readonly_git_is_rejected_before_writing_objects(self):
        with patch.object(REPAIR.os, "access", return_value=False):
            self.expect_rejection_without_mutation("read-only here")

    def test_existing_index_lock_is_preserved(self):
        lock = self.root / ".git/index.lock"
        lock.write_bytes(b"other process owns this lock\n")
        before_index = (self.root / ".git/index").read_bytes()
        with self.assertRaisesRegex(RuntimeError, "Another Git operation"):
            self.invoke(apply=True)
        self.assertEqual(lock.read_bytes(), b"other process owns this lock\n")
        self.assertEqual((self.root / ".git/index").read_bytes(), before_index)
        self.assertEqual(self.git("rev-parse", "main").decode().strip(), self.big2)


if __name__ == "__main__":
    unittest.main()

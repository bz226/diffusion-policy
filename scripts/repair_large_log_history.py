#!/usr/bin/env python3
"""Remove oversized command-log versions from unpublished main history.

The default is a read-only preview. --apply rewrites only the affected local
commits, retains backups under refs/backups, and leaves working files untouched.
This is deliberately scoped to the rejected Gradmean command log, not a general
history-cleaning utility. It never fetches, pushes, or starts experiments.
"""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import uuid


LOG_PATH = "results/stage3/gradmean/runs/commands.jsonl"
MAX_BLOB_BYTES = 100 * 1024 * 1024
MAIN_REF = "refs/heads/main"
STUDY_REF = "refs/heads/stage3-gradmean"
BASE_REF = "refs/remotes/origin/main"


def git(root, *args, data=None, index=None, allowed=(0,)):
    env = os.environ.copy()
    env.update(GIT_OPTIONAL_LOCKS="0", GIT_NO_REPLACE_OBJECTS="1")
    if index is not None:
        env["GIT_INDEX_FILE"] = str(index)
    result = subprocess.run(
        ["git", "-C", str(root), "-c", "core.splitIndex=false", *args],
        input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
    )
    if result.returncode not in allowed:
        raise RuntimeError(result.stderr.decode(errors="replace").strip()
                           or "git " + args[0] + " failed")
    return result


def oid(root, ref, optional=False):
    result = git(root, "rev-parse", "--verify", ref, allowed=(0, 128) if optional else (0,))
    return result.stdout.decode().strip() if result.returncode == 0 else None


def header_and_message(raw):
    try:
        header, message = raw.split(b"\n\n", 1)
    except ValueError as exc:
        raise RuntimeError("Malformed Git commit; repair refused.") from exc
    lines = header.split(b"\n")
    if any(line.startswith((b"gpgsig ", b"gpgsig-sha256 ", b"mergetag ")) for line in lines):
        raise RuntimeError("Signed commits are not supported; repair refused.")
    parents = [line[7:].decode() for line in lines if line.startswith(b"parent ")]
    if len(parents) != 1:
        raise RuntimeError("Only a linear unpublished history is supported; merge repair refused.")
    return lines, message, parents[0]


def oversized_blobs(root, base, head):
    objects = git(root, "rev-list", "--objects", "--no-object-names", base + ".." + head).stdout
    if not objects:
        return {}
    output = git(root, "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)",
                 data=objects).stdout
    result = {}
    for row in output.splitlines():
        object_id, kind, size = row.split()
        if kind == b"blob" and int(size) > MAX_BLOB_BYTES:
            result[object_id.decode()] = int(size)
    return result


def log_entry(root, revision):
    output = git(root, "ls-tree", "-z", revision, "--", LOG_PATH).stdout
    if not output:
        return None
    records = output.rstrip(b"\0").split(b"\0")
    if len(records) != 1:
        raise RuntimeError("Unexpected command-log tree entries; repair refused.")
    metadata, path = records[0].split(b"\t", 1)
    mode, kind, object_id = metadata.split()
    if path != LOG_PATH.encode() or kind != b"blob" or mode not in (b"100644", b"100755"):
        raise RuntimeError("Command-log path must be an ordinary file; repair refused.")
    return object_id.decode()


def check_study_worktrees(root):
    records = git(root, "worktree", "list", "--porcelain", "-z").stdout.split(b"\0")
    if ("branch " + STUDY_REF).encode() in records:
        raise RuntimeError("stage3-gradmean is checked out in another worktree; repair refused.")


def check_staging(root):
    if git(root, "ls-files", "--unmerged", "-z").stdout:
        raise RuntimeError("Resolve index conflicts before repairing history.")
    result = git(root, "diff", "--cached", "--quiet", "--ita-visible-in-index", "HEAD", "--",
                 allowed=(0, 1))
    if result.returncode:
        raise RuntimeError("Staged changes exist. Unstage or commit them before repairing history.")


def plan(root):
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR",
                 "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_NAMESPACE"):
        if os.environ.get(name):
            raise RuntimeError("Unset " + name + " before repairing history.")
    top = Path(git(root, "rev-parse", "--show-toplevel").stdout.decode().strip()).resolve()
    if top != root.resolve():
        raise RuntimeError("Run the helper from its parent repository.")
    branch = git(root, "symbolic-ref", "--quiet", "HEAD", allowed=(0, 1)).stdout.decode().strip()
    if branch != MAIN_REF:
        raise RuntimeError("The current branch must be main; repair refused.")
    check_staging(root)
    head, base = oid(root, MAIN_REF), oid(root, BASE_REF)
    if git(root, "merge-base", "--is-ancestor", base, head, allowed=(0, 1)).returncode:
        raise RuntimeError("origin/main is not an ancestor of main; repair refused.")
    if log_entry(root, base) is not None:
        raise RuntimeError("The log already exists on origin/main. Published history will not be rewritten.")
    large = oversized_blobs(root, base, head)
    commits = []
    first_large = None
    known_log_blobs = set()
    previous = base
    for commit in git(root, "rev-list", "--reverse", base + ".." + head).stdout.decode().splitlines():
        raw = git(root, "cat-file", "commit", commit).stdout
        _lines, _message, parent = header_and_message(raw)
        if parent != previous:
            raise RuntimeError("Only a linear unpublished history is supported; repair refused.")
        entry = log_entry(root, commit)
        if entry:
            known_log_blobs.add(entry)
        if entry in large and first_large is None:
            first_large = len(commits)
        commits.append((commit, raw))
        previous = commit
    unrelated = set(large) - known_log_blobs
    if unrelated:
        raise RuntimeError("Other oversized outgoing blobs also need attention: "
                           + ", ".join(f"{key} ({large[key]} bytes)" for key in sorted(unrelated)))
    if first_large is None:
        return None
    git_dir = Path(git(root, "rev-parse", "--absolute-git-dir").stdout.decode().strip())
    index = git_dir / "index"
    if not index.is_file() or index.is_symlink():
        raise RuntimeError("A regular Git index is required; repair refused.")
    if git(root, "rev-parse", "--shared-index-path").stdout.strip():
        raise RuntimeError("Split indexes are not supported by this repair.")
    rewritten = commits[first_large:]
    study = oid(root, STUDY_REF, optional=True)
    mapped_study = study if study in {commit for commit, _ in rewritten} else None
    if mapped_study:
        check_study_worktrees(root)
    return dict(head=head, base=base, commits=rewritten, study=mapped_study,
                index=index, index_bytes=index.read_bytes(), git_dir=git_dir,
                index_mode=stat.S_IMODE(index.stat().st_mode),
                entries=git(root, "ls-files", "--stage", "-z").stdout,
                oversized=large, preserved_count=first_large)


def require_writable_git(state):
    directory, index = state["git_dir"], state["index"]
    readonly = bool(os.statvfs(directory).f_flag & getattr(os, "ST_RDONLY", 1))
    if readonly or not os.access(directory, os.W_OK) or not os.access(index, os.W_OK):
        raise RuntimeError("The parent .git directory is read-only here. Run this helper with --apply "
                           "in your normal terminal; no history has been changed.")


def recheck(root, state):
    if git(root, "symbolic-ref", "--quiet", "HEAD", allowed=(0, 1)).stdout.strip() != MAIN_REF.encode():
        raise RuntimeError("The current branch changed during repair; nothing published.")
    if oid(root, MAIN_REF) != state["head"] or oid(root, BASE_REF) != state["base"]:
        raise RuntimeError("Git refs changed during repair; nothing published.")
    if state["index"].read_bytes() != state["index_bytes"]:
        raise RuntimeError("The staging index changed during repair; nothing published.")
    if state["study"]:
        if oid(root, STUDY_REF, optional=True) != state["study"]:
            raise RuntimeError("stage3-gradmean changed during repair; nothing published.")
        check_study_worktrees(root)


def without_log(entries):
    return b"".join(record + b"\0" for record in entries.split(b"\0")
                    if record and record.split(b"\t", 1)[1] != LOG_PATH.encode())


def write_index_lock(descriptor, data, mode):
    with os.fdopen(descriptor, "wb") as stream:
        os.fchmod(stream.fileno(), mode)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def file_identity(path):
    try:
        value = path.stat()
        return value.st_dev, value.st_ino
    except FileNotFoundError:
        return None


def apply_plan(root, state):
    require_writable_git(state)
    mapping = {}
    with tempfile.TemporaryDirectory(prefix="repair-command-log-") as temporary:
        temp_index = Path(temporary) / "rewrite-index"
        for previous, raw in state["commits"]:
            lines, message, parent = header_and_message(raw)
            git(root, "read-tree", previous, index=temp_index)
            git(root, "update-index", "--force-remove", "--", LOG_PATH, index=temp_index)
            tree = git(root, "write-tree", index=temp_index).stdout.strip()
            replacement = []
            for line in lines:
                if line.startswith(b"tree "):
                    line = b"tree " + tree
                elif line.startswith(b"parent "):
                    line = b"parent " + mapping.get(parent, parent).encode()
                replacement.append(line)
            commit = git(root, "hash-object", "-t", "commit", "-w", "--stdin",
                         data=b"\n".join(replacement) + b"\n\n" + message).stdout.decode().strip()
            changed = git(root, "diff-tree", "--no-commit-id", "--name-only", "-r", "-z",
                          previous, commit).stdout
            if changed not in (b"", LOG_PATH.encode() + b"\0"):
                raise RuntimeError("Unexpected source changes in repaired tree; nothing published.")
            mapping[previous] = commit
        new_head = mapping[state["head"]]
        if oversized_blobs(root, state["base"], new_head):
            raise RuntimeError("Oversized blobs remain in repaired history; nothing published.")
        git(root, "merge-base", "--is-ancestor", state["base"], new_head)

        # Build the exact replacement index separately. No working files are read
        # or written, so concurrently produced experiment outputs remain intact.
        final_index = Path(temporary) / "final-index"
        final_index.write_bytes(state["index_bytes"])
        git(root, "update-index", "--force-remove", "--", LOG_PATH, index=final_index)
        if git(root, "ls-files", "--stage", "-z", index=final_index).stdout != without_log(state["entries"]):
            raise RuntimeError("Unexpected index changes; nothing published.")
        new_index_bytes = final_index.read_bytes()
        prefix = "refs/backups/large-command-log-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        refs = [(MAIN_REF, state["head"], new_head, prefix + "/main")]
        if state["study"]:
            refs.append((STUDY_REF, state["study"], mapping[state["study"]], prefix + "/stage3-gradmean"))
        transaction = ["start", "verify " + BASE_REF + " " + state["base"]]
        for ref, old, new, backup in refs:
            transaction += ["create " + backup + " " + old, "update " + ref + " " + new + " " + old]
        transaction += ["prepare", "commit"]
        recheck(root, state)
        lock = state["index"].with_name("index.lock")
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
        except FileExistsError as exc:
            raise RuntimeError("Another Git operation holds index.lock; retry after it finishes.") from exc
        lock_stat = os.fstat(descriptor)
        lock_identity = lock_stat.st_dev, lock_stat.st_ino
        publication_attempted = False
        try:
            write_index_lock(descriptor, new_index_bytes, state["index_mode"])
            recheck(root, state)
            publication_attempted = True
            git(root, "update-ref", "--stdin", "-m", "remove oversized unpublished command log",
                data=("\n".join(transaction) + "\n").encode())
            os.replace(lock, state["index"])
        except BaseException:
            published = False
            if publication_attempted:
                # The subprocess can finish its transaction before Python sees
                # an interrupt. Inspect the refs instead of trusting a flag set
                # after subprocess.run returns.
                try:
                    current_refs = {ref: oid(root, ref, optional=True) for ref, *_ in refs}
                    if all(current_refs[ref] == new for ref, _old, new, _backup in refs):
                        published = True
                    elif not all(current_refs[ref] == old for ref, old, _new, _backup in refs):
                        raise RuntimeError("Affected refs no longer match either the original or repaired tips.")
                except (RuntimeError, OSError) as exc:
                    raise RuntimeError("Could not determine interrupted publication state: " + str(exc)
                                       + ". Original tips were scheduled for backup under " + prefix
                                       + ". Stop and inspect the refs and index before continuing.") from exc
            if published:
                try:
                    installed = state["index"].read_bytes() == new_index_bytes
                    if installed:
                        # A signal may arrive immediately after a successful
                        # rename. Reacquire the index lock before rolling back.
                        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
                        lock_stat = os.fstat(descriptor)
                        lock_identity = lock_stat.st_dev, lock_stat.st_ino
                        write_index_lock(descriptor, state["index_bytes"], state["index_mode"])
                    expected = new_index_bytes if installed else state["index_bytes"]
                    if state["index"].read_bytes() != expected:
                        raise RuntimeError("The index changed during interrupted publication.")
                    rollback = ["start"] + ["update " + ref + " " + old + " " + new for ref, old, new, _ in refs]
                    rollback += ["prepare", "commit"]
                    git(root, "update-ref", "--stdin", "-m", "rollback interrupted command-log repair",
                        data=("\n".join(rollback) + "\n").encode())
                    if installed:
                        os.replace(lock, state["index"])
                except (RuntimeError, OSError) as exc:
                    raise RuntimeError("Could not fully roll back interrupted publication: " + str(exc)
                                       + ". Original commits remain under " + prefix
                                       + "; the index and refs may disagree at the command-log path. "
                                       "Stop and inspect Git state before continuing.") from exc
            raise
        finally:
            # A concurrent Git command can create a new lock after our rename.
            # Only unlink the lock this invocation owns.
            if file_identity(lock) == lock_identity:
                lock.unlink()
    return refs


def repair(root, apply=False):
    root = Path(root).resolve()
    state = plan(root)
    if state is None:
        print("No oversized command-log versions in unpublished main history. No changes needed.")
        return
    print(f"Found {len(state['oversized'])} oversized command-log versions.")
    print(f"Preserve {state['preserved_count']} earlier unpublished commits and all published history.")
    print(f"{'Will rewrite' if apply else 'Would rewrite'} {len(state['commits'])} local commits, "
          "removing only " + LOG_PATH + ".")
    if state["study"]:
        print("Also update stage3-gradmean to its corresponding repaired commit.")
    print("Working files, including the local log and pending results, stay untouched.")
    if not apply:
        print("Preview only: no objects, indexes, or refs written. Run with --apply in your normal terminal.")
        return
    refs = apply_plan(root, state)
    for ref, old, new, backup in refs:
        print(f"{ref}: {old[:12]} -> {new[:12]}; original retained as {backup}")
    print("Repair complete. Pending working changes were not staged. A normal git push origin main is sufficient.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="apply the bounded local history repair")
    mode.add_argument("--dry-run", action="store_true", help="preview only (the default)")
    args = parser.parse_args()
    repair(Path(__file__).resolve().parent.parent, apply=args.apply)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError) as exc:
        print("Error: " + str(exc), file=sys.stderr)
        sys.exit(1)

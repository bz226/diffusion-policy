#!/usr/bin/env python3
"""Stage experiment sources and results as ordinary files, then run git add -A.

Nested results/*/dppo checkouts retain their own Git history and clean status
for experiment provenance checks. --dry-run does not write either Git index.
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def git(root, *args, data=None, allowed=(0,)):
    command = ["git", "-C", str(root)]
    if args[0] in {"add", "rm"}:
        command.append("--literal-pathspecs")
    result = subprocess.run(
        [*command, *args],
        input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode not in allowed:
        raise RuntimeError(os.fsdecode(result.stderr).strip() or f"git {args[0]} failed")
    return result.stdout


def entries(root):
    result = []
    for record in git(root, "ls-files", "--stage", "-z").split(b"\0"):
        if record:
            info, path = record.split(b"\t", 1)
            mode, _oid, stage = info.split()
            result.append((os.fsdecode(path), mode, stage))
    return result


def nul_paths(paths):
    return b"".join(os.fsencode(path) + b"\0" for path in paths)


def ignored_by_parent(root, checkout, paths):
    """Respect surrounding ignore rules, including study-specific exclusions.

    A checkout's own ignore rules may cover its tracked scripts and selected
    result files; these are precisely the files we need to seed explicitly.
    """
    if not paths:
        return set()
    # Git reports only the last matching rule. Evaluate the ancestor ignore
    # files in an empty temporary worktree so child rules cannot mask exclusions.
    # The real Git directory supplies its config/global excludes, read-only.
    git_dir = os.fsdecode(git(root, "rev-parse", "--absolute-git-dir")).strip()
    with tempfile.TemporaryDirectory(prefix="experiment-parent-ignores-") as temporary:
        probe = Path(temporary)
        ancestor = checkout.parent
        while True:
            ignore = ancestor / ".gitignore"
            if ignore.is_file():
                target = probe / ignore.relative_to(root)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(ignore.read_bytes())
            if ancestor == root:
                break
            ancestor = ancestor.parent
        for path in paths:
            target = probe / path
            (target if path.endswith("/") else target.parent).mkdir(parents=True, exist_ok=True)
        output = git(probe, "--git-dir", git_dir, "--work-tree", str(probe),
                     "check-ignore", "--no-index", "-z", "--stdin",
                     data=nul_paths(paths), allowed=(0, 1))
    return {os.fsdecode(path) for path in output.split(b"\0") if path}


def selected_results(checkout):
    logs = checkout / "log" / "gym-finetune"
    if not logs.is_dir():
        return []
    selected = []
    # Prune checkpoint/data/cache directories rather than walking large assets.
    skip = {".git", "checkpoint", "checkpoints", "data", "datasets", "__pycache__", "cache", ".cache"}
    for directory, subdirs, files in os.walk(logs, followlinks=False):
        subdirs[:] = [name for name in subdirs
                      if name not in skip and not (Path(directory) / name).is_symlink()]
        for name in files:
            path = Path(directory) / name
            if (name in {"result.pkl", "run.log"}
                    or (path.parent.name == ".hydra" and path.suffix == ".yaml")):
                selected.append(path)
    return selected


def declared_submodules(root):
    if not (root / ".gitmodules").is_file():
        return set()
    output = git(root, "config", "-z", "--file", ".gitmodules", "--get-regexp",
                 r"^submodule\..*\.path$", allowed=(0, 1))
    return {os.fsdecode(record.split(b"\n", 1)[1])
            for record in output.split(b"\0") if record}


def plan(root):
    top = Path(os.fsdecode(git(root, "rev-parse", "--show-toplevel")).strip()).resolve()
    if top != root:
        raise RuntimeError("Run the copy of this script in the parent repository's scripts directory.")
    index = entries(root)
    if any(stage != b"0" for _, _, stage in index):
        raise RuntimeError("Resolve parent index conflicts before staging.")
    gitlinks = {path for path, mode, _ in index if mode == b"160000"}
    submodules = declared_submodules(root)
    plans = []
    for checkout in sorted((root / "results").glob("*/dppo")):
        relative = checkout.relative_to(root).as_posix()
        if relative in submodules or not (checkout / ".git").exists():
            continue
        if checkout.is_symlink() or not checkout.resolve().is_relative_to(root):
            raise RuntimeError(f"Refusing checkout outside the project: {relative}")
        if ignored_by_parent(root, checkout, [relative + "/"]):
            continue
        inner_top = Path(os.fsdecode(git(checkout, "rev-parse", "--show-toplevel")).strip())
        if inner_top.resolve() != checkout.resolve():
            raise RuntimeError(f"Not an independent checkout: {relative}")
        source = entries(checkout)
        if any(mode == b"160000" or stage != b"0" for _, mode, stage in source):
            raise RuntimeError(f"{relative} contains submodules or conflicts; resolve them first.")
        paths = [str((checkout / path).relative_to(root)) for path, _, _ in source
                 if (checkout / path).is_file() or (checkout / path).is_symlink()]
        excluded = ignored_by_parent(root, checkout, paths)
        paths = [path for path in paths if path not in excluded]
        if not paths:
            raise RuntimeError(f"{relative} has no eligible tracked source files; no changes made.")
        artifacts = [str(path.relative_to(root)) for path in selected_results(checkout)]
        excluded = ignored_by_parent(root, checkout, artifacts)
        artifacts = [path for path in artifacts if path not in excluded]
        plans.append((relative, sorted(set(paths + artifacts)), len(paths), len(artifacts)))
    unsupported = gitlinks - submodules - {relative for relative, *_ in plans}
    if unsupported:
        raise RuntimeError("Unregistered embedded repositories outside the supported checkouts: "
                           + ", ".join(sorted(unsupported)))
    return plans, gitlinks, submodules


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="show the plan without staging")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    plans, gitlinks, submodules = plan(root)
    for relative, paths, source_count, artifact_count in plans:
        action = "Would stage" if args.dry_run else "Staging"
        print(f"{action} {relative}: {source_count} source files, {artifact_count} result records.",
              flush=True)
        if not args.dry_run:
            if relative in gitlinks:
                git(root, "rm", "--cached", "-f", "--", relative)
            git(root, "update-index", "--add", "-z", "--stdin", data=nul_paths(paths))
    if args.dry_run:
        print("Would then run git add -A. No files or Git indexes changed.")
        return
    git(root, "add", "-A")
    unexpected = {path for path, mode, _ in entries(root)
                  if mode == b"160000" and path not in submodules}
    if unexpected:
        raise RuntimeError("Other embedded repositories need explicit handling: "
                           + ", ".join(sorted(unexpected)))
    print("Staged ordinary source files and results; nested Git metadata preserved.")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

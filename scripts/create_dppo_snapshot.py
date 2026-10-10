#!/usr/bin/env python3
"""Create results/<study>/dppo as ordinary source files from a local Git revision.

Use a local independent checkout or an archived Git directory as --source.
The study must not exist. This exports committed source, never stages files,
and never creates a nested .git directory or pointer in the new study.
"""

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit, urlunsplit


SNAPSHOT_IGNORES = """# Ordinary source snapshot; original upstream rules are in ../provenance/.
# Keep source, result.pkl, run.log, and Hydra configurations versionable.
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.venv/
venv/
.env
.env.*
!.env.example
wandb/
build/
dist/
*.egg-info/
.tox/
.nox/
cache/
.cache/
checkpoint/
checkpoints/
data/
datasets/
*.pt
*.pth
*.ckpt
*.safetensors
*.onnx
*.h5
*.hdf5
*.zarr/
*.zarr.zip
"""


def git(metadata, *args, data=None, allowed=(0,)):
    environment = dict(os.environ, GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0")
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR",
                 "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_PREFIX"):
        environment.pop(name, None)
    result = subprocess.run(
        ["git", "--git-dir", str(metadata), *args], input=data,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment,
    )
    if result.returncode not in allowed:
        raise RuntimeError(os.fsdecode(result.stderr).strip() or "Git command failed")
    return result.stdout


def source_metadata(source, subdir):
    source = Path(source).resolve()
    marker = source / ".git"
    if marker.exists():
        # Resolve linked worktrees as well, without letting Git discover a parent.
        metadata = Path(os.fsdecode(git(marker, "rev-parse", "--absolute-git-dir")).strip())
        dirty = git(metadata, "-C", str(source), "--work-tree", str(source), "--literal-pathspecs",
                    "status", "--porcelain", "-z", "--", subdir)
        if dirty:
            raise RuntimeError("Source has uncommitted changes; commit the intended source first, "
                               "or explicitly select its archived Git directory to export a commit.")
        return metadata
    # A .git-free source folder must not silently resolve to the parent repository.
    if not (source / "HEAD").is_file() or not (source / "objects").is_dir():
        raise RuntimeError("--source must be an independent checkout or an explicit Git directory; "
                           "ordinary source folders use the parent repository for later provenance.")
    git(source, "rev-parse", "--git-dir")
    return source


def committed_files(metadata, revision, subdir):
    commit = git(metadata, "rev-parse", "--verify", "--end-of-options",
                 revision + "^{commit}").decode().strip()
    entries = []
    tree = commit if subdir == "." else commit + ":" + subdir
    for item in git(metadata, "ls-tree", "-rz", "--full-tree", tree).split(b"\0"):
        if not item:
            continue
        header, raw_path = item.split(b"\t", 1)
        mode, kind, oid = header.split()
        path = PurePosixPath(os.fsdecode(raw_path))
        if path.is_absolute() or any(part in {"..", "."} or part.lower() == ".git"
                                     for part in path.parts):
            raise RuntimeError("Unsafe source path: " + str(path))
        if mode not in {b"100644", b"100755", b"120000"} or kind != b"blob":
            raise RuntimeError("Unsupported source entry (submodule or special file): " + str(path))
        if path.name == ".gitignore" and mode == b"120000":
            raise RuntimeError("Symlinked .gitignore is unsupported: " + str(path))
        entries.append((path, mode, oid))
    if not entries:
        raise RuntimeError("Source revision has no files")
    # Batch object reads preserve committed bytes, executable bits, and symlinks;
    # unlike git archive, this includes files marked export-ignore upstream.
    blobs = git(metadata, "cat-file", "--batch", data=b"".join(oid + b"\n" for _, _, oid in entries))
    offset = 0
    files = []
    for path, mode, oid in entries:
        end = blobs.index(b"\n", offset)
        found_oid, kind, size = blobs[offset:end].split()
        if found_oid != oid or kind != b"blob":
            raise RuntimeError("Unexpected source object: " + str(path))
        offset = end + 1
        content = blobs[offset:offset + int(size)]
        offset += int(size) + 1
        files.append((path, mode, content))
    return commit, files


def safe_remote(metadata):
    remote = os.fsdecode(git(metadata, "config", "--get", "remote.origin.url", allowed=(0, 1))).strip()
    if "://" in remote:
        parsed = urlsplit(remote)
        # Do not persist embedded credentials or query-string tokens.
        remote = urlunsplit((parsed.scheme, parsed.netloc.rsplit("@", 1)[-1], parsed.path, "", ""))
    return remote or None


def create_snapshot(root, source, revision, study, subdir="."):
    root = Path(root).resolve()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", study):
        raise RuntimeError("Use a single study name containing letters, digits, '.', '_' or '-'")
    subtree = PurePosixPath(subdir)
    if subtree.is_absolute() or any(part == ".." or part.lower() == ".git" for part in subtree.parts):
        raise RuntimeError("--subdir must be a relative source directory without '..' or '.git'")
    subdir = str(subtree)
    results = root / "results"
    destination = results / study
    if results.is_symlink() or destination.exists() or destination.is_symlink():
        raise RuntimeError("Study destination already exists or results is a symlink: " + str(destination))
    metadata = source_metadata(source, subdir)
    commit, files = committed_files(metadata, revision, subdir)
    remote = safe_remote(metadata)
    with tempfile.TemporaryDirectory(prefix="dppo-source-snapshot-") as temporary:
        prepared = Path(temporary)
        exported = prepared / "dppo"
        exported.mkdir()
        provenance = prepared / "provenance"
        provenance.mkdir()
        saved_ignores = []
        links = []
        for path, mode, content in files:
            if path.name == ".gitignore":
                saved = Path("provenance/upstream_gitignores") / (str(path) + ".txt")
                target = prepared / saved
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                saved_ignores.append({"source_path": str(path), "saved_path": saved.as_posix()})
                continue
            target = exported / path
            target.parent.mkdir(parents=True, exist_ok=True)
            if mode == b"120000":
                target.symlink_to(os.fsdecode(content))
                links.append(target)
            else:
                target.write_bytes(content)
                target.chmod(0o755 if mode == b"100755" else 0o644)
        for link in links:
            try:
                resolved = link.resolve()
                resolved.relative_to(exported)
                if Path(os.readlink(link)).is_absolute():
                    raise ValueError("absolute symlink")
            except (ValueError, RuntimeError) as exc:
                raise RuntimeError("Source symlink escapes the snapshot or forms a cycle: "
                                   + str(link.relative_to(exported))) from exc
        (exported / ".gitignore").write_text(SNAPSHOT_IGNORES)
        record = {
            "format_version": 1,
            "source_directory": "results/{}/dppo".format(study),
            "source_git_directory": str(metadata),
            "source_subdirectory": subdir,
            "upstream_url": remote,
            "commit": commit,
            "source_file_count": len(files),
            "snapshot_file_count": len(files) - len(saved_ignores) + 1,
            "upstream_gitignores": saved_ignores,
            "note": "Initial committed source snapshot. Upstream ignore files are preserved separately "
                    "and replaced with project exclusions. Later source edits belong to the parent "
                    "repository; each run must record its parent revision and relevant diff or snapshot. "
                    "No nested Git history is required or included.",
        }
        (provenance / "dppo_source_snapshot.json").write_text(json.dumps(record, indent=2) + "\n")
        results.mkdir(exist_ok=True)
        # Exclusive creation refuses even an empty study; copytree preserves links
        # and modes across filesystems without touching either Git index.
        destination.mkdir()
        try:
            shutil.copytree(exported, destination / "dppo", symlinks=True)
            shutil.copytree(provenance, destination / "provenance")
        except BaseException:
            shutil.rmtree(destination)
            raise
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="local checkout or archived Git directory")
    parser.add_argument("--revision", required=True, help="approved source commit or ref to export")
    parser.add_argument("--study", required=True, help="new directory name under results/")
    parser.add_argument("--subdir", default=".", help="source subtree, e.g. results/previous/dppo in a parent commit")
    args = parser.parse_args()
    record = create_snapshot(Path(__file__).resolve().parents[1], args.source, args.revision, args.study, args.subdir)
    print("Created {}: {} files from {} (no nested .git).".format(
        record["source_directory"], record["snapshot_file_count"], record["commit"]))
    print("Provenance: results/{}/provenance/dppo_source_snapshot.json".format(args.study))
    print("No files staged. Review parent ignore rules before the first commit.")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError) as exc:
        print("Error: {}".format(exc), file=sys.stderr)
        sys.exit(1)

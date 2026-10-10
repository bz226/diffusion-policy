"""Address DPPO's own Git history explicitly, including its local archive."""

import json
from pathlib import Path


def source_git_command(repo, *args):
    """Build a Git command without discovering the enclosing project repository.

    Completed studies may move their nested metadata into an ignored backup.
    The provenance record maps that backup to this exact source directory;
    a custom --repo must have its own metadata or matching archive record.
    """
    repo = Path(repo).resolve()
    metadata = repo / ".git"
    if not metadata.exists():
        record_path = repo.parent / "provenance" / "dppo_git_archive.json"
        if not record_path.is_file():
            raise RuntimeError(
                "No DPPO Git metadata for {}. Restore its local Git backup or "
                "use an independent checkout; archive record missing: {}".format(repo, record_path)
            )
        try:
            record = json.loads(record_path.read_text())
            source_path = record["source_directory"]
            archive_path = record["archived_git_directory"]
            if not all(isinstance(path, str) and path for path in (source_path, archive_path)):
                raise ValueError("source_directory and archived_git_directory must be nonempty paths")
            if any(Path(path).is_absolute() for path in (source_path, archive_path)):
                raise ValueError("archive paths must be relative to the project root")
            project = repo.parents[2]
            source = (project / source_path).resolve()
            metadata = (project / archive_path).resolve()
            source.relative_to(project)
            metadata.relative_to(project)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            raise RuntimeError("Invalid DPPO Git archive record {}: {}".format(record_path, exc)) from exc
        if source != repo:
            raise RuntimeError(
                "DPPO Git archive record {} belongs to {}, not requested source {}".format(
                    record_path, source, repo
                )
            )
        if not metadata.is_dir():
            raise RuntimeError(
                "Archived DPPO Git metadata is unavailable at {}. Restore that local backup "
                "or use an independent checkout; the parent repository cannot replace its history.".format(
                    metadata
                )
            )
    return ["git", "-C", str(repo), "--git-dir", str(metadata), "--work-tree", str(repo), *args]

# Repository instructions

Follow `research_agent_workflow.md` for research work. Repository maintenance does not authorize new experiments.

Use `main` for this workspace's ongoing work unless the user requests another branch. Do not create or switch to a study branch by default. Bind research provenance to the approved revision and source contents; administrative branch-name checks must allow `main`.

## Creating study source

Experiment source and selected results belong in the parent repository as ordinary files, including `results/<study>/dppo`. Do not turn these checkouts into submodules or exclude their source to suppress an embedded-repository warning.

**Create every new study with ordinary source files from the start.** Do not run `git clone`, `git init`, `git worktree add`, or `git submodule add` inside a study directory, and never copy a `.git` directory or pointer into one. Do not introduce provenance checks that require a nested repository.

Use `scripts/create_dppo_snapshot.py` as the first study scaffolding step, before creating its directory. Choose the source and revision for the authorized study. For example, to reuse source already committed in this parent repository (replace `NEW_STUDY` and select the appropriate revision/source subtree):

```bash
python3 scripts/create_dppo_snapshot.py --source .git --revision HEAD \
  --subdir results/stage2_noclip/dppo --study NEW_STUDY
```

The helper can also export an independent local checkout or archived Git directory with `--source PATH --revision COMMIT --study NEW_STUDY` (omit `--subdir`). If upstream source must be downloaded, put that checkout under the ignored `.codex-runtime/upstream/` directory, then export it with the helper. Never clone directly into `results/`.

The helper preserves committed source and executable bits, stores the imported revision in `provenance/dppo_source_snapshot.json`, and preserves original upstream ignore files separately. Its replacement ignore rules keep source scripts and selected results versionable while excluding runtime logs, generated data, checkpoints, and caches. It refuses existing study directories; never delete an existing study to make the command succeed.

For subsequent edits and per-run provenance, use the parent repository's revision plus the relevant study diff or a source snapshot. The imported commit records the initial source only. Refer explicitly to the parent repository when running Git commands; do not mistake `git -C results/<study>/dppo rev-parse HEAD` for a separate DPPO revision. Keep run outputs outside the source directory. Verify that new source contains no nested `.git` and that intended source is included before its first parent commit.

## Staging and existing checkouts

New snapshots support ordinary `git add -A`. While this repository also contains legacy nested checkouts, use this command from the project root for updates across studies:

```bash
python3 scripts/git_add_all.py
```

Use `--dry-run` to preview. The helper handles legacy `results/*/dppo` checkouts, replaces accidental Git repository links in the parent index with ordinary source files, respects parent exclusions, and preserves nested history. It untracks runtime logs under `results/` without deleting local files, prevents older child ignore rules from adding them again, and rejects prospective files larger than GitHub's 100 MiB limit before staging. Use the helper for routine updates even after nested checkouts are gone. Its availability is not permission to create new nested checkouts.

Keep runtime logs local: console and scheduler output (`*.log`, rotated `*.log.*`, `*.out`, `*.err`), command transcripts (`commands.jsonl`, `*_commands.jsonl`), monitor histories (`monitor_history.jsonl`, `*_monitor_history.jsonl`, `*_monitor.jsonl`), and controller `events.jsonl` under `results/`. Do not force-add these files. Keep source, configurations, reports, selected metrics (including `result.pkl` and scientific `replicates.jsonl`), and concise provenance such as `runtime_estimates.jsonl` versionable. Do not blanket-ignore JSONL files or entire `log/` directories. New log formats should get explicit ignore rules and matching staging-helper exclusions.

Do not delete or truncate live logs to make a push succeed. An ignore rule does not untrack an existing file or remove historical blobs: the staging helper handles untracking, but rejected commits containing oversized logs still require an unpublished-history repair. Preserve original commits in local backup refs and push only the intended branch. Never reset or check out active experiment files as part of a history repair.

Preserve existing nested `.git` metadata while runs, verification, or finalization depend on it. Follow the research workflow when archiving a completed study's metadata. Do not modify active experiments merely to change their source storage.

If the parent `.git` directory is read-only, run the helper with `--dry-run`, verify the intended changes, and give the user the command above to run in their normal terminal. Do not claim that a dry run staged files.

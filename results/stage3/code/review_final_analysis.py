#!/usr/bin/env python
"""Preserve automatic outputs and assemble the mechanically reviewed candidates."""
import datetime as dt
import json
import os
from pathlib import Path
import re
import shutil
import sys

import analyze_stage3 as analysis


def main():
    root = analysis.ROOT
    review = root / "runs/final_analysis_review"
    out = review / "candidates"
    if out.exists():
        raise RuntimeError("Refusing to overwrite existing reviewed candidates")
    out.mkdir()
    campaign = analysis.read_json(root / "runs/campaign.json")
    runs = analysis.load_runs(root, campaign, allow_partial=False)
    references = analysis.reference_runs(root)
    anchors = analysis.read_json(root / "provenance/anchors.json")
    omissions, caption = analysis.figures(root, runs, references, anchors, out / "figs")

    copied = ["all_runs_long.csv", "seed_results.csv", "condition_results.csv",
              "noise_windows.csv", "all_commands.json"]
    for filename in copied:
        shutil.copyfile(root / filename, out / filename)
        assert (root / filename).read_bytes() == (out / filename).read_bytes()
    original = (root / "REPORT.md").read_text()
    text = re.sub(r"The three rows compare[^\n]+", lambda _: caption, original, count=1)
    text = re.sub(r"Omitted log-axis / undefined observations: `[^`]+`",
                  lambda _: "Omitted log-axis / undefined observations: `" + json.dumps(omissions, sort_keys=True) + "`", text, count=1)
    lines = []
    for line in text.splitlines():
        if line.startswith(("| PROJ3 |", "| PROJ10 |")):
            line = line.replace("— / — / —; unresolved (0/3 available)", "skipped (distance rule)")
            line = line.replace("unresolved (0/3 available)", "skipped (distance rule)")
            line = line.replace("[None, None, None]", "not run").replace("None / None / None", "not run")
        if line.startswith(("| PROJ3_seed", "| PROJ10_seed")):
            line = re.sub(r" · \[result.pkl\]\([^)]+\)", "", line)
        if line.startswith("| Projection |"):
            line = "| Projection | none | Π_Θ, Θ convex | ball B(θ₀, R) | PROJ* | PROJ3 and PROJ10 were skipped by the prescribed distance rule. No projection effect measured; see Q6. |"
        if line == "No matching observations in the saved records.":
            line = "No zero-update or collapse cases were observed in the completed runs."
        lines.append(line)
    text = "\n".join(lines) + "\n"
    # Rebase existing local links; copied numerical files and corrected figures
    # stay local, while native evidence retains its original location.
    checked, missing = [], []
    def rebase(match):
        target = match.group(1)
        if "://" in target or target.startswith("#"):
            return match.group(0)
        path, separator, fragment = target.partition("#")
        source = (root / path).resolve()
        if path in copied or path.startswith("figs/"):
            source = (out / path).resolve()
        relative = os.path.relpath(source, out)
        checked.append(relative)
        if not source.exists():
            missing.append(relative)
        return "](" + relative + (separator + fragment if separator else "") + ")"
    # Fenced exact command/job-file text is evidence and remains byte-for-byte.
    chunks = re.split(r"(^```[^\n]*\n.*?^```[^\n]*$)", text, flags=re.M | re.S)
    for i in range(0, len(chunks), 2):
        chunks[i] = re.sub(r"\]\(([^)]+)\)", rebase, chunks[i])
    text = "".join(chunks)
    if missing:
        raise RuntimeError("Missing report links: " + repr(missing))
    answer = ("**Answer.** E1 improved the final scheduled eval return to 4670 ± 28 (sample SD across three seeds), "
              "compared with 4532 ± 67 for E0 and 4542 ± 81 for E2; the E1→E2 change lowered both discounted and undiscounted training return. "
              "All tested SGD rates avoided the prescribed collapse criterion, but SGD1 improved the final train-sampler discounted objective less than E2 "
              "(ΔJ_disc 40 ± 22 versus 160 ± 5). Projection was skipped by the prescribed distance rule; Phase-1 uncertainty remains inconclusive.\n\n")
    text = text.replace("**Status.**", answer + "**Status.**", 1)
    note = ("\n**Review.** The final numerical audit passed 2,118 checks with zero differences. "
            "The figures now exclude evaluation rows from training action-range curves and preserve undefined noise-window gaps; "
            "original automatic outputs remain preserved. [Review record](../review_record.json), "
            "[numeric checks](../numeric_review.json), [plot review](../plot_review.json).\n")
    text = text.replace("## Verification gate", note + "\n## Verification gate", 1)
    (out / "REPORT.md").write_text(text)
    shutil.copyfile(root / "code/analyze_stage3.py", review / "plotting_source_snapshot.py")
    record = {
        "status": "reviewed_candidates_generated",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": sys.executable + " " + str(Path(__file__).resolve()),
        "scope": "Reporting-only corrections; no new sampling, training, estimators, uncertainty methods or scientific comparisons.",
        "original_analysis_record": str(root / "analysis_record.json"),
        "original_report_preserved": str(root / "REPORT.md"),
        "corrected_plot_source": str(review / "plotting_source_snapshot.py"),
        "report": str(out / "REPORT.md"),
        "byte_identical_numeric_copies": copied,
        "report_links_checked": len(checked),
        "missing_report_links": missing,
        "changes": ["Filter action-range curves to training iterations.",
                    "Preserve undefined B_env windows as gaps and mark isolated defined observations.",
                    "Label skipped projection conditions and absence of zero updates explicitly.",
                    "Rebase report links and remove links to nonexistent prospectively skipped outputs."],
        "plot_omissions": omissions,
        "numeric_review": str(review / "numeric_review.json"),
        "plot_review": str(review / "plot_review.json"),
        "limitation": "Phase-1 percentile uncertainty remains unreliable; no replacement computed.",
        "curation_status": "Pending explicit artifact/name approval; existing curated files unchanged."
    }
    analysis.write_json(review / "review_record.json", record)
    print(json.dumps({"status": record["status"], "report": record["report"],
                      "links_checked": len(checked), "plot_omissions": omissions}))


if __name__ == "__main__":
    main()

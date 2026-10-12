"""Verify published clean-v3 artifacts and independently recompute their table.

Read-only except a new checks.json beside the immutable render candidates.
Use --require-all for the final seven-point measurement, after the renderer.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import re

from PIL import Image

from check_clean_v3_render import verify


FIGURES = ("eval_reward", "train_reward", "gradient_noise_V6", "gradient_noise_V4", "gradient_noise_V5")
METHODS = ("DPPO", "without_PPO", "without_PPO_and_GAE")
POINTS = ("P0", "D1", "D2", "N1", "N2", "C1", "C2")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/soalnas/share/data/zbao7/diffusion_policy/results/stage3/gradmean"))
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    published = args.root.resolve() / "figs_clean"
    record = json.loads((published / "render_record.json").read_text())
    run = Path(record["run"])
    candidates = run / "candidates"
    output = run / "checks.json"
    if output.exists():
        output = run / ("checks_revalidation_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + ".json")
    expected = 7 if args.require_all else None
    result = verify(candidates, expected)
    rows = list(csv.DictReader((candidates / "points_table.csv").open(newline="")))
    if len(rows) != 21 or {row["point"] for row in rows} != set(POINTS):
        raise ValueError("Expected three estimator rows per measurement point")
    if args.require_all and any(row["K"] != "500" or row["groups"] != "10" for row in rows):
        raise ValueError("Final table contains an incomplete cohort")
    if args.require_all and result["compared_numeric_cells"] != 1008:
        raise ValueError("Final table recomputation did not cover all expected cells")
    if set(record["rewards"]["curves"]) != set(METHODS):
        raise ValueError("The figure includes the wrong training methods")
    files = []
    for name in FIGURES:
        for suffix in (".png", ".pdf", ".caption.md"):
            candidate, current = candidates / (name + suffix), published / (name + suffix)
            if not candidate.is_file() or candidate.read_bytes() != current.read_bytes():
                raise ValueError("Published figure differs from its recorded candidate: " + name + suffix)
            files.append(str(candidate))
        with Image.open(str(candidates / (name + ".png"))) as image:
            dpi = image.info.get("dpi")
            if dpi is None or any(abs(value - 300) > .02 for value in dpi):
                raise ValueError("PNG is not 300dpi: " + name)
            if not image.info.get("Description"):
                raise ValueError("PNG lacks its embedded caption: " + name)
        if not (candidates / (name + ".pdf")).read_bytes().startswith(b"%PDF"):
            raise ValueError("Invalid PDF: " + name)
        caption = (candidates / (name + ".caption.md")).read_text()
        if not caption.strip() or re.search(r"\b(?:NC4|E1C)\b", caption):
            raise ValueError("Missing caption or unapproved method label: " + name)
        if name.startswith("gradient_noise") and "500 batches of 80" not in caption:
            raise ValueError("Gradient figure lacks the shared sample size")
    for filename in ("points_table.csv", "points_table.md", "NOTE.md", "render_record.json"):
        if (candidates / filename).read_bytes() != (published / filename).read_bytes():
            raise ValueError("Published artifact differs from candidate: " + filename)
        files.append(str(candidates / filename))
    # Independently verify native checkpoint counters and reward log values.
    native_checks = []
    for middle in ("D1", "N1", "C1"):
        mapping = record["mapping"][middle]
        path = Path(mapping["result_path"])
        with path.open("rb") as stream:
            native = pickle.load(stream)
        if [row["itr"] for row in native] != list(range(140)):
            raise ValueError("Seed-0 reference has incomplete native iterations")
        for point, itr, updates in ((middle, 70, 63), (middle[0] + "2", 139, 126)):
            count = sum("train_episode_reward" in row for row in native if row["itr"] <= itr)
            saved = next(row for row in native if row["itr"] == itr)
            entry = record["mapping"][point]
            if count != updates or entry["update_number"] != updates or saved["step"] != 80000 * updates:
                raise ValueError("Native checkpoint update mapping mismatch: " + point)
            if not Path(entry["checkpoint_path"]).is_file():
                raise ValueError("Checkpoint missing: " + point)
            native_checks.append(dict(point=point, checkpoint=entry["checkpoint_path"], updates=updates, source=str(path)))
        final = next(row for row in native if row["itr"] == 130)
        if final["step"] != 9360000 or sum("train_episode_reward" in row for row in native if row["itr"] < 130) != 117:
            raise ValueError("Final evaluation update mapping mismatch")
        matched = re.findall(r"eval:.*?avg episode reward\s+([-+0-9.eE]+)", (path.parent / "run.log").read_text())
        if not matched or abs(float(matched[-1]) - final["eval_episode_reward"]) > .00005001:
            raise ValueError("Final evaluation fails its native log sanity check")
    if record["mapping"]["P0"]["update_number"] != 0:
        raise ValueError("Shared pretrained origin is not zero updates")
    result.update(require_all=args.require_all, artifact_files=files, png_dpi=300,
                  native_checkpoint_checks=native_checks, source_script=str(Path(__file__).resolve()),
                  independent_recompute_source=str(Path(__file__).with_name("check_clean_v3_render.py").resolve()))
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": True, "output": str(output), "completed_points": result["completed_points"],
                      "compared_numeric_cells": result["compared_numeric_cells"],
                      "max_relative_difference": result["max_relative_difference"]}, indent=2))


if __name__ == "__main__":
    main()

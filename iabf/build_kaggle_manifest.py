#!/usr/bin/env python3
"""Build an audited manifest from the Kaggle-hosted processed Multi-PIE subset.

Accepts CMU-style names with or without a _crop_<resolution> suffix, e.g.
001_02_01_051_00_crop_128.png. Camera IDs and angles are the
documented Idiap/Bob pose analysis mapping. No inferred or guessed labels.
This implements a custom subset evaluation, NOT official Bob Protocol P.
"""
import argparse
import csv
import json
import re
from pathlib import Path
from collections import Counter, defaultdict

CAMERA_TO_YAW = {
    "11_0": -90, "12_0": -75, "09_0": -60, "08_0": -45,
    "13_0": -30, "14_0": -15, "05_1": 0, "05_0": 15,
    "04_1": 30, "19_0": 45, "20_0": 60, "01_0": 75, "24_0": 90,
}
PATTERN = re.compile(
    r"^(?P<subject>\d{3,5})_(?P<session>\d{2})_(?P<recording>\d{2})_"
    r"(?P<camera>\d{3}|\d{2}_\d)_(?P<shot>\d{2})"
    r"(?:_crop_\d+)?\.(?:png|jpg|jpeg)$", re.I
)

def parse(path):
    m = PATTERN.fullmatch(path.name)
    if not m:
        return None
    d = m.groupdict()
    c = d["camera"]
    camera = c if "_" in c else c[:2] + "_" + c[2]
    if camera not in CAMERA_TO_YAW:
        return None
    if int(d["session"]) not in (1, 2, 3, 4) or int(d["shot"]) > 19:
        return None
    return {
        "subject": d["subject"], "camera": camera,
        "yaw_deg": CAMERA_TO_YAW[camera], "session": d["session"],
        "recording": d["recording"], "illumination": d["shot"]
    }

def create(root):
    images = [p for p in root.rglob("*") if p.is_file()
              and p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    images.sort()
    by_condition = defaultdict(list)
    unparsed = []
    for path in images:
        data = parse(path)
        if data is None:
            if len(unparsed) < 30:
                unparsed.append(path.relative_to(root).as_posix())
            continue
        data["path"] = path.relative_to(root).as_posix()
        by_condition[(data["subject"], data["camera"], data["session"])].append(data)
    # Keep pose comparisons reasonably manageable and prioritize neutral/no-flash.
    # This is a deliberate custom subset selection, not the official protocol.
    def priority(d):
        shot = int(d["illumination"])
        return (int(d["recording"]) != 1, shot not in (0, 19),
                shot != 0, shot, d["path"])
    rows = sorted((min(group, key=priority) for group in by_condition.values()),
                  key=lambda x: x["path"])
    by_subject = defaultdict(list)
    for r in rows:
        by_subject[r["subject"]].append(r)
    eligible = [key for key, group in by_subject.items()
                if sum(r["yaw_deg"] == 0 for r in group) >= 1
                and sum(r["yaw_deg"] != 0 for r in group) >= 4]
    usable = [key for key, group in by_subject.items()
              if any(r["yaw_deg"] == 0 for r in group)
              and any(r["yaw_deg"] != 0 for r in group)]
    aud = {
        "dataset": "aliates/multi-pie Kaggle processed subset",
        "protocol": "custom, 50 total adaptation images; not official Bob Protocol P",
        "selection_rule": "one image per subject/session/camera; prioritize neutral recording and no-flash shot",
        "image_files_found": len(images),
        "parsed_images_before_selection": sum(map(len, by_condition.values())),
        "images_in_manifest": len(rows),
        "unparsed_image_count": len(images)-sum(map(len, by_condition.values())),
        "unparsed_examples": unparsed,
        "subjects": len(by_subject),
        "adaptation_eligible_subjects": len(eligible),
        "frontal_and_rotated_subjects": len(usable),
        "cameras": dict(sorted(Counter(r["camera"] for r in rows).items())),
        "yaw_degrees": sorted({r["yaw_deg"] for r in rows}),
        "supports_50_image_protocol": len(eligible) >= 10 and len(usable) >= 14,
    }
    return rows, aud

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--audit", type=Path, required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        ap.error(f"Missing dataset root: {root}")
    rows, audit = create(root)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2) + "\n")
    print("IABF_KAGGLE_INVENTORY", json.dumps(audit), flush=True)
    if not rows:
        raise SystemExit("No correctly labeled Multi-PIE filenames found; cannot infer pose.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cols = ["path", "subject", "camera", "yaw_deg", "session",
            "recording", "illumination"]
    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows({k: r[k] for k in cols} for r in rows)
    if not audit["supports_50_image_protocol"]:
        raise SystemExit(
            "Kaggle upload lacks >=10 subjects with 1 frontal + 4 rotated images "
            "and >=14 subjects with both views. No GPU model fitting attempted."
        )
    print("IABF_MANIFEST_READY", len(rows), flush=True)

if __name__ == "__main__":
    main()

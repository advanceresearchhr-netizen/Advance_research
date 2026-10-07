#!/usr/bin/env python3
"""Paper 2 NTU RGB+D 60 preflight. No training and no paid GPU required."""
from pathlib import Path
import argparse, json, re, sys

PAT=re.compile(r"^(S\d{3}C\d{3}P\d{3}R\d{3}A\d{3})")

def ids(root, suffixes):
    out=set()
    for p in Path(root).rglob("*"):
        if p.is_file() and p.suffix.lower() in suffixes:
            m=PAT.match(p.stem)
            if m: out.add(m.group(1))
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--rgb", required=True)
    ap.add_argument("--skeleton", required=True)
    ap.add_argument("--out", default="paper2/preflight_report.json")
    a=ap.parse_args()
    rgb=ids(a.rgb,{".avi",".mp4",".mkv"})
    sk=ids(a.skeleton,{".skeleton"})
    paired=rgb & sk
    rep={
      "rgb_samples":len(rgb),"skeleton_samples":len(sk),"paired_samples":len(paired),
      "rgb_without_skeleton":len(rgb-sk),"skeleton_without_rgb":len(sk-rgb),
      "pass": len(paired)>1000 and len(rgb-sk)==0
    }
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(rep,indent=2))
    print(json.dumps(rep,indent=2))
    if not rep["pass"]:
        sys.exit("PREFLIGHT_FAILED: synchronized NTU RGB+skeleton corpus not ready")
if __name__=="__main__": main()

"""Fail-closed validation of Paper 3 prediction artifacts before publication."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

KEYS=["seed","condition","severity","variant","video_id"]
REQUIRED=KEYS+["class_id","prediction","correct","confidence"]

def audit_predictions(frame,expected_test_size=1068):
    missing=set(REQUIRED)-set(frame.columns)
    if missing:raise ValueError(f"Missing columns: {sorted(missing)}")
    df=frame.copy()
    if df[REQUIRED].isna().any().any():raise ValueError("Null prediction fields")
    df["video_id"]=df["video_id"].astype(str).str.zfill(4)
    if df.duplicated(KEYS).any():raise ValueError("Duplicate prediction identity")
    if not df["class_id"].between(0,14).all() or not df["prediction"].between(0,14).all():
        raise ValueError("Class indices outside [0,14]")
    if not df["confidence"].between(0,1).all():raise ValueError("Invalid confidence")
    if not np.array_equal(df["correct"].to_numpy().astype(int),(df["class_id"]==df["prediction"]).to_numpy().astype(int)):
        raise ValueError("Incorrect correctness flags")
    reference=None
    groups=0
    for key,part in df.groupby(["seed","condition","severity","variant"],dropna=False):
        if len(part)!=expected_test_size:raise ValueError(f"Incomplete evaluation {key}: {len(part)} != {expected_test_size}")
        labels=part.set_index("video_id")["class_id"].sort_index()
        if reference is None:reference=labels
        elif not reference.equals(labels):raise ValueError(f"Video IDs or ground truth differ: {key}")
        groups+=1
    if groups==0:raise ValueError("No predictions")
    return {"valid":True,"groups":groups,"videos_per_group":expected_test_size,
            "seeds":sorted(df.seed.astype(int).unique().tolist()),
            "variants":sorted(df.variant.unique().tolist()),
            "conditions":sorted(df.condition.unique().tolist())}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--predictions",required=True)
    parser.add_argument("--out",required=True)
    parser.add_argument("--expected-test-size",type=int,default=1068)
    args=parser.parse_args()
    report=audit_predictions(pd.read_csv(args.predictions,dtype={"video_id":str}),args.expected_test_size)
    Path(args.out).write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()

"""Paired clean/corrupted reliability audit from saved score rows.
Scores are sigmoid reliability (higher means healthier), not corruption probability.
No labels, checkpoints or test data are used for model selection.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score

def paired_quality_audit(frame,n_boot=1000,seed=2026):
    required={"seed","condition","severity","modality","video_id","state","reliability"}
    if missing:=required-set(frame.columns):raise ValueError(f"Missing columns: {sorted(missing)}")
    df=frame.copy()
    if df[list(required)].isna().any().any():raise ValueError("Null reliability rows")
    if not df.reliability.between(0,1).all():raise ValueError("Reliability must be in [0,1]")
    if not set(df.state.unique()).issubset({"clean","corrupt"}):raise ValueError("State must be clean/corrupt")
    df["video_id"]=df.video_id.astype(str).str.zfill(4)
    keys=["seed","condition","severity","modality","video_id","state"]
    if df.duplicated(keys).any():raise ValueError("Duplicate sample/state")
    out=[]
    for group,part in df.groupby(["seed","condition","severity","modality"]):
        pivot=part.pivot(index="video_id",columns="state",values="reliability")
        if set(pivot.columns)!={"clean","corrupt"} or pivot.isna().any().any():
            raise ValueError(f"Missing paired clean/corrupt samples: {group}")
        clean=pivot["clean"].to_numpy();bad=pivot["corrupt"].to_numpy()
        y=np.r_[np.zeros(len(clean),dtype=int),np.ones(len(bad),dtype=int)]
        score=np.r_[1-clean,1-bad]
        auc=float(roc_auc_score(y,score))
        rng=np.random.default_rng(seed+int(group[0]))
        draws=[]
        for _ in range(n_boot):
            idx=rng.integers(0,len(clean),len(clean))
            yy=np.r_[np.zeros(len(idx),dtype=int),np.ones(len(idx),dtype=int)]
            zz=np.r_[1-clean[idx],1-bad[idx]]
            draws.append(roc_auc_score(yy,zz))
        out.append(dict(seed=int(group[0]),condition=group[1],severity=float(group[2]),
            modality=group[3],n_paired_videos=len(clean),auroc=auc,
            auprc=float(average_precision_score(y,score)),
            auroc_ci95_low=float(np.percentile(draws,2.5)),
            auroc_ci95_high=float(np.percentile(draws,97.5)),
            mean_clean_reliability=float(clean.mean()),
            mean_corrupt_reliability=float(bad.mean()),
            mean_reliability_change=float((bad-clean).mean()),
            polarity_warning=bool(auc<.5)))
    return pd.DataFrame(out)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--bootstrap",type=int,default=1000)
    args=p.parse_args()
    if args.bootstrap<1:raise ValueError("bootstrap must be positive")
    result=paired_quality_audit(pd.read_csv(args.input,dtype={"video_id":str}),args.bootstrap)
    result.to_csv(args.out,index=False)
    print(json.dumps({"groups":len(result),"polarity_warnings":int(result.polarity_warning.sum())},indent=2))
if __name__=="__main__":main()

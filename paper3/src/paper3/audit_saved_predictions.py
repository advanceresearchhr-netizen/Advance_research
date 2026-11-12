"""Audit saved Paper 3 predictions. No training, no GPU, no test-set tuning."""
import argparse
from pathlib import Path
import pandas as pd
from scipy.stats import binomtest
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--results",required=True)
 p.add_argument("--out",required=True)
 a=p.parse_args()
 root=Path(a.results);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 d=pd.read_csv(root/"predictions_by_video.csv")
 assert not d.duplicated(["seed","condition","severity","variant","video_id"]).any()
 rows=[]
 for (seed,condition,severity),g in d.groupby(["seed","condition","severity"]):
  x=g[g.variant=="legacy_zero"].set_index("video_id")
  y=g[g.variant=="v2"].set_index("video_id")
  if x.empty or y.empty:continue
  assert x.index.equals(y.index) or set(x.index)==set(y.index)
  z=x[["correct","class_id"]].join(y[["correct","class_id"]],lsuffix="_reference",rsuffix="_v2",validate="one_to_one")
  assert (z.class_id_reference==z.class_id_v2).all()
  n=len(z);b=int(((z.correct_reference==1)&(z.correct_v2==0)).sum());c=int(((z.correct_reference==0)&(z.correct_v2==1)).sum())
  rows.append({"seed":seed,"condition":condition,"severity":severity,"n":n,"reference_accuracy":z.correct_reference.mean(),"v2_accuracy":z.correct_v2.mean(),"delta_percentage_points":100*(z.correct_v2.mean()-z.correct_reference.mean()),"reference_only_correct":b,"v2_only_correct":c,"exact_mcnemar_p_exploratory":binomtest(min(b,c),b+c,0.5).pvalue if b+c else 1.0})
 r=pd.DataFrame(rows);r.to_csv(out/"paired_tests_recomputed.csv",index=False)
 q=pd.read_csv(root/"clean_vs_corrupt_detection.csv")
 q.to_csv(out/"reliability_auroc_as_recorded.csv",index=False)
 assert r.shape[0]>=9
 print("PAIRED_AUDIT_COMPLETE",len(r),"matched seed-condition-severity comparisons")
 print(r[r.condition.isin(["clean","missing_rgb","missing_pose"])].to_string(index=False))
 print("AUROC direction NOT verified: requires scoring-label implementation audit")
if __name__=="__main__":main()

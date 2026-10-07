#!/usr/bin/env python3
from pathlib import Path
import re,json,sys,zipfile,hashlib
PAT=re.compile(r"a(\d+)_s(\d+)_t(\d+)",re.I)
def scan(zpath,kind):
    with zipfile.ZipFile(zpath) as z:
        bad=z.testzip()
        if bad: raise SystemExit(f"{kind} corrupt member: {bad}")
        rows={}
        for n in z.namelist():
            m=PAT.search(Path(n).stem)
            if m:
                a,s,t=map(int,m.groups()); rows[(a,s,t)]=n
        return rows
rgb=scan("RGB.zip","rgb"); sk=scan("Skeleton.zip","skeleton")
paired=set(rgb)&set(sk); union=set(rgb)|set(sk)
train={x for x in paired if x[1] in {1,3,5,7}}
test={x for x in paired if x[1] in {2,4,6,8}}
rep={"rgb_ids":len(rgb),"skeleton_ids":len(sk),"paired_ids":len(paired),
     "union_ids":len(union),"rgb_only":len(set(rgb)-set(sk)),"skeleton_only":len(set(sk)-set(rgb)),
     "train_ids":len(train),"test_ids":len(test),
     "actions":sorted({x[0] for x in paired}),"subjects":sorted({x[1] for x in paired}),
     "trials":sorted({x[2] for x in paired})}
rep["pass"]=len(paired)==861 and rep["rgb_only"]==0 and rep["skeleton_only"]==0 and rep["actions"]==list(range(1,28)) and rep["subjects"]==list(range(1,9))
Path("paper2/data_integrity.json").write_text(json.dumps(rep,indent=2))
print(json.dumps(rep,indent=2))
if not rep["pass"]: sys.exit("DATA_INTEGRITY_FAILED")

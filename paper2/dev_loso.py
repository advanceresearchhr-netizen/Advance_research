#!/usr/bin/env python3
"""Development-only LOSO selection on frozen UTD training subjects. Never reads final-test subjects."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
sys.path.insert(0,str(Path(__file__).parent))
from experiment import UTD,FaithfulHAR,seed_all,metrics
TRAIN=[1,3,5,7]
def cache(rgbzip,skzip,out):
 out=Path(out)
 if out.exists():
  d=torch.load(out,map_location="cpu"); assert set(d["subjects"].tolist())<=set(TRAIN); return d
 ds=UTD(rgbzip,skzip,TRAIN); R=[];S=[];Y=[];K=[]
 for i in range(len(ds)):
  r,s,y,k=ds[i]; R.append(r.half());S.append(s);Y.append(y);K.append(k)
  if (i+1)%50==0: print(f"CACHE {i+1}/{len(ds)}",flush=True)
 d={"rgb":torch.stack(R),"sk":torch.stack(S),"y":torch.stack(Y),"keys":torch.stack(K)}
 d["subjects"]=d["keys"][:,1]; assert len(d["y"])==431 and set(d["subjects"].tolist())==set(TRAIN)
 torch.save(d,out); return d
@torch.no_grad()
def evaluate(m,dl,dev):
 m.eval(); L=[];Y=[]
 for r,s,y in dl: L.append(m(r.float().to(dev),s.to(dev)).cpu());Y.append(y)
 return metrics(torch.cat(L),torch.cat(Y))
def main(a):
 d=cache(a.rgb,a.skel,a.cache); dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 print("DEVICE",dev,"CUDA",torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",flush=True)
 rows=[]
 for seed in a.seeds:
  for held in TRAIN:
   seed_all(seed); tr=d["subjects"]!=held; va=d["subjects"]==held
   train=TensorDataset(d["rgb"][tr],d["sk"][tr],d["y"][tr]); val=TensorDataset(d["rgb"][va],d["sk"][va],d["y"][va])
   g=torch.Generator().manual_seed(seed); tl=DataLoader(train,batch_size=a.batch,shuffle=True,generator=g); vl=DataLoader(val,batch_size=a.batch)
   m=FaithfulHAR().to(dev); opt=torch.optim.AdamW(m.parameters(),lr=a.lr,weight_decay=1e-4)
   best=None
   for ep in range(a.epochs):
    m.train(); losses=[]
    for r,s,y in tl:
     r,s,y=r.float().to(dev),s.to(dev),y.to(dev); opt.zero_grad(set_to_none=True); z=m(r,s); loss=nn.functional.cross_entropy(z,y); loss.backward(); opt.step(); losses.append(loss.detach().item())
    met=evaluate(m,vl,dev)
    if best is None or met["macro_f1"]>best["macro_f1"]: best={**met,"epoch":ep+1}
    if ep==0 or (ep+1)%5==0: print(f"LOSO seed={seed} held={held} epoch={ep+1} loss={np.mean(losses):.4f} f1={met['macro_f1']:.4f}",flush=True)
   rows.append({"seed":seed,"held_subject":held,**best}); print("FOLD_DONE",json.dumps(rows[-1]),flush=True)
 out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rows,indent=2))
 summary={"folds":len(rows),"mean_top1":float(np.mean([r["top1"] for r in rows])),"mean_macro_f1":float(np.mean([r["macro_f1"] for r in rows])),"epochs":a.epochs,"seeds":a.seeds,"final_test_touched":False}
 (out.parent/"dev_loso_summary.json").write_text(json.dumps(summary,indent=2)); print("PAPER2_DEV_RESULTS",json.dumps(summary),flush=True); print("PAPER2_DEV_DONE",flush=True)
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--rgb",default="RGB.zip");p.add_argument("--skel",default="Skeleton.zip");p.add_argument("--cache",default="/workspace/paper2/utd_train_cache.pt");p.add_argument("--out",default="/workspace/paper2/results/dev_loso.json");p.add_argument("--epochs",type=int,default=20);p.add_argument("--batch",type=int,default=16);p.add_argument("--lr",type=float,default=3e-4);p.add_argument("--seeds",type=int,nargs="+",default=[42,123,2026]);main(p.parse_args())

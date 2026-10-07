#!/usr/bin/env python3
"""Paper 2 production experiment core. Frozen UTD-MHAD protocol."""
import os, re, json, random, argparse, zipfile, tempfile
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import cv2, scipy.io
PAT=re.compile(r"a(\d+)_s(\d+)_t(\d+)",re.I); TRAIN={1,3,5,7}; TEST={2,4,6,8}
def seed_all(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
def index_zip(z): return {tuple(map(int,PAT.search(Path(n).stem).groups())):n for n in z.namelist() if PAT.search(Path(n).stem)}
def decode_rgb(path,T=8,S=112):
 cap=cv2.VideoCapture(str(path)); fs=[]
 while True:
  ok,f=cap.read()
  if not ok: break
  fs.append(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))
 cap.release(); assert fs
 ids=np.linspace(0,len(fs)-1,T).round().astype(int)
 a=np.stack([cv2.resize(fs[i],(S,S)) for i in ids]).astype("float32")/255.
 a=(a-np.array([.43216,.394666,.37645],np.float32))/(np.array([.22803,.22145,.216989],np.float32)+1e-6)
 return torch.from_numpy(a).permute(3,0,1,2)
def decode_skel(path,T=32):
 x=scipy.io.loadmat(path)["d_skel"].astype("float32").transpose(2,0,1)
 x=x[np.linspace(0,len(x)-1,T).round().astype(int)]; x=x-x[:,0:1]; x=x/(np.linalg.norm(x,axis=-1).mean()+1e-6)
 return torch.from_numpy(x)
class UTD(Dataset):
 def __init__(self,rgbzip,skzip,subjects,limit=0):
  self.rz=zipfile.ZipFile(rgbzip); self.sz=zipfile.ZipFile(skzip); R=index_zip(self.rz); S=index_zip(self.sz)
  self.R,self.S=R,S; self.keys=[k for k in sorted(set(R)&set(S)) if k[1] in set(subjects)]
  if limit:self.keys=self.keys[:limit]
  self.tmp=tempfile.TemporaryDirectory()
 def __len__(self): return len(self.keys)
 def __getitem__(self,i):
  k=self.keys[i]; rp=self.rz.extract(self.R[k],self.tmp.name); sp=self.sz.extract(self.S[k],self.tmp.name)
  return decode_rgb(rp),decode_skel(sp),torch.tensor(k[0]-1),torch.tensor(k)
class RGBEncoder(nn.Module):
 def __init__(self,d=128):
  super().__init__(); self.net=nn.Sequential(nn.Conv3d(3,32,3,padding=1),nn.BatchNorm3d(32),nn.ReLU(),nn.MaxPool3d((1,2,2)),nn.Conv3d(32,64,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool3d((1,1,1))); self.fc=nn.Linear(64,d)
 def forward(self,x): return self.fc(self.net(x).flatten(1))
class SkeletonEncoder(nn.Module):
 def __init__(self,d=128):
  super().__init__(); self.net=nn.Sequential(nn.Conv1d(60,128,3,padding=1),nn.ReLU(),nn.Conv1d(128,d,3,padding=1),nn.ReLU())
 def forward(self,x,return_joint=False):
  z=self.net(x.flatten(2).transpose(1,2)); feat=z.mean(-1)
  if return_joint:
   # gradient-ready joint evidence proxy retained from input
   return feat,x
  return feat
class FaithfulHAR(nn.Module):
 def __init__(self,d=128,pdrop=.2):
  super().__init__(); self.rgb=RGBEncoder(d); self.skel=SkeletonEncoder(d); self.motion=nn.Sequential(nn.Linear(60,d),nn.ReLU())
  self.gate=nn.Sequential(nn.Linear(d*3,d),nn.ReLU(),nn.Linear(d,3)); self.cls=nn.Linear(d,27); self.pdrop=pdrop
 def forward(self,rgb,sk,force_drop=None,return_parts=False):
  r=self.rgb(rgb); s=self.skel(sk); vel=torch.diff(sk,dim=1,prepend=sk[:,:1]); m=self.motion(vel.abs().mean(1).flatten(1))
  parts=torch.stack([r,s,m],1)
  if self.training and self.pdrop and force_drop is None:
   keep=(torch.rand(parts.shape[:2],device=parts.device)>self.pdrop).float(); keep[keep.sum(1)==0,0]=1; parts=parts*keep[:,:,None]
  if force_drop is not None: parts[:,force_drop]=0
  w=self.gate(torch.cat([r,s,m],1)).softmax(-1); fused=(parts*w[:,:,None]).sum(1); logits=self.cls(fused)
  return (logits,{"weights":w,"parts":parts,"velocity":vel}) if return_parts else logits
def metrics(logits,y):
 p=logits.argmax(1); acc=(p==y).float().mean().item()
 f=[]; rec=[]
 for c in range(27):
  tp=((p==c)&(y==c)).sum().item(); fp=((p==c)&(y!=c)).sum().item(); fn=((p!=c)&(y==c)).sum().item()
  rr=tp/(tp+fn) if tp+fn else float("nan"); pp=tp/(tp+fp) if tp+fp else 0.; rec.append(rr)
  f.append(2*pp*rr/(pp+rr) if pp+rr and not np.isnan(rr) else float("nan"))
 return {"top1":acc,"macro_f1":float(np.nanmean(f)),"per_class_recall":rec}
@torch.no_grad()
def modality_faithfulness(model,rgb,sk,y):
 model.eval(); base,parts=model(rgb,sk,return_parts=True); prob=base.softmax(-1).gather(1,y[:,None]).squeeze(1); out={}
 for i,n in enumerate(["rgb","skeleton","motion"]):
  z=model(rgb,sk,force_drop=i); q=z.softmax(-1).gather(1,y[:,None]).squeeze(1); out[n]=float((prob-q).mean())
 return out,parts["weights"].mean(0).tolist()
def run_smoke(args):
 seed_all(args.seed); dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 ds=UTD(args.rgb,args.skel,[1,3,5,7],limit=args.limit); dl=DataLoader(ds,batch_size=2,shuffle=False,num_workers=0)
 rgb,sk,y,k=next(iter(dl)); rgb,sk,y=rgb.to(dev),sk.to(dev),y.to(dev); model=FaithfulHAR().to(dev); opt=torch.optim.AdamW(model.parameters(),1e-3)
 model.train(); opt.zero_grad(); out=model(rgb,sk); loss=nn.functional.cross_entropy(out,y); loss.backward(); grad=sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None); opt.step()
 faith,w=modality_faithfulness(model,rgb,sk,y)
 rep={"pass":bool(torch.isfinite(loss) and grad>0),"device":str(dev),"seed":args.seed,"keys":k.tolist(),"rgb_shape":list(rgb.shape),"skeleton_shape":list(sk.shape),"logits_shape":list(out.shape),"loss":float(loss.detach()),"grad_l1":grad,"recognition":metrics(out.detach(),y),"modality_confidence_drop":faith,"mean_modality_weights":w}
 Path(args.out).parent.mkdir(parents=True,exist_ok=True); Path(args.out).write_text(json.dumps(rep,indent=2)); print(json.dumps(rep,indent=2))
if __name__=="__main__":
 p=argparse.ArgumentParser(); p.add_argument("--rgb",default="RGB.zip");p.add_argument("--skel",default="Skeleton.zip");p.add_argument("--seed",type=int,default=42);p.add_argument("--limit",type=int,default=8);p.add_argument("--out",default="paper2/production_smoke.json");a=p.parse_args();run_smoke(a)

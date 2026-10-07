#!/usr/bin/env python3
import argparse, zipfile, re, json, tempfile
from pathlib import Path
import cv2, numpy as np, scipy.io, torch
from torch import nn
from model_smoke import Model
PAT=re.compile(r"a(\d+)_s(\d+)_t(\d+)",re.I)
def index(z):
    return {tuple(map(int,PAT.search(Path(n).stem).groups())):n for n in z.namelist() if PAT.search(Path(n).stem)}
def video_tensor(path,T=8,S=64):
    cap=cv2.VideoCapture(str(path)); frames=[]
    while True:
        ok,f=cap.read()
        if not ok: break
        f=cv2.cvtColor(f,cv2.COLOR_BGR2RGB); frames.append(f)
    cap.release(); assert frames
    ids=np.linspace(0,len(frames)-1,T).round().astype(int)
    a=np.stack([cv2.resize(frames[i],(S,S)) for i in ids]).astype("float32")/255.
    return torch.from_numpy(a).permute(3,0,1,2)
def skel_tensor(path,T=32):
    d=scipy.io.loadmat(path)["d_skel"].astype("float32") # J,3,time
    x=np.transpose(d,(2,0,1))
    ids=np.linspace(0,len(x)-1,T).round().astype(int); x=x[ids]
    root=x[:,0:1,:]; x=x-root
    scale=np.linalg.norm(x,axis=-1).mean()+1e-6; x=x/scale
    return torch.from_numpy(x)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--rgb",default="RGB.zip"); ap.add_argument("--skel",default="Skeleton.zip"); ap.add_argument("--batch",type=int,default=2); a=ap.parse_args()
    with zipfile.ZipFile(a.rgb) as rz, zipfile.ZipFile(a.skel) as sz, tempfile.TemporaryDirectory() as td:
        R,S=index(rz),index(sz); keys=sorted(set(R)&set(S)); assert len(keys)==861
        train=[k for k in keys if k[1] in {1,3,5,7}]; test=[k for k in keys if k[1] in {2,4,6,8}]
        assert len(train)==431 and len(test)==430 and not(set(train)&set(test))
        chosen=[]
        seen=set()
        for k in train:
            if k[0] not in seen: chosen.append(k); seen.add(k[0])
            if len(chosen)>=a.batch: break
        rgbs=[]; sks=[]; ys=[]
        for k in chosen:
            rp=rz.extract(R[k],td); sp=sz.extract(S[k],td)
            rgbs.append(video_tensor(rp)); sks.append(skel_tensor(sp)); ys.append(k[0]-1)
        rgb=torch.stack(rgbs); sk=torch.stack(sks); y=torch.tensor(ys)
        m=Model(); opt=torch.optim.AdamW(m.parameters(),1e-3)
        opt.zero_grad(); out,g=m(rgb,sk); loss=nn.functional.cross_entropy(out,y); loss.backward()
        grad=sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None); opt.step()
        result={"pass":True,"paired":len(keys),"train":len(train),"test":len(test),"subjects_train":[1,3,5,7],"subjects_test":[2,4,6,8],"batch_keys":chosen,"rgb_shape":list(rgb.shape),"skeleton_shape":list(sk.shape),"logits_shape":list(out.shape),"gate_shape":list(g.shape),"loss":float(loss.detach()),"grad_l1":grad}
        Path("paper2/real_data_smoke_result.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=="__main__": main()

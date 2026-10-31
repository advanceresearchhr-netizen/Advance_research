import numpy as np,pandas as pd,torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from scipy.io import loadmat
from .core import corrupt_images
IMAGENET_MEAN=torch.tensor([.485,.456,.406]).view(3,1,1)
IMAGENET_STD=torch.tensor([.229,.224,.225]).view(3,1,1)

def make_splits(manifest,seed):
    from sklearn.model_selection import train_test_split
    df=pd.read_csv(manifest,dtype={'sequence':str});df['sequence']=df.sequence.str.zfill(4)
    assert len(df)==2326 and df.sequence.is_unique and df.action.nunique()==15
    assert df.aligned.all() and df.finite_pose.all()
    trpool=df[df.train==1];test=df[df.train==-1]
    assert len(trpool)==1258 and len(test)==1068
    tr,val=train_test_split(trpool,test_size=.2,random_state=seed,stratify=trpool.action)
    assert len(tr)==1006 and len(val)==252
    sets=[set(x.sequence) for x in [tr,val,test]]
    assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    return tr,val,test

class PennDataset(Dataset):
    def __init__(self,frame,root,class_to_id,visual_fault='clean',severity=0.0,seed=42):
        self.df=frame.reset_index(drop=True);self.root=Path(root);self.class_to_id=class_to_id
        self.fault=visual_fault;self.severity=severity;self.seed=seed
    def __len__(self):return len(self.df)
    def __getitem__(self,i):
        row=self.df.iloc[i];sid=str(row.sequence).zfill(4)
        paths=sorted((self.root/'frames'/sid).glob('*.jpg'))
        d=loadmat(self.root/'labels'/f'{sid}.mat',squeeze_me=True)
        px=np.asarray(d['x'],np.float32);py=np.asarray(d['y'],np.float32)
        if not paths or len(paths)!=len(px) or len(px)!=len(py) or px.shape[1]!=13:raise ValueError(f'Bad sample {sid}')
        indices=np.linspace(0,len(paths)-1,16).round().astype(int)
        rgbs=[];poses=[]
        for idx in indices:
            with Image.open(paths[idx]) as im:
                im=im.convert('RGB');w,h=im.size;arr=np.asarray(im.resize((112,112))).copy()
            rgbs.append(torch.from_numpy(arr).permute(2,0,1).float()/255)
            poses.append(np.stack([(px[idx]/max(w,1))*2-1,(py[idx]/max(h,1))*2-1],axis=-1))
        rgb=torch.stack(rgbs)
        if self.fault!='clean':rgb=corrupt_images(rgb,self.fault,self.severity,self.seed+i)
        rgb=(rgb-IMAGENET_MEAN)/IMAGENET_STD
        return rgb,torch.tensor(np.stack(poses),dtype=torch.float32),int(self.class_to_id[row.action]),sid

def frozen_features(loader,backbone,device):
    backbone.eval();out=[]
    with torch.inference_mode():
        for rgb,pose,y,sid in loader:
            b,t,c,h,w=rgb.shape
            feat=backbone(rgb.to(device).reshape(b*t,c,h,w)).reshape(b,t,512)
            out.append((feat.cpu(),pose.cpu(),y.long(),list(sid)))
    if not out:raise ValueError('No data')
    return torch.cat([a[0] for a in out]),torch.cat([a[1] for a in out]),torch.cat([a[2] for a in out]),sum([a[3] for a in out],[])

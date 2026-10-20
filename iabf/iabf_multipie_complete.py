#!/usr/bin/env python3
"""IABF Multi-PIE: 50-image gate-only adaptation + identity-disjoint cross-pose evaluation.

Usage:
 python iabf_multipie_50shot.py --manifest multipie_manifest.csv \
   --baseline best_standard_recognizer.pth --proposed best_casia_recognizer.pth \
   --output runs/seed42 --seed 42

CSV REQUIRED columns: path,subject,yaw_deg
CSV optional: session,illumination,camera. yaw_deg comes from known CAMERA metadata,
NOT from the model's estimated head pose. Frontal is abs(yaw_deg)<=7.5.
Exactly 50 unique images are selected for adaptation. Validation and test SUBJECTS are disjoint.
No network downloads in this script. Install dependencies and provide trusted checkpoints separately.
"""
import argparse, copy, hashlib, json, math, random, sys, time, csv
from pathlib import Path
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from sklearn.metrics import roc_auc_score, roc_curve

REFERENCE_5PTS=np.float32([[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],[41.5493,92.3655],[70.7299,92.2041]])
POSE_OBJECT_POINTS=np.float64([[-30,-30,-30],[30,-30,-30],[0,0,-50],[-25,30,-20],[25,30,-20]])
MEAN=np.array([.485,.456,.406],np.float32)[:,None,None]
STD=np.array([.229,.224,.225],np.float32)[:,None,None]

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for buf in iter(lambda:f.read(2**20),b''):h.update(buf)
    return h.hexdigest()

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

class CASIAWebFaceRecognizer(nn.Module):
    """Exact inference topology observed in user's uploaded AR verification notebook."""
    def __init__(self):
        super().__init__()
        from facenet_pytorch import InceptionResnetV1
        self.backbone=InceptionResnetV1(pretrained=None).eval()
        self.face_proj=nn.Sequential(nn.Linear(512,512,bias=False),nn.BatchNorm1d(512),nn.PReLU())
        self.meta_gate=nn.Sequential(nn.Linear(11,64),nn.BatchNorm1d(64),nn.ReLU(),nn.Linear(64,512),nn.Sigmoid())
        self.emb_head=nn.Sequential(nn.Linear(512,512,bias=False),nn.BatchNorm1d(512))
    def forward(self,x,pose,illum,quality):
        z=self.face_proj(self.backbone(x))
        # IMPORTANT: original trained architecture is residual, not z*gate.
        z=z*(1.0+self.meta_gate(torch.cat([pose,illum,quality],dim=1)))
        return F.normalize(self.emb_head(z),dim=1)

class StandardRecognizer(nn.Module):
    def __init__(self):
        super().__init__()
        from facenet_pytorch import InceptionResnetV1
        self.backbone=InceptionResnetV1(pretrained=None).eval()
        self.face_proj=nn.Sequential(nn.Linear(512,512,bias=False),nn.BatchNorm1d(512),nn.PReLU())
        self.emb_head=nn.Sequential(nn.Linear(512,512,bias=False),nn.BatchNorm1d(512))
    def forward(self,x,*unused):
        return F.normalize(self.emb_head(self.face_proj(self.backbone(x))),dim=1)

def load_model(path,cls,device):
    # Original research checkpoints contain NumPy scalar validation metadata.
    # Use an explicit NumPy allowlist and retain safe weights-only loading.
    numpy_safe=[np._core.multiarray.scalar,np.dtype]
    numpy_safe += [type(np.dtype(t)) for t in ('float32','float64','int32','int64','uint8','bool')]
    with torch.serialization.safe_globals(numpy_safe):
        ckpt=torch.load(path,map_location='cpu',weights_only=True)
    state=ckpt.get('model_state_dict',ckpt) if isinstance(ckpt,dict) else ckpt
    state={k:v for k,v in state.items() if not k.startswith('backbone.logits.')}
    model=cls();model.load_state_dict(state,strict=True)
    return model.to(device).eval()

def five_points(lm):
    lm=np.asarray(lm,np.float64)
    if lm.shape!=(68,2) or not np.isfinite(lm).all():raise ValueError('invalid landmarks')
    return np.ascontiguousarray([lm[36:42].mean(0),lm[42:48].mean(0),lm[30],lm[48],lm[54]],dtype=np.float64)

def estimate_pose(pts,shape):
    h,w=shape[:2]; eye=np.linalg.norm(pts[1]-pts[0])
    if eye<5:raise ValueError('eye separation too small')
    if np.any(pts<0) or np.any(pts[:,0]>=w) or np.any(pts[:,1]>=h):raise ValueError('landmarks outside image')
    camera=np.float64([[w,0,w/2],[0,w,h/2],[0,0,1]])
    distort=np.zeros((4,1),np.float64)
    ok,r,t=cv2.solvePnP(POSE_OBJECT_POINTS,pts,camera,distort,flags=cv2.SOLVEPNP_EPNP)
    if not ok:raise ValueError('EPNP failed')
    ok,r,t=cv2.solvePnP(POSE_OBJECT_POINTS,pts,camera,distort,r,t,True,flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:raise ValueError('iterative failed')
    R=cv2.Rodrigues(r)[0];sy=np.hypot(R[0,0],R[1,0])
    if sy<1e-6:raise ValueError('Euler singularity')
    depth=(R@POSE_OBJECT_POINTS.T+t.reshape(3,1))[2]
    if np.any(depth<=0):raise ValueError('invalid depth')
    proj=cv2.projectPoints(POSE_OBJECT_POINTS,r,t,camera,distort)[0].reshape(5,2)
    error=float(np.sqrt(np.mean(np.sum((proj-pts)**2,axis=1))))
    # Matrix convention: ZYX Euler: (roll_z, yaw_y, pitch_x); output explicitly (yaw_y, pitch_x, roll_z).
    rz=np.arctan2(R[1,0],R[0,0]); ry=np.arctan2(-R[2,0],sy); rx=np.arctan2(R[2,1],R[2,2])
    angles=np.degrees([ry,rx,rz])
    if error/eye>.10 or np.any(np.abs(angles)>90):raise ValueError('pose quality rejected')
    return angles.astype(np.float32),error

def process_image(path,detector):
    rgb=np.asarray(ImageOps.exif_transpose(Image.open(path)).convert('RGB'))
    landmarks=detector.get_landmarks_from_image(rgb)
    if landmarks is None or len(landmarks)!=1:raise ValueError('expected exactly one detected face')
    pts=five_points(landmarks[0]);ang,rmse=estimate_pose(pts,rgb.shape)
    M,_=cv2.estimateAffinePartial2D(pts.astype(np.float32),REFERENCE_5PTS*2,method=cv2.LMEDS)
    if M is None:raise ValueError('alignment failed')
    a=cv2.warpAffine(rgb,M,(224,224),borderMode=cv2.BORDER_REFLECT)
    g=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY).astype(np.float32)/255.
    bl=cv2.GaussianBlur(g,(15,15),0)
    illum=np.float32([g.mean(),g.std(),(g<.2).mean(),(g-bl).std()])
    qual=np.float32([g.mean(),g.std(),np.log1p(cv2.Laplacian(g,cv2.CV_32F).var())/10,1-((g<.02)|(g>.98)).mean()])
    x=((a.astype(np.float32)/255).transpose(2,0,1)-MEAN)/STD
    return x,ang/90.,illum,qual,rmse

def read_manifest(p):
    df=pd.read_csv(p,dtype={'subject':str,'camera':str,'session':str,'illumination':str})
    req={'path','subject','yaw_deg'}
    if not req.issubset(df):raise ValueError(f'Manifest needs columns {sorted(req)}')
    df=df.copy(); df['yaw_deg']=pd.to_numeric(df.yaw_deg,errors='raise')
    if df[['path','subject','yaw_deg']].isna().any().any():raise ValueError('Missing manifest fields')
    if not np.isfinite(df.yaw_deg).all() or (df.yaw_deg.abs()>90).any():raise ValueError('Invalid camera yaw values')
    if 'camera' not in df or df.camera.isna().any():raise ValueError('Verified camera IDs required; add camera column and map to official geometry')
    if (df.groupby('camera').yaw_deg.nunique()>1).any():raise ValueError('Camera ID maps to inconsistent yaw angles')
    base=Path(p).resolve().parent
    df['path']=df.path.map(lambda x:str((base/str(x)).resolve()))
    if df.path.duplicated().any():raise ValueError('duplicate paths')
    for path in df.path:
        if not Path(path).is_file():raise FileNotFoundError(path)
    df['frontal']=df.yaw_deg.abs()<=7.5
    return df.reset_index(drop=True)

def split_subjects(df,seed,val_fraction=.2):
    # Adaptation 50 images: sample ONLY subjects with 1 frontal plus >=1 nonfrontal.
    usable=[]
    for sub,g in df.groupby('subject'):
        if g.frontal.any() and (~g.frontal).any():usable.append(str(sub))
    rng=np.random.default_rng(seed); usable=np.asarray(sorted(usable));rng.shuffle(usable)
    if len(usable)<14:raise ValueError('Need >=14 subjects: 10 adaptation + at least 2 validation + 2 test')
    # Use 10 identities, 5 images per identity, allowing 1 frontal + 4 rotated.
    eligible=[]
    for s in usable:
        g=df[df.subject==s]
        if g.frontal.sum()>=1 and (~g.frontal).sum()>=4:eligible.append(s)
    if len(eligible)<10:raise ValueError('Need >=10 adaptation subjects with 1 frontal and >=4 rotated photos')
    train_ids=set(eligible[:10]);remaining=[x for x in usable if x not in train_ids]
    if len(remaining)<4:raise ValueError('Need >=2 validation and >=2 test subjects')
    nv=max(2,int(round(len(remaining)*val_fraction)));nv=min(nv,len(remaining)-2)
    val_ids=set(remaining[:nv]);test_ids=set(remaining[nv:]);
    train=[]
    for s in sorted(train_ids):
        g=df[df.subject==s].copy();fr=g[g.frontal].sort_values('path').iloc[0]
        rot=g[~g.frontal].copy();rot['abs_yaw']=rot.yaw_deg.abs()
        # Favor multiple camera directions and magnitudes; deterministic jitter for ties.
        rot['order']=rot.abs_yaw.rank(method='first')
        ordered=rot.sort_values(['abs_yaw','path'])
        idxs=np.linspace(0,len(ordered)-1,4).round().astype(int)
        choose=ordered.iloc[idxs]
        train.extend([int(fr.name)]+[int(i) for i in choose.index])
    if len(train)!=50 or len(set(train))!=50:raise AssertionError('50 unique adaptation images not selected')
    assert train_ids.isdisjoint(val_ids|test_ids) and val_ids.isdisjoint(test_ids)
    return df.loc[train].copy(),df[df.subject.isin(val_ids)].copy(),df[df.subject.isin(test_ids)].copy()

def precompute(df,detector,out,subset):
    path=out/f'{subset}_features.npz'; records=[];xs=[];ps=[];il=[];ql=[];rows=[]
    for k,row in enumerate(df.itertuples()):
        try:
            x,p,i,q,e=process_image(row.path,detector)
            xs.append(x);ps.append(p);il.append(i);ql.append(q);rows.append(row.Index)
            records.append({'row_id':int(row.Index),'path':row.path,'status':'ok','estimated_pose_rmse_px':e})
        except Exception as ex:
            records.append({'row_id':int(row.Index),'path':row.path,'status':str(ex),'estimated_pose_rmse_px':np.nan})
        if (k+1)%100==0:print(f'[{subset}] preprocessing {k+1}/{len(df)}',flush=True)
    pd.DataFrame(records).to_csv(out/f'{subset}_preprocessing_audit.csv',index=False)
    if not rows:raise RuntimeError(f'{subset}: all preprocessing failed')
    if subset=='adapt' and len(rows)!=50:
        raise RuntimeError(f'Only {len(rows)}/50 selected training images passed pose/alignment QC. STOP: no quiet replacements or relaxed budget. Fix images/selection and rerun.')
    features={'x':np.asarray(xs,np.float32),'p':np.asarray(ps,np.float32),'i':np.asarray(il,np.float32),'q':np.asarray(ql,np.float32),'row_ids':np.asarray(rows,np.int64)}
    np.savez_compressed(path,**features)
    kept=df.loc[rows].copy().reset_index(drop=True)
    if not (kept.frontal.any() and (~kept.frontal).any()):raise RuntimeError(f'{subset}: missing frontal or rotated samples after QC')
    return features,kept

@torch.no_grad()
def embed(model,data,device,batch_size=32,zero_pose=False,zero_illum=False,zero_quality=False):
    model.eval();out=[]
    for start in range(0,len(data['x']),batch_size):
        sl=slice(start,start+batch_size)
        x=torch.from_numpy(data['x'][sl]).to(device)
        p=torch.from_numpy(data['p'][sl]).to(device)
        i=torch.from_numpy(data['i'][sl]).to(device)
        q=torch.from_numpy(data['q'][sl]).to(device)
        if zero_pose:p=torch.zeros_like(p)
        if zero_illum:i=torch.zeros_like(i)
        if zero_quality:q=torch.zeros_like(q)
        out.append(model(x,p,i,q).cpu().numpy())
    return np.concatenate(out)

def optimize_gate(model,features,frame,device,epochs,lr,weight_decay,out):
    # Freeze everything except the existing gate, including all BatchNorm running statistics.
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    for p in model.meta_gate.parameters():p.requires_grad_(True)
    # Gate BN runs in eval mode because 50 images do not justify running-statistic updates.
    model.meta_gate.eval()
    ref=copy.deepcopy(model).eval()
    for p in ref.parameters():p.requires_grad_(False)
    x=torch.from_numpy(features['x']).to(device)
    p=torch.from_numpy(features['p']).to(device)
    il=torch.from_numpy(features['i']).to(device)
    q=torch.from_numpy(features['q']).to(device)
    identities=frame.subject.astype(str).to_numpy()
    labels=torch.tensor(pd.Categorical(identities).codes,device=device)
    mask=(labels[:,None]==labels[None,:]);diag=torch.eye(len(labels),dtype=torch.bool,device=device)
    positives=mask&~diag
    if (~positives.any(dim=1)).any():raise RuntimeError('Adaptation set needs >=2 samples per identity')
    optimizer=torch.optim.AdamW(model.meta_gate.parameters(),lr=lr,weight_decay=weight_decay)
    hist=[]
    # Cache frozen visual trunk: adaptation updates ONLY the gate, so this is exact and efficient.
    with torch.no_grad():
        visual=model.face_proj(model.backbone(x)).detach()
        ref_g=ref.meta_gate(torch.cat([p,il,q],dim=1))
        ref_emb=F.normalize(ref.emb_head(visual*(1+ref_g)),dim=1)
    for ep in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        gate=model.meta_gate(torch.cat([p,il,q],dim=1))
        emb=F.normalize(model.emb_head(visual*(1+gate)),dim=1)
        similarities=emb@emb.T/.07
        similarities=similarities.masked_fill(diag,-1e4)
        log_prob=similarities-torch.logsumexp(similarities,dim=1,keepdim=True)
        supervised=-(log_prob*positives.float()).sum(dim=1)/positives.sum(dim=1).clamp_min(1)
        # Constrain a tiny 50-image update to avoid destroying pretrained representations.
        preserve=(emb-ref_emb).square().sum(dim=1).mean()
        gate_reg=(gate-ref_g).square().mean()
        loss=supervised.mean()+2.0*preserve+.1*gate_reg
        if not torch.isfinite(loss):raise RuntimeError('non-finite training loss')
        loss.backward();nn.utils.clip_grad_norm_(model.meta_gate.parameters(),1.0)
        optimizer.step()
        hist.append({'epoch':ep+1,'loss':float(loss.detach()),'supervised_contrastive':float(supervised.mean().detach()),'embedding_preservation':float(preserve.detach())})
    pd.DataFrame(hist).to_csv(out/'adaptation_training_history.csv',index=False)
    model.eval();return model

def gallery_probe(frame,emb,seed):
    rng=np.random.default_rng(seed)
    subjects=sorted(frame.subject.unique()); gallery=[];probes=[]
    for subj in subjects:
        idx=np.flatnonzero(frame.subject.to_numpy()==subj)
        frontal=[int(k) for k in idx if bool(frame.iloc[k].frontal)]
        nonfront=[int(k) for k in idx if not bool(frame.iloc[k].frontal)]
        if not frontal or not nonfront:continue
        # First path is a fixed deterministic gallery, never selected by score.
        g=min(frontal,key=lambda k:frame.iloc[k].path)
        gallery.append((subj,g))
        for k in nonfront:probes.append((subj,k))
    if len(gallery)<2 or not probes:raise RuntimeError('At least two gallery identities with rotated probes needed')
    gallery_ids=np.array([t[0] for t in gallery]);gidx=np.array([t[1] for t in gallery]);pidx=np.array([t[1] for t in probes]);pids=np.array([t[0] for t in probes])
    sim=emb[pidx]@emb[gidx].T
    pred=gallery_ids[sim.argmax(1)]; rank1=(pred==pids).astype(int)
    score=sim.max(axis=1)
    true_col=np.array([np.where(gallery_ids==subj)[0][0] for subj in pids])
    genuine_score=sim[np.arange(len(sim)),true_col]
    rows=frame.iloc[pidx].copy();rows['predicted_subject']=pred;rows['correct_rank1']=rank1;rows['max_similarity']=score;rows['genuine_similarity']=genuine_score
    rows['gallery_path']=[frame.iloc[gidx[k]].path for k in true_col]
    return rows,sim,pids,gallery_ids

def verification(scores,pids,gallery_ids):
    labels=(pids[:,None]==gallery_ids[None,:]).astype(int).ravel()
    allscores=scores.ravel()
    if len(np.unique(labels))<2:return {'auc':np.nan,'eer':np.nan,'tar_at_far_1pct':np.nan},(labels,allscores)
    auc=roc_auc_score(labels,allscores);fpr,tpr,_=roc_curve(labels,allscores)
    k=np.argmin(np.abs(fpr-(1-tpr)));eer=float((fpr[k]+1-tpr[k])/2)
    # Conservative observed TAR at empirical FAR <= target, without interpolating sparse FAR points.
    tar=float(tpr[fpr<=.01].max()) if np.any(fpr<=.01) else np.nan
    return {'auc':float(auc),'eer':eer,'tar_at_far_1pct':tar},(labels,allscores)

def run_eval(models,sets,device,out,seed):
    metrics=[];all_pairs={};predictions={}
    for split,(data,frame) in sets.items():
        for name,net in models.items():
            # No post-hoc changes to evaluation samples; same galleries/probes per model.
            em=embed(net,data,device)
            pred,mat,pids,gids=gallery_probe(frame,em,seed)
            score,ys=verification(mat,pids,gids)
            metrics.append({'split':split,'model':name,'subjects':len(np.unique(pids)),'probes':len(pred),'rank1':float(pred.correct_rank1.mean()),**score})
            pred.to_csv(out/f'{split}_{name}_per_probe.csv',index=False)
            predictions[(split,name)]=pred
            all_pairs[(split,name)]=ys
            by=(pred.assign(pose_side=np.where(pred.yaw_deg<0,'left','right'))
                 .groupby(['yaw_deg','pose_side'],as_index=False).agg(n=('correct_rank1','size'),rank1=('correct_rank1','mean')))
            by.to_csv(out/f'{split}_{name}_by_camera_yaw.csv',index=False)
            if split=='test' and name=='IABF_50':
                for tag,kw in [('zero_pose',{'zero_pose':True}),('zero_illum',{'zero_illum':True}),('zero_quality',{'zero_quality':True})]:
                    ab=embed(net,data,device,**kw)
                    abp,abmat,apid,agid=gallery_probe(frame,ab,seed)
                    av,_=verification(abmat,apid,agid)
                    metrics.append({'split':'test_ablation','model':tag,'subjects':len(np.unique(apid)),'probes':len(abp),'rank1':float(abp.correct_rank1.mean()),**av})
                    abp.to_csv(out/f'test_{tag}_per_probe.csv',index=False)
    result=pd.DataFrame(metrics);result.to_csv(out/'evaluation_summary.csv',index=False)
    if 'test' in sets:
        plot_results(out,predictions,all_pairs)
    return result

def plot_results(out,predictions,all_pairs):
    keys=[k for k in predictions if k[0]=='test']
    fig,ax=plt.subplots(figsize=(9,5.2))
    for k in keys:
        g=predictions[k].groupby('yaw_deg').correct_rank1.agg(['mean','size']).reset_index().sort_values('yaw_deg')
        ax.plot(g.yaw_deg,g['mean']*100,marker='o',label=k[1]);
    ax.set(xlabel='Camera yaw (degrees)',ylabel='Rank-1 identification accuracy (%)',title='Frontal gallery → rotated probes (held-out test subjects)',ylim=(-3,103))
    ax.axvline(0,color='gray',alpha=.4,linewidth=.8);ax.grid(alpha=.2);ax.legend();fig.tight_layout()
    fig.savefig(out/'figure_rank1_vs_pose.png',dpi=250);fig.savefig(out/'figure_rank1_vs_pose.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5))
    for k in keys:
        labels,scores=all_pairs[k];fpr,tpr,_=roc_curve(labels,scores)
        ax.plot(fpr,tpr,label=f'{k[1]} (AUC={roc_auc_score(labels,scores):.3f})')
    ax.plot([0,1],[0,1],ls='--',color='gray',lw=.8)
    ax.set(xlabel='False positive rate',ylabel='True positive rate',title='Cross-pose verification ROC, held-out identities')
    ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(out/'figure_roc.png',dpi=250);fig.savefig(out/'figure_roc.pdf');plt.close(fig)
    # Show same probe/gallery images with predictions from all models and include failures.
    base=predictions.get(('test','Standard'));prop=predictions.get(('test','IABF_50'))
    if base is None or prop is None:return
    a=base[['path','correct_rank1']].rename(columns={'correct_rank1':'baseline_correct'})
    b=prop[['path','gallery_path','subject','yaw_deg','correct_rank1','genuine_similarity']].rename(columns={'correct_rank1':'iabf_correct'})
    joined=b.merge(a,on='path',validate='one_to_one')
    cases=[('IABF corrected',joined[(joined.iabf_correct==1)&(joined.baseline_correct==0)]),
           ('Both correct',joined[(joined.iabf_correct==1)&(joined.baseline_correct==1)]),
           ('Both wrong',joined[(joined.iabf_correct==0)&(joined.baseline_correct==0)]),
           ('IABF newly wrong',joined[(joined.iabf_correct==0)&(joined.baseline_correct==1)])]
    fig,axs=plt.subplots(4,2,figsize=(6.5,10));n=0
    for j,(title,df) in enumerate(cases):
        if df.empty:
            axs[j,0].axis('off');axs[j,1].axis('off');axs[j,0].set_title(title+' (none)');continue
        item=df.sort_values('path').iloc[0]
        for k,impath in enumerate((item.gallery_path,item.path)):
            axs[j,k].imshow(Image.open(impath).convert('RGB'));axs[j,k].axis('off')
            axs[j,k].set_title(('Frontal gallery' if k==0 else f'Probe yaw={item.yaw_deg:g}°'),fontsize=9)
        axs[j,0].set_ylabel(title,fontsize=9)
        n+=1
    fig.suptitle('Real held-out Multi-PIE samples (not synthetic)',fontsize=11)
    fig.tight_layout();fig.savefig(out/'figure_actual_face_pairs.png',dpi=200);plt.close(fig)


# ========================= Complete protocol: extension of the original 50-image script =========================
DESCRIPTORS={
    'pose':(1,0,0),'illumination':(0,1,0),'quality':(0,0,1),
    'pose_illumination':(1,1,0),'pose_quality':(1,0,1),
    'illumination_quality':(0,1,1),'full':(1,1,1)
}

def validated_pose_geometry():
    """Known-angle synthetic checks protect yaw/pitch/roll output ordering."""
    def euler(R):
        sy=np.hypot(R[0,0],R[1,0]);rz=np.arctan2(R[1,0],R[0,0]);ry=np.arctan2(-R[2,0],sy);rx=np.arctan2(R[2,1],R[2,2]);return np.degrees([ry,rx,rz])
    for axis in range(3):
        v=np.zeros(3);v[axis]=20
        # yaw about camera Y, pitch X, roll Z
        rotvec=np.zeros(3); rotvec[[1,0,2][axis]]=np.deg2rad(v[axis]);R=cv2.Rodrigues(rotvec)[0]
        estimated=euler(R)
        if not np.allclose(estimated,v,atol=1e-5):raise AssertionError(f'Euler convention failed axis {axis}: {estimated} vs {v}')
    return True

def masked_inputs(data,mode):
    enable=DESCRIPTORS[mode];return [torch.from_numpy(data[k]) if enable[j] else torch.zeros_like(torch.from_numpy(data[k])) for j,k in enumerate(('p','i','q'))]

def adapt_arcface(model,data,frame,validation,device,out,mode,epochs,lr,wd,margin,scale,eval_every=3):
    """Only gate + temporary ArcFace classifier updated; pretrained model frozen.
    Val is identity-disjoint, used for epoch selection ONLY; final test remains sealed.
    """
    from copy import deepcopy
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    for p in model.meta_gate.parameters():p.requires_grad_(True)
    model.meta_gate.eval()
    x=torch.from_numpy(data['x']).to(device)
    p,i,q=[t.to(device) for t in masked_inputs(data,mode)]
    labs=torch.tensor(pd.Categorical(frame.subject.astype(str)).codes,dtype=torch.long,device=device)
    nclasses=int(labs.max().item())+1
    if len(labs)!=50 or nclasses!=10:raise AssertionError('ArcFace adaptation requires exactly 50 samples and 10 identities')
    with torch.no_grad():
        vis=model.face_proj(model.backbone(x)).detach()
        original_gate=model.meta_gate(torch.cat([p,i,q],1)).detach()
        reference=F.normalize(model.emb_head(vis*(1+original_gate)),dim=1).detach()
    classifier=nn.Parameter(F.normalize(torch.randn(nclasses,512,device=device),dim=1))
    opt=torch.optim.AdamW([{'params':model.meta_gate.parameters()},{'params':[classifier]}],lr=lr,weight_decay=wd)
    records=[];best=-np.inf;best_state=deepcopy(model.meta_gate.state_dict());best_epoch=0
    for epoch in range(1,epochs+1):
        opt.zero_grad(set_to_none=True)
        gate=model.meta_gate(torch.cat([p,i,q],1))
        em=F.normalize(model.emb_head(vis*(1+gate)),dim=1)
        cosine=F.linear(em,F.normalize(classifier,dim=1)).clamp(-1+1e-6,1-1e-6)
        # Additive angular margin; standard ArcFace logit substitution.
        theta=torch.acos(cosine); target=torch.cos(theta+margin)
        oh=F.one_hot(labs,nclasses).bool()
        logits=torch.where(oh,target,cosine)*scale
        ce=F.cross_entropy(logits,labs)
        preservation=(em-reference).square().sum(1).mean()
        loss=ce+2.0*preservation
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite ArcFace objective')
        loss.backward();nn.utils.clip_grad_norm_(list(model.meta_gate.parameters())+[classifier],1.0)
        opt.step()
        row={'epoch':epoch,'mode':mode,'arcface_ce':float(ce.detach()),'preserve':float(preservation.detach()),'loss':float(loss.detach())}
        if epoch==1 or epoch%eval_every==0 or epoch==epochs:
            model.eval()
            with torch.no_grad():
                vemb=embed(model,validation[0],device,zero_pose=(mode not in ('pose','pose_illumination','pose_quality','full')),zero_illum=(mode not in ('illumination','pose_illumination','illumination_quality','full')),zero_quality=(mode not in ('quality','pose_quality','illumination_quality','full')))
                vp,*_=gallery_probe(validation[1],vemb,0)
                row['validation_rank1']=float(vp.correct_rank1.mean())
            if row['validation_rank1']>best+1e-12:
                best=row['validation_rank1'];best_state=deepcopy(model.meta_gate.state_dict());best_epoch=epoch
        records.append(row)
    model.meta_gate.load_state_dict(best_state);model.eval()
    pd.DataFrame(records).to_csv(out/f'training_{mode}.csv',index=False)
    return model,{'mode':mode,'best_epoch':best_epoch,'validation_rank1':float(best)}

def embeddings_for_mode(model,data,device,mode='full'):
    flags=DESCRIPTORS[mode]
    return embed(model,data,device,zero_pose=not flags[0],zero_illum=not flags[1],zero_quality=not flags[2])

def bootstrap_rank1(per_probe,reps=500,seed=123):
    # Cluster bootstrap: resample identities, never pretend thousands of probes are independent.
    subjects=per_probe.subject.unique()
    if len(subjects)<2:return [np.nan,np.nan]
    by=per_probe.groupby('subject').correct_rank1.agg(['sum','count'])
    rng=np.random.default_rng(seed);draws=[]
    for _ in range(reps):
        sample=rng.choice(subjects,len(subjects),replace=True)
        draws.append(by.loc[sample]['sum'].sum()/by.loc[sample]['count'].sum())
    return np.percentile(draws,[2.5,97.5]).tolist()

def full_evaluate(models,validation,test,device,out,seed):
    (out/'predictions').mkdir(exist_ok=True)
    rows=[];pred_map={};score_map={};summary=[]
    for split,(feat,df) in [('validation',validation),('test',test)]:
        for name,(model,mode) in models.items():
            emb=embeddings_for_mode(model,feat,device,mode)
            pred,sim,pids,gids=gallery_probe(df,emb,seed)
            stats,_=verification(sim,pids,gids)
            ci=bootstrap_rank1(pred,seed=seed)
            rows.append({'split':split,'model':name,'rank1':pred.correct_rank1.mean(),'ci95_low':ci[0],'ci95_high':ci[1],'n_probes':len(pred),'n_subjects':pred.subject.nunique(),**stats})
            pred.to_csv(out/'predictions'/f'{split}_{name}.csv',index=False)
            if split=='test':pred_map[name]=pred;score_map[name]=sim
            if split=='test':
                for group, cols in [('camera',['camera','yaw_deg']),('illumination',['illumination']),('session',['session']),('pose_illumination',['yaw_deg','illumination']),('session_pose',['session','yaw_deg'])]:
                    if all(c in pred.columns for c in cols):
                        gg=pred.groupby(cols,dropna=False).correct_rank1.agg(['mean','sum','count']).reset_index()
                        gg.to_csv(out/f'test_{name}_{group}.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'full_model_comparison.csv',index=False)
    # Post-hoc descriptor interventions (distinct from independently retrained ablations)
    model,mode=models['IABF_50_full']
    for mode0 in DESCRIPTORS:
        emb=embeddings_for_mode(model,test[0],device,mode0)
        pred,sim,pids,gids=gallery_probe(test[1],emb,seed)
        st,_=verification(sim,pids,gids)
        summary.append({'intervention':mode0,'rank1':float(pred.correct_rank1.mean()),'n_probes':len(pred),**st})
    pd.DataFrame(summary).to_csv(out/'posthoc_descriptor_interventions.csv',index=False)
    # Error-matched real image panels; no generated face imagery.
    pstd=pred_map['Standard'];pnew=pred_map['IABF_50_full']
    a=pstd[['path','correct_rank1']].rename(columns={'correct_rank1':'standard_correct'})
    joined=pnew.merge(a,on='path',validate='one_to_one')
    joined['outcome']=np.select([joined.correct_rank1.gt(joined.standard_correct),joined.correct_rank1.lt(joined.standard_correct),joined.correct_rank1.eq(1)],['corrected','regression','both_correct'],default='both_wrong')
    joined.to_csv(out/'matched_test_error_analysis.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,4.4))
    for name in ('Standard','IABF_original','IABF_50_full'):
        if name in pred_map:
            g=pred_map[name].groupby('yaw_deg').correct_rank1.agg(['mean','size']).reset_index().sort_values('yaw_deg')
            ax.plot(g.yaw_deg,g['mean']*100,marker='o',label=name)
    ax.set(xlabel='Multi-PIE camera yaw (degrees)',ylabel='Rank-1 identification (%)',ylim=(-2,102),title='One frontal gallery image per identity');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(out/'rank1_by_pose.pdf');fig.savefig(out/'rank1_by_pose.png',dpi=250);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.4))
    for name in ('Standard','IABF_original','IABF_50_full'):
        if name not in score_map:continue
        # Matrix shape n_probe x n_gallery, known gallery labels from deterministic gallery_probe
        _,sim,pids,gids=gallery_probe(test[1],embeddings_for_mode(models[name][0],test[0],device,models[name][1]),seed)
        y=(pids[:,None]==gids[None,:]).ravel().astype(int);s=sim.ravel();fpr,tpr,_=roc_curve(y,s);ax.plot(fpr,tpr,label=f'{name}: AUC={roc_auc_score(y,s):.3f}')
    ax.plot([0,1],[0,1],'--',color='gray',lw=.8);ax.set(xlabel='False accept rate',ylabel='True accept rate',title='Held-out cross-pose verification');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'verification_roc.pdf');fig.savefig(out/'verification_roc.png',dpi=250);plt.close(fig)
    cases=['corrected','regression','both_correct','both_wrong'];fig,axs=plt.subplots(4,2,figsize=(7,10))
    for r,key in enumerate(cases):
        examples=joined[joined.outcome==key]
        if examples.empty:
            for a in axs[r]:a.axis('off')
            axs[r,0].set_title(key+': none');continue
        item=examples.sort_values('path').iloc[0]
        for j,path in enumerate([item.gallery_path,item.path]):
            axs[r,j].imshow(Image.open(path).convert('RGB'));axs[r,j].axis('off');axs[r,j].set_title(('frontal gallery' if j==0 else f'probe yaw {item.yaw_deg}°'))
        axs[r,0].set_ylabel(key)
    fig.suptitle('Real Multi-PIE gallery and pose probes; held-out identities');fig.tight_layout();fig.savefig(out/'actual_pair_examples.png',dpi=250);plt.close(fig)
    return pd.DataFrame(rows)

def quality_robustness(models,validation,test,device,out,seed):
    """Deterministic probe-only degradation; descriptor recalculated from degraded aligned RGB."""
    def degrade(data,level,seed):
        clone={k:v.copy() for k,v in data.items()};rng=np.random.default_rng(seed)
        for idx in range(len(clone['x'])):
            rgb=np.clip((clone['x'][idx]*STD+MEAN).transpose(1,2,0)*255,0,255).astype(np.uint8)
            if level=='blur_low':rgb=cv2.GaussianBlur(rgb,(5,5),1.0)
            if level=='blur_high':rgb=cv2.GaussianBlur(rgb,(11,11),2.5)
            if level=='low_light':rgb=np.clip(rgb.astype(np.float32)*.40,0,255).astype(np.uint8)
            if level=='noise':rgb=np.clip(rgb.astype(np.float32)+rng.normal(0,12,rgb.shape),0,255).astype(np.uint8)
            gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
            local=cv2.GaussianBlur(gray,(15,15),0)
            clone['i'][idx]=[gray.mean(),gray.std(),(gray<.2).mean(),(gray-local).std()]
            clone['q'][idx]=[gray.mean(),gray.std(),np.log1p(cv2.Laplacian(gray,cv2.CV_32F).var())/10,1-((gray<.02)|(gray>.98)).mean()]
            clone['x'][idx]=((rgb.astype(np.float32)/255).transpose(2,0,1)-MEAN)/STD
        return clone
    records=[]
    # Keep clean frontal gallery embeddings; degrade ONLY rotated probe representations.
    for name,(model,mode) in models.items():
        clean=embeddings_for_mode(model,test[0],device,mode)
        for severity in ['clean','blur_low','blur_high','low_light','noise']:
            degraded=test[0] if severity=='clean' else degrade(test[0],severity,seed)
            modified=embeddings_for_mode(model,degraded,device,mode)
            frame=test[1];is_fr=frame.frontal.to_numpy();modified[is_fr]=clean[is_fr]
            pred,sim,pids,gids=gallery_probe(frame,modified,seed)
            stat,_=verification(sim,pids,gids)
            records.append({'model':name,'degradation':severity,'rank1':float(pred.correct_rank1.mean()),'n_probes':len(pred),**stat})
    pd.DataFrame(records).to_csv(out/'quality_robustness.csv',index=False)

def session_cross_evaluation(models,frame,data,device,out,seed):
    # Gallery Session 1 frontal, probes from later sessions, evaluation identities unchanged.
    if 'session' not in frame.columns:return
    ses=sorted(frame.session.dropna().astype(str).unique())
    if len(ses)<2:return
    ref=ses[0];records=[]
    for name,(model,mode) in models.items():
        em=embeddings_for_mode(model,data,device,mode)
        for target in ses[1:]:
            gallery=[];probes=[]
            for sub,g in frame.groupby('subject'):
                f=g[(g.session.astype(str)==ref)&g.frontal]
                r=g[(g.session.astype(str)==target)&(~g.frontal)]
                if not f.empty and not r.empty:
                    gallery.append((sub,int(frame.index.get_loc(f.index[0]))))
                    probes.extend((sub,int(frame.index.get_loc(k))) for k in r.index)
            if len(gallery)<2 or not probes:continue
            gids=np.asarray([t[0] for t in gallery]);pid=np.asarray([t[0] for t in probes]);gidx=[x[1] for x in gallery];pidx=[x[1] for x in probes]
            sim=em[pidx]@em[gidx].T;rank1=(gids[sim.argmax(1)]==pid).mean();st,_=verification(sim,pid,gids)
            records.append({'model':name,'gallery_session':ref,'probe_session':target,'subjects':len(gids),'probes':len(pid),'rank1':float(rank1),**st})
    pd.DataFrame(records).to_csv(out/'cross_session_evaluation.csv',index=False)


def lfw_tenfold(models,pairs_csv,detector,device,out):
    """Official-fold-compatible LFW pair scoring. CSV: path1,path2,is_same,fold (0..9).
    No model updates; fold-specific score threshold learned from other nine folds.
    """
    pairs=pd.read_csv(pairs_csv)
    needed={'path1','path2','is_same','fold'}
    if not needed.issubset(pairs.columns):raise ValueError('LFW pairs CSV requires path1,path2,is_same,fold')
    if set(pairs.fold.unique())!=set(range(10)):raise ValueError('LFW must have exactly folds 0..9')
    if not pairs.is_same.isin([0,1]).all():raise ValueError('is_same must be binary')
    base=Path(pairs_csv).resolve().parent
    for c in ('path1','path2'):pairs[c]=pairs[c].map(lambda x:str((base/str(x)).resolve()))
    paths=sorted(set(pairs.path1)|set(pairs.path2))
    cache={};errors=[]
    for j,path in enumerate(paths):
        try:cache[path]=process_image(path,detector)
        except Exception as ex:errors.append({'path':path,'reason':str(ex)})
        if (j+1)%200==0:print('LFW preprocessing:',j+1,'/',len(paths),flush=True)
    pd.DataFrame(errors,columns=['path','reason']).to_csv(out/'lfw_preprocessing_failures.csv',index=False)
    pairs['processed']=pairs.path1.isin(cache)&pairs.path2.isin(cache)
    pairs.to_csv(out/'lfw_pair_coverage.csv',index=False)
    if pairs.processed.sum()<10:raise RuntimeError('Insufficient LFW pairs passed preprocessing')
    retained=pairs[pairs.processed].copy();kept_paths=sorted(set(retained.path1)|set(retained.path2));position={path:i for i,path in enumerate(kept_paths)}
    feature={k:np.array([cache[path][i] for path in kept_paths],np.float32) for k,i in [('x',0),('p',1),('i',2),('q',3)]}
    report=[];fold_rows=[]
    for name,(net,mode) in models.items():
        emb=embeddings_for_mode(net,feature,device,mode)
        scores=np.array([float(emb[position[a]]@emb[position[b]]) for a,b in zip(retained.path1,retained.path2)])
        labels=retained.is_same.to_numpy(dtype=int);folds=retained.fold.to_numpy(dtype=int)
        for fold in range(10):
            train=folds!=fold;test=folds==fold
            if not test.any() or not train.any() or len(set(labels[train]))<2:continue
            # Frozen calibration from the other 9 folds, maximizing balanced accuracy.
            cand=np.unique(np.r_[-1.,scores[train],1.])
            pos=labels[train]==1;neg=~pos
            tpr=(scores[train,None]>=cand[None,:])[pos].mean(0)
            tnr=(scores[train,None]<cand[None,:])[neg].mean(0)
            threshold=float(cand[np.argmax((tpr+tnr)/2)])
            pred=(scores[test]>=threshold).astype(int)
            correct=(pred==labels[test]).mean()
            fold_rows.append({'model':name,'fold':fold,'accuracy':float(correct),'threshold':threshold,'test_pairs_retained':int(test.sum()),'test_pairs_original':int((pairs.fold==fold).sum())})
        if len(set(labels))>1:auc=float(roc_auc_score(labels,scores))
        else:auc=np.nan
        report.append({'model':name,'auc_all_retained_pairs':auc,'evaluated_pairs':len(retained),'total_pairs':len(pairs),'coverage':len(retained)/len(pairs)})
    pd.DataFrame(fold_rows).to_csv(out/'lfw_10fold_results.csv',index=False)
    pd.DataFrame(report).to_csv(out/'lfw_coverage_and_auc.csv',index=False)
    if fold_rows:
        t=pd.DataFrame(fold_rows)
        t.groupby('model').accuracy.agg(['mean','std','count']).to_csv(out/'lfw_10fold_summary.csv')

def main():
    ap=argparse.ArgumentParser(description='IABF complete 50-image Multi-PIE evaluation; full methodology')
    for name in ('manifest','baseline','proposed'):ap.add_argument('--'+name,required=True)
    ap.add_argument('--output',default='iabf_multipie_complete_results')
    ap.add_argument('--seed',type=int,default=42);ap.add_argument('--epochs',type=int,default=30)
    ap.add_argument('--eval-every',type=int,default=3);ap.add_argument('--learning-rate',type=float,default=1e-4)
    ap.add_argument('--weight-decay',type=float,default=1e-4);ap.add_argument('--arcface-margin',type=float,default=.3)
    ap.add_argument('--arcface-scale',type=float,default=30);ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--dry-run',action='store_true');ap.add_argument('--skip-ablations',action='store_true')
    ap.add_argument('--lfw-pairs',default=None,help='Optional official LFW pairs converted to CSV with path1,path2,is_same,fold')
    args=ap.parse_args();seed_all(args.seed);validated_pose_geometry()
    out=Path(args.output).resolve();out.mkdir(exist_ok=True,parents=True)
    frame=read_manifest(args.manifest)
    adapt,val,test=split_subjects(frame,args.seed)
    if len(adapt)!=50 or adapt.subject.nunique()!=10:raise AssertionError('Exactly 50 images / 10 train subjects required')
    for label,df in [('adapt',adapt),('validation',val),('test',test)]:df.to_csv(out/f'{label}_manifest.csv',index=False)
    checks={'training_images':len(adapt),'training_subjects':adapt.subject.nunique(),'validation_subjects':val.subject.nunique(),'test_subjects':test.subject.nunique(),'camera_ids':sorted(frame.camera.unique().tolist()),'camera_yaw_map':frame.groupby('camera').yaw_deg.first().to_dict()}
    (out/'manifest_audit.json').write_text(json.dumps(checks,indent=2,default=str))
    if args.dry_run:print('DRY RUN PASS',json.dumps(checks,indent=2,default=str));return
    for path in (args.baseline,args.proposed):
        if not Path(path).is_file():raise FileNotFoundError(path)
    device=torch.device(args.device)
    std=load_model(args.baseline,StandardRecognizer,device);original=load_model(args.proposed,CASIAWebFaceRecognizer,device)
    import face_alignment
    detector=face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,device=str(device),flip_input=False)
    sets={}
    for label,df in [('adapt',adapt),('validation',val),('test',test)]:sets[label]=precompute(df,detector,out,label)
    coverage=[]
    for label,(features,df) in sets.items():
        initial={'adapt':adapt,'validation':val,'test':test}[label]
        coverage.append({'split':label,'selected_images':len(initial),'processed_images':len(df),'processing_coverage':len(df)/len(initial),'original_subjects':initial.subject.nunique(),'surviving_subjects':df.subject.nunique()})
        if df.subject.nunique()<2:raise RuntimeError(label+' has insufficient surviving identities after quality-control')
    pd.DataFrame(coverage).to_csv(out/'preprocessing_coverage_summary.csv',index=False)
    models={'Standard':(std,'full'),'IABF_original':(original,'full')}
    best=[]
    modes=list(DESCRIPTORS) if not args.skip_ablations else ['full']
    for mode in modes:
        net=copy.deepcopy(original)
        net,record=adapt_arcface(net,*sets['adapt'],sets['validation'],device,out,mode,args.epochs,args.learning_rate,args.weight_decay,args.arcface_margin,args.arcface_scale,args.eval_every)
        best.append(record)
        models['IABF_50_'+mode]=(net,mode)
        torch.save({'model_state_dict':net.state_dict(),'model':'original-IABF-gate-only','mode':mode,'training_images':50,'arcface_margin':args.arcface_margin,'arcface_scale':args.arcface_scale,'epoch_selected_by_validation':record['best_epoch']},out/f'iabf_50_{mode}.pth')
    pd.DataFrame(best).to_csv(out/'validation_selected_epochs.csv',index=False)
    result=full_evaluate(models,sets['validation'],sets['test'],device,out,args.seed)
    quality_robustness({k:v for k,v in models.items() if k in ('Standard','IABF_original','IABF_50_full')},sets['validation'],sets['test'],device,out,args.seed)
    session_cross_evaluation({k:v for k,v in models.items() if k in ('Standard','IABF_original','IABF_50_full')},sets['test'][1],sets['test'][0],device,out,args.seed)
    if args.lfw_pairs:lfw_tenfold({k:v for k,v in models.items() if k in ('Standard','IABF_original','IABF_50_full')},args.lfw_pairs,detector,device,out)
    provenance={'seed':args.seed,'exact_adaptation_images':50,'adaptation_subjects':sorted(adapt.subject.unique().tolist()),'validation_subjects':sorted(val.subject.unique().tolist()),'test_subjects':sorted(test.subject.unique().tolist()),'baseline_sha256':digest(args.baseline),'proposed_sha256':digest(args.proposed),'manifest_sha256':digest(args.manifest),'architecture':'original 512D InceptionResNetV1 + residual (1+sigmoid) 11->64->512 gate','supervision':'ArcFace on 50 images; trainable temporary class weights + gate only; frozen backbone, projection, embedding head','quality_experiment':'deterministic perturbation applied to nonfrontal probes; gallery clean','retrained_descriptor_ablations':modes,'pose_convention':'YXZ returned in yaw_y, pitch_x, roll_z order; signs require dataset camera verification','warning':'No test accuracy is evidence of out-of-domain generalization unless full protocol and original checkpoints verified'}
    (out/'experiment_provenance.json').write_text(json.dumps(provenance,indent=2))
    print(result.to_string(index=False));print('OUTPUT:',out)

if __name__=='__main__':main()
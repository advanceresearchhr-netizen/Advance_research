"""Reconstruct exactly the original notebook's Penn Action manifest from .mat source labels."""
from pathlib import Path
from scipy.io import loadmat
import pandas as pd,numpy as np,argparse,json

def build(root):
 root=Path(root);rows=[]
 for path in sorted((root/'labels').glob('*.mat')):
  seq=path.stem;d=loadmat(path,squeeze_me=True)
  frames=sorted((root/'frames'/seq).glob('*.jpg'))
  xx=np.asarray(d['x']); yy=np.asarray(d['y'])
  action=str(np.asarray(d['action']).item())
  train=int(np.asarray(d['train']).item())
  rows.append({'sequence':seq,'action':action,'train':train,'rgb_frames':len(frames),'pose_frames':len(xx),'joints':xx.shape[1] if xx.ndim==2 else -1,'aligned':xx.ndim==2 and yy.shape==xx.shape and len(frames)==len(xx),'finite_pose':bool(np.isfinite(xx).all() and np.isfinite(yy).all())})
 df=pd.DataFrame(rows)
 if len(df)!=2326 or df.action.nunique()!=15 or not df.aligned.all() or not df.finite_pose.all():raise ValueError('Dataset integrity mismatch')
 if (df.train==1).sum()!=1258 or (df.train==-1).sum()!=1068:raise ValueError('Official split mismatch')
 return df
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 frame=build(a.data_root);Path(a.out).parent.mkdir(parents=True,exist_ok=True);frame.to_csv(a.out,index=False)
 print(json.dumps({'videos':len(frame),'classes':frame.action.nunique(),'train':int((frame.train==1).sum()),'test':int((frame.train==-1).sum())}))

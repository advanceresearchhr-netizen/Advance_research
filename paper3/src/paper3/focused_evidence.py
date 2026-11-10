"""Single-GPU focused evidence: mask-indicator ablation and ModDrop-style control.
All methods use the same frozen per-seed reference and train/validation split.
ModDrop-style is an independently implemented masking control, NOT a verified
reproduction of any published author's code. Official test is exploratory.
"""
import argparse,copy,json,os
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
from sklearn.metrics import f1_score
from torchvision.models import resnet18,ResNet18_Weights
from .data import make_splits
from .reviewer_campaign import extract,SEEDS
from .final_campaign import MaskAwareReference,mask_inputs
from .core import TemporalFusion
from .train import seed_all

class Control(nn.Module):
 def __init__(self,base_state,kind):
  super().__init__()
  self.kind=kind
  self.base=TemporalFusion();self.base.load_state_dict(base_state,strict=True)
  for p in self.base.parameters():p.requires_grad_(False)
  width=17 if kind=='availability' else 15
  self.adapter=nn.Sequential(nn.Linear(width,64),nn.GELU(),nn.Linear(64,15))
  nn.init.zeros_(self.adapter[-1].weight);nn.init.zeros_(self.adapter[-1].bias)
 def train(self,mode=True):
  super().train(mode);self.base.eval();return self
 def forward(self,r,p,m):
  with torch.no_grad():z=self.base(r,p)
  x=torch.cat((z,m.float()),dim=1) if self.kind=='availability' else z
  return z+self.adapter(x)

def train_control(base,tr,val,seed,kind,epochs):
 seed_all(seed+901)
 model=Control(base,kind).cuda()
 optimizer=torch.optim.AdamW(model.adapter.parameters(),lr=.001)
 best=-1.;checkpoint=None;history=[]
 train_r,train_p,train_y=tr;val_r,val_p,val_y=val
 masks=torch.tensor([[1,1],[0,1],[1,0]],device='cuda',dtype=torch.float32)
 for ep in range(epochs):
  model.train()
  order=torch.randperm(len(train_y),generator=torch.Generator().manual_seed(seed+ep+901))
  for ix in order.split(64):
   r=train_r[ix].cuda();p=train_p[ix].cuda();y=train_y[ix].cuda()
   m=masks[torch.randint(3,(len(ix),),device='cuda')]
   r,p=mask_inputs(r,p,m)
   optimizer.zero_grad(set_to_none=True)
   loss=nn.functional.cross_entropy(model(r,p,m),y)
   loss.backward();optimizer.step()
  model.eval();scores=[]
  with torch.inference_mode():
   for pair in [(1,1),(0,1),(1,0)]:
    pred=[]
    for ix in torch.arange(len(val_y)).split(128):
     m=torch.tensor(pair,device='cuda',dtype=torch.float32).expand(len(ix),-1)
     r,p=mask_inputs(val_r[ix].cuda(),val_p[ix].cuda(),m)
     pred.append(model(r,p,m).argmax(1).cpu())
    scores.append(float((torch.cat(pred)==val_y).float().mean()))
  score=float(np.mean(scores));history.append({'epoch':ep+1,'clean':scores[0],'missing_rgb':scores[1],'missing_pose':scores[2],'selection':score})
  if score>best:
   best=score;checkpoint={'model':copy.deepcopy(model.state_dict()),'epoch':ep+1,'validation':history[-1]}
 return checkpoint,history

def main():
 ap=argparse.ArgumentParser()
 for k in ('manifest','data-root','out','prior'):ap.add_argument('--'+k,required=True)
 ap.add_argument('--smoke',action='store_true')
 args=ap.parse_args()
 out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 prior=Path(args.prior)
 manifest=pd.read_csv(args.manifest,dtype={'sequence':str});manifest.sequence=manifest.sequence.str.zfill(4)
 pool=manifest[manifest.train==1].reset_index(drop=True)
 _,_,test=make_splits(args.manifest,42)
 if args.smoke:pool=pool.head(25).copy();test=test.head(6).copy()
 classes=sorted(manifest.action.unique());mapping={v:i for i,v in enumerate(classes)}
 bb=resnet18(weights=ResNet18_Weights.DEFAULT);bb.fc=nn.Identity();bb=bb.cuda().eval()
 for p in bb.parameters():p.requires_grad_(False)
 def cache(df):return extract(df,args.data_root,mapping,bb,'cuda',32)
 train=cache(pool);held=cache(test)
 index={str(z).zfill(4):i for i,z in enumerate(train[3])}
 def subset(frame):
  ix=torch.tensor([index[str(z).zfill(4)] for z in frame.sequence])
  return tuple(x[ix] for x in train[:3])
 del bb;torch.cuda.empty_cache()
 rows=[];history={}
 for seed in ((42,) if args.smoke else SEEDS):
  tr,val,_=make_splits(args.manifest,seed)
  if args.smoke:tr=pool.head(15);val=pool.tail(10)
  base=torch.load(prior/f'reference_seed{seed}.pt',map_location='cpu',weights_only=True)['model']
  a=subset(tr);b=subset(val)
  for kind in ('availability','logits_only'):
   ck,h=train_control(base,a,b,seed,kind,1 if args.smoke else 18)
   torch.save(ck,out/f'{kind}_seed{seed}.pt')
   history[f'{seed}_{kind}']=h
   net=Control(base,kind).cuda();net.load_state_dict(ck['model']);net.eval()
   for condition,pair in [('clean',(1,1)),('missing_rgb',(0,1)),('missing_pose',(1,0))]:
    preds=[];labels=held[2].numpy()
    with torch.inference_mode():
     for ix in torch.arange(len(labels)).split(128):
      m=torch.tensor(pair,device='cuda',dtype=torch.float32).expand(len(ix),-1)
      r,p=mask_inputs(held[0][ix].cuda(),held[1][ix].cuda(),m)
      preds.append(net(r,p,m).argmax(1).cpu())
    pred=torch.cat(preds).numpy()
    for video,y,yp in zip(held[3],labels,pred):
     rows.append({'seed':seed,'variant':kind,'condition':condition,'video_id':str(video).zfill(4),'label':int(y),'prediction':int(yp),'correct':int(y==yp)})
  print('FOCUSED_EVIDENCE_SEED_COMPLETE',seed,flush=True)
 df=pd.DataFrame(rows)
 if df.duplicated(['seed','variant','condition','video_id']).any():raise ValueError('Duplicate predictions')
 df.to_csv(out/'focused_predictions.csv',index=False)
 stats=[]
 for (seed,variant,condition),g in df.groupby(['seed','variant','condition']):
  stats.append({'seed':int(seed),'variant':variant,'condition':condition,'n':len(g),'accuracy':float(g.correct.mean()),'macro_f1':float(f1_score(g.label,g.prediction,labels=list(range(15)),average='macro',zero_division=0))})
 pd.DataFrame(stats).to_csv(out/'focused_metrics.csv',index=False)
 (out/'histories.json').write_text(json.dumps(history))
 (out/'provenance.json').write_text(json.dumps({'completed':not args.smoke,'smoke':args.smoke,'seeds':[42] if args.smoke else list(SEEDS),'comparators':['availability','logits_only'],'parameter_matching':'same 64-hidden adapter; input width differs by two availability flags','published_baseline_reproduction':False,'independent_external_dataset':False,'test_status':'Exploratory; official Penn Action test informed earlier design','source_commit':os.environ.get('PAPER3_COMMIT')},indent=2))
 print('FOCUSED_EVIDENCE_COMPLETE',flush=True)
if __name__=='__main__':main()

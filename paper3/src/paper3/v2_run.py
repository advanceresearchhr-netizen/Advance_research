"""One comprehensive Penn Action V2 job: 3 matched seeds, baseline + reliability router.
The published V1 results remain immutable. No test-based checkpoint selection.
"""
import argparse,copy,hashlib,json,os,random,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import accuracy_score,f1_score
from torchvision.models import resnet18,ResNet18_Weights
from .data import PennDataset,make_splits,frozen_features
from .train import fit,evaluate,seed_all
from .core import TemporalFusion
from .v2_model import ReliabilityRouter,augment_pose

def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def metric(logits,target):
 y=target.cpu().numpy();pred=logits.argmax(1).cpu().numpy()
 return {'accuracy':float(accuracy_score(y,pred)),
         'macro_f1':float(f1_score(y,pred,labels=list(range(15)),average='macro',zero_division=0))}
def extract(frame,root,map_,backbone,device,batch,mode='clean',sev=0.0,seed=42):
 ds=PennDataset(frame,root,map_,visual_fault=mode,severity=sev,seed=seed)
 dat=DataLoader(ds,batch_size=batch,shuffle=False,num_workers=4,pin_memory=(device=='cuda'))
 return frozen_features(dat,backbone,device)
def model_evaluate(net,arr,dev,availability=None):
 net.eval();logits=[];ys=[]
 loader=DataLoader(TensorDataset(*arr),batch_size=64,shuffle=False)
 with torch.inference_mode():
  for r,p,y in loader:
   logits.append(net(r.to(dev),p.to(dev),availability=availability).cpu())
   ys.append(y)
 return metric(torch.cat(logits),torch.cat(ys))
def fit_router(base_state,clean,corr,seed,dev,epochs=20):
 seed_all(seed);net=ReliabilityRouter(base_state).to(dev)
 optimizer=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=5e-4,weight_decay=1e-4)
 scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)
 xtrain,xval=clean
 # corr {name:(train_arrays,val_arrays,bad_modality_index)}
 history=[];best=-1.;result=None;bad_epochs=0
 names=['clean']+sorted(corr)
 n=xtrain[0].shape[0]
 for epoch in range(1,epochs+1):
  net.train()
  order=torch.randperm(n,generator=torch.Generator().manual_seed(seed+epoch))
  avg_loss=0.;seen=0
  for loc in order.split(64):
   which=names[(epoch+seen//64)%len(names)]
   if which=='clean':arr=xtrain;label=torch.ones(len(loc),2)
   else:
    arr=corr[which][0];label=torch.ones(len(loc),2)
    label[:,corr[which][2]]=0.
   rgb=arr[0][loc].to(dev);pose=arr[1][loc].to(dev);y=arr[2][loc].to(dev);label=label.to(dev)
   optimizer.zero_grad(set_to_none=True)
   out,aux=net(rgb,pose,return_aux=True)
   ce=nn.functional.cross_entropy(out,y)
   qloss=nn.functional.binary_cross_entropy_with_logits(aux['quality_logits'],label)
   # Auxiliary pure-modality heads must be discriminative, even under clean data.
   if which=='clean':
    aux_loss=(nn.functional.cross_entropy(net.rgb_head(net.reference.rgb_encoder(rgb)),y)+
       nn.functional.cross_entropy(net.pose_head(net.reference.pose_encoder(pose.flatten(2))),y))/2
   else:
    # Train only the undamaged specialist for corrupted examples.
    if corr[which][2]==0:aux_loss=nn.functional.cross_entropy(net.pose_head(net.reference.pose_encoder(pose.flatten(2))),y)
    else:aux_loss=nn.functional.cross_entropy(net.rgb_head(net.reference.rgb_encoder(rgb)),y)
   loss=ce+.2*aux_loss+.3*qloss
   loss.backward();nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad],1.)
   optimizer.step();avg_loss+=float(loss.item())*len(loc);seen+=len(loc)
  scheduler.step()
  clean_val=model_evaluate(net,xval,dev)
  # validation-matched RGB damage, pose failure, and clean accuracy
  corr_vals={}
  for name,(tr,va,bad) in corr.items():corr_vals[name]=model_evaluate(net,va,dev)
  mean_fault=float(np.mean([v['macro_f1'] for v in corr_vals.values()]))
  score=.6*clean_val['macro_f1']+.4*mean_fault
  record={'epoch':epoch,'train_loss':avg_loss/max(1,seen),'val_clean':clean_val,
          'val_corruptions':corr_vals,'selection_score':score}
  history.append(record)
  print('V2_EPOCH',seed,epoch,'loss',round(record['train_loss'],5),
        'clean_f1',round(clean_val['macro_f1'],4),'fault_f1',round(mean_fault,4),flush=True)
  if score>best:
   best=score;bad_epochs=0
   result={'model':copy.deepcopy(net.state_dict()),'epoch':epoch,'seed':seed,
           'selection_score':score,'validation':record,'architecture':'ReliabilityRouter'}
  else:bad_epochs+=1
  if bad_epochs>=8:break
 assert result is not None
 return result,history

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--manifest',required=True);p.add_argument('--data-root',required=True);p.add_argument('--out',required=True)
 p.add_argument('--device',default='cuda');p.add_argument('--feature-batch',type=int,default=32)
 p.add_argument('--baseline-epochs',type=int,default=35);p.add_argument('--v2-epochs',type=int,default=25)
 p.add_argument('--smoke',action='store_true')
 args=p.parse_args();seed_all(42)
 if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA GPU unavailable')
 root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
 manifest=pd.read_csv(args.manifest,dtype={'sequence':str});manifest.sequence=manifest.sequence.str.zfill(4)
 labels=sorted(manifest.action.unique());assert len(labels)==15
 mapping={a:i for i,a in enumerate(labels)}
 tr0,val0,test=make_splits(args.manifest,42)
 pool=manifest[manifest.train==1].reset_index(drop=True)
 if args.smoke:
  pool=pool.head(8).copy();test=test.head(8).copy()
 device=args.device
 bb=resnet18(weights=ResNet18_Weights.DEFAULT).to(device).eval();bb.fc=nn.Identity()
 for v in bb.parameters():v.requires_grad_(False)
 def cache(frame,mode='clean',sev=0.0,seed=42):
  return extract(frame,args.data_root,mapping,bb,device,args.feature_batch,mode,sev,seed)
 print('V2_CACHE_CLEAN_TRAIN',flush=True)
 pr,pp,py,pids=cache(pool)
 idx={s:i for i,s in enumerate(pids)}
 def select(ids,arr):
  loc=torch.tensor([idx[str(z).zfill(4)] for z in ids],dtype=torch.long)
  return tuple(v[loc] for v in arr)
 clean=(pr,pp,py)
 kinds=['frame_dropout','blur','body_occlusion','temporal_gap','low_light']
 rgb_cache={}
 for name in kinds:
  print('V2_CACHE_TRAIN',name,flush=True)
  r,p,y,ids=cache(pool,name,.5)
  assert ids==pids and torch.equal(y,py)
  rgb_cache[name]=(r,p,y)
 # pose failures generated on raw 2D pose annotations, at known training severity.
 gen=torch.Generator().manual_seed(739)
 pose_cache={}
 for name in ['pose_gap','pose_joint_dropout','pose_noise']:
  pose_cache[name]=(pr,augment_pose(pp,name,.5,gen),py)
 print('V2_CACHE_TEST',flush=True)
 tclean=cache(test)
 tcorrupt={}
 for name in kinds:
  print('V2_CACHE_TEST',name,flush=True)
  tcorrupt[name]=cache(test,name,.5)
 del bb
 if device=='cuda':torch.cuda.empty_cache()
 result_rows=[];history_all=[];selected=[]
 for seed in [42,123,2026] if not args.smoke else [42]:
  tr,val,te=make_splits(args.manifest,seed)
  if args.smoke:
   tr=tr[tr.sequence.isin(pids)].head(6)
   val=val[val.sequence.isin(pids)].head(2)
   if not len(val):val=pool.tail(2)
  assert len(set(tr.sequence)&set(val.sequence))==0
  if not args.smoke:assert set(te.sequence)==set(test.sequence)
  xtr=select(tr.sequence,clean);xval=select(val.sequence,clean)
  print('V2_TRAIN_BASELINE',seed,flush=True)
  base=fit(xtr,xval,seed,model_kind='paper2',epochs=args.baseline_epochs,device=device)
  torch.save(base,root/f'paper2_retrained_seed{seed}.pt')
  corrections={}
  for name,full in rgb_cache.items():
   corrections[name]=(select(tr.sequence,full),select(val.sequence,full),0)
  for name,full in pose_cache.items():
   corrections[name]=(select(tr.sequence,full),select(val.sequence,full),1)
  print('V2_TRAIN_ROUTER',seed,flush=True)
  trained,curve=fit_router(base['model'],(xtr,xval),corrections,seed,device,epochs=args.v2_epochs)
  torch.save(trained,root/f'paper3_v2_seed{seed}.pt')
  history_all.append({'seed':seed,'curve':curve})
  candidate=ReliabilityRouter(base['model']).to(device)
  candidate.load_state_dict(trained['model'],strict=True)
  reference=TemporalFusion().to(device);reference.load_state_dict(base['model'],strict=True)
  test_sets={'clean':tuple(tclean[:3])}
  for name,values in tcorrupt.items():test_sets[name]=tuple(values[:3])
  for name in pose_cache:
   gen=torch.Generator().manual_seed(8600+seed)
   test_sets[name]=(tclean[0],augment_pose(tclean[1],name,.5,gen),tclean[2])
  for name,arr in test_sets.items():
   for model,network in [('paper2',reference),('paper3_v2',candidate)]:
    met=model_evaluate(network,arr,device)
    result_rows.append({'seed':seed,'model':model,'condition':name,
                        'severity':0 if name=='clean' else .5,**met})
  for model,network in [('paper2',reference),('paper3_v2',candidate)]:
   for absent,avail in [('missing_rgb',[False,True]),('missing_pose',[True,False])]:
    # Original reference has no missing modality API. For baseline use zeroed modality.
    mask=torch.tensor(avail,dtype=torch.bool,device=device)[None,:].expand(len(tclean[0]),-1)
    rr=tclean[0].clone();qq=tclean[1].clone()
    if not avail[0]:rr.zero_()
    if not avail[1]:qq.zero_()
    if model=='paper2':met=model_evaluate(network,(rr,qq,tclean[2]),device)
    else:
     network.eval();logit=[]
     with torch.inference_mode():
      for a in range(0,len(rr),64):
       logit.append(network(rr[a:a+64].to(device),qq[a:a+64].to(device),
         availability=mask[a:a+64]).cpu())
     met=metric(torch.cat(logit),tclean[2])
    result_rows.append({'seed':seed,'model':model,'condition':absent,'severity':1.,**met})
  selected.append({'seed':seed,'paper2_epoch':base['epoch'],'v2_epoch':trained['epoch'],
                   'v2_selection_score':trained['selection_score']})
  (root/'progress.json').write_text(json.dumps({'completed_seeds':selected},indent=2))
  print('V2_SEED_COMPLETE',seed,flush=True)
 pd.DataFrame(result_rows).to_csv(root/'results_long.csv',index=False)
 (root/'epoch_history.json').write_text(json.dumps(history_all,indent=2))
 (root/'selection.json').write_text(json.dumps(selected,indent=2))
 tab=pd.DataFrame(result_rows)
 summary=tab.groupby(['model','condition'],as_index=False).agg(
   accuracy_mean=('accuracy','mean'),accuracy_std=('accuracy','std'),
   macro_f1_mean=('macro_f1','mean'),macro_f1_std=('macro_f1','std'))
 summary.to_csv(root/'summary.csv',index=False)
 meta={'dataset':'Penn Action','train_pool_size':1258,'official_test_size':1068,
       'seeds':[42,123,2026],'test_evaluated_after_validation_selection':True,
       'train_corruptions':kinds+list(pose_cache),'severity':.5,
       'reference':'Retrained frozen baseline per seed',
       'source_commit':os.environ.get('PAPER3_COMMIT','unknown'),
       'manifest_sha256':digest(args.manifest),'completed':not args.smoke,
       'smoke':args.smoke}
 (root/'provenance.json').write_text(json.dumps(meta,indent=2))
 print('PAPER3_V2_COMPLETED' if not args.smoke else 'PAPER3_V2_SMOKE_COMPLETED',flush=True)
if __name__=='__main__':main()

"""Final Penn Action reviewer evidence: independent matched-head training and diagnostics.

Calls the frozen 2026-10-19 reviewer campaign first, then adds NEW independently
trained mask-aware and unsupervised-gate controls per seed, paired predictions,
per-class errors, valid clean-versus-corrupted quality labels, and qualitative
sample descriptors. No cross-dataset/predicted-pose result is invented.
This is *exploratory*: prior Penn Action official test observations informed design.
"""
import argparse,copy,json,os,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import roc_auc_score,average_precision_score,confusion_matrix,classification_report
from torchvision.models import resnet18,ResNet18_Weights
from .reviewer_campaign import main as old_main,extract,stats,paired_uncertainty,SEEDS,POSE,RGB_SEEN,RGB_UNSEEN
from .data import make_splits
from .core import TemporalFusion
from .v2_model import ReliabilityRouter,augment_pose
from .train import seed_all

class MaskAwareReference(nn.Module):
 """Frozen original reference logits + trained availability-conditional correction.
 Unlike zero-fill baseline this adapter is explicitly trained on all 3 masks."""
 def __init__(self,base_state):
  super().__init__()
  self.base=TemporalFusion();self.base.load_state_dict(base_state)
  for p in self.base.parameters():p.requires_grad_(False)
  self.adapter=nn.Sequential(nn.Linear(17,64),nn.GELU(),nn.Linear(64,15))
  nn.init.zeros_(self.adapter[-1].weight);nn.init.zeros_(self.adapter[-1].bias)
 def train(self,mode=True):
  super().train(mode);self.base.eval();return self
 def forward(self,r,p,mask):
  with torch.no_grad():logits=self.base(r,p)
  return logits+self.adapter(torch.cat([logits,mask.float()],1))

def mask_inputs(r,p,m):
 return r*m[:,0,None,None],p*m[:,1,None,None,None]

def train_mask(base_state,tr,val,seed,device,epochs=18):
 seed_all(seed+750)
 net=MaskAwareReference(base_state).to(device)
 opt=torch.optim.AdamW(net.adapter.parameters(),lr=.001)
 hist=[];best=float('-inf');ck=None
 tr_x,tr_p,tr_y=tr;vl_x,vl_p,vl_y=val
 for e in range(epochs):
  net.train()
  ix=torch.randperm(len(tr_y),generator=torch.Generator().manual_seed(seed+e+700))
  for ids in ix.split(64):
   x=tr_x[ids].to(device);p=tr_p[ids].to(device);y=tr_y[ids].to(device)
   masks=torch.tensor([[1,1],[0,1],[1,0]],device=device,dtype=torch.float32)
   m=masks[torch.randint(3,(len(ids),),device=device)]
   xr,pr=mask_inputs(x,p,m)
   opt.zero_grad(set_to_none=True)
   loss=nn.functional.cross_entropy(net(xr,pr,m),y)
   loss.backward();opt.step()
  net.eval();scores=[]
  with torch.inference_mode():
   for mask in [(1,1),(0,1),(1,0)]:
    preds=[]
    for ids in torch.arange(len(vl_y)).split(128):
     r=vl_x[ids].to(device);p=vl_p[ids].to(device)
     m=torch.tensor(mask,device=device).float()[None,:].expand(len(ids),-1)
     rr,pp=mask_inputs(r,p,m)
     preds.append(net(rr,pp,m).argmax(1).cpu())
    scores.append(float((torch.cat(preds)==vl_y).float().mean()))
  score=float(np.mean(scores))
  hist.append({'epoch':e+1,'val_mask_average_accuracy':score,'clean':scores[0],'missing_rgb':scores[1],'missing_pose':scores[2]})
  if score>best:best=score;ck={'model':copy.deepcopy(net.state_dict()),'epoch':e+1,'validation':hist[-1],'seed':seed}
 assert ck
 return ck,hist

def train_no_quality(base_state,tr,val,corrupt,seed,device,epochs=20):
 """True TRAINED no-reliability-supervision ablation, with same CE+aux weights,
 qloss coefficient exactly zero. Selection uses identical val score formula."""
 seed_all(seed)
 net=ReliabilityRouter(base_state).to(device)
 opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=5e-4,weight_decay=1e-4)
 sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
 names=['clean']+sorted(corrupt)
 history=[];best=-1e9;result=None;bad=0;n=len(tr[2])
 for ep in range(1,epochs+1):
  net.train();idx=torch.randperm(n,generator=torch.Generator().manual_seed(seed+ep))
  seen=0
  for loc in idx.split(64):
   which=names[(ep+seen//64)%len(names)]
   arr=tr if which=='clean' else corrupt[which][0]
   r=arr[0][loc].to(device);p=arr[1][loc].to(device);y=arr[2][loc].to(device)
   opt.zero_grad(set_to_none=True)
   logits,aux=net(r,p,return_aux=True)
   with torch.no_grad():
    emb_r=net.reference.rgb_encoder(r)
    emb_p=net.reference.pose_encoder(p.flatten(2))
   if which=='clean':
    extra=(nn.functional.cross_entropy(net.rgb_head(emb_r),y)+nn.functional.cross_entropy(net.pose_head(emb_p),y))/2
   elif corrupt[which][2]==0:extra=nn.functional.cross_entropy(net.pose_head(emb_p),y)
   else:extra=nn.functional.cross_entropy(net.rgb_head(emb_r),y)
   loss=nn.functional.cross_entropy(logits,y)+.2*extra # no reliability BCE
   loss.backward();nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad],1.)
   opt.step();seen+=len(loc)
  sch.step()
  net.eval()
  def accuracy(a):
   logits=[]
   with torch.inference_mode():
    for loc in torch.arange(len(a[2])).split(128):
     logits.append(net(a[0][loc].to(device),a[1][loc].to(device)).argmax(1).cpu())
   from sklearn.metrics import f1_score
   return float(f1_score(a[2].numpy(),torch.cat(logits).numpy(),labels=list(range(15)),average='macro',zero_division=0))
  scores=[accuracy(val)]+[accuracy(va) for _,va,_ in corrupt.values()]
  score=.6*scores[0]+.4*float(np.mean(scores[1:]))
  history.append({'epoch':ep,'selection_score':score,'val_clean_f1':scores[0]})
  if score>best:
   best=score;result={'model':copy.deepcopy(net.state_dict()),'seed':seed,'epoch':ep,'selection_score':score};bad=0
  else:bad+=1
  if bad>=8:break
 return result,history

def inference(model,r,p,availability,device,batch=128):
 logits=[];qvalues=[]
 model.eval()
 with torch.inference_mode():
  for i in range(0,len(r),batch):
   a=r[i:i+batch].to(device);b=p[i:i+batch].to(device)
   m=torch.tensor(availability,dtype=torch.float32,device=device)[None,:].expand(len(a),-1)
   a,b=mask_inputs(a,b,m)
   if isinstance(model,MaskAwareReference):log=model(a,b,m)
   else:
    log,aux=model(a,b,availability=m.bool(),return_aux=True)
    qvalues.append(aux['reliability'].cpu())
   logits.append(log.cpu())
 return torch.cat(logits),torch.cat(qvalues) if qvalues else None

def matched_final_comparisons(frame,smoke=False):
 """Validate identical video identity and labels BEFORE paired statistical tests.
 CSV readers often infer numeric Penn Action IDs as integers, unlike fresh strings."""
 from scipy.stats import binomtest
 frame=frame.copy()
 frame['video_id']=frame['video_id'].astype(str).str.zfill(4)
 key=['seed','condition','severity','variant','video_id']
 if frame[key].isna().any().any() or frame.duplicated(key).any():
  raise ValueError('Missing/duplicate prediction identities in final comparisons')
 results=[]
 for (seed,condition,severity),part in frame.groupby(['seed','condition','severity']):
  pivot=part.pivot(index='video_id',columns='variant',values='correct')
  if 'v2' not in pivot:continue
  label_map=part[part.variant=='v2'].set_index('video_id')['class_id']
  for alternative in ('trained_mask_aware','trained_no_quality_supervision','specialist_rule'):
   if alternative not in pivot:continue
   # Every evaluated model must predict EXACTLY the same held-out video IDs.
   a_ids=set(part.loc[part.variant=='v2','video_id'])
   b_ids=set(part.loc[part.variant==alternative,'video_id'])
   if not a_ids or a_ids!=b_ids:
    raise ValueError(f'Paired prediction IDs mismatch: {seed} {condition} {severity} {alternative}; V2={len(a_ids)} competitor={len(b_ids)} overlap={len(a_ids & b_ids)}')
   sub=pivot[['v2',alternative]].dropna()
   if len(sub)!=len(a_ids):
    raise ValueError(f'Incomplete paired predictions: {seed} {condition} {alternative}')
   a=sub['v2'].to_numpy(dtype=float);b=sub[alternative].to_numpy(dtype=float)
   gold=label_map.loc[sub.index].to_numpy()
   other_labels=part[part.variant==alternative].set_index('video_id').loc[sub.index,'class_id'].to_numpy()
   if not np.array_equal(gold,other_labels):
    raise ValueError(f'Label mismatch between V2 and {alternative}')
   rng=np.random.default_rng(int(seed)+int(float(severity)*1000)+len(condition))
   clusters=[np.flatnonzero(gold==c) for c in np.unique(gold)]
   if not clusters or any(len(group)==0 for group in clusters):
    raise ValueError('No valid strata for paired bootstrap')
   boot=[]
   for _ in range(80 if smoke else 1500):
    ix=np.concatenate([rng.choice(group,len(group),replace=True) for group in clusters])
    boot.append(float((a[ix]-b[ix]).mean()))
   n10=int(((a==1)&(b==0)).sum());n01=int(((a==0)&(b==1)).sum())
   results.append({'seed':seed,'condition':condition,'severity':severity,'comparator':alternative,
    'matched_samples':len(a),'delta_accuracy_v2_minus_comparator':float((a-b).mean()),
    'ci95_lower':float(np.percentile(boot,2.5)),'ci95_upper':float(np.percentile(boot,97.5)),
    'mcnemar_p_exploratory':float(binomtest(min(n10,n01),n10+n01,.5).pvalue) if n10+n01 else 1.})
 if not results:raise ValueError('No paired model comparisons; inspect variant IDs and conditions')
 return pd.DataFrame(results)

def classification_outputs(df,out,action_names):
 """Human-readable class confusion/error cases using existing predictions."""
 names=list(action_names)
 conf=[];rank=[];cases=[]
 for (seed,cond,severity,variant),t in df.groupby(['seed','condition','severity','variant']):
  if cond not in ['clean','missing_rgb','missing_pose','frame_dropout','temporal_gap','pose_gap']:continue
  true=t.class_id.to_numpy();pred=t.prediction.to_numpy()
  matrix=confusion_matrix(true,pred,labels=list(range(15)))
  for i in range(15):
   denom=(true==i).sum()
   rank.append({'seed':seed,'condition':cond,'severity':severity,'variant':variant,'class_id':i,'class_name':names[i],
                'support':int(denom),'correct':int(matrix[i,i]),'recall':float(matrix[i,i]/denom) if denom else None})
   for j in range(15):
    if i!=j and matrix[i,j]:
     conf.append({'seed':seed,'condition':cond,'severity':severity,'variant':variant,
                  'true_class':i,'predicted_class':j,'count':int(matrix[i,j])})
  if variant=='v2':
   missed=t[t.correct==0].sort_values('confidence',ascending=False).head(12)
   for _,v in missed.iterrows():
    cases.append({'seed':seed,'condition':cond,'severity':severity,
      'video_id':v.video_id,'true_class':int(v.class_id),'predicted_class':int(v.prediction),
      'wrong_prediction_confidence':float(v.confidence),
      'interpretation':'high-confidence error; inspect raw RGB frames and pose trajectory before making a causal assertion'})
 pd.DataFrame(rank).to_csv(out/'per_class_metrics.csv',index=False)
 pd.DataFrame(conf).to_csv(out/'confusion_errors.csv',index=False)
 pd.DataFrame(cases).to_csv(out/'qualitative_case_index.csv',index=False)

def main():
 ap=argparse.ArgumentParser()
 for k in ['manifest','data-root','out']:ap.add_argument('--'+k,required=True)
 ap.add_argument('--device',default='cuda')
 ap.add_argument('--smoke',action='store_true')
 ap.add_argument('--feature-batch',type=int,default=32)
 ap.add_argument('--baseline-epochs',type=int,default=35)
 ap.add_argument('--v2-epochs',type=int,default=25)
 args=ap.parse_args()
 out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 # First, execute the original full reviewer campaign, with identical configs.
 print('FINAL_CAMPAIGN_PHASE_1_REFERENCE_V2_AND_INFERENCE_ABLATIONS',flush=True)
 old_main()
 print('FINAL_CAMPAIGN_PHASE_2_TRAINED_CONTROLS',flush=True)
 device=args.device
 manifest=pd.read_csv(args.manifest,dtype={'sequence':str});manifest.sequence=manifest.sequence.str.zfill(4)
 pool=manifest[manifest.train==1].reset_index(drop=True)
 _,_,test=make_splits(args.manifest,42)
 if args.smoke:pool=pool.head(25).copy();test=test.head(6).copy()
 classes=sorted(manifest.action.unique());mapping={x:i for i,x in enumerate(classes)}
 bb=resnet18(weights=ResNet18_Weights.DEFAULT);bb.fc=nn.Identity()
 bb=bb.to(device).eval()
 for par in bb.parameters():par.requires_grad_(False)
 def cache(df,mode='clean',severity=0.):
  return extract(df,args.data_root,mapping,bb,device,args.feature_batch,mode,severity)
 pr,pp,py,pids=cache(pool)
 index={z:i for i,z in enumerate(pids)}
 def select(frame,arr):
  ix=torch.tensor([index[str(z).zfill(4)] for z in frame.sequence])
  return tuple(q[ix] for q in arr)
 train_clean=(pr,pp,py)
 rgbcache={}
 for name in RGB_SEEN:
  features=cache(pool,name,.5)
  if features[3]!=pids or not torch.equal(features[2],py):
   raise ValueError(f'RGB training cache is misaligned for {name}')
  rgbcache[name]=features[:3]
 gen=torch.Generator().manual_seed(739)
 posecache={name:(pr,augment_pose(pp,name,.5,gen),py) for name in POSE}
 test_clean=cache(test)
 if len(test_clean[3])!=len(set(test_clean[3])) or len(test_clean[3])!=len(test):
  raise ValueError('Test video IDs must be unique and cover every official-test sample')
 testsets={('clean',0.):test_clean[:3],('missing_rgb',1.):test_clean[:3],
           ('missing_pose',1.):test_clean[:3]}
 for name in ['frame_dropout','temporal_gap','blur','compression']:
  for s in ([.5] if args.smoke else [.25,.5,.75]):
   testsets[(name,s)]=cache(test,name,s)[:3]
 for name in ['pose_gap','pose_noise']:
  for s in ([.5] if args.smoke else [.25,.5,.75]):
   gen=torch.Generator().manual_seed(8600+int(s*100))
   testsets[(name,s)]=(test_clean[0],augment_pose(test_clean[1],name,s,gen),test_clean[2])
 del bb
 if device=='cuda':torch.cuda.empty_cache()
 detailed=[];metrics=[];histories=[]
 for seed in ([42] if args.smoke else SEEDS):
  seed_all(seed)
  tr,val,_=make_splits(args.manifest,seed)
  if args.smoke:tr=pool.head(15).copy();val=pool.tail(10).copy()
  xtr=select(tr,train_clean);xval=select(val,train_clean)
  bck=torch.load(out/f'reference_seed{seed}.pt',map_location='cpu',weights_only=False)
  maskck,maskhist=train_mask(bck['model'],xtr,xval,seed,device,epochs=1 if args.smoke else 18)
  torch.save(maskck,out/f'mask_aware_seed{seed}.pt')
  masknet=MaskAwareReference(bck['model']).to(device);masknet.load_state_dict(maskck['model'])
  corrupt={}
  for name,data in rgbcache.items():corrupt[name]=(select(tr,data),select(val,data),0)
  for name,data in posecache.items():corrupt[name]=(select(tr,data),select(val,data),1)
  nocheck,nohist=train_no_quality(bck['model'],xtr,xval,corrupt,seed,device,epochs=1 if args.smoke else args.v2_epochs)
  torch.save(nocheck,out/f'no_quality_seed{seed}.pt')
  noqnet=ReliabilityRouter(bck['model']).to(device);noqnet.load_state_dict(nocheck['model'])
  histories.append({'seed':seed,'mask_aware':maskhist,'without_reliability_supervision':nohist})
  for (cond,sev),arr in testsets.items():
   availability=(False,True) if cond=='missing_rgb' else (True,False) if cond=='missing_pose' else (True,True)
   for label,net in [('trained_mask_aware',masknet),('trained_no_quality_supervision',noqnet)]:
    lg,q=inference(net,arr[0],arr[1],availability,device)
    m=stats(lg,arr[2])
    metrics.append({'seed':seed,'condition':cond,'severity':sev,'variant':label,**m})
    probs=lg.softmax(-1).numpy();pred=probs.argmax(1)
    for i,z in enumerate(test_clean[3]):
     detailed.append({'seed':seed,'condition':cond,'severity':sev,'variant':label,'video_id':z,
       'class_id':int(arr[2][i]),'prediction':int(pred[i]),
       'confidence':float(probs[i,pred[i]]),'correct':int(pred[i]==int(arr[2][i]))})
   print('FINAL_TRAINED_BASELINES_EVALUATED',seed,cond,sev,flush=True)
  print('FINAL_SEED_COMPLETE',seed,flush=True)
 pd.DataFrame(metrics).to_csv(out/'trained_controls_metrics.csv',index=False)
 pd.DataFrame(detailed).to_csv(out/'trained_controls_predictions.csv',index=False)
 (out/'trained_controls_history.json').write_text(json.dumps(histories,indent=2))
 first=pd.read_csv(out/'predictions_by_video.csv',dtype={'video_id':str})
 combo=pd.concat([first,pd.DataFrame(detailed)],ignore_index=True)
 combo['video_id']=combo['video_id'].astype(str).str.zfill(4)
 classification_outputs(combo,out,classes)
 comparisons=matched_final_comparisons(combo,smoke=args.smoke)
 comparisons.to_csv(out/'final_matched_comparisons.csv',index=False)
 # High-quality per-sample reliability detection: matched clean/corrupt pairs.
 # Evaluate actual trained V2 router, rather than pooling corruption types
 # without a clean negative control.
 detector=[]
 for seed in ([42] if args.smoke else SEEDS):
  bck=torch.load(out/f'reference_seed{seed}.pt',map_location='cpu',weights_only=False)
  v2ck=torch.load(out/f'v2_seed{seed}.pt',map_location='cpu',weights_only=False)
  v2net=ReliabilityRouter(bck['model']).to(device);v2net.load_state_dict(v2ck['model'])
  _,clean_q=inference(v2net,test_clean[0],test_clean[1],(True,True),device)
  for name,sensor in [('frame_dropout',0),('temporal_gap',0),('blur',0),('compression',0),
                      ('pose_gap',1),('pose_noise',1)]:
   for sev in ([.5] if args.smoke else [.25,.5,.75]):
    arr=testsets[(name,sev)]
    _,dam_q=inference(v2net,arr[0],arr[1],(True,True),device)
    y=np.r_[np.zeros(len(clean_q)),np.ones(len(dam_q))]
    z=np.r_[1-clean_q[:,sensor].numpy(),1-dam_q[:,sensor].numpy()]
    detector.append({'seed':seed,'condition':name,'severity':sev,'modality':['rgb','pose'][sensor],
        'clean_corrupt_detection_auroc':float(roc_auc_score(y,z)),
        'clean_corrupt_detection_auprc':float(average_precision_score(y,z)),
        'mean_clean_reliability':float(clean_q[:,sensor].mean()),
        'mean_corrupt_reliability':float(dam_q[:,sensor].mean())})
 pd.DataFrame(detector).to_csv(out/'clean_vs_corrupt_detection.csv',index=False)
 old=json.loads((out/'provenance.json').read_text())
 old.update({'final_additional_controls':True,'completed':not args.smoke,
  'trained_mask_aware':True,'trained_no_quality_supervision':True,
  'qualitative_cases':'Automatically indexed high-confidence misclassifications; not manually inspected video examples',
  'external_dataset':'Not run: no verified independent RGB-pose dataset supplied',
  'predicted_keypoints':'Not run: no validated prediction pipeline in provided protocol',
  'test_status':'Exploratory: earlier Penn Action official test already informed design'})
 (out/'provenance.json').write_text(json.dumps(old,indent=2))
 print('PAPER3_FINAL_CAMPAIGN_COMPLETE' if not args.smoke else 'PAPER3_FINAL_SMOKE_COMPLETE',flush=True)
if __name__=='__main__':main()

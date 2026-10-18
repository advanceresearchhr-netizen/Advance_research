"""Paper 3 confirmatory-style Penn Action audit, ONE bounded GPU campaign.

Primary comparisons: V2 versus specialist-only routing, using identical trained
heads, seed, labels, and samples. All earlier test-informed design is disclosed.
Independent cross-dataset/predicted-keypoint validation is NOT claimed here.
"""
import argparse, csv, hashlib, json, math, os, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from scipy.stats import binomtest
from torchvision.models import resnet18, ResNet18_Weights
from .data import PennDataset, make_splits, frozen_features
from .train import fit, seed_all
from .core import TemporalFusion
from .v2_model import ReliabilityRouter, augment_pose
from .v2_run import fit_router

SEEDS=(42,123,2026)
SEVERITIES=(.25,.5,.75)
RGB_SEEN=('frame_dropout','blur','body_occlusion','temporal_gap','low_light')
RGB_UNSEEN=('compression','irregular_sampling')
POSE=('pose_gap','pose_joint_dropout','pose_noise')

def sha256(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def extract(df,root,labels,backbone,device,bs,mode='clean',sev=0.,seed=42):
 dataset=PennDataset(df,root,labels,visual_fault=mode,severity=sev,seed=seed)
 loader=torch.utils.data.DataLoader(dataset,batch_size=bs,shuffle=False,num_workers=4,pin_memory=device=='cuda')
 return frozen_features(loader,backbone,device)

def pred_logits(base,router,arr,device,variant='v2',availability=(True,True),bs=128):
 """Identical V2 checkpoints: ablate routing at INFERENCE, not extra training."""
 base.eval();router.eval()
 r,p,y=arr
 out=[];quality=[];refout=[];components=[]
 for s in range(0,len(y),bs):
  a=r[s:s+bs].to(device);b=p[s:s+bs].to(device)
  present=torch.tensor(availability,dtype=torch.bool,device=device)[None,:].expand(len(a),-1)
  aa=a if availability[0] else torch.zeros_like(a)
  bb=b if availability[1] else torch.zeros_like(b)
  with torch.inference_mode():
   if variant=='legacy_zero':
    logits=base(aa,bb)
    aux=None
   else:
    final,aux=router(aa,bb,availability=present,return_aux=True)
    rlog=router.rgb_head(router.reference.rgb_encoder(aa))
    plog=router.pose_head(router.reference.pose_encoder(bb.flatten(2)))
    ref=aux['baseline_logits']
    if variant=='v2':logits=final
    elif variant=='specialist_rule':
     if not availability[0]:logits=plog
     elif not availability[1]:logits=rlog
     else:logits=ref
    elif variant=='specialist_equal':
     if not availability[0]:logits=plog
     elif not availability[1]:logits=rlog
     else:logits=.5*(rlog+plog)
    elif variant=='specialist_quality_no_supervision':
     # DO NOT call this a trained no-supervision ablation. This is the
     # same trained quality net with its routing DISABLED, as a sensitivity test.
     if not availability[0]:logits=plog
     elif not availability[1]:logits=rlog
     else:logits=.5*(ref+.5*(rlog+plog))
    else:raise ValueError(variant)
  out.append(logits.cpu())
  if aux is not None:
   quality.append(aux['reliability'].cpu())
   refout.append(ref.cpu())
 return torch.cat(out), (torch.cat(quality) if quality else None), (torch.cat(refout) if refout else None)

def stats(logits,y):
 probs=logits.softmax(-1).numpy(); labels=y.numpy(); pred=probs.argmax(-1)
 conf=probs.max(1)
 nll=float(-np.log(np.clip(probs[np.arange(len(labels)),labels],1e-10,1)).mean())
 ece=0.
 for k in range(15):
  lo=k/15;hi=(k+1)/15
  mask=(conf>=lo)&((conf<hi) if k<14 else (conf<=hi))
  if mask.any():ece+=mask.mean()*abs((pred[mask]==labels[mask]).mean()-conf[mask].mean())
 return {'accuracy':float(accuracy_score(labels,pred)),
         'macro_f1':float(f1_score(labels,pred,labels=list(range(15)),average='macro',zero_division=0)),
         'nll':nll,'ece15':float(ece)}

def paired_uncertainty(rows,seed=2026,n_boot=1000):
 """Stratified paired bootstrap; separate within-seed uncertainty from
 between-seed variability. Holm-adjust exact McNemar descriptive p-values."""
 tab=pd.DataFrame(rows);res=[]
 for key,group in tab.groupby(['seed','condition','severity']):
  piv=group.pivot(index='video_id',columns='variant',values='correct')
  label_map=group[group.variant=='v2'].set_index('video_id')['class_id']
  if not {'v2','specialist_rule','legacy_zero'}.issubset(piv.columns):continue
  labels=label_map.loc[piv.index].to_numpy()
  rng=np.random.default_rng(seed+int(key[0])*11+len(str(key)))
  for comparator in ['specialist_rule','legacy_zero']:
   a=piv['v2'].to_numpy(dtype=float);b=piv[comparator].to_numpy(dtype=float)
   delta=float((a-b).mean())
   draws=[]
   for k in range(n_boot):
    parts=[]
    for cls in np.unique(labels):
     ix=np.flatnonzero(labels==cls)
     parts.append(rng.choice(ix,len(ix),replace=True))
    j=np.concatenate(parts)
    draws.append(float((a[j]-b[j]).mean()))
   b10=int(((a==1)&(b==0)).sum());b01=int(((a==0)&(b==1)).sum())
   p=float(binomtest(min(b10,b01),b10+b01,.5).pvalue) if (b10+b01)>0 else 1.
   res.append({'seed':int(key[0]),'condition':key[1],'severity':key[2],
               'comparison':'v2_vs_'+comparator,'delta_accuracy':delta,
               'ci95_low':float(np.percentile(draws,2.5)),
               'ci95_high':float(np.percentile(draws,97.5)),
               'discordant_v2_only':b10,'discordant_comparator_only':b01,
               'mcnemar_exact_p_exploratory':p})
 frame=pd.DataFrame(res)
 # Familywise multiplicity adjustment is across all reported exploratory tests.
 if len(frame):
  order=np.argsort(frame.mcnemar_exact_p_exploratory.to_numpy())
  p=frame.mcnemar_exact_p_exploratory.to_numpy();adj=np.empty(len(p))
  cumul=0.
  for rank,j in enumerate(order):
   cumul=max(cumul,(len(p)-rank)*p[j]);adj[j]=min(1.,cumul)
  frame['holm_adjusted_p_exploratory']=adj
 return frame

def latency(net,example,device,repeats=80):
 net.eval();r,p=example;r=r[:1].to(device);p=p[:1].to(device)
 with torch.inference_mode():
  for _ in range(15):net(r,p)
  if device=='cuda':torch.cuda.synchronize()
  times=[]
  for _ in range(repeats):
   t=time.perf_counter();net(r,p)
   if device=='cuda':torch.cuda.synchronize()
   times.append((time.perf_counter()-t)*1000)
 return {'median_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)),
         'all_parameters':int(sum(t.numel() for t in net.parameters())),
         'trainable_parameters':int(sum(t.numel() for t in net.parameters() if t.requires_grad))}

def main():
 ap=argparse.ArgumentParser()
 for v in ['manifest','data-root','out']:ap.add_argument('--'+v,required=True)
 ap.add_argument('--device',default='cuda')
 ap.add_argument('--feature-batch',type=int,default=32)
 ap.add_argument('--baseline-epochs',type=int,default=35)
 ap.add_argument('--v2-epochs',type=int,default=25)
 ap.add_argument('--smoke',action='store_true')
 args=ap.parse_args();seed_all(42)
 device=args.device
 if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('No GPU')
 out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 manifest=pd.read_csv(args.manifest,dtype={'sequence':str})
 manifest.sequence=manifest.sequence.str.zfill(4)
 tr0,va0,te=make_splits(args.manifest,42)
 pool=manifest[manifest.train==1].reset_index(drop=True)
 if args.smoke:
  # Small integrity test only; not valid performance evidence.
  pool=pool.head(25).copy();te=te.head(6).copy()
 categories=sorted(manifest.action.unique()); mapping={c:i for i,c in enumerate(categories)}
 backbone=resnet18(weights=ResNet18_Weights.DEFAULT)
 backbone.fc=nn.Identity();backbone=backbone.to(device).eval()
 for param in backbone.parameters():param.requires_grad_(False)
 def cache(frame,name='clean',severity=0,seed=42):
  return extract(frame,args.data_root,mapping,backbone,device,args.feature_batch,name,severity,seed)
 print('CAMPAIGN_CACHING_TRAIN',flush=True)
 pr,pp,py,pids=cache(pool);idx={k:i for i,k in enumerate(pids)}
 def pick(frame,arr):
  loc=torch.tensor([idx[str(k).zfill(4)] for k in frame.sequence],dtype=torch.long)
  return tuple(v[loc] for v in arr)
 clean_train=(pr,pp,py)
 rgbtrain={}
 for name in RGB_SEEN:
  vals=cache(pool,name,.5)
  assert vals[3]==pids and torch.equal(vals[2],py)
  rgbtrain[name]=vals[:3]
 gen=torch.Generator().manual_seed(739)
 posetrain={name:(pr,augment_pose(pp,name,.5,gen),py) for name in POSE}
 print('CAMPAIGN_CACHING_TEST',flush=True)
 tclean=cache(te);assert len(set(tclean[3]))==len(te)
 # Reference cache evaluated under predefined severities, and held-out types.
 testsets={('clean',0.):tclean[:3]}
 for name in RGB_SEEN+RGB_UNSEEN:
  for severity in SEVERITIES if not args.smoke else (.5,):
   print('CAMPAIGN_TEST_CACHE',name,severity,flush=True)
   vals=cache(te,name,severity)
   assert vals[3]==tclean[3] and torch.equal(vals[2],tclean[2])
   testsets[(name,severity)]=vals[:3]
 for name in POSE:
  for severity in SEVERITIES if not args.smoke else (.5,):
   gen=torch.Generator().manual_seed(8600+int(severity*100))
   testsets[(name,severity)]=(tclean[0],augment_pose(tclean[1],name,severity,gen),tclean[2])
 del backbone
 if device=='cuda':torch.cuda.empty_cache()
 per_video=[];summary=[];curves=[];selection=[];quality_rows=[];profiles=[]
 for seed in SEEDS if not args.smoke else (42,):
  tr,val,tst=make_splits(args.manifest,seed)
  if args.smoke:
   tr=pool.head(15).copy();val=pool.tail(10).copy()
  assert not (set(tr.sequence)&set(val.sequence))
  a=pick(tr,clean_train);b=pick(val,clean_train)
  print('CAMPAIGN_TRAIN_BASELINE',seed,flush=True)
  base_ck=fit(a,b,seed,model_kind='paper2',epochs=args.baseline_epochs,device=device)
  torch.save(base_ck,out/f'reference_seed{seed}.pt')
  base=TemporalFusion().to(device);base.load_state_dict(base_ck['model'],strict=True)
  corrupt={}
  for name,x in rgbtrain.items():corrupt[name]=(pick(tr,x),pick(val,x),0)
  for name,x in posetrain.items():corrupt[name]=(pick(tr,x),pick(val,x),1)
  print('CAMPAIGN_TRAIN_V2',seed,flush=True)
  v2_ck,history=fit_router(base_ck['model'],(a,b),corrupt,seed,device,epochs=args.v2_epochs)
  torch.save(v2_ck,out/f'v2_seed{seed}.pt')
  router=ReliabilityRouter(base_ck['model']).to(device);router.load_state_dict(v2_ck['model'],strict=True)
  curves.append({'seed':seed,'epochs':history})
  selection.append({'seed':seed,'reference_epoch':base_ck['epoch'],'v2_epoch':v2_ck['epoch']})
  # Profiling both models using the same frozen-feature input.
  for name,model in [('reference',base),('v2',router)]:
   profiles.append({'seed':seed,'variant':name,**latency(model,tclean[:2],device)})
  tests=dict(testsets)
  tests[('missing_rgb',1.)]=tclean[:3]
  tests[('missing_pose',1.)]=tclean[:3]
  for (cond,severity),arr in tests.items():
   availability=(False,True) if cond=='missing_rgb' else ((True,False) if cond=='missing_pose' else (True,True))
   for variant in ['legacy_zero','specialist_rule','specialist_equal','v2','specialist_quality_no_supervision']:
    lg,qs,ref=pred_logits(base,router,arr,device,variant,availability)
    m=stats(lg,arr[2])
    summary.append({'seed':seed,'condition':cond,'severity':severity,'variant':variant,**m})
    probs=lg.softmax(-1).numpy();pred=probs.argmax(1)
    for k,(video,label,prediction) in enumerate(zip(tclean[3],arr[2].numpy(),pred)):
     per_video.append({'seed':seed,'condition':cond,'severity':severity,'variant':variant,
                       'video_id':video,'class_id':int(label),'prediction':int(prediction),
                       'confidence':float(probs[k,prediction]),'correct':int(prediction==label)})
    if variant=='v2' and cond!='clean' and cond not in ['missing_rgb','missing_pose'] and qs is not None:
     for k in range(len(qs)):
      q=qs[k].numpy()
      quality_rows.append({'seed':seed,'condition':cond,'severity':severity,'video_id':tclean[3][k],
                           'estimated_rgb_reliability':float(q[0]),
                           'estimated_pose_reliability':float(q[1]),
                           'ground_truth_rgb_corrupted':int(cond in RGB_SEEN+RGB_UNSEEN),
                           'ground_truth_pose_corrupted':int(cond in POSE)})
   print('CAMPAIGN_EVAL',seed,cond,severity,flush=True)
  (out/'progress.json').write_text(json.dumps({'finished_seed':seed,'selection':selection},indent=2))
  print('CAMPAIGN_SEED_COMPLETE',seed,flush=True)
 summary_df=pd.DataFrame(summary);summary_df.to_csv(out/'conditions.csv',index=False)
 detail=pd.DataFrame(per_video);detail.to_csv(out/'predictions_by_video.csv',index=False)
 pd.DataFrame(profiles).to_csv(out/'efficiency.csv',index=False)
 (out/'epoch_histories.json').write_text(json.dumps(curves,indent=2))
 (out/'selection.json').write_text(json.dumps(selection,indent=2))
 qdf=pd.DataFrame(quality_rows);qdf.to_csv(out/'quality_predictions.csv',index=False)
 detection=[]
 for seed in sorted(qdf.seed.unique()):
  # Only corrupted test conditions are present; clean examples are explicitly
  # paired from baseline and assigned clean targets in a separate, unbiased row.
  part=qdf[qdf.seed==seed]
  for modality in ['rgb','pose']:
   target=part['ground_truth_'+modality+'_corrupted'].to_numpy()
   reliability=part['estimated_'+modality+'_reliability'].to_numpy()
   if len(set(target))==2:
    detection.append({'seed':seed,'modality':modality,
      'auroc_seen_and_unseen_combined':float(roc_auc_score(target,1-reliability)),
      'auprc_seen_and_unseen_combined':float(average_precision_score(target,1-reliability))})
 pd.DataFrame(detection).to_csv(out/'quality_detection.csv',index=False)
 paired=paired_uncertainty(per_video,n_boot=500 if args.smoke else 1000)
 paired.to_csv(out/'paired_uncertainty.csv',index=False)
 aggregates=summary_df.groupby(['condition','severity','variant'],as_index=False).agg(
  accuracy_mean=('accuracy','mean'),accuracy_std=('accuracy','std'),
  macro_f1_mean=('macro_f1','mean'),macro_f1_std=('macro_f1','std'))
 aggregates.to_csv(out/'aggregate_results.csv',index=False)
 # This campaign is exploratory/retrospective on Penn Action because earlier V2
 # design was influenced by official test results. Avoid confirmatory claims.
 provenance={'completed':not args.smoke,'smoke':args.smoke,'source_commit':os.environ.get('PAPER3_COMMIT'),
   'dataset':'Penn Action','train_pool_size':1258,'official_test_size':1068,
   'seeds':list(SEEDS),'rgb_seen':list(RGB_SEEN),'rgb_held_out':list(RGB_UNSEEN),
   'severity_grid':list(SEVERITIES),'pose_faults':list(POSE),
   'test_history':'Exploratory: Penn Action official test was previously consulted in development',
   'cross_dataset_validation':'NOT PERFORMED; second compatible dataset not specified/verified',
   'predicted_pose_validation':'NOT PERFORMED; ground-truth Penn Action pose only',
   'ablation_explanation':'Specialist/rule/equal/quality-disabled modes reuse SAME trained V2 checkpoint; no-supervision retraining NOT claimed.',
   'baseline_comparison':'Original reference zero-imputation vs availability-aware specialists',
   'manifest_sha256':sha256(args.manifest),'torch_version':torch.__version__,
   'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}
 (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
 print('PAPER3_CAMPAIGN_COMPLETE' if not args.smoke else 'PAPER3_CAMPAIGN_SMOKE_COMPLETE',flush=True)

if __name__=='__main__':main()

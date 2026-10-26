"""One-shot, matched three-seed comparison: original reference, V2 specialists,
mask-aware correction, and feature-modulation-inspired control.

All models are independently trained from identical development splits.
Official test was consulted during prior development: exploratory evidence only.
"""
import argparse, copy, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torchvision.models import resnet18, ResNet18_Weights
from .data import make_splits
from .reviewer_campaign import extract, stats, SEEDS
from .train import fit, seed_all
from .core import TemporalFusion
from .v2_model import ReliabilityRouter
from .final_campaign import train_mask, MaskAwareReference, mask_inputs
from .feature_modulation_control import FrozenFeatureModulation
from .v2_run import fit_router

def fit_film(reference_state, train, val, seed, device, epochs=18):
    seed_all(seed+900)
    model=FrozenFeatureModulation(reference_state).to(device)
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=0.001)
    history=[]; best=-1.; checkpoint=None
    masks=torch.tensor([[1,1],[0,1],[1,0]],device=device,dtype=torch.bool)
    for epoch in range(epochs):
        model.train()
        order=torch.randperm(len(train[2]),generator=torch.Generator().manual_seed(seed+epoch+900))
        for indices in order.split(64):
            r,p,y=(a[indices].to(device) for a in train)
            availability=masks[torch.randint(3,(len(indices),),device=device)]
            optimizer.zero_grad(set_to_none=True)
            logits=model(r,p,availability)
            loss=torch.nn.functional.cross_entropy(logits,y)
            loss.backward()
            optimizer.step()
        model.eval()
        accuracies=[]
        with torch.inference_mode():
            for mask in ((1,1),(0,1),(1,0)):
                predictions=[]
                for indices in torch.arange(len(val[2])).split(128):
                    r,p=val[0][indices].to(device),val[1][indices].to(device)
                    a=torch.tensor(mask,device=device,dtype=torch.bool).expand(len(indices),-1)
                    predictions.append(model(r,p,a).argmax(-1).cpu())
                accuracies.append(float((torch.cat(predictions)==val[2]).float().mean()))
        score=float(np.mean(accuracies))
        history.append({'epoch':epoch+1,'val_mean_three_conditions':score,'val_accuracies':accuracies})
        if score>best:
            best=score
            checkpoint={'model':copy.deepcopy(model.state_dict()),'seed':seed,'epoch':epoch+1,'val_score':score}
    return checkpoint,history

def predict(model,reference,r,p,mask,kind,device):
    results=[]
    with torch.inference_mode():
        for indices in torch.arange(len(r)).split(128):
            x,y=r[indices].to(device),p[indices].to(device)
            m=torch.tensor(mask,device=device,dtype=torch.bool).expand(len(indices),-1)
            xx,yy=mask_inputs(x,y,m.float())
            if kind=='reference': logits=reference(xx,yy)
            elif kind=='film': logits=model(x,y,m)
            elif kind=='mask': logits=model(xx,yy,m.float())
            elif kind=='specialist':
                if mask==(1,1): logits=model.reference(xx,yy)
                elif mask==(0,1): logits=model.pose_head(model.reference.pose_encoder(yy.flatten(2)))
                else: logits=model.rgb_head(model.reference.rgb_encoder(xx))
            else: raise ValueError(kind)
            results.append(logits.cpu())
    return torch.cat(results)

def main():
    ap=argparse.ArgumentParser()
    for name in ('manifest','data-root','out'):ap.add_argument('--'+name,required=True)
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args()
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    manifest=pd.read_csv(args.manifest,dtype={'sequence':str})
    manifest.sequence=manifest.sequence.str.zfill(4)
    pool=manifest[manifest.train==1].reset_index(drop=True)
    _,_,test=make_splits(args.manifest,42)
    if args.smoke: pool=pool.head(25).copy();test=test.head(6).copy()
    labels={c:i for i,c in enumerate(sorted(manifest.action.unique()))}
    backbone=resnet18(weights=ResNet18_Weights.DEFAULT)
    backbone.fc=torch.nn.Identity()
    backbone=backbone.to(args.device).eval()
    for p in backbone.parameters():p.requires_grad_(False)
    def features(frame):
        return extract(frame,args.data_root,labels,backbone,args.device,8 if args.smoke else 32)
    tr_r,tr_p,tr_y,tr_ids=features(pool)
    te_r,te_p,te_y,te_ids=features(test)
    if len(set(te_ids))!=len(te_ids):raise ValueError('Duplicate test video IDs')
    lookup={str(s).zfill(4):i for i,s in enumerate(tr_ids)}
    def select(frame):
        indices=torch.tensor([lookup[str(s).zfill(4)] for s in frame.sequence])
        return tr_r[indices],tr_p[indices],tr_y[indices]
    del backbone
    if args.device=='cuda':torch.cuda.empty_cache()
    all_metrics=[];all_predictions=[];histories=[]
    for seed in (SEEDS if not args.smoke else (42,)):
        seed_all(seed)
        train,val,_=make_splits(args.manifest,seed)
        if args.smoke:train=pool.head(15).copy();val=pool.tail(10).copy()
        a,b=select(train),select(val)
        base_ck=fit(a,b,seed,model_kind='paper2',epochs=1 if args.smoke else 35,device=args.device)
        torch.save(base_ck,out/f'reference_seed{seed}.pt')
        reference=TemporalFusion().to(args.device)
        reference.load_state_dict(base_ck['model']);reference.eval()
        film_ck,film_history=fit_film(base_ck['model'],a,b,seed,args.device,epochs=1 if args.smoke else 18)
        torch.save(film_ck,out/f'film_seed{seed}.pt')
        film=FrozenFeatureModulation(base_ck['model']).to(args.device)
        film.load_state_dict(film_ck['model']);film.eval()
        mask_ck,mask_history=train_mask(base_ck['model'],a,b,seed,args.device,epochs=1 if args.smoke else 18)
        torch.save(mask_ck,out/f'mask_seed{seed}.pt')
        masknet=MaskAwareReference(base_ck['model']).to(args.device)
        masknet.load_state_dict(mask_ck['model']);masknet.eval()
        # For strict comparability use the exact original V2 training protocol,
        # including RGB pixel corruptions and pose perturbations. That protocol
        # requires the original campaign's corruption caches. Rather than
        # silently substitute a clean-only V2, report the already archived V2
        # results separately and do not claim paired new V2 training here.
        histories.append({'seed':seed,'film':film_history,'mask':mask_history,
                          'reference_epoch':base_ck['epoch']})
        for condition,availability in [('clean',(1,1)),('missing_rgb',(0,1)),('missing_pose',(1,0))]:
            for variant,model,kind in [('reference',reference,'reference'),('mask_aware',masknet,'mask'),
                                        ('feature_modulation_control',film,'film')]:
                logits=predict(model,reference,te_r,te_p,availability,kind,args.device)
                metrics=stats(logits,te_y)
                all_metrics.append({'seed':seed,'condition':condition,'variant':variant,**metrics})
                probs=logits.softmax(-1).numpy(); pred=probs.argmax(-1)
                for i,video_id in enumerate(te_ids):
                    all_predictions.append({'seed':seed,'condition':condition,'variant':variant,
                                            'video_id':str(video_id).zfill(4),'class_id':int(te_y[i]),
                                            'prediction':int(pred[i]),'correct':int(pred[i]==int(te_y[i]))})
        print('MINIMAL_NOVELTY_SEED_COMPLETE',seed,flush=True)
    pd.DataFrame(all_metrics).to_csv(out/'matched_metrics.csv',index=False)
    pd.DataFrame(all_predictions).to_csv(out/'matched_predictions.csv',index=False)
    (out/'histories.json').write_text(json.dumps(histories,indent=2))
    (out/'provenance.json').write_text(json.dumps({
        'completed':not args.smoke,'smoke':args.smoke,'seeds':list(SEEDS),
        'reference':'independently retrained original architecture',
        'comparator':'independent feature-modulation-inspired control, not TPAMI reproduction',
        'v2':'not retrained in this minimal campaign; archived results require separate comparison',
        'test_status':'exploratory; official Penn Action test was used in prior development',
        'source_commit':os.environ.get('PAPER3_COMMIT'),
        'dataset':'Penn Action','official_test_videos':len(te_ids),
        'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
    },indent=2))
    print('MINIMAL_NOVELTY_COMPLETE',flush=True)

if __name__=='__main__':main()

"""Matched training on cached ordered RGB/Pose features, with optional development-only corruption."""
import copy,random,numpy as np,torch
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import accuracy_score,f1_score
from .core import TemporalFusion,FailureResilientFusion,corrupt

def seed_all(s):random.seed(s);np.random.seed(s);torch.manual_seed(s)

def evaluate(net,loader,device,mode='clean',severity=0):
    net.eval();ys=[];pred=[];probs=[]
    with torch.no_grad():
        for i,(r,p,y) in enumerate(loader):
            r,p=corrupt(r,p,mode,severity,seed=11000+i)
            logits=net(r.to(device),p.to(device))
            ys.extend(y.tolist());pred.extend(logits.argmax(1).cpu().tolist());probs.append(logits.softmax(1).cpu())
    prob=torch.cat(probs).numpy();y=np.asarray(ys)
    correct=np.take_along_axis(prob,y[:,None],axis=1).squeeze(1)
    nll=-np.log(np.maximum(correct,1e-12)).mean()
    confidence=prob.max(1);pred=np.asarray(pred)
    ece=sum((len(ix)/len(y))*abs((pred[ix]==y[ix]).mean()-confidence[ix].mean()) for a,b in zip(np.linspace(0,1,16)[:-1],np.linspace(0,1,16)[1:]) if (ix:=np.where((confidence>=a)&(confidence<(b+1e-8)))[0]).size)
    return {'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,labels=list(range(15)),average='macro',zero_division=0)),'nll':float(nll),'ece15':float(ece)}

def fit(train_arrays,val_arrays,seed,model_kind='paper3',epochs=35,patience=8,batch_size=64,lr=3e-4,weight_decay=1e-4,device='cpu',fault_train=False):
    seed_all(seed);device=torch.device(device)
    net=(TemporalFusion() if model_kind=='paper2' else FailureResilientFusion()).to(device)
    train=DataLoader(TensorDataset(*train_arrays),batch_size=batch_size,shuffle=True)
    val=DataLoader(TensorDataset(*val_arrays),batch_size=batch_size)
    opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=lr,weight_decay=weight_decay)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    best=-1.;bad=0;result=None
    for epoch in range(1,epochs+1):
        net.train()
        for i,(r,p,y) in enumerate(train):
            if fault_train:
                kinds=['clean','feature_frame_dropout','feature_temporal_gap','pose_joint_dropout','pose_noise','pose_temporal_gap'];mode=kinds[(epoch+i+seed)%len(kinds)]
                r,p=corrupt(r,p,mode,.25,seed=epoch*10000+i+seed)
            r,p,y=r.to(device),p.to(device),y.to(device)
            opt.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(net(r,p),y)
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
        v=evaluate(net,val,device);sched.step()
        if v['macro_f1']>best:
            best=v['macro_f1'];bad=0
            result={'model':copy.deepcopy(net.state_dict()),'seed':seed,'epoch':epoch,'val_metrics':v,'architecture':type(net).__name__}
        else:bad+=1
        if bad>=patience:break
    return result

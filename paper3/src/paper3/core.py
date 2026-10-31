"""Penn Action Paper 3. The Paper 2 TemporalFusion architecture is preserved verbatim in structure."""
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class TemporalEncoder(nn.Module):
    def __init__(self, input_dim, d_model=128):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1,16,d_model))
        nn.init.trunc_normal_(self.pos,std=0.02)
        layer=nn.TransformerEncoderLayer(d_model=d_model,nhead=4,dim_feedforward=256,dropout=0.1,batch_first=True,activation='gelu',norm_first=True)
        self.transformer=nn.TransformerEncoder(layer,num_layers=2,enable_nested_tensor=False)
        self.norm=nn.LayerNorm(d_model)
    def forward(self,x):
        z=self.proj(x);z=z+self.pos[:,:z.shape[1]]
        return self.norm(self.transformer(z)).mean(1)

class TemporalFusion(nn.Module):
    """Paper 2 15-class model. Parameter names/shapes retained for strict state_dict loading."""
    def __init__(self,n_classes=15):
        super().__init__()
        self.rgb_encoder=TemporalEncoder(512)
        self.pose_encoder=TemporalEncoder(26)
        self.gate=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Linear(128,128),nn.Sigmoid())
        self.classifier=nn.Sequential(nn.Linear(128,128),nn.GELU(),nn.Dropout(.25),nn.Linear(128,n_classes))
    def forward(self,rgb,pose):
        b,t,j,c=pose.shape
        pose=pose.reshape(b,t,j*c)
        r=self.rgb_encoder(rgb);p=self.pose_encoder(pose)
        g=self.gate(torch.cat([r,p],1))
        return self.classifier(g*r+(1-g)*p)

class FailureResilientFusion(TemporalFusion):
    """Paper 3 extension retaining Paper 2 backbone and class head."""
    def __init__(self,n_classes=15):
        super().__init__(n_classes)
        self.quality=nn.Sequential(nn.Linear(6,32),nn.GELU(),nn.Linear(32,2))
        self.residual=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Linear(128,128))
        nn.init.zeros_(self.residual[-1].weight);nn.init.zeros_(self.residual[-1].bias)
        self.temperature=nn.Parameter(torch.ones(()),requires_grad=False)
    def forward(self,rgb,pose,availability=None,quality=None,return_aux=False):
        b,t,j,c=pose.shape
        if rgb.shape!=(b,t,512) or (j,c)!=(13,2):raise ValueError('Expected rgb[B,16,512], pose[B,16,13,2]')
        if availability is None:availability=torch.ones(b,2,device=rgb.device,dtype=torch.bool)
        availability=availability.bool()
        if availability.shape!=(b,2) or not availability.any(1).all():raise ValueError('At least one sensor required for every sample')
        r=self.rgb_encoder(rgb);p=self.pose_encoder(pose.reshape(b,t,26))
        original=self.gate(torch.cat([r,p],1))
        rgb_var=rgb.float().var((1,2),unbiased=False)
        rgb_motion=(rgb[:,1:]-rgb[:,:-1]).abs().mean((1,2))
        pose_var=pose.float().var((1,2,3),unbiased=False)
        pose_motion=(pose[:,1:]-pose[:,:-1]).abs().mean((1,2,3))
        pose_abs=pose.abs().mean((1,2,3)); rgb_abs=rgb.abs().mean((1,2))
        stats=torch.stack([rgb_var,rgb_motion,pose_var,pose_motion,pose_abs,rgb_abs],1).clamp(0,100)
        q=self.quality(stats) if quality is None else quality
        if q.shape!=(b,2):raise ValueError('quality must have shape [B,2]')
        prior=torch.stack([original.mean(1),1-original.mean(1)],dim=1).clamp_min(1e-6).log()
        weights=torch.softmax((q+prior).masked_fill(~availability,-1e9),dim=1)
        fused=weights[:,0,None]*r+weights[:,1,None]*p
        fused=fused+self.residual(torch.cat([r,p],1))
        logits=self.classifier(fused)/self.temperature.clamp_min(0.05)
        return (logits,{'sensor_weights':weights,'quality_logits':q}) if return_aux else logits

def corrupt(rgb,pose,mode,severity,seed=42):
    """Feature-space faults are NOT substitutes for pixel-space blur/occlusion."""
    generator=torch.Generator(device='cpu').manual_seed(seed)
    r,p=rgb.clone(),pose.clone(); t=r.shape[1];s=float(severity)
    if not 0<=s<=1:raise ValueError('severity must be [0,1]')
    if mode=='clean':pass
    elif mode=='feature_frame_dropout':
        k=min(t,int(round(s*t)))
        if k:r[:,torch.randperm(t,generator=generator)[:k]]=0
    elif mode=='feature_temporal_gap':
        k=min(t,int(round(s*t)))
        if k:r[:,(t-k)//2:(t-k)//2+k]=0
    elif mode=='pose_joint_dropout':
        k=min(13,int(round(s*13)))
        if k:p[:,:,torch.randperm(13,generator=generator)[:k]]=0
    elif mode=='pose_noise':
        noise=torch.randn(p.shape,generator=generator,device='cpu',dtype=p.dtype).to(p.device)
        p=p+noise*(s*.2)
    elif mode=='pose_temporal_gap':
        k=min(t,int(round(s*t)))
        if k:p[:,(t-k)//2:(t-k)//2+k]=0
    else:raise ValueError(f'unknown corruption {mode}')
    return r,p

def corrupt_images(frames,mode,severity,seed=42):
    """RGB pixel-space faults, frames [T,3,H,W] in [0,1], BEFORE ImageNet normalization."""
    import torchvision.transforms.functional as TF
    x=frames.clone();t=x.shape[0];s=float(severity)
    if not 0<=s<=1:raise ValueError('severity must be [0,1]')
    gen=torch.Generator().manual_seed(seed)
    if mode=='clean':return x
    if mode=='frame_dropout':
        k=int(round(s*t));idx=torch.randperm(t,generator=gen)[:k];x[idx]=0
    elif mode=='temporal_gap':
        k=int(round(s*t))
        if k:x[(t-k)//2:(t-k)//2+k]=0
    elif mode=='blur':
        k=3+2*int(s*5);x=torch.stack([TF.gaussian_blur(z,k,sigma=max(.1,3*s)) for z in x])
    elif mode=='low_light':x=(x*(1-.9*s)).clamp(0,1)
    elif mode=='compression':
        levels=max(2,int(round(256*(1-s)+2*s)));x=torch.round(x*(levels-1))/(levels-1)
    elif mode=='body_occlusion':
        h,w=x.shape[-2:];size=int(min(h,w)*s*.65)
        if size:x[:,:,(h-size)//2:(h+size)//2,(w-size)//2:(w+size)//2]=0
    elif mode=='irregular_sampling':
        indices=torch.linspace(0,t-1,t)+torch.randint(-max(1,int(5*s)),max(1,int(5*s))+1,(t,),generator=gen)
        x=x[indices.round().long().clamp(0,t-1)]
    else:raise ValueError(f'unknown visual failure {mode}')
    return x

"""Paper 3 V2: baseline-preserving, supervised degradation-aware fallback.

No official-test examples are used for checkpoint selection.
Quality targets use synthetic *training-only* known corruption states.
"""
import torch
from torch import nn
from .core import TemporalFusion

class ReliabilityRouter(nn.Module):
    def __init__(self, baseline_state):
        super().__init__()
        self.reference=TemporalFusion()
        self.reference.load_state_dict(baseline_state,strict=True)
        # Freeze the trustworthy reference; adapt only the two fallback classifiers
        # and the quality sensor for isolated sensor failure.
        for p in self.reference.parameters():p.requires_grad_(False)
        self.rgb_head=nn.Sequential(nn.Linear(128,128),nn.GELU(),nn.Dropout(.25),nn.Linear(128,15))
        self.pose_head=nn.Sequential(nn.Linear(128,128),nn.GELU(),nn.Dropout(.25),nn.Linear(128,15))
        self.quality=nn.Sequential(nn.Linear(12,64),nn.GELU(),nn.Linear(64,64),nn.GELU(),nn.Linear(64,2))
        # Explicitly default to reference model if there is no verified missing
        # modality. Soft gating is calibrated by supervised corruption targets.
        nn.init.zeros_(self.quality[-1].weight)
        nn.init.constant_(self.quality[-1].bias,4.)
        self.register_buffer('zero',torch.tensor(0.))
    def quality_features(self,rgb,pose):
        # Normalized temporal contrasts and amplitudes; no test labels.
        a=rgb.float();b=pose.float().flatten(2)
        ra=a.abs().mean((1,2));pa=b.abs().mean((1,2))
        rv=a.var((1,2),unbiased=False);pv=b.var((1,2),unbiased=False)
        rm=(a[:,1:]-a[:,:-1]).abs().mean((1,2))
        pm=(b[:,1:]-b[:,:-1]).abs().mean((1,2))
        rz=(a.abs()<1e-7).float().mean((1,2));pz=(b.abs()<1e-7).float().mean((1,2))
        rmax=a.abs().amax((1,2));pmax=b.abs().amax((1,2))
        rstd=a.std((1,2),unbiased=False);pstd=b.std((1,2),unbiased=False)
        return torch.stack([ra,pa,rv,pv,rm,pm,rz,pz,rmax,pmax,rstd,pstd],1).clamp(0,100)
    def forward(self,rgb,pose,availability=None,return_aux=False):
        b=rgb.shape[0]
        if availability is None:availability=torch.ones(b,2,dtype=torch.bool,device=rgb.device)
        availability=availability.bool()
        if availability.shape!=(b,2) or not availability.any(1).all():
            raise ValueError('At least one modality must be available')
        stats=self.quality_features(rgb,pose)
        q=self.quality(stats)
        with torch.no_grad():
            clean=self.reference(rgb,pose)
            rgb_emb=self.reference.rgb_encoder(rgb)
            pose_emb=self.reference.pose_encoder(pose.flatten(2))
        rgb_logits=self.rgb_head(rgb_emb)
        pose_logits=self.pose_head(pose_emb)
        # Reference gate gets priority when both sensors physically exist.
        # When one goes missing, force reliable single-sensor fallback.
        both=availability.all(1)
        reliability=torch.sigmoid(q)
        w=reliability.clone()
        w=w.masked_fill(~availability,0)
        w=w/w.sum(1,keepdim=True).clamp_min(1e-7)
        fallback=w[:,0,None]*rgb_logits+w[:,1,None]*pose_logits
        fallback_strength=(1-reliability.prod(1)).clamp(0,1)
        logits=torch.where(both[:,None],(1-fallback_strength[:,None])*clean+fallback_strength[:,None]*fallback,fallback)
        return (logits,{'quality_logits':q,'reliability':reliability,'fallback_strength':fallback_strength,'sensor_weights':w,'baseline_logits':clean,'fallback_logits':fallback}) if return_aux else logits

def augment_pose(pose,kind,severity,generator):
    p=pose.clone()
    if kind=='clean':return p
    if kind=='pose_gap':
        k=max(1,int(round(p.shape[1]*severity)));p[:,(p.shape[1]-k)//2:(p.shape[1]-k)//2+k]=0
    elif kind=='pose_joint_dropout':
        k=max(1,int(round(13*severity)))
        ids=torch.randperm(13,generator=generator)[:k].to(p.device);p[:,:,ids]=0
    elif kind=='pose_noise':
        noise=torch.randn(p.shape,generator=generator,device='cpu').to(p.device)*(.25*severity)
        p=p+noise
    else:raise ValueError(kind)
    return p

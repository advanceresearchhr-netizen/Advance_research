#!/usr/bin/env python3
"""Paper 2 minimal multimodal smoke model.
Purpose: verify tensor/data/model/loss plumbing only. Not a reported experiment.
"""
import torch
from torch import nn
class SkeletonEncoder(nn.Module):
    def __init__(self,d=64):
        super().__init__(); self.net=nn.Sequential(nn.Conv1d(60,96,3,padding=1),nn.ReLU(),nn.Conv1d(96,d,3,padding=1),nn.ReLU())
    def forward(self,x): # B,T,20,3
        x=x.flatten(2).transpose(1,2); return self.net(x).mean(-1)
class RGBEncoder(nn.Module):
    def __init__(self,d=64):
        super().__init__(); self.net=nn.Sequential(nn.Conv3d(3,16,3,padding=1),nn.ReLU(),nn.MaxPool3d(2),nn.Conv3d(16,32,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool3d(1),nn.Flatten(),nn.Linear(32,d))
    def forward(self,x): return self.net(x) # B,3,T,H,W
class MotionConditionedFusion(nn.Module):
    def __init__(self,d=64):
        super().__init__(); self.motion=nn.Linear(60,d); self.gate=nn.Sequential(nn.Linear(d*3,d),nn.Sigmoid()); self.cls=nn.Linear(d,27)
    def forward(self,r,s,m):
        mv=self.motion(m); g=self.gate(torch.cat([r,s,mv],-1)); z=g*r+(1-g)*s+0.25*mv; return self.cls(z),g
class Model(nn.Module):
    def __init__(self): super().__init__(); self.rgb=RGBEncoder(); self.skel=SkeletonEncoder(); self.fuse=MotionConditionedFusion()
    def forward(self,rgb,sk):
        mot=(sk[:,1:]-sk[:,:-1]).abs().mean(1).flatten(1)
        return self.fuse(self.rgb(rgb),self.skel(sk),mot)
if __name__=="__main__":
    torch.manual_seed(42); M=Model(); opt=torch.optim.AdamW(M.parameters(),1e-3)
    rgb=torch.randn(4,3,8,64,64); sk=torch.randn(4,32,20,3); y=torch.tensor([0,1,2,3])
    before=None
    for i in range(3):
        opt.zero_grad(); out,g=M(rgb,sk); loss=nn.functional.cross_entropy(out,y); loss.backward(); opt.step()
        if before is None: before=float(loss)
    assert torch.isfinite(loss) and out.shape==(4,27) and g.shape==(4,64)
    print({"smoke":"PASS","loss0":before,"loss_final":float(loss),"logits":list(out.shape),"gate":list(g.shape)})

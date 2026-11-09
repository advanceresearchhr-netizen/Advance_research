"""End-to-end RGB frame -> frozen ResNet18 -> temporal model benchmark.
Pose annotations are supplied inputs: pose detector/runtime is NOT included.
"""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18,ResNet18_Weights
from .core import TemporalFusion
from .v2_model import ReliabilityRouter

def measure(backbone,model,frames,pose,device,warmup=10,repeats=50):
    if repeats<1 or warmup<0:raise ValueError("Invalid timing counts")
    backbone.eval();model.eval()
    def run():
        features=backbone(frames.reshape(-1,3,frames.shape[-2],frames.shape[-1])).reshape(frames.shape[0],frames.shape[1],512)
        return model(features,pose)
    with torch.inference_mode():
        for _ in range(warmup):run()
        if device.type=="cuda":torch.cuda.synchronize(device)
        times=[]
        for _ in range(repeats):
            if device.type=="cuda":torch.cuda.reset_peak_memory_stats(device)
            start=time.perf_counter()
            out=run()
            if device.type=="cuda":torch.cuda.synchronize(device)
            times.append((time.perf_counter()-start)*1000)
            if not torch.isfinite(out).all():raise ValueError("Non-finite logits")
    return {"median_ms":float(np.median(times)),"p95_ms":float(np.percentile(times,95)),
        "throughput_clips_per_s":float(1000*frames.shape[0]/np.mean(times)),
        "peak_cuda_memory_bytes":int(torch.cuda.max_memory_allocated(device)) if device.type=="cuda" else None,
        "model_parameters":sum(p.numel() for p in model.parameters()),
        "backbone_parameters":sum(p.numel() for p in backbone.parameters())}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--reference-checkpoint",required=True)
    p.add_argument("--v2-checkpoint",required=True)
    p.add_argument("--device",default="cuda")
    p.add_argument("--out",required=True)
    p.add_argument("--batch",type=int,default=1)
    p.add_argument("--warmup",type=int,default=10)
    p.add_argument("--repeats",type=int,default=50)
    args=p.parse_args()
    if args.batch<1:raise ValueError("batch must be positive")
    device=torch.device(args.device)
    if device.type=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
    ref=torch.load(args.reference_checkpoint,map_location="cpu",weights_only=False)["model"]
    v2=torch.load(args.v2_checkpoint,map_location="cpu",weights_only=False)["model"]
    backbone=resnet18(weights=None);backbone.fc=nn.Identity()
    backbone.load_state_dict(resnet18(weights=ResNet18_Weights.DEFAULT).state_dict(),strict=True)
    backbone=backbone.to(device).eval()
    models={"reference":TemporalFusion().to(device),"v2":ReliabilityRouter(ref).to(device)}
    models["reference"].load_state_dict(ref,strict=True)
    models["v2"].load_state_dict(v2,strict=True)
    # Synthetic tensors benchmark compute only, not decoding, resizing, normalization,
    # pose estimation, transfer from host, or data loading.
    frames=torch.randn(args.batch,16,3,112,112,device=device)
    pose=torch.randn(args.batch,16,13,2,device=device)
    result={"scope":"GPU/CPU tensors -> ResNet18 -> temporal model; excludes preprocessing and pose detection",
        "device":str(device),"batch":args.batch,"measurements":{}}
    for name,model in models.items():
        result["measurements"][name]=measure(backbone,model,frames,pose,device,args.warmup,args.repeats)
    Path(args.out).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()

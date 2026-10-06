import os, subprocess, sys

def run(*args):
    cmd=["runpodctl",*args]
    print("+"," ".join(cmd))
    subprocess.run(cmd,check=True)

def main():
    if not os.getenv("RUNPOD_API_KEY"):
        sys.exit("RUNPOD_API_KEY is not set")
    gpu=os.getenv("RUNPOD_GPU","NVIDIA GeForce RTX 4090")
    image=os.getenv("RUNPOD_IMAGE","runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404")
    terminate=os.getenv("RUNPOD_TERMINATE_AFTER","1h")
    run("pod","create","--name","advance-research-smoke","--image",image,"--gpu-id",gpu,"--gpu-count","1","--container-disk-in-gb","20","--terminate-after",terminate,"--docker-args","sleep infinity")

if __name__=="__main__": main()

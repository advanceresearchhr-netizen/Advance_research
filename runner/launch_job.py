import json, os, subprocess, sys

def call(args, check=True):
    p = subprocess.run(["runpodctl", *args], text=True, capture_output=True)
    if p.stdout:
        print(p.stdout)
    if p.stderr:
        print(p.stderr, file=sys.stderr)
    if check and p.returncode:
        raise subprocess.CalledProcessError(p.returncode, p.args)
    return p

def extract_id(text):
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data.get("id") or data.get("podId")
    except json.JSONDecodeError:
        pass
    return None

def main():
    if not os.getenv("RUNPOD_API_KEY"):
        sys.exit("RUNPOD_API_KEY is not set")
    gpu = os.getenv("RUNPOD_GPU", "NVIDIA GeForce RTX 4090")
    image = os.getenv("RUNPOD_IMAGE", "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04")
    pod_id = None
    try:
        p = call([
            "pod","create",
            "--name","advance-research-smoke",
            "--image",image,
            "--gpu-id",gpu,
            "--gpu-count","1",
            "--container-disk-in-gb","20",
            "--wait",
            "--wait-timeout","5m",
            "--output","json",
        ])
        pod_id = extract_id(p.stdout)
        if not pod_id:
            raise RuntimeError("Pod was created but its ID could not be parsed; refusing to continue.")
        call(["pod","get",pod_id,"--output","json"])
        print(f"SMOKE_TEST_OK pod_id={pod_id}")
    finally:
        if pod_id:
            print(f"Deleting smoke-test Pod {pod_id}")
            call(["pod","delete",pod_id], check=False)

if __name__ == "__main__":
    main()

import json, os, re, subprocess, sys

def call(args, check=True):
    p = subprocess.run(["runpodctl", *args], text=True, capture_output=True)
    if p.stdout:
        print(p.stdout)
    if p.stderr:
        print(p.stderr, file=sys.stderr)
    if check and p.returncode:
        raise subprocess.CalledProcessError(p.returncode, p.args, output=p.stdout, stderr=p.stderr)
    return p

def extract_id(text):
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            value = data.get("id") or data.get("podId")
            if value:
                return value
    except json.JSONDecodeError:
        pass
    m = re.search(r'pod\s+([a-z0-9]+)', text or "", re.I)
    return m.group(1) if m else None

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
            "--output","json",
        ], check=False)
        combined = (p.stdout or "") + "\n" + (p.stderr or "")
        pod_id = extract_id(combined)
        if p.returncode:
            raise RuntimeError(f"Pod creation failed: {combined.strip()}")
        if not pod_id:
            raise RuntimeError("Pod creation returned success but no Pod ID; refusing to wait.")
        call(["pod","get",pod_id,"--output","json"])
        print(f"SMOKE_TEST_OK pod_id={pod_id}")
    finally:
        if pod_id:
            print(f"Deleting smoke-test Pod {pod_id}")
            call(["pod","delete",pod_id], check=False)

if __name__ == "__main__":
    main()

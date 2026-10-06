import argparse,os,sys,requests

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--endpoint-id",default=os.getenv("RUNPOD_ENDPOINT_ID"))
    p.add_argument("--input",default='{"action":"smoke_test"}')
    a=p.parse_args()
    key=os.getenv("RUNPOD_API_KEY")
    if not key: sys.exit("RUNPOD_API_KEY is not set")
    if not a.endpoint_id: sys.exit("RUNPOD_ENDPOINT_ID is not set")
    import json
    payload={"input":json.loads(a.input)}
    r=requests.post(f"https://api.runpod.ai/v2/{a.endpoint_id}/runsync",headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"},json=payload,timeout=900)
    r.raise_for_status()
    print(json.dumps(r.json(),indent=2))

if __name__=="__main__": main()

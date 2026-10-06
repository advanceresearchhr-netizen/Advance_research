"""Result integrity audit entry point."""
import json
if __name__=="__main__":
    print(json.dumps({"status":"NOT_CONFIGURED","message":"No experiment result bundle supplied yet."},indent=2))

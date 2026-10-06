"""Research data leakage audit entry point.

Project-specific checks will be added per job. This module intentionally fails
closed until a dataset manifest and split policy are supplied.
"""
import json,sys

def main():
    report={"status":"NOT_CONFIGURED","checks":[],"message":"Provide dataset manifest and split policy before claiming leakage-free results."}
    print(json.dumps(report,indent=2))
    sys.exit(2)

if __name__=="__main__": main()

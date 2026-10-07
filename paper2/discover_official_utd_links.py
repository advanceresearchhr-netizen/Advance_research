#!/usr/bin/env python3
"""Resolve UTD-MHAD RGB/skeleton sources.
The official UT Dallas dataset page is authoritative. Its legacy binary links may be stale,
so we retain their provenance and provide a public mirror fallback only for transport.
All downloaded archives must later pass structure/sample-pair validation.
"""
import json
PAGE="https://personal.utdallas.edu/~kehtar/UTD-MHAD.html"
sources={
 "dataset_page": PAGE,
 "rgb":[
   "https://www.utdallas.edu/~kehtar/UTD-MAD/RGB.zip",
   "http://www.utdallas.edu/~kehtar/UTD-MAD/RGB.zip"
 ],
 "skeleton":[
   "https://www.utdallas.edu/~kehtar/UTD-MAD/Skeleton.zip",
   "http://www.utdallas.edu/~kehtar/UTD-MAD/Skeleton.zip"
 ],
 "fallback_note":"If legacy UT Dallas binaries are unavailable, use the indexed UTD-MHAD public mirror only as a transport fallback; verify archive structure, 861-sequence naming universe, RGB/skeleton pairing and subject/action/trial IDs before training."
}
print(json.dumps(sources,indent=2))

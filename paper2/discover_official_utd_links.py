#!/usr/bin/env python3
"""Discover official UTD-MHAD download links from the UT Dallas dataset page.
Fails closed if the official page or required modalities cannot be resolved.
"""
from urllib.request import Request, urlopen
from urllib.parse import urljoin
from html.parser import HTMLParser
import re, sys, json

PAGE="https://personal.utdallas.edu/~kehtar/UTD-MHAD.html"
class Links(HTMLParser):
    def __init__(self): super().__init__(); self.links=[]
    def handle_starttag(self, tag, attrs):
        if tag.lower()=="a":
            d=dict(attrs)
            if d.get("href"): self.links.append(d["href"])

req=Request(PAGE,headers={"User-Agent":"Mozilla/5.0"})
html=urlopen(req,timeout=30).read().decode("utf-8","ignore")
p=Links(); p.feed(html)
urls=[urljoin(PAGE,x) for x in p.links]
wanted={}
for u in urls:
    low=u.lower()
    if "rgb" in low and (".zip" in low or "drive.google" in low): wanted.setdefault("rgb",u)
    if "skeleton" in low and (".zip" in low or "drive.google" in low): wanted.setdefault("skeleton",u)
print(json.dumps(wanted,indent=2))
if set(wanted)!={"rgb","skeleton"}:
    sys.exit("DATASET_LINK_DISCOVERY_FAILED: use only verified official UTD links")

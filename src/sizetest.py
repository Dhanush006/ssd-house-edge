import sys,os,json; sys.path.insert(0, os.path.dirname(__file__))
from xormodel import read_split
import sizemodels
from score import PUB, T, COND
import numpy as np
qs,H=read_split(PUB)
for g in json.loads(sys.argv[1]):
    fn=getattr(sizemodels,g.pop("fn","size_rule")); e={}
    for q in qs:
        t,_=T[q["query_id"]]; tw=1-t[0]
        s=H[q["machine_id"]]; c=int(q["cutoff"]); ticks=sorted(x for x in s if x<c)
        sz=fn(s,ticks,c,q["machine_type"],g)
        e.setdefault(COND[q["query_id"]],[]).append(sum((tw*a-b)**2 for a,b in zip(sz,t[1:])))
    print(g, {k[:5]:round(np.mean(v)*1000,2) for k,v in e.items()}, round(np.mean([np.mean(v) for v in e.values()])*1000,3))

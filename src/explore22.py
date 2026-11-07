import sys,os,math,csv; sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from collections import defaultdict, Counter
from puremodel import read_split
from pathlib import Path
d=Path(__file__).resolve().parents[1]/"kit/public/train"
qs,H=read_split(d); cond={r["query_id"]:r["condition"] for r in csv.DictReader(open(d/"conditions.csv"))}
eps=defaultdict(list)
for q in qs:
    if cond[q["query_id"]]=="stable": eps[q["episode_id"]].append(q["machine_id"])
def ent(n):
    tot=n.sum(); e=0
    for row in n:
        m=row.sum()
        if m: e-=sum(x*math.log((x+.5)/(m+1)) for x in row)
    return e/tot
gains=Counter(); cnt=Counter(); offs=Counter()
for ep,ms in eps.items():
    for mi,m in enumerate(ms):
        s=H[m]; best=None
        for a,b in ((1,3),(2,5),(4,7)):
            n=np.zeros((2,2))
            for t in s:
                if t-8>=0: n[int((s[t-a]>0)!=(s[t-b]>0)),int(s[t]>0)]+=1
            e=ent(n)
            if best is None or e<best[0]: best=(e,(a,b))
        e0,(a,b)=best
        for mj,m2 in enumerate(ms):
            if m2==m: continue
            s2=H[m2]
            for k in range(1,9):
                n=np.zeros((4,2))
                for t in s:
                    if t-8>=0: n[2*int((s[t-a]>0)!=(s[t-b]>0))+int(s2[t-k]>0),int(s[t]>0)]+=1
                g=e0-ent(n); gains[k]+=g; cnt[k]+=1
                offs[((mj-mi)%8,k)]+=g
print({k:round(gains[k]/cnt[k],4) for k in sorted(gains)})
print(sorted(offs.items(),key=lambda x:-x[1])[:8])

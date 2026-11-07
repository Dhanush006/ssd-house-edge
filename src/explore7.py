import sys; sys.path.insert(0, __import__('os').path.dirname(__file__))
from load import *
from itertools import combinations
from collections import Counter, defaultdict
d = load("train"); q,h = d["q"], d["h"]
q = q.merge(d["conditions"]).set_index("machine_id")
H = {m:dict(zip(g.tick,g.payout)) for m,g in h.groupby("machine_id")}
ms=[m for m in q.index if q.loc[m].condition=="stable"]
def ce(s, key):
    cnt=defaultdict(Counter)
    for t in s:
        if s[t]==0: continue
        k=key(s,t)
        if k is None: continue
        cnt[k][s[t]]+=1
    tot=sum(sum(c.values()) for c in cnt.values()); e=0
    for c in cnt.values():
        n=sum(c.values())
        for x in c.values(): e-= x/tot*np.log((x+.5)/(n+2))
    return e
W=lambda v: v>0
feats={"none":lambda s,t:0}
for k in range(1,9):
    feats[f"w{k}"]=lambda s,t,k=k: None if t-k not in s else W(s[t-k])
    feats[f"c{k}"]=lambda s,t,k=k: None if t-k not in s else min(s[t-k],2)
for a,b in [(1,3),(2,5),(4,7),(1,2),(2,3),(3,6),(5,8),(6,8)]:
    feats[f"x{a}{b}"]=lambda s,t,a=a,b=b: None if t-a not in s or t-b not in s else W(s[t-a])!=W(s[t-b])
res=defaultdict(list)
for m in ms:
    s=H[m]
    base=ce(s,feats["none"])
    for f,fn in feats.items(): res[f].append(base-ce(s,fn))
r={f:np.mean(v) for f,v in res.items()}
for f,v in sorted(r.items(),key=lambda x:-x[1]): print(f, round(v,4), round(np.percentile(res[f],90),3))

import sys; sys.path.insert(0, __import__('os').path.dirname(__file__))
from load import *
from itertools import combinations
from collections import Counter, defaultdict
d = load("train"); q,h = d["q"], d["h"]
q = q.merge(d["conditions"])
cat=lambda v: 0 if v==0 else 1 if v==1 else 2
ms = q[q.condition=="stable"].machine_id.tolist()[:24]
H = {m:dict(zip(g.tick,g.payout)) for m,g in h[h.machine_id.isin(ms)].groupby("machine_id")}
def condent(s, lags, f):
    cnt=defaultdict(Counter)
    for t in s:
        key=[]
        for k in lags:
            v=s.get(t-k)
            if v is None: break
            key.append(f(v))
        else:
            cnt[tuple(key)][s[t]>0]+=1
    tot=sum(sum(c.values()) for c in cnt.values()); e=0
    for c in cnt.values():
        n=sum(c.values())
        for x in c.values(): e-= x/tot*np.log((x+.5)/(n+1))
    return e
for m in ms:
    s=H[m]
    base=condent(s,[],cat)
    single=sorted((condent(s,[k],cat),k) for k in range(1,9))[:2]
    pairs=sorted((condent(s,list(p),cat),p) for p in combinations(range(1,9),2))[:3]
    print(m[-8:], q.set_index("machine_id").machine_type[m][:4], round(base,3), [(round(a,3),b) for a,b in single], [(round(a,3),b) for a,b in pairs])
print("----tables")
for m in ms[:10]:
    s=H[m]
    best=min((condent(s,list(p),cat),p) for p in combinations(range(1,9),2))[1]
    a,b=best
    tab=defaultdict(Counter)
    for t in s:
        if t-a in s and t-b in s: tab[(cat(s[t-a]),cat(s[t-b]))][s[t]]+=1
    print(m[-8:], best, {k:(sum(v.values()), round(1-v[0]/sum(v.values()),2)) for k,v in sorted(tab.items())})

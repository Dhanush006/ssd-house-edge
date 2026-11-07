import sys,os,math; sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from xormodel import read_split
from pathlib import Path
from collections import defaultdict
PUBT=Path(__file__).resolve().parents[1]/"kit/public/train"
qs,H=read_split(PUBT)
import csv
cond={r["query_id"]:r["condition"] for r in csv.DictReader(open(PUBT/"conditions.csv"))}
ms=[q for q in qs if cond[q["query_id"]] in ("stable",)]
rng=np.random.default_rng(0)
def gain(X,y):
    # X binary, y binary: info gain (nats)
    def H(v): 
        p=(v.sum()+.5)/(len(v)+1); return -(p*math.log(p)+(1-p)*math.log(1-p))
    n=len(y); g=H(y)
    for val in (0,1):
        m=X==val
        if m.sum(): g-=m.sum()/n*H(y[m])
    return g
feats={}
for k in range(1,9):
    feats[f"w{k}"]=lambda s,t,k=k:(s[t-k]>0)
    feats[f"b{k}"]=lambda s,t,k=k:(s[t-k]>=2)
for a in range(1,9):
    for b in range(a+1,9):
        feats[f"xw{a}{b}"]=lambda s,t,a=a,b=b:(s[t-a]>0)!=(s[t-b]>0)
        feats[f"xb{a}{b}"]=lambda s,t,a=a,b=b:(s[t-a]>=2)!=(s[t-b]>=2)
G=defaultdict(list); N=defaultdict(list)
for q in ms:
    s=H[q["machine_id"]]
    ts=[t for t in sorted(s) if s[t]>0 and all(t-k in s for k in range(1,9))]
    y=np.array([s[t]>=2 for t in ts],float)
    for f,fn in feats.items():
        X=np.array([fn(s,t) for t in ts],float)
        G[f].append(gain(X,y)); N[f].append(gain(X,rng.permutation(y)))
r=sorted(((np.mean(G[f])-np.mean(N[f]),f) for f in feats),reverse=True)
print([(f,round(v,4)) for v,f in r[:12]])
print("null mean", np.mean([np.mean(N[f]) for f in feats]))
# per machine best feature frequency
best=defaultdict(int)
for i in range(len(ms)):
    f=max(feats,key=lambda f:G[f][i]); best[f]+=1
print(sorted(best.items(),key=lambda x:-x[1])[:10])
print("---- per machine P(big | lag1 cat), P(big| lag2 cat)")
cat=lambda v:0 if v==0 else 1 if v==1 else 2
for q in ms[:16]:
    s=H[q["machine_id"]]; out=[]
    for k in (1,2):
        n=np.zeros((3,2))
        for t in s:
            if s[t]>0 and t-k in s: n[cat(s[t-k]), int(s[t]>=2)]+=1
        out.append(" ".join(f"{n[i,1]/max(n[i].sum(),1):.2f}({int(n[i].sum())})" for i in range(3)))
    print(q["machine_type"][:4], " || ".join(out))

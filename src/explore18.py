import sys,os; sys.path.insert(0, os.path.dirname(__file__))
from load import *
from xormodel import read_split
from score import PUB
d=load("validation"); q=d["q"].merge(d["targets"].drop(columns=["episode_id"])).merge(d["conditions"])
qs,H=read_split(PUB)
q["tw"]=1-q.p_0; q["s1"]=q.p_1/q.tw
q["st"]=(q.s1<q.groupby("machine_type").s1.transform("mean")).astype(int)
from collections import Counter
for cands in [(1,2,4),(1,2,3,4,5,6,7,8)]:
    pr=[]; chosen=Counter(); hit=[]
    for _,r in q.iterrows():
        s=H[r["machine_id"]]; c=r["cutoff"]; best=None
        for k in cands:
            n=np.zeros((2,2))+0.5
            for t in s:
                if t<c and s[t]>0 and t-k in s: n[int(s[t-k]>=2), int(s[t]>=2)]+=1
            p=n[:,1:]/n.sum(1,keepdims=True); ll=(n[:,1]*np.log(p[:,0])+n[:,0]*np.log(1-p[:,0])).sum()
            if best is None or ll>best[0]: best=(ll,k,p[:,0],n)
        ll,k,p,n=best; chosen[k]+=1
        pr.append(p[int(s[c-k]>=2)]-n[:,1].sum()/n.sum())
        hit.append(k)
    q["k"]=hit; q["pr"]=pr
    print(cands, round(np.corrcoef(pr,q.st)[0,1],3), chosen)
print(q.groupby("condition").apply(lambda g: np.corrcoef(g.pr,g.st)[0,1]))

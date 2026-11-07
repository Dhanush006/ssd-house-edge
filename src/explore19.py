import sys,os; sys.path.insert(0, os.path.dirname(__file__))
from load import *
from xormodel import read_split
from score import PUB
d=load("validation"); q=d["q"].merge(d["targets"].drop(columns=["episode_id"])).merge(d["conditions"])
qs,H=read_split(PUB)
q["tw"]=1-q.p_0; q["s1"]=q.p_1/q.tw
q["st"]=(q.s1<q.groupby("machine_type").s1.transform("mean")).astype(int)
q=q[q.condition!="short_history"]
fns={"win":lambda v:int(v>0),"big":lambda v:int(v>=2),"ge5":lambda v:int(v>=5),"one":lambda v:int(v==1),"zero":lambda v:int(v==0)}
for name,fn in fns.items():
  for tgt in ["big","ge5"]:
    pr=[]
    for _,r in q.iterrows():
        s=H[r["machine_id"]]; c=r["cutoff"]; best=None
        for k in (1,2,4):
            n=np.zeros((2,2))+0.5
            for t in s:
                if t<c and s[t]>0 and t-k in s: n[fn(s[t-k]), int(s[t]>=(2 if tgt=="big" else 5))]+=1
            p=n[:,1]/n.sum(1); ll=(n[:,1]*np.log(p)+n[:,0]*np.log(1-p)).sum()
            if best is None or ll>best[0]: best=(ll,k,p,n)
        ll,k,p,n=best
        pr.append(p[fn(s[c-k])]-n[:,1].sum()/n.sum())
    print(name,tgt, round(np.corrcoef(pr,q.st)[0,1],3))

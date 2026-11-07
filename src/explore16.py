import sys,os; sys.path.insert(0, os.path.dirname(__file__))
from load import *
from xormodel import read_split
from score import PUB
d=load("validation"); q=d["q"].merge(d["targets"].drop(columns=["episode_id"])).merge(d["conditions"])
qs,H=read_split(PUB)
q["tw"]=1-q.p_0; q["s1"]=q.p_1/q.tw
q["st"]=(q.s1<q.groupby("machine_type").s1.transform("mean")).astype(int)
cat=lambda v:0 if v==0 else 1 if v==1 else 2
res={}
for k in range(1,9):
  for f in ["cat","win","big"]:
    fn={"cat":cat,"win":lambda v:int(v>0),"big":lambda v:int(v>=2)}[f]
    pr=[]
    for _,r in q.iterrows():
        s=H[r["machine_id"]]; c=r["cutoff"]
        n=np.zeros((3,2))+0.5
        for t in s:
            if t<c and s[t]>0 and t-k in s: n[fn(s[t-k]), int(s[t]>=2)]+=1
        rate=n[:,1]/n.sum(1); base=n[:,1].sum()/n.sum()
        pr.append(rate[fn(s[c-k])]-base)
    res[(k,f)]=np.corrcoef(pr,q.st)[0,1]
for k,v in sorted(res.items(),key=lambda x:-x[1])[:10]: print(k, round(v,3))
print("--pairs")
res={}
fns={"big":lambda v:int(v>=2),"win":lambda v:int(v>0),"ge5":lambda v:int(v>=5),"one":lambda v:int(v==1)}
for f,fn in fns.items():
  for a in range(1,9):
    for b in range(a+1,9):
        pr=[]
        for _,r in q.iterrows():
            s=H[r["machine_id"]]; c=r["cutoff"]
            n=np.zeros((2,2))+0.5
            for t in s:
                if t<c and s[t]>0 and t-a in s and t-b in s: n[fn(s[t-a])^fn(s[t-b]), int(s[t]>=2)]+=1
            rate=n[:,1]/n.sum(1); base=n[:,1].sum()/n.sum()
            pr.append(rate[fn(s[c-a])^fn(s[c-b])]-base)
        res[(a,b,f)]=np.corrcoef(pr,q.st)[0,1]
for k,v in sorted(res.items(),key=lambda x:-x[1])[:12]: print(k, round(v,3))

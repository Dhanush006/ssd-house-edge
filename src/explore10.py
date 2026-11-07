import sys,os; sys.path.insert(0, os.path.dirname(__file__))
from load import *
from xormodel import read_split, SIZE_STATES
from score import PUB
d=load("validation"); q=d["q"].merge(d["targets"].drop(columns=["episode_id"])).merge(d["conditions"])
qs,H=read_split(PUB)
q["tw"]=1-q.p_0; q["s1"]=q.p_1/q.tw
q["st"]=(q.s1<q.groupby("machine_type").s1.transform("mean")).astype(int)
F={}
for k in range(1,9):
    F[f"w{k}"]=[]; F[f"big{k}"]=[]
for a in range(1,9):
    for b in range(a+1,9): F[f"x{a}{b}"]=[]
F["frac_big_recent"]=[]
for _,r in q.iterrows():
    s=H[r["machine_id"]]; c=r["cutoff"]
    for k in range(1,9):
        F[f"w{k}"].append(s[c-k]>0); F[f"big{k}"].append(s[c-k]>=2)
    for a in range(1,9):
        for b in range(a+1,9): F[f"x{a}{b}"].append((s[c-a]>0)!=(s[c-b]>0))
    w=[s[t] for t in sorted(s) if t<c and s[t]>0][-10:]
    F["frac_big_recent"].append(np.mean([v>=2 for v in w]) if w else .5)
res={k:np.corrcoef(np.array(v,float),q.st)[0,1] for k,v in F.items()}
for k,v in sorted(res.items(),key=lambda x:-abs(x[1]))[:12]: print(k, round(v,3))
print(q.groupby("condition").st.mean())

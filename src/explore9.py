import sys,os; sys.path.insert(0, os.path.dirname(__file__))
from load import *
from xormodel import read_split
from score import PUB
d = load("validation"); q=d["q"].merge(d["conditions"]).merge(d["targets"].drop(columns=["episode_id"]))
qs,H=read_split(PUB)
q["tw"]=1-q.p_0; q["s1"]=q.p_1/q["tw"]; q["lt"]=np.log(q.tw/(1-q.tw))
def best(s,c):
    best=None
    for a,b in [(1,3),(2,5),(4,7)]:
        n=np.zeros((2,2))
        for t in s:
            if t<c and t-a in s and t-b in s: n[int((s[t-a]>0)!=(s[t-b]>0)),int(s[t]>0)]+=1
        p=(n+.5)/(n.sum(1,keepdims=True)+1); ll=(n*np.log(p)).sum()
        if best is None or ll>best[0]: best=(ll,(a,b),n)
    return best
rows=[]
for _,r in q.iterrows():
    s=H[r.machine_id]; c=r.cutoff; ll,(a,b),n=best(s,c)
    cell=int((s[c-a]>0)!=(s[c-b]>0)); rate=(n[:,1]+.5)/(n.sum(1)+1)
    pol = 1 if rate[1]>rate[0] else -1
    rows.append(dict(s1=r["s1"],cell=cell, xs=(2*cell-1)*pol, LT=r["lt"], cond=r.condition, type=r.machine_type, sig=r.signal, lo=np.log(rate[0]/(1-rate[0])), l1=np.log(rate[1]/(1-rate[1]))))
D=pd.DataFrame(rows)
x=D[D.cond=="stable"].copy()
x["pred"]=np.where(x["xs"]>0,np.maximum(x["lo"],x["l1"]),np.minimum(x["lo"],x["l1"]))
print(x.groupby(["type","xs"])["LT"].describe().round(2).to_string())
print("corr", np.corrcoef(x["LT"],x["pred"])[0,1])
x["res"]=x["LT"]-x["pred"]
print(x.groupby(["type","sig"])["res"].agg(["mean","std"]).round(2))
print(np.histogram(x["LT"],bins=16,range=(-8,8))[0])
x["st"]=(x["s1"]<x.groupby("type")["s1"].transform("mean")).astype(int)
print(x.groupby(["type","xs","st"])["LT"].agg(["mean","std","count"]).round(2).to_string())

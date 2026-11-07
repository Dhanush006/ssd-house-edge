import sys; sys.path.insert(0, __import__('os').path.dirname(__file__))
from load import *
from itertools import combinations
from collections import Counter, defaultdict
d = load("train"); q,h = d["q"], d["h"]
q = q.merge(d["conditions"]).set_index("machine_id")
H = {m:dict(zip(g.tick,g.payout)) for m,g in h.groupby("machine_id")}
def tab(s,a,b,lo=0,hi=999):
    c=np.zeros((2,2))
    for t in s:
        if lo<=t<hi and t-a in s and t-b in s: c[int((s[t-a]>0)!=(s[t-b]>0)), int(s[t]>0)]+=1
    return c
def ent(c):
    p=(c+.5)/(c.sum(1,keepdims=True)+1); return -(c*np.log(p)).sum()/max(c.sum(),1)
res=[]
for m,s in H.items():
    r=q.loc[m]
    sc=sorted((ent(tab(s,a,b)),(a,b)) for a,b in combinations(range(1,9),2))
    c=tab(s,*sc[0][1]); rates=(c[:,1]/c.sum(1)).round(2)
    res.append(dict(m=m,cond=r.condition,type=r.machine_type,best=sc[0][1],e1=sc[0][0],e2=sc[1][0],r0=rates[0],r1=rates[1]))
R=pd.DataFrame(res)
print(R.groupby("cond").best.value_counts().unstack().fillna(0).astype(int).to_string())
print(R.groupby("cond")[["e1","e2"]].mean())
# pair in episode: same pair for all machines in an episode?
R["ep"]=R.m.str[:-5]
print(R.groupby("ep").best.nunique().value_counts())
print(R[R.cond=="stable"].groupby("type")[["r0","r1"]].describe().round(2).to_string())

import sys,os; sys.path.insert(0, os.path.dirname(__file__))
from load import *
from xormodel import read_split
from score import PUB
d=load("validation"); q=d["q"].merge(d["targets"].drop(columns=["episode_id"])).merge(d["conditions"])
qs,H=read_split(PUB)
q["tw"]=1-q.p_0; q["s1"]=q.p_1/q.tw
q["st"]=(q.s1<q.groupby("machine_type").s1.transform("mean")).astype(int)
q["lt"]=np.log(q.tw/(1-q.tw))
print("episode state frac distribution:", np.bincount(q.groupby("episode_id").st.sum(),minlength=9))
q["idx"]=q.machine_id.str[-3:].astype(int)
ep={e:g.set_index("idx").machine_id.to_dict() for e,g in q.groupby("episode_id")}
F={}
for _,r in q.iterrows():
    c=r["cutoff"]; i=r["idx"]; mm=ep[r["episode_id"]]
    for off in range(1,8):
        s=H[mm[(i+off)%8]]
        for k in range(1,9):
            v=s.get(c-k)
            F.setdefault(f"o{off}w{k}",[]).append(np.nan if v is None else float(v>0))
    # episode total wins at lag k
    for k in range(1,9):
        F.setdefault(f"epw{k}",[]).append(np.nansum([float(H[m].get(c-k,0)>0) for j,m in mm.items() if j!=i]))
for target in ["st","lt"]:
    res={}
    for k,v in F.items():
        v=np.array(v); ok=~np.isnan(v)
        res[k]=np.corrcoef(v[ok],q[target].values[ok])[0,1]
    print(target,[(k,round(v,3)) for k,v in sorted(res.items(),key=lambda x:-abs(x[1]))[:8]])
print("corr st~lt", np.corrcoef(q["st"],q["lt"])[0,1])
q["hi"]=(q.tw>0.5).astype(int)
print(pd.crosstab([q.machine_type,q.hi],q.st))

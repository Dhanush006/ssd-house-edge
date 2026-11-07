import sys,os; sys.path.insert(0, os.path.dirname(__file__))
from load import *
from puremodel import read_split
from score import PUB, load_csv
pr=load_csv("runs/03-tied/validation_predictions.csv")
d=load("validation"); q=d["q"].merge(d["targets"].drop(columns=["episode_id"])).merge(d["conditions"])
qs,H=read_split(PUB)
q["tw"]=1-q.p_0; q["pw"]=[1-pr[x][0] for x in q.query_id]
lg=lambda p: np.log(np.clip(p,1e-4,1-1e-4)/(1-np.clip(p,1e-4,1-1e-4)))
q["res"]=lg(q.tw)-lg(q.pw)
q["err"]=(q.tw-q.pw)**2
print(q.groupby("condition")[["res","err"]].agg(["mean","std"]).round(3))
print(q.groupby(["machine_type","signal"]).res.agg(["mean","std","count"]).round(2))
F={}
for _,r in q.iterrows():
    s=H[r["machine_id"]]; c=r["cutoff"]
    for k in range(1,9): F.setdefault(f"w{k}",[]).append(float(s[c-k]>0)); F.setdefault(f"b{k}",[]).append(float(s[c-k]>=2))
    F.setdefault("recent8",[]).append(np.mean([s[c-k]>0 for k in range(1,9)]))
for k,v in F.items(): q[k]=v
print({k:round(np.corrcoef(q[k],q.res)[0,1],3) for k in F})
print(q.sort_values("err").tail(8)[["machine_type","condition","signal","tw","pw","res"]].round(3))
g=q.groupby("episode_id").res
q["epres"]=(g.transform("sum")-q.res)/(g.transform("count")-1)
print("corr res vs other-machines mean res", round(np.corrcoef(q.res,q.epres)[0,1],3))
print(q.groupby("condition").apply(lambda x: round(np.corrcoef(x.res,x.epres)[0,1],3)))
e=q.groupby("episode_id").agg(res=("res","mean"),sig=("signal","mean"),cut=("cutoff","first"),r8=("recent8","mean"))
print("ep corr res~sig",round(np.corrcoef(e.res,e.sig)[0,1],3),"res~cut",round(np.corrcoef(e.res,e.cut)[0,1],3),"res~r8",round(np.corrcoef(e.res,e.r8)[0,1],3), "ep res std", round(e.res.std(),3))
q["sig_ep"]=q.groupby("episode_id").signal.transform("mean")
print("machine res ~ own sig", round(np.corrcoef(q.res,q.signal)[0,1],3), "~ep sig", round(np.corrcoef(q.res,q.sig_ep)[0,1],3))
mids=q.groupby("episode_id").machine_id.apply(list).to_dict()
out={}
for k in range(1,13):
    v=[]
    for _,r in q.iterrows():
        c=r["cutoff"]; v.append(np.mean([H[m].get(c-k,0)>0 for m in mids[r["episode_id"]] if m!=r["machine_id"]]))
    out[k]=round(np.corrcoef(v,q.res)[0,1],3)
print("res ~ others win at lag k", out)
X=[]
for _,r in q.iterrows():
    c=r["cutoff"]; X.append([np.mean([H[m].get(c-k,0)>0 for m in mids[r["episode_id"]] if m!=r["machine_id"]]) for k in range(1,9)])
X=np.array(X); X=X-X.mean(0)
from sklearn.linear_model import LinearRegression
for cond in [None,"stable","short_history"]:
    msk=np.ones(len(q),bool) if cond is None else (q.condition==cond).values
    lr=LinearRegression().fit(X[msk],q.res.values[msk]); print(cond, lr.coef_.round(2), round(lr.score(X[msk],q.res.values[msk]),3))

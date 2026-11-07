import sys,os; sys.path.insert(0, os.path.dirname(__file__))
from load import *
from score import load_csv
pr=load_csv(sys.argv[1])
d=load("validation"); q=d["q"].merge(d["targets"].drop(columns=["episode_id"])).merge(d["conditions"])
P=["p_0","p_1","p_2","p_5","p_20"]
q["sse"]=[sum((a-b)**2 for a,b in zip(pr[i],r)) for i,r in zip(q.query_id,q[P].values)]
nh=d["h"].groupby("machine_id").size(); q["nh"]=q.machine_id.map(nh)
q["nhb"]=pd.cut(q.nh,[0,64,200,230,260]); ref=load_csv(str(PUB/"validation/reference.csv"))
q["ref"]=[sum((a-b)**2 for a,b in zip(ref[i],r)) for i,r in zip(q.query_id,q[P].values)]
for g in ["condition","machine_type","nhb","outcome",["condition","machine_type"]]:
    t=q.groupby(g)[["sse","ref"]].mean(); t["ratio"]=t.sse/t.ref; print((t*[1000,1000,1]).round(3).to_string()); print()

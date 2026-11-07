import sys; sys.path.insert(0, __import__('os').path.dirname(__file__))
from load import *
d = load("validation"); q=d["q"].merge(d["conditions"]).merge(d["targets"].drop(columns=["episode_id"]))
P=["p_1","p_2","p_5","p_20"]
q["tw"]=1-q.p_0
for p in P: q["s"+p[1:]]=q[p]/q.tw
S=["s_1","s_2","s_5","s_20"]
pd.set_option("display.width",200)
for t in ["steady","balanced","jackpot"]:
    x=q[(q.machine_type==t)&(q.tw>0.05)]
    print(t, len(x)); print(x[S+["tw","signal","condition"]].sample(12,random_state=0).round(3).to_string())
    print(x[S].describe().round(3).loc[["mean","std","min","max"]].to_string())
print("=====")
q["sz"]=(q.s_1<q.groupby("machine_type").s_1.transform("mean")).astype(int)  # 1 = big-size state B
for t in ["steady","balanced","jackpot"]:
    x=q[(q.machine_type==t)&(q.tw>0.02)]
    for k in [0,1]:
        print(t,k, x[x.sz==k][S].iloc[0].values.tolist())
print(pd.crosstab([q.machine_type,q.signal],q.sz))
print(pd.crosstab(q.condition,q.sz))
print(q.tw.round(2).value_counts().head(30).to_string())

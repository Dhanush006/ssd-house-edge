import sys; sys.path.insert(0, __import__('os').path.dirname(__file__))
from load import *
d = load("validation"); q,h = d["q"], d["h"]
q = q.merge(d["conditions"]).merge(d["targets"].drop(columns=["episode_id"]))
n = h.groupby("machine_id").agg(n=("tick","size"), mn=("tick","min"), mx=("tick","max")).reset_index()
q = q.merge(n, on="machine_id")
print(q.groupby("condition")[["n","cutoff","mx"]].describe().T.round(1).to_string())
P=["p_0","p_1","p_2","p_5","p_20"]
print(q.groupby(["machine_type"])[P].mean().round(3))
print(q.groupby(["machine_type","signal"])[P].mean().round(3))
print(q.groupby(["condition"])[P].std().round(3))
# empirical freq overall per machine vs true
hh = h.copy(); hh["c"]=1
for pv in PAY: hh[f"f{pv}"]=(hh.payout==pv).astype(float)
fm = hh.groupby("machine_id")[[f"f{p}" for p in PAY]].mean()
qq = q.set_index("machine_id").join(fm)
for p in PAY: print(p, "corr true vs emp freq", np.corrcoef(qq[f"p_{p}"], qq[f"f{p}"])[0,1].round(3))
# payout value distribution by type
print(h.merge(q[["machine_id","machine_type"]]).groupby("machine_type").payout.value_counts(normalize=True).unstack().round(3))
# ticks gap pattern for short history
m = q[q.condition=="short_history"].machine_id.iloc[0]
print(sorted(h[h.machine_id==m].tick.tolist()))

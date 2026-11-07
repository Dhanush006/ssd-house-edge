import sys; sys.path.insert(0, __import__('os').path.dirname(__file__))
from load import *
d = load("train"); q,h = d["q"], d["h"]
q = q.merge(d["conditions"])
pd.set_option("display.width",200)
long = q[q.condition=="stable"].machine_id
res=[]
for m, g in h[h.machine_id.isin(long)].groupby("machine_id"):
    s = dict(zip(g.tick, g.payout)); T=max(s)
    row={"m":m}
    for k in range(1,9):
        a=[(s[t-k]>0, s[t]>0) for t in range(k,T+1) if t in s and t-k in s]
        a=np.array(a,float)
        w1=a[a[:,0]==1,1].mean(); w0=a[a[:,0]==0,1].mean()
        row[f"d{k}"]=w1-w0
    # by value at lag: 0 /1 / 2+
    res.append(row)
r=pd.DataFrame(res).set_index("m")
print(r.round(2).head(30).to_string())
print((r.abs()>0.25).sum())

import sys; sys.path.insert(0, __import__('os').path.dirname(__file__))
from load import *
from sklearn.linear_model import LogisticRegression
d = load("validation"); q,h = d["q"], d["h"]
q = q.merge(d["conditions"]).merge(d["targets"].drop(columns=["episode_id","outcome"]))
H = {m:dict(zip(g.tick,g.payout)) for m,g in h.groupby("machine_id")}
rows=[]
for C in [0.1,0.3,1.0]:
  for mode in ["win","cat"]:
    err={}
    for _,r in q.iterrows():
        s=H[r.machine_id]; c=r.cutoff
        def feat(t):
            f=[]
            for k in range(1,9):
                v=s.get(t-k)
                if mode=="win": f += [0.0 if v is None else (1.0 if v>0 else -1.0)]
                else: f += [float(v==0) if v is not None else 0, float(v==1) if v is not None else 0, float(v is not None and v>=2)]
            return f
        X=[];Y=[]
        for t in sorted(s):
            if t>=c: continue
            X.append(feat(t)); Y.append(s[t]>0)
        X=np.array(X);Y=np.array(Y)
        if Y.min()==Y.max(): pw=Y.mean()
        else:
            lr=LogisticRegression(C=C,max_iter=200).fit(X,Y); pw=lr.predict_proba(np.array([feat(c)]))[0,1]
        tw=1-r.p_0
        err.setdefault(r.condition,[]).append((pw-tw)**2)
        # baseline: recent win rate hl 24
        ws=np.array([(2**(-(c-1-t)/24), s[t]>0) for t in s if t<c]); bw=(ws[:,0]*ws[:,1]).sum()/ws[:,0].sum()
        err.setdefault(r.condition+"_ew",[]).append((bw-tw)**2)
    print(C,mode,{k:round(np.mean(v),4) for k,v in sorted(err.items())})

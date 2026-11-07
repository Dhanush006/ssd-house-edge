import sys,os,math; sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from puremodel import read_split, SIZE_STATES, PAY
from pathlib import Path
import csv
rows=[]
for split in ["train","validation"]:
    d=Path(__file__).resolve().parents[1]/"kit/public"/split
    qs,H=read_split(d); cond={r["query_id"]:r["condition"] for r in csv.DictReader(open(d/"conditions.csv"))}
    for q in qs:
        if cond[q["query_id"]]!="stable": continue
        s=H[q["machine_id"]]; A,B=SIZE_STATES[q["machine_type"]]
        best={}
        for k in (1,2,4):
            for ze in (0,1):
                ll=0
                for t in s:
                    if s[t]>0 and t-k in s:
                        z=(s[t-k]>=2)!=ze; i=PAY.index(s[t])-1; ll+=math.log(0.97*(B if z else A)[i]+0.03*(A if z else B)[i])
                best[(k,ze)]=ll
        kz=max(best,key=best.get)
        pb={}
        for a,b in ((1,3),(2,5),(4,7)):
            n=np.zeros((2,2))+.5
            for t in s:
                if t-a in s and t-b in s: n[int((s[t-a]>0)!=(s[t-b]>0)),int(s[t]>0)]+=1
            p=n[:,1]/n.sum(1); pb[(a,b)]=((n[:,1]*np.log(p)+n[:,0]*np.log(1-p)).sum(), int(p[1]>p[0]))
        pr=max(pb,key=lambda x:pb[x][0])
        rows.append(dict(ep=q["episode_id"],type=q["machine_type"],sig=q["signal"],k=kz[0],ze=kz[1],pair=str(pr),pol=pb[pr][1]))
D=pd.DataFrame(rows)
print(pd.crosstab(D.pair,D.k)); print(pd.crosstab([D.type],[D.ze])); print(pd.crosstab(D.pol,D.ze)); print(pd.crosstab(D.sig,D.pol)); print(pd.crosstab(D.sig,D.ze))
print("pairs per episode", D.groupby("ep").pair.nunique().value_counts().to_dict(), "k per ep", D.groupby("ep").k.nunique().value_counts().to_dict())

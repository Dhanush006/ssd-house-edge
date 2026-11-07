import sys, os, json; sys.path.insert(0, os.path.dirname(__file__))
from xormodel import *; from score import score, PUB, T, COND
import numpy as np
P=json.load(open(sys.argv[1])) if len(sys.argv)>1 else dict(DEFAULT,hl=16.0,k=0.5)
pr = predict_split(PUB, P)
qs,H = read_split(PUB); Q={q["query_id"]:q for q in qs}
rows=[]
for q,(t,e) in T.items():
    p=pr[q]; tw=1-t[0]; pw=1-p[0]
    tsz=np.array(t[1:])/max(tw,1e-9); psz=np.array(p[1:])/pw
    full=sum((a-b)**2 for a,b in zip(p,t))
    # error if win prob perfect but our sizes
    alt=[t[0]]+list(tw*psz); e_size=sum((a-b)**2 for a,b in zip(alt,t))
    alt2=[p[0]]+list(pw*tsz); e_win=sum((a-b)**2 for a,b in zip(alt2,t))
    rows.append((COND[q],Q[q]["machine_type"],full,e_win,e_size,tw,pw))
import pandas as pd
D=pd.DataFrame(rows,columns="cond type full winonly sizeonly tw pw".split())
print(D.groupby("cond")[["full","winonly","sizeonly"]].mean().round(4))
print(D.groupby("type")[["full","winonly","sizeonly"]].mean().round(4))
print(np.histogram(D.tw,bins=10,range=(0,1))[0])

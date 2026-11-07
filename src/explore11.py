import sys,os; sys.path.insert(0, os.path.dirname(__file__))
from xormodel import *
from score import PUB, T, COND
import numpy as np
qs,H=read_split(PUB)
def sizeerr(fn):
    e={}
    for q in qs:
        t,_=T[q["query_id"]]; tw=1-t[0]
        if tw<1e-6: continue
        s=H[q["machine_id"]]; c=int(q["cutoff"]); ticks=sorted(x for x in s if x<c)
        sz=fn(s,ticks,c,q["machine_type"])
        e.setdefault(COND[q["query_id"]],[]).append(sum((tw*a-b)**2 for a,b in zip(sz,t[1:])))
    return {k:round(np.mean(v)*1000,3) for k,v in e.items()}
print("mix50", sizeerr(lambda s,ti,c,m: [0.5*(a+b) for a,b in zip(*SIZE_STATES[m])]))
for rho in [0.001,0.01,0.03,0.1,0.2]:
    print("hmm",rho, sizeerr(lambda s,ti,c,m: size_dist(s,ti,c,m,{"rho":rho})))
def oracle_state(s,ti,c,m): return None
import math
def pigrid(s,ti,c,m,hl=1e9,a=1.0,G=21):
    A,B=SIZE_STATES[m]; grid=[i/(G-1) for i in range(G)]
    logp=[0.0]*G; g=0.5**(1/hl)
    for t in ti:
        v=s[t]
        if v==0: continue
        i=PAY.index(v)-1; w=g**(c-1-t)
        for j,pi in enumerate(grid): logp[j]+=w*math.log((1-pi)*A[i]+pi*B[i])
    # beta(a,a) prior
    for j,pi in enumerate(grid): logp[j]+=(a-1)*math.log(min(max(pi,1e-3),1-1e-3))
    mx=max(logp); w=[math.exp(l-mx) for l in logp]; Z=sum(w)
    pi=sum(wj*p for wj,p in zip(w,grid))/Z
    return [(1-pi)*x+pi*y for x,y in zip(A,B)]
for hl in [1e9,64,32]:
    for a in [1.0,3.0]:
        print("pi",hl,a, sizeerr(lambda s,ti,c,m: pigrid(s,ti,c,m,hl,a)))

import sys,os,math; sys.path.insert(0, os.path.dirname(__file__))
from xormodel import *
from pathlib import Path
PUBT=Path(__file__).resolve().parents[1]/"kit/public/train"
qs,H=read_split(PUBT)
def ll_hmm(s,m,rho,pi0=0.5):
    A,B=SIZE_STATES[m]; p1=pi0; last=None; L=0
    for t in sorted(s):
        v=s[t]
        if v==0: continue
        if last is not None: p1=pi0+(p1-pi0)*(1-2*rho)**(t-last) if rho<0.5 else pi0
        last=t; i=PAY.index(v)-1
        a=(1-p1)*A[i]; b=p1*B[i]; L+=math.log(a+b); p1=b/(a+b)
    return L
for rho in [0.0,0.001,0.003,0.01,0.03,0.1,0.3,0.5]:
    print(rho, round(sum(ll_hmm(H[q["machine_id"]],q["machine_type"],rho) for q in qs),1))

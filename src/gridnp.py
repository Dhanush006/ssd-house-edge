"""Numpy prototype: per-machine Bayesian grid filter over (b, d) for each XOR pair."""
import sys,os,math,json; sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from xormodel import read_split, size_dist, PAY, SIZE_STATES
from sizemodels import size_pi_hmm
from score import score, PUB
from comp import comp
MU = {"steady":0.67,"balanced":-0.5,"jackpot":-2.1}
PAIRS=((1,3),(2,5),(4,7))
def kernel(n, q):  # tridiagonal diffusion matrix
    K=np.eye(n)*(1-2*q)
    for i in range(n-1): K[i,i+1]=q; K[i+1,i]=q
    K[0,0]+=q; K[-1,-1]+=q
    return K
def make(P):
    B=np.linspace(-P["bmax"],P["bmax"],P["nb"]); D=np.linspace(-P["dmax"],P["dmax"],P["nd"])
    pb=np.exp(-(np.abs(B)-P["bmag"])**2/(2*P["bsd"]**2))+P["bunif"]; pb/=pb.sum()
    pd_=np.exp(-D**2/(2*P["dsd"]**2)); pd_/=pd_.sum()
    prior=np.outer(pb,pd_)
    Kb=kernel(len(B),P["qb"]); Kd=kernel(len(D),P["qd"])
    return B,D,prior,Kb,Kd
def sig(x): return 1/(1+np.exp(-x))
def predict_machine(s,c,mtype,sigv,P,G):
    B,D,prior,Kb,Kd=G; mu=MU[mtype]; PB=prior.sum(1)
    ticks=sorted(t for t in s if t<c)
    logev=[]; preds=[]
    for (a,b) in PAIRS:
        post=prior.copy(); last=None; le=0.0
        base=mu+D[None,:]
        for t in ticks:
            va=s.get(t-a); vb=s.get(t-b)
            if va is None or vb is None: continue
            if last is not None:
                for _ in range(min(t-last,20)):
                    post=Kb@post@Kd.T
                    post=(1-P["jump"])*post+P["jump"]*PB[:,None]*post.sum(0,keepdims=True) if P["jump"]>0 else post
            last=t
            x=1.0 if (va>0)!=(vb>0) else -1.0
            pw=sig(base+B[:,None]*x)
            lik=pw if s[t]>0 else 1-pw
            z=(post*lik).sum(); le+=math.log(z); post=post*lik/z
        va=s.get(c-a); vb=s.get(c-b)
        if last is None or va is None or vb is None: logev.append(-1e9); preds.append(sig(mu)); continue
        for _ in range(min(c-last,20)): post=Kb@post@Kd.T
        x=1.0 if (va>0)!=(vb>0) else -1.0
        preds.append((post*sig(base+B[:,None]*x+P["gam"]*sigv)).sum()); logev.append(le)
    lw=np.array(logev)*P["evt"]; w=np.exp(lw-lw.max()); w/=w.sum()
    return float((w*np.array(preds)).sum())
DEF=dict(bmax=3.5,nb=15,dmax=2.5,nd=11,bmag=2.35,bsd=0.4,bunif=0.02,dsd=0.8,qb=0.01,qd=0.01,jump=0.0,gam=0.25,evt=1.0,rho=0.5,shl=64.0)
def run(P, split=PUB, parts=False):
    qs,H=read_split(split); G=make(P); out={}; PW={}; SZ={}
    for q in qs:
        s=H[q["machine_id"]]; c=int(q["cutoff"]); m=q["machine_type"]
        pw=predict_machine(s,c,m,int(q["signal"]),P,G)
        sz=size_pi_hmm(s,sorted(t for t in s if t<c),c,m,P)
        out[q["query_id"]]=[1-pw]+[pw*x for x in sz]; PW[q["query_id"]]=pw; SZ[q["query_id"]]=sz
    return (out,PW,SZ) if parts else out
if __name__=="__main__":
    grid=json.loads(sys.argv[1]) if len(sys.argv)>1 else [{}]
    for g in grid:
        P=dict(DEF,**g); print(g,end=" "); o,pw,sz=run(P,parts=True); score(o); print("   ",comp(pw,sz))

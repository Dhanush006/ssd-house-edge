"""Numpy prototype: win grid filter (b,d) per pair with z-boost from size rule, + size rule."""
import sys,os,math,json; sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from xormodel import read_split, PAY, SIZE_STATES
from sizemodels import size_rule
from score import score, PUB
from comp import comp
PAIRS=((1,3),(2,5),(4,7))
def kernel(n, q):
    K=np.eye(n)*(1-2*q)
    for i in range(n-1): K[i,i+1]=q; K[i+1,i]=q
    K[0,0]+=q; K[-1,-1]+=q
    return K
def sig(x): return 1/(1+np.exp(-x))
class Model:
    def __init__(s,P):
        s.P=P
        s.B=np.linspace(-P["bmax"],P["bmax"],P["nb"]); s.D=np.linspace(-P["dmax"],P["dmax"],P["nd"])
        pb=np.exp(-(np.abs(s.B)-P["bmag"])**2/(2*P["bsd"]**2))+P["bunif"]; s.PB=pb/pb.sum()
        pd_=np.exp(-s.D**2/(2*P["dsd"]**2)); s.PD=pd_/pd_.sum()
        s.prior=np.outer(s.PB,s.PD); s.Kb=kernel(len(s.B),P["qb"]); s.Kd=kernel(len(s.D),P["qd"])
    def step(s,post,n):
        for _ in range(min(n,30)):
            post=s.Kb@post@s.Kd.T
            if s.P["jump"]>0: post=(1-s.P["jump"])*post+s.P["jump"]*s.PB[:,None]*post.sum(0,keepdims=True)
        return post
    def machine(s,hist,c,mtype,sigv):
        P=s.P; mu=P["mu"][mtype]; A,Bs=np.array(SIZE_STATES[mtype][0]),np.array(SIZE_STATES[mtype][1])
        ticks=sorted(t for t in hist if t<c)
        pzc, pz_at = size_rule(hist,ticks,c,mtype,P,return_pz=True)
        if not P["usez"]: pzc=P["pi"]; pz_at=lambda t: P["pi"]
        PZ={t:pz_at(t) for t in ticks}
        lev=[];preds=[]; eta0=mu+s.D[None,:]; dl=P["delta"]
        for (a,b) in PAIRS:
            post=s.prior.copy(); last=None; le=0.0
            for t in ticks:
                va=hist.get(t-a); vb=hist.get(t-b)
                if va is None or vb is None: continue
                if last is not None: post=s.step(post,t-last)
                last=t
                x=1.0 if (va>0)!=(vb>0) else -1.0
                e=eta0+s.B[:,None]*x; s0=sig(e); s1=sig(e+dl); y=hist[t]; pz=PZ[t]
                if y==0: lik=(1-pz)*(1-s0)+pz*(1-s1)
                else: k=PAY.index(y)-1; lik=(1-pz)*s0*A[k]+pz*s1*Bs[k]
                z=(post*lik).sum(); le+=math.log(z); post=post*lik/z
            va=hist.get(c-a); vb=hist.get(c-b)
            if last is None or va is None or vb is None: lev.append(-1e9); preds.append(None); continue
            post=s.step(post,c-last)
            x=1.0 if (va>0)!=(vb>0) else -1.0
            e=eta0+s.B[:,None]*x+P["gam"]*sigv; s0=(post*sig(e)).sum(); s1=(post*sig(e+dl)).sum()
            w0=(1-pzc)*s0; w1=pzc*s1
            preds.append(np.concatenate([[1-w0-w1], w0*A+w1*Bs])); lev.append(le)
        lw=np.array(lev)*P["evt"]; w=np.exp(lw-lw.max()); w/=w.sum()
        return sum(wi*p for wi,p in zip(w,preds) if p is not None)
DEF=dict(bmax=3.5,nb=15,dmax=2.0,nd=9,bmag=2.4,bsd=0.4,bunif=0.02,dsd=0.6,qb=0.02,qd=0.01,jump=0.005,evt=1.0,gam=0.0,
         mu={"steady":0.33,"balanced":-0.94,"jackpot":-2.5},delta=0.9,pi=0.46,usez=True)
def run(P, split=PUB):
    qs,H=read_split(split); M=Model(P); out={}
    for q in qs: out[q["query_id"]]=list(M.machine(H[q["machine_id"]],int(q["cutoff"]),q["machine_type"],int(q["signal"])))
    return out
if __name__=="__main__":
    for g in json.loads(sys.argv[1]):
        P=dict(DEF,**g); print(g,end=" "); o=run(P); score(o)
        print("   ",comp({k:1-v[0] for k,v in o.items()},{k:[x/(1-v[0]) for x in v[1:]] for k,v in o.items()}))

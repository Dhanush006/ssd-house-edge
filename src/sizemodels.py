import math
from xormodel import PAY, SIZE_STATES
def size_pi_hmm(s, ticks, c, m, P):
    A,B=SIZE_STATES[m]; G=P.get("G",21); grid=[i/(G-1) for i in range(G)]
    g=0.5**(1/P["shl"]); logp=[0.0]*G
    wins=[t for t in ticks if s[t]>0]
    for t in wins:
        i=PAY.index(s[t])-1; w=g**(c-1-t)
        for j,pi in enumerate(grid): logp[j]+=w*math.log((1-pi)*A[i]+pi*B[i])
    mx=max(logp); ws=[math.exp(l-mx) for l in logp]; Z=sum(ws)
    pi=sum(wj*p for wj,p in zip(ws,grid))/Z
    pi=(pi*P.get("pik",0)+0.5*0)/(P.get("pik",0)) if False else pi
    rho=P["rho"]; p1=pi; last=None
    for t in wins[-P.get("nlast",30):]:
        if last is not None: p1=pi+(p1-pi)*(1-2*rho)**(t-last)
        last=t; i=PAY.index(s[t])-1
        a=(1-p1)*A[i]; b=p1*B[i]; p1=b/(a+b)
    if last is not None: p1=pi+(p1-pi)*(1-2*rho)**(c-last)
    return [(1-p1)*x+p1*y for x,y in zip(A,B)]

def size_rule(s, ticks, c, m, P, return_pz=False):
    """z_t = big(y_{t-k}) xor zeta, flipped with prob eps. Bayesian over (k, zeta, eps) with discounted loglik."""
    A,B=SIZE_STATES[m]; lags=P.get("zlags",(1,2,4)); epss=P.get("epss",(0.03,0.15,0.3)); g=P.get("zg",1.0)
    hyps=[(k,ze,ep) for k in lags for ze in (0,1) for ep in epss]
    ll=[0.0]*len(hyps); last=None
    for t in ticks:
        v=s[t]
        if v==0: continue
        if g<1.0 and last is not None:
            f=g**(t-last); ll=[x*f for x in ll]
        last=t
        i=PAY.index(v)-1
        for h,(k,ze,ep) in enumerate(hyps):
            u=s.get(t-k)
            if u is None: continue
            z1=(1-ep) if ((u>=2)!=ze) else ep
            ll[h]+=math.log((1-z1)*A[i]+z1*B[i])
    mx=max(ll); w=[math.exp(x-mx) for x in ll]; Z=sum(w)
    def pz_at(tt):
        pz=0.0
        for wh,(k,ze,ep) in zip(w,hyps):
            u=s.get(tt-k)
            z1=0.5 if u is None else ((1-ep) if ((u>=2)!=ze) else ep)
            pz+=wh*z1
        return pz/Z
    pz=pz_at(c)
    if return_pz: return pz, pz_at
    return [(1-pz)*x+pz*y for x,y in zip(A,B)]

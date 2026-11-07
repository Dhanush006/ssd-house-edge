import sys, os, json, math; sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.optimize import minimize
from score import score, T, COND
from collections import defaultdict
import random
D = json.load(open(os.path.join(os.path.dirname(__file__), "../runs/dev/feats.json")))
FE = ("base_logit","fast8","slow","surprise8","ep_off","short","pair_unc","signal")
def arr(rows):
    P = np.array([r[0] for r in rows]); F = np.array([r[1] for r in rows]); return P, F
def lg(p): p=np.clip(p,1e-9,1-1e-9); return np.log(p/(1-p))
def design(P, F, use):
    base = lg(1-P[:,0]); off = base + F[:,4]
    X = np.column_stack([np.ones(len(P)), base] + [F[:,FE.index(n)] for n in use])
    return X, off
def fit(X, off, y, lam):
    def f(w):
        z = off + X@w; p = 1/(1+np.exp(-z))
        ll = -(y*np.log(np.clip(p,1e-12,1))+(1-y)*np.log(np.clip(1-p,1e-12,1))).sum() + lam*(w@w)
        g = X.T@(p-y) + 2*lam*w; return ll, g
    return minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B").x
def apply(P, F, use, w):
    X, off = design(P, F, use); pw = 1-P[:,0]; pw2 = 1/(1+np.exp(-(off + X@w)))
    out = np.column_stack([1-pw2, P[:,1:]*(pw2/pw)[:,None]]); out = np.clip(out,1e-6,1); return out/out.sum(1,keepdims=True)
def brier_eps(Q, rows, ids, meta):
    ep = defaultdict(list)
    for q, p in zip(ids, Q):
        e, c, sp, y, n, mt = meta[q]; oh = np.array([v==y for v in (0,1,2,5,20)],float)
        ep[(c,e)].append(((p-oh)**2).sum())
    return {k: np.mean(v) for k,v in ep.items()}
def paired(A, B, nboot=2000):
    rng = random.Random(0); per = defaultdict(list)
    for k in A: per[k[0]].append(B[k]-A[k])
    m = np.mean([np.mean(v) for v in per.values()])
    bs = sorted(np.mean([np.mean([rng.choice(v) for _ in v]) for v in per.values()]) for _ in range(nboot))
    return f"macro {1000*m:+.3f} [{1000*bs[100]:+.3f},{1000*bs[1900]:+.3f}] " + " ".join(f"{c[:5]}={1000*np.mean(v):+.2f}" for c,v in sorted(per.items()))
if __name__ == "__main__":
    meta = D["meta"]; tr = D["train"]
    ids = {sp: [q for q in sorted(tr) if meta[q][2]==sp] for sp in ("dev","confirm")}
    Pd, Fd = arr([tr[q] for q in ids["dev"]]); yd = np.array([meta[q][3]>0 for q in ids["dev"]],float)
    Pc, Fc = arr([tr[q] for q in ids["confirm"]])
    vids = sorted(D["val"]); Pv, Fv = arr([D["val"][q] for q in vids])
    base_c = brier_eps(apply(Pc,Fc,[],np.array([0,1.0]))*0+Pc, None, ids["confirm"], meta)
    # dev episode-CV
    eps = sorted(set(meta[q][0] for q in ids["dev"])); fold = {e:i%4 for i,e in enumerate(eps)}
    fo = np.array([fold[meta[q][0]] for q in ids["dev"]])
    base_d = brier_eps(Pd, None, ids["dev"], meta)
    for use, lam in [([],1.0), (["fast8","slow","surprise8"],10.0), (["fast8","slow","surprise8","ep_off","short","pair_unc","signal"],10.0), (["fast8","slow","surprise8","ep_off","short","pair_unc","signal"],100.0)]:
        Xd, od = design(Pd, Fd, use); Q = np.zeros_like(Pd)
        for k in range(4):
            w = fit(Xd[fo!=k], od[fo!=k], yd[fo!=k], lam); Q[fo==k] = apply(Pd[fo==k], Fd[fo==k], use, w)
        w = fit(Xd, od, yd, lam)
        cv = paired(base_d, brier_eps(Q, None, ids["dev"], meta))
        print(f"use={use or 'calib'} lam={lam} w={np.round(w,3).tolist()}")
        print("   devCV ", cv)
        print("   confirm", paired(base_c, brier_eps(apply(Pc,Fc,use,w), None, ids["confirm"], meta)))
        Qv = apply(Pv, Fv, use, w); print("   val   ", end=" "); score({q:list(p) for q,p in zip(vids,Qv)})

"""Component diagnostics: win-only and size-only SSE (x1000) per condition, given pw dict and sz dict."""
import sys,os; sys.path.insert(0, os.path.dirname(__file__))
from score import T, COND, REFERR
import numpy as np
def comp(pw, sz):
    ew={}; es={}
    for q,(t,_) in T.items():
        tw=1-t[0]; ts=[x/max(tw,1e-12) for x in t[1:]]
        p=pw[q]; ew.setdefault(COND[q],[]).append((p-tw)**2+sum((p*a-tw*a)**2 for a in ts))
        es.setdefault(COND[q],[]).append(sum((tw*a-tw*b)**2 for a,b in zip(sz[q],ts)))
    f=lambda e:" ".join(f"{k[:5]}={np.mean(v)*1000:.2f}" for k,v in e.items())
    return "WIN " + f(ew) + " | SIZE " + f(es)

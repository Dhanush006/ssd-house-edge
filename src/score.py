"""Fast replica of the contest score (validation). preds: dict qid -> 5 probs."""
import csv
from pathlib import Path
from collections import defaultdict
PUB = Path(__file__).resolve().parents[1] / "kit/public/validation"
def _read(p): return list(csv.DictReader(open(p)))
T = {r["query_id"]: ([float(r[f"p_{k}"]) for k in (0,1,2,5,20)], r["episode_id"]) for r in _read(PUB/"targets.csv")}
REF = {r["query_id"]: [float(r[f"p_{k}"]) for k in (0,1,2,5,20)] for r in _read(PUB/"reference.csv")}
COND = {r["query_id"]: r["condition"] for r in _read(PUB/"conditions.csv")}
def cond_err(preds):
    out = {}
    for c in ("stable","short_history","drift","shared_change"):
        ep = defaultdict(list)
        for q,(t,e) in T.items():
            if COND[q]==c: ep[e].append(sum((a-b)**2 for a,b in zip(preds[q],t)))
        out[c] = sum(sum(v)/len(v) for v in ep.values())/len(ep)
    return out
REFERR = cond_err(REF)
def score(preds, verbose=True):
    e = cond_err(preds); r = {c: e[c]/REFERR[c] for c in e}
    R = sum(r.values())/4; pts = 60/(1+R)
    if verbose: print(f"points={pts:.3f} R={R:.4f} " + " ".join(f"{c}={v:.3f}" for c,v in r.items()))
    return pts, r
def load_csv(p): return {r["query_id"]:[float(r[f"p_{k}"]) for k in (0,1,2,5,20)] for r in _read(p)}
if __name__=="__main__":
    import sys; score(load_csv(sys.argv[1]))

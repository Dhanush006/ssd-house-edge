"""House Edge gate model: per-query combination of model B (structured filter) and model C (trees).

  p = w * p_B + (1 - w) * p_C,   w = sigmoid(theta . [1, f])
  f = B's own confidence for this query: posterior weight of its top rule hypothesis, posterior weight of the
      "no rule" hypothesis, screening evidence gap between the two best hypotheses (/10, capped at 2),
      log(1 + history length) / 6, and the B-C disagreement (L1 distance).
theta is fitted offline on a dev split of the kit's training data (observed outcomes, never validation targets).
Pure Python standard library at prediction time (fit_gate uses numpy, offline only).
"""
import json, math, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

FEAT = ["top_post", "none_post", "gap", "loghist", "l1_bc"]


def sig(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x); return e / (1.0 + e)


def fvec(gf, pb, pc):
    return [1.0] + list(gf[:4]) + [math.fsum(abs(a - b) for a, b in zip(pb, pc))]


def combine(PB, PC, F, th):
    out = {}
    for q, pb in PB.items():
        pc = PC.get(q)
        if pc is None:
            out[q] = pb; continue
        w = sig(math.fsum(a * b for a, b in zip(th, fvec(F.get(q, [0.5, 0.0, 1.0, 0.8]), pb, pc))))
        out[q] = [w * a + (1 - w) * b for a, b in zip(pb, pc)]
    return out


def _bwork(args):
    import model_b
    P, items = args
    return model_b._work((dict(P, gatefeat=True), items))


def b_run(items, Pb, workers=1):
    """-> ({qid: B probs}, {qid: B confidence features}); episodes split across processes (deterministic)."""
    import model_b
    if workers > 1:
        import multiprocessing as mp
        eps = sorted(set(it[1] for it in items)); slot = {e: i % workers for i, e in enumerate(eps)}
        chunks = [[it for it in items if slot[it[1]] == i] for i in range(workers)]
        with mp.get_context("spawn").Pool(workers) as pool:
            res = pool.map(_bwork, [(Pb, ch) for ch in chunks])
        rows = [r for rr in res for r in rr]
    else:
        rows = model_b._work((dict(Pb, gatefeat=True), items))
    return {q: v[0] for q, v in rows}, {q: v[1] for q, v in rows}


def b_feats(items, Pb, workers=1):
    return b_run(items, Pb, workers)[1]


def predict_split(d, P, workers=1):
    import model_b, model_c
    Pb = json.loads((HERE / P["parts"][0][1]).read_text()); Pc = json.loads((HERE / P["parts"][1][1]).read_text())
    qs, H = model_b.read_split(d, Pb.get("payvals"))
    items = [(q["query_id"], q["episode_id"], H[q["machine_id"]], int(q["cutoff"]), q["machine_type"], q["signal"])
             for q in sorted(qs, key=lambda r: r["query_id"])]
    PB, F = b_run(items, Pb, workers)
    PC = model_c.predict_items(items, Pc)
    return combine(PB, PC, F, P["theta"])


def fit_gate(DB, DC, F, meta, lam=1e-3):
    """theta minimizing the mean Brier of the mixture on the 'dev' split items; reports dev/confirm vs B, C."""
    import numpy as np
    from scipy.optimize import minimize
    qs = [q for q in DB if q in DC and q in F and meta[q][2] == "dev"]
    if len(qs) < 200:
        return None, "too few dev items"
    X = np.array([fvec(F[q], DB[q], DC[q]) for q in qs]); PBm = np.array([DB[q] for q in qs]); PCm = np.array([DC[q] for q in qs])
    Y = np.zeros_like(PBm); Y[np.arange(len(qs)), [meta[q][3] for q in qs]] = 1.0
    D = PBm - PCm; R0 = PCm - Y

    def f(th):
        w = 1 / (1 + np.exp(-(X @ th))); E = R0 + w[:, None] * D
        L = (E ** 2).sum(1).mean() + lam * (th[1:] ** 2).sum()
        g = X.T @ (2 * (E * D).sum(1) * w * (1 - w)) / len(qs); g[1:] += 2 * lam * th[1:]
        return L, g
    r = minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 300})
    th = [round(float(v), 5) for v in r.x]
    w = 1 / (1 + np.exp(-(X @ r.x)))
    return th, f"theta={th} (features 1,{','.join(FEAT)}) mean w on dev={w.mean():.2f} sd={w.std():.2f}"

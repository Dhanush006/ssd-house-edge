"""House Edge model P: safe pooled fallback (pure Python standard library, deterministic, < 5 s).

p(next payout) = (own exponentially weighted payout counts + k * type frequencies) / (own weight + k)
  * type frequencies: payout frequencies per machine type over organizer train + validation histories (+ train labels);
    types not seen at fit time use the pooled frequencies of all types;
  * (half-life, k) chosen on validation targets (grid), else defaults.
Only ticks < cutoff of the query's own machine are read at prediction time.
usage (fit): python model_p.py fit <kit_public_dir> <out_params.json>
"""
import json, math, sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import kitschema as KS


def pred_one(s, c, mt, P):
    n = len(P["payvals"]); fr = P["freq"].get(mt) or P["freq"]["_pool"]
    g = 0.5 ** (1.0 / P["hl"]) if P["hl"] > 0 else 1.0; k = P["k"]
    acc = [0.0] * n; wsum = 0.0
    for t, v in s.items():
        if t < c and 0 <= v < n:
            w = g ** (c - 1 - t); acc[v] += w; wsum += w
    p = [(a + k * f) / (wsum + k) for a, f in zip(acc, fr)]
    p = [max(v, 1e-6) for v in p]; z = math.fsum(p)
    return [v / z for v in p]


def predict_items(items, P):
    return {qid: pred_one(s, c, mt, P) for qid, ep, s, c, mt, sg in items}


def predict_split(d, P, workers=1):
    qs, H = KS.read_split(d, P.get("payvals"))
    return {q["query_id"]: pred_one(H.get(q["machine_id"], {}), int(q["cutoff"]), q["machine_type"], P) for q in qs}


def fit(pub):
    pub = Path(pub); cols, vals = KS.find_schema(pub); n = len(vals)
    cnt = defaultdict(lambda: [1.0] * n)
    for sp in ("train", "validation"):
        d = pub / sp
        if not (d / "queries.csv").exists():
            continue
        qs, H = KS.read_split(d, vals); lab = KS.read_labels(d, vals) if sp == "train" else {}
        for q in qs:
            s = H.get(q["machine_id"], {}); c = int(q["cutoff"])
            for t, v in s.items():
                if t < c:
                    cnt[q["machine_type"]][v] += 1; cnt["_pool"][v] += 1
            if q["query_id"] in lab:
                cnt[q["machine_type"]][lab[q["query_id"]]] += 1; cnt["_pool"][lab[q["query_id"]]] += 1
    freq = {t: [x / math.fsum(c) for x in c] for t, c in sorted(cnt.items()) if math.fsum(c) > 200 or t == "_pool"}
    P = {"cols": list(cols), "payvals": list(vals), "freq": freq, "hl": 40.0, "k": 20.0}
    v = pub / "validation"; tg = KS.read_targets(v, cols)
    if tg:
        qs, H = KS.read_split(v, vals); best = None
        for hl in (0.0, 20.0, 40.0, 80.0):
            for k in (5.0, 20.0, 60.0, 200.0):
                Q = dict(P, hl=hl, k=k); err = 0.0
                for q in qs:
                    if q["query_id"] in tg:
                        p = pred_one(H.get(q["machine_id"], {}), int(q["cutoff"]), q["machine_type"], Q)
                        err += math.fsum((a - b) ** 2 for a, b in zip(p, tg[q["query_id"]]))
                if best is None or err < best[0] - 1e-12:
                    best = (err, hl, k)
        P["hl"], P["k"] = best[1], best[2]
    return P


if __name__ == "__main__" and len(sys.argv) > 3 and sys.argv[1] == "fit":
    P = fit(sys.argv[2]); Path(sys.argv[3]).parent.mkdir(parents=True, exist_ok=True)
    Path(sys.argv[3]).write_text(json.dumps(P, indent=1, sort_keys=True) + "\n")
    print(f"fit_p payouts={P['payvals']} types={sorted(P['freq'])} hl={P['hl']} k={P['k']}")

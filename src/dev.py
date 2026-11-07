"""Chronological development splits from TRAIN (observed outcomes only; train has no true probabilities).
Pseudo-queries: for each train query machine, cutoffs c-j (j in JS); label = observed payout at that tick;
history = ticks < c-j only (same episode). Episodes split deterministically (stratified by condition):
  confirm = every 3rd episode within each condition (untouched during tuning), dev = the rest.
Metric: Brier (sum of squared errors vs one-hot outcome) -> per episode mean -> per condition mean.
Expected Brier = true-prob SSE + constant, so differences between models estimate SSE differences."""
import sys, os, csv, json, random, importlib
sys.path.insert(0, os.path.dirname(__file__))
from pathlib import Path
from collections import defaultdict
HE = Path(__file__).resolve().parents[1]; TR = HE / "kit/public/train"
JS = (0, 2, 4, 6, 8, 10, 12, 14)
CONDS = ("stable", "short_history", "drift", "shared_change")

def build():
    import puremodel as M
    qs, H = M.read_split(TR)
    lab = {r["query_id"]: int(r["outcome"]) for r in csv.DictReader(open(TR / "labels.csv"))}
    cond = {r["query_id"]: r["condition"] for r in csv.DictReader(open(TR / "conditions.csv"))}
    epc = {}
    for q in qs: epc[q["episode_id"]] = cond[q["query_id"]]
    split = {}
    for cnd in CONDS:
        eps = sorted(e for e, c in epc.items() if c == cnd)
        for i, e in enumerate(eps): split[e] = "confirm" if i % 3 == 2 else "dev"
    items = []; meta = {}
    for q in sorted(qs, key=lambda r: r["query_id"]):
        s = H[q["machine_id"]]; c = int(q["cutoff"])
        for j in JS:
            cj = c - j
            y = lab[q["query_id"]] if j == 0 else s.get(cj)
            if y is None: continue
            hist = {t: v for t, v in s.items() if t < cj}
            qid = f'{q["query_id"]}@{j}'; pep = f'{q["episode_id"]}@{j}'
            items.append((qid, pep, hist, cj, q["machine_type"], int(q["signal"])))
            meta[qid] = (q["episode_id"], epc[q["episode_id"]], split[q["episode_id"]], y, len(hist), q["machine_type"])
    return items, meta

def predict(items, P, mod="puremodel", workers=6):
    M = importlib.import_module(mod)
    eps = sorted(set(it[1] for it in items)); slot = {e: i % workers for i, e in enumerate(eps)}
    chunks = [[it for it in items if slot[it[1]] == i] for i in range(workers)]
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(workers) as pool:
        res = pool.map(M._work, [(P, ch) for ch in chunks])
    out = {}
    for r in res: out.update(dict(r))
    return out

def brier_by_episode(pred, meta, which):
    ep = defaultdict(list)
    for qid, p in pred.items():
        e, c, sp, y, n, mt = meta[qid]
        if sp != which: continue
        oh = [1.0 if v == y else 0.0 for v in (0, 1, 2, 5, 20)]
        ep[(c, e)].append(sum((a - b) ** 2 for a, b in zip(p, oh)))
    return {k: sum(v) / len(v) for k, v in ep.items()}

def compare(pa, pb, meta, which, nboot=2000, seed=0):
    """Paired episode-level difference B - A (negative = B better), per condition and macro-avg, with bootstrap 90% CI."""
    A = brier_by_episode(pa, meta, which); B = brier_by_episode(pb, meta, which)
    rng = random.Random(seed); out = {}
    per = {c: [B[k] - A[k] for k in sorted(A) if k[0] == c] for c in CONDS}
    for c, d in per.items():
        bs = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(nboot))
        out[c] = (1000 * sum(d) / len(d), 1000 * bs[int(.05 * nboot)], 1000 * bs[int(.95 * nboot)], sum(x < 0 for x in d), len(d))
    bs = []
    for _ in range(nboot):
        bs.append(sum(sum(rng.choice(d) for _ in d) / len(d) for d in per.values()) / 4)
    bs.sort(); m = sum(sum(d) / len(d) for d in per.values()) / 4
    out["macro"] = (1000 * m, 1000 * bs[int(.05 * nboot)], 1000 * bs[int(.95 * nboot)])
    return out

def fmt(o):
    s = " ".join(f"{c[:5]}={v[0]:+.2f}[{v[1]:+.2f},{v[2]:+.2f}]w{v[3]}/{v[4]}" for c, v in o.items() if c != "macro")
    m = o["macro"]; return f"macro={m[0]:+.3f}[{m[1]:+.3f},{m[2]:+.3f}] | " + s

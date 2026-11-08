"""Offline check of the per-machine online mixer (Fixed-Share style exponential weights over B and C).
For every validation query at cutoff c, both models also forecast the same episode truncated at c-k (k = 1..M);
the observed payout at c-k scores them. Weight on B = w0*exp(-eta*L_B) / (w0*exp(-eta*L_B) + (1-w0)*exp(-eta*L_C)).
eta = 0 is the global blend. Scores against the validation true probabilities.
usage: python src/mix_offline.py <kit_public> <run_dir> [M]"""
import sys, json, math, time
from pathlib import Path
from collections import defaultdict

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import kitschema as KS, r2score as S, model_b as MB, model_c as MC
from round2_run import pool_predict


def main():
    pub, run = Path(sys.argv[1]), Path(sys.argv[2]); M = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    S.set_schema(pub); vals = KS.find_schema(pub)[1]
    Pb = json.loads((run / "B" / "params.json").read_text()); Pc = json.loads((run / "C" / "params.json").read_text())
    qs, H = KS.read_split(pub / "validation", vals)
    base = [(q["query_id"], q["episode_id"], H[q["machine_id"]], int(q["cutoff"]), q["machine_type"], q["signal"])
            for q in sorted(qs, key=lambda r: r["query_id"])]
    items = list(base)
    for k in range(1, M + 1):
        items += [(f"{q}@{k}", f"{e}@{k}", s, c - k, mt, sg) for q, e, s, c, mt, sg in base]
    cache = run / f"mix_cache_M{M}.json"
    if cache.exists():
        pb, pc = json.loads(cache.read_text())
    else:
        t = time.time(); pb = pool_predict(MB, items, Pb, 3); print(f"B {time.time() - t:.0f}s", flush=True)
        t = time.time(); pc = MC.predict_items(items, Pc); print(f"C {time.time() - t:.0f}s", flush=True)
        cache.write_text(json.dumps([pb, pc]))
    br = lambda p, y: math.fsum((a - (1.0 if j == y else 0.0)) ** 2 for j, a in enumerate(p))
    # per query, per k: (brier B, brier C) on the observed payout at c-k
    L = {}
    for q, e, s, c, mt, sg in base:
        L[q] = [(k, br(pb[f"{q}@{k}"], s[c - k]), br(pc[f"{q}@{k}"], s[c - k])) for k in range(1, M + 1) if (c - k) in s]
    ep_of = {q: e for q, e, *_ in base}; by_ep = defaultdict(list)
    for q in L: by_ep[ep_of[q]].append(q)
    Sc = S.Scorer(pub)
    pbv = {q: pb[q] for q, *_ in base}; pcv = {q: pc[q] for q, *_ in base}

    def mixed(w0, eta, gam, pool, m):
        out = {}
        for q in L:
            src = by_ep[ep_of[q]] if pool == "ep" else [q]
            dl = math.fsum(gam ** (k - 1) * (lb - lc) for r in src for k, lb, lc in L[r] if k <= m)
            if pool == "ep": dl /= len(src)
            z = max(-50.0, min(50.0, eta * dl))
            wb = w0 / (w0 + (1 - w0) * math.exp(z))
            out[q] = [wb * a + (1 - wb) * b for a, b in zip(pbv[q], pcv[q])]
        return out

    print(f"B     {Sc.line('B', pbv)}"); print(f"C     {Sc.line('C', pcv)}")
    best = {}
    for w0 in (1.0, 0.9, 0.7, 0.5, 0.3, 0.0):
        best[w0] = Sc.score(mixed(w0, 0, 1, "m", M))[0]
    print("global blend w0 -> points:", " ".join(f"{w}:{v:.3f}" for w, v in best.items()))
    w0b = max(best, key=best.get); w0s = sorted({w0b, 0.9, 0.5})
    rows = []
    for w0 in w0s:
        if w0 in (0.0, 1.0): continue
        for pool in ("m", "ep"):
            for m in (4, M):
                for gam in (1.0, 0.8):
                    for eta in (0.5, 1, 2, 4, 8, 16):
                        rows.append((Sc.score(mixed(w0, eta, gam, pool, m))[0], w0, pool, m, gam, eta))
    rows.sort(reverse=True)
    print("top mixer settings (points, w0, pool, M, gamma, eta):")
    for r in rows[:8]: print("  ", f"{r[0]:.3f}", r[1:])
    print(f"best global blend {best[w0b]:.3f} (w0={w0b}) | best mixer {rows[0][0]:.3f} | gain {rows[0][0] - best[w0b]:+.3f}")


if __name__ == "__main__":
    main()

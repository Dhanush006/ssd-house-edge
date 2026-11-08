"""Neighbor-machine coupling test on a kit (analysis only).

usage: python src/neighbor.py <kit_public> <B_params.json> [--workers 6]
Runs model B's filter over the full observed history of every train + validation machine and records the
one-step predictive P(win) at every tick (before the episode offset). Residual u_t = win_t - p_t.
For machine i, other machine j (same episode) and lag k = 1..8:  z_ijk = sum_t u_t x_j(t-k) / sqrt(sum_t p(1-p))
with x_j = +1 win / -1 loss. Also the episode count: x = (wins of the other machines at t-k) - their mean.
Null: the same statistics with j taken from a different episode (ticks aligned) -> no coupling possible.
Prints, per lag: mean z (common sign), mean z^2 (any sign; 1 under null) for real vs null pairs, and the
distribution of the per-machine maximum |z| over (j, k).
"""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(__file__))
from pathlib import Path
from collections import defaultdict
import numpy as np
import model_b as M
import kitschema as KS


def run(args):
    P, rows = args
    F = M.WinFilter(P); out = []
    for key, s, c, mt, sg, ctx in rows:
        ticks = sorted(t for t in s if t < c)
        if len(ticks) < 40: out.append((key, {})); continue
        _, rec = F.machine(s, c, mt, ticks, sg, ctx)
        out.append((key, {t: pw for t, pw, y in rec}))
    return out


def main():
    pub = Path(sys.argv[1]); P = json.loads(Path(sys.argv[2]).read_text()); P["eL"] = 100000
    W = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 6
    vals = P.get("payvals"); mach = {}
    for sp in ("train", "validation"):
        qs, H = KS.read_split(pub / sp, vals); lab = KS.read_labels(pub / sp, vals)
        for q in qs:
            s = {t: v for t, v in H[q["machine_id"]].items() if t < q["cutoff"]}
            if q["query_id"] in lab: s[q["cutoff"]] = lab[q["query_id"]]
            mach[q["query_id"]] = dict(ep=q["episode_id"], s=s, c=q["cutoff"] + 1, mt=q["machine_type"], sg=q["signal"])
    byep = defaultdict(list)
    for k in sorted(mach): byep[mach[k]["ep"]].append(k)
    for ep, ks in byep.items():
        hs = [mach[k]["s"] for k in ks]
        for n, k in enumerate(ks): mach[k]["ctx"] = (hs, n)
    rows = [(k, m["s"], m["c"], m["mt"], m["sg"], m["ctx"]) for k, m in sorted(mach.items())]
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(W) as pool:
        res = pool.map(run, [(P, rows[i::W]) for i in range(W)])
    pw = {k: r for rr in res for k, r in rr}
    eps = sorted(byep)

    def stats(i, others):
        s = mach[i]["s"]; p = pw[i]
        ts = [t for t in sorted(p) if t in s]
        if len(ts) < 40: return None
        u = np.array([(1.0 if s[t] > 0 else 0.0) - p[t] for t in ts]); v = np.array([p[t] * (1 - p[t]) for t in ts]); sd = math.sqrt(v.sum())
        Z = np.zeros((len(others), 8)); C = np.zeros(8)
        for k in range(1, 9):
            cnt = np.zeros(len(ts)); nn = np.zeros(len(ts))
            for a, j in enumerate(others):
                sj = mach[j]["s"]
                x = np.array([0.0 if (t - k) not in sj else (1.0 if sj[t - k] > 0 else -1.0) for t in ts])
                Z[a, k - 1] = (u * x).sum() / sd; cnt += (x > 0); nn += (x != 0)
            xc = np.where(nn > 0, cnt / np.maximum(nn, 1), 0.0); xc = xc - xc.mean()
            C[k - 1] = (u * xc).sum() / math.sqrt((v * xc ** 2).sum() + 1e-12)
        return Z, C

    real, null = [], []
    for e, ep in enumerate(eps):
        ks = byep[ep]; ks2 = byep[eps[(e + 1) % len(eps)]]
        for i in ks:
            r = stats(i, [j for j in ks if j != i]); n = stats(i, ks2[:len(ks) - 1])
            if r is not None and n is not None: real.append(r); null.append(n)
    for name, L in (("real", real), ("null", null)):
        Z = np.stack([z for z, c in L]); C = np.stack([c for z, c in L])
        print(f"{name}: machines={len(L)}")
        print("  pair z mean by lag   " + " ".join(f"{x:+.3f}" for x in Z.mean((0, 1))))
        print("  pair z^2 mean by lag " + " ".join(f"{x:.3f}" for x in (Z ** 2).mean((0, 1))))
        print("  count z mean by lag  " + " ".join(f"{x:+.3f}" for x in C.mean(0)))
        print("  count z^2 by lag     " + " ".join(f"{x:.3f}" for x in (C ** 2).mean(0)))
        mx = np.abs(Z).reshape(len(L), -1).max(1)
        print("  per-machine max|z| over (j,k): quantiles 50/90/99 " + " ".join(f"{np.quantile(mx, q):.2f}" for q in (.5, .9, .99))
              + f" | frac > 3.5: {(mx > 3.5).mean():.3f} > 4: {(mx > 4).mean():.3f}")
        # relative index structure: is the strongest partner a fixed offset?
    np.save(Path(sys.argv[2]).parent / "neighbor_z.npy", np.stack([z for z, c in real]))


if __name__ == "__main__":
    main()

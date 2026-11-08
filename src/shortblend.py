"""Offline: blend weight on B chosen separately for short (gate loghist feature < log1p(100)/6) and long histories, on dev; report confirm."""
import sys, json, math
sys.path.insert(0, "src")
import round2_run as R
for tag in sys.argv[1:]:
    out = R.HE / "runs" / f"R2-{tag}"
    J = json.loads((out / "dev_preds.json").read_text()); dev = J["preds"]; meta = {k: tuple(v) for k, v in J["meta"].items()}
    fs = sorted(out.glob("gate_feats*.json"))
    if not fs or "C" not in dev: print(tag, "skip"); continue
    F = json.loads(fs[0].read_text()); thr = math.log1p(100) / 6
    short = {q: F.get(q, [0, 0, 0, 1])[3] < thr for q in dev["B"]}
    def mixw(ws, wl): return {q: [(ws if short[q] else wl) * a + (1 - (ws if short[q] else wl)) * b for a, b in zip(dev["B"][q], dev["C"][q])] for q in dev["B"]}
    grid = [i / 10 for i in range(11)]
    g1 = min(grid, key=lambda w: R.dev_brier(mixw(w, w), meta, "dev")[0])
    g2 = min(((a, b) for a in grid for b in grid), key=lambda ab: R.dev_brier(mixw(*ab), meta, "dev")[0])
    r = lambda p: f"dev={1e3 * R.dev_brier(p, meta, 'dev')[0]:.2f} conf={1e3 * R.dev_brier(p, meta, 'confirm')[0]:.2f}"
    print(f"{tag:<12} n_short={sum(short.values())} B {r(dev['B'])} | blend w={g1} {r(mixw(g1, g1))} | split w_short={g2[0]} w_long={g2[1]} {r(mixw(*g2))}")

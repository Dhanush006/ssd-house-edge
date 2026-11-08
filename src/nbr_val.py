"""Validation diagnostic: B's final win logit + gam * (others' win share at c-k minus their EW win rate); scores per k, gam."""
import sys, os, math
sys.path.insert(0, os.path.dirname(__file__))
from collections import defaultdict
import r2score as S, kitschema as KS, model_b as M
pub = sys.argv[1]; bcsv = sys.argv[2]
S.set_schema(pub); Sc = S.Scorer(pub)
qs, H = KS.read_split(pub + "/validation"); PB = S.load_csv(bcsv)
by = defaultdict(list)
for q in qs: by[q["episode_id"]].append(q)
def feat(q, k, hl=16.0):
    g = 0.5 ** (1 / hl); acc = []; 
    for o in by[q["episode_id"]]:
        if o is q: continue
        s = H[o["machine_id"]]; v = s.get(q["cutoff"] - k)
        if v is None: continue
        a, b = 0.5, 1.0
        for t in sorted(t for t in s if t < q["cutoff"] - k): a = a * g + (s[t] > 0); b = b * g + 1
        acc.append((v > 0) - a / b)
    return sum(acc) / len(acc) if acc else 0.0
def adj(p, e):
    pw = 1 - p[0]; pw2 = M.sig(M.logit(pw) + e); return [1 - pw2] + [x * pw2 / pw for x in p[1:]]
print(Sc.line("B", PB))
for k in (1, 2, 3, 4, 5):
    F = {q["query_id"]: feat(q, k) for q in qs}
    res = []
    for gam in (0.5, 1.0, 1.5, 2.0, 3.0):
        res.append((gam, Sc.score({qq: adj(p, gam * F[qq]) for qq, p in PB.items()})[0]))
    print(f"k={k} " + " ".join(f"g{g}:{v:.3f}" for g, v in res))
def feat2(q, k, mode):
    acc = []
    for o in by[q["episode_id"]]:
        if o is q: continue
        if mode == "same" and o["machine_type"] != q["machine_type"]: continue
        if mode == "diff" and o["machine_type"] == q["machine_type"]: continue
        s = H[o["machine_id"]]; v = s.get(q["cutoff"] - k)
        if v is None: continue
        rate = sum(1 for t in s if s[t] > 0 and t < q["cutoff"] - k) / max(1, sum(1 for t in s if t < q["cutoff"] - k))
        x = (v >= 2) - 0.5 if mode == "big" else (v > 0) - rate
        acc.append(x)
    return sum(acc) / len(acc) if acc else 0.0
for mode in ("all", "same", "diff", "big"):
    F = {q["query_id"]: feat2(q, 3, mode) for q in qs}
    print(mode, " ".join(f"g{g}:{Sc.score({qq: adj(p, g * F[qq]) for qq, p in PB.items()})[0]:.3f}" for g in (-0.5, 0.25, 0.5, 0.75, 1.0)))

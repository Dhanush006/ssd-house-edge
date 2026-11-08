"""Is the rule hypothesis (MAP among top population hypotheses) / size lag associated with machine type or signal? (analysis only)"""
import sys, json, math
sys.path.insert(0, "src")
from collections import Counter, defaultdict
import model_b as M, kitschema as KS
pub = sys.argv[1]; P = json.load(open(sys.argv[2]))
top = [M.to_h(h) for h, w in sorted(zip(P["hyps"], P["hprior"]), key=lambda r: -r[1])[:3]]
tab = defaultdict(Counter); tabs = defaultdict(Counter); tabe = defaultdict(list)
for sp in ("train", "validation"):
    qs, H = KS.read_split(pub + "/" + sp)
    for q in qs:
        s = {t: v for t, v in H[q["machine_id"]].items() if t < q["cutoff"]}
        ticks = sorted(s); tc = M.common_ticks(s, ticks, 8)
        if len(tc) < 150: continue
        ll = [M.screen_ll(s, tc, h, 64) for h in top]; j = max(range(3), key=lambda i: ll[i])
        if sorted(ll)[-1] - sorted(ll)[-2] < 5: continue
        tab[q["machine_type"]][M.hkey(top[j])] += 1; tabs[q["signal"]][M.hkey(top[j])] += 1; tabe[q["episode_id"]].append(M.hkey(top[j]))
for name, T in (("type", tab), ("signal", tabs)):
    for k in sorted(T): print(name, k, dict(T[k]))
same = sum(sum(c * (c - 1) for c in Counter(v).values()) for v in tabe.values()); tot = sum(len(v) * (len(v) - 1) for v in tabe.values())
print(f"within-episode same-rule pair rate {same / tot:.3f} (1/3 if independent)")

"""Oracle population prior for B (analysis only): python src/oracle_b.py <B_params.json> <out.json> "<h1;h2;..>" <tie|uniform>
Sets the rule prior to the true hypothesis set (uniform) and the size-driver prior to the true rule
(tie = first lag, tie2 = second lag, uniform = lags 1..8), keeping everything else fitted."""
import sys, json
from pathlib import Path
import model_b as M
P = json.loads(Path(sys.argv[1]).read_text()); hs = sys.argv[3].split(";"); mode = sys.argv[4]
P["hprior"] = [(0.998 / len(hs) if h in hs else 0.002 / (len(P["hyps"]) - len(hs))) for h in P["hyps"]]
nk = P.get("nsz", len(M.SZ)); kp = {}
for h in P["hyps"]:
    v = [1e-4] * nk
    if h in hs:
        a, b = [int(x) for x in h.split(",")]
        if mode == "uniform":
            for k in range(1, 9): v[k] = 1.0
        else: v[a if mode == "tie" else b] = 1.0
    z = sum(v); kp[h] = [x / z for x in v]
kp["-1,-1"] = [1.0 / nk] * nk; P["kprior"] = kp
Path(sys.argv[2]).write_text(json.dumps(P, indent=1, sort_keys=True) + "\n")

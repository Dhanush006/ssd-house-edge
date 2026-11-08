"""Linear blend of entry models (pure Python standard library, deterministic).
params.json: {"parts": [["model_b", "params_b.json", 0.7], ["model_c", "params_c.json", 0.3]]}
Each part is a module and its params file, both next to this file."""
import importlib, json, math, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def predict_split(d, P, workers=1):
    acc = None; tot = math.fsum(w for _, _, w in P["parts"])
    for mod, pf, w in P["parts"]:
        M = importlib.import_module(mod)
        pr = M.predict_split(d, json.loads((HERE / pf).read_text()), workers)
        if acc is None:
            acc = {q: [0.0] * len(p) for q, p in pr.items()}
        for q, p in pr.items():
            a = acc[q]
            for j in range(len(a)):
                a[j] += (w / tot) * p[j]
    return acc

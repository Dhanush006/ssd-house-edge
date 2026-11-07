"""Kit-parameterized contest score on validation (true probabilities) + CSV helpers.
usage: python src/r2score.py <kit_public_dir> <validation_predictions.csv> [more.csv ...]"""
import csv, json, sys
from pathlib import Path
from collections import defaultdict

CONDS = ("stable", "short_history", "drift", "shared_change")
PK = (0, 1, 2, 5, 20)


def _rows(p):
    return list(csv.DictReader(open(p, newline="", encoding="utf-8")))


def load_csv(p):
    return {r["query_id"]: [float(r[f"p_{k}"]) for k in PK] for r in _rows(p)}


def write_csv(path, preds):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["query_id", "p_0", "p_1", "p_2", "p_5", "p_20"])
        for q in sorted(preds):
            p = [max(0.0, float(x)) for x in preds[q]]; z = sum(p)
            w.writerow([q] + [repr(x / z) for x in p])


class Scorer:
    def __init__(self, pub):
        v = Path(pub) / "validation"
        self.T = {r["query_id"]: ([float(r[f"p_{k}"]) for k in PK], r["episode_id"]) for r in _rows(v / "targets.csv")}
        self.COND = {r["query_id"]: r["condition"] for r in _rows(v / "conditions.csv")}
        self.maxp = json.loads((Path(pub) / "metadata.json").read_text()).get("max_points", 60)
        self.REF = self.cond_err(load_csv(v / "reference.csv"))

    def cond_err(self, preds):
        out = {}
        for c in CONDS:
            ep = defaultdict(list)
            for q, (t, e) in self.T.items():
                if self.COND[q] == c:
                    ep[e].append(sum((a - b) ** 2 for a, b in zip(preds[q], t)))
            out[c] = sum(sum(v) / len(v) for v in ep.values()) / len(ep)
        return out

    def score(self, preds):
        e = self.cond_err(preds); r = {c: e[c] / max(self.REF[c], 1e-9) for c in CONDS}
        R = sum(r.values()) / 4
        return self.maxp / (1 + R), R, r

    def line(self, name, preds):
        pts, R, r = self.score(preds)
        return f"{name:<14} points={pts:7.3f}/{self.maxp} R={R:.4f} " + " ".join(f"{c[:6]}={r[c]:.4f}" for c in CONDS)


if __name__ == "__main__":
    S = Scorer(sys.argv[1])
    for p in sys.argv[2:]:
        print(S.line(Path(p).parent.name, load_csv(p)))

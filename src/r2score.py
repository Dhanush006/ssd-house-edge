"""Kit-parameterized contest score on validation (true probabilities) + CSV helpers.
usage: python src/r2score.py <kit_public_dir> <validation_predictions.csv> [more.csv ...]
Payout columns / conditions / max points come from the kit (sample_submission header, conditions.csv, metadata)."""
import csv, json, sys
from pathlib import Path
from collections import defaultdict
import kitschema as KS

CONDS = ("stable", "short_history", "drift", "shared_change")
COLS = list(KS.DEFAULT_COLS)
PK = (0, 1, 2, 5, 20)


def set_schema(pub):
    global COLS, CONDS
    COLS = KS.find_schema(pub)[0]
    cs = []
    for sp in ("validation", "train"):
        f = Path(pub) / sp / "conditions.csv"
        if f.exists():
            for r in _rows(f):
                if r.get("condition") and r["condition"] not in cs: cs.append(r["condition"])
    if cs:
        main = [c for c in ("stable", "short_history", "drift", "shared_change") if c in cs]
        CONDS = tuple(main + sorted(c for c in cs if c not in main))
    return COLS, CONDS


def _rows(p):
    return list(csv.DictReader(open(p, newline="", encoding="utf-8-sig")))


def load_csv(p):
    rows = _rows(p)
    cols = KS.cols_from_header(rows[0].keys())[0] if rows else COLS
    return {r["query_id"]: [float(r[c]) for c in cols] for r in rows}


def write_csv(path, preds):
    KS.write_submission(path, preds, COLS)


class Scorer:
    def __init__(self, pub):
        v = Path(pub) / "validation"
        tg = KS.read_targets(v, COLS)
        ep = {r["query_id"]: r.get("episode_id", r["query_id"]) for r in _rows(v / "targets.csv")} if (v / "targets.csv").exists() else {}
        self.T = {q: (p, ep[q]) for q, p in tg.items()}
        self.COND = {r["query_id"]: r["condition"] for r in _rows(v / "conditions.csv")} if (v / "conditions.csv").exists() else {q: "all" for q in self.T}
        self.conds = [c for c in CONDS if c in set(self.COND.values())] or sorted(set(self.COND.values()))
        self.maxp = KS.metadata(pub).get("max_points", 40)
        self.ok = bool(self.T)
        self.REF = self.cond_err(load_csv(v / "reference.csv")) if (self.ok and (v / "reference.csv").exists()) else None

    def cond_err(self, preds):
        out = {}
        for c in self.conds:
            ep = defaultdict(list)
            for q, (t, e) in self.T.items():
                if self.COND.get(q) == c and q in preds:
                    ep[e].append(sum((a - b) ** 2 for a, b in zip(preds[q], t)))
            out[c] = sum(sum(v) / len(v) for v in ep.values()) / max(1, len(ep))
        return out

    def score(self, preds):
        if not self.ok:
            return float("nan"), float("nan"), {c: float("nan") for c in self.conds}
        e = self.cond_err(preds)
        ref = self.REF or {c: 1.0 for c in self.conds}  # no reference: raw error (points not comparable)
        r = {c: e[c] / max(ref[c], 1e-9) for c in self.conds}
        R = sum(r.values()) / max(1, len(r))
        return self.maxp / (1 + R), R, r

    def line(self, name, preds):
        pts, R, r = self.score(preds)
        return f"{name:<14} points={pts:7.3f}/{self.maxp} R={R:.4f} " + " ".join(f"{c[:6]}={r[c]:.4f}" for c in self.conds)


if __name__ == "__main__":
    set_schema(sys.argv[1])
    S = Scorer(sys.argv[1])
    for p in sys.argv[2:]:
        print(S.line(Path(p).parent.name, load_csv(p)))

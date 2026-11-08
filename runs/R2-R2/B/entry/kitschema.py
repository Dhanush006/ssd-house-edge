"""Kit schema helpers (Python standard library only; shipped inside every entry).

Nothing about the kit layout is assumed beyond the documented file names:
  * payout set = the p_<value> columns of sample_submission.csv (any count, any values), else the values seen in
    history/labels, else the main-round default (0, 1, 2, 5, 20);
  * payouts are handled internally as an index 0..n-1 in increasing value order (index 0 = loss, index >= 2 = "big");
    a history value that is not in the payout set is mapped to the nearest one;
  * missing query columns get neutral defaults (signal 0, machine_type "_pool"); signal may be any number;
  * cutoff / tick may be written as integers or integral floats.
"""
import csv, json, math
from collections import defaultdict
from pathlib import Path

DEFAULT_PAY = (0.0, 1.0, 2.0, 5.0, 20.0)
DEFAULT_COLS = ("p_0", "p_1", "p_2", "p_5", "p_20")


def _rows(p):
    with open(p, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _header(p):
    with open(p, newline="", encoding="utf-8-sig") as f:
        return next(csv.reader(f), [])


def num(v, default=0.0):
    try:
        x = float(str(v).strip())
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def to_int(v, default=0):
    x = num(v, None)
    return default if x is None else int(round(x))


def cols_from_header(h):
    """p_<value> columns in increasing value order -> (cols, values)."""
    cs = []
    for c in h:
        c = c.strip()
        if c.startswith("p_"):
            v = num(c[2:], None)
            if v is not None:
                cs.append((v, c))
    cs.sort()
    return tuple(c for _, c in cs), tuple(v for v, _ in cs)


def find_schema(d):
    """Payout columns/values for a split dir or a public dir (looks at */sample_submission.csv, then data)."""
    d = Path(d)
    cands = [d / "sample_submission.csv"] + [d / s / "sample_submission.csv" for s in ("test", "validation", "train")]
    cands += [d.parent / s / "sample_submission.csv" for s in ("test", "validation", "train")]
    for p in cands:
        if p.exists():
            cols, vals = cols_from_header(_header(p))
            if len(cols) >= 2:
                return list(cols), list(vals)
    seen = set()
    for p in [d / "history.csv"] + [d / s / "history.csv" for s in ("train", "validation", "test")]:
        if p.exists():
            for r in _rows(p):
                v = num(r.get("payout"), None)
                if v is not None:
                    seen.add(v)
    if len(seen) >= 2:
        vals = sorted(seen)
        return [f"p_{int(v) if v == int(v) else v}" for v in vals], vals
    return list(DEFAULT_COLS), list(DEFAULT_PAY)


class PayMap:
    def __init__(self, vals):
        self.vals = [float(v) for v in vals]
        self.exact = {v: i for i, v in enumerate(self.vals)}

    def idx(self, v):
        x = num(v, None)
        if x is None:
            return None
        i = self.exact.get(x)
        if i is None:
            i = min(range(len(self.vals)), key=lambda j: (abs(self.vals[j] - x), j))
        return i


def read_split(d, vals=None):
    """-> (queries [dicts with normalized fields], histories {machine_id: {tick: payout index}})."""
    d = Path(d)
    if vals is None:
        vals = find_schema(d)[1]
    pm = PayMap(vals)
    qs = []
    for r in _rows(d / "queries.csv"):
        q = dict(r)
        q["machine_type"] = (r.get("machine_type") or "").strip() or "_pool"
        q["signal"] = num(r.get("signal"), 0.0)
        q["cutoff"] = to_int(r.get("cutoff"), 0)
        if not q.get("episode_id"):
            q["episode_id"] = "ep-" + str(r.get("machine_id"))
        qs.append(q)
    H = defaultdict(dict)
    hp = d / "history.csv"
    if hp.exists():
        for r in _rows(hp):
            i = pm.idx(r.get("payout"))
            if i is None or r.get("tick") in (None, ""):
                continue
            H[r["machine_id"]][to_int(r["tick"])] = i
    return qs, H


def read_labels(d, vals):
    """labels.csv outcome -> payout index (or {} if absent)."""
    p = Path(d) / "labels.csv"
    if not p.exists():
        return {}
    pm = PayMap(vals); out = {}
    for r in _rows(p):
        i = pm.idx(r.get("outcome"))
        if i is not None:
            out[r["query_id"]] = i
    return out


def read_targets(d, cols):
    p = Path(d) / "targets.csv"
    if not p.exists():
        return {}
    out = {}
    for r in _rows(p):
        try:
            v = [float(r[c]) for c in cols]
        except (KeyError, ValueError, TypeError):
            return {}
        out[r["query_id"]] = v
    return out


def write_submission(path, preds, cols):
    """Rows sorted by query_id; probabilities clipped at 0 and renormalized; repr floats (byte-stable)."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    n = len(cols)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["query_id"] + list(cols))
        for q in sorted(preds):
            p = [max(0.0, float(x)) if math.isfinite(float(x)) else 0.0 for x in preds[q]][:n]
            p += [0.0] * (n - len(p))
            z = math.fsum(p)
            p = [x / z for x in p] if z > 0 else [1.0 / n] * n
            w.writerow([q] + [repr(x) for x in p])


def validate_submission(sub, split_dir, cols=None):
    """Mirror of the organizer checker rules -> list of problems ([] = valid).
    Columns exactly query_id + the payout columns (as a set), no duplicates/empties, every test query once,
    finite probabilities in [0, 1] summing to 1 within 1e-6."""
    split_dir = Path(split_dir); errs = []
    if cols is None:
        cols = find_schema(split_dir)[0]
    try:
        with open(sub, newline="", encoding="utf-8-sig") as f:
            rd = csv.reader(f, strict=True); h = next(rd, None)
            if h is None or len(set(h)) != len(h) or set(h) != {"query_id", *cols}:
                return [f"header {h} != query_id + {cols}"]
            seen = set()
            for ln, row in enumerate(rd, 2):
                if len(row) != len(h) or any(not v.strip() for v in row):
                    errs.append(f"row {ln}: missing/extra/empty field"); continue
                r = dict(zip(h, row)); q = r["query_id"]
                if q in seen:
                    errs.append(f"row {ln}: duplicate {q}")
                seen.add(q)
                try:
                    p = [float(r[c]) for c in cols]
                except ValueError:
                    errs.append(f"row {ln}: non-numeric"); continue
                if not all(math.isfinite(x) and 0 <= x <= 1 for x in p):
                    errs.append(f"row {ln}: probability outside [0,1] or not finite")
                if abs(math.fsum(p) - 1) > 1e-6:
                    errs.append(f"row {ln}: sum {math.fsum(p)}")
                if len(errs) > 20:
                    break
    except (OSError, csv.Error, UnicodeError) as e:
        return [f"unreadable: {e}"]
    want = {r["query_id"] for r in _rows(split_dir / "queries.csv")}
    miss = want - seen; extra = seen - want
    if miss: errs.append(f"missing {len(miss)} query ids e.g. {sorted(miss)[:2]}")
    if extra: errs.append(f"unexpected {len(extra)} query ids e.g. {sorted(extra)[:2]}")
    return errs


def metadata(pub):
    p = Path(pub) / "metadata.json"
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}

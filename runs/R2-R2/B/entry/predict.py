"""House Edge entry: python predict.py --data /data --out /output/submission.csv
Pure Python standard library; deterministic. Loads params.json + model.py next to this file.
Safety: every query in queries.csv gets a valid row. If the main model fails (or returns a bad row), the bundled
pooled fallback (model_p.py + params_p.json) is used for those queries, and a note goes to stderr."""
import argparse, json, math, os, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kitschema as KS


def ok_row(p, n):
    try:
        p = [float(x) for x in p]
    except (TypeError, ValueError):
        return None
    if len(p) != n or not all(math.isfinite(x) and x >= 0 for x in p) or math.fsum(p) <= 0:
        return None
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    t0 = time.time()
    data = Path(a.data)
    split_dir = data / a.split if (data / a.split / "queries.csv").exists() else data
    params = json.loads((HERE / "params.json").read_text())
    out_cols, out_vals = KS.find_schema(split_dir)
    mvals = params.get("payvals") or out_vals
    preds = {}
    try:
        import model
        try:
            preds = model.predict_split(split_dir, params, max(1, a.workers))
        except Exception as e:  # multiprocessing unavailable etc.: single process
            if a.workers > 1:
                print("note: parallel run failed, single process:", repr(e)[:200], file=sys.stderr)
                preds = model.predict_split(split_dir, params, 1)
            else:
                raise
    except Exception as e:
        print("WARNING: main model failed, using pooled fallback:", repr(e)[:300], file=sys.stderr)
        preds = {}
    qids = [q["query_id"] for q in KS.read_split(split_dir, mvals)[0]]
    nm = len(mvals); bad = [q for q in qids if ok_row(preds.get(q), nm) is None]
    if bad:
        fb = {}
        try:
            import model_p
            fb = model_p.predict_split(split_dir, json.loads((HERE / "params_p.json").read_text()))
        except Exception as e:
            print("WARNING: fallback failed, uniform rows:", repr(e)[:200], file=sys.stderr)
        for q in bad:
            preds[q] = ok_row(fb.get(q), nm) or [1.0 / nm] * nm
        print(f"note: {len(bad)} of {len(qids)} queries used the fallback", file=sys.stderr)
    # map model payout order onto the output columns (identity when the schema matches)
    pm = KS.PayMap(mvals); mp = [pm.idx(v) for v in out_vals]
    rows = {q: [ok_row(preds[q], nm)[j] for j in mp] for q in qids}
    KS.write_submission(a.out, rows, out_cols)
    print(f"wrote {a.out}: {len(rows)} rows, {time.time() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()

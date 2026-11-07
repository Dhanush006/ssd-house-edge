"""House Edge entry: python predict.py --data /data --out /output/submission.csv
Pure Python standard library; deterministic. Loads params.json + model.py next to this file."""
import argparse, csv, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    data = Path(a.data)
    split_dir = data / a.split if (data / a.split / "queries.csv").exists() else data
    params = json.loads((HERE / "params.json").read_text())
    preds = model.predict_split(split_dir, params)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["query_id", "p_0", "p_1", "p_2", "p_5", "p_20"])
        for qid in sorted(preds):
            p = [max(0.0, float(x)) for x in preds[qid]]
            z = sum(p); p = [x / z for x in p]
            w.writerow([qid] + [repr(x) for x in p])

if __name__ == "__main__":
    main()

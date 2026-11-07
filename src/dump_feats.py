"""Dump baseline preds + correction features for train pseudo-queries (dev+confirm) and validation."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import dev, puremodel as M
import multiprocessing as mp
def run(items, P, workers=6):
    eps = sorted(set(it[1] for it in items)); slot = {e: i % workers for i, e in enumerate(eps)}
    chunks = [[it for it in items if slot[it[1]] == i] for i in range(workers)]
    with mp.get_context("spawn").Pool(workers) as pool:
        res = pool.map(M._work, [(P, ch) for ch in chunks])
    return {qid: (p, f) for r in res for qid, p, f in r}
if __name__ == "__main__":
    P = json.load(open(sys.argv[1])); P["feat"] = True
    items, meta = dev.build()
    tr = run(items, P)
    qs, H = M.read_split(dev.HE / "kit/public/validation")
    vitems = [(q["query_id"], q["episode_id"], H[q["machine_id"]], int(q["cutoff"]), q["machine_type"], int(q["signal"])) for q in qs]
    va = run(vitems, P)
    json.dump({"train": tr, "meta": meta, "val": va}, open(dev.HE / "runs/dev/feats.json", "w"))
    print("dumped", len(tr), len(va))

"""Small bounded random search over existing dynamics params, scored on DEV split only."""
import sys, os, json, math, random, time
sys.path.insert(0, os.path.dirname(__file__))
import dev
if __name__ == "__main__":
    base = json.load(open(dev.HE / "runs/05-eps/params.json"))
    items, meta = dev.build()
    A = json.load(open(dev.HE / "runs/dev/base.json"))
    rng = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    lu = lambda a, b: math.exp(rng.uniform(math.log(a), math.log(b)))
    for i in range(int(sys.argv[2]) if len(sys.argv) > 2 else 8):
        g = dict(qb=round(lu(0.005, 0.02), 4), jump=round(lu(0.001, 0.005), 4), qd=round(lu(0.001, 0.006), 4), etau=round(rng.uniform(0.2, 0.45), 3))
        P = dict(base, **g); t = time.time()
        B = dev.predict(items, P)
        json.dump(B, open(dev.HE / f"runs/dev/rs{i}.json", "w"))
        print(f"rs{i}", g, f"{time.time()-t:.0f}s", dev.fmt(dev.compare(A, B, meta, "dev")), flush=True)

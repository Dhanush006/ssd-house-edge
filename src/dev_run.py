"""dev_run.py NAME PARAMS_JSON [module] : cache dev+confirm predictions to runs/dev/NAME.json"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
import dev
if __name__ == "__main__":
    name, pj = sys.argv[1], sys.argv[2]; mod = sys.argv[3] if len(sys.argv) > 3 else "puremodel"
    P = json.load(open(pj)) if pj.endswith(".json") else json.loads(pj)
    items, meta = dev.build(); t = time.time()
    pred = dev.predict(items, P, mod)
    os.makedirs(dev.HE / "runs/dev", exist_ok=True)
    json.dump(pred, open(dev.HE / f"runs/dev/{name}.json", "w"))
    print(name, len(items), f"{time.time()-t:.0f}s")

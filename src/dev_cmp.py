import sys, os, json; sys.path.insert(0, os.path.dirname(__file__))
import dev
if __name__ == "__main__":
    a, split = sys.argv[1], sys.argv[-1]
    items, meta = dev.build()
    A = json.load(open(dev.HE / f"runs/dev/{a}.json"))
    for b in sys.argv[2:-1]:
        B = json.load(open(dev.HE / f"runs/dev/{b}.json"))
        print(f"{b} vs {a} [{split}]", dev.fmt(dev.compare(A, B, meta, split)))

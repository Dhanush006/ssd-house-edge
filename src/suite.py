"""Synthetic stress-test suite: python src/suite.py <tagprefix> S1 S2 ... [--par 2]
Generates _synth/<S> kits (if missing), runs the round-2 runner on each, prints one table from summary.json."""
import sys, json, subprocess
from pathlib import Path
HE = Path(__file__).resolve().parents[1]; PY = sys.executable
a = sys.argv[1:]; pre = a[0]; par = int(a[a.index("--par") + 1]) if "--par" in a else 2
scs = [s for s in a[1:] if s.startswith("S") or s == "main"]
if "--table" not in a:
    procs = []
    for s in scs:
        kit = HE / "kit" if s == "main" else HE / "_synth" / s
        if s != "main" and not (kit / "public/metadata.json").exists():
            subprocess.run([PY, str(HE / "src/synth_kit.py"), str(kit), s], check=True, capture_output=True)
        lf = open(HE / "_synth" / f"log-{pre}-{s}.txt", "w")
        procs.append(subprocess.Popen([PY, str(HE / "src/round2_run.py"), str(kit), "--tag", f"{pre}-{s}", "--workers", str(10 // par)], stdout=lf, stderr=subprocess.STDOUT))
        if len(procs) >= par:
            procs.pop(0).wait()
    for p in procs: p.wait()
M = ("A", "B0", "Bn", "B", "Bt", "C", "D", "blend")
print(f"{'kit':<6} " + " ".join(f"{m:>7}" for m in M) + "  pick    pick_val best   flags")
for s in scs:
    f = HE / "runs" / f"R2-{pre}-{s}" / "summary.json"
    if not f.exists(): print(s, "missing"); continue
    J = json.loads(f.read_text()); v = J["val_points"]; best = max((x for x in M if x in v), key=lambda x: v[x])
    pk = J["pick"]
    print(f"{s:<6} " + " ".join(f"{v.get(m, float('nan')):7.2f}" for m in M) + f"  {pk:<7} {v.get(pk, 0):7.2f} {best:<6} {'; '.join(x.split(' ')[0] for x in J['flags'])}")

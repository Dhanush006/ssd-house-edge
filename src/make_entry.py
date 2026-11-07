"""Build an entry: make_entry.py <run_dir> <model.py> <params.json> "<desc>"
Creates <run_dir>/entry/, test+validation CSVs from a fresh copy, checks, zips."""
import sys, shutil, subprocess, json, hashlib, os
from pathlib import Path
HE = Path(__file__).resolve().parents[1]; KIT = HE/"kit"; PY = sys.executable
run, modelpy, params, desc = Path(sys.argv[1]).resolve(), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
ent = run/"entry"
if ent.exists(): shutil.rmtree(ent)
ent.mkdir(parents=True)
shutil.copy(HE/"entry_src/predict.py", ent/"predict.py")
shutil.copy(modelpy, ent/"model.py"); shutil.copy(params, ent/"params.json")
(ent/"reproduce.json").write_text(json.dumps({"image":"python","command":["python","predict.py","--data","/data","--out","/output/submission.csv"]}, indent=2)+"\n")
meta = json.loads((KIT/"public/metadata.json").read_text())
(ent/"README.md").write_text(f"""# House Edge entry ({meta['round']} round)

- Round: {meta['round']}; season_id: {meta['season_id']}
- Environment: `python` image (Python 3.13), **standard library only**; no requirements. Tested locally with CPython 3.12/3.13 on macOS.
- Hardware: any CPU; runs in well under a minute, < 200 MB RAM. No GPU.
- Seeds: none needed (fully deterministic, no randomness; iteration order is sorted).

## Method
{desc}

## Recreate the forecast
Run from this folder:
```sh
python predict.py --data /data --out /output/submission.csv
```
`--data` is the round's `public/` folder (reads `test/queries.csv` and `test/history.csv`).
Saved settings: `params.json` (fitted/tuned offline on organizer training+validation data). Model code: `model.py`.
Optional: `--split validation` writes validation predictions for local scoring.
For a query at cutoff c only ticks < c of machines in the same episode are used; condition labels are never used.
""")
fresh = run/"fresh"; 
if fresh.exists(): shutil.rmtree(fresh)
shutil.copytree(ent, fresh/"code")
py313 = subprocess.run(["uv","python","find","3.13"],capture_output=True,text=True).stdout.strip() or PY
for split, name in (("test","submission.csv"),("validation","validation_predictions.csv")):
    subprocess.run([py313, "-I", str(fresh/"code/predict.py"), "--data", str(KIT/"public"), "--out", str(run/name), "--split", split], check=True, cwd=fresh/"code")
shutil.rmtree(fresh/"code/__pycache__", ignore_errors=True)
# reproduce check: run again from entry dir, compare bytes
subprocess.run([py313, "-I", str(ent/"predict.py"), "--data", str(KIT/"public"), "--out", str(fresh/"again.csv")], check=True, cwd=ent)
shutil.rmtree(ent/"__pycache__", ignore_errors=True)
same = (fresh/"again.csv").read_bytes() == (run/"submission.csv").read_bytes()
print("py:", py313, "reproduce identical:", same)
r1 = subprocess.run([PY, "check_submission.py", str(run/"submission.csv")], cwd=KIT, capture_output=True, text=True); print(r1.stdout[-600:], r1.stderr[-300:])
r2 = subprocess.run([PY, "check_submission.py", str(run/"validation_predictions.csv"), "--split", "validation"], cwd=KIT, capture_output=True, text=True); print(r2.stdout[-900:], r2.stderr[-300:])
z = run/"entry.zip"
if z.exists(): z.unlink()
r3 = subprocess.run([PY, "build_entry.py", "--source", str(ent), "--out", str(z)], cwd=KIT, capture_output=True, text=True); print(r3.stdout[-400:], r3.stderr[-300:])

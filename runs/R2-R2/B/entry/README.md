# House Edge entry (adaptation round) - Team SSD

- Round: adaptation; season_id: 5fffdc4d0c700bbf61c3f911
- Environment: `python` image (Python 3.13), **standard library only**; no requirements.txt needed.
- Hardware: any CPU; uses up to 2 processes (`--workers 2`, default), < 2 GB RAM. No GPU.
- Seeds: none needed (deterministic, no randomness; sorted iteration order; identical output for 1 or 2 workers).

## Method
Generalized structured model: win logit = mu_type + d + b*x + delta*z; x from a machine-specific rule hypothesis (268 candidates: XOR / AND / OR lag pairs, single lags, win counts, 3-lag parity, cross-machine lags, none); population priors over rule and size-state hypotheses fitted by EM on organizer train+validation data; Bayesian grid filter per machine.

## Recreate the forecast
```sh
python predict.py --data /data --out /output/submission.csv
```
`--data` is the round's `public/` folder (reads `test/queries.csv`, `test/history.csv`, `test/sample_submission.csv` header).
Saved settings: `params*.json` (fitted offline on organizer training + validation data of this round). Code: `model*.py`, `kitschema.py`.
Optional: `--split validation` writes validation predictions for local scoring; `--workers 1` for a single process.
Safety: any query the main model cannot score gets the bundled pooled fallback (`model_p.py`, `params_p.json`); a note is printed.
For a query at cutoff c only ticks < c of machines in the same episode are used; condition labels are never used.
Main-round data is not used by this entry.

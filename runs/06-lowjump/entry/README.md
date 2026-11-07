# House Edge entry (main round)

- Round: main; season_id: 5fffdc4d0c700bbf61c3f911
- Environment: `python` image (Python 3.13), **standard library only**; no requirements. Tested locally with CPython 3.12/3.13 on macOS.
- Hardware: any CPU; runs in well under a minute, < 200 MB RAM. No GPU.
- Seeds: none needed (fully deterministic, no randomness; iteration order is sorted).

## Method
Bayesian per-machine model of the slot simulator structure observed in the training data. Win: logit P(win) = mu_type + d + b*x + delta*z (+ shared episode shift), x = XOR of win at lags (a,b), pair in {(1,3),(2,5),(4,7)}; (b,d) grid filter with diffusion + jumps (drift / shared changes); pairs weighted by evidence. Size: two size regimes per type; regime z_t = [payout at lag a >= 2] xor polarity (a = first lag of the machine's pair), Bayesian over polarity/noise. Episode shift: one logit offset fitted to recent outcomes of the other machines in the same episode (ticks < cutoff only). Filter dynamics (qb, qd, jump, etau) chosen by a small random search on chronological development splits of the training episodes. See model.py docstring.

## Recreate the forecast
Run from this folder:
```sh
python predict.py --data /data --out /output/submission.csv
```
`--data` is the round's `public/` folder (reads `test/queries.csv` and `test/history.csv`).
Saved settings: `params.json` (fitted/tuned offline on organizer training+validation data). Model code: `model.py`.
Optional: `--split validation` writes validation predictions for local scoring.
For a query at cutoff c only ticks < c of machines in the same episode are used; condition labels are never used.

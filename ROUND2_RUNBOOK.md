# House Edge round 2: release-day runbook (Team SSD)

Goal: a valid upload in under 10 minutes after the kit is released, and the best entry in about 20 minutes.
Everything below is rehearsed. Do not change code during the window.

## 0. Before the release (T-15 min)
1. Plug in the laptop. Close other heavy jobs (`uptime`: load should be < 4).
2. `cd ~/Documents/Academia/Projects/Datathon27/house-edge`
3. Check that Python works: `.venv/bin/python -c "import numpy, scipy, lightgbm; print('ok')"` and `uv python find 3.13`.
4. Clear old Desktop output: `rm -rf ~/Desktop/SSD-HouseEdge-R2`.
5. Open the Google Form for round 2 in the browser. Choose the team in the list.

## 1. At the release (T = 0)
1. Download the round-2 kit. Unzip it into a NEW folder: `house-edge/_r2kit/` (it must contain `public/` or be the `public/` folder itself).
2. Run ONE command:

```sh
.venv/bin/python src/round2_run.py _r2kit --tag R2 --fast | tee _r2kit.log
```

3. Watch the log. Each line has the clock time and the seconds since start. The important lines:

| line | meaning | expected time (rehearsed) |
|---|---|---|
| `schema: ...` / `schema test: ...` | what the kit contains (payout columns, machines per episode, cutoffs, types) | 1 s |
| `DESKTOP SAFE <- P` | pooled fallback entry is valid and on the Desktop | < 10 s |
| `DESKTOP SAFE <- B` | structured model B is valid and on the Desktop (replaces P) | 2-4 min |
| `DESKTOP FINAL <- <model>` | the picked entry is valid and on the Desktop | 3-8 min |
| `DONE` | finished | 3-8 min |

4. Upload as soon as `DESKTOP SAFE <- B` appears (a valid upload early is insurance; the latest Form response counts):
   - CSV: `~/Desktop/SSD-HouseEdge-R2/SAFE/submission.csv`
   - ZIP: `~/Desktop/SSD-HouseEdge-R2/SAFE/entry.zip`
5. When `DONE` appears, upload again from `~/Desktop/SSD-HouseEdge-R2/FINAL/` (if it differs from SAFE: compare `INFO.txt`).

## 2. If there are more than 25 minutes left after `DONE`
Run the full mode (reuses the fitted models; adds the filter-rate retune Bt and more dev pseudo-queries; about 10-15 min):

```sh
.venv/bin/python src/round2_run.py _r2kit --tag R2 | tee _r2kit_full.log
```

Upload from `FINAL/` again only if it finishes with `DONE` at least 5 minutes before the deadline. If not, keep the fast upload.

## 3. Warning flags (printed on the `flags:` line and as `FLAG ...` lines)

| flag | meaning | action |
|---|---|---|
| `NEW_TEST_TYPES [...]` | test has machine types not in train/validation | none: they use the pooled fallback automatically |
| `MISSING validation/targets.csv` / `NO_VALIDATION_TARGETS` | no validation answers: B fits its tables from histories | none; the pick uses the dev split only |
| `NO_TARGETS_FIT` | B fitted without targets (less exact) | none; the pick decides between B, C and the blend |
| `NO_DOMINANT_RULE_FAMILY` | B's rule families may not contain the new rule | trust the pick (it falls to C or a blend if B is worse on dev) |
| `NONE_MASS=...` | many machines look structureless | as above; C or a blend usually wins |
| `SIZE_TABLES_MISFIT` | more than 2 payout-size states | none; B uses mixtures of the 2 extreme tables |
| `C_BEATS_B_ON_DEV` | trees beat the structured model on the dev split | none; the picker already accounts for it |
| `B_C_DISAGREE_TEST` | B and C differ a lot on test | check that the pick is a blend/gate or C; if PICK = B, still keep it |
| `VALIDATION_DISAGREES_WITH_DEV_PICK` | validation points prefer another model by > 1 point | B and C are fitted with validation data, so validation is optimistic for them. Keep the pick unless the gap is > 2 points; then package the validation winner (section 4) |
| `POOLED_P_BEATS_PICK_ON_DEV` | something is badly wrong with all models | upload SAFE = P (`--package P --desk FINAL`) |
| `SLOW_ENTRY` | packaged entry took > 10 min here | still valid (limit 30 min, 2 cores); prefer B or C over gate/blend if time is short |
| `WARN step failed (continuing)` on `kit checker` | the organizer checker rejected our CSV | read the message. If it is a format problem, our validator is wrong: upload SAFE/P and tell the team. If it says "requires supported round data" on a synthetic kit, ignore |
| `PACKAGE <m> NOT OK` / `PACKAGE <m> FAILED` | that entry did not pass rerun/validation | it is not copied to the Desktop; use the previous SAFE |

## 4. Manual fallback (only if the runner stops with an error)
Each step is independent and uses the cached fits in `runs/R2-R2/`:

```sh
.venv/bin/python src/round2_run.py _r2kit --tag R2 --package B --desk FINAL      # structured model
.venv/bin/python src/round2_run.py _r2kit --tag R2 --package C --desk FINAL      # trees
.venv/bin/python src/round2_run.py _r2kit --tag R2 --package P --desk FINAL      # pooled fallback (always works)
```

Order of preference if the runner fails: B > C > P. Package `blend` or `gate` only if the runner printed them as PICK.
Each `--package` run prints `byte-identical=True`, `own validator: VALID`, and the zip check. Exit code 0 = OK.

## 5. Where the files are
- Desktop: `~/Desktop/SSD-HouseEdge-R2/SAFE/` and `.../FINAL/`: `submission.csv`, `entry.zip`, `INFO.txt` (model, time, SHA-256, rows).
- Runs: `runs/R2-R2/<model>/` (`submission.csv`, `validation_predictions.csv`, `entry/`, `entry.zip`, `fresh/`), `runs/R2-R2/summary.json`.

## 6. Final checklist before the last upload
- [ ] `INFO.txt` rows = number of test queries (`schema test: N queries` line).
- [ ] Log line `own validator: VALID` for the uploaded model, and `kit checker (test): VALID FORMAT` (real kit).
- [ ] `byte-identical=True` (fresh-folder Python 3.13 run vs. 1-worker rerun).
- [ ] `zip ... root README/reproduce ok=True`; ZIP < 100 MB, CSV < 10 MB.
- [ ] Test runtime on the `package` line well under 30 min.
- [ ] Both files attached in the Form, team chosen, Submit clicked, confirmation seen. Keep the exact files.

## 7. What the entry does (for the Form / judges)
- `predict.py --data /data --out /output/submission.csv`; Python 3.13 standard library only; deterministic (no seeds).
- Reads `test/queries.csv`, `test/history.csv`, and the payout columns from `test/sample_submission.csv`.
- Uses only ticks before each cutoff, and only machines of the same episode. No condition labels.
- Any query the main model cannot score gets the bundled pooled fallback (a note is printed).

# House Edge experiments (main round)

Validation technical points (max 60) and per-condition loss ratios vs frozen reference (lower is better).

| # | Run | Change | Points | stable | short | drift | shared | Kept |
|---|-----|--------|--------|--------|-------|-------|--------|------|
| 0 | 00-starter | pooled starter (reference) | 30.000 | 1 | 1 | 1 | 1 | baseline |
| 0b | 00-starter/comparison | pooled_recent | 30.826 | | | | | baseline |
| 1 | (scratch) | per-machine logistic on lag-win indicators (additive) | ~no gain vs EW win rate | | | | | no |
| 2 | 01-xor-hl16 | **Key finding:** each machine's win prob is driven by XOR of win(lag a), win(lag b) with pairs (1,3),(2,5),(4,7); polarity flips under drift/shared change. Track forgotten per-cell win rates (hl=16), Bayesian avg over 3 pairs; sizes from type prior + machine counts | 53.538 | 0.098 | 0.146 | 0.123 | 0.116 | yes (entry saved 12:10) |
| 2a | grid | same with hl=40 / 80 / 8 / 12 / 24 | 45.65 / 39.91 / 52.53 / 53.30 / 51.87 | | | | | hl 12–16 best |
| 3 | (scratch) size 2-state HMM / pi-grid | sizes per type are exactly 2 fixed distributions (A small, B big); HMM sticky state no help; machine pi-grid size-only err 9.8 vs 11 | — | | | | | partial |
| 4 | (scratch) gridnp.py numpy (b,d) grid filter per pair + jump | 54.955 | 0.086 | 0.108 | 0.086 | 0.087 | superseded |
| 5 | 02-xor-sizerule (src/puremodel.py) | **Key finding 2:** size regime z_t = [payout_{t-k}>=2] xor polarity, k in {1,2,4} per machine; z also adds ~+0.9 to win logit. Bayesian size rule + win grid filter with z-boost. Pure stdlib, ~45s CPU / 768 queries | **56.715** | 0.043 | 0.099 | 0.042 | 0.047 | yes |
| 6 | (pure) dynamics tune qb=.01 qd=.003 jump=.0025 | | 57.023 | 0.037 | 0.095 | 0.036 | 0.041 | yes |
| 7 | 03-tied | **Key finding 3:** size lag k == first lag of the machine's win pair ((1,3)->1,(2,5)->2,(4,7)->4); tie size rule to pair | **57.820** | 0.025 | 0.066 | 0.030 | 0.030 | yes (entry saved, zip clean) |
| 8 | grid | signal term per type / mu shifts / evt / epss / delta / bmag | flat (57.77-57.82) | | | | | no |
| 9 | (scratch) residual analysis | true-vs-pred win logit residual std ~0.55; correlates 0.5 within episode, with others' recent wins (lag 3 strongest); no specific neighbor coupling; own fast offset no help | | | | | | insight |
| 10 | 04-epoffset | + shared episode logit shift fitted (LOO, hl=4, tau=0.3) to other machines' recent outcomes | **57.913** | 0.022 | 0.066 | 0.028 | 0.028 | yes (entry saved, zip clean) |
| 11 | grid | temperature by history length, own-machine fast offset | flat/worse | | | | | no |
| — | train Brier (labels, 768) | 02: 0.3592, 03: 0.3537, 04: 0.3535 (lower better) -> improvements generalize to train | | | | | | check |
| 12 | grid | size polarity flips (zflip 0.002/0.01) | 57.69 / 57.07 | | | | | no (polarity static) |
| 13 | 05-eps | size rule near-deterministic: eps hypotheses {0.001,0.01,0.05} | **58.159** | 0.016 | 0.060 | 0.025 | 0.025 | yes (entry saved, zip clean) |
| 14 | grid | mu shifts, dsd 0.4/0.9, delta 0.7/1.1 on 05 | 58.08-58.19 (noise) | | | | | no (kept 05) |
| — | fresh-folder check of runs/05-eps/entry.zip | Python 3.13 -I, 54 s wall for test, output byte-identical to submission.csv | | | | | | ok |

## Status (checkpoint)
- CURRENT BEST: `runs/05-eps/` (58.16). Older: `runs/04-epoffset/` (57.91), Older: `runs/03-tied/` (57.82), Older: `runs/02-xor-sizerule/` (56.71), (entry/ folder, entry.zip, submission.csv, validation_predictions.csv, params.json). Previous fallback: `runs/01-xor-hl16/` (53.54).
- Model code: `src/puremodel.py` (pure stdlib, this is what ships as entry/model.py). Prototypes: src/combo.py (numpy). Build: `.venv/bin/python src/make_entry.py <run> src/puremodel.py <params.json> "<desc>"`.
- Remaining error: win part ~6-10 (x1e-3 SSE), size part ~2 for long histories, ~7.5 for short_history.
- Next to try: tune mu/delta/bmag per type; short-history priors; episode-level shared-change pooling; pair prior / size-rule eps; check runtime.

## Final main-round pick (13:10)
- `runs/05-eps/submission.csv` + `runs/05-eps/entry.zip` (validation 58.159 / 60). Checker: VALID 1536 test queries. Zip contains only README.md, reproduce.json, predict.py, model.py, params.json (grep for AI/tool names: clean).
- Remaining error is mostly the win probability (per-tick hidden logit noise, std ~0.5); size is nearly exact for long histories.

## Round 2 plan (data at 9 PM, "changed temporal relationships")
1. Run runs/05-eps model unchanged on new validation (`--split validation`) for a baseline.
2. Re-run the structure probes on new train: best XOR pair per machine (src/explore5.py), size-rule lag (src/explore18/19.py), size-state distributions from targets (src/explore8.py), tie between pair and size lag (src/explore20.py). Expect the pair set / size lags / maybe the XOR rule to change.
3. Make PAIRS, zlags, SIZE_STATES, mu, delta into params.json (currently constants in model.py) and refit them from new data; if the pair set is unknown, use all 28 pairs with an evidence prior.
4. Retune qb/qd/jump/eps on new validation; check train Brier (src/train_brier.py) to avoid overfitting.
5. Package with src/make_entry.py into runs/R2-xx; /main_data not needed.

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

## Phase B (15:25-) improvement study
- Fallback stays `runs/05-eps/` (untouched). Baseline code snapshot: `src/puremodel_base.py`.
- Dev harness: `src/dev.py` (train episodes only; chronological pseudo-queries at c-j, j=0,2..14, observed outcomes; Brier).
  Split stratified by condition: confirm = every 3rd train episode per condition (8/cond, untouched), dev = 16/cond.
  Limitation: train has no true probabilities, so dev/confirm use observed-outcome Brier (proper, noisy); official validation (true probs) already used for tuning.
- Audit (validation, 05): SSE x1e-3 stable 3.2, drift 4.7, shared 5.2, short 10.7; jackpot worst ratio (short jackpot 0.124); outcome=5 errors largest.
- Availability: for cutoff c only ticks < c of the same machine/episode; short histories = 40 consecutive recent + 24 sampled older ticks (older ones mostly lack the lag pairs, so the filter skips them).

| exp | change | dev dBrier x1e-3 macro [90% CI] | short | verdict |
|---|---|---|---|---|
| B1 useold | use older ticks with lag-pair unknown (x marginalised) | +0.66 [+0.03,+1.24] | +2.67 worse | drop |
| B2 epshort | slower episode offset for short histories (hl16,tau.5) | -0.08 [-0.35,+0.17] | -0.32 (6/16 eps better) | no consistent gain |
| B3 dsd0.4 | tighter offset prior (type-level shrinkage) | -0.03 [-0.18,+0.13] | -0.05 | no gain |
- Phase-2 verdict: existing priors already implement type-level shrinkage (mu_type, d~N(0,dsd), b-magnitude prior, uniform pair/polarity priors) and own evidence dominates as history grows; 3 trials gave no consistent gain -> family stopped.
| C1 calib | residual logistic on P(win): intercept+slope (offset = base logit + episode shift), fit on dev pseudo-queries (observed wins), 4-fold episode CV | CV -0.10 [-0.62,+0.37]; confirm -0.00; val 58.105 | | no gain |
| C2 linear3 | + fast8, slow, surprise8 (lam 10) | CV -0.13 [-0.68,+0.40]; confirm -0.34 [-0.94,+0.29]; val 58.194 | | CI spans 0 |
| C3 linear7 | + ep_off, short, pair_unc, signal (lam 10 / 100) | CV -0.25/-0.24 (CIs span 0, shared/short worse on CV); confirm -0.30/-0.17; val 58.19/58.17 | | CI spans 0 |
- Phase-3 verdict: learned corrections are tiny (coeffs <0.15), signs only mildly consistent, no CI excludes 0 and validation moves <=0.04 pts -> not promoted. Baseline is already close to calibrated.
| D1 ff1 | slow+fast filter (fast: qb .04, jump .01, qd .01) blended by recent win log-lik (hl 8, prior .2) | dev +0.56 [+0.37,+0.76] worse in all 4 conds; val 58.055 | | drop |
| D2 ff2 | same, hl 16, prior .1 | dev +0.28 [+0.17,+0.38] worse; val 58.110 | | drop |
- Phase-4 verdict: the single filter with jumps already adapts; a fast filter adds variance even on drift/shared -> family stopped. (Runtime would also double.)
| E rs0-7 | bounded random search on DEV only (seed 0, 8 draws): qb [.005,.02], jump [.001,.005], qd [.001,.006], etau [.2,.45] (log-uniform except etau) | best rs6 (qb .0096, jump .0012, qd .0022, etau .353): dev -0.475 [-0.71,-0.25], all 4 conds better (12/8/11/12 of 16 eps); higher jump/qb draws were worse | | select rs6 |
| E confirm | rs6 on untouched confirm split | -0.269 [-0.52,-0.03]; stable +0.10 [-0.14,+0.38] (3/8), short -0.61 (6/8), drift -0.53 (7/8), shared -0.04 (3/8) | | pass |
| E val | rs6 on official validation (true probs, not used for selection) | **58.206** vs 58.159; paired dSSE x1e-3 -0.162 [-0.246,-0.078]; eps better: stable 17/24, short 14/24, drift 16/24, shared 19/24 | | pass |
- PROMOTED: `runs/06-lowjump/` = lower jump/diffusion rates (filters were adapting too eagerly). Fresh-folder run from zip: 53.6 s, byte-identical; checker VALID; zip clean. Shipped model.py with 05 params reproduces 05 test CSV byte-for-byte.
- Not promoted: short-history shrinkage, learned residual correction, fast/slow blend (see rows above).
- CURRENT BEST (Phase B): `runs/06-lowjump/` (58.206). Fallback: `runs/05-eps/` (58.159).

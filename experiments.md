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

## Round-2 prep (17:06-) — structure-agnostic models + one-command runner
Goal: be ready for the adaptation kit (changed temporal patterns). All numbers = main-kit validation points /60 (true probs).
- **A** = `runs/06-lowjump` (hard-coded pairs (1,3),(2,5),(4,7), size lag tied, fixed tables): **58.206** | stable .0157 short .0599 drift .0236 shared .0241.
- **B** `src/model_b.py` + `src/fit_b.py` (fit 3 s): hypotheses = 28 XOR pairs + 8 single lags + none; size lag k in 0..8 (0 = constant state).
  Fit from kit data only: 2 size tables/type by 2-means on validation targets' p_i/(1-p0); joint prior pi(h,k) by EM over 1536 train+val
  machines (windowed beta-binomial win evidence + size log-lik); mu/delta/bmag by sign-mixture regression on validation targets.
  Runtime: per query, top-K=4 hypotheses (prior >= .004, windowed BB screen) run the (b,d) grid filter; evidence on common tick set.
  - On main kit EM recovers exactly (1,3),(2,5),(4,7) (prior .33 each, rest <.001) and the tie k|pair = first lag (0.98-1.00); tables exact (max dev 0).
  - fitted mu {st .25, ba -1.03, ja -2.45} delta .77 bmag 2.42 (A: .33/-.94/-2.5, .9, 2.4).
  - signal term gam from the regression (+0.38) HURT: 57.774 -> gam=0: **58.182** | .0167 .0621 .0228 .0233 (gap to A -0.024). gam now forced 0.
  - plugging A's mu/delta/bmag into B: 58.13-58.15 (fitted constants are slightly better); own-tick evidence (evcommon=false) 58.185 (same).
  - **told nothing** (`--naive`: uniform prior over all 37 h, uniform k, no tie): **56.621** | .0297 .1238 .0435 .0417 — degrades mainly on short history.
  - single core py3.13, 1536 test queries: 24-27 s fitted, 40 s naive.
  - **Bt** = B with filter rates retuned by 8-draw random search on train dev split (kept only if confirm also better): qb .0089 jump .00053 qd .00217 etau .383; dev -0.24, confirm -0.13 (x1e-3 Brier); val 58.196.
- **C** `src/model_c.py` + `src/fit_c.py`: per-tick examples from train+val histories (271k, past-only features): type, signal, EW win rates (hl 4/8/32),
  per-machine adaptive lag-agreement x_h*r_h for 36 lag hypotheses at two memories (hl 40 and 20), strongest-hypothesis x type, size-state
  agreement terms k=1..8, episode others' last-tick surprise.
  - multinomial LR (L2 1e-4), one memory (hl 40): 46.06 (drift .57, shared .41); + fast memory hl 10: 53.24; hl 20: 53.84.
  - **LightGBM** (rounds by early stopping on an episode-grouped train holdout; trees exported to JSON, pure-Python inference, 3 s): **55.912** | .0587 .0924 .0792 .0622.
- D starter pooled = reference: 30.00.
- Blend chosen on train dev split (never on validation): A+B 0.3/0.7 -> 58.199; with Bt: 0.1 A + 0.9 Bt -> 58.202. B+C 50/50 = 57.73 (C does not help on main).
- **Synthetic changed-structure kit** (`src/synth_kit.py`: pairs (3,6),(1,8),(2,4), size lag untied uniform 1..8, drift/shared/short like main), points /40:
  A 21.55 | **B 37.68** | Bn 36.99 | C 36.49 | D 21.47. fit_b recovers the 3 new pairs (.32-.35 each) and an untied k (≈uniform). Dev blend picks B alone.
- **Runner** `src/round2_run.py <kit> --tag <t>` (~4 min main kit, 6 procs; restartable, cached per step): fits B/Bn/C/C_dev, starter, predicts val+test,
  fast + official scores, dev/confirm Brier, B retune (Bt), dev-chosen blend -> runs/R2-<t>/<model>/. `--package <A|B|Bt|C|blend>` builds entry +
  zip, fresh-folder py3.13 -I rerun byte-identical: verified B (24 s), C (3 s, 0.96 MB), blend A+B (77 s).

## Round-2 robustness (17:50-) — synthetic suite + generalized B
Suite: `src/synth_kit.py <dir> S1..S8` (main-sized kits, 4 conditions, fixed seeds; crude pooled reference, so only compare models within a kit),
`src/suite.py <tag> S1 .. main [--from <tag>] [--table]`. Points are /40 (main /60).
| kit | change |
|---|---|
| S1 | new XOR pairs (3,6),(1,8),(2,4) + untied size lag |
| S2 | AND / OR of two lags |
| S3 | single lag, or momentum count of wins in last 3/5/8 |
| S4 | 3-lag parity |
| S5 | cross-machine: partner machine's win at lag 1-4 |
| S6 | 3 size states (sum of two size lags), size lag != win lag |
| S7 | informative signal (win logit +0.8*signal; size state forced on signal=+1) |
| S8 | no structure: drifting win rate, one size table |
Changes to B (all fitted per kit, nothing hard-coded):
- rule families: XOR pairs, single lags, none, AND/OR/A&~B/~A&B of every lag pair, win-count windows 2..8, 3-lag parity (56), cross-machine
  (other machine at relative index r=1..7, lag 1..8) = 268 hypotheses; population prior by EM; per query windowed BB screen keeps top 4 for the filter.
- size states: K in 2..4 chosen from validation targets (farthest-point k-means); the two extreme clusters are tables A/B, intermediate states =
  mixtures; size-state driver = one lag k in 0..8 or the mean of two lags (28) -> 37 size hypotheses with a joint prior over (rule, size driver).
- main kit: 58.181 (268 rules) / 58.177 (+ size generalization) vs 58.182 before; runtime single core 59 s fitted, 153 s told-nothing (machine loaded).
C: + per-pair cell win-rate deviations (any 2-lag boolean rule), cross-machine agreement, count windows (main 55.87 vs 55.91).
Runner: diagnostics block (rule-family mass, top hypotheses, size K / mixture misfit, per-condition ratios), flags, conservative PICK =
first of B, Bt, blend, C, A within 0.3e-3 dev+confirm Brier of the best.
- Bt search now also tries two fixed fast-drift settings (qd .015/.04, wider d grid): S8 Bt 32.09 -> 34.02; on main they lose on dev (not picked).
- Bugs found by the suite and fixed: naive prior below pmin (no candidates), single size state (A=B -> NaN). 
**Suite v2 (final code; validation points, /40 synthetic, /60 main; pick = runner's conservative dev+confirm choice):**
| kit | A | B0 (old family) | Bn (told nothing) | B | Bt | C | D | dev blend | pick | pick val | best |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 xor pairs | 21.50 | 37.40 | 36.28 | 37.41 | 37.38 | 36.13 | 21.61 | 37.36 | B | 37.41 | B |
| S2 and/or | 24.69 | 30.00 | 35.86 | 36.69 | - | 33.23 | 23.88 | 36.69 | B | 36.69 | B |
| S3 lag/count | 24.80 | 33.83 | 33.61 | 35.04 | - | 33.62 | 23.43 | 35.09 | blend B.7/C.3 | 35.09 | blend |
| S4 3-parity | 21.01 | 21.79 | 36.49 | 36.87 | 36.88 | 22.28 | 21.12 | 36.88 | B | 36.87 | Bt |
| S5 cross-machine | 21.20 | 22.46 | 35.83 | 36.96 | - | 27.72 | 21.72 | 36.96 | B | 36.96 | B |
| S6 3 size states | 36.20 | 37.39 | 36.62 | 37.54 | - | 36.85 | 21.20 | 37.57 | blend B.8/C.2 | 37.57 | blend |
| S7 signal | 36.89 | 38.14 | 36.85 | 38.14 | 38.15 | 37.01 | 22.41 | 38.12 | blend Bt.8/C.2 | 38.12 | Bt |
| S8 no structure | 28.88 | 31.27 | 31.26 | 32.09 | 34.02 | 34.47 | 31.03 | 34.50 | blend Bt.2/C.8 | 34.50 | blend |
| main | 58.21 | 58.18 | 55.23 | 58.18 | 58.19 | 55.87 | 30.83 | 58.20 | B | 58.18 | A |
- Pick is within 0.03 pts of the best model on every kit. "-" = retune not kept (failed dev+confirm). S8 flags: NONE_MASS .88, C_BEATS_B_ON_DEV.
- Runtime: packaged B (final code) on main test 184 s single core under load 80+ (byte-identical, 58.177); told-nothing 153 s; full runner on main 12 min under load.

## Round-2 release-window readiness (22:00-) — schema robustness, staged runner, gate
Kit released ~1 h before the deadline -> everything pre-built; one command (`ROUND2_RUNBOOK.md`).
- **Schema**: `src/kitschema.py` (stdlib, shipped in every entry). Payout set = `p_*` columns of sample_submission (any count/values),
  payouts handled as indices (0 = loss, >= 2 = big), unknown history values -> nearest payout; missing signal -> 0, any numeric signal;
  missing machine_type -> `_pool`; unknown test types -> pooled tables/mu in B (`_pool`), all-zero type in C, pooled freq in P.
  fit_b: types with >= 30 validation targets get own tables/mu, rest + unknown -> `_pool`; no validation targets -> tables from histories
  (split by previous big payout, then EM) and mu from observed win rates. Main kit unchanged: **B 58.177, C 55.869** (same as before).
- **P** (`src/model_p.py`): pooled fallback, own EW counts shrunk to type frequencies, (hl, k) by validation grid: main 30.85 (starter 30.83).
  Bundled in every entry: queries the main model cannot score (exception / bad row) get P; tested with a deliberately broken B params -> valid CSV.
- **Runner** staged: P packaged + verified -> Desktop SAFE (6 s); B fitted, packaged -> SAFE; C/C_dev fit in background; dev split; blend; gate; pick -> FINAL.
  Packaging runs fresh-folder py3.13 -I test (2 workers), 1-worker rerun (must be byte-identical), validation in parallel; own validator mirrors
  checker rules (header set, every query once, finite, [0,1], sum 1e-6); kit checker soft; own zip fallback. `--fast` = no retune/A, 4 dev offsets.
- **Schema-change kits** (`src/synth_kit.py X1..X3`, all end-to-end, every package byte-identical + VALID; kit checker rejects them by design):
| kit | change | P | B | C | blend | pick (val) | wall (3 kits in parallel, load 15-30) |
|---|---|---|---|---|---|---|---|
| X1 | payouts 0/1/3/10/50, 10 machines/ep, signal {0,1,2}, 4 types + test-only "mystery" | 21.52 | 37.44 | 36.15 | 37.44 (B 1.0) | B 37.44 | SAFE-P 6 s, SAFE-B 369 s, FINAL 545 s |
| X2 | payouts 0/2/4/8, 6 machines/ep, renamed types, no signal column, cutoffs 300-480 | 21.42 | 37.05 | 36.09 | 36.87 | B 37.05 | 6 s / 341 s / 502 s |
| X3 | main schema, no targets/reference/metadata | - | (no val) | - | - | blend B.8/C.2 | 3 s / 356 s / 560 s |
- **Main fast rehearsal** (machine idle, 6 workers): SAFE-P 6 s, SAFE-B 99 s (package 46 s, test runtime 27 s on 2 workers), C fit 124 s (background),
  FINAL (= B, 58.177) at **186 s**.
- Gate (B vs C per query, w = sigmoid(theta . [top posterior, none posterior, evidence gap, log hist, |B-C|]), fit on dev only;
  eligible only if it also wins on confirm): main dev 367.42 vs B 367.50 but confirm 357.86 vs 357.44 -> not eligible; X1/X2/X3 same pattern
  (X2 dev -2.0, confirm +2.4 vs B) -> gate overfits the dev part; kept as guarded candidate only.

### Refocus (22:37): S1-type adaptation + neighbor machines (real main data)
- **Neighbor coupling, main data** (`src/neighbor.py`): B's one-step predictive P(win) at every history tick (2314 machines-with->=40 ticks
  of train+val, 1157 analysed), residual u = win - p; z for other machine j at lag k, vs a null with j from another episode.
  Machine-specific coupling (fixed partner, own sign): **absent**: pair z^2 by lag 0.94-1.01 (no excess), per-machine max|z| over 56 (j,k)
  quantiles 2.45/3.06/3.70 vs null 2.20/2.95/3.75. Common-sign effect: others' win at lag k raises own win, all lags +0.06..+0.08 (shared
  drift), **lag 3 spike +0.22** (episode count z +0.56 vs +0.09..+0.17 at other lags; null ~0).
  Value (`src/nbr_val.py`, validation true probs, B final logit + g * mean_j(win_j(c-3) - rate_j)): g .25 58.208, **g .5 58.216**, g .75 58.201,
  g 1 58.162 (B 58.177; same-type only 58.173, other-type 58.202, big payouts 58.189; lags 1,2,4,5 all <= 58.13). => +0.04 pts at best.
  Built into B as a per-kit learnable term (`nbk`, `nbg`; k 1..8, g grid on dev, kept only if confirm also improves).
  Main: dev picks the same lag 3 / g .5 (dev 367.50 -> 367.27) but confirm worsens (357.44 -> 357.90) -> not kept. Clear small-effect result.
- Rule vs type / signal / episode on main (`src/typecheck.py`, MAP among the 3 pairs): uniform over types and signals; within-episode same-rule
  rate 0.336 (1/3 = independent) -> no type- or episode-conditional prior to exploit.
- **Oracle prior gap** (`src/oracle_b.py`: true pair set uniform + true size-lag rule, rest fitted; validation points /40):
| kit | truth | B (EM-fitted) | oracle prior | short_history ratio B / oracle |
|---|---|---|---|---|
| S1 | (3,6),(1,8),(2,4), size lag uniform 1..8 | 37.408 | 37.409 | .1065 / .1073 |
| S1b | (1,3),(2,6),(5,8) (partial overlap with main), size lag = first lag | 37.895 | 37.902 | .0610 / .0603 |
| S1c | (2,7),(4,5),(1,6),(3,8), size lag = second lag | 37.743 | 37.752 | .0867 / .0871 |
  EM recovers the new pairs exactly (S1b 2,6 .357 / 1,3 .316 / 5,8 .315; S1c .23-.26 each; rest < .001). Discovery is at oracle level;
  short history is limited by identifying each machine's own pair/sign/offset from ~64 ticks, not by the population prior.
  Main-kit prior as an extra prior: unnecessary (new kit alone is at oracle).
- Short/long-history split blend weights (`src/shortblend.py`, dev-chosen, confirm Brier x1e3): main B 357.44 vs split 358.47 (w_short .6);
  S1 374.70 vs 374.94; S2 386.30 vs 387.06; S3 372.57 vs 371.78 (global blend 371.80); S4 equal. -> no short-history gain from mixing; dropped.
  Gate summary: never eligible (dev gain, confirm loss on main, S1, X1-X3); remains in the runner as a confirm-guarded candidate (R2_NOGATE=1 skips it).

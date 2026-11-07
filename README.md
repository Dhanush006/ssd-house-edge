# House Edge — Team SSD (TAMU Datathon 2026)

We forecast the next payout of simulated slot machines. For each machine, the model gives five probabilities: a payout of 0×, 1×, 2×, 5× or 20× the wager.

**Result:** 58.21 of 60 validation points. The starter baseline scores 30.83. The model takes 54 seconds on one CPU core and uses only the Python standard library.

The rest of this file explains the challenge, then our approach step by step, then the model, the evidence, and what did not work.

---

## 1. The challenge

### 1.1 The task

Each **query** names one machine and one **cutoff tick**. A tick is one time step, which is one spin. We must predict the payout at the cutoff tick, using only spins before it.

The answer is five probabilities in the order 0×, 1×, 2×, 5×, 20×. They must add up to 1.

### 1.2 The data

The data comes from a simulator. Machines are grouped in **episodes**. An episode is a set of 8 related machines observed up to the same cutoff.

| Split | Queries | Episodes | History rows | What it includes |
|---|---|---|---|---|
| train | 768 | 96 | 139,944 | Histories, the observed next payout, condition labels |
| validation | 768 | 96 | 141,392 | The same, plus the simulator's **true probabilities** |
| test | 1,536 | 192 | 282,664 | Histories only. The graders keep the answers. |

Each machine has a **type**: steady, balanced or jackpot. The test set has 498 steady, 549 balanced and 489 jackpot queries. Each query also has a **signal** value of −1, 0 or +1.

Across all training spins, the payout mix is 58.3% 0×, 19.3% 1×, 14.7% 2×, 6.4% 5× and 1.3% 20×.

What the files look like:

```text
queries.csv   query_id, episode_id, machine_id, machine_type, signal, cutoff
              main-validation-e00000-m000-next, main-validation-e00000, ...-m000, balanced, 0, 221

history.csv   episode_id, machine_id, tick, payout
              main-validation-e00000, ...-m000, 0, 20
              main-validation-e00000, ...-m000, 1, 2

targets.csv   (validation only) the true probabilities p_0 ... p_20 for each query
```

### 1.3 The four conditions

Each episode belongs to one **condition**. The score weights the four conditions equally.

| Condition | What happens in the simulator |
|---|---|
| stable | The machines keep the same behaviour. |
| short history | Only 40 recent consecutive ticks and 24 sampled older ticks are visible. |
| drift | A machine's behaviour changes over time. |
| shared change | Several machines in the episode change together. |

The condition label is available for train and validation only. The test model must work without it.

### 1.4 The score

1. For each query, add the five squared differences between our probabilities and the true probabilities.
2. Average this error within each episode, then across the episodes of each condition.
3. Divide each condition's error by the error of a fixed **reference model** (a pooled frequency model).
4. Average the four ratios. This average is **R**.
5. Points = 60 / (1 + R).

So R = 1 (same error as the reference) gives 30 points. R = 0 (perfect odds) gives 60 points. Our R is about 0.03.

The score compares against the **true** probabilities, not the random outcome. A lucky jackpot cannot help or hurt a forecast.

### 1.5 The rules that shaped the work

- For a cutoff c, use only ticks below c. This applies to other machines too.
- Use other machines only from the same episode.
- Do not use condition labels at test time.
- Keep whole episodes together in any data split.
- The graders rerun our code with no network access, two CPU cores and a 30-minute limit. The output must match our CSV to within 1e-8.

---

## 2. Our approach, step by step

The table lists every main step in order. "Points" are validation points out of 60.

| Step | What we did | Points | Decision |
|---|---|---|---|
| 0 | Ran the starter. Its best method, `pooled_recent`, weights recent spins more. | 30.83 | Baseline |
| 1 | Fit a per-machine logistic model on "won at lag k" features. | about 30 | Dropped. No gain over a recent win rate. |
| 2 | **Discovery 1.** Each machine's win chance depends on whether two specific past spins disagree (Section 3.1). | 53.54 | Kept. First saved entry. |
| 3 | Tested a hidden two-state model for payout size. | — | Partly useful. It showed that sizes come from exactly two fixed tables per type. |
| 4 | Replaced simple counts with a Bayesian grid filter that allows jumps. | 54.96 | Superseded |
| 5 | **Discovery 2.** The size table in use depends on one earlier payout (Section 3.2). | 56.72 | Kept |
| 6 | Tuned how fast the filter forgets. | 57.02 | Kept |
| 7 | **Discovery 3.** The size lag is always the first lag of the machine's win pair. | 57.82 | Kept |
| 8 | Added a shared shift for the whole episode, fitted to the other machines' recent results. | 57.91 | Kept |
| 9 | Made the size rule nearly deterministic (very small noise levels). | 58.16 | Kept. First upload. |
| 10 | Audited errors, built new validation splits, and ran a study of possible improvements (Section 5). | — | — |
| 11 | Lowered the jump and drift rates of the filter, after a random search on the new splits. | **58.21** | Kept. Final main-round entry. |

The big jump came from steps 2, 5 and 7. We stopped treating the data as noisy counts and recovered the rules of the simulator instead.

---

## 3. The model

The model is in `runs/06-lowjump/entry/model.py`. Its settings are in `params.json` in the same folder.

### 3.1 Discovery 1: the win rule

A spin is a **win** when the payout is above 0×. Each machine has one **lag pair** (a, b). A lag of k means the spin k ticks earlier. The pair is one of three: (1, 3), (2, 5) or (4, 7).

Define x = +1 if spins t−a and t−b disagree (one won and one lost). Otherwise x = −1. Then:

```text
logit P(win at t) = mu_type + d + b · x + delta · z
```

| Term | Meaning |
|---|---|
| mu_type | A base level for the machine type: steady +0.33, balanced −0.94, jackpot −2.50 |
| d | A small offset for this machine. It changes slowly over time. |
| b | The strength and direction of the lag-pair effect. It is specific to the machine. Its sign can flip. |
| delta · z | A boost of +0.9 when the machine is in its "big payout" state (Section 3.2) |

Two machines of the same type can react in opposite directions. This explains why the organizers warned that same-type machines can behave differently.

### 3.2 Discoveries 2 and 3: the payout-size rule

When a machine wins, the size (1×, 2×, 5× or 20×) comes from one of two fixed tables for its type. Table A has smaller payouts. Table B has bigger payouts.

The state z picks the table. z depends on one earlier payout: was the payout k ticks ago 2× or more? Each machine also has a fixed polarity that can invert this test. And k is always the first lag of the machine's win pair: pair (1, 3) uses k = 1, pair (2, 5) uses k = 2, pair (4, 7) uses k = 4.

The two tables per type, in the order 1×, 2×, 5×, 20×:

| Type | Table A (z = 0) | Table B (z = 1) |
|---|---|---|
| steady | 0.754, 0.197, 0.048, 0.001 | 0.455, 0.436, 0.107, 0.002 |
| balanced | 0.500, 0.348, 0.130, 0.022 | 0.214, 0.547, 0.205, 0.034 |
| jackpot | 0.335, 0.245, 0.315, 0.105 | 0.121, 0.324, 0.416, 0.139 |

### 3.3 How the model makes one forecast

1. **Read the history.** Take all spins of the machine before the cutoff.
2. **Fit the size rule.** Score each hypothesis (lag k, polarity, small noise level) against the past payouts. Keep the probability of each one.
3. **Run the win filter, once per lag pair.** The filter holds a probability grid over the unknowns b and d. At each past spin it does two things:
   - It spreads the grid a little, so that b and d can drift. With a small probability it also resets b, so that b can jump or flip.
   - It multiplies the grid by the likelihood of the observed payout, then normalizes.
4. **Weight the three lag pairs** by how well each one explains the history.
5. **Forecast the next spin.** Combine the win probability with the size tables for the expected state z.
6. **Apply the episode shift.** Fit one small logit shift to the recent results of the other 7 machines in the episode. This handles changes that hit several machines at once.
7. **Clip and normalize** so that the five probabilities are positive and add up to 1.

This is a **Bayesian filter**: a running estimate of hidden values that updates with each new observation and forgets old evidence at a controlled rate. That forgetting rate is the main lever for drift. It is the same idea as the constant step size used to track a moving target in reinforcement learning.

### 3.4 One worked example

Validation query `main-validation-e00000-m000-next`, a balanced machine at cutoff 221:

| | p_0 | p_1 | p_2 | p_5 | p_20 | Squared error |
|---|---|---|---|---|---|---|
| True odds | 0.260 | 0.370 | 0.257 | 0.097 | 0.016 | — |
| Our model | 0.142 | 0.423 | 0.302 | 0.113 | 0.019 | 0.019 |
| Reference model | 0.506 | 0.132 | 0.289 | 0.055 | 0.018 | 0.120 |

The reference model uses only long-run frequencies, so it expects a loss half of the time. Our model knows that this machine is in a winning phase of its lag rule, and its error is about six times smaller.

---

## 4. Results

### 4.1 Validation points

| Entry | Points | stable | short history | drift | shared change |
|---|---|---|---|---|---|
| Reference model | 30.00 | 1.000 | 1.000 | 1.000 | 1.000 |
| Starter `pooled_recent` | 30.83 | | | | |
| `runs/05-eps` (first upload) | 58.16 | 0.0163 | 0.0601 | 0.0249 | 0.0253 |
| `runs/06-lowjump` (final) | **58.21** | **0.0157** | **0.0599** | **0.0236** | **0.0241** |

Condition columns are error ratios against the reference. Lower is better.

### 4.2 Where the remaining error is

- **Short history** is the weakest condition, with a ratio of 0.060 against 0.016–0.024 for the others. With few spins, the filter cannot pin down the lag pair and its strength. The 24 older sampled ticks rarely have both lags of a pair, so they add little.
- **Jackpot machines** have the worst ratio. Their 5× and 20× outcomes are rare, so each one carries a lot of information.
- Most of the remaining error is in the **win probability**. The true win logit has random noise at each tick, with a standard deviation of about 0.5. No feature we tested predicts this noise. The size part is almost exact for long histories.

---

## 5. How we checked the final improvement

By step 9 we had used the official validation set many times. A further gain measured only on that set could be overfitting. So before we changed the model again, we built new checks.

### 5.1 New splits from the training data

The training data has observed outcomes but no true probabilities. So:

- We made **pseudo-queries** from the training histories at cutoffs c, c−2, …, c−14. Each one uses only earlier spins.
- We scored them with the **Brier score**: the squared error against the observed outcome. This is noisier than the official metric, but it is a fair test.
- We split the training episodes by condition: 16 per condition for development, 8 per condition kept untouched for a final **confirmation** check.

### 5.2 The improvement study

| Idea | What we tested | Result | Decision |
|---|---|---|---|
| Stronger shrinkage for short histories | Use the older sampled ticks; a slower episode shift for short histories; a tighter type-level prior | No consistent gain. The older ticks made it worse. | Stopped after 3 trials. The existing priors already shrink to the type level. |
| Learned correction to the win logit | A regularized linear model on fast and slow win rates, recent surprises, history length, lag-pair uncertainty and signal | Coefficients under 0.15. No confidence interval excluded zero. Validation moved by at most 0.04 points. | Not used. The model is already close to calibrated. |
| Fast and slow filter blend | A second, faster filter, blended by recent predictive accuracy | Worse in all four conditions on development data. It also doubled the runtime. | Dropped |
| Retune the filter's drift settings | Bounded random search, 8 draws, scored on development data only | Best draw: lower jump and drift rates | **Kept** |

### 5.3 Evidence for the final change

The final entry changes four settings and nothing else:

| Setting | Meaning | Before | After |
|---|---|---|---|
| jump | Chance per tick that b resets | 0.0025 | 0.0012 |
| qd | Drift rate of the offset d | 0.003 | 0.0022 |
| qb | Drift rate of the strength b | 0.010 | 0.0096 |
| etau | Prior width of the episode shift | 0.30 | 0.353 |

The old filter was adapting too eagerly. Draws with higher jump rates scored worse.

| Check | Change in error (×1e-3, lower is better) | 90% interval | Episodes better |
|---|---|---|---|
| Development split (Brier) | −0.48 | −0.71 to −0.25 | All four conditions improved, 8–12 of 16 per condition |
| Untouched confirmation split (Brier) | −0.27 | −0.52 to −0.03 | short 6/8, drift 7/8, shared 3/8, stable 3/8 |
| Official validation (true odds, not used to select) | −0.16 | −0.25 to −0.08 | stable 17/24, short 14/24, drift 16/24, shared 19/24 |

The gain is small but consistent across all three checks, and no single episode drives it. The one weak spot is stable episodes on the confirmation split (+0.10, with an interval that includes zero).

---

## 6. Submission and reproduction

The final main-round entry is in `runs/06-lowjump/entry/`:

| File | Purpose |
|---|---|
| `predict.py` | Entry point: `python predict.py --data <public folder> --out <csv>` |
| `model.py` | The model (standard library only, no randomness) |
| `params.json` | The fitted settings |
| `reproduce.json` | The command the graders run |
| `README.md` | Setup notes for the graders |

Checks done before upload:

- The official checker accepts the test CSV (1,536 queries).
- A run from a fresh folder with Python 3.13 in isolated mode takes 53.6 seconds and gives a byte-identical CSV.
- With the old settings, the same `model.py` reproduces the first upload byte for byte. The extra code paths in `model.py` are experiments that stay off unless `params.json` turns them on.

To rebuild the entry:

```bash
python src/make_entry.py runs/06-lowjump src/puremodel.py runs/06-lowjump/params.json "<description>"
```

---

## 7. Limitations and the plan for round 2

- **The structure is hard-coded.** The lag pairs, the size lags and the size tables are fixed constants. If the hidden test data uses different rules, these constants will not transfer. The main-round kit gives no sign of this.
- **The size tables come from validation targets.** This uses released data, which the rules allow.
- **Short histories stay weak.** The model needs more spins to identify each machine's rule.

Round 2 changes some of the timing patterns on purpose. Our plan:

1. Run the current model unchanged on the new validation data, to get a baseline.
2. Re-run the structure probes in `src/explore*.py` on the new training data: the lag pairs, the size lag, the two size tables and the link between them.
3. Move these constants into `params.json` and refit them. If the pair set is unclear, allow all 28 possible lag pairs with a prior.
4. Retune the filter settings and check them on the training Brier score.
5. Package and check the entry as before.

---

## 8. How to explore the work yourself

Put the organizer kit in `kit/` first.

```bash
cd kit
# score any validation predictions
python check_submission.py ../runs/06-lowjump/validation_predictions.csv --split validation
# check the test file format
python check_submission.py ../runs/06-lowjump/submission.csv
```

| Path | What it holds |
|---|---|
| `experiments.md` | Every experiment with its score and decision |
| `src/puremodel.py` | The model source (shipped as `entry/model.py`) |
| `src/explore*.py` | The probes that found the lag-pair and size rules |
| `src/dev.py` | The development and confirmation split harness |
| `src/make_entry.py` | Builds an entry folder and zip |
| `runs/05-eps/entry/` | The first upload (58.16), kept as the fallback |
| `runs/06-lowjump/entry/` | The final main-round entry (58.21) |

Good first steps for a review:

1. Open `kit/public/validation/history.csv` and pick one machine. Check its wins against the lag pair rule in Section 3.1.
2. Compare a few rows of `runs/06-lowjump/validation_predictions.csv` with `kit/public/validation/targets.csv`.
3. Read `experiments.md` from top to bottom. It follows Section 2.

---

## Glossary

| Term | Meaning |
|---|---|
| Tick | One time step: one spin of one machine |
| Cutoff | The tick to predict. Only earlier ticks may be used. |
| Episode | A group of 8 related machines with the same cutoff |
| Query | One request for five payout probabilities |
| Condition | The scenario of an episode: stable, short history, drift or shared change |
| Win | A spin with a payout above 0× |
| Lag k | The spin k ticks before the current one |
| Lag pair | The two lags (a, b) whose agreement drives a machine's win chance |
| Logit | log(p / (1 − p)). It turns a probability into an unbounded number. |
| Polarity | The direction of an effect for one machine. It can be inverted. |
| Bayesian filter | A running probability estimate of hidden values, updated at each observation |
| Jump rate | The chance per tick that the filter allows a sudden change |
| Brier score | The squared error between forecast probabilities and the observed outcome |
| Reference model | The organizers' fixed pooled frequency model. The score is relative to it. |
| R | The average of the four condition error ratios. Points = 60 / (1 + R). |

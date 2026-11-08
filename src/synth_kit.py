"""Synthetic round-2 stress-test kits (pipeline testing only; never used for fitting real entries).
usage: python src/synth_kit.py <out_dir> <scenario S1..S8> [seed]
Base family as on main (win logit = mu_type + d + b*x + delta*z + noise sd .5; 2 size tables per type), with one
change per scenario:
  S1 new XOR pairs (3,6),(1,8),(2,4); size lag k uniform 1..8 (untied)
  S2 x = AND or OR of two lags (random pair), size lag untied
  S3 half machines: single lag a; half: momentum count of wins in last k in {3,5,8} (sign of b = momentum/anti)
  S4 three-lag parity (XOR of 3 random lags <= 8)
  S5 cross-machine: x = win of partner machine (same episode) at lag 1..4
  S6 three size states: z = [y(t-k1)>=2] + [y(t-k2)>=2] (k1,k2 random), mid table = average of A,B; win XOR main pairs
  S7 signal informative: win logit += 0.8*signal; size state forced to B when signal = +1; main pairs, tied k
  S8 no structure: x = 0, d follows a random walk (sd .12/tick), one size table
Conditions: stable; short_history (40 recent + 24 sampled older); drift (b reverses over time for half the
machines; S8 has extra walk); shared_change (5 of 8 machines share a logit shift from a random tick)."""
import sys, csv, json, math, random
from pathlib import Path

MAIN_PAIRS = ((1, 3), (2, 5), (4, 7))
TABLES = {"steady": [[0.754, 0.197, 0.048, 0.001], [0.455, 0.436, 0.107, 0.002]],
          "balanced": [[0.5, 0.348, 0.13, 0.022], [0.214, 0.547, 0.205, 0.034]],
          "jackpot": [[0.335, 0.245, 0.315, 0.105], [0.121, 0.324, 0.416, 0.139]]}
MU = {"steady": 0.33, "balanced": -0.94, "jackpot": -2.5}; DELTA = 0.9
PAY = (0, 1, 2, 5, 20); CONDS = ("stable", "short_history", "drift", "shared_change")


def sig(x): return 1 / (1 + math.exp(-x))


def W(s, t):
    v = s.get(t)
    return None if v is None else v > 0


def machine_spec(rng, sc, m):
    a, b = rng.choice(MAIN_PAIRS); sp = dict(k=a, zeta=rng.randint(0, 1))
    if sc == "S1":
        a, b = rng.choice(((3, 6), (1, 8), (2, 4))); sp.update(kind="xor", lags=(a, b), k=rng.randint(1, 8))
    elif sc == "S2":
        a, b = sorted(rng.sample(range(1, 9), 2)); sp.update(kind=rng.choice(("and", "or")), lags=(a, b), k=rng.randint(1, 8))
    elif sc == "S3":
        if m % 2: sp.update(kind="lag", lags=(rng.randint(1, 8),), k=rng.randint(1, 8))
        else: sp.update(kind="count", lags=(rng.choice((3, 5, 8)),), k=rng.randint(1, 8))
    elif sc == "S4":
        sp.update(kind="xor3", lags=tuple(sorted(rng.sample(range(1, 9), 3))), k=rng.randint(1, 8))
    elif sc == "S5":
        sp.update(kind="cross", lags=(rng.randint(1, 4),), partner=(m + rng.randint(1, 7)) % 8, k=rng.randint(1, 8))
    elif sc == "S6":
        sp.update(kind="xor", lags=(a, b), k=rng.randint(1, 8), k2=rng.randint(1, 8))
    elif sc == "S7":
        sp.update(kind="xor", lags=(a, b))
    elif sc == "S8":
        sp.update(kind="none", lags=())
    else:
        sp.update(kind="xor", lags=(a, b))
    return sp


def xval(sp, s, t, eps_hist):
    L = sp["lags"]; kd = sp["kind"]
    if kd == "none": return 0.0
    if kd == "cross":
        v = W(eps_hist[sp["partner"]], t - L[0]); return 0.0 if v is None else (1.0 if v else -1.0)
    if kd == "count":
        v = [W(s, t - j) for j in range(1, L[0] + 1)]
        if None in v: return 0.0
        return (2.0 * sum(v) - L[0]) / L[0]
    v = [W(s, t - j) for j in L]
    if None in v: return 0.0
    if kd == "xor": r = v[0] != v[1]
    elif kd == "xor3": r = (v[0] ^ v[1] ^ v[2])
    elif kd == "and": r = v[0] and v[1]
    elif kd == "or": r = v[0] or v[1]
    elif kd == "lag": r = v[0]
    return 1.0 if r else -1.0


def episode(rng, sc, cond, c):
    shift_t = rng.randint(100, c - 15); shift = rng.choice((-1.2, 1.2)); shared = set(rng.sample(range(8), 5))
    specs = []
    for m in range(8):
        sp = machine_spec(rng, sc, m)
        sp.update(mt=rng.choice(list(MU)), sig=rng.randint(-1, 1), b0=rng.choice((-1, 1)) * rng.gauss(2.4, 0.3),
                  d=rng.gauss(0, 0.6), drift=(cond == "drift" and m % 2 == 0))
        specs.append(sp)
    S = [dict() for _ in range(8)]; tg = [None] * 8; walk = [0.0] * 8
    for t in range(c + 1):
        new = []
        for m, sp in enumerate(specs):
            s = S[m]; mt = sp["mt"]
            x = xval(sp, s, t, S)
            g = lambda k: 0 if s.get(t - k) is None else int(s[t - k] >= 2)
            if sc == "S6":
                z = g(sp["k"]) + g(sp["k2"]); A, B = TABLES[mt]; tab = [A, [(u + v) / 2 for u, v in zip(A, B)], B][z]; zb = z / 2
            elif sc == "S8":
                z = 0; tab = TABLES[mt][0]; zb = 0
            else:
                z = g(sp["k"]) ^ sp["zeta"]
                if sc == "S7" and sp["sig"] == 1: z = 1
                tab = TABLES[mt][z]; zb = z
            bt = sp["b0"] * math.cos(math.pi * t / c) if sp["drift"] else sp["b0"]
            e = shift if (cond == "shared_change" and m in shared and t >= shift_t) else 0.0
            if sc == "S8":
                walk[m] += rng.gauss(0, 0.12 if not sp["drift"] else 0.2); e += walk[m]
            if sc == "S7": e += 0.8 * sp["sig"]
            pw = sig(MU[mt] + sp["d"] + bt * x + DELTA * zb + e + rng.gauss(0, 0.5))
            p = [1 - pw] + [pw * q for q in tab]
            if t == c: tg[m] = p
            r = rng.random(); acc = 0.0; y = 20
            for v, q in zip(PAY, p):
                acc += q
                if r < acc: y = v; break
            new.append(y)
        for m in range(8): S[m][t] = new[m]
    ms = []
    for m, sp in enumerate(specs):
        hist = {t: v for t, v in S[m].items() if t < c}
        if cond == "short_history":
            rec = set(range(c - 40, c)); old = set(rng.sample(range(0, c - 40), 24))
            hist = {t: v for t, v in hist.items() if t in rec or t in old}
        ms.append(dict(mt=sp["mt"], sig=sp["sig"], hist=hist, y=S[m][c], tgt=tg[m]))
    return ms


def write(pub, split, eps, ref_tab=None):
    d = pub / split; d.mkdir(parents=True, exist_ok=True)
    fq = csv.writer(open(d / "queries.csv", "w", newline=""), lineterminator="\n"); fq.writerow(["query_id", "episode_id", "machine_id", "machine_type", "signal", "cutoff"])
    fh = csv.writer(open(d / "history.csv", "w", newline=""), lineterminator="\n"); fh.writerow(["episode_id", "machine_id", "tick", "payout"])
    fs = csv.writer(open(d / "sample_submission.csv", "w", newline=""), lineterminator="\n"); fs.writerow(["query_id", "p_0", "p_1", "p_2", "p_5", "p_20"])
    lab = []; cnd = []; tg = []; rf = []; counts = {}
    for e, (cond, c, ms) in enumerate(eps):
        ep = f"syn-{split}-e{e:05d}"
        for m, M in enumerate(ms):
            mid = f"{ep}-m{m:03d}"; qid = f"{mid}-next"
            fq.writerow([qid, ep, mid, M["mt"], M["sig"], c]); fs.writerow([qid] + [0.2] * 5)
            for t in sorted(M["hist"]): fh.writerow([ep, mid, t, M["hist"][t]])
            lab.append([qid, M["y"]]); cnd.append([qid, cond]); tg.append([qid, ep] + M["tgt"] + [M["y"]])
            for t, v in M["hist"].items(): counts.setdefault(M["mt"], [0] * 5)[PAY.index(v)] += 1
            rf.append((qid, M["mt"]))
    if split == "test": return counts
    def w(name, hdr, rows):
        f = csv.writer(open(d / name, "w", newline=""), lineterminator="\n"); f.writerow(hdr); f.writerows(rows)
    w("labels.csv", ["query_id", "outcome"], lab); w("conditions.csv", ["query_id", "condition"], cnd)
    if split == "validation":
        w("targets.csv", ["query_id", "episode_id", "p_0", "p_1", "p_2", "p_5", "p_20", "outcome"], tg)
        rows = []
        for qid, mt in rf:  # pooled reference: per-type training frequencies, strength 20 toward uniform
            cc = ref_tab[mt]; z = sum(cc) + 100; rows.append([qid] + [(v + 20) / z for v in cc])
        w("reference.csv", ["query_id", "p_0", "p_1", "p_2", "p_5", "p_20"], rows)
    return counts


def main():
    out = Path(sys.argv[1]); sc = sys.argv[2]; seed = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    rng = random.Random(f"{sc}-{seed}"); pub = out / "public"
    def gen(n):
        return [(cond, c, episode(rng, sc, cond, c)) for cond in CONDS for _ in range(n) for c in [rng.randint(192, 256)]]
    trc = write(pub, "train", gen(24))
    write(pub, "validation", gen(24), trc); write(pub, "test", gen(48))
    (pub / "metadata.json").write_text(json.dumps({"round": f"synthetic-{sc}", "max_points": 40, "season_id": "synthetic",
                                                   "conditions": list(CONDS), "machines_per_episode": 8}, indent=2) + "\n")
    print("synthetic kit", sc, "->", pub)


if __name__ == "__main__":
    main()

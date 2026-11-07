"""Synthetic stress-test kit with CHANGED temporal structure (for testing the round-2 pipeline only).
usage: python src/synth_kit.py <out_dir> [seed]
Same family as the main-round simulator as we understand it, but with different lag pairs
((3,6), (1,8), (2,4)), an UNTIED size lag (uniform 1..8) and per-tick logit noise sd 0.5.
Conditions: stable; short_history (40 recent + 24 sampled older ticks); drift (b reverses over time
for half the machines); shared_change (5 of 8 machines get a common logit shift at a random tick).
Writes public/{train,validation,test} in the organizer file format + metadata.json."""
import sys, csv, json, math, random
from pathlib import Path

PAIRS = ((3, 6), (1, 8), (2, 4))
TABLES = {"steady": [[0.754, 0.197, 0.048, 0.001], [0.455, 0.436, 0.107, 0.002]],
          "balanced": [[0.5, 0.348, 0.13, 0.022], [0.214, 0.547, 0.205, 0.034]],
          "jackpot": [[0.335, 0.245, 0.315, 0.105], [0.121, 0.324, 0.416, 0.139]]}
MU = {"steady": 0.33, "balanced": -0.94, "jackpot": -2.5}; DELTA = 0.9
PAY = (0, 1, 2, 5, 20); CONDS = ("stable", "short_history", "drift", "shared_change")


def sig(x): return 1 / (1 + math.exp(-x))


def episode(rng, cond, c):
    ms = []
    shift_t = rng.randint(100, c - 15); shift = rng.choice((-1.2, 1.2)); shared = set(rng.sample(range(8), 5))
    for m in range(8):
        mt = rng.choice(list(MU)); a, b = rng.choice(PAIRS); k = rng.randint(1, 8); zeta = rng.randint(0, 1)
        b0 = rng.choice((-1, 1)) * rng.gauss(2.4, 0.3); d = rng.gauss(0, 0.6); drift = cond == "drift" and m % 2 == 0
        s = {}; tgt = None
        for t in range(c + 1):
            ya = s.get(t - a); yb = s.get(t - b)
            x = 0.0 if ya is None or yb is None else (1.0 if (ya > 0) != (yb > 0) else -1.0)
            u = s.get(t - k); z = (0 if u is None else int(u >= 2)) ^ zeta
            bt = b0 * math.cos(math.pi * t / c) if drift else b0
            e = shift if (cond == "shared_change" and m in shared and t >= shift_t) else 0.0
            pw = sig(MU[mt] + d + bt * x + DELTA * z + e + rng.gauss(0, 0.5))
            tab = TABLES[mt][z]; p = [1 - pw] + [pw * q for q in tab]
            if t == c:
                tgt = p
            r = rng.random(); acc = 0.0; y = 20
            for v, q in zip(PAY, p):
                acc += q
                if r < acc: y = v; break
            s[t] = y
        hist = {t: v for t, v in s.items() if t < c}
        if cond == "short_history":
            rec = set(range(c - 40, c)); old = rng.sample(range(0, c - 40), 24)
            hist = {t: v for t, v in hist.items() if t in rec or t in old}
        ms.append(dict(mt=mt, sig=rng.randint(-1, 1), hist=hist, y=s[c], tgt=tgt))
    return ms


def write(pub, split, eps, rng, ref_tab=None):
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
        for qid, mt in rf:
            cc = ref_tab[mt]; z = sum(cc) + 100; rows.append([qid] + [(v + 20) / (z) for v in cc])
        w("reference.csv", ["query_id", "p_0", "p_1", "p_2", "p_5", "p_20"], rows)
    return counts


def main():
    out = Path(sys.argv[1]); seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7; rng = random.Random(seed)
    pub = out / "public"
    def gen(n):
        return [(cond, c, episode(rng, cond, c)) for cond in CONDS for _ in range(n) for c in [rng.randint(192, 256)]]
    tr = gen(24); trc = write(pub, "train", tr, rng)
    write(pub, "validation", gen(24), rng, trc); write(pub, "test", gen(48), rng)
    (pub / "metadata.json").write_text(json.dumps({"round": "synthetic", "max_points": 40, "season_id": "synthetic",
                                                   "conditions": list(CONDS), "machines_per_episode": 8}, indent=2) + "\n")
    print("synthetic kit written to", pub)


if __name__ == "__main__":
    main()

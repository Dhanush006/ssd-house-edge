"""Fit model B's structure constants from a kit's organizer data (train + validation).

usage: python src/fit_b.py <kit_public_dir> <out_params.json> [--naive] [--novaltab]
  * size tables (2 per type): 2-means on validation targets' conditional size vectors p_i/(1-p_0)
    (--novaltab: EM from train+val histories instead, no targets).
  * population prior over win hypotheses h (28 XOR pairs, 8 single lags, none) and over size lag k|h
    (k=0..8; joint prior => the k=first-lag tie only if the data shows it): EM over machines using
    windowed beta-binomial evidence (win given x) + size log-lik.
  * mu per type, delta (big-size state boost), gam (signal), bmag/bsd: regression on validation targets'
    logit P(win) with the target z state and the machine's MAP hypothesis.
  --naive: skip the population prior (uniform over all 37 h, uniform k) -- "told nothing" ablation.
Filter dynamics (qb, qd, jump, etau, ...) are copied from the main-round entry (runs/06-lowjump).
"""
import sys, os, csv, json, math
sys.path.insert(0, os.path.dirname(__file__))
from pathlib import Path
from collections import defaultdict
import numpy as np
import model_b as M

BASE = dict(bmax=3.5, nb=15, dmax=2.0, nd=9, bmag=2.4, bsd=0.4, bunif=0.02, dsd=0.6, qb=0.0096, qd=0.0022,
            jump=0.0012, evt=1.0, epss=[0.001, 0.01, 0.05], floor=1e-06, eL=30, ehl=4, etau=0.353,
            K=4, sgap=12.0, pmin=0.004, swin=64, evs=1.0)
TYPES = ("steady", "balanced", "jackpot")


def load(pub, split):
    d = Path(pub) / split
    qs, H = M.read_split(d)
    lab = {r["query_id"]: int(r["outcome"]) for r in csv.DictReader(open(d / "labels.csv"))} if (d / "labels.csv").exists() else {}
    tg = {}
    if (d / "targets.csv").exists():
        for r in csv.DictReader(open(d / "targets.csv")):
            tg[r["query_id"]] = [float(r[f"p_{k}"]) for k in PAYK]
    return qs, H, lab, tg


PAYK = (0, 1, 2, 5, 20)


def two_means(Q):
    Q = np.asarray(Q); ev = Q @ np.array([1, 2, 5, 20.0])
    o = np.argsort(ev); n = len(Q)
    c = np.stack([Q[o[: n // 4]].mean(0), Q[o[-n // 4:]].mean(0)])
    for _ in range(50):
        d = ((Q[:, None, :] - c[None]) ** 2).sum(-1); a = d.argmin(1)
        c = np.stack([np.median(Q[a == j], 0) if (a == j).any() else c[j] for j in (0, 1)])
    c = c / c.sum(1, keepdims=True)
    if c[0] @ np.array([1, 2, 5, 20.0]) > c[1] @ np.array([1, 2, 5, 20.0]):
        c = c[::-1]
    d = ((Q[:, None, :] - c[None]) ** 2).sum(-1); a = d.argmin(1)
    spread = float(np.sqrt(d.min(1)).max())
    return c.tolist(), a, spread


def main():
    pub = Path(sys.argv[1]); out = Path(sys.argv[2]); naive = "--naive" in sys.argv; novaltab = "--novaltab" in sys.argv
    P = dict(BASE)
    tq, tH, tlab, _ = load(pub, "train")
    vq, vH, vlab, vtg = load(pub, "validation")
    # machines: full observed history (+ observed label at cutoff)
    mach = []
    for qs, H, lab, tg in ((tq, tH, tlab, {}), (vq, vH, vlab, vtg)):
        for q in qs:
            s = dict(H[q["machine_id"]]); c = int(q["cutoff"])
            s = {t: v for t, v in s.items() if t < c}
            if q["query_id"] in lab:
                s[c] = lab[q["query_id"]]
            mach.append(dict(q=q, s=s, c=c, mt=q["machine_type"], sig=int(q["signal"]), tg=tg.get(q["query_id"])))
    # ---- size tables
    SS = {}; zq = {}
    for mt in TYPES:
        rows = [m for m in mach if m["mt"] == mt and m["tg"]]
        Q = [[p / (1 - m["tg"][0]) for p in m["tg"][1:]] for m in rows]
        c2, a, spread = two_means(Q)
        SS[mt] = c2
        for m, ai in zip(rows, a):
            zq[m["q"]["query_id"]] = int(ai)
        print(f"fit_b tables {mt}: n={len(rows)} z1={int(a.sum())} max_dev={spread:.4f} A={[round(x, 3) for x in c2[0]]} B={[round(x, 3) for x in c2[1]]}")
    if novaltab:
        SS = em_tables(mach, SS)
    P["size_states"] = SS
    # ---- per-machine evidence
    hyps = M.ALL_HYPS; nh = len(hyps); nk = len(M.KS)
    BB = np.zeros((len(mach), nh)); LS = np.zeros((len(mach), nk))
    for i, m in enumerate(mach):
        s = m["s"]; ticks = sorted(s)
        tc = M.common_ticks(s, ticks, 8)
        for j, h in enumerate(hyps):
            BB[i, j] = M.screen_ll(s, tc, h, P["swin"])
        szh, szl = M.size_ll(s, ticks, SS[m["mt"]], [0.01])
        for k in M.KS:
            LS[i, k] = M.lse([szl[n] for n, (kk, ze, ep) in enumerate(szh) if kk == k]) - math.log(2)
    # ---- EM for joint prior pi[h,k]
    L = BB[:, :, None] + LS[:, None, :]
    pi = np.full((nh, nk), 1.0 / (nh * nk))
    for it in range(300):
        lw = L + np.log(pi)[None]; lw -= lw.max((1, 2), keepdims=True); r = np.exp(lw); r /= r.sum((1, 2), keepdims=True)
        pi = r.mean(0) + 1e-6; pi /= pi.sum()
    ph = pi.sum(1); pk = pi.sum(0)
    rh = r.sum(2)
    top = np.argsort(-ph)[:8]
    print("fit_b win hyps (prior): " + " ".join(f"{hyps[j]}:{ph[j]:.3f}" for j in top))
    print("fit_b size k marginal: " + " ".join(f"k{k}:{pk[k]:.3f}" for k in M.KS))
    for j in top[:6]:
        if ph[j] < 0.01: continue
        ck = pi[j] / ph[j]
        print(f"fit_b   k|{hyps[j]}: " + " ".join(f"k{k}:{ck[k]:.2f}" for k in M.KS if ck[k] > 0.02))
    if naive:
        P["hyps"] = [list(h) for h in hyps]; P["hprior"] = [1.0 / nh] * nh
        P["kprior"] = {"-1,-1": [1.0 / nk] * nk}
    else:
        rho = 0.002
        hp = (1 - rho) * ph + rho / nh
        P["hyps"] = [list(h) for h in hyps]; P["hprior"] = [float(x) for x in hp]
        kp = {}
        for j, h in enumerate(hyps):
            ck = (pi[j] + 0.02 * ph[j] * pk) / (ph[j] * 1.02)
            ck = 0.98 * ck / ck.sum() + 0.02 / nk
            kp[M.hkey(h)] = [float(x) for x in ck]
        kp["-1,-1"] = [float(x) for x in (0.98 * pk + 0.02 / nk)]
        P["kprior"] = kp
    # ---- mu / delta / gam / bmag from validation targets
    X = []; Y = []; XV = []
    for i, m in enumerate(mach):
        if not m["tg"]: continue
        pw = 1 - m["tg"][0]; z = zq[m["q"]["query_id"]]
        h = hyps[int(rh[i].argmax())]; xv = M.xval(m["s"], m["c"], h) or 0.0
        X.append([1.0 * (m["mt"] == t) for t in TYPES] + [z, m["sig"]]); Y.append(M.logit(pw)); XV.append(xv)
    X = np.array(X); Y = np.array(Y); XV = np.array(XV)
    # mixture regression: Y = type + delta z + gam sig + bmag * (sign_m * x) + d ; sign_m alternated
    sg = np.ones(len(Y)); coef = np.linalg.lstsq(X, Y, rcond=None)[0]
    for _ in range(30):
        r0 = Y - X @ coef
        bm = abs(np.median(np.abs(r0[XV != 0]))) if (XV != 0).any() else 0.0
        sg = np.where(r0 * XV >= 0, 1.0, -1.0)
        Xf = np.column_stack([X, sg * XV]); cf = np.linalg.lstsq(Xf, Y, rcond=None)[0]
        coef = cf[:-1]; bmag = cf[-1]
    res = Y - Xf @ cf
    P["mu"] = {t: round(float(coef[n]), 4) for n, t in enumerate(TYPES)}
    P["delta"] = round(float(coef[3]), 4); g = float(coef[4])
    P["gam"] = 0.0  # signal term: raw target-regression estimate is unreliable (val -0.4 pts on main); enable only via dev check
    P["gam_raw"] = round(g, 4)
    P["bmag"] = round(float(bmag), 3) if (XV != 0).any() else 0.5
    P["bsd"] = round(float(np.clip(np.abs(res).std(), 0.3, 0.8)), 3)
    P["bmax"] = round(max(3.5, P["bmag"] + 1.2), 3)
    res = res  # residual sd reported below
    print(f"fit_b mu={P['mu']} delta={P['delta']} gam_raw={g:.3f} bmag={P['bmag']} bsd={P['bsd']} resid_sd={res.std():.3f}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(P, indent=1, sort_keys=True) + "\n")
    print("fit_b wrote", out)


def em_tables(mach, SS):
    """Refit the two size tables per type from histories only (EM over machine size rule (k,zeta) and z)."""
    for mt in TYPES:
        ms = [m for m in mach if m["mt"] == mt]
        A, B = np.array(SS[mt][0]), np.array(SS[mt][1])
        for _ in range(15):
            ca = np.zeros(4); cb = np.zeros(4)
            for m in ms:
                s = m["s"]; ticks = sorted(s)
                szh, szl = M.size_ll(s, ticks, (A.tolist(), B.tolist()), [0.01])
                w = np.exp(np.array(szl) - max(szl)); w /= w.sum()
                for t in ticks:
                    v = s[t]
                    if v == 0: continue
                    i = PAYK.index(v) - 1; pz = 0.0
                    for wh, (k, ze, ep) in zip(w, szh):
                        if wh < 1e-4: continue
                        u = s.get(t - k) if k else 0
                        z1 = 0.5 if u is None else ((1 - ep) if ((u >= 2 if k else False) != ze) else ep)
                        pz += wh * z1
                    qa = (1 - pz) * A[i]; qb = pz * B[i]; tot = qa + qb
                    ca[i] += qa / tot; cb[i] += qb / tot
            A = (ca + 1) / (ca + 1).sum(); B = (cb + 1) / (cb + 1).sum()
        SS[mt] = [A.tolist(), B.tolist()]
        print(f"fit_b EM tables {mt}: A={[round(x, 3) for x in A]} B={[round(x, 3) for x in B]}")
    return SS


if __name__ == "__main__":
    main()

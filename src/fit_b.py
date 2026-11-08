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
import kitschema as KS
COLS, VALS = list(KS.DEFAULT_COLS), list(KS.DEFAULT_PAY)
MINTYPE = 30  # machine types with fewer validation targets use the pooled tables / mu ("_pool")

BASE = dict(bmax=3.5, nb=15, dmax=2.0, nd=9, bmag=2.4, bsd=0.4, bunif=0.02, dsd=0.6, qb=0.0096, qd=0.0022,
            jump=0.0012, evt=1.0, epss=[0.001, 0.01, 0.05], floor=1e-06, eL=30, ehl=4, etau=0.353,
            K=4, sgap=12.0, pmin=0.004, swin=64, evs=1.0)


def load(pub, split):
    d = Path(pub) / split
    if not (d / "queries.csv").exists():
        return [], {}, {}, {}
    qs, H = M.read_split(d, VALS)
    lab = KS.read_labels(d, VALS)
    tg = KS.read_targets(d, COLS)
    return qs, H, lab, tg


PAYK = (0, 1, 2, 5, 20)


def evv():
    return np.array(VALS[1:], dtype=float)


def kmeans(Q, K):
    K = max(1, min(K, len(Q)))
    ev = Q @ evv()
    cs = [Q[int(np.argmin(ev))]]  # deterministic farthest-point init (keeps rare extreme states)
    while len(cs) < K:
        d = np.min(((Q[:, None, :] - np.stack(cs)[None]) ** 2).sum(-1), 1); cs.append(Q[int(np.argmax(d))])
    c = np.stack(cs)
    for _ in range(50):
        d = ((Q[:, None, :] - c[None]) ** 2).sum(-1); a = d.argmin(1)
        c = np.stack([np.median(Q[a == j], 0) if (a == j).any() else c[j] for j in range(K)])
    c = c / c.sum(1, keepdims=True)
    c = c[np.argsort(c @ evv())]
    d = ((Q[:, None, :] - c[None]) ** 2).sum(-1)
    return c, float(np.sqrt(d.min(1)).max())


def size_tables(Q):
    """Size states from targets: smallest K in 2..4 whose clusters fit (max dev < .02); the two extreme clusters
    become tables A, B; every target is then represented as a mixture (1-z)A + zB (z clipped to [0,1]).
    Returns tables, per-target z, K, and max deviation of that two-table mixture representation."""
    Q = np.asarray(Q)
    for K in (2, 3, 4):
        c, sp = kmeans(Q, K)
        if sp < 0.02: break
    A, B = c[0], c[-1]; dv = B - A
    if float(np.sqrt(dv @ dv)) < 0.01:  # one size state only: B = A, z = 0 (delta unidentified -> 0)
        B = A.copy(); dv = B - A; z = np.zeros(len(Q))
    else:
        z = np.clip(((Q - A) @ dv) / (dv @ dv), 0, 1)
    res = Q - (A[None] + z[:, None] * dv[None])
    return [A.tolist(), B.tolist()], z, K, float(np.sqrt((res ** 2).sum(1)).max()), sp


def main():
    pub = Path(sys.argv[1]); out = Path(sys.argv[2]); naive = "--naive" in sys.argv; novaltab = "--novaltab" in sys.argv
    global COLS, VALS
    COLS, VALS = KS.find_schema(pub)
    P = dict(BASE); P["payvals"] = list(VALS); P["cols"] = list(COLS)
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
            t_ = tg.get(q["query_id"])
            if t_ is not None and (len(t_) != len(VALS) or t_[0] >= 1 - 1e-9): t_ = None
            mach.append(dict(q=q, s=s, c=c, mt=q["machine_type"], sig=q["signal"], tg=t_))
    ntg = defaultdict(int)
    for m in mach:
        if m["tg"]: ntg[m["mt"]] += 1
    mc = max([m["c"] for m in mach] + [0])
    if mc > 1500: P["maxhist"] = 1500  # runtime guard: the filter reads the most recent 1500 ticks only
    notg = sum(ntg.values()) < 2 * MINTYPE
    if notg:
        for m in mach: ntg[m["mt"]] += 1
    TYPES = [t for t in sorted(ntg) if ntg[t] >= MINTYPE and t != "_pool"]
    if set(TYPES) == {"steady", "balanced", "jackpot"}: TYPES = ["steady", "balanced", "jackpot"]
    for m in mach:
        if m["mt"] not in TYPES: m["mt"] = "_pool"
    print(f"fit_b schema: payouts={VALS} types={TYPES} (+_pool) targets={sum(ntg.values())} machines={len(mach)}")
    # ---- size tables
    SS = {}; zq = {}; DIAG = {"max_dev": {}}
    for mt in (TYPES + ["_pool"]) if not notg else []:
        rows = [m for m in mach if (m["mt"] == mt or mt == "_pool") and m["tg"]]
        Q = [[p / (1 - m["tg"][0]) for p in m["tg"][1:]] for m in rows]
        c2, a, K, spread, ksp = size_tables(Q)
        SS[mt] = c2; DIAG["max_dev"][mt] = round(spread, 4); DIAG.setdefault("K", {})[mt] = K
        for m, ai in zip(rows, a):
            if mt != "_pool" or m["mt"] == "_pool": zq[m["q"]["query_id"]] = float(ai)
        print(f"fit_b tables {mt}: n={len(rows)} K={K} (cluster dev {ksp:.3f}) z_mean={a.mean():.2f} mix_max_dev={spread:.4f} A={[round(x, 3) for x in c2[0]]} B={[round(x, 3) for x in c2[1]]}")
    if notg:  # no usable validation targets: empirical size frequencies split by previous big payout, then EM
        SS = init_tables(mach, TYPES + ["_pool"]); novaltab = True
    if novaltab:
        SS = em_tables(mach, SS, TYPES + ["_pool"])
    P["size_states"] = SS
    # ---- per-machine evidence
    hyps = M.ALL_HYPS if "--base" in sys.argv else M.FULL_HYPS; nh = len(hyps); nk = 9 if "--base" in sys.argv else len(M.SZ)
    # episode context (other machines of the same episode, sorted by query id) for cross-machine hypotheses
    byep = defaultdict(list)
    for i, m in enumerate(mach): byep[m["q"]["episode_id"]].append(i)
    for ep, ix in byep.items():
        ix.sort(key=lambda i: mach[i]["q"]["query_id"]); hs = [mach[i]["s"] for i in ix]
        for n, i in enumerate(ix): mach[i]["ctx"] = (hs, n)
    NW = int(os.environ.get("R2_FITW", "6"))
    eps = sorted(byep); chunks = [[i for e in eps[w::NW] for i in byep[e]] for w in range(NW)]
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(NW) as pool:
        res = pool.map(_evid, [([(mach[i]["s"], mach[i]["ctx"], SS[mach[i]["mt"]]) for i in ch], hyps, P["swin"], nk) for ch in chunks])
    BB = np.zeros((len(mach), nh)); LS = np.zeros((len(mach), nk))
    for ch, (bb, ls) in zip(chunks, res):
        BB[ch] = bb; LS[ch] = ls
    # ---- EM for joint prior pi[h,k]
    L = BB[:, :, None] + LS[:, None, :]
    pi = np.full((nh, nk), 1.0 / (nh * nk))
    for it in range(300):
        lw = L + np.log(pi)[None]; lw -= lw.max((1, 2), keepdims=True); r = np.exp(lw); r /= r.sum((1, 2), keepdims=True)
        pi = r.mean(0) + 1e-6; pi /= pi.sum()
    ph = pi.sum(1); pk = pi.sum(0)
    rh = r.sum(2)
    top = np.argsort(-ph)[:8]
    print("fit_b win hyps (prior): " + " ".join(f"{M.hkey(hyps[j])}:{ph[j]:.3f}" for j in top))
    print("fit_b size k marginal: " + " ".join(f"k{k}:{pk[k]:.3f}" for k in M.KS) + f" two-lag-sum:{pk[9:].sum():.3f}")
    for j in top[:6]:
        if ph[j] < 0.01: continue
        ck = pi[j] / ph[j]
        print(f"fit_b   k|{M.hkey(hyps[j])}: " + " ".join(f"{M.SZ[k]}:{ck[k]:.2f}" for k in range(nk) if ck[k] > 0.02))
    if naive:
        P["hyps"] = [M.hkey(h) for h in hyps]; P["hprior"] = [1.0 / nh] * nh
        P["kprior"] = {"-1,-1": [1.0 / nk] * nk}
    else:
        rho = 0.002
        hp = (1 - rho) * ph + rho / nh
        P["hyps"] = [M.hkey(h) for h in hyps]; P["hprior"] = [float(x) for x in hp]
        kp = {}
        for j, h in enumerate(hyps):
            ck = (pi[j] + 0.02 * ph[j] * pk) / (ph[j] * 1.02)
            ck = 0.98 * ck / ck.sum() + 0.02 / nk
            kp[M.hkey(h)] = [float(x) for x in ck]
        kp["-1,-1"] = [float(x) for x in (0.98 * pk + 0.02 / nk)]
        P["kprior"] = kp
    # ---- mu / delta / gam / bmag from validation targets
    X = []; Y = []; XV = []
    RT = TYPES + (["_pool"] if any(m["mt"] == "_pool" and m["tg"] for m in mach) else [])
    for i, m in enumerate(mach):
        if not m["tg"] or m["q"]["query_id"] not in zq: continue
        pw = 1 - m["tg"][0]; z = zq[m["q"]["query_id"]]
        h = hyps[int(rh[i].argmax())]; xv = M.xval(m["s"], m["c"], h, m["ctx"]) or 0.0
        X.append([1.0 * (m["mt"] == t) for t in RT] + [z, m["sig"]]); Y.append(M.logit(pw)); XV.append(xv)
    if len(Y) < 10:  # no targets: logit of observed win rates per type; delta/bmag defaults
        return finish_notg(P, mach, TYPES, hyps, ph, pk, nk, DIAG, out)
    NT = len(RT)
    X = np.array(X); Y = np.array(Y); XV = np.array(XV)
    if np.ptp(X[:, NT]) < 1e-9: X[:, NT] = 0.0  # no size-state variation: delta -> 0 via min-norm least squares
    if np.ptp(X[:, NT + 1]) < 1e-9: X[:, NT + 1] = 0.0  # constant signal: no signal term
    # mixture regression: Y = type + delta z + gam sig + bmag * (sign_m * x) + d ; sign_m alternated
    sg = np.ones(len(Y)); coef = np.linalg.lstsq(X, Y, rcond=None)[0]
    for _ in range(30):
        r0 = Y - X @ coef
        bm = abs(np.median(np.abs(r0[XV != 0]))) if (XV != 0).any() else 0.0
        sg = np.where(r0 * XV >= 0, 1.0, -1.0)
        Xf = np.column_stack([X, sg * XV]); cf = np.linalg.lstsq(Xf, Y, rcond=None)[0]
        coef = cf[:-1]; bmag = cf[-1]
    res = Y - Xf @ cf
    P["mu"] = {t: round(float(coef[n]), 4) for n, t in enumerate(RT)}
    if "_pool" not in P["mu"]:
        cnt = [sum(1 for m in mach if m["tg"] and m["mt"] == t) for t in TYPES]
        P["mu"]["_pool"] = round(float(sum(c * P["mu"][t] for c, t in zip(cnt, TYPES)) / max(1, sum(cnt))), 4)
    P["delta"] = round(float(coef[NT]), 4); g = float(coef[NT + 1])
    P["gam"] = 0.0  # signal term: raw target-regression estimate is unreliable (val -0.4 pts on main); enable only via dev check
    P["gam_raw"] = round(g, 4)
    P["bmag"] = round(float(bmag), 3) if (XV != 0).any() else 0.5
    P["bsd"] = round(float(np.clip(np.abs(res).std(), 0.3, 0.8)), 3)
    P["bmax"] = round(max(3.5, P["bmag"] + 1.2), 3)
    res = res  # residual sd reported below
    DIAG["top_h"] = [[M.hkey(hyps[j]), round(float(ph[j]), 4)] for j in top]
    DIAG["k_marg"] = [round(float(x), 4) for x in pk[:9]] + [round(float(pk[9:].sum()), 4)]; P["nsz"] = nk; DIAG["none_mass"] = round(float(ph[hyps.index(("none",))]), 4)
    DIAG["resid_sd"] = round(float(res.std()), 3); P["diag"] = DIAG
    print(f"fit_b mu={P['mu']} delta={P['delta']} gam_raw={g:.3f} bmag={P['bmag']} bsd={P['bsd']} resid_sd={res.std():.3f}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(P, indent=1, sort_keys=True) + "\n")
    print("fit_b wrote", out)


def _evid(args):
    rows, hyps, swin, nk = args
    bb = []; ls = []
    for s, ctx, tabs in rows:
        ticks = sorted(s); tc = M.common_ticks(s, ticks, 8)
        bb.append([M.screen_ll(s, tc, h, swin, ctx) for h in hyps])
        szh, szl = M.size_ll(s, ticks, tabs, [0.01], nk)
        ls.append([M.lse([szl[n] for n, (kk, ze, ep) in enumerate(szh) if kk == k]) - math.log(2) for k in range(nk)])
    return bb, ls


def em_tables(mach, SS, types):
    """Refit the two size tables per type from histories only (EM over machine size rule (k,zeta) and z)."""
    npos = len(VALS) - 1
    for mt in types:
        ms = [m for m in mach if m["mt"] == mt or mt == "_pool"]
        if not ms or npos < 1: continue
        A, B = np.array(SS[mt][0]), np.array(SS[mt][1])
        for _ in range(15):
            ca = np.zeros(npos); cb = np.zeros(npos)
            for m in ms:
                s = m["s"]; ticks = sorted(s)
                szh, szl = M.size_ll(s, ticks, (A.tolist(), B.tolist()), [0.01])
                w = np.exp(np.array(szl) - max(szl)); w /= w.sum()
                for t in ticks:
                    v = s[t]
                    if v == 0: continue
                    i = v - 1; pz = 0.0
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


def init_tables(mach, types):
    """Two size tables per type from histories: size frequencies after a small vs a big previous payout."""
    npos = len(VALS) - 1; SS = {}
    for mt in types:
        ca = np.ones(npos); cb = np.ones(npos)
        for m in mach:
            if not (m["mt"] == mt or mt == "_pool"): continue
            s = m["s"]
            for t, v in s.items():
                if v == 0: continue
                u = s.get(t - 1)
                if u is None: continue
                (cb if u >= 2 else ca)[v - 1] += 1
        SS[mt] = [(ca / ca.sum()).tolist(), (cb / cb.sum()).tolist()]
    return SS


def finish_notg(P, mach, TYPES, hyps, ph, pk, nk, DIAG, out):
    P["mu"] = {}
    for t in TYPES + ["_pool"]:
        w = l = 1
        for m in mach:
            if m["mt"] == t or t == "_pool":
                for v in m["s"].values():
                    if v > 0: w += 1
                    else: l += 1
        P["mu"][t] = round(M.logit(w / (w + l)), 4)
    P["delta"] = 0.5; P["gam"] = 0.0; P["gam_raw"] = 0.0; P["bmag"] = 2.0; P["bsd"] = 0.6; P["bmax"] = 3.5
    DIAG["top_h"] = [[M.hkey(hyps[j]), round(float(ph[j]), 4)] for j in np.argsort(-ph)[:8]]
    DIAG["k_marg"] = [round(float(x), 4) for x in pk[:9]] + [round(float(pk[9:].sum()), 4)]; P["nsz"] = nk
    DIAG["none_mass"] = round(float(ph[hyps.index(("none",))]), 4); DIAG["resid_sd"] = -1; DIAG["no_targets"] = True
    P["diag"] = DIAG
    print(f"fit_b NO TARGETS fallback mu={P['mu']}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(P, indent=1, sort_keys=True) + "\n")
    print("fit_b wrote", out)


if __name__ == "__main__":
    main()

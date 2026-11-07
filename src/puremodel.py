"""House Edge model, pure Python standard library (deterministic).

Generative structure found in the data:
  * win:   logit P(win_t) = mu_type + d + b * x_t + delta * z_t,
           x_t = +1 if win(y_{t-a}) != win(y_{t-b}) else -1, machine-specific pair (a,b) in {(1,3),(2,5),(4,7)};
           b (polarity/strength) can drift or jump; d is a slowly varying offset.
  * size:  given a win, payout size ~ A_type (z=0) or B_type (z=1);
           z_t = [y_{t-k} >= 2] xor zeta (noise eps), machine-specific k in {1,2,4}.
We run a Bayesian grid filter over (b, d) for each pair, weight pairs by evidence, and a Bayesian
posterior over size rules (k, zeta, eps). Only ticks < cutoff of the same machine are read.
"""
import csv, math
from collections import defaultdict
from pathlib import Path

PAY = (0, 1, 2, 5, 20)
PAIRS = ((1, 3), (2, 5), (4, 7))
SIZE_STATES = {
 "steady": [[0.753989389636146, 0.19680848829108313, 0.04821807963131535, 0.0009840424414554146],
            [0.4551220076282822, 0.43590239389737434, 0.10679608650485668, 0.002179511969486871]],
 "balanced": [[0.4998531084689386, 0.3479282723694339, 0.13047310213853774, 0.021745517023089637],
              [0.2140661471457723, 0.5467365932898975, 0.20502622248371163, 0.03417103708061861]],
 "jackpot": [[0.3351460319013039, 0.2449461987732039, 0.31493082699411934, 0.10497694233137306],
             [0.12078673011007354, 0.3239206783804992, 0.4164694436320704, 0.13882314787735675]]}

def read_split(d):
    d = Path(d)
    qs = list(csv.DictReader(open(d / "queries.csv", newline="", encoding="utf-8")))
    H = defaultdict(dict)
    for r in csv.DictReader(open(d / "history.csv", newline="", encoding="utf-8")):
        H[r["machine_id"]][int(r["tick"])] = int(r["payout"])
    return qs, H

def sig(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x); return e / (1.0 + e)

def linspace(a, b, n):
    return [a + (b - a) * i / (n - 1) for i in range(n)]

# ---------------- size rule ----------------
def size_rule(s, ticks, c, mtype, P, lags=None):
    if P.get("zflip"):
        return size_rule_flip(s, ticks, c, mtype, P, lags)
    A, B = SIZE_STATES[mtype]
    hyps = [(k, ze, ep) for k in (lags or P["zlags"]) for ze in (0, 1) for ep in P["epss"]]
    ll = [0.0] * len(hyps)
    for t in ticks:
        v = s[t]
        if v == 0:
            continue
        i = PAY.index(v) - 1
        for h, (k, ze, ep) in enumerate(hyps):
            u = s.get(t - k)
            if u is None:
                continue
            z1 = (1 - ep) if ((u >= 2) != ze) else ep
            ll[h] += math.log((1 - z1) * A[i] + z1 * B[i])
    mx = max(ll); w = [math.exp(x - mx) for x in ll]; Z = math.fsum(w)
    w = [x / Z for x in w]
    def pz_at(tt):
        acc = 0.0
        for wh, (k, ze, ep) in zip(w, hyps):
            u = s.get(tt - k)
            acc += wh * (0.5 if u is None else ((1 - ep) if ((u >= 2) != ze) else ep))
        return acc
    return pz_at

def size_rule_flip(s, ticks, c, mtype, P, lags=None):
    """Same rule, but the polarity zeta may flip over time (2-state HMM, flip prob per tick)."""
    A, B = SIZE_STATES[mtype]; f = P["zflip"]
    hyps = [(k, ep) for k in (lags or P["zlags"]) for ep in P["epss"]]
    ll = [0.0] * len(hyps); pre = [dict() for _ in hyps]; fin = [0.5] * len(hyps)
    for h, (k, ep) in enumerate(hyps):
        p1 = 0.5; last = None; L = 0.0
        for t in ticks:
            if last is not None:
                p1 = 0.5 + (p1 - 0.5) * (1 - 2 * f) ** (t - last)
            last = t
            pre[h][t] = p1
            v = s[t]
            if v == 0:
                continue
            u = s.get(t - k)
            if u is None:
                continue
            i = PAY.index(v) - 1; g = u >= 2
            # zeta=0: z=g w.p. 1-ep ; zeta=1: z=not g
            zA = (1 - ep) if g else ep            # P(z=1 | zeta=0)
            l0 = (1 - zA) * A[i] + zA * B[i]; l1 = zA * A[i] + (1 - zA) * B[i]
            m = (1 - p1) * l0 + p1 * l1; L += math.log(m); p1 = p1 * l1 / m
        if last is not None:
            p1 = 0.5 + (p1 - 0.5) * (1 - 2 * f) ** (c - last)
        ll[h] = L; fin[h] = p1
    mx = max(ll); w = [math.exp(x - mx) for x in ll]; Z = math.fsum(w)
    w = [x / Z for x in w]
    def pz_at(tt):
        acc = 0.0
        for h, (k, ep) in enumerate(hyps):
            u = s.get(tt - k)
            if u is None:
                acc += w[h] * 0.5; continue
            p1 = fin[h] if tt >= c else pre[h].get(tt, 0.5)
            zA = (1 - ep) if u >= 2 else ep
            acc += w[h] * ((1 - p1) * zA + p1 * (1 - zA))
        return acc
    return pz_at

# ---------------- win filter ----------------
class WinFilter:
    def __init__(self, P):
        self.P = P
        self.B = linspace(-P["bmax"], P["bmax"], P["nb"]); self.D = linspace(-P["dmax"], P["dmax"], P["nd"])
        nb, nd = len(self.B), len(self.D); self.nb, self.nd = nb, nd
        pb = [math.exp(-(abs(b) - P["bmag"]) ** 2 / (2 * P["bsd"] ** 2)) + P["bunif"] for b in self.B]
        zb = math.fsum(pb); self.PB = [x / zb for x in pb]
        pdd = [math.exp(-d * d / (2 * P["dsd"] ** 2)) for d in self.D]
        zd = math.fsum(pdd); self.PD = [x / zd for x in pdd]
        self.prior = [self.PB[ib] * self.PD[id_] for ib in range(nb) for id_ in range(nd)]
        self.tables = {}
        for mt, mu in P["mu"].items():
            for x in (-1.0, 1.0):
                s0 = [sig(mu + self.D[id_] + self.B[ib] * x) for ib in range(nb) for id_ in range(nd)]
                s1 = [sig(mu + self.D[id_] + self.B[ib] * x + P["delta"]) for ib in range(nb) for id_ in range(nd)]
                self.tables[(mt, x)] = (s0, s1)

    def sigtables(self, mtype, x, sigv):
        P = self.P; g = P["gam"][mtype] if isinstance(P["gam"], dict) else P["gam"]
        mu = P["mu"][mtype] + g * sigv; nb, nd = self.nb, self.nd
        s0 = [sig(mu + self.D[id_] + self.B[ib] * x) for ib in range(nb) for id_ in range(nd)]
        s1 = [sig(mu + self.D[id_] + self.B[ib] * x + P["delta"]) for ib in range(nb) for id_ in range(nd)]
        return s0, s1

    def step(self, p, n):
        P = self.P; nb, nd = self.nb, self.nd; qb, qd, j = P["qb"], P["qd"], P["jump"]
        PB = self.PB
        for _ in range(min(n, 30)):
            # b diffusion (reflecting)
            q = [0.0] * (nb * nd)
            for ib in range(nb):
                lo = ib - 1 if ib > 0 else ib; hi = ib + 1 if ib < nb - 1 else ib
                o, ol, oh = ib * nd, lo * nd, hi * nd
                for id_ in range(nd):
                    q[o + id_] += (1 - 2 * qb) * p[o + id_]
                    q[ol + id_] += qb * p[o + id_]
                    q[oh + id_] += qb * p[o + id_]
            # d diffusion
            r = [0.0] * (nb * nd)
            for ib in range(nb):
                o = ib * nd
                for id_ in range(nd):
                    v = q[o + id_]
                    lo = id_ - 1 if id_ > 0 else id_; hi = id_ + 1 if id_ < nd - 1 else id_
                    r[o + id_] += (1 - 2 * qd) * v
                    r[o + lo] += qd * v
                    r[o + hi] += qd * v
            if j > 0:
                md = [math.fsum(r[ib * nd + id_] for ib in range(nb)) for id_ in range(nd)]
                r = [(1 - j) * r[ib * nd + id_] + j * PB[ib] * md[id_] for ib in range(nb) for id_ in range(nd)]
            p = r
        return p

    def machine(self, s, c, mtype, ticks, pz_at, sigv=0):
        P = self.P
        A, Bs = SIZE_STATES[mtype]
        tie = P.get("tie", False)
        if not tie:
            PZ = {t: pz_at(t) for t in ticks}
            pzc = pz_at(c)
        lev = []; preds = []; recs = []
        for (a, b) in PAIRS:
            if tie:
                pz_p = size_rule(s, ticks, c, mtype, P, lags=[a])
                PZ = {t: pz_p(t) for t in ticks}; pzc = pz_p(c)
            post = list(self.prior); last = None; le = 0.0; rec = {}
            L = P.get("eL", 0)
            for t in ticks:
                va = s.get(t - a); vb = s.get(t - b)
                if va is None or vb is None:
                    continue
                if last is not None:
                    post = self.step(post, t - last)
                last = t
                x = 1.0 if (va > 0) != (vb > 0) else -1.0
                s0, s1 = self.tables[(mtype, x)]
                y = s[t]; pz = PZ[t]; qz = 1 - pz
                if L and t >= c - L:
                    rec[t] = qz * math.fsum(u * w for u, w in zip(post, s0)) + pz * math.fsum(u * w for u, w in zip(post, s1))
                if y == 0:
                    lik = [qz * (1 - u) + pz * (1 - w) for u, w in zip(s0, s1)]
                else:
                    k = PAY.index(y) - 1; ak = qz * A[k]; bk = pz * Bs[k]
                    lik = [ak * u + bk * w for u, w in zip(s0, s1)]
                post = [u * w for u, w in zip(post, lik)]
                z = math.fsum(post); le += math.log(z)
                post = [u / z for u in post]
            va = s.get(c - a); vb = s.get(c - b)
            if last is None or va is None or vb is None:
                lev.append(None); preds.append(None); recs.append(None); continue
            post = self.step(post, c - last)
            x = 1.0 if (va > 0) != (vb > 0) else -1.0
            s0, s1 = self.tables[(mtype, x)]
            if P.get("gam") and sigv:
                s0, s1 = self.sigtables(mtype, x, sigv)
            e0 = math.fsum(u * w for u, w in zip(post, s0)); e1 = math.fsum(u * w for u, w in zip(post, s1))
            w0 = (1 - pzc) * e0; w1 = pzc * e1
            preds.append([1 - w0 - w1] + [w0 * A[i] + w1 * Bs[i] for i in range(4)]); lev.append(le); recs.append(rec)
        ok = [i for i in range(len(PAIRS)) if lev[i] is not None]
        if not ok:
            pzc = pz_at(c)
            mu = P["mu"][mtype]; e0 = sig(mu); e1 = sig(mu + P["delta"])
            w0 = (1 - pzc) * e0; w1 = pzc * e1
            return [1 - w0 - w1] + [w0 * A[i] + w1 * Bs[i] for i in range(4)], []
        mx = max(lev[i] for i in ok)
        wts = {i: math.exp(P["evt"] * (lev[i] - mx)) for i in ok}; Z = math.fsum(wts.values())
        common = sorted(set.intersection(*[set(recs[i]) for i in ok]))
        rec = [(t, math.fsum(wts[i] * recs[i][t] for i in ok) / Z, 1 if s[t] > 0 else 0) for t in common]
        return [math.fsum(wts[i] * preds[i][j] for i in ok) / Z for j in range(5)], rec

def predict_query(F, s, c, mtype, P, sigv=0):
    ticks = sorted(t for t in s if t < c)
    pz_at = size_rule(s, ticks, c, mtype, P)
    return F.machine(s, c, mtype, ticks, pz_at, sigv)

def logit(p):
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))

def episode_offset(recs, c, P, hl=None, tau=None):
    """Shared episode-level logit shift e fitted to recent outcomes of all machines in the episode
    (only ticks < c), with Gaussian prior N(0, etau^2) and exponential recency weights."""
    g = 0.5 ** (1.0 / (hl or P["ehl"])); obs = []
    for rec in recs:
        for t, pw, y in rec:
            obs.append((g ** (c - 1 - t), logit(pw), y))
    e = 0.0; inv = 1.0 / ((tau or P["etau"]) ** 2)
    for _ in range(20):
        gr = -e * inv; h = -inv
        for w, l, y in obs:
            p = sig(l + e); gr += w * (y - p); h -= w * p * (1 - p)
        e -= gr / h
    return e

def finalize(p, e, P, nh=999):
    tmp = P.get("tshort", 1.0) if nh < 100 else P.get("tlong", 1.0)
    if e or tmp != 1.0:
        pw = 1 - p[0]; pw2 = sig(tmp * logit(pw) + P.get("ekap", 1.0) * e)
        p = [1 - pw2] + [v * pw2 / pw for v in p[1:]]
    p = [min(max(v, P["floor"]), 1.0) for v in p]
    z = math.fsum(p)
    return [v / z for v in p]

def _work(args):
    P, items = args
    F = WinFilter(P)
    raw = [(qid, ep, c) + predict_query(F, s, c, mt, P, sg) for qid, ep, s, c, mt, sg in items]
    nhist = {it[0]: sum(1 for t in it[2] if t < it[3]) for it in items}
    by_ep = defaultdict(list)
    for qid, ep, c, p, rec in raw:
        by_ep[ep].append(rec)
    out = []
    for qid, ep, c, p, rec in raw:
        e = 0.0
        if P.get("eL", 0):
            others = [r for r in by_ep[ep] if r is not rec] if P.get("eloo") else by_ep[ep]
            e = episode_offset(others, c, P)
        if P.get("stau"):
            e += episode_offset([rec], c, P, P["shl"], P["stau"])
        out.append((qid, finalize(p, e, P, nhist[qid])))
    return out

def predict_split(d, P, workers=1):
    qs, H = read_split(d)
    items = [(q["query_id"], q["episode_id"], H[q["machine_id"]], int(q["cutoff"]), q["machine_type"], int(q["signal"])) for q in sorted(qs, key=lambda r: r["query_id"])]
    if workers > 1:
        import multiprocessing as mp
        eps = sorted(set(it[1] for it in items)); slot = {e: i % workers for i, e in enumerate(eps)}
        chunks = [[it for it in items if slot[it[1]] == i] for i in range(workers)]
        with mp.get_context("spawn").Pool(workers) as pool:
            res = pool.map(_work, [(P, ch) for ch in chunks])
        out = {}
        for r in res:
            out.update(dict(r))
        return out
    return dict(_work((P, items)))

DEFAULT = dict(bmax=3.5, nb=15, dmax=2.0, nd=9, bmag=2.4, bsd=0.4, bunif=0.02, dsd=0.6, qb=0.02, qd=0.01,
               jump=0.005, evt=1.0, mu={"steady": 0.33, "balanced": -0.94, "jackpot": -2.5}, delta=0.9,
               zlags=[1, 2, 4], epss=[0.03, 0.15, 0.3], floor=1e-6)

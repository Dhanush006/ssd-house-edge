"""House Edge model B: generalized structured model (pure Python standard library, deterministic).

Same generative family as model A, but the rule constants are fitted from the kit (src/fit_b.py):
  * win:  logit P(win_t) = mu_type + gam*signal + d + b * x_t + delta * z_t
          x_t from a machine-specific hypothesis h:
            (a, b>0): x = +1 if win(t-a) != win(t-b) else -1      (XOR pair, 28 candidates)
            (a, 0)  : x = +1 if win(t-a) else -1                  (single lag, 8 candidates)
            (0, 0)  : x = 0                                       (no lag effect)
          population prior over h is fitted (params "hyps"/"hprior"); per query the hypotheses are
          screened cheaply (windowed beta-binomial evidence) and the top few run the Bayesian (b,d) filter.
  * size: given a win, size ~ table A_type (z=0) or B_type (z=1);
          z_t = [y_{t-k} >= 2] xor zeta (noise eps), k in 1..8, or k=0 (constant state zeta).
          prior over k may depend on h (fitted joint prior "kprior"), so the "size lag = first win lag"
          tie is used only when the data supports it.
Only ticks < cutoff of the same machine are read; the episode shift uses other machines of the same
episode at ticks < cutoff. No condition labels.
"""
import csv, math
from collections import defaultdict
from pathlib import Path
import kitschema as KS_

PAY = (0, 1, 2, 5, 20)  # default only; histories are payout *indices* (0 = loss, >= 2 = big), see kitschema
KS = tuple(range(9))  # 0 = constant state
SZ = [(k,) for k in range(9)] + [(a, b) for a in range(1, 9) for b in range(a + 1, 9)]  # size-state drivers


def read_split(d, vals=None):
    return KS_.read_split(d, vals)


def rtype(P, mtype):
    """Machine type known to the fitted params, else the pooled fallback."""
    if mtype in P["mu"] and mtype in P["size_states"]:
        return mtype
    if "_pool" in P["mu"] and "_pool" in P["size_states"]:
        return "_pool"
    return sorted(k for k in P["mu"] if k in P["size_states"])[0]


def sig(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x); return e / (1.0 + e)


def logit(p):
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def linspace(a, b, n):
    return [a + (b - a) * i / (n - 1) for i in range(n)]


OLDK = ("xor", "lag", "none")


def to_h(h):
    """Accept old [a, b] lists (xor / single lag / none) or key strings 'kind:i,j'."""
    if isinstance(h, str):
        if ":" not in h:
            h = [int(v) for v in h.split(",")]
        else:
            k, a = h.split(":"); return (k,) + tuple(int(v) for v in a.split(",")) if a else (k,)
    a, b = int(h[0]), int(h[1])
    if a < 0: return ("fallback",)
    if a == 0: return ("none",)
    if b == 0: return ("lag", a)
    return ("xor", a, b)


def hkey(h):
    k = h[0]
    if k == "xor": return f"{h[1]},{h[2]}"
    if k == "lag": return f"{h[1]},0"
    if k == "none": return "0,0"
    return k + ":" + ",".join(str(v) for v in h[1:])


def own_lmax(h):
    k = h[0]
    if k in ("none", "cr"): return 1
    return max(h[1:])


def xval(s, t, h, ctx=None):
    """x for hypothesis h at tick t, or None if a needed observation is missing."""
    k = h[0]
    if k == "none":
        return 0.0
    if k == "cr":
        hs, mi = ctx; v = hs[(mi + h[1]) % len(hs)].get(t - h[2])
        return None if v is None else (1.0 if v > 0 else -1.0)
    if k == "cnt":
        n = h[1]; w = 0
        for j in range(1, n + 1):
            v = s.get(t - j)
            if v is None: return None
            w += v > 0
        return (2.0 * w - n) / n
    va = s.get(t - h[1])
    if va is None:
        return None
    wa = va > 0
    if k == "lag":
        return 1.0 if wa else -1.0
    vb = s.get(t - h[2])
    if vb is None:
        return None
    wb = vb > 0
    if k == "xor": r = wa != wb
    elif k == "and": r = wa and wb
    elif k == "or": r = wa or wb
    elif k == "anb": r = wa and not wb
    elif k == "bna": r = wb and not wa
    elif k == "p3":
        vc = s.get(t - h[3])
        if vc is None: return None
        r = (wa != wb) != (vc > 0)
    return 1.0 if r else -1.0


def lbb(w, l):
    """log Beta-binomial marginal of an ordered sequence with w wins, l losses, Beta(1,1) prior."""
    return math.lgamma(w + 1) + math.lgamma(l + 1) - math.lgamma(w + l + 2)


def common_ticks(s, ticks, lmax):
    return [t for t in ticks if all((t - j) in s for j in range(1, lmax + 1))]


def screen_ll(s, ticks_c, h, win, ctx=None):
    """Windowed beta-binomial evidence of win/loss given x under h (separate rates per x per window)."""
    cnt = defaultdict(lambda: [0, 0])
    for t in ticks_c:
        x = xval(s, t, h, ctx)
        cnt[(t // win, x)][0 if s[t] > 0 else 1] += 1
    return math.fsum(lbb(w, l) for w, l in cnt.values())


# ---------------- size rule ----------------
def fz(s, t, sz):
    """Size-state driver f in [0,1]: mean of [payout >= 2] at the lags in sz (missing -> 0.5); sz=(0,) -> 0."""
    if sz[0] == 0:
        return 0.0
    acc = 0.0
    for k in sz:
        u = s.get(t - k)
        acc += 0.5 if u is None else (1.0 if u >= 2 else 0.0)
    return acc / len(sz)


def size_ll(s, ticks, tabs, epss, nsz=9):
    """log-lik of positive payout sizes for each (size hyp j, zeta, eps); P(z=1) = ep + (1-2ep) * f_j (xor zeta)."""
    A, B = tabs; SZn = SZ[:nsz]
    hyps = [(j, ze, ep) for j in range(nsz) for ze in (0, 1) for ep in epss]
    ll = [0.0] * len(hyps)
    for t in ticks:
        v = s[t]
        if v == 0:
            continue
        i = v - 1; ai = A[i]; bi = B[i]
        f = [fz(s, t, sz) for sz in SZn]
        for h, (j, ze, ep) in enumerate(hyps):
            g = f[j] if ze == 0 else 1.0 - f[j]
            z1 = ep + (1 - 2 * ep) * g
            ll[h] += math.log((1 - z1) * ai + z1 * bi)
    return hyps, ll


def lse(v):
    m = max(v)
    return m + math.log(math.fsum(math.exp(x - m) for x in v))


class SizePost:
    """Posterior over size hypotheses (j, zeta, eps) given a prior over j; pz(t) = P(z_t = 1) = sum_j c0_j + a_j f_j(t)."""
    def __init__(self, s, hyps, ll, kp, nep):
        self.s = s
        lw = [ll[i] + math.log(max(kp[j] if j < len(kp) else 0.0, 1e-300)) - math.log(2 * nep) for i, (j, ze, ep) in enumerate(hyps)]
        self.logev = lse(lw)
        w = [math.exp(x - self.logev) for x in lw]
        agg = {}
        for wh, (j, ze, ep) in zip(w, hyps):
            a = agg.setdefault(j, [0.0, 0.0])
            if ze == 0:
                a[0] += wh * ep; a[1] += wh * (1 - 2 * ep)
            else:
                a[0] += wh * (1 - ep); a[1] -= wh * (1 - 2 * ep)
        self.agg = [(SZ[j], a[0], a[1]) for j, a in sorted(agg.items()) if abs(a[0]) + abs(a[1]) > 1e-14]

    def pz(self, t):
        s = self.s; acc = 0.0
        for sz, c0, a in self.agg:
            acc += c0 + a * fz(s, t, sz)
        return acc


# ---------------- win filter ----------------
class WinFilter:
    def __init__(self, P):
        self.P = P
        self.SS = P["size_states"]
        self.B = linspace(-P["bmax"], P["bmax"], P["nb"]); self.D = linspace(-P["dmax"], P["dmax"], P["nd"])
        nb, nd = len(self.B), len(self.D); self.nb, self.nd = nb, nd
        pb = [math.exp(-(abs(b) - P["bmag"]) ** 2 / (2 * P["bsd"] ** 2)) + P["bunif"] for b in self.B]
        zb = math.fsum(pb); self.PB = [x / zb for x in pb]
        pdd = [math.exp(-d * d / (2 * P["dsd"] ** 2)) for d in self.D]
        zd = math.fsum(pdd); self.PD = [x / zd for x in pdd]
        self.prior = [self.PB[ib] * self.PD[id_] for ib in range(nb) for id_ in range(nd)]
        self.tables = {}
        self.hyps = [to_h(h) for h in P["hyps"]]
        self.hprior = dict(zip(self.hyps, P["hprior"]))
        self.kprior = {to_h(k): p for k, p in P["kprior"].items()}

    def tab(self, mtype, x, sigv):
        key = (mtype, x, sigv)
        if key not in self.tables:
            P = self.P; mu = P["mu"][mtype] + P.get("gam", 0.0) * sigv; nb, nd = self.nb, self.nd
            s0 = [sig(mu + self.D[id_] + self.B[ib] * x) for ib in range(nb) for id_ in range(nd)]
            s1 = [sig(mu + self.D[id_] + self.B[ib] * x + P["delta"]) for ib in range(nb) for id_ in range(nd)]
            self.tables[key] = (s0, s1)
        return self.tables[key]

    def step(self, p, n):
        P = self.P; nb, nd = self.nb, self.nd; qb, qd, j = P["qb"], P["qd"], P["jump"]
        PB = self.PB
        for _ in range(min(n, 30)):
            q = [0.0] * (nb * nd)
            for ib in range(nb):
                lo = ib - 1 if ib > 0 else ib; hi = ib + 1 if ib < nb - 1 else ib
                o, ol, oh = ib * nd, lo * nd, hi * nd
                for id_ in range(nd):
                    q[o + id_] += (1 - 2 * qb) * p[o + id_]
                    q[ol + id_] += qb * p[o + id_]
                    q[oh + id_] += qb * p[o + id_]
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

    def candidates(self, s, ticks, mtype, sz_hyps, sz_ll, ctx=None):
        """Screen hypotheses; return list of (h, SizePost, logprior)."""
        P = self.P
        thr = min(P["pmin"], 0.999 * max(self.hprior.values()))
        cand = [h for h in self.hyps if self.hprior[h] >= thr]
        nep = len(P["epss"])
        out = []
        lmax = max([own_lmax(h) for h in cand] + [1])
        tc = common_ticks(s, ticks, lmax)
        for h in cand:
            kp = self.kprior.get(h) or self.kprior[("fallback",)]
            sp = SizePost(s, sz_hyps, sz_ll, kp, nep)
            sc = math.log(self.hprior[h]) + P["evs"] * screen_ll(s, tc, h, P["swin"], ctx) + sp.logev
            out.append((sc, h, sp))
        out.sort(key=lambda r: (-r[0], hkey(r[1])))
        best = out[0][0]
        kept = [r for r in out if r[0] >= best - P["sgap"]][:P["K"]]
        return kept, set(tc)

    def machine(self, s, c, mtype, ticks, sigv=0, ctx=None):
        P = self.P
        mtype = rtype(P, mtype)
        mh = P.get("maxhist", 0)
        if mh and len(ticks) > mh:  # runtime guard for very long histories: keep the most recent ticks
            ticks = ticks[-mh:]
        A, Bs = self.SS[mtype]
        sz_hyps, sz_ll = size_ll(s, ticks, self.SS[mtype], P["epss"], P.get("nsz", 9))
        kept, tcs = self.candidates(s, ticks, mtype, sz_hyps, sz_ll, ctx)
        gap = min(kept[0][0] - kept[1][0], 20.0) if len(kept) > 1 else 20.0
        self.gf = [0.0, 0.0, gap / 10.0, math.log1p(len(ticks)) / 6.0]
        lev = []; preds = []; recs = []; hs_ = []
        L = P.get("eL", 0)
        for sc, h, sp in kept:
            post = list(self.prior); last = None; le = 0.0; rec = {}
            for t in ticks:
                x = xval(s, t, h, ctx)
                if x is None:
                    if h[0] != "cr": continue
                    x = 0.0
                if last is not None:
                    post = self.step(post, t - last)
                last = t
                s0, s1 = self.tab(mtype, x, 0)
                y = s[t]; pz = sp.pz(t); qz = 1 - pz
                if L and t >= c - L:
                    rec[t] = qz * math.fsum(u * w for u, w in zip(post, s0)) + pz * math.fsum(u * w for u, w in zip(post, s1))
                if y == 0:
                    lik = [qz * (1 - u) + pz * (1 - w) for u, w in zip(s0, s1)]
                else:
                    k = y - 1; ak = qz * A[k]; bk = pz * Bs[k]
                    lik = [ak * u + bk * w for u, w in zip(s0, s1)]
                post = [u * w for u, w in zip(post, lik)]
                z = math.fsum(post)
                if t in tcs or not P.get("evcommon", True):
                    le += math.log(z)
                post = [u / z for u in post]
            x = xval(s, c, h, ctx)
            if x is None and h[0] == "cr": x = 0.0
            if last is None or x is None:
                continue
            post = self.step(post, c - last)
            s0, s1 = self.tab(mtype, x, sigv)
            pzc = sp.pz(c)
            e0 = math.fsum(u * w for u, w in zip(post, s0)); e1 = math.fsum(u * w for u, w in zip(post, s1))
            w0 = (1 - pzc) * e0; w1 = pzc * e1
            preds.append([1 - w0 - w1] + [w0 * A[i] + w1 * Bs[i] for i in range(len(A))])
            lev.append(P["evt"] * le + math.log(self.hprior[h])); recs.append(rec); hs_.append(h)
        if not preds:
            sp = SizePost(s, sz_hyps, sz_ll, self.kprior[("fallback",)], len(P["epss"]))
            pzc = sp.pz(c)
            mu = P["mu"][mtype] + P.get("gam", 0.0) * sigv; e0 = sig(mu); e1 = sig(mu + P["delta"])
            w0 = (1 - pzc) * e0; w1 = pzc * e1
            return [1 - w0 - w1] + [w0 * A[i] + w1 * Bs[i] for i in range(len(A))], []
        mx = max(lev)
        wts = [math.exp(v - mx) for v in lev]; Z = math.fsum(wts)
        self.gf[0] = max(wts) / Z; self.gf[1] = math.fsum(w for w, h in zip(wts, hs_) if h[0] == "none") / Z
        common = sorted(set.intersection(*[set(r) for r in recs]))
        rec = [(t, math.fsum(w * r[t] for w, r in zip(wts, recs)) / Z, 1 if s[t] > 0 else 0) for t in common]
        return [math.fsum(w * p[j] for w, p in zip(wts, preds)) / Z for j in range(len(preds[0]))], rec


def episode_offset(recs, c, P):
    g = 0.5 ** (1.0 / P["ehl"]); obs = []
    for rec in recs:
        for t, pw, y in rec:
            obs.append((g ** (c - 1 - t), logit(pw), y))
    e = 0.0; inv = 1.0 / (P["etau"] ** 2)
    for _ in range(20):
        gr = -e * inv; h = -inv
        for w, l, y in obs:
            p = sig(l + e); gr += w * (y - p); h -= w * p * (1 - p)
        e -= gr / h
    return e


def finalize(p, e, P):
    if e:
        pw = 1 - p[0]; pw2 = sig(logit(pw) + e)
        p = [1 - pw2] + [v * pw2 / pw for v in p[1:]]
    p = [min(max(v, P["floor"]), 1.0) for v in p]
    z = math.fsum(p)
    return [v / z for v in p]


def _work(args):
    P, items = args
    F = WinFilter(P)
    raw = []; gfs = {}
    ctxs = episode_ctx(items)
    for qid, ep, s, c, mt, sg in items:
        ticks = sorted(t for t in s if t < c)
        p, rec = F.machine(s, c, mt, ticks, sg, ctxs[qid])
        raw.append((qid, ep, c, p, rec))
        if P.get("gatefeat"): gfs[qid] = list(F.gf)
    by_ep = defaultdict(list)
    for qid, ep, c, p, rec in raw:
        by_ep[ep].append(rec)
    out = []
    for qid, ep, c, p, rec in raw:
        e = 0.0
        if P.get("eL", 0):
            others = [r for r in by_ep[ep] if r is not rec]
            e = episode_offset(others, c, P)
        out.append((qid, finalize(p, e, P)) if not P.get("gatefeat") else (qid, (finalize(p, e, P), gfs[qid])))
    return out


def episode_ctx(items):
    """Per query: (histories of the episode's machines at ticks < cutoff, sorted by query id; own index)."""
    by = defaultdict(list)
    for it in items:
        by[it[1]].append(it)
    ctx = {}
    for ep, its in by.items():
        its = sorted(its, key=lambda r: r[0]); hs = [{t: v for t, v in it[2].items() if t < it[3]} for it in its]
        for i, it in enumerate(its):
            ctx[it[0]] = (hs, i)
    return ctx


def predict_split(d, P, workers=1):
    qs, H = read_split(d, P.get("payvals"))
    items = [(q["query_id"], q["episode_id"], H[q["machine_id"]], int(q["cutoff"]), q["machine_type"], q["signal"])
             for q in sorted(qs, key=lambda r: r["query_id"])]
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


PAIRS8 = [(a, b) for a in range(1, 9) for b in range(a + 1, 9)]
ALL_HYPS = [("xor", a, b) for a, b in PAIRS8] + [("lag", a) for a in range(1, 9)] + [("none",)]
FULL_HYPS = (ALL_HYPS + [(k, a, b) for k in ("and", "or", "anb", "bna") for a, b in PAIRS8]
             + [("cnt", n) for n in range(2, 9)]
             + [("p3", a, b, c) for a in range(1, 9) for b in range(a + 1, 9) for c in range(b + 1, 9)]
             + [("cr", r, l) for r in range(1, 8) for l in range(1, 9)])

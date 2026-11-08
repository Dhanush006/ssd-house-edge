"""House Edge model C: generic multinomial logistic regression (pure Python inference, deterministic).

Features at tick t use only ticks < t of the same machine (and other machines of the same episode):
  type one-hot, signal, short-history flag, EW win rates (hl 8, 32, logit), EW big-win share,
  for every lag hypothesis h (28 XOR pairs + 8 single lags): x_h(t) * r_h(t), where r_h is the machine's
  own exponentially weighted past agreement between x_h and win/loss (adaptive per-machine polarity),
  the strongest such term (by |r_h|) per type, size-state terms g_k(t) * r^s_k(t) for k = 1..8
  (g_k = +-1 for payout >= 2 at lag k; r^s_k = EW agreement of g_k with big wins), and the episode's
  other machines' last-tick win share minus their EW rate.
Weights are fitted offline (src/fit_c.py) on per-tick examples from organizer histories.
"""
import csv, math
from collections import defaultdict
from pathlib import Path

PAY = (0, 1, 2, 5, 20)
TYPES = ("steady", "balanced", "jackpot")
HYPS = [(a, b) for a in range(1, 9) for b in range(a + 1, 9)] + [(a, 0) for a in range(1, 9)]
HL_R = 40.0
FEAT2 = False
PAIRS8 = [(a, b) for a in range(1, 9) for b in range(a + 1, 9)]
CNTK = (3, 5, 8)
HL_F = 10.0


def read_split(d):
    d = Path(d)
    qs = list(csv.DictReader(open(d / "queries.csv", newline="", encoding="utf-8")))
    H = defaultdict(dict)
    for r in csv.DictReader(open(d / "history.csv", newline="", encoding="utf-8")):
        H[r["machine_id"]][int(r["tick"])] = int(r["payout"])
    return qs, H


def lg(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def xval(s, t, h):
    a, b = h
    va = s.get(t - a)
    if va is None:
        return 0.0
    if b == 0:
        return 1.0 if va > 0 else -1.0
    vb = s.get(t - b)
    if vb is None:
        return 0.0
    return 1.0 if (va > 0) != (vb > 0) else -1.0


def machine_feats(s, mtype, sigv, targets, epwin, ctx=None):
    """Yield (t, feature list) for each t in targets (sorted), using ticks < t only.
    epwin(t) -> other machines' (last-tick win share - EW rate) or 0."""
    ticks = sorted(s)
    lam = 0.5 ** (1.0 / HL_R); g8 = 0.5 ** (1 / 8.0); g32 = 0.5 ** (1 / 32.0)
    nh = len(HYPS)
    num = [0.0] * nh; den = [0.0] * nh
    fnum = [0.0] * nh; fden = [0.0] * nh; lf = 0.5 ** (1.0 / HL_F); g4 = 0.5 ** 0.25; w4 = [0.5, 1.0]
    snum = [0.0] * 8; sden = [0.0] * 8
    w8 = [0.5, 1.0]; w32 = [0.5, 1.0]; bg = [0.5, 1.0]
    nobs = 0; i = 0; out = []
    if FEAT2:
        cw = [[0.0] * 4 for _ in PAIRS8]; cd = [[0.0] * 4 for _ in PAIRS8]
        ncr = [[0.0, 0.0] for _ in range(14)]; ncn = [[0.0, 0.0] for _ in CNTK]
        others = [] if ctx is None else [ctx[0][(ctx[1] + r) % len(ctx[0])] for r in range(1, len(ctx[0]))]

        def cell(tt, a, b):
            va = s.get(tt - a); vb = s.get(tt - b)
            return None if va is None or vb is None else 2 * (va > 0) + (vb > 0)

        def crx(tt, j):
            r, l = divmod(j, 2)
            if r >= len(others): return 0.0
            v = others[r].get(tt - l - 1)
            return 0.0 if v is None else (1.0 if v > 0 else -1.0)

        def cnx(tt, n):
            v = [s.get(tt - j) for j in range(1, n + 1)]
            return 0.0 if None in v else (2.0 * sum(u > 0 for u in v) - n) / n
    oh = [1.0 if mtype == t else 0.0 for t in TYPES]
    for tq in targets:
        while i < len(ticks) and ticks[i] < tq:
            t = ticks[i]; y = s[t]; w = 1.0 if y > 0 else -1.0
            for j, h in enumerate(HYPS):
                x = xval(s, t, h)
                if x:
                    num[j] = num[j] * lam + x * w; den[j] = den[j] * lam + 1.0
                    fnum[j] = fnum[j] * lf + x * w; fden[j] = fden[j] * lf + 1.0
            if y > 0:
                big = 1.0 if y >= 2 else -1.0
                for k in range(1, 9):
                    u = s.get(t - k)
                    if u is not None:
                        snum[k - 1] = snum[k - 1] * lam + (1.0 if u >= 2 else -1.0) * big; sden[k - 1] = sden[k - 1] * lam + 1.0
                bg = [bg[0] * g32 + (1.0 if y >= 2 else 0.0), bg[1] * g32 + 1.0]
            w4 = [w4[0] * g4 + (y > 0), w4[1] * g4 + 1.0]; w8 = [w8[0] * g8 + (y > 0), w8[1] * g8 + 1.0]; w32 = [w32[0] * g32 + (y > 0), w32[1] * g32 + 1.0]
            if FEAT2:
                for j, (a, b) in enumerate(PAIRS8):
                    cc = cell(t, a, b)
                    if cc is not None:
                        cw[j][cc] = cw[j][cc] * lam + (y > 0); cd[j][cc] = cd[j][cc] * lam + 1.0
                for j in range(14):
                    x = crx(t, j)
                    if x: ncr[j][0] = ncr[j][0] * lam + x * w; ncr[j][1] = ncr[j][1] * lam + 1.0
                for j, n in enumerate(CNTK):
                    x = cnx(t, n)
                    if x: ncn[j][0] = ncn[j][0] * lam + x * w; ncn[j][1] = ncn[j][1] * lam + 1.0
            nobs += 1; i += 1
        f = list(oh) + [float(sigv), 1.0 if nobs < 100 else 0.0, lg(w4[0] / w4[1]), lg(w8[0] / w8[1]), lg(w32[0] / w32[1]), bg[0] / bg[1] - 0.5]
        best = 0.0; bestr = 0.0
        for j, h in enumerate(HYPS):
            r = num[j] / (den[j] + 4.0); v = xval(s, tq, h) * r
            f.append(v)
            if abs(r) > bestr:
                bestr = abs(r); best = v
        f += [best * o for o in oh]
        fb = 0.0; fbr = 0.0
        for j, h in enumerate(HYPS):
            r = fnum[j] / (fden[j] + 2.0); v = xval(s, tq, h) * r
            f.append(v)
            if abs(r) > fbr:
                fbr = abs(r); fb = v
        f += [fb * o for o in oh]
        for k in range(1, 9):
            u = s.get(tq - k)
            g = 0.0 if u is None else (1.0 if u >= 2 else -1.0)
            f.append(g * snum[k - 1] / (sden[k - 1] + 4.0))
        f.append(epwin(tq))
        if FEAT2:
            base = w32[0] / w32[1]
            for j, (a, b) in enumerate(PAIRS8):
                cc = cell(tq, a, b)
                f.append(0.0 if cc is None else (cw[j][cc] + 2 * base) / (cd[j][cc] + 2) - base)
            for j in range(14):
                f.append(crx(tq, j) * ncr[j][0] / (ncr[j][1] + 4.0))
            for j, n in enumerate(CNTK):
                f.append(cnx(tq, n) * ncn[j][0] / (ncn[j][1] + 4.0))
        out.append((tq, f))
    return out


def episode_signal(Hs, cutoff_of):
    """Hs: {machine_id: history}. Returns fn(mid, t) = mean over other machines of (win at t-1 - EW win rate before t-1)."""
    g = 0.5 ** (1 / 16.0)
    st = {}
    for m, s in Hs.items():
        rate = {}; a, b = 0.5, 1.0
        for t in sorted(s):
            rate[t] = (1.0 if s[t] > 0 else 0.0) - a / b
            a = a * g + (s[t] > 0); b = b * g + 1.0
        st[m] = rate
    mids = sorted(Hs)

    def f(mid, t):
        v = [st[o][t - 1] for o in mids if o != mid and (t - 1) in st[o] and t - 1 < cutoff_of[o]]
        return sum(v) / len(v) if v else 0.0
    return f


NF = 3 + 6 + 2 * (len(HYPS) + 3) + 8 + 1


def softmax_pred(W, f):
    """W: list of 5 rows (bias + NF weights); class 0 row fixed at zero."""
    z = [w[0] + math.fsum(a * b for a, b in zip(w[1:], f)) for w in W]
    m = max(z); e = [math.exp(v - m) for v in z]; Z = math.fsum(e)
    return [v / Z for v in e]


def tree_val(n, f):
    while isinstance(n, list):
        n = n[2] if f[n[0]] <= n[1] else n[3]
    return n


def gbm_pred(T, f):
    z = [0.0] * 5
    for i, tr in enumerate(T):
        z[i % 5] += tree_val(tr, f)
    m = max(z); e = [math.exp(v - m) for v in z]; Z = math.fsum(e)
    return [v / Z for v in e]


def predict_items(items, P):
    """items: (qid, episode, history{t:payout}, cutoff, type, signal); histories may include ticks >= cutoff."""
    global HL_R, HL_F
    global FEAT2
    HL_R = P.get("hlr", 40.0); HL_F = P.get("hlf", 10.0); FEAT2 = P.get("feat2", False)
    by_ep = defaultdict(list)
    for it in items:
        by_ep[it[1]].append(it)
    out = {}
    for ep in sorted(by_ep):
        qq = sorted(by_ep[ep], key=lambda r: r[0])
        cut = {it[0]: it[3] for it in qq}
        Hs = {it[0]: {t: v for t, v in it[2].items() if t < it[3]} for it in qq}
        es = episode_signal(Hs, cut); hl = [Hs[it[0]] for it in qq]
        for n, (qid, _, _, c, mt, sg) in enumerate(qq):
            (_, f), = machine_feats(Hs[qid], mt, sg, [c], lambda t, q=qid: es(q, t), (hl, n))
            if "gbm" in P:
                p = gbm_pred(P["gbm"], f)
            else:
                f = [(a - mu) / s_ for a, mu, s_ in zip(f, P["mean"], P["sd"])]
                p = softmax_pred(P["W"], f)
            p = [min(max(v, 1e-6), 1.0) for v in p]; z = math.fsum(p)
            out[qid] = [v / z for v in p]
    return out


def predict_split(d, P, workers=1):
    qs, H = read_split(d)
    items = [(q["query_id"], q["episode_id"], H[q["machine_id"]], int(q["cutoff"]), q["machine_type"], int(q["signal"]))
             for q in qs]
    return predict_items(items, P)

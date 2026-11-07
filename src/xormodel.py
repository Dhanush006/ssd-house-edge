"""Pure-stdlib XOR-pair tracking model."""
import csv, math
from collections import defaultdict
from pathlib import Path
PAY = (0,1,2,5,20)
PAIRS3 = ((1,3),(2,5),(4,7))
ALLPAIRS = tuple((a,b) for a in range(1,9) for b in range(a+1,9))

def read_split(d):
    d = Path(d)
    qs = list(csv.DictReader(open(d/"queries.csv")))
    H = defaultdict(dict)
    for r in csv.DictReader(open(d/"history.csv")):
        H[r["machine_id"]][int(r["tick"])] = int(r["payout"])
    return qs, H

TYPE_SIZE = {  # P(size | win), placeholder; refit later
 "steady":[0.35/0.577,0.181/0.577,0.045/0.577,0.001/0.577],
 "balanced":[0.143/0.416,0.193/0.416,0.069/0.416,0.013/0.416],
 "jackpot":[0.046/0.22,0.065/0.22,0.081/0.22,0.028/0.22]}

SIZE_STATES = {
 "steady": [[0.753989389636146, 0.19680848829108313, 0.04821807963131535, 0.0009840424414554146],
            [0.4551220076282822, 0.43590239389737434, 0.10679608650485668, 0.002179511969486871]],
 "balanced": [[0.4998531084689386, 0.3479282723694339, 0.13047310213853774, 0.021745517023089637],
              [0.2140661471457723, 0.5467365932898975, 0.20502622248371163, 0.03417103708061861]],
 "jackpot": [[0.3351460319013039, 0.2449461987732039, 0.31493082699411934, 0.10497694233137306],
             [0.12078673011007354, 0.3239206783804992, 0.4164694436320704, 0.13882314787735675]]}

def size_dist(s, ticks, c, mtype, P):
    A, B = SIZE_STATES[mtype]; rho = P["rho"]
    p1 = 0.5; last = None
    for t in ticks:
        v = s[t]
        if v == 0: continue
        if last is not None:
            p1 = 0.5 + (p1 - 0.5) * (1 - 2 * rho) ** (t - last)
        last = t
        i = PAY.index(v) - 1
        a = (1 - p1) * A[i]; b = p1 * B[i]
        p1 = b / (a + b)
    if last is not None:
        p1 = 0.5 + (p1 - 0.5) * (1 - 2 * rho) ** (c - last)
    return [(1 - p1) * x + p1 * y for x, y in zip(A, B)]

def predict_one(s, c, mtype, P):
    hl = P["hl"]; g = 0.5**(1.0/hl)
    pairs = P["pairs"]
    ticks = sorted(t for t in s if t < c)
    # per pair sequential tracking
    logpost = []; rates = []
    pw_prior = P["a0"]
    for (a,b) in pairs:
        n = [0.0,0.0]; w = [0.0,0.0]; ll = 0.0; last = None
        for t in ticks:
            va = s.get(t-a); vb = s.get(t-b)
            if va is None or vb is None: continue
            if last is not None:
                f = g**(t-last); n=[x*f for x in n]; w=[x*f for x in w]; ll*=P["llg"]**(t-last)
            last = t
            cell = (va>0) != (vb>0)
            p = (w[cell]+pw_prior*P["k"]) / (n[cell]+P["k"])
            y = s[t]>0
            ll += math.log(p if y else 1-p)
            n[cell]+=1; w[cell]+=y
        va = s.get(c-a); vb = s.get(c-b)
        if va is None or vb is None or last is None:
            logpost.append(-1e9); rates.append(pw_prior); continue
        f = g**(c-last); n=[x*f for x in n]; w=[x*f for x in w]
        cell = (va>0) != (vb>0)
        rates.append((w[cell]+pw_prior*P["k"])/(n[cell]+P["k"]))
        logpost.append(ll)
    mx = max(logpost); ws = [math.exp(l-mx) for l in logpost]; Z=sum(ws)
    pwin = sum(wi*r for wi,r in zip(ws,rates))/Z
    sz = size_dist(s, ticks, c, mtype, P)
    pwin = min(max(pwin, P["eps"]), 1-P["eps"])
    return [1-pwin]+[pwin*x for x in sz]

DEFAULT = dict(hl=40.0, pairs=PAIRS3, a0=0.5, k=1.0, llg=0.98, ks=10.0, hls=60.0, eps=0.005, rho=0.01)
def predict_split(d, P=DEFAULT):
    qs,H = read_split(d)
    return {q["query_id"]: predict_one(H[q["machine_id"]], int(q["cutoff"]), q["machine_type"], P) for q in qs}

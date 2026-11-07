import pandas as pd, numpy as np
from pathlib import Path
PUB = Path(__file__).resolve().parents[1] / "kit/public"
PAY = [0,1,2,5,20]
def load(split):
    q = pd.read_csv(PUB/split/"queries.csv")
    h = pd.read_csv(PUB/split/"history.csv")
    out = {"q": q, "h": h}
    for f in ["labels","conditions","targets","reference"]:
        p = PUB/split/f"{f}.csv"
        if p.exists(): out[f] = pd.read_csv(p)
    return out

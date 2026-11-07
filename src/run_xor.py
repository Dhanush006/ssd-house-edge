import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from xormodel import *; from score import score, PUB
base = dict(DEFAULT, hl=16.0, k=0.5)
for g in [dict(rho=0.0),dict(rho=0.003),dict(rho=0.01),dict(rho=0.03)]:
    P = dict(base, **g); print(g, end=" "); score(predict_split(PUB, P))

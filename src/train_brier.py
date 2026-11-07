import sys,os,json,csv; sys.path.insert(0, os.path.dirname(__file__))
from pathlib import Path
from collections import defaultdict
import puremodel as M, xormodel as X
d=Path(__file__).resolve().parents[1]/"kit/public/train"
lab={r["query_id"]:int(r["outcome"]) for r in csv.DictReader(open(d/"labels.csv"))}
cond={r["query_id"]:r["condition"] for r in csv.DictReader(open(d/"conditions.csv"))}
def brier(o):
    e=defaultdict(list)
    for q,p in o.items():
        y=[1.0 if v==lab[q] else 0.0 for v in (0,1,2,5,20)]
        e[cond[q]].append(sum((a-b)**2 for a,b in zip(p,y)))
    return {k[:5]:round(sum(v)/len(v),4) for k,v in sorted(e.items())}, round(sum(sum(v) for v in e.values())/768,4)
if __name__=="__main__":
    for f in sys.argv[1:]:
        P=json.load(open(f)) if f.endswith(".json") else dict(M.DEFAULT,**json.loads(f))
        print(f[:60], brier(M.predict_split(d,P,workers=6)))

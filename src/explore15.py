import sys,os; sys.path.insert(0, os.path.dirname(__file__))
from load import *
from xormodel import read_split
from score import PUB
d=load("validation"); q=d["q"].merge(d["targets"].drop(columns=["episode_id"])).merge(d["conditions"])
qs,H=read_split(PUB)
q["tw"]=1-q.p_0; q["s1"]=q.p_1/q.tw
q["st"]=(q.s1<q.groupby("machine_type").s1.transform("mean")).astype(int)
q["twb"]=pd.cut(q.tw,[0,.1,.3,.5,.7,.9,1])
print(pd.crosstab([q.machine_type,q.twb],q.st))
# per machine history: P(big|win) in high-cell vs low-cell, using best pair

import sys,os; sys.path.insert(0, os.path.dirname(__file__))
from load import *
from xormodel import read_split
from score import PUB
from sklearn.linear_model import LogisticRegression
import warnings; warnings.filterwarnings("ignore")
d=load("validation"); q=d["q"].merge(d["targets"].drop(columns=["episode_id"])).merge(d["conditions"])
qs,H=read_split(PUB)
q["tw"]=1-q.p_0; q["s1"]=q.p_1/q.tw
q["st"]=(q.s1<q.groupby("machine_type").s1.transform("mean")).astype(int)
def feats(s,t,lags,kind):
    f=[]
    for k in lags:
        v=s[t-k]
        if kind=="big": f.append(float(v>=2))
        elif kind=="cat": f+= [float(v==1), float(v>=2)]
        elif kind=="lv": f.append(np.log1p(v))
    return f
for lags in [(1,2,4),(1,2,3,4),(1,2,4,8),tuple(range(1,9))]:
  for kind in ["big","cat","lv"]:
    for C in [0.3,1.0]:
        pr=[]
        for _,r in q.iterrows():
            s=H[r["machine_id"]]; c=r["cutoff"]
            X=[];Y=[]
            for t in s:
                if t<c and s[t]>0 and all(t-k in s for k in lags): X.append(feats(s,t,lags,kind)); Y.append(s[t]>=2)
            if len(set(Y))<2: pr.append(0.5); continue
            m=LogisticRegression(C=C).fit(X,Y); pr.append(m.predict_proba([feats(s,c,lags,kind)])[0,1]-np.mean(Y))
        print(lags,kind,C, round(np.corrcoef(pr,q.st)[0,1],3))

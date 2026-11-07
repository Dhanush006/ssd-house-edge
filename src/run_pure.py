import sys,os,json,time; sys.path.insert(0, os.path.dirname(__file__))
import puremodel as M
from score import score, PUB
if __name__=="__main__":
    for g in json.loads(sys.argv[1]):
        P=dict(M.DEFAULT,**g); t=time.time(); o=M.predict_split(PUB,P,workers=int(os.environ.get("W","4")))
        print(g,f"{time.time()-t:.0f}s",end=" "); score(o)

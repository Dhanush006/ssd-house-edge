import sys,os,json; sys.path.insert(0, os.path.dirname(__file__))
import puremodel as M
from score import score, PUB
from comp import comp
if __name__=='__main__':
  P=json.load(open(sys.argv[1])); P.update(json.loads(sys.argv[2]) if len(sys.argv)>2 else {})
  o=M.predict_split(PUB,P,workers=6); score(o)
  print(comp({k:1-v[0] for k,v in o.items()},{k:[x/(1-v[0]) for x in v[1:]] for k,v in o.items()}))

"""python src/oracle_score.py <kit_public> <params.json> [<params2.json> ...]: validation points of B params (6 workers)."""
import sys, json
sys.path.insert(0, "src")
import r2score as S, model_b as M
if __name__ == "__main__":
    pub = sys.argv[1]; S.set_schema(pub); Sc = S.Scorer(pub)
    for pf in sys.argv[2:]:
        print(Sc.line(pf.split("/")[-2] + "/" + pf.split("/")[-1][:14], M.predict_split(pub + "/validation", json.load(open(pf)), 4)))

"""Fit model C weights: usage  python src/fit_c.py <kit_public_dir> <out_params.json> [--lam 1e-4] [--trainonly]
Examples: every observed tick t >= 9 of every train machine (label = payout at t; features from ticks < t), plus
the train label at the cutoff; and (unless --trainonly) every observed tick of validation histories (ticks < the
validation cutoff, so validation query outcomes are never used). Multinomial logistic regression, L2, L-BFGS."""
import sys, os, csv, json, time
sys.path.insert(0, os.path.dirname(__file__))
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.optimize import minimize
import model_c as M
import kitschema as KS
COLS, VALS = None, None


def examples(pub, split, use_label):
    d = Path(pub) / split
    if not (d / "queries.csv").exists():
        return [], [], []
    qs, H = M.read_split(d, VALS)
    lab = KS.read_labels(d, VALS) if use_label else {}
    by_ep = defaultdict(list)
    for q in qs: by_ep[q["episode_id"]].append(q)
    X = []; Y = []; G = []
    for ep in sorted(by_ep):
        qq = sorted(by_ep[ep], key=lambda r: r["query_id"])
        Hs = {}; cut = {}
        for q in qq:
            c = int(q["cutoff"]); s = {t: v for t, v in H[q["machine_id"]].items() if t < c}
            if q["query_id"] in lab: s[c] = lab[q["query_id"]]
            Hs[q["machine_id"]] = s; cut[q["machine_id"]] = c + 1
        es = episode_signal = M.episode_signal(Hs, cut)
        hl = [Hs[q["machine_id"]] for q in qq]
        for n, q in enumerate(qq):
            m = q["machine_id"]; s = Hs[m]
            tg = [t for t in sorted(s) if t >= 9]
            for t, f in M.machine_feats(s, q["machine_type"], q["signal"], tg, lambda t, m=m: es(m, t), (hl, n)):
                X.append(f); Y.append(s[t]); G.append(ep)
    return X, Y, G


def fit(X, Y, lam):
    X = np.asarray(X); Y = np.asarray(Y); n, p = X.shape
    mean = X.mean(0); sd = X.std(0); sd[sd < 1e-9] = 1.0
    Z = np.column_stack([np.ones(n), (X - mean) / sd]); Yo = np.eye(5)[Y]

    def f(w):
        W = np.vstack([np.zeros(p + 1), w.reshape(4, p + 1)])
        S = Z @ W.T; S -= S.max(1, keepdims=True); E = np.exp(S); Pr = E / E.sum(1, keepdims=True)
        ll = -(np.log(Pr[np.arange(n), Y] + 1e-300)).sum() / n
        reg = lam * (W[:, 1:] ** 2).sum()
        g = ((Pr - Yo).T @ Z / n)[1:]; g[:, 1:] += 2 * lam * W[1:, 1:]
        return ll + reg, g.ravel()
    r = minimize(f, np.zeros(4 * (p + 1)), jac=True, method="L-BFGS-B", options={"maxiter": 500})
    W = np.vstack([np.zeros(p + 1), r.x.reshape(4, p + 1)])
    return W, mean, sd, r.fun


def fit_gbm(X, Y, G, nround=None, nc=5):
    """LightGBM multiclass; rounds chosen on an episode-grouped holdout (every 5th episode), then refit on all.
    Trees exported as nested lists for pure-Python inference."""
    import lightgbm as lgb
    X = np.asarray(X); Y = np.asarray(Y); eps = sorted(set(G)); ho = {e for i, e in enumerate(eps) if i % 5 == 4}
    m = np.array([g in ho for g in G])
    prm = dict(objective="multiclass", num_class=nc, learning_rate=0.05, num_leaves=31, min_data_in_leaf=200,
               feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0,
               deterministic=True, num_threads=4)
    if nround is None:
        b = lgb.train(prm, lgb.Dataset(X[~m], Y[~m]), 2000, valid_sets=[lgb.Dataset(X[m], Y[m])],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        nround = b.best_iteration; loss = b.best_score["valid_0"]["multi_logloss"]
    b = lgb.train(prm, lgb.Dataset(X, Y), nround)
    trees = []
    def conv(n):
        if "leaf_value" in n: return float(n["leaf_value"])
        return [int(n["split_feature"]), float(n["threshold"]), conv(n["left_child"]), conv(n["right_child"])]
    for ti in b.dump_model()["tree_info"]:
        trees.append(conv(ti["tree_structure"]))
    print(f"fit_c gbm rounds={nround} holdout_logloss={loss:.4f}")
    return {"gbm": trees, "nclass": nc, "n": int(len(Y))}, loss


def main():
    pub, out = sys.argv[1], Path(sys.argv[2])
    lam = float(sys.argv[sys.argv.index("--lam") + 1]) if "--lam" in sys.argv else 1e-4
    t0 = time.time()
    arg = lambda k, d: float(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
    M.HL_R = arg("--hlr", 40.0); M.HL_F = arg("--hlf", 10.0); M.FEAT2 = "--feat1" not in sys.argv
    global COLS, VALS
    COLS, VALS = KS.find_schema(Path(pub))
    tc = defaultdict(int)
    for sp in ("train", "validation"):
        if (Path(pub) / sp / "queries.csv").exists():
            for q in KS.read_split(Path(pub) / sp, VALS)[0]: tc[q["machine_type"]] += 1
    ty = sorted(t for t, n in tc.items() if n >= 30 and t != "_pool")
    M.TYPES = ("steady", "balanced", "jackpot") if set(ty) == {"steady", "balanced", "jackpot"} else tuple(ty)
    X, Y, G = ([], [], []) if "--valonly" in sys.argv else examples(pub, "train", True)
    if "--trainonly" not in sys.argv:
        X2, Y2, G2 = examples(pub, "validation", False); X += X2; Y += Y2; G += G2
    if "--gbm" in sys.argv:
        P, loss = fit_gbm(X, Y, G, nc=len(VALS))
    else:
        W, mean, sd, loss = fit(X, Y, lam)
        P = {"W": W.tolist(), "mean": mean.tolist(), "sd": sd.tolist(), "lam": lam, "n": len(Y)}
    P["hlr"] = M.HL_R; P["hlf"] = M.HL_F; P["feat2"] = M.FEAT2; P["types"] = list(M.TYPES); P["payvals"] = list(VALS); P["cols"] = list(COLS)
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(P) + "\n")
    print(f"fit_c n={len(Y)} nf={len(X[0])} lam={lam} loss={loss:.4f} time={time.time() - t0:.0f}s wrote {out}")


if __name__ == "__main__":
    main()

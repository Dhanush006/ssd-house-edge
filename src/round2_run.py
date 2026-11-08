"""Round-2 one-command runner (restartable; prints summary lines only).

  python src/round2_run.py <kit_dir> [--tag NAME] [--force] [--workers 6]
      fits B (+ naive B), C (+ C_dev for blend weights), runs the starter (D), predicts validation + test
      for A, B, Bn, C, D, scores validation (fast replica + official checker), picks a blend on the train
      dev split (observed-outcome Brier; never on validation), writes runs/R2-<tag>/<model>/.
  python src/round2_run.py <kit_dir> --package <A|B|C|blend> [--tag NAME]
      builds runs/R2-<tag>/<model>/entry (+ entry.zip), reruns from a fresh folder with Python 3.13 -I
      and checks the CSV is byte-identical, runs the checker on test + validation.
<kit_dir> is the organizer kit folder (with public/ and check_submission.py) or its public/ folder.
"""
import sys, os, json, csv, time, shutil, subprocess, random, importlib, math
from pathlib import Path
from collections import defaultdict

HE = Path(__file__).resolve().parents[1]; SRC = HE / "src"; AENT = HE / "runs/06-lowjump/entry"
sys.path.insert(0, str(SRC)); sys.path.insert(0, str(AENT))
import r2score as S

PY = sys.executable
CONDS = S.CONDS
MODS = {"A": ("model", AENT / "model.py"), "B": ("model_b", SRC / "model_b.py"), "Bn": ("model_b", SRC / "model_b.py"),
        "Bt": ("model_b", SRC / "model_b.py"), "B0": ("model_b", SRC / "model_b.py"), "C": ("model_c", SRC / "model_c.py")}


def log(*a):
    print("[r2]", *a, flush=True)


def paths(kit):
    kit = Path(kit).resolve()
    pub = kit / "public" if (kit / "public" / "metadata.json").exists() else kit
    kd = pub.parent if (pub.parent / "check_submission.py").exists() else HE / "kit"
    return pub, kd


def sh(cmd, cwd=None, soft=False):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0 and soft:
        log("WARN step failed (continuing):", " ".join(map(str, cmd))[:200], (r.stderr or r.stdout).strip()[-300:]); return ""
    if r.returncode != 0:
        log("FAILED", " ".join(map(str, cmd))); print(r.stdout[-2000:], r.stderr[-2000:]); sys.exit(1)
    return r.stdout


def mod(name):
    return importlib.import_module(MODS[name][0])


# ---------------- dev split (train pseudo-queries, observed outcomes) ----------------
JS = (0, 2, 4, 6, 8, 10, 12, 14)


def dev_items(pub):
    tr = pub / "train"
    qs, H = mod("B").read_split(tr)
    lab = {r["query_id"]: int(r["outcome"]) for r in csv.DictReader(open(tr / "labels.csv"))}
    cond = {r["query_id"]: r["condition"] for r in csv.DictReader(open(tr / "conditions.csv"))}
    epc = {q["episode_id"]: cond[q["query_id"]] for q in qs}
    split = {}
    for c in CONDS:
        for i, e in enumerate(sorted(e for e, cc in epc.items() if cc == c)):
            split[e] = "confirm" if i % 3 == 2 else "dev"
    items, meta = [], {}
    for q in sorted(qs, key=lambda r: r["query_id"]):
        s = H[q["machine_id"]]; c = int(q["cutoff"])
        for j in JS:
            cj = c - j; y = lab[q["query_id"]] if j == 0 else s.get(cj)
            if y is None: continue
            hist = {t: v for t, v in s.items() if t < cj}
            qid = f'{q["query_id"]}@{j}'
            items.append((qid, f'{q["episode_id"]}@{j}', hist, cj, q["machine_type"], int(q["signal"])))
            meta[qid] = (q["episode_id"], epc[q["episode_id"]], split[q["episode_id"]], y)
    return items, meta


def pool_predict(M, items, P, workers):
    eps = sorted(set(it[1] for it in items)); slot = {e: i % workers for i, e in enumerate(eps)}
    chunks = [[it for it in items if slot[it[1]] == i] for i in range(workers)]
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(workers) as pool:
        res = pool.map(M._work, [(P, ch) for ch in chunks])
    out = {}
    for r in res: out.update(dict(r))
    return out


def dev_brier(pred, meta, which):
    ep = defaultdict(list)
    for q, p in pred.items():
        e, c, sp, y = meta[q]
        if sp != which: continue
        ep[(c, e)].append(sum((a - (1.0 if v == y else 0.0)) ** 2 for a, v in zip(p, S.PK)))
    per = {c: [sum(v) / len(v) for k, v in ep.items() if k[0] == c] for c in CONDS}
    return sum(sum(v) / len(v) for v in per.values()) / 4, {c: sum(v) / len(v) for c, v in per.items()}


def mix(preds, ws):
    tot = sum(ws.values())
    return {q: [sum(ws[m] / tot * preds[m][q][j] for m in ws) for j in range(5)] for q in preds[next(iter(ws))]}


# ---------------- main evaluation ----------------
def evaluate(kit, tag, force, workers, tune=True):
    pub, kd = paths(kit); out = HE / "runs" / f"R2-{tag}"; out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((pub / "metadata.json").read_text())
    log(f"kit={pub} round={meta.get('round')} max_points={meta.get('max_points')} out={out}")
    t0 = time.time()
    # fits
    for name, args in (("B", []), ("B0", ["--base"]), ("Bn", ["--naive"])):
        pf = out / name / "params.json"
        if force or not pf.exists():
            o = sh([PY, str(SRC / "fit_b.py"), str(pub), str(pf)] + args)
            if name == "B":
                for l in o.splitlines():
                    if l.startswith("fit_b") and "wrote" not in l: log(l)
    if not (out / "A" / "params.json").exists():
        (out / "A").mkdir(exist_ok=True); shutil.copy(AENT / "params.json", out / "A" / "params.json")
    for name, args in (("C", []), ("C_dev", ["--valonly"])):
        pf = out / name / "params.json"
        if force or not pf.exists():
            o = sh([PY, str(SRC / "fit_c.py"), str(pub), str(pf), "--hlf", "20", "--gbm"] + args)
            log(name, o.strip().splitlines()[-1][:120])
    log(f"fits done {time.time() - t0:.0f}s")
    # starter D
    dd = out / "D"
    if force or not (dd / "submission.csv").exists():
        sh([PY, "run_starter.py", "--data", str(pub), "--out", str(dd), "--method", "pooled_recent"], cwd=kd, soft=True)
    # predictions
    preds = {}
    for name in ("A", "B", "B0", "Bn", "C"):
        P = json.loads((out / name / "params.json").read_text()); M = mod(name)
        for split, fn in (("validation", "validation_predictions.csv"), ("test", "submission.csv")):
            f = out / name / fn
            if force or not f.exists():
                t = time.time()
                if name == "C":
                    pr = M.predict_split(pub / split, P)
                else:
                    pr = pool_predict(M, read_items(M, pub / split), P, workers)
                S.write_csv(f, pr); log(f"pred {name} {split} {time.time() - t:.0f}s ({workers if name != 'C' else 1} proc)")
        preds[name] = S.load_csv(out / name / "validation_predictions.csv")
    if (dd / "validation_predictions.csv").exists():
        preds["D"] = S.load_csv(dd / "validation_predictions.csv")
    Sc = S.Scorer(pub)
    log("---- validation (fast replica) ----")
    for name in ("A", "B", "B0", "Bn", "C", "D"):
        if name in preds: log(Sc.line(name, preds[name]))
    log("---- official checker ----")
    for name in ("A", "B", "C"):
        o = sh([PY, "check_submission.py", str(out / name / "validation_predictions.csv"), "--data", str(pub), "--split", "validation"], cwd=kd, soft=True)
        pts = [l for l in o.splitlines() if "points" in l.lower()]
        log(name, pts[0].strip() if pts else o.strip()[-200:])
    # dev split blend
    dp = out / "dev_preds.json"
    J = json.loads(dp.read_text()) if (dp.exists() and not force) else {"preds": {}}
    need = [n for n in ("A", "B", "C") if n not in J["preds"]]
    if need:
        items, dmeta = dev_items(pub); J["meta"] = dmeta
        for name in need:
            P = json.loads((out / ("C_dev" if name == "C" else name) / "params.json").read_text()); M = mod(name)
            t = time.time()
            J["preds"][name] = M.predict_items(items, P) if name == "C" else pool_predict(M, items, P, workers)
            log(f"dev pred {name} {time.time() - t:.0f}s")
        dp.write_text(json.dumps(J))
    J = json.loads(dp.read_text()); dev, dmeta = J["preds"], {k: tuple(v) for k, v in J["meta"].items()}
    if tune:
        tune_b(out, pub, dev, dmeta, workers, force)
        if (out / "Bt" / "params.json").exists():
            dev["Bt"] = json.loads((out / "Bt" / "dev_preds.json").read_text())
            P = json.loads((out / "Bt" / "params.json").read_text()); M = mod("Bt")
            for split, fn in (("validation", "validation_predictions.csv"), ("test", "submission.csv")):
                if force or not (out / "Bt" / fn).exists():
                    S.write_csv(out / "Bt" / fn, pool_predict(M, read_items(M, pub / split), P, workers))
            preds["Bt"] = S.load_csv(out / "Bt" / "validation_predictions.csv")
            log(Sc.line("Bt", preds["Bt"]))
    log("---- dev split (train pseudo-queries, Brier x1e3; C fit on validation histories only) ----")
    for name in [m for m in ("A", "B", "Bt", "C") if m in dev]:
        m, pc = dev_brier(dev[name], dmeta, "dev"); mc, _ = dev_brier(dev[name], dmeta, "confirm")
        log(f"{name}: dev={1e3 * m:.2f} confirm={1e3 * mc:.2f} " + " ".join(f"{c[:6]}={1e3 * v:.2f}" for c, v in pc.items()))
    best = None
    cands = [m for m in ("A", "B", "Bt", "C") if m in dev]
    for a, b in [(x, y) for i, x in enumerate(cands) for y in cands[i + 1:]]:
        for i in range(11):
            w = i / 10; ws = {a: round(1 - w, 1), b: w}
            m, _ = dev_brier(mix(dev, ws), dmeta, "dev")
            if best is None or m < best[0] - 1e-12: best = (m, ws)
    ws = {k: v for k, v in best[1].items() if v > 0}
    mc, _ = dev_brier(mix(dev, ws), dmeta, "confirm")
    bl = mix(preds, ws)
    (out / "blend").mkdir(exist_ok=True)
    (out / "blend" / "weights.json").write_text(json.dumps(ws) + "\n")
    S.write_csv(out / "blend" / "validation_predictions.csv", bl)
    tp = {m: S.load_csv(out / m / "submission.csv") for m in ws}
    S.write_csv(out / "blend" / "submission.csv", mix(tp, ws))
    log(f"blend (dev-selected) {ws}: dev={1e3 * best[0]:.2f} confirm={1e3 * mc:.2f}")
    log(Sc.line("blend", bl))
    for a, b in (("B", "C"),):
        log(Sc.line(f"{a}+{b} 50/50", mix(preds, {a: .5, b: .5})))
    # ---- recommendation (conservative: dev + confirm Brier on the new kit's train split) ----
    allm = {m: dev[m] for m in cands}; allm["blend"] = mix(dev, ws)
    rows = {}
    for m, pr in allm.items():
        d_, pc = dev_brier(pr, dmeta, "dev"); c_, _ = dev_brier(pr, dmeta, "confirm"); rows[m] = (d_, c_, (d_ + c_) / 2)
    bestm = min(rows, key=lambda m: (rows[m][2], m)); tol = float(os.environ.get("R2_TOL", "0.0003"))
    pref = [m for m in ("B", "Bt", "blend", "C", "A") if m in rows]  # conservative: simplest within noise of the best
    pick = next(m for m in pref + [bestm] if rows[m][2] <= rows[bestm][2] + tol)
    vp = {m: Sc.score(preds[m])[0] for m in preds}; vp["blend"] = Sc.score(bl)[0]
    vr = {m: Sc.score(preds[m])[2] for m in preds}
    Pb = json.loads((out / "B" / "params.json").read_text()); dg = Pb.get("diag", {})
    flags = []
    th = dg.get("top_h", [])
    fam = defaultdict(float)
    for h, w in zip(Pb["hyps"], Pb["hprior"]):
        k = h if isinstance(h, str) else f"{h[0]},{h[1]}"
        fam[k.split(":")[0] if ":" in k else ("none" if k == "0,0" else ("lag" if k.endswith(",0") else "xor"))] += w
    fams = sorted(fam.items(), key=lambda kv: -kv[1])
    if fams and fams[0][1] < 0.35: flags.append("NO_DOMINANT_RULE_FAMILY (B's family may not contain the rule)")
    if th and sum(v for _, v in th[:12]) < 0.15 and fams and fams[0][0] in ("none",): flags.append("NO_SHARP_HYPOTHESES")
    if dg.get("none_mass", 0) > 0.2: flags.append(f"NONE_MASS={dg['none_mass']} (many machines look structureless)")
    if max(dg.get("max_dev", {"x": 0}).values()) > 0.02: flags.append(f"SIZE_TABLES_MISFIT max_dev={dg['max_dev']} (more than 2 size states?)")
    if "C" in rows and rows["C"][2] < rows["B"][2]: flags.append("C_BEATS_B_ON_DEV (structured model misspecified)")
    dif = sum(sum(abs(a - b) for a, b in zip(preds["B"][q], preds["C"][q])) for q in preds["B"]) / len(preds["B"])
    if dif > 0.15: flags.append(f"B_C_DISAGREE mean L1={dif:.3f}")
    if pick != "blend" and pick in vp and max(vp.values()) - vp[pick] > 1.0: flags.append("VALIDATION_DISAGREES_WITH_DEV_PICK (check)")
    log("==== DIAGNOSTICS ====")
    log("top win hypotheses (prior mass):", " ".join(f"({k}):{v}" for k, v in th))
    log("rule families (prior mass):", " ".join(f"{k}:{v:.3f}" for k, v in fams[:6]))
    log("size-lag marginal k0..k8:", dg.get("k_marg"), "| size table max_dev:", dg.get("max_dev"), "| resid_sd:", dg.get("resid_sd"))
    for m in sorted(vr): log(f"val ratios {m:<5}", " ".join(f"{c[:6]}={vr[m][c]:.4f}" for c in CONDS))
    log("==== RECOMMENDATION ====")
    for m in sorted(rows, key=lambda m: rows[m][2]):
        log(f"{m:<6} dev={1e3 * rows[m][0]:.2f} confirm={1e3 * rows[m][1]:.2f} mean={1e3 * rows[m][2]:.2f} val_points={vp.get(m, float('nan')):.3f}")
    log(f"best by dev+confirm = {bestm}; tolerance {tol * 1e3:.2f}e-3 -> conservative pick")
    log(f"PICK = {pick}" + (f" weights={ws}" if pick == "blend" else ""), "| flags:", "; ".join(flags) or "none")
    (out / "summary.json").write_text(json.dumps({"val_points": vp, "dev": {m: r[0] for m, r in rows.items()},
        "confirm": {m: r[1] for m, r in rows.items()}, "pick": pick, "blend": ws, "flags": flags, "diag": dg}, indent=1) + "\n")
    log(f"done {time.time() - t0:.0f}s. package with: {PY} src/round2_run.py {kit} --tag {tag} --package {pick}")


def tune_b(out, pub, dev, dmeta, workers, force, ndraw=int(os.environ.get("R2_NDRAW", "8"))):
    """Bounded random search of B's filter rates on the DEV split only; keep the best draw as Bt only if it
    also beats B on the untouched confirm split."""
    td = out / "Bt"
    if (td / "done").exists() and not force:
        return
    td.mkdir(exist_ok=True)
    base = json.loads((out / "B" / "params.json").read_text()); M = mod("B")
    items, _ = dev_items(pub)
    b_dev, _ = dev_brier(dev["B"], dmeta, "dev"); b_conf, _ = dev_brier(dev["B"], dmeta, "confirm")
    rng = random.Random(0); best = None
    for i in range(ndraw):
        P = dict(base)
        P["qb"] = round(math.exp(rng.uniform(math.log(.003), math.log(.03))), 5)
        P["jump"] = round(math.exp(rng.uniform(math.log(.0004), math.log(.006))), 5)
        P["qd"] = round(math.exp(rng.uniform(math.log(.0008), math.log(.008))), 5)
        P["etau"] = round(rng.uniform(.2, .5), 3)
        pr = pool_predict(M, items, P, workers); m, _ = dev_brier(pr, dmeta, "dev")
        log(f"tune {i}: qb={P['qb']} jump={P['jump']} qd={P['qd']} etau={P['etau']} dev={1e3 * m:.2f} (B {1e3 * b_dev:.2f})")
        if best is None or m < best[0]: best = (m, P, pr)
    gr = base.get("gam_raw", 0.0)
    if abs(gr) > 0.1:  # signal term: candidate only, kept if dev says so
        P = dict(base); P["gam"] = gr
        pr = pool_predict(M, items, P, workers); m, _ = dev_brier(pr, dmeta, "dev")
        log(f"tune gam={gr}: dev={1e3 * m:.2f}")
        if m < best[0]: best = (m, P, pr)
    mc, _ = dev_brier(best[2], dmeta, "confirm")
    ok = best[0] < b_dev and mc < b_conf
    log(f"tune best dev={1e3 * best[0]:.2f} confirm={1e3 * mc:.2f} vs B dev={1e3 * b_dev:.2f} confirm={1e3 * b_conf:.2f} -> {'keep Bt' if ok else 'no Bt'}")
    if ok:
        (td / "params.json").write_text(json.dumps(best[1], indent=1, sort_keys=True) + "\n")
        (td / "dev_preds.json").write_text(json.dumps(best[2]))
    (td / "done").write_text("1\n")


def read_items(M, d):
    qs, H = M.read_split(d)
    return [(q["query_id"], q["episode_id"], H[q["machine_id"]], int(q["cutoff"]), q["machine_type"], int(q["signal"]))
            for q in sorted(qs, key=lambda r: r["query_id"])]


# ---------------- packaging ----------------
def package(kit, tag, name):
    pub, kd = paths(kit); out = HE / "runs" / f"R2-{tag}"; md = out / name; ent = md / "entry"
    meta = json.loads((pub / "metadata.json").read_text())
    if ent.exists(): shutil.rmtree(ent)
    ent.mkdir(parents=True)
    shutil.copy(HE / "entry_src/predict.py", ent / "predict.py")
    desc = {"A": "Main-round model (fixed lag-pair/size-rule constants) with Bayesian (b,d) grid filter.",
            "Bt": "Model B with filter rates (qb, jump, qd, etau) retuned by random search on a train dev split.",
            "B": "Generalized structured model: win logit = mu_type + d + b*x + delta*z, x over 28 XOR lag pairs + 8 single lags + none; "
                 "population priors over lag hypotheses and size lag fitted by EM on organizer train+validation data; Bayesian grid filter per machine.",
            "C": "Gradient-boosted trees (LightGBM, trained offline) on past-only features (adaptive per-machine lag agreement, EW win rates, size-state terms, episode signal); "
                 "trees exported to JSON and evaluated in pure Python."}
    if name == "blend":
        ws = json.loads((md / "weights.json").read_text())
        shutil.copy(SRC / "model_blend.py", ent / "model.py"); parts = []
        for m, w in sorted(ws.items()):
            src = MODS[m][1]; mn = {"A": "model_a"}.get(m, src.stem)
            shutil.copy(src, ent / f"{mn}.py"); shutil.copy(out / m / "params.json", ent / f"params_{m}.json")
            parts.append([mn, f"params_{m}.json", w])
        (ent / "params.json").write_text(json.dumps({"parts": parts}, indent=1) + "\n")
        method = "Linear blend (weights chosen on a train dev split): " + "; ".join(f"{w:g} x {m}: {desc[m]}" for m, w in sorted(ws.items()))
    else:
        shutil.copy(MODS[name][1], ent / "model.py"); shutil.copy(out / name / "params.json", ent / "params.json")
        method = desc[name]
    (ent / "reproduce.json").write_text(json.dumps({"image": "python", "command": ["python", "predict.py", "--data", "/data", "--out", "/output/submission.csv"]}, indent=2) + "\n")
    (ent / "README.md").write_text(f"""# House Edge entry ({meta.get('round')} round)

- Round: {meta.get('round')}; season_id: {meta.get('season_id')}
- Environment: `python` image (Python 3.13), **standard library only**; no requirements.
- Hardware: any CPU, one core, < 1 GB RAM. No GPU.
- Seeds: none needed (deterministic, no randomness; sorted iteration order).

## Method
{method}

## Recreate the forecast
```sh
python predict.py --data /data --out /output/submission.csv
```
`--data` is the round's `public/` folder (reads `test/queries.csv`, `test/history.csv`).
Saved settings: `params*.json` (fitted offline on organizer training + validation data of this round). Code: `model*.py`.
Optional: `--split validation` writes validation predictions for local scoring.
For a query at cutoff c only ticks < c of machines in the same episode are used; condition labels are never used.
""")
    py313 = subprocess.run(["uv", "python", "find", "3.13"], capture_output=True, text=True).stdout.strip() or PY
    fresh = md / "fresh"
    if fresh.exists(): shutil.rmtree(fresh)
    shutil.copytree(ent, fresh / "code")
    t = time.time()
    sh([py313, "-I", str(fresh / "code/predict.py"), "--data", str(pub), "--out", str(md / "submission.csv")], cwd=fresh / "code")
    dt = time.time() - t
    sh([py313, "-I", str(fresh / "code/predict.py"), "--data", str(pub), "--out", str(md / "validation_predictions.csv"), "--split", "validation"], cwd=fresh / "code")
    sh([py313, "-I", str(ent / "predict.py"), "--data", str(pub), "--out", str(fresh / "again.csv")], cwd=ent)
    shutil.rmtree(ent / "__pycache__", ignore_errors=True)
    same = (fresh / "again.csv").read_bytes() == (md / "submission.csv").read_bytes()
    log(f"package {name}: py={py313} test runtime={dt:.0f}s (1 core) reproduce identical={same}")
    o = sh([PY, "check_submission.py", str(md / "submission.csv"), "--data", str(pub)], cwd=kd); log("test check:", o.strip().splitlines()[0])
    o = sh([PY, "check_submission.py", str(md / "validation_predictions.csv"), "--data", str(pub), "--split", "validation"], cwd=kd)
    log("val check:", " | ".join(l.strip() for l in o.splitlines() if "points" in l.lower() or "VALID" in l))
    z = md / "entry.zip"
    if z.exists(): z.unlink()
    o = sh([PY, "build_entry.py", "--source", str(ent), "--out", str(z)], cwd=kd)
    log("zip:", z, f"{z.stat().st_size / 1e6:.2f} MB", o.strip().splitlines()[-1][:150] if o.strip() else "")
    if not same:
        log("WARNING: not byte-identical"); sys.exit(1)


def main():
    a = sys.argv[1:]
    kit = a[0]; tag = a[a.index("--tag") + 1] if "--tag" in a else Path(kit).resolve().name
    workers = int(a[a.index("--workers") + 1]) if "--workers" in a else 6
    if "--package" in a:
        package(kit, tag, a[a.index("--package") + 1])
    else:
        evaluate(kit, tag, "--force" in a, workers, "--notune" not in a)


if __name__ == "__main__":
    main()

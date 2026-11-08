"""Round-2 one-command runner (staged, restartable; timestamped summary lines only).

  python src/round2_run.py <kit_dir> [--tag NAME] [--fast] [--force] [--workers 6] [--notune] [--ablations] [--nodesk] [--gate] [--nb]
    stage 0  schema report; pooled fallback P fitted + packaged + verified       -> Desktop SAFE   (~1 min)
    stage 1  B fitted (C, C_dev fit in parallel); B predicted, packaged, verified -> Desktop SAFE   (~5 min)
    stage 2  dev split (train pseudo-queries, observed outcomes): B, C, (A), retune Bt (not with --fast),
             blend, per-query gate; conservative PICK by dev + confirm Brier; PICK packaged -> Desktop FINAL
  python src/round2_run.py <kit_dir> --package <P|A|B|Bt|C|blend|gate> [--tag NAME] [--desk SLOT]
<kit_dir> = the organizer kit folder (with public/ and check_submission.py) or its public/ folder.
Outputs: runs/R2-<tag>/<model>/{submission.csv, validation_predictions.csv, entry/, entry.zip}; summary.json;
~/Desktop/SSD-HouseEdge-R2/{SAFE,FINAL}/ (submission.csv + entry.zip + INFO.txt) unless --nodesk.
"""
import sys, os, json, csv, time, shutil, subprocess, random, importlib, math, zipfile
from pathlib import Path
from collections import defaultdict

HE = Path(__file__).resolve().parents[1]; SRC = HE / "src"; AENT = HE / "runs/06-lowjump/entry"
sys.path.insert(0, str(SRC)); sys.path.insert(0, str(AENT))
import r2score as S
import kitschema as KS

PY = sys.executable
DESK = Path(os.environ.get("R2_DESK", str(Path.home() / "Desktop" / "SSD-HouseEdge-R2")))
T0 = time.time()
MODS = {"A": ("model", AENT / "model.py"), "B": ("model_b", SRC / "model_b.py"), "Bn": ("model_b", SRC / "model_b.py"),
        "Bt": ("model_b", SRC / "model_b.py"), "B0": ("model_b", SRC / "model_b.py"), "C": ("model_c", SRC / "model_c.py"),
        "P": ("model_p", SRC / "model_p.py"), "gate": ("model_gate", SRC / "model_gate.py")}
OPT = {"desk": True, "workers": 6}
VALS = list(KS.DEFAULT_PAY); STD = False


def log(*a):
    print(f"[r2 {time.strftime('%H:%M:%S')} +{time.time() - T0:4.0f}s]", *a, flush=True)


def paths(kit):
    kit = Path(kit).resolve()
    pub = kit / "public" if (kit / "public" / "test" / "queries.csv").exists() else kit
    if not (pub / "test" / "queries.csv").exists():  # nested unzip folders: find the shallowest */test/queries.csv
        hits = sorted((p for p in kit.rglob("queries.csv") if p.parent.name == "test"),
                      key=lambda p: ("adaptation" not in str(p).lower(), "main" in str(p).lower(), len(p.parts), str(p)))
        if hits: pub = hits[0].parent.parent
        if len(hits) > 1: print(f"[r2] NOTE several kits under {kit}; using {pub} (pass that public/ folder explicitly to be sure)", flush=True)
    kd = next((d for d in (pub.parent, pub.parent.parent, pub.parent.parent.parent) if (d / "check_submission.py").exists()), HE / "kit")
    return pub, kd


def sh(cmd, cwd=None, soft=False):
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0 and soft:
        log("WARN step failed (continuing):", " ".join(map(str, cmd))[-160:], "|", (r.stderr or r.stdout).strip()[-300:]); return ""
    if r.returncode != 0:
        log("FAILED", " ".join(map(str, cmd))); print(r.stdout[-2000:], r.stderr[-3000:]); raise RuntimeError("step failed")
    return r.stdout


def mod(name):
    return importlib.import_module(MODS[name][0])


def setup_schema(pub):
    global VALS, STD
    cols, VALS = KS.find_schema(pub); S.set_schema(pub)
    tq = {sp: KS.read_split(pub / sp, VALS) if (pub / sp / "queries.csv").exists() else ([], {}) for sp in ("train", "validation", "test")}
    types = {sp: sorted(set(q["machine_type"] for q in v[0])) for sp, v in tq.items()}
    fitted = set(types["train"]) | set(types["validation"])
    sigs = set(q["signal"] for v in tq.values() for q in v[0])
    STD = sigs <= {-1.0, 0.0, 1.0} and VALS == list(KS.DEFAULT_PAY) and set(types["test"]) <= {"steady", "balanced", "jackpot"} and fitted >= {"steady", "balanced", "jackpot"}
    md = KS.metadata(pub)
    log(f"schema: payout columns {cols} | conditions {list(S.CONDS)} | metadata round={md.get('round')} max_points={md.get('max_points')}")
    for sp, (qs, H) in tq.items():
        if not qs: log(f"schema {sp}: MISSING"); continue
        ep = defaultdict(int)
        for q in qs: ep[q["episode_id"]] += 1
        cut = [q["cutoff"] for q in qs]; nh = [len(H.get(q["machine_id"], {})) for q in qs]
        sg = sorted(set(q["signal"] for q in qs))
        log(f"schema {sp}: {len(qs)} queries, {len(ep)} episodes, machines/episode {min(ep.values())}-{max(ep.values())}, "
            f"cutoff {min(cut)}-{max(cut)}, history len {min(nh)}-{max(nh)} (mean {sum(nh) / len(nh):.0f}), types {types[sp]}, signal {sg[:6]}")
    if str(md.get("round", "")).lower() == "main":
        log("FLAG KIT_IS_MAIN_ROUND: metadata says round=main. For round 2 pass the adaptation public/ folder!")
    unk = sorted(set(types["test"]) - fitted)
    if unk: log(f"FLAG NEW_TEST_TYPES {unk}: these use the pooled fallback (type '_pool') in B, C (all-zero type) and P")
    if not STD: log("schema differs from main round: model A disabled (hard-coded main schema)")
    for sp in ("validation",):
        for f in ("targets.csv", "reference.csv", "conditions.csv"):
            if not (pub / sp / f).exists(): log(f"FLAG MISSING {sp}/{f}")
    return types


# ---------------- items / dev split ----------------
def read_items(name, d):
    if name == "A":
        M = mod("A"); qs, H = M.read_split(d)
        return [(q["query_id"], q["episode_id"], H[q["machine_id"]], int(q["cutoff"]), q["machine_type"], int(q["signal"]))
                for q in sorted(qs, key=lambda r: r["query_id"])]
    qs, H = KS.read_split(d, VALS)
    return [(q["query_id"], q["episode_id"], H[q["machine_id"]], int(q["cutoff"]), q["machine_type"], q["signal"])
            for q in sorted(qs, key=lambda r: r["query_id"])]


def dev_items(pub, js):
    tr = pub / "train"
    qs, H = KS.read_split(tr, VALS)
    lab = KS.read_labels(tr, VALS)
    cf = tr / "conditions.csv"
    cond = {r["query_id"]: r["condition"] for r in csv.DictReader(open(cf))} if cf.exists() else {}
    epc = {q["episode_id"]: cond.get(q["query_id"], "all") for q in qs}
    split = {}
    for c in sorted(set(epc.values())):
        for i, e in enumerate(sorted(e for e, cc in epc.items() if cc == c)):
            split[e] = "confirm" if i % 3 == 2 else "dev"
    items, meta = [], {}
    for q in sorted(qs, key=lambda r: r["query_id"]):
        s = H[q["machine_id"]]; c = int(q["cutoff"])
        for j in js:
            cj = c - j; y = lab.get(q["query_id"]) if j == 0 else s.get(cj)
            if y is None: continue
            hist = {t: v for t, v in s.items() if t < cj}
            qid = f'{q["query_id"]}@{j}'
            items.append((qid, f'{q["episode_id"]}@{j}', hist, cj, q["machine_type"], q["signal"]))
            meta[qid] = (q["episode_id"], epc[q["episode_id"]], split[q["episode_id"]], y)
    return items, meta


def a_items(items):  # model A reads raw payout values and integer signals
    return [(q, e, {t: int(VALS[v]) for t, v in s.items()}, c, mt, int(round(sg))) for q, e, s, c, mt, sg in items]


def pool_predict(M, items, P, workers):
    eps = sorted(set(it[1] for it in items)); slot = {e: i % workers for i, e in enumerate(eps)}
    chunks = [[it for it in items if slot[it[1]] == i] for i in range(workers)]
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(workers) as pool:
        res = pool.map(M._work, [(P, ch) for ch in chunks])
    out = {}
    for r in res: out.update({k: v[:len(VALS)] if isinstance(v, list) else v for k, v in r})
    return out


def predict_any(name, M, items, P, workers):
    if name == "C": return M.predict_items(items, P)
    if name == "P": return M.predict_items(items, P)
    if name == "A": return pool_predict(M, a_items(items), P, workers)
    return pool_predict(M, items, P, workers)


def dev_brier(pred, meta, which):
    ep = defaultdict(list)
    for q, p in pred.items():
        e, c, sp, y = meta[q]
        if sp != which: continue
        ep[(c, e)].append(sum((a - (1.0 if j == y else 0.0)) ** 2 for j, a in enumerate(p)))
    cs = sorted(set(k[0] for k in ep))
    per = {c: [sum(v) / len(v) for k, v in ep.items() if k[0] == c] for c in cs}
    per = {c: sum(v) / len(v) for c, v in per.items()}
    return sum(per.values()) / max(1, len(per)), per


def mix(preds, ws):
    tot = sum(ws.values()); k0 = next(iter(ws))
    return {q: [sum(ws[m] / tot * preds[m][q][j] for m in ws) for j in range(len(preds[k0][q]))] for q in preds[k0]}


def predict_splits(out, pub, name, force, workers):
    P = json.loads((out / name / "params.json").read_text()); M = mod(name)
    for split, fn in (("validation", "validation_predictions.csv"), ("test", "submission.csv")):
        f = out / name / fn
        if not (pub / split / "queries.csv").exists(): continue
        if force or not f.exists():
            t = time.time(); pr = predict_any(name, M, read_items(name, pub / split), P, workers)
            S.write_csv(f, pr); log(f"pred {name} {split} {time.time() - t:.0f}s")


def val_line(out, name, Sc):
    f = out / name / "validation_predictions.csv"
    if Sc.ok and f.exists():
        pr = S.load_csv(f); log("val", Sc.line(name, pr)); return Sc.score(pr)[0]
    return float("nan")


# ---------------- main evaluation ----------------
def evaluate(kit, tag, force, workers, fast=False, tune=True, ablations=False):
    pub, kd = paths(kit); out = HE / "runs" / f"R2-{tag}"; out.mkdir(parents=True, exist_ok=True)
    log(f"kit={pub} out={out} fast={fast} workers={workers}")
    if OPT["desk"]:
        old = [d for d in (DESK / "SAFE", DESK / "FINAL") if d.exists()]
        if old:
            arc = DESK / "_previous" / time.strftime("%H%M%S"); arc.mkdir(parents=True, exist_ok=True)
            for d in old: shutil.move(str(d), str(arc / d.name))
            log(f"moved old Desktop SAFE/FINAL to {arc} (never upload from _previous)")
    setup_schema(pub); Sc = S.Scorer(pub)
    if not Sc.ok: log("FLAG NO_VALIDATION_TARGETS: validation points unavailable; picking by dev split only")
    vp = {}
    # ---- stage 0: pooled fallback P (always valid, seconds)
    pf = out / "P" / "params.json"
    if force or not pf.exists():
        Pp = mod("P").fit(pub); pf.parent.mkdir(parents=True, exist_ok=True); pf.write_text(json.dumps(Pp, indent=1, sort_keys=True) + "\n")
        log(f"fit P hl={Pp['hl']} k={Pp['k']} types={sorted(Pp['freq'])}")
    predict_splits(out, pub, "P", force, workers); vp["P"] = val_line(out, "P", Sc)
    if force or not (out / "P" / "entry.zip").exists():
        package(kit, tag, "P", desk="SAFE")
    # ---- stage 1: fits (B now; C, C_dev in the background)
    procs = {}
    for name, args in (("C", []), ("C_dev", ["--valonly"])):
        if force or not (out / name / "params.json").exists():
            (out / name).mkdir(parents=True, exist_ok=True)
            procs[name] = subprocess.Popen([PY, str(SRC / "fit_c.py"), str(pub), str(out / name / "params.json"), "--hlf", "20", "--gbm"] + args,
                                           stdout=open(out / name / "fit.log", "w"), stderr=subprocess.STDOUT)
    bl = [("B", [])] + ([("B0", ["--base"]), ("Bn", ["--naive"])] if ablations else [])
    for name, args in bl:
        pf = out / name / "params.json"
        if force or not pf.exists():
            o = sh([PY, SRC / "fit_b.py", pub, pf] + args)
            if name == "B":
                for l in o.splitlines():
                    if l.startswith("fit_b") and "wrote" not in l and "k|" not in l: log(l[:230])
    log("fit B done")
    predict_splits(out, pub, "B", force, workers); vp["B"] = val_line(out, "B", Sc)
    if force or not (out / "B" / "entry.zip").exists():
        package(kit, tag, "B", desk="SAFE" if not (vp["B"] < vp["P"]) else None)
    for name in [n for n, _ in bl[1:]]:
        predict_splits(out, pub, name, force, workers); vp[name] = val_line(out, name, Sc)
    for name, p in procs.items():
        if p.wait() != 0: log(f"WARN fit {name} failed:", (out / name / "fit.log").read_text()[-300:])
        else: log(f"fit {name}:", (out / name / "fit.log").read_text().strip().splitlines()[-1][:120])
    hasC = (out / "C" / "params.json").exists() and (out / "C_dev" / "params.json").exists()
    if hasC:
        try:
            predict_splits(out, pub, "C", force, workers); vp["C"] = val_line(out, "C", Sc)
        except Exception as e:
            log("WARN model C prediction failed (C dropped):", repr(e)[:300]); hasC = False
    hasA = False
    if STD and not fast:
        try:
            if not (out / "A" / "params.json").exists():
                (out / "A").mkdir(exist_ok=True); shutil.copy(AENT / "params.json", out / "A" / "params.json")
            predict_splits(out, pub, "A", force, workers); vp["A"] = val_line(out, "A", Sc); hasA = True
        except Exception as e:
            log("WARN model A failed (dropped):", repr(e)[:300])
    dd = out / "D"
    try:
        if STD and (force or not (dd / "validation_predictions.csv").exists()):
            sh([PY, "run_starter.py", "--data", pub, "--out", dd, "--method", "pooled_recent"], cwd=kd, soft=True)
        if (dd / "validation_predictions.csv").exists() and Sc.ok:
            vp["D"] = Sc.score(S.load_csv(dd / "validation_predictions.csv"))[0]; log("val", Sc.line("D starter", S.load_csv(dd / "validation_predictions.csv")))
    except Exception as e:
        log("WARN starter D skipped:", repr(e)[:200])
    # ---- stage 2: dev split
    js = (0, 4, 8, 12) if fast else (0, 2, 4, 6, 8, 10, 12, 14)
    dp = out / "dev_preds.json"
    J = json.loads(dp.read_text()) if (dp.exists() and not force) else {"preds": {}}
    if J.get("js") != list(js): J = {"preds": {}}
    want = ["B", "P"] + (["C"] if hasC else []) + (["A"] if hasA else [])
    need = [n for n in want if n not in J["preds"]]
    if need:
        items, dmeta = dev_items(pub, js); J["meta"] = dmeta; J["js"] = list(js)
        for name in need:
            try:
                P = json.loads((out / ("C_dev" if name == "C" else name) / "params.json").read_text()); M = mod(name)
                t = time.time(); J["preds"][name] = predict_any(name, M, items, P, workers)
                log(f"dev pred {name} {time.time() - t:.0f}s ({len(items)} pseudo-queries)")
                dp.write_text(json.dumps(J))
            except Exception as e:
                if name == "B": raise
                log(f"WARN dev pred {name} failed (dropped):", repr(e)[:300])
    dev, dmeta = J["preds"], {k: tuple(v) for k, v in J["meta"].items()}
    if OPT.get("nb"):  # opt-in (--nb): false positives on kits without neighbor effects (S6 -0.06), main gain <= +0.04
        try:
            if nb_tune(out, pub, dev, dmeta, js):
                dp.write_text(json.dumps(J))
                predict_splits(out, pub, "B", True, workers); vp["B"] = val_line(out, "B", Sc)
                (out / "B" / "entry.zip").unlink(missing_ok=True)
        except Exception as e:
            log("WARN neighbor tune failed (ignored):", repr(e)[:300])
    if not os.environ.get("R2_NOEV"):
        try:
            if ev_tune(out, pub, dev, dmeta, js, workers, (40, 80) if fast else (25, 40, 80)):
                dp.write_text(json.dumps(J))
                predict_splits(out, pub, "B", True, workers); vp["B"] = val_line(out, "B", Sc)
                (out / "B" / "entry.zip").unlink(missing_ok=True)
        except Exception as e:
            log("WARN evidence half-life tune failed (ignored):", repr(e)[:300])
    if tune and not fast:
        try:
            tune_b(out, pub, dev, dmeta, workers, force, js)
        except Exception as e:
            log("WARN retune failed (no Bt):", repr(e)[:300])
        if (out / "Bt" / "params.json").exists() and (out / "Bt" / "dev_preds.json").exists():
            dev["Bt"] = json.loads((out / "Bt" / "dev_preds.json").read_text())
            predict_splits(out, pub, "Bt", force, workers); vp["Bt"] = val_line(out, "Bt", Sc)
    cands = [m for m in ("A", "B", "Bt", "C") if m in dev]
    log("---- dev split (train pseudo-queries, Brier x1e3; C fit on validation histories only) ----")
    for name in cands + ["P"]:
        m, pc = dev_brier(dev[name], dmeta, "dev"); mc, _ = dev_brier(dev[name], dmeta, "confirm")
        log(f"{name}: dev={1e3 * m:.2f} confirm={1e3 * mc:.2f} " + " ".join(f"{c[:6]}={1e3 * v:.2f}" for c, v in pc.items()))
    # blend (pairwise, weights on dev only)
    best = None
    for a, b in [(x, y) for i, x in enumerate(cands) for y in cands[i + 1:]]:
        for i in range(11):
            w = i / 10; ws = {a: round(1 - w, 1), b: w}
            m, _ = dev_brier(mix(dev, ws), dmeta, "dev")
            if best is None or m < best[0] - 1e-12: best = (m, ws)
    rows = {}
    if best:
        ws = {k: v for k, v in best[1].items() if v > 0}
        (out / "blend").mkdir(exist_ok=True); (out / "blend" / "weights.json").write_text(json.dumps(ws) + "\n")
        for fn in ("validation_predictions.csv", "submission.csv"):
            if all((out / m / fn).exists() for m in ws):
                S.write_csv(out / "blend" / fn, mix({m: S.load_csv(out / m / fn) for m in ws}, ws))
        dev["blend"] = mix(dev, ws); vp["blend"] = val_line(out, "blend", Sc)
        log(f"blend (dev-selected) {ws}")
    # per-query gate (B vs C weight from B's confidence), fitted on dev only
    if hasC and OPT.get("gate"):  # opt-in (--gate): never beat B/blend on confirm in rehearsals
        try:
            g = gate_stage(out, pub, dev, dmeta, workers, force, js)
            if g is not None:
                dev["gate"] = g; vp["gate"] = val_line(out, "gate", Sc)
        except Exception as e:
            log("WARN gate stage failed (ignored):", repr(e)[:300])
    for m in [x for x in ("A", "B", "Bt", "C", "blend", "gate", "P") if x in dev]:
        d_, _ = dev_brier(dev[m], dmeta, "dev"); c_, _ = dev_brier(dev[m], dmeta, "confirm"); rows[m] = (d_, c_, (d_ + c_) / 2)
    if "gate" in rows:
        rc = min(rows[m][1] for m in ("B", "Bt", "blend", "C") if m in rows)
        if rows["gate"][1] >= rc - 1e-5:
            log(f"gate not eligible: confirm {1e3 * rows['gate'][1]:.2f} >= best other confirm {1e3 * rc:.2f}"); rows["gate_x"] = rows.pop("gate")
    elig = {m: r for m, r in rows.items() if m not in ("gate_x", "P")}
    bestm = min(elig, key=lambda m: (elig[m][2], m)); tol = float(os.environ.get("R2_TOL", "0.0003"))
    pref = [m for m in ("B", "Bt", "blend", "gate", "C", "A") if m in rows]  # conservative: simplest within noise of the best
    pick = next(m for m in pref + [bestm] if m in elig and elig[m][2] <= elig[bestm][2] + tol)
    # ---- diagnostics / flags
    Pb = json.loads((out / "B" / "params.json").read_text()); dg = Pb.get("diag", {})
    flags = []; th = dg.get("top_h", [])
    fam = defaultdict(float)
    for h, w in zip(Pb["hyps"], Pb["hprior"]):
        k = h if isinstance(h, str) else f"{h[0]},{h[1]}"
        fam[k.split(":")[0] if ":" in k else ("none" if k == "0,0" else ("lag" if k.endswith(",0") else "xor"))] += w
    fams = sorted(fam.items(), key=lambda kv: -kv[1])
    if fams and fams[0][1] < 0.35: flags.append("NO_DOMINANT_RULE_FAMILY (B's family may not contain the rule)")
    if dg.get("none_mass", 0) > 0.2: flags.append(f"NONE_MASS={dg['none_mass']} (many machines look structureless)")
    mdv = [v for k, v in dg.get("max_dev", {}).items() if k != "_pool"]
    if mdv and max(mdv) > 0.02: flags.append(f"SIZE_TABLES_MISFIT max_dev={max(mdv)} (more than 2 size states?)")
    if dg.get("no_targets"): flags.append("NO_TARGETS_FIT (B tables/mu from histories only)")
    if "C" in rows and rows["C"][2] < rows["B"][2]: flags.append("C_BEATS_B_ON_DEV (structured model misspecified)")
    if hasC:
        pb_, pc_ = S.load_csv(out / "B" / "submission.csv"), S.load_csv(out / "C" / "submission.csv")
        dif = sum(sum(abs(a - b) for a, b in zip(pb_[q], pc_[q])) for q in pb_) / len(pb_)
        if dif > 0.15: flags.append(f"B_C_DISAGREE_TEST mean L1={dif:.3f}")
    vv = {m: v for m, v in vp.items() if v == v}
    if pick in vv and max(vv.values()) - vv[pick] > 1.0: flags.append(f"VALIDATION_DISAGREES_WITH_DEV_PICK (val best {max(vv, key=vv.get)})")
    if "P" in rows and rows["P"][2] < rows[pick][2]: flags.append("POOLED_P_BEATS_PICK_ON_DEV (use SAFE=P?)")
    log("==== DIAGNOSTICS ====")
    log("top win hypotheses (prior mass):", " ".join(f"({k}):{v}" for k, v in th))
    log("rule families (prior mass):", " ".join(f"{k}:{v:.3f}" for k, v in fams[:6]))
    log("size K:", dg.get("K"), "| size-lag marginal k0..k8,2lag:", dg.get("k_marg"), "| table max_dev:", dg.get("max_dev"))
    log("==== RECOMMENDATION ====")
    for m in sorted(rows, key=lambda m: rows[m][2]):
        log(f"{m:<6} dev={1e3 * rows[m][0]:.2f} confirm={1e3 * rows[m][1]:.2f} mean={1e3 * rows[m][2]:.2f} val_points={vp.get(m, float('nan')):.3f}")
    log(f"best by dev+confirm = {bestm}; tolerance {tol * 1e3:.2f}e-3 -> PICK = {pick}" + (f" weights={ws}" if pick == "blend" else ""))
    log("flags:", "; ".join(flags) or "none")
    (out / "summary.json").write_text(json.dumps({"val_points": vp, "dev": {m: r[0] for m, r in rows.items()},
        "confirm": {m: r[1] for m, r in rows.items()}, "pick": pick, "blend": best[1] if best else None, "flags": flags, "diag": dg,
        "elapsed_s": round(time.time() - T0)}, indent=1) + "\n")
    # ---- final package
    if force or not (out / pick / "entry.zip").exists() or pick != "B":
        package(kit, tag, pick, desk="FINAL")
    else:
        publish(out / pick, "FINAL", pick, pub)
    log(f"DONE {time.time() - T0:.0f}s. FINAL = {pick} -> " + (str(DESK / "FINAL") if OPT["desk"] else str(out / pick) + " (desktop copy disabled)"))


def nb_tune(out, pub, dev, dmeta, js):
    """Episode neighbor term for B (others' win surprise at lag k, coefficient g): (k, g) chosen on the dev split from
    B's dev predictions (post-hoc logit shift), kept only if the confirm split also improves. Returns True if B changed."""
    pf = out / "B" / "params.json"; P = json.loads(pf.read_text())
    if "nb_done" in P:
        return False
    M = mod("B"); items, _ = dev_items(pub, js); ctx = M.episode_ctx(items)
    F = {k: {it[0]: M.nb_feat(ctx[it[0]], it[3], k) for it in items} for k in range(1, 9)}
    adj = lambda k, g: {q: M.finalize(p, g * F[k][q], P) for q, p in dev["B"].items()}
    b_dev, _ = dev_brier(dev["B"], dmeta, "dev"); b_conf, _ = dev_brier(dev["B"], dmeta, "confirm"); best = None
    for k in range(1, 9):
        for g in (-0.5, -0.25, 0.25, 0.5, 0.75, 1.0):
            m, _ = dev_brier(adj(k, g), dmeta, "dev")
            if best is None or m < best[0] - 1e-12: best = (m, k, g)
    m, k, g = best; pr = adj(k, g); mc, _ = dev_brier(pr, dmeta, "confirm")
    ok = m < b_dev - 2e-5 and mc < b_conf
    log(f"neighbor term: best lag {k} g={g} dev={1e3 * m:.2f} confirm={1e3 * mc:.2f} vs B dev={1e3 * b_dev:.2f} confirm={1e3 * b_conf:.2f} -> {'KEEP' if ok else 'no'}")
    P["nb_done"] = 1
    if ok:
        P["nbk"] = k; P["nbg"] = g; dev["B"].clear(); dev["B"].update(pr)
    pf.write_text(json.dumps(P, indent=1, sort_keys=True) + "\n")
    return ok


def ev_tune(out, pub, dev, dmeta, js, workers, hls):
    """Evidence half-life for B's rule hypotheses (forget old evidence when rules change within a history):
    each candidate = one B pass on the dev items; kept only if dev AND confirm improve. Returns True if B changed."""
    pf = out / "B" / "params.json"; P = json.loads(pf.read_text())
    if "ev_done" in P:
        return False
    M = mod("B"); items, _ = dev_items(pub, js)
    b_dev, _ = dev_brier(dev["B"], dmeta, "dev"); b_conf, _ = dev_brier(dev["B"], dmeta, "confirm"); best = None
    for hl in hls:
        t = time.time(); pr = pool_predict(M, items, dict(P, evhl=hl), workers); m, _ = dev_brier(pr, dmeta, "dev")
        log(f"evidence half-life {hl}: dev={1e3 * m:.2f} (B {1e3 * b_dev:.2f}) {time.time() - t:.0f}s")
        if best is None or m < best[0]: best = (m, hl, pr)
    m, hl, pr = best; mc, _ = dev_brier(pr, dmeta, "confirm")
    ok = m < b_dev - 2e-5 and mc < b_conf
    log(f"evidence half-life: best {hl} dev={1e3 * m:.2f} confirm={1e3 * mc:.2f} vs B dev={1e3 * b_dev:.2f} confirm={1e3 * b_conf:.2f} -> {'KEEP' if ok else 'no'}")
    P["ev_done"] = 1
    if ok:
        P["evhl"] = hl; dev["B"].clear(); dev["B"].update(pr)
    pf.write_text(json.dumps(P, indent=1, sort_keys=True) + "\n")
    return ok


def tune_b(out, pub, dev, dmeta, workers, force, js, ndraw=int(os.environ.get("R2_NDRAW", "8"))):
    """Bounded random search of B's filter rates on the DEV split only; keep the best draw as Bt only if it
    also beats B on the untouched confirm split."""
    td = out / "Bt"
    if (td / "done").exists() and not force:
        return
    td.mkdir(exist_ok=True)
    base = json.loads((out / "B" / "params.json").read_text()); M = mod("B")
    items, _ = dev_items(pub, js)
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
    for qd in (0.015, 0.04):  # fixed fast-drift candidates (pure drifting win rates)
        P = dict(base); P["qd"] = qd; P["dmax"] = max(base["dmax"], 3.0); P["nd"] = max(base["nd"], 13)
        pr = pool_predict(M, items, P, workers); m, _ = dev_brier(pr, dmeta, "dev")
        log(f"tune fast-drift qd={qd}: dev={1e3 * m:.2f} (B {1e3 * b_dev:.2f})")
        if m < best[0]: best = (m, P, pr)
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


def gate_stage(out, pub, dev, dmeta, workers, force, js):
    """Per-query B/C weight w = sigmoid(theta . f(B confidence)); theta fitted on the dev split; kept as a
    candidate only (the picker decides with dev + confirm)."""
    if not (SRC / "model_gate.py").exists():
        return None
    G = mod("gate")
    gd = out / "gate"; gd.mkdir(exist_ok=True)
    fp = out / f"gate_feats_{len(js)}.json"
    if force or not fp.exists():
        items, _ = dev_items(pub, js)
        Pb = json.loads((out / "B" / "params.json").read_text())
        t = time.time(); F = G.b_feats(items, Pb, workers); fp.write_text(json.dumps(F)); log(f"gate features (dev) {time.time() - t:.0f}s")
    F = json.loads(fp.read_text())
    th, info = G.fit_gate(dev["B"], dev["C"], F, dmeta)
    log("gate fit:", info)
    if th is None:
        return None
    Pg = {"theta": th, "parts": [["model_b", "params_b.json"], ["model_c", "params_c.json"]], "feat": G.FEAT}
    (gd / "params.json").write_text(json.dumps(Pg, indent=1) + "\n")
    # validation / test predictions from the already computed B and C rows + B features
    for split, fn in (("validation", "validation_predictions.csv"), ("test", "submission.csv")):
        if not (pub / split / "queries.csv").exists() or not (out / "C" / fn).exists(): continue
        if True:  # theta may change between runs: always recompute
            Pb = json.loads((out / "B" / "params.json").read_text())
            Fs = G.b_feats(read_items("B", pub / split), Pb, workers)
            S.write_csv(gd / fn, G.combine(S.load_csv(out / "B" / fn), S.load_csv(out / "C" / fn), Fs, th))
    return G.combine(dev["B"], dev["C"], F, th)


# ---------------- packaging ----------------
DESC = {"A": "Main-round model (fixed lag-pair/size-rule constants) with Bayesian (b,d) grid filter.",
        "Bt": "Model B with filter rates (qb, jump, qd, etau) retuned by random search on a train dev split.",
        "B": "Generalized structured model: win logit = mu_type + d + b*x + delta*z; x from a machine-specific rule hypothesis "
             "(268 candidates: XOR / AND / OR lag pairs, single lags, win counts, 3-lag parity, cross-machine lags, none); "
             "population priors over rule and size-state hypotheses fitted by EM on organizer train+validation data; Bayesian grid filter per machine.",
        "C": "Gradient-boosted trees (LightGBM, trained offline) on past-only features (adaptive per-machine lag agreement, EW win rates, size-state terms, "
             "episode signal); trees exported to JSON and evaluated in pure Python.",
        "P": "Pooled fallback: own exponentially weighted payout counts shrunk toward per-type payout frequencies (organizer train + validation).",
        "gate": "Per-query combination of model B and model C: weight on B = logistic function of B's posterior confidence (fitted on a train dev split)."}


def py313():
    return subprocess.run(["uv", "python", "find", "3.13"], capture_output=True, text=True).stdout.strip() or PY


def own_zip(ent, z):
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(ent.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                zi = zipfile.ZipInfo(str(p.relative_to(ent)), (2026, 1, 1, 0, 0, 0)); zi.compress_type = zipfile.ZIP_DEFLATED
                zf.writestr(zi, p.read_bytes())


def package(kit, tag, name, desk=None):
    pub, kd = paths(kit); out = HE / "runs" / f"R2-{tag}"; md = out / name; ent = md / "entry"
    if VALS == list(KS.DEFAULT_PAY): setup_quiet(pub)
    meta = KS.metadata(pub); t0 = time.time()
    if ent.exists(): shutil.rmtree(ent)
    ent.mkdir(parents=True)
    for f in ("predict.py",): shutil.copy(HE / "entry_src" / f, ent / f)
    shutil.copy(SRC / "kitschema.py", ent / "kitschema.py"); shutil.copy(SRC / "model_p.py", ent / "model_p.py")
    shutil.copy(out / "P" / "params.json", ent / "params_p.json")
    if name == "blend":
        ws = json.loads((md / "weights.json").read_text())
        shutil.copy(SRC / "model_blend.py", ent / "model.py"); parts = []
        for m, w in sorted(ws.items()):
            src = MODS[m][1]; mn = {"A": "model_a"}.get(m, src.stem)
            shutil.copy(src, ent / f"{mn}.py"); shutil.copy(out / m / "params.json", ent / f"params_{m}.json")
            parts.append([mn, f"params_{m}.json", w])
        (ent / "params.json").write_text(json.dumps({"parts": parts, "payvals": VALS}, indent=1) + "\n")
        method = "Linear blend (weights chosen on a train dev split): " + "; ".join(f"{w:g} x {m}: {DESC[m]}" for m, w in sorted(ws.items()))
    elif name == "gate":
        Pg = json.loads((md / "params.json").read_text())
        shutil.copy(SRC / "model_gate.py", ent / "model.py")
        for src, pf in ((SRC / "model_b.py", out / "B" / "params.json"), (SRC / "model_c.py", out / "C" / "params.json")):
            shutil.copy(src, ent / src.name)
        shutil.copy(out / "B" / "params.json", ent / "params_b.json"); shutil.copy(out / "C" / "params.json", ent / "params_c.json")
        Pg["payvals"] = VALS; (ent / "params.json").write_text(json.dumps(Pg, indent=1) + "\n")
        method = DESC["gate"] + " B: " + DESC["B"] + " C: " + DESC["C"]
    else:
        shutil.copy(MODS[name][1], ent / "model.py"); shutil.copy(out / name / "params.json", ent / "params.json")
        method = DESC[name]
    (ent / "reproduce.json").write_text(json.dumps({"image": "python", "command": ["python", "predict.py", "--data", "/data", "--out", "/output/submission.csv"]}, indent=2) + "\n")
    (ent / "README.md").write_text(f"""# House Edge entry ({meta.get('round')} round) - Team SSD

- Round: {meta.get('round')}; season_id: {meta.get('season_id')}
- Environment: `python` image (Python 3.13), **standard library only**; no requirements.txt needed.
- Hardware: any CPU; uses up to 2 processes (`--workers 2`, default), < 2 GB RAM. No GPU.
- Seeds: none needed (deterministic, no randomness; sorted iteration order; identical output for 1 or 2 workers).

## Method
{method}

## Recreate the forecast
```sh
python predict.py --data /data --out /output/submission.csv
```
`--data` is the round's `public/` folder (reads `test/queries.csv`, `test/history.csv`, `test/sample_submission.csv` header).
Saved settings: `params*.json` (fitted offline on organizer training + validation data of this round). Code: `model*.py`, `kitschema.py`.
Optional: `--split validation` writes validation predictions for local scoring; `--workers 1` for a single process.
Safety: any query the main model cannot score gets the bundled pooled fallback (`model_p.py`, `params_p.json`); a note is printed.
For a query at cutoff c only ticks < c of machines in the same episode are used; condition labels are never used.
Main-round data is not used by this entry.
""")
    p313 = py313(); fresh = md / "fresh"
    if fresh.exists(): shutil.rmtree(fresh)
    shutil.copytree(ent, fresh / "code")
    runs = [("test", [p313, "-I", fresh / "code/predict.py", "--data", pub, "--out", md / "submission.csv"], fresh / "code"),
            ("again1w", [p313, "-I", ent / "predict.py", "--data", pub, "--out", fresh / "again.csv", "--workers", "1"], ent)]
    if (pub / "validation" / "queries.csv").exists():
        runs.append(("val", [p313, "-I", fresh / "code/predict.py", "--data", pub, "--out", md / "validation_predictions.csv", "--split", "validation"], fresh / "code"))
    ps = {k: (subprocess.Popen([str(c) for c in cmd], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True), time.time()) for k, cmd, cwd in runs}
    dt = {}; err = {}
    for k, (p, ts) in ps.items():
        o, e = p.communicate(); dt[k] = time.time() - ts
        if p.returncode != 0: err[k] = e[-500:]
        elif "WARNING" in e or "fallback" in e: log(f"package {name} {k} stderr:", e.strip()[-300:])
    shutil.rmtree(ent / "__pycache__", ignore_errors=True)
    if "test" in err or "again1w" in err:
        log(f"PACKAGE {name} FAILED:", err); return False
    same = (fresh / "again.csv").read_bytes() == (md / "submission.csv").read_bytes()
    bad = KS.validate_submission(md / "submission.csv", pub / "test")
    log(f"package {name}: test runtime {dt['test']:.0f}s (2 workers, concurrent with 2 other runs) | 1-worker rerun byte-identical={same} | own validator: {'VALID' if not bad else bad[:3]}")
    if (pub / "validation" / "queries.csv").exists() and "val" not in err:
        bv = KS.validate_submission(md / "validation_predictions.csv", pub / "validation")
        if bv: log("validation csv problems:", bv[:3])
    o = sh([PY, "check_submission.py", md / "submission.csv", "--data", pub], cwd=kd, soft=True)
    log("kit checker (test):", (o.strip().splitlines() or ["(failed - kit checker may not accept this schema; own validator is authoritative)"])[0])
    z = md / "entry.zip"
    if z.exists(): z.unlink()
    o = sh([PY, "build_entry.py", "--source", ent, "--out", z], cwd=kd, soft=True)
    if not z.exists():
        log("kit build_entry failed -> own zip"); own_zip(ent, z)
    names = zipfile.ZipFile(z).namelist()
    zok = "README.md" in names and "reproduce.json" in names and "predict.py" in names and not any("__pycache__" in n for n in names)
    log(f"zip {z} {z.stat().st_size / 1e6:.2f} MB files={len(names)} root README/reproduce ok={zok} | package total {time.time() - t0:.0f}s")
    ok = same and not bad and zok
    if not ok:
        log(f"PACKAGE {name} NOT OK (identical={same}, valid={not bad}, zip={zok})"); return False
    if dt["test"] > 600: log(f"FLAG SLOW_ENTRY test runtime {dt['test']:.0f}s on this machine (limit 30 min on 2 cores)")
    if desk: publish(md, desk, name, pub)
    return True


def setup_quiet(pub):
    global VALS
    VALS = KS.find_schema(pub)[1]; S.set_schema(pub)


def publish(md, slot, name, pub):
    if not OPT["desk"]: return
    d = DESK / slot; d.mkdir(parents=True, exist_ok=True)
    for f in ("submission.csv", "entry.zip"): shutil.copy(md / f, d / f)
    sha = __import__("hashlib").sha256((d / "submission.csv").read_bytes()).hexdigest()
    (d / "INFO.txt").write_text(f"model={name}\nrun={md.parent.name}\nkit={pub}\ntime={time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                                f"submission.csv sha256={sha}\nrows={sum(1 for _ in open(d / 'submission.csv')) - 1}\n")
    log(f"DESKTOP {slot} <- {name}: {d}/submission.csv + entry.zip (sha {sha[:12]})")


def main():
    a = sys.argv[1:]
    kit = a[0]; tag = a[a.index("--tag") + 1] if "--tag" in a else Path(kit).resolve().name
    workers = int(a[a.index("--workers") + 1]) if "--workers" in a else 6
    OPT["desk"] = "--nodesk" not in a; OPT["gate"] = "--gate" in a; OPT["nb"] = "--nb" in a
    if "--package" in a:
        pub, _ = paths(kit); setup_quiet(pub)
        ok = package(kit, tag, a[a.index("--package") + 1], desk=a[a.index("--desk") + 1] if "--desk" in a else None)
        sys.exit(0 if ok else 1)
    try:
        evaluate(kit, tag, "--force" in a, workers, "--fast" in a, "--notune" not in a, "--ablations" in a)
    except BaseException as e:
        import traceback
        log("RUNNER ERROR:", repr(e)[:300]); print(traceback.format_exc()[-1500:])
        if OPT["desk"] and (DESK / "SAFE" / "submission.csv").exists() and not (DESK / "FINAL" / "submission.csv").exists():
            shutil.copytree(DESK / "SAFE", DESK / "FINAL", dirs_exist_ok=True); log("copied SAFE -> FINAL (SAFE is valid and verified)")
        log("SAFE on the Desktop is valid. Manual fallback: see ROUND2_RUNBOOK.md section 4 (--package B / C / P).")
        sys.exit(2)


if __name__ == "__main__":
    main()

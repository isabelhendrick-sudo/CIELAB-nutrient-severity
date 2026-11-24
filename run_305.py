"""
run_305.py — AI Hub 305 pipeline: tip burn vs yellowing, training-free colour + placement.

Run order (each step writes to --out and can be re-run; segmentation resumes from the cache):

  python run_305.py segment   --leaf-table leaves.csv --cache cache --out results [--sam-checkpoint sam_vit_b_01ec64.pth]
  python run_305.py overlays  --leaf-table leaves.csv --cache cache --out results --sample 30
  python run_305.py calibrate --leaf-table leaves.csv --cache cache --out results
  python run_305.py features  --leaf-table leaves.csv --cache cache --out results
  python run_305.py analyse   --leaf-table leaves.csv --cache cache --out results

Leaf table (CSV, one row per annotated leaf box):
  image_path, plant_id, class_name (tipburn | yellowing), captured_at (date/time),
  x1, y1, x2, y2 (box in original-image pixels), symptomatic (1 = Y, 0 = N), [leaf_id]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from colour_zones import (ZoneConfig, WB_MODES, GLARE_MODES, estimate_illuminant, apply_white_balance, to_lch,
                          classify_pixels, zone_fractions, severity_grid, baseline_indices,
                          background_features, class_map_to_rgb, CLASS_COLOURS, CLASS_NAMES, BG)
from placement import erode_proportional, placement_features
from segment import (Segmenter, load_rgb, to_working, save_mask, load_mask, box_in_working,
                     refine_mask, recover_symptom_margin)
import stats_utils as su

REQUIRED = ["image_path", "plant_id", "class_name", "captured_at", "x1", "y1", "x2", "y2", "symptomatic"]
B1_GRID = [60, 65, 70, 75, 80]
B2_GRID = [100, 105, 110, 115, 120]
POSITIVE_CLASS = "tipburn"
EROSION_SENS = (0.0, 0.01, 0.02)
TREATMENT_EN = {"칼슘부족": "Ca deficiency", "질소부족": "N deficiency", "철결핍": "Fe deficiency",
                "EC 3.0": "high EC (3.0)", "EC 0.75": "low EC (0.75)", "pH 5.0": "pH 5.0", "pH 7.5": "pH 7.5"}


# ================================================================ data
def read_leaf_table(path, day_step: int = 1) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"leaf table is missing columns: {missing}")
    df["captured_at"] = pd.to_datetime(df["captured_at"])
    df["symptomatic"] = df["symptomatic"].astype(int)
    df["plant_id"] = df["plant_id"].astype(str)
    df["class_name"] = df["class_name"].astype(str).str.strip().str.lower()
    if "leaf_id" not in df.columns:
        df["leaf_id"] = [f"L{i:07d}" for i in range(len(df))]
    df["leaf_id"] = df["leaf_id"].astype(str)
    if df["leaf_id"].duplicated().any():
        raise ValueError("leaf_id values must be unique")
    if "cause_method" in df.columns:                      # AI Hub 305 induced treatment, in English
        cm = df["cause_method"].astype(str).str.strip()
        df["treatment"] = cm.map(TREATMENT_EN).fillna(cm)   # per leaf row (a plant can have several)
    else:
        df["treatment"] = "unknown"
    t0 = df.groupby("plant_id")["captured_at"].transform("min")
    df["day"] = (df["captured_at"] - t0).dt.total_seconds() / 86400.0
    if day_step > 1:
        df = df[np.floor(df["day"]).astype(int) % day_step == 0].copy()
        print(f"[data] day_step={day_step}: keeping {len(df)} leaves")
    return df.reset_index(drop=True)


def make_split(df: pd.DataFrame, out: Path, frac_cal: float = 0.4, seed: int = 42) -> pd.DataFrame:
    """Plant-level split stratified by class AND treatment, so every treatment appears on both sides
    where it has enough plants. Re-uses an existing split.csv if present."""
    path = out / "split.csv"
    if path.exists():
        return pd.read_csv(path, dtype={"plant_id": str})
    n_treat = df.groupby("plant_id")["cause_method"].nunique() if "cause_method" in df.columns else None
    if n_treat is not None and (n_treat > 1).any():
        print(f"[split] note: {int((n_treat > 1).sum())} plants have more than one treatment label; the split "
              "uses each plant's most common one, while per-treatment results use each leaf's own label")
    rng = np.random.default_rng(seed)
    rows = []
    # stratify by each plant's class and its most common treatment
    plants = pd.DataFrame({
        "class_name": df.groupby("plant_id")["class_name"].first(),
        "treatment": df.groupby("plant_id")["treatment"].agg(lambda x: x.value_counts().index[0])})
    for (cls, trt), pl in plants.groupby(["class_name", "treatment"]):
        ids = np.array(sorted(pl.index))
        rng.shuffle(ids)
        n_cal = int(round(frac_cal * len(ids)))
        rows += [{"plant_id": p, "class_name": cls, "treatment": trt,
                  "split": "calibration" if i < n_cal else "test"} for i, p in enumerate(ids)]
    split = pd.DataFrame(rows)
    split.to_csv(path, index=False)
    print(f"[split] {split['split'].value_counts().to_dict()} plants -> {path}")
    print(pd.crosstab([split["class_name"], split["treatment"]], split["split"]).to_string())
    return split


def _image_groups(df):
    return [(p, g.to_dict("records")) for p, g in df.groupby("image_path", sort=True)]


# ================================================================ segment
def cmd_segment(a):
    df = read_leaf_table(a.leaf_table, a.day_step)
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    seg = Segmenter(a.sam_checkpoint, a.sam_model)
    groups = _image_groups(df)
    t0, done, log = time.time(), 0, []
    for i, (img_path, recs) in enumerate(groups):
        todo = [r for r in recs if not (cache / f"{r['leaf_id']}.npz").exists()]
        if not todo:
            continue
        try:
            img = load_rgb(img_path, a.ignore_exif)
        except IOError as e:
            print(f"[segment] {e}")
            continue
        boxes = [(r["x1"], r["y1"], r["x2"], r["y2"]) for r in todo]
        for r, (geom, m) in zip(todo, seg.masks_for_image(img, boxes)):
            save_mask(cache, r["leaf_id"], geom, m, seg.method)
            log.append({"leaf_id": r["leaf_id"], "method": seg.method, "mask_frac": float(m.mean())})
            done += 1
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"[segment] {i + 1}/{len(groups)} images, {done} leaves, {el / max(done, 1):.2f} s/leaf")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if log:
        lp = out / "segment_log.csv"
        pd.DataFrame(log).to_csv(lp, mode="a", header=not lp.exists(), index=False)
    print(f"[segment] finished: {done} new masks in {cache}")


# ================================================================ per-leaf processing
def _box_w(rec, geom):
    return box_in_working((rec["x1"], rec["y1"], rec["x2"], rec["y2"]), geom)


def _leaf_arrays(img, rec, cache, wb_ill, cfg: ZoneConfig, return_raw: bool = False):
    """Crop, refined leaf mask, white-balanced crop and geometry for one leaf box."""
    geom, raw = load_mask(cache, rec["leaf_id"])
    crop = to_working(img, geom)
    if crop.shape[:2] != raw.shape:                        # guard against rounding differences
        import cv2
        crop = cv2.resize(crop, raw.shape[::-1], interpolation=cv2.INTER_AREA)
    rgb01 = apply_white_balance(crop, wb_ill)
    full = refine_mask(raw, _box_w(rec, geom), cfg.trim_frac)
    if cfg.recover_tips:
        L, C, h, _, _ = to_lch(rgb01)
        full = recover_symptom_margin(full, L, C, h, _box_w(rec, geom))
    if return_raw:
        return crop, full, rgb01, geom, raw
    return crop, full, rgb01, geom


def _outside_box(rec, geom, shape):
    """Pixels of the padded working crop that lie outside the annotated box (the padding ring)."""
    x1, y1, x2, y2 = box_in_working((rec["x1"], rec["y1"], rec["x2"], rec["y2"]), geom)
    ring = np.ones(shape, bool)
    ring[max(0, int(np.floor(y1))):int(np.ceil(y2)), max(0, int(np.floor(x1))):int(np.ceil(x2))] = False
    return ring


def _calibrate_image(args):
    img_path, recs, cache, ignore_exif, base = args
    cfg = ZoneConfig.from_dict(base)
    img = load_rgb(img_path, ignore_exif)
    ills = {m: estimate_illuminant(img, m) for m in WB_MODES}
    rows = []
    for r in recs:
        try:
            geom, raw = load_mask(cache, r["leaf_id"])
        except FileNotFoundError:
            continue
        crop = to_working(img, geom)
        if crop.shape[:2] != raw.shape:
            continue
        base_mask = refine_mask(raw, _box_w(r, geom), cfg.trim_frac)
        for m in WB_MODES:
            L, C, h, _, _ = to_lch(apply_white_balance(crop, ills[m]))
            full = recover_symptom_margin(base_mask, L, C, h, _box_w(r, geom)) if cfg.recover_tips else base_mask
            ev = erode_proportional(full, cfg.erosion_frac)
            for g in GLARE_MODES:
                cfg.glare_mode = g
                for (b1, b2), sev in severity_grid(L, C, h, ev, B1_GRID, B2_GRID, cfg).items():
                    rows.append({"leaf_id": r["leaf_id"], "plant_id": r["plant_id"],
                                 "symptomatic": r["symptomatic"], "wb_mode": m, "glare_mode": g,
                                 "b1": b1, "b2": b2, "severity": sev})
    return rows


def _features_image(args):
    img_path, recs, cache, ignore_exif, cfg_d, veins = args
    cfg = ZoneConfig.from_dict(cfg_d)
    img = load_rgb(img_path, ignore_exif)
    ill = estimate_illuminant(img, cfg.wb_mode)
    rows = []
    for r in recs:
        try:
            crop, full, rgb01, geom = _leaf_arrays(img, r, cache, ill, cfg)
        except FileNotFoundError:
            continue
        L, C, h, a_, b_ = to_lch(rgb01)
        classes = classify_pixels(L, C, h, full, cfg)
        ev = erode_proportional(full, cfg.erosion_frac)
        row = {"leaf_id": r["leaf_id"]}
        row.update(zone_fractions(classes, ev))
        for e in EROSION_SENS:                                     # erosion sensitivity
            row[f"sev_e{int(round(e * 1000)):03d}"] = zone_fractions(
                classes, erode_proportional(full, e))["severity"]
        evalpx = ev & ~np.isin(classes, (0, 1, 2))
        row.update(baseline_indices(rgb01, L, C, h, a_, evalpx))
        row.update(placement_features(classes, full, ev, L, veins=veins))
        # background for the shortcut test: raw colours of the padding ring OUTSIDE the annotated box,
        # so symptom tissue the mask may have missed cannot leak into "background"
        Lr, Cr, hr, ar, br = to_lch(crop.astype(np.float64) / 255.0)
        row.update(background_features(None, Lr, Cr, hr, ar, br, _outside_box(r, geom, full.shape) & ~full))
        row["mask_frac_of_crop"] = float(full.mean())
        rows.append(row)
    return rows


def _run_parallel(fn, jobs, workers):
    out = []
    if workers <= 1:
        for i, j in enumerate(jobs):
            out.extend(fn(j))
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(jobs)} images")
        return out
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, res in enumerate(ex.map(fn, jobs, chunksize=4)):
            out.extend(res)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(jobs)} images")
    return out


# ================================================================ calibrate
def cmd_calibrate(a):
    df = read_leaf_table(a.leaf_table, a.day_step)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    split = make_split(df, out, seed=a.seed)
    cal_plants = set(split.loc[split["split"] == "calibration", "plant_id"])
    cal = df[df["plant_id"].isin(cal_plants)]
    print(f"[calibrate] {cal['plant_id'].nunique()} calibration plants, {len(cal)} leaves "
          f"({cal['symptomatic'].sum()} Y)")
    base = ZoneConfig(erosion_frac=a.erosion_frac, trim_frac=a.trim_frac, recover_tips=a.recover_tips)
    jobs = [(p, recs, a.cache, a.ignore_exif, base.to_dict()) for p, recs in _image_groups(cal)]
    grid = pd.DataFrame(_run_parallel(_calibrate_image, jobs, a.workers))
    grid.to_csv(out / "calibration_leaf_grid.csv", index=False)

    keys = ["wb_mode", "glare_mode", "b1", "b2"]
    res = pd.DataFrame([{**dict(zip(keys, k)),
                         "auc": su.safe_auc(d["symptomatic"], d["severity"]),
                         "pr_auc": su.safe_ap(d["symptomatic"], d["severity"]), "n": len(d)}
                        for k, d in grid.groupby(keys)])
    res["dist_from_default"] = np.hypot(res["b1"] - 70, res["b2"] - 110)
    res["auc_r"] = res["auc"].round(3)
    res = res.sort_values(["auc_r", "dist_from_default"], ascending=[False, True])
    res.drop(columns="auc_r").to_csv(out / "calibration_grid.csv", index=False)
    best = res.iloc[0]
    cfg = ZoneConfig(b1=float(best["b1"]), b2=float(best["b2"]), wb_mode=str(best["wb_mode"]),
                     glare_mode=str(best["glare_mode"]), erosion_frac=a.erosion_frac,
                     trim_frac=a.trim_frac, recover_tips=a.recover_tips)
    sel = np.ones(len(grid), bool)
    for k in keys:
        sel &= (grid[k] == best[k]).values
    d = grid[sel]
    cfg.severity_cutoff = su.youden_cutoff(d["symptomatic"], d["severity"])
    _write_config(cfg, out, extra={"calibration_auc": float(best["auc"]), "split_seed": a.seed,
                                   "n_calibration_plants": len(cal_plants)})
    print(res.head(10).drop(columns=["auc_r"]).to_string(index=False))
    print(f"[calibrate] chosen: wb={cfg.wb_mode} glare={cfg.glare_mode} b1={cfg.b1} b2={cfg.b2} "
          f"cut-off={cfg.severity_cutoff:.2f}% (calibration AUC {best['auc']:.3f}); "
          f"trim={cfg.trim_frac} recover_tips={cfg.recover_tips}")


def _write_config(cfg: ZoneConfig, out: Path, extra: dict):
    body = {"zone_config": cfg.to_dict(), **extra}
    text = json.dumps(body, indent=2, sort_keys=True)
    body["sha256"] = hashlib.sha256(text.encode()).hexdigest()
    body["frozen_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out / "frozen_config.json").write_text(json.dumps(body, indent=2, sort_keys=True))
    print(f"[calibrate] frozen config -> {out / 'frozen_config.json'} (sha256 {body['sha256'][:12]}…)")


def load_config(out: Path) -> ZoneConfig:
    p = out / "frozen_config.json"
    if not p.exists():
        raise FileNotFoundError("run `calibrate` first: frozen_config.json not found")
    return ZoneConfig.from_dict(json.loads(p.read_text())["zone_config"])


# ================================================================ features
def cmd_features(a):
    df = read_leaf_table(a.leaf_table, a.day_step)
    out = Path(a.out)
    cfg = load_config(out)
    print(f"[features] config: {cfg}")
    jobs = [(p, recs, a.cache, a.ignore_exif, cfg.to_dict(), not a.no_veins)
            for p, recs in _image_groups(df)]
    feats = pd.DataFrame(_run_parallel(_features_image, jobs, a.workers))
    full = df.merge(feats, on="leaf_id", how="inner")
    split = make_split(df, out, seed=a.seed)
    full = full.merge(split[["plant_id", "split"]], on="plant_id", how="left")
    full.to_csv(out / "features.csv", index=False)
    print(f"[features] {len(full)} leaves -> {out / 'features.csv'}")


# ================================================================ analyse
def _clr(df, cols, eps=1e-3):
    x = np.log(df[cols].astype(float).fillna(0).values + eps)
    x = x - x.mean(axis=1, keepdims=True)
    return pd.DataFrame(x[:, :-1], columns=[f"clr_{c}" for c in cols[:-1]], index=df.index)


def _detection(f, cfg, out, n_boot):
    cal, test = f[f.split == "calibration"], f[f.split == "test"]
    rows = []

    def add(name, col, data_cal, data_test, cutoff=None, sign=1.0):
        d = data_test.assign(_s=sign * data_test[col])
        cut = cutoff if cutoff is not None else su.youden_cutoff(
            data_cal["symptomatic"], sign * data_cal[col])
        bs = su.group_bootstrap(d, "plant_id",
                                lambda x: su.binary_metrics(x["symptomatic"], x["_s"], cut),
                                n_boot=n_boot)
        bs.update({"score": name, "cutoff": cut, "sign": sign})
        rows.append(bs)

    add("severity (zones)", "severity", cal, test, cutoff=cfg.severity_cutoff)
    add("discoloured (zones + red)", "discoloured", cal, test)
    for e in EROSION_SENS:
        add(f"severity, erosion {e:.0%}", f"sev_e{int(round(e * 1000)):03d}", cal, test)
    for col in [c for c in f.columns if c.startswith("bl_")]:
        sign = 1.0 if su.safe_auc(cal["symptomatic"], cal[col]) >= 0.5 else -1.0  # direction from calibration
        add(f"baseline {col[3:]}", col, cal, test, sign=sign)
    for cls in sorted(test["class_name"].unique()):            # per class, main score only
        t, c = test[test.class_name == cls], cal[cal.class_name == cls]
        if t["symptomatic"].nunique() == 2:
            add(f"severity (zones), {cls} plants only", "severity", c, t, cutoff=cfg.severity_cutoff)
    for (cls, trt), t in test.groupby(["class_name", "treatment"]):   # per treatment, main score only
        if t["symptomatic"].nunique() == 2 and t["plant_id"].nunique() >= 2:
            c = cal[(cal.class_name == cls) & (cal.treatment == trt)]
            add(f"severity (zones), {cls}: {trt}", "severity", c, t, cutoff=cfg.severity_cutoff)
    det = pd.DataFrame(rows)
    first = ["score", "auc", "auc_lo", "auc_hi", "pr_auc", "pr_auc_lo", "pr_auc_hi", "sensitivity",
             "specificity", "balanced_accuracy", "cutoff", "n", "n_groups"]
    det = det[[c for c in first if c in det.columns] + [c for c in det.columns if c not in first]]
    det.to_csv(out / "detection.csv", index=False)
    return det


def _progression(f, out):
    t = f[f.split == "test"].copy()
    t["day_i"] = np.floor(t["day"]).astype(int)
    pd_rows = []
    for (p, d), g in t.groupby(["plant_id", "day_i"]):
        pd_rows.append({"plant_id": p, "class_name": g["class_name"].iloc[0], "day": d,
                        "sev_all": g["severity"].mean(),
                        "sev_Y": g.loc[g.symptomatic == 1, "severity"].mean(),
                        "sev_N": g.loc[g.symptomatic == 0, "severity"].mean(),
                        "bg_L": g["bgf_L"].mean(), "bg_hue": g["bgf_hue"].mean(),
                        "frac_Y_boxes": g["symptomatic"].mean()})
    pdays = pd.DataFrame(pd_rows)
    pdays.to_csv(out / "progression_plant_days.csv", index=False)

    per_plant = None
    summary = []
    for series in ["sev_all", "sev_Y", "sev_N", "bg_L", "bg_hue", "frac_Y_boxes"]:
        sp = su.per_group_spearman(pdays, "plant_id", "day", series).rename(columns={"rho": f"rho_{series}"})
        per_plant = sp[["plant_id", f"rho_{series}"]] if per_plant is None else per_plant.merge(
            sp[["plant_id", f"rho_{series}"]], on="plant_id")
        w = su.wilcoxon_greater(sp[f"rho_{series}"])
        summary.append({"series": series, **w})
    raudpc = pd.DataFrame([{"plant_id": p, "rel_audpc_all": su.relative_audpc(g["day"], g["sev_all"])}
                           for p, g in pdays.groupby("plant_id")])
    per_plant = per_plant.merge(raudpc, on="plant_id").merge(
        pdays.groupby("plant_id")["class_name"].first().reset_index(), on="plant_id")
    per_plant.to_csv(out / "progression_per_plant.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "progression_summary.csv", index=False)

    mm = su.mixed_model(t[["severity", "day", "symptomatic", "plant_id"]].dropna(),
                        "severity ~ day * symptomatic", "plant_id", re_formula="~day")
    if mm is not None:
        mm.to_csv(out / "progression_mixed_model.csv")
    return pdays, pd.DataFrame(summary), mm


def _discrimination(f, out, n_repeats, n_boot, seed, y_df=None, label_col="class_name",
                    positive=POSITIVE_CLASS, prefix="discrimination"):
    """Grouped-CV logistic regression on Y leaves: baselines vs colour vs colour + placement,
    plus a background-only shortcut test. Default task: tip burn vs yellowing."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if y_df is None:
        y_df = f[f.symptomatic == 1]
    y_df = y_df.copy().reset_index(drop=True)
    y = (y_df[label_col] == positive).astype(int).values
    groups = y_df["plant_id"].values
    plants_per_label = y_df.groupby(label_col)["plant_id"].nunique()
    n_splits = int(min(5, plants_per_label.min())) if len(plants_per_label) == 2 else 0
    if n_splits < 2:
        print(f"[{prefix}] skipped: needs at least 2 plants per group, have {plants_per_label.to_dict()}")
        return None, None, None, None
    frac_cols = ["frac_necrotic", "frac_chlorotic", "frac_red", "frac_other", "frac_healthy"]
    colour = _clr(y_df, frac_cols)
    placement = y_df[[c for c in y_df.columns if c.startswith("pl_")]]
    sets = {
        "baselines": y_df[[c for c in y_df.columns if c.startswith("bl_")]],
        "colour fractions": colour,
        "colour + placement": pd.concat([colour, placement], axis=1),
        "placement only": placement,
        "background only (shortcut test)": y_df[[c for c in y_df.columns if c.startswith("bgf_")]],
    }
    # drop features that are missing for every leaf (e.g. vein features with --no-veins)
    sets = {k: v.loc[:, v.notna().any()] for k, v in sets.items()}

    def model():
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000))

    oof = {}
    for name, X in sets.items():
        X = X.astype(float).values
        acc = np.zeros(len(y))
        for r in range(n_repeats):
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
            for tr, te in cv.split(X, y, groups):
                acc[te] += model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        oof[name] = acc / n_repeats
    pred = y_df[["leaf_id", "plant_id", "class_name", "treatment"]].copy()
    pred["y"] = y
    for name, p in oof.items():
        pred[f"p_{name}"] = p
    pred.to_csv(out / f"{prefix}_oof_predictions.csv", index=False, encoding="utf-8-sig")

    def metrics_for(col):
        def fn(d):
            leaf_auc = su.safe_auc(d["y"], d[col])
            bal = su.binary_metrics(d["y"], d[col], 0.5)["balanced_accuracy"]
            pl = d.groupby("plant_id").agg(y=("y", "first"), p=(col, "mean"))
            return {"leaf_auc": leaf_auc, "balanced_accuracy": bal, "plant_auc": su.safe_auc(pl["y"], pl["p"])}
        return fn

    rows = []
    for name in sets:
        bs = su.group_bootstrap(pred, "plant_id", metrics_for(f"p_{name}"), n_boot=n_boot)
        rows.append({"feature_set": name, **bs})
    gain = su.group_bootstrap(
        pred, "plant_id",
        lambda d: su.safe_auc(d["y"], d["p_colour + placement"]) - su.safe_auc(d["y"], d["p_colour fractions"]),
        n_boot=n_boot)
    rows.append({"feature_set": "AUC gain: colour + placement minus colour", "leaf_auc": gain["value"],
                 "leaf_auc_lo": gain["value_lo"], "leaf_auc_hi": gain["value_hi"], "n_groups": gain["n_groups"]})
    disc = pd.DataFrame(rows)
    disc.to_csv(out / f"{prefix}.csv", index=False)

    # per treatment: share of leaves classified correctly, and (where a treatment occurs in both
    # classes, e.g. low EC) AUC within that treatment, which cannot come from treatment-linked artefacts
    trows = []
    for (cls, trt), g in pred.groupby(["class_name", "treatment"]):
        r = {"class_name": cls, "treatment": trt, "n_plants": g["plant_id"].nunique(), "n_leaves": len(g)}
        for name in ("colour fractions", "colour + placement", "background only (shortcut test)"):
            r[f"correct_{name}"] = float(np.mean((g[f"p_{name}"] >= 0.5).astype(int) == g["y"]))
        trows.append(r)
    for trt, g in pred.groupby("treatment"):
        if g["y"].nunique() == 2:
            r = {"class_name": "both classes", "treatment": trt, "n_plants": g["plant_id"].nunique(), "n_leaves": len(g)}
            for name in ("colour fractions", "colour + placement", "background only (shortcut test)"):
                r[f"within_treatment_auc_{name}"] = su.safe_auc(g["y"], g[f"p_{name}"])
            trows.append(r)
    by_trt = pd.DataFrame(trows)
    by_trt.to_csv(out / f"{prefix}_by_treatment.csv", index=False, encoding="utf-8-sig")

    X = sets["colour + placement"]
    m = model().fit(X.astype(float).values, y)
    coefs = pd.DataFrame({"feature": X.columns, f"std_coef_towards_{positive}": m[-1].coef_[0]})
    coefs = coefs.reindex(coefs.iloc[:, 1].abs().sort_values(ascending=False).index)
    coefs.to_csv(out / f"{prefix}_coefficients.csv", index=False)

    cols = [c for c in y_df.columns if c.startswith("pl_share")] + \
           ["pl_chl_off_minus_on", "frac_necrotic", "frac_chlorotic", "frac_red"]
    cols = [c for c in cols if c in y_df.columns]
    means = y_df.groupby(label_col)[cols].mean().T
    means.to_csv(out / f"{prefix}_placement_means.csv", encoding="utf-8-sig")
    y_df.groupby(["class_name", "treatment"])[cols].mean().T.to_csv(
        out / f"{prefix}_placement_means_by_treatment.csv", encoding="utf-8-sig")
    return disc, coefs, means, by_trt


def _figures(f, pdays, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve
    except ImportError:
        return
    test = f[f.split == "test"].dropna(subset=["severity"])
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    if test["symptomatic"].nunique() == 2:
        fpr, tpr, _ = roc_curve(test["symptomatic"], test["severity"])
        ax.plot(fpr, tpr, label=f"severity (AUC {su.safe_auc(test['symptomatic'], test['severity']):.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="1 − specificity", ylabel="Sensitivity", title="Y vs N leaves, test plants")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig_roc_detection.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    cols = {"tipburn": "tab:red", "yellowing": "tab:olive"}
    for p, g in pdays.groupby("plant_id"):
        g = g.sort_values("day")
        ax.plot(g["day"], g["sev_all"], color=cols.get(g["class_name"].iloc[0], "grey"), alpha=0.5, lw=1)
    for k, c in cols.items():
        ax.plot([], [], color=c, label=k)
    ax.set(xlabel="Days since first image", ylabel="Mean leaf severity (%)", title="Progression, test plants")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig_progression.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)
    for ax, (s, lab) in zip(axes, [(1, "Y leaves"), (0, "N leaves")]):
        for k, c in cols.items():
            v = f[(f.symptomatic == s) & (f.class_name == k)]["severity"].dropna()
            ax.hist(v, bins=30, range=(0, 100), alpha=0.5, color=c, label=k)
        ax.set(title=lab, xlabel="Severity (%)")
    axes[0].set_ylabel("Leaves")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out / "fig_severity_histograms.png", dpi=150)
    plt.close(fig)


def render_leaf_panel(rec: dict, cache, cfg: ZoneConfig, path: Path, ignore_exif=False, title="", dpi=110):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    img = load_rgb(rec["image_path"], ignore_exif)
    crop, full, rgb01, geom, raw = _leaf_arrays(img, rec, cache, estimate_illuminant(img, cfg.wb_mode), cfg,
                                                return_raw=True)
    L, C, h, _, _ = to_lch(rgb01)
    classes = classify_pixels(L, C, h, full, cfg)
    ev = erode_proportional(full, cfg.erosion_frac)
    classes[full & ~ev] = BG
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    axes[0].imshow(crop)
    axes[0].set_title("crop (raw)")
    axes[1].imshow(crop)
    axes[1].contour(raw, levels=[0.5], colors="red", linewidths=0.8, linestyles="dashed")
    axes[1].contour(full, levels=[0.5], colors="cyan", linewidths=1.2)
    bx1, by1, bx2, by2 = _box_w(rec, geom)
    axes[1].plot([bx1, bx2, bx2, bx1, bx1], [by1, by1, by2, by2, by1], color="white", lw=0.8, ls=":")
    axes[1].set_title("mask: cyan = used, red dashed = raw SAM")
    axes[2].imshow(class_map_to_rgb(classes))
    axes[2].set_title("pixel classes")
    for ax in axes:
        ax.axis("off")
    fig.legend(handles=[Patch(color=np.array(c) / 255, label=CLASS_NAMES[k]) for k, c in CLASS_COLOURS.items()
                        if k != BG], loc="lower center", ncol=7, fontsize=8)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def cmd_analyse(a):
    out = Path(a.out)
    cfg = load_config(out)
    f = pd.read_csv(out / "features.csv", dtype={"plant_id": str, "leaf_id": str})
    print(f"[analyse] {len(f)} leaves, {f['plant_id'].nunique()} plants")
    det = _detection(f, cfg, out, a.n_boot)
    pdays, prog, mm = _progression(f, out)
    disc, coefs, means, by_trt = _discrimination(f, out, a.n_repeats, a.n_boot, a.seed)
    # exploratory: within yellowing, N deficiency (older leaves, veins yellow too) vs
    # Fe deficiency (young leaves, chlorosis between green veins) - a direct test of the vein features
    nfe = f[(f.symptomatic == 1) & (f.class_name == "yellowing") &
            (f.treatment.isin(["N deficiency", "Fe deficiency"]))]
    nfe_res = _discrimination(f, out, a.n_repeats, a.n_boot, a.seed, y_df=nfe, label_col="treatment",
                              positive="Fe deficiency", prefix="exploratory_N_vs_Fe") if len(nfe) else (None,) * 4
    _figures(f, pdays, out)

    # audit: highest-scoring N leaves on test plants
    t = f[(f.split == "test") & (f.symptomatic == 0)].sort_values("severity", ascending=False).head(a.audit_n)
    aud = t[["leaf_id", "image_path", "plant_id", "class_name", "day", "severity", "frac_excluded",
             "frac_shadow", "frac_chlorotic", "frac_necrotic"]].copy()
    aud["error_tag"] = ""   # fill in: young pale leaf / shadow / glare / neighbour in mask / real symptom / other
    aud.to_csv(out / "audit_top_N_leaves.csv", index=False)
    (out / "audit").mkdir(exist_ok=True)
    recs = f.set_index("leaf_id").loc[aud["leaf_id"]].reset_index().to_dict("records")
    for r in recs:
        try:
            render_leaf_panel(r, Path(a.cache), cfg, out / "audit" / f"{r['leaf_id']}.png", a.ignore_exif,
                              f"{r['leaf_id']}  plant {r['plant_id']}  severity {r['severity']:.1f}%")
        except Exception as e:  # noqa: BLE001
            print(f"[audit] {r['leaf_id']}: {e}")

    pd.set_option("display.width", 160)
    lines = ["== Detection (test plants) ==",
             det[["score", "auc", "auc_lo", "auc_hi", "pr_auc", "sensitivity", "specificity", "n_groups"]]
             .round(3).to_string(index=False),
             "", "== Progression (test plants; one-sided Wilcoxon on per-plant Spearman rho) ==",
             prog.round(4).to_string(index=False)]
    if mm is not None:
        lines += ["", f"== Mixed model: severity ~ day * symptomatic; {mm.attrs.get('structure')}; "
                      f"converged cleanly: {mm.attrs.get('converged')} ==", mm.round(4).to_string()]
    lines += ["", "== Tip burn vs yellowing (Y leaves; 5-fold CV grouped by plant) ==",
              disc[[c for c in ["feature_set", "leaf_auc", "leaf_auc_lo", "leaf_auc_hi", "plant_auc",
                                "balanced_accuracy", "n_groups"] if c in disc.columns]].round(3).to_string(index=False),
              "", "== Placement means by class (Y leaves) ==", means.round(3).to_string(),
              "", "== Tip burn vs yellowing, by treatment (share correct; within-treatment AUC where both classes) ==",
              by_trt.round(3).to_string(index=False),
              "", "== Top coefficients (colour + placement) ==", coefs.head(12).round(3).to_string(index=False)]
    if nfe_res[0] is not None:
        d2, _, m2, _ = nfe_res
        lines += ["", "== EXPLORATORY: N vs Fe deficiency within yellowing (Y leaves; grouped CV) ==",
                  d2[[c for c in ["feature_set", "leaf_auc", "leaf_auc_lo", "leaf_auc_hi", "plant_auc", "n_groups"]
                      if c in d2.columns]].round(3).to_string(index=False),
                  "", m2.round(3).to_string()]
    text = "\n".join(lines)
    (out / "summary.txt").write_text(text, encoding="utf-8")
    print(text)


def cmd_overlays(a):
    df = read_leaf_table(a.leaf_table, a.day_step)
    out = Path(a.out)
    try:
        cfg = load_config(out)
        print("[overlays] using the frozen config")
    except FileNotFoundError:
        cfg = ZoneConfig(glare_mode=a.glare_mode, trim_frac=a.trim_frac, recover_tips=a.recover_tips,
                         erosion_frac=a.erosion_frac)
        print(f"[overlays] no frozen config yet; glare={cfg.glare_mode} trim={cfg.trim_frac} "
              f"recover_tips={cfg.recover_tips}")
    d = out / a.overlay_dir
    d.mkdir(parents=True, exist_ok=True)
    cached = df[[(Path(a.cache) / f"{i}.npz").exists() for i in df["leaf_id"]]]
    if a.leaf_ids:
        pick = cached[cached["leaf_id"].isin([x.strip() for x in a.leaf_ids.split(",")])]
    elif a.pick == "severe":                                # most severe Y leaves per class (needs features)
        f = pd.read_csv(out / "features.csv", dtype={"leaf_id": str})[["leaf_id", "severity"]]
        y = cached[cached.symptomatic == 1].merge(f, on="leaf_id")
        per = max(1, a.sample // 2)
        pick = (y.sort_values("severity", ascending=False).groupby("class_name").head(per))
    else:
        per = max(1, a.sample // 4)
        idx = []
        for _, g in cached.groupby(["class_name", "symptomatic"]):
            idx += list(g.sample(min(len(g), per), random_state=a.seed).index)
        pick = cached.loc[idx]
    for r in pick.to_dict("records"):
        render_leaf_panel(r, Path(a.cache), cfg, d / f"{r['class_name']}_{'Y' if r['symptomatic'] else 'N'}_{r['leaf_id']}.png",
                          a.ignore_exif, f"{r['leaf_id']}  {r['class_name']}  {'Y' if r['symptomatic'] else 'N'}",
                          dpi=a.dpi)
    print(f"[overlays] {len(pick)} panels -> {d}")


# ================================================================ CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["segment", "overlays", "calibrate", "features", "analyse", "all"])
    ap.add_argument("--leaf-table", required=True)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="results")
    ap.add_argument("--sam-checkpoint", default=None)
    ap.add_argument("--sam-model", default="vit_b")
    ap.add_argument("--ignore-exif", action="store_true", help="ignore EXIF orientation when reading images")
    ap.add_argument("--day-step", type=int, default=1, help="keep every n-th day per plant (speed fallback)")
    ap.add_argument("--erosion-frac", type=float, default=0.01)
    ap.add_argument("--trim-frac", type=float, default=0.04,
                    help="mask refinement: trims protrusions (e.g. petioles); 0 = off")
    ap.add_argument("--recover-tips", action="store_true",
                    help="add brown/bleached tissue that touches the mask (dried tips SAM left out)")
    ap.add_argument("--glare-mode", choices=["broad", "strict"], default="broad",
                    help="overlays before calibration only; calibration chooses between the two")
    ap.add_argument("--overlay-dir", default="overlays", help="sub-folder of --out for overlay panels")
    ap.add_argument("--leaf-ids", default=None, help="overlays: comma-separated leaf ids to draw")
    ap.add_argument("--pick", choices=["random", "severe"], default="random",
                    help="overlays: random sample, or the most severe Y leaves per class (after features)")
    ap.add_argument("--dpi", type=int, default=110, help="overlays: image resolution (use 250+ for figures)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--no-veins", action="store_true", help="skip the (slower) vein features")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-repeats", type=int, default=20)
    ap.add_argument("--audit-n", type=int, default=20)
    ap.add_argument("--sample", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    cmds = {"segment": cmd_segment, "overlays": cmd_overlays, "calibrate": cmd_calibrate,
            "features": cmd_features, "analyse": cmd_analyse}
    if a.command == "all":
        for c in ["segment", "calibrate", "features", "analyse"]:
            print(f"\n######## {c}")
            cmds[c](a)
    else:
        cmds[a.command](a)


if __name__ == "__main__":
    main()

"""
known_severity.py — known-severity AND known-placement test on real AI Hub 305 leaves.

Lesion colours are sampled from real symptomatic (Y) leaves of CALIBRATION plants, then painted
with soft edges onto symptom-free (N) leaves of TEST plants at 0–60% cover, in three placements
(random, margin, tip). The true severity counts each pixel by its lesion opacity (summed alpha),
plus any symptom pixels the leaf already had, so soft edges are not a source of artificial bias.

Outputs (in --out/known_severity/):
  known_severity_leaves.csv   one row per painted leaf
  agreement.csv               r, slope, intercept, bias, limits of agreement, Lin's CCC, Cb
  placement_recovery.csv      measured margin/tip share of the painted class, by placement mode
  fig_known_severity.png      measured vs true severity

  python known_severity.py --leaf-table leaves.csv --cache cache --out results --n-leaves 40

Caveat (state it in the write-up): the lesion colours are ones the zones already recognise, so this
tests area and placement measurement, not whether the colour zones themselves are right.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from colour_zones import (estimate_illuminant, apply_white_balance, to_lch, classify_pixels,
                          zone_fractions, NECROTIC, CHLOROTIC, BG, GLARE, SHADOW)
from placement import erode_proportional, placement_features, distance_to_edge
from segment import load_rgb, to_working, load_mask
from run_305 import read_leaf_table, load_config, make_split, _leaf_arrays
import stats_utils as su

COVERS = [0, 5, 10, 20, 30, 40, 50, 60]
PLACEMENTS = ["random", "margin", "tip"]
LESIONS = {"necrotic": NECROTIC, "chlorotic": CHLOROTIC}


def _prepare(rec, cache, cfg, ignore_exif):
    img = load_rgb(rec["image_path"], ignore_exif)
    try:
        _, full, rgb01, _ = _leaf_arrays(img, rec, cache, estimate_illuminant(img, cfg.wb_mode), cfg)
    except (FileNotFoundError, ValueError):
        return None
    return rgb01, full


def collect_lesion_colours(recs, cache, cfg, ignore_exif, per_class=40000, seed=0):
    rng = np.random.default_rng(seed)
    pools = {k: [] for k in LESIONS}
    for r in recs:
        prep = _prepare(r, cache, cfg, ignore_exif)
        if prep is None:
            continue
        rgb01, full = prep
        L, C, h, _, _ = to_lch(rgb01)
        cls = classify_pixels(L, C, h, erode_proportional(full, cfg.erosion_frac), cfg)
        for k, code in LESIONS.items():
            px = rgb01[cls == code]
            if len(px):
                pools[k].append(px[rng.choice(len(px), min(len(px), 2000), replace=False)])
    out = {}
    for k, lst in pools.items():
        if not lst:
            raise RuntimeError(f"no {k} pixels found on calibration Y leaves")
        allpx = np.concatenate(lst)
        out[k] = allpx[rng.choice(len(allpx), min(len(allpx), per_class), replace=False)]
        print(f"[known] {k} colour pool: {len(out[k])} pixels")
    return out


def lesion_alpha(ev, full, cover, placement, rng, blob_sigma=6.0, edge_sigma=1.5):
    """Soft lesion opacity (0-1) inside `ev` covering roughly `cover` % of it."""
    if cover <= 0:
        return np.zeros(ev.shape)
    field = ndi.gaussian_filter(rng.random(ev.shape), blob_sigma)
    field = (field - field[ev].mean()) / (field[ev].std() + 1e-9)
    if placement == "margin":
        d = distance_to_edge(full)
        field -= 2.5 * d / (d[ev].max() + 1e-9) * 2
    elif placement == "tip":
        yy, xx = np.nonzero(full)
        cy, cx = yy.mean(), xx.mean()
        Y, X = np.mgrid[0:ev.shape[0], 0:ev.shape[1]]
        r = np.hypot(Y - cy, X - cx)
        field += 5.0 * r / (r[ev].max() + 1e-9)
    thr = np.quantile(field[ev], 1 - cover / 100.0)
    binary = ev & (field >= thr)
    alpha = np.clip(ndi.gaussian_filter(binary.astype(float), edge_sigma), 0, 1)
    alpha[~ev] = 0
    # rescale so summed opacity matches the requested cover (blurring leaks a little mass)
    s = alpha[ev].sum()
    if s > 0:
        alpha = np.clip(alpha * (binary.sum() / s), 0, 1)
    return alpha


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leaf-table", required=True)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="results")
    ap.add_argument("--n-leaves", type=int, default=40)
    ap.add_argument("--n-source-leaves", type=int, default=150)
    ap.add_argument("--ignore-exif", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    out = Path(a.out)
    cfg = load_config(out)
    df = read_leaf_table(a.leaf_table)
    split = make_split(df, out)
    df = df.merge(split[["plant_id", "split"]], on="plant_id")
    cached = df[[(Path(a.cache) / f"{i}.npz").exists() for i in df["leaf_id"]]]
    rng = np.random.default_rng(a.seed)

    src = cached[(cached.split == "calibration") & (cached.symptomatic == 1)]
    src = src.sample(min(len(src), a.n_source_leaves), random_state=a.seed)
    pools = collect_lesion_colours(src.to_dict("records"), a.cache, cfg, a.ignore_exif, seed=a.seed)

    tgt = cached[(cached.split == "test") & (cached.symptomatic == 0)]
    tgt = tgt.sample(min(len(tgt), a.n_leaves), random_state=a.seed)
    rows = []
    for i, r in enumerate(tgt.to_dict("records")):
        prep = _prepare(r, a.cache, cfg, a.ignore_exif)
        if prep is None:
            continue
        rgb01, full = prep
        ev_mask = erode_proportional(full, cfg.erosion_frac)
        L, C, h, _, _ = to_lch(rgb01)
        cls0 = classify_pixels(L, C, h, full, cfg)
        ev = ev_mask & ~np.isin(cls0, (BG, GLARE, SHADOW))
        if ev.sum() < 500:
            continue
        base_sym = np.isin(cls0, (NECROTIC, CHLOROTIC)) & ev
        base_sev = zone_fractions(cls0, ev_mask)["severity"]
        for lesion, code in LESIONS.items():
            for placement in PLACEMENTS:
                for cover in COVERS:
                    alpha = lesion_alpha(ev, full, cover, placement, rng)
                    pool = pools[lesion]
                    paint = pool[rng.integers(0, len(pool), size=ev.shape)]
                    img = rgb01 * (1 - alpha[..., None]) + paint * alpha[..., None]
                    Lp, Cp, hp, _, _ = to_lch(img)
                    cls = classify_pixels(Lp, Cp, hp, full, cfg)
                    meas = zone_fractions(cls, ev_mask)
                    truth = 100.0 * (alpha[ev].sum() + ((1 - alpha) * base_sym)[ev].sum()) / ev.sum()
                    pl = placement_features(cls, full, ev_mask, Lp, veins=False)
                    k = "nec" if lesion == "necrotic" else "chl"
                    rows.append({"leaf_id": r["leaf_id"], "plant_id": r["plant_id"], "lesion": lesion,
                                 "placement": placement, "target_cover": cover, "base_severity": base_sev,
                                 "true_severity": truth, "measured_severity": meas["severity"],
                                 "frac_excluded": meas["frac_excluded"],
                                 "share_margin": pl[f"pl_share_margin_{k}"], "share_tip": pl[f"pl_share_tip_{k}"]})
        print(f"[known] {i + 1}/{len(tgt)} leaves")
    res = pd.DataFrame(rows)
    d = out / "known_severity"
    d.mkdir(parents=True, exist_ok=True)
    res.to_csv(d / "known_severity_leaves.csv", index=False)

    agr = [{"subset": "all", **su.agreement_stats(res.true_severity, res.measured_severity)}]
    for (les, pla), g in res.groupby(["lesion", "placement"]):
        agr.append({"subset": f"{les}, {pla}", **su.agreement_stats(g.true_severity, g.measured_severity)})
    agr = pd.DataFrame(agr)
    agr.to_csv(d / "agreement.csv", index=False)

    rec = (res[res.target_cover >= 10].groupby(["lesion", "placement"])[["share_margin", "share_tip"]]
           .mean().reset_index())
    rec["expected_if_even"] = "margin 0.33, tip 0.25"
    rec.to_csv(d / "placement_recovery.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 5))
        for les, mk in [("necrotic", "o"), ("chlorotic", "s")]:
            g = res[res.lesion == les]
            ax.scatter(g.true_severity, g.measured_severity, s=10, alpha=0.5, marker=mk, label=les)
        lim = max(res.true_severity.max(), res.measured_severity.max()) + 5
        ax.plot([0, lim], [0, lim], "k--", lw=0.8)
        ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="True severity (%)", ylabel="Measured severity (%)",
               title=f"Known severity (CCC {agr.iloc[0]['ccc']:.3f})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(d / "fig_known_severity.png", dpi=150)
        plt.close(fig)
    except ImportError:
        pass
    print(agr.round(3).to_string(index=False))
    print(rec.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

"""
placement.py — leaf-relative placement features, computed from geometry (no learning).

All positions are relative to the leaf mask, never to image coordinates, so leaf
angle and scale do not matter:

  margin bands : three equal-area bands by distance from the leaf edge (margin/middle/core)
  margin share : share of each symptom class inside the margin band (1/3 if spread evenly)
  tip region   : outer 25% of leaf area by distance from the mask centroid (≈ leaflet tips)
  veins        : ridge filter on L*, top 12% response inside the leaf, margin band excluded
  clumping     : share of a symptom class in its largest connected patch

Distances use the UNERODED mask so that erosion does not shift margin positions.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import sato

from colour_zones import BG, GLARE, SHADOW, NECROTIC, CHLOROTIC, RED

SYMPTOMS = {"nec": NECROTIC, "chl": CHLOROTIC, "red": RED}


def equivalent_diameter(mask: np.ndarray) -> float:
    return float(np.sqrt(4.0 * mask.sum() / np.pi))


def erode_proportional(mask: np.ndarray, frac: float) -> np.ndarray:
    """Erode by `frac` x equivalent diameter (e.g. 0.01 = 1%). frac=0 returns the mask."""
    if frac <= 0 or mask.sum() == 0:
        return mask.copy()
    r = int(round(frac * equivalent_diameter(mask)))
    if r < 1:
        return mask.copy()
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    disk = (xx * xx + yy * yy) <= r * r
    out = ndi.binary_erosion(mask, structure=disk)
    return out if out.sum() > 0 else mask.copy()


def distance_to_edge(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance (px) to the nearest non-leaf pixel; the crop border counts as edge."""
    padded = np.pad(mask, 1, constant_values=False)
    return ndi.distance_transform_edt(padded)[1:-1, 1:-1]


def _share(count_in: int, count_total: int, min_px: int) -> float:
    return count_in / count_total if count_total >= min_px else np.nan


def vein_mask(L: np.ndarray, region: np.ndarray, top_frac: float = 0.12,
              sigmas=(1, 2, 3)) -> np.ndarray:
    """Pixels in `region` with the strongest ridge response (light or dark ridges) on L*."""
    if region.sum() < 200:
        return np.zeros_like(region)
    Lf = L.copy()
    Lf[~region] = float(L[region].mean())        # suppress outside-leaf edges
    resp = np.maximum(sato(Lf, sigmas=sigmas, black_ridges=True),
                      sato(Lf, sigmas=sigmas, black_ridges=False))
    thr = np.quantile(resp[region], 1.0 - top_frac)
    return region & (resp >= thr)


def placement_features(classes: np.ndarray, full_mask: np.ndarray, eval_mask: np.ndarray,
                       L: np.ndarray, veins: bool = True, min_px: int = 20) -> dict:
    """Leaf-relative placement features (prefix 'pl_').

    classes   : class map from colour_zones.classify_pixels (computed on the eroded mask)
    full_mask : uneroded leaf mask (for distances and centroid)
    eval_mask : eroded leaf mask
    """
    ev = eval_mask & ~np.isin(classes, (BG, GLARE, SHADOW))
    feats: dict[str, float] = {}
    n_ev = int(ev.sum())
    names = [f"pl_{b}_{k}" for b in ("margin", "middle", "core") for k in SYMPTOMS] + \
            [f"pl_share_margin_{k}" for k in SYMPTOMS] + \
            [f"pl_tip_{k}" for k in SYMPTOMS] + [f"pl_share_tip_{k}" for k in SYMPTOMS] + \
            ["pl_chl_onvein", "pl_chl_offvein", "pl_chl_off_minus_on"] + \
            [f"pl_largest_patch_{k}" for k in ("nec", "chl")]
    if n_ev < 100:
        return {n: np.nan for n in names}

    # --- margin bands (equal-area tertiles of distance to the edge)
    d_all = distance_to_edge(full_mask)
    d = d_all[ev]
    q1, q2 = np.quantile(d, [1 / 3, 2 / 3])
    band = np.full(d.shape, 1, dtype=np.int8)       # 0 margin, 1 middle, 2 core
    band[d <= q1] = 0
    band[d > q2] = 2
    cls = classes[ev]
    for k, code in SYMPTOMS.items():
        is_k = cls == code
        for bi, bname in enumerate(("margin", "middle", "core")):
            in_b = band == bi
            feats[f"pl_{bname}_{k}"] = is_k[in_b].mean() if in_b.any() else np.nan
        feats[f"pl_share_margin_{k}"] = _share(int((is_k & (band == 0)).sum()), int(is_k.sum()), min_px)

    # --- tip region (outer 25% of area by distance from the centroid)
    yy, xx = np.nonzero(full_mask)
    cy, cx = yy.mean(), xx.mean()
    ey, ex = np.nonzero(ev)
    r = np.hypot(ey - cy, ex - cx)
    tip = r >= np.quantile(r, 0.75)
    for k, code in SYMPTOMS.items():
        is_k = cls == code
        feats[f"pl_tip_{k}"] = is_k[tip].mean() if tip.any() else np.nan
        feats[f"pl_share_tip_{k}"] = _share(int((is_k & tip).sum()), int(is_k.sum()), min_px)

    # --- veins (margin band excluded: the leaf edge itself is a strong ridge)
    if veins:
        margin_px = np.zeros_like(ev)
        margin_px[ev] = band == 0
        region = ev & ~margin_px
        vm = vein_mask(L, region)
        on, off = region & vm, region & ~vm
        chl = classes == CHLOROTIC
        on_f = chl[on].mean() if on.sum() >= min_px else np.nan
        off_f = chl[off].mean() if off.sum() >= min_px else np.nan
        feats.update({"pl_chl_onvein": on_f, "pl_chl_offvein": off_f,
                      "pl_chl_off_minus_on": off_f - on_f})
    else:
        feats.update({"pl_chl_onvein": np.nan, "pl_chl_offvein": np.nan, "pl_chl_off_minus_on": np.nan})

    # --- clumping (share of the class in its largest 8-connected patch)
    s8 = np.ones((3, 3), dtype=bool)
    for k in ("nec", "chl"):
        m = ev & (classes == SYMPTOMS[k])
        tot = int(m.sum())
        if tot >= min_px:
            lab, n = ndi.label(m, structure=s8)
            sizes = np.bincount(lab.ravel())[1:]
            feats[f"pl_largest_patch_{k}"] = sizes.max() / tot
        else:
            feats[f"pl_largest_patch_{k}"] = np.nan
    return feats

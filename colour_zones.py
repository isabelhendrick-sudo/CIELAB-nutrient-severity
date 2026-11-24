"""
colour_zones.py — training-free CIELAB colour analysis for strawberry leaves.

Pixel classes are checked in a fixed order (first match wins):

    1 GLARE      glare_rule (broad or strict, chosen at calibration)    excluded
    2 SHADOW     L* < 24 and C >= 10 and b2 <= h < 250                 excluded
    3 NECROTIC   L* < 24 or C < 10 or 45 <= h < b1                    severity
    4 CHLOROTIC  b1 <= h < b2 and C >= 15                             severity
    5 HEALTHY    b2 <= h < 250                                        denominator
    6 RED        (h < 45 or h >= 300) and C >= 15                     discoloured
    7 OTHER      everything left                                       denominator

b1 is the shared necrotic|chlorotic edge, b2 the shared chlorotic|healthy edge,
so moving either one moves both neighbouring classes (no inert settings, no gaps).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
import numpy as np
from skimage.color import rgb2lab, rgb2hsv

# class codes
BG, GLARE, SHADOW, NECROTIC, CHLOROTIC, HEALTHY, RED, OTHER = 0, 1, 2, 3, 4, 5, 6, 7
CLASS_NAMES = {BG: "background", GLARE: "glare", SHADOW: "shadow", NECROTIC: "necrotic",
               CHLOROTIC: "chlorotic", HEALTHY: "healthy", RED: "red", OTHER: "other"}
CLASS_COLOURS = {BG: (0, 0, 0), GLARE: (255, 255, 255), SHADOW: (90, 90, 90),
                 NECROTIC: (120, 60, 20), CHLOROTIC: (240, 220, 0), HEALTHY: (40, 160, 40),
                 RED: (200, 30, 120), OTHER: (0, 160, 220)}
WB_MODES = ("raw", "greyworld", "lowsat")
GLARE_MODES = ("broad", "strict")


def glare_rule(L, C, mode: str = "broad"):
    """broad  : (L* > 90 and C < 10) or (L* > 65 and C < 8)   (original; also removes pale, bleached tissue)
    strict : L* > 92 and C < 6                                (specular spots only; bleached tissue -> necrotic)"""
    if mode == "strict":
        return (L > 92) & (C < 6)
    return ((L > 90) & (C < 10)) | ((L > 65) & (C < 8))


@dataclass
class ZoneConfig:
    b1: float = 70.0             # necrotic | chlorotic hue edge (deg)
    b2: float = 110.0            # chlorotic | healthy hue edge (deg)
    healthy_hi: float = 250.0    # upper hue edge of healthy
    necrotic_hue_lo: float = 45.0
    red_hi: float = 45.0         # red if h < red_hi ...
    red_lo: float = 300.0        # ... or h >= red_lo
    dark_L: float = 24.0
    lowC: float = 10.0
    minC_coloured: float = 15.0
    wb_mode: str = "raw"
    erosion_frac: float = 0.01   # erosion radius as fraction of leaf equivalent diameter
    glare_mode: str = "broad"    # "broad": original rule; "strict": only bright colourless specular spots
    trim_frac: float = 0.04      # mask refinement: trims protrusions (petioles) narrower than ~2x this
    recover_tips: bool = False   # mask refinement: add brown/bleached tissue touching the mask
    severity_cutoff: float | None = None   # set by Youden's J at calibration

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in keys})


# ---------------------------------------------------------------- colour correction
def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def estimate_illuminant(image_rgb: np.ndarray, mode: str, lowsat_quantile: float = 0.10,
                        max_side: int = 800) -> np.ndarray | None:
    """Estimate the light colour (linear RGB, sums to 3) from a WHOLE image.

    greyworld : mean of all pixels (original plan; biased on green-dominated frames)
    lowsat    : mean of the least-saturated pixels (bottom `lowsat_quantile`),
                excluding clipped (any channel >= 250) and very dark (V < 0.15) pixels.
    Returns None for mode 'raw'.
    """
    if mode == "raw":
        return None
    img = image_rgb
    h, w = img.shape[:2]
    if max(h, w) > max_side:                       # speed: subsample
        step = int(np.ceil(max(h, w) / max_side))
        img = img[::step, ::step]
    f = img.astype(np.float64) / 255.0
    lin = srgb_to_linear(f).reshape(-1, 3)
    if mode == "greyworld":
        ill = lin.mean(axis=0)
    elif mode == "lowsat":
        hsv = rgb2hsv(f).reshape(-1, 3)
        ok = (img.reshape(-1, 3).max(axis=1) < 250) & (hsv[:, 2] > 0.15)
        if ok.sum() < 100:
            return None
        s = hsv[ok, 1]
        thr = np.quantile(s, lowsat_quantile)
        sel = lin[ok][s <= thr]
        ill = sel.mean(axis=0)
    else:
        raise ValueError(f"unknown wb mode {mode!r}")
    ill = np.maximum(ill, 1e-6)
    return 3 * ill / ill.sum()


def apply_white_balance(rgb_uint8: np.ndarray, illuminant: np.ndarray | None) -> np.ndarray:
    """Von Kries correction in linear RGB; green gain fixed at 1. Returns sRGB float [0,1]."""
    f = rgb_uint8.astype(np.float64) / 255.0
    if illuminant is None:
        return f
    gains = illuminant[1] / illuminant
    lin = srgb_to_linear(f) * gains[None, None, :]
    return linear_to_srgb(lin)


def to_lch(rgb01: np.ndarray):
    """sRGB float [0,1] -> (L*, C, h in degrees [0,360), a*, b*)."""
    lab = rgb2lab(rgb01)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    C = np.hypot(a, b)
    h = np.degrees(np.arctan2(b, a)) % 360.0
    return L, C, h, a, b


# ---------------------------------------------------------------- pixel classes
def classify_pixels(L, C, h, leaf_mask, cfg: ZoneConfig) -> np.ndarray:
    """Return an int8 class map (codes above). Pixels outside leaf_mask = BG."""
    out = np.full(L.shape, BG, dtype=np.int8)
    todo = leaf_mask.copy()

    def take(cond, code):
        nonlocal todo
        sel = todo & cond
        out[sel] = code
        todo &= ~sel

    take(glare_rule(L, C, cfg.glare_mode), GLARE)
    take((L < cfg.dark_L) & (C >= cfg.lowC) & (h >= cfg.b2) & (h < cfg.healthy_hi), SHADOW)
    take((L < cfg.dark_L) | (C < cfg.lowC) | ((h >= cfg.necrotic_hue_lo) & (h < cfg.b1)), NECROTIC)
    take((h >= cfg.b1) & (h < cfg.b2) & (C >= cfg.minC_coloured), CHLOROTIC)
    take((h >= cfg.b2) & (h < cfg.healthy_hi), HEALTHY)
    take(((h < cfg.red_hi) | (h >= cfg.red_lo)) & (C >= cfg.minC_coloured), RED)
    out[todo] = OTHER
    return out


def zone_fractions(classes: np.ndarray, eval_mask: np.ndarray | None = None) -> dict:
    """Fractions (0-1) of each class among evaluated pixels, plus severity scores (%).

    Evaluated = leaf pixels that are not glare or shadow (optionally restricted by eval_mask).
    """
    leaf = classes != BG
    if eval_mask is not None:
        leaf &= eval_mask
    n_leaf = int(leaf.sum())
    excluded = leaf & np.isin(classes, (GLARE, SHADOW))
    ev = leaf & ~excluded
    n = int(ev.sum())
    res = {"n_leaf_px": n_leaf, "n_eval_px": n,
           "frac_excluded": excluded.sum() / n_leaf if n_leaf else np.nan,
           "frac_glare": (leaf & (classes == GLARE)).sum() / n_leaf if n_leaf else np.nan,
           "frac_shadow": (leaf & (classes == SHADOW)).sum() / n_leaf if n_leaf else np.nan}
    for code in (NECROTIC, CHLOROTIC, HEALTHY, RED, OTHER):
        res[f"frac_{CLASS_NAMES[code]}"] = (ev & (classes == code)).sum() / n if n else np.nan
    res["severity"] = 100 * (res["frac_necrotic"] + res["frac_chlorotic"]) if n else np.nan
    res["discoloured"] = res["severity"] + 100 * res["frac_red"] if n else np.nan
    return res


def severity_grid(L, C, h, eval_leaf_mask, b1_values, b2_values, base_cfg: ZoneConfig) -> dict:
    """Severity (%) for every (b1, b2) pair, computed cheaply from shared pixel arrays.

    Only depends on pixels inside eval_leaf_mask (already eroded).
    """
    Lm, Cm, hm = L[eval_leaf_mask], C[eval_leaf_mask], h[eval_leaf_mask]
    glare = glare_rule(Lm, Cm, base_cfg.glare_mode)
    dark = Lm < base_cfg.dark_L
    lowc = Cm < base_cfg.lowC
    out = {}
    for b2 in b2_values:
        shadow = ~glare & dark & ~lowc & (hm >= b2) & (hm < base_cfg.healthy_hi)
        ev = ~glare & ~shadow
        n = ev.sum()
        for b1 in b1_values:
            nec = ev & (dark | lowc | ((hm >= base_cfg.necrotic_hue_lo) & (hm < b1)))
            chl = ev & ~nec & (hm >= b1) & (hm < b2) & (Cm >= base_cfg.minC_coloured)
            out[(b1, b2)] = 100.0 * (nec.sum() + chl.sum()) / n if n else np.nan
    return out


def class_map_to_rgb(classes: np.ndarray) -> np.ndarray:
    rgb = np.zeros(classes.shape + (3,), dtype=np.uint8)
    for code, col in CLASS_COLOURS.items():
        rgb[classes == code] = col
    return rgb


# ---------------------------------------------------------------- baselines
def baseline_indices(rgb01: np.ndarray, L, C, h, a, eval_mask: np.ndarray) -> dict:
    """Six training-free whole-leaf indices over evaluated pixels.

    mean_hue (circular), mean_a, mean_C, DGCI (Karcher & Richardson 2003),
    ExG and ExGR (Woebbecke et al. 1995; Meyer & Neto 2008) on chromatic coordinates.
    """
    if eval_mask.sum() == 0:
        return {k: np.nan for k in ("bl_mean_hue", "bl_mean_a", "bl_mean_C", "bl_mean_L",
                                    "bl_dgci", "bl_exg", "bl_exgr")}
    hr = np.radians(h[eval_mask])
    mean_hue = np.degrees(np.arctan2(np.sin(hr).mean(), np.cos(hr).mean())) % 360
    px = rgb01[eval_mask]
    s = px.sum(axis=1, keepdims=True)
    s[s == 0] = 1e-9
    r, g, b = (px / s).T
    exg = 2 * g - r - b
    exr = 1.4 * r - g
    hsv = rgb2hsv(rgb01[None, eval_mask])[0]          # (n,3); H in [0,1)
    H = hsv[:, 0] * 360.0
    dgci = ((H - 60.0) / 60.0 + (1 - hsv[:, 1]) + (1 - hsv[:, 2])) / 3.0
    return {"bl_mean_hue": mean_hue, "bl_mean_a": float(a[eval_mask].mean()),
            "bl_mean_C": float(C[eval_mask].mean()), "bl_mean_L": float(L[eval_mask].mean()),
            "bl_dgci": float(dgci.mean()), "bl_exg": float(exg.mean()),
            "bl_exgr": float((exg - exr).mean())}


def background_features(rgb01: np.ndarray, L, C, h, a, b, bg_mask: np.ndarray) -> dict:
    """Colour summary of NON-leaf pixels (for the shortcut test)."""
    if bg_mask.sum() < 50:
        return {k: np.nan for k in ("bgf_L", "bgf_a", "bgf_b", "bgf_C", "bgf_Lsd", "bgf_hue")}
    hr = np.radians(h[bg_mask])
    return {"bgf_L": float(L[bg_mask].mean()), "bgf_a": float(a[bg_mask].mean()),
            "bgf_b": float(b[bg_mask].mean()), "bgf_C": float(C[bg_mask].mean()),
            "bgf_Lsd": float(L[bg_mask].std()),
            "bgf_hue": float(np.degrees(np.arctan2(np.sin(hr).mean(), np.cos(hr).mean())) % 360)}

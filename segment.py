"""
segment.py — leaf segmentation from an annotated box.

SAM ViT-B (box prompt) when `segment_anything` and a checkpoint are available,
otherwise GrabCut. Every leaf is handled in a padded crop resized so its longest
side is WORK_SIDE px, which makes all pixel settings proportional across images.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import cv2
from scipy import ndimage as ndi

WORK_SIDE = 512
PAD_FRAC = 0.10


def load_rgb(path: str | Path, ignore_exif: bool = False) -> np.ndarray:
    """Read an image as RGB uint8. Works with non-ASCII (e.g. Korean) paths."""
    data = np.fromfile(str(path), dtype=np.uint8)
    flags = cv2.IMREAD_COLOR | (cv2.IMREAD_IGNORE_ORIENTATION if ignore_exif else 0)
    bgr = cv2.imdecode(data, flags)
    if bgr is None:
        raise IOError(f"could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def crop_geometry(box, img_shape, pad=PAD_FRAC, work_side=WORK_SIDE):
    """Padded crop (x1, y1, x2, y2) in image coords and the resize scale."""
    x1, y1, x2, y2 = [float(v) for v in box]
    h, w = img_shape[:2]
    pw, ph = pad * (x2 - x1), pad * (y2 - y1)
    cx1, cy1 = int(max(0, np.floor(x1 - pw))), int(max(0, np.floor(y1 - ph)))
    cx2, cy2 = int(min(w, np.ceil(x2 + pw))), int(min(h, np.ceil(y2 + ph)))
    scale = work_side / max(cx2 - cx1, cy2 - cy1, 1)
    return (cx1, cy1, cx2, cy2, scale)


def to_working(arr: np.ndarray, geom, nearest: bool = False) -> np.ndarray:
    cx1, cy1, cx2, cy2, scale = geom
    crop = arr[cy1:cy2, cx1:cx2]
    size = (max(1, int(round((cx2 - cx1) * scale))), max(1, int(round((cy2 - cy1) * scale))))
    if nearest:
        interp = cv2.INTER_NEAREST
    else:
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(crop, size, interpolation=interp)


def box_in_working(box, geom):
    cx1, cy1, _, _, s = geom
    x1, y1, x2, y2 = box
    return ((x1 - cx1) * s, (y1 - cy1) * s, (x2 - cx1) * s, (y2 - cy1) * s)


def postprocess_mask(mask: np.ndarray, min_component_frac: float = 0.05) -> np.ndarray:
    """Fill holes (keeps necrotic patches SAM may drop) and remove small fragments."""
    m = ndi.binary_fill_holes(mask)
    lab, n = ndi.label(m)
    if n > 1:
        sizes = np.bincount(lab.ravel())[1:]
        keep = np.flatnonzero(sizes >= min_component_frac * sizes.sum()) + 1
        m = np.isin(lab, keep)
    return m


class Segmenter:
    def __init__(self, sam_checkpoint: str | None = None, model_type: str = "vit_b",
                 device: str | None = None):
        self.predictor = None
        if sam_checkpoint:
            try:
                import torch
                from segment_anything import sam_model_registry, SamPredictor
                dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
                sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(dev)
                self.predictor = SamPredictor(sam)
                print(f"[segment] SAM {model_type} on {dev}")
            except Exception as e:  # noqa: BLE001
                print(f"[segment] SAM unavailable ({e}); using GrabCut")
        else:
            print("[segment] no SAM checkpoint given; using GrabCut")

    @property
    def method(self) -> str:
        return "sam" if self.predictor is not None else "grabcut"

    def masks_for_image(self, image_rgb: np.ndarray, boxes: list) -> list[tuple]:
        """Return [(geom, working_mask_bool), ...] for each box of one image."""
        out = []
        if self.predictor is not None:
            self.predictor.set_image(image_rgb)
        for box in boxes:
            geom = crop_geometry(box, image_rgb.shape)
            if self.predictor is not None:
                masks, _, _ = self.predictor.predict(box=np.asarray(box, dtype=float),
                                                     multimask_output=False)
                m = to_working(masks[0].astype(np.uint8), geom, nearest=True) > 0
            else:
                m = self._grabcut(to_working(image_rgb, geom), box_in_working(box, geom))
            out.append((geom, postprocess_mask(m)))
        return out

    @staticmethod
    def _grabcut(crop_rgb: np.ndarray, wbox, iters: int = 5) -> np.ndarray:
        h, w = crop_rgb.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in wbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if crop_rgb.std() < 2:                      # blank/flat crop: GrabCut can stall on it
            m = np.zeros((h, w), bool)
            m[y1:y2, x1:x2] = True
            return m
        mask = np.zeros((h, w), np.uint8)
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR), mask,
                        (x1, y1, max(1, x2 - x1), max(1, y2 - y1)), bgd, fgd, iters,
                        cv2.GC_INIT_WITH_RECT)
            return np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD))
        except cv2.error:
            m = np.zeros((h, w), bool)
            m[y1:y2, x1:x2] = True
            return m


# ---------------------------------------------------------------- cache
def save_mask(cache_dir: Path, leaf_id: str, geom, mask: np.ndarray, method: str):
    np.savez_compressed(Path(cache_dir) / f"{leaf_id}.npz", mask=np.packbits(mask),
                        shape=np.array(mask.shape), geom=np.array(geom, dtype=float),
                        method=np.array(method))


def load_mask(cache_dir: Path, leaf_id: str):
    z = np.load(Path(cache_dir) / f"{leaf_id}.npz")
    shape = tuple(int(v) for v in z["shape"])
    mask = np.unpackbits(z["mask"])[: shape[0] * shape[1]].reshape(shape).astype(bool)
    g = z["geom"]
    geom = (int(g[0]), int(g[1]), int(g[2]), int(g[3]), float(g[4]))
    return geom, mask


# ---------------------------------------------------------------- mask refinement (applied at load time)
def _disk(r: int) -> np.ndarray:
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    return (xx * xx + yy * yy) <= r * r


def _bbox_iou(m: np.ndarray, box) -> float:
    ys, xs = np.nonzero(m)
    a = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    bx1, by1, bx2, by2 = box
    iw = max(0.0, min(a[2], bx2) - max(a[0], bx1))
    ih = max(0.0, min(a[3], by2) - max(a[1], by1))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def refine_mask(mask: np.ndarray, box_w, trim_frac: float = 0.04) -> np.ndarray:
    """Geometry-only clean-up of a cached SAM mask (no colour used):

    1. keep the connected part whose bounding box best matches the annotated box
       (drops a second leaf or a stray leaflet that SAM merged in);
    2. trim thin protrusions such as petioles: open the mask with a disk of radius
       trim_frac x equivalent diameter, keep its largest part as the leaf 'core', and keep
       original pixels within that radius of the core (so serrations and tips survive);
    3. fill holes.
    """
    if mask.sum() == 0:
        return mask
    lab, n = ndi.label(mask)
    if n > 1:
        best = max(range(1, n + 1), key=lambda i: (_bbox_iou(lab == i, box_w), (lab == i).sum()))
        mask = lab == best
    if trim_frac > 0:
        r = int(round(trim_frac * np.sqrt(4.0 * mask.sum() / np.pi)))
        if r >= 2:
            opened = ndi.binary_opening(mask, structure=_disk(r))
            lab, n = ndi.label(opened)
            if n >= 1:
                sizes = np.bincount(lab.ravel())[1:]
                core = lab == (np.argmax(sizes) + 1)
                near = ndi.distance_transform_edt(~core) <= r
                trimmed = mask & near
                lab2, n2 = ndi.label(trimmed)
                if n2 > 1:
                    sizes2 = np.bincount(lab2.ravel())[1:]
                    trimmed = lab2 == (np.argmax(sizes2) + 1)
                if trimmed.sum() > 0.5 * mask.sum():      # safety: never drop most of the leaf
                    mask = trimmed
    return ndi.binary_fill_holes(mask)


def recover_symptom_margin(mask: np.ndarray, L: np.ndarray, C: np.ndarray, h: np.ndarray, box_w,
                           frac: float = 0.15) -> np.ndarray:
    """Add brown or bleached leaf tissue that SAM left out (dried tips and margins).

    Candidate pixels lie inside the annotated box, within frac x equivalent diameter of the mask,
    and look like dead tissue: tan/brown (45 <= h < 75, C >= 10, 25 < L* < 85) or bleached
    (L* > 60, 6 <= C < 15, 60 <= h < 110). Only candidate patches that touch the mask are added.
    Excludes soil (dark), white plastic (C < 6) and green tissue (other leaves).
    """
    if mask.sum() == 0:
        return mask
    hgt, wid = mask.shape
    x1, y1, x2, y2 = box_w
    inbox = np.zeros_like(mask)
    inbox[max(0, int(y1)):min(hgt, int(np.ceil(y2))), max(0, int(x1)):min(wid, int(np.ceil(x2)))] = True
    r = frac * np.sqrt(4.0 * mask.sum() / np.pi)
    near = ndi.distance_transform_edt(~mask) <= r
    brown = (h >= 45) & (h < 75) & (C >= 10) & (L > 25) & (L < 85)
    bleached = (L > 60) & (C >= 6) & (C < 15) & (h >= 60) & (h < 110)
    cand = inbox & near & ~mask & (brown | bleached)
    cand = ndi.binary_opening(cand, structure=_disk(3))        # drop speckle and thin streaks
    if not cand.any():
        return mask
    lab, n = ndi.label(cand, structure=np.ones((3, 3), bool))
    touching = np.unique(lab[ndi.binary_dilation(mask, structure=np.ones((3, 3), bool)) & cand])
    touching = touching[touching > 0]
    if len(touching) == 0:
        return mask
    out = mask | np.isin(lab, touching)
    return ndi.binary_fill_holes(out)

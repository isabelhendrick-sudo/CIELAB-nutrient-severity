"""
make_synthetic_test_data.py — fake AI Hub 305-like data to smoke-test the pipeline end to end.

NOT for results. Draws trifoliate 'leaves' on a cluttered background; tip-burn plants get
brown necrosis at leaflet tips, yellowing plants get diffuse chlorotic patches; both grow
with time. Writes images + a leaf table in the format run_305.py expects.

  python make_synthetic_test_data.py --out synth
  python run_305.py all --leaf-table synth/leaves.csv --cache synth/cache --out synth/results --n-boot 200 --n-repeats 3
"""
import argparse
from pathlib import Path
import numpy as np
import cv2
import pandas as pd


def leaflet_masks(shape, centre, angle_deg, size):
    """Three elliptical leaflets around a centre (trifoliate). Returns list of (mask, tip_xy)."""
    out = []
    for k, da in enumerate((-50, 0, 50)):
        a = np.radians(angle_deg + da)
        L = size * (1.0 if k == 1 else 0.85)
        cx = centre[0] + 0.55 * L * np.cos(a)
        cy = centre[1] + 0.55 * L * np.sin(a)
        m = np.zeros(shape, np.uint8)
        cv2.ellipse(m, (int(cx), int(cy)), (int(0.55 * L), int(0.33 * L)), np.degrees(a), 0, 360, 1, -1)
        tip = (centre[0] + 1.05 * L * np.cos(a), centre[1] + 1.05 * L * np.sin(a))
        out.append((m.astype(bool), tip, a))
    return out


def draw_leaf(img, rng, centre, angle, size, kind, stage):
    h, w = img.shape[:2]
    lfs = leaflet_masks((h, w), centre, angle, size)
    leaf = np.zeros((h, w), bool)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.array([55, 115, 45], float) * rng.uniform(0.9, 1.1)
    for m, tip, a in lfs:
        leaf |= m
    img[leaf] = np.clip(base + rng.normal(0, 6, (leaf.sum(), 3)), 0, 255)
    # light veins: midrib + a few side veins per leaflet
    for m, tip, a in lfs:
        vm = np.zeros((h, w), np.uint8)
        cv2.line(vm, (int(centre[0]), int(centre[1])), (int(tip[0]), int(tip[1])), 1, 2)
        for t in np.linspace(0.3, 0.9, 4):
            px = centre[0] + t * (tip[0] - centre[0])
            py = centre[1] + t * (tip[1] - centre[1])
            for s in (-1, 1):
                b = a + s * np.radians(55)
                cv2.line(vm, (int(px), int(py)), (int(px + 0.25 * size * np.cos(b)), int(py + 0.25 * size * np.sin(b))), 1, 1)
        v = vm.astype(bool) & m
        img[v] = np.clip(img[v] * 1.25 + 10, 0, 255)
    symptomatic = stage > 0.08
    if kind == "tipburn" and stage > 0:
        for m, tip, a in lfs:
            d = np.hypot(xx - tip[0], yy - tip[1])
            nec = m & (d < (0.15 + 0.55 * stage) * size * 0.6)
            nec &= rng.random((h, w)) < 0.97
            img[nec] = np.clip(np.array([100, 58, 36]) + rng.normal(0, 8, (nec.sum(), 3)), 0, 255)
    if kind == "yellowing" and stage > 0:
        noise = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), size * 0.06)
        noise = (noise - noise.min()) / (np.ptp(noise) + 1e-9)
        chl = leaf & (noise < 0.15 + 0.6 * stage)
        img[chl] = np.clip(np.array([195, 185, 60]) + rng.normal(0, 8, (chl.sum(), 3)), 0, 255)
    ys, xs = np.nonzero(leaf)
    return (xs.min(), ys.min(), xs.max(), ys.max()), int(symptomatic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synth")
    ap.add_argument("--plants", type=int, default=12)
    ap.add_argument("--days", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for p in range(a.plants):
        kind = "tipburn" if p < round(a.plants * 0.58) else "yellowing"
        cause = (["칼슘부족", "EC 0.75", "EC 3.0"] if kind == "tipburn" else ["질소부족", "철결핍", "EC 0.75"])[p % 3]
        for d in range(a.days):
            day = 2 * d
            h, w = 700, 900
            img = np.clip(rng.normal(120, 18, (h, w, 3)) + np.array([10, 0, -15]), 0, 255)
            for _ in range(6):                       # clutter: other foliage + a white sticky trap
                cv2.circle(img, (int(rng.uniform(0, w)), int(rng.uniform(0, h))), int(rng.uniform(20, 60)),
                           (50, 105, 40), -1)
            cv2.rectangle(img, (20, 20), (90, 140), (235, 235, 230), -1)
            for j, (cx, cy) in enumerate([(230, 220), (620, 230), (430, 520)]):
                stage = min(1.0, max(0.0, (day / 10.0) * rng.uniform(0.6, 1.2) - (0.25 if j == 2 else 0)))
                box, sym = draw_leaf(img, rng, (cx, cy), rng.uniform(0, 360), 130, kind, stage)
                rows.append({"image_path": None, "plant_id": f"P{p:02d}", "class_name": kind,
                             "captured_at": pd.Timestamp("2023-05-01") + pd.Timedelta(days=day, hours=10),
                             "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3], "symptomatic": sym,
                             "cause_method": cause})
            cast = rng.uniform(0.92, 1.08, 3)       # small daily lighting change
            img = np.clip(img * cast, 0, 255).astype(np.uint8)
            path = out / "images" / f"P{p:02d}_d{day:02d}.jpg"
            cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
            for r in rows[-3:]:
                r["image_path"] = str(path)
    pd.DataFrame(rows).to_csv(out / "leaves.csv", index=False)
    print(f"{len(rows)} leaves, {a.plants} plants -> {out / 'leaves.csv'}")


if __name__ == "__main__":
    main()

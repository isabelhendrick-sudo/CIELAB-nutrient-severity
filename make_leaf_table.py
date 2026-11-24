"""
make_leaf_table.py — build leaves.csv from AI Hub 305 label JSONs (시설작물개체이미지 및 시설작물질병이미지).

Reads every label JSON under the given root folder(s), keeps strawberry LEAF boxes (category name 잎;
stem boxes 줄기 are dropped) for the tip burn (잎끝마름) and yellowing (황화) classes, finds each image
by its file name anywhere under the root(s), and writes one row per leaf box.

By default it keeps ONE image per plant per day (the one taken closest to --target-time), which
keeps lighting comparable across days and the run time manageable; use --per-day 0 for all images.
Images still inside zip files are extracted on the fly (only the ones selected) into --extract-dir.

  python make_leaf_table.py --root "C:\\...\\01-1.정식개방데이터\\Training" --out leaves.csv
  (add a second --root for the Validation folder if there is one; run_305.py makes its own plant split)

Also prints what each class was induced by (disease_cause_method), date ranges per class, and an
EXIF orientation check that tells you whether run_305.py needs --ignore-exif.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

CLASS_MAP = {"잎끝마름": "tipburn", "황화": "yellowing"}
LEAF_CATEGORY = "잎"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def scan(roots):
    """Index images by file name and list label JSONs under all roots."""
    images, jsons, zips = {}, [], []
    dup_names = 0
    for root in roots:
        for dirpath, _, files in os.walk(root):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                p = os.path.join(dirpath, f)
                if ext == ".json":
                    jsons.append(p)
                elif ext in IMG_EXT:
                    if f in images:
                        dup_names += 1
                    images[f] = p
                elif ext == ".zip":
                    zips.append(p)
    return images, jsons, zips, dup_names


def box_from_annotation(a):
    """Return (x1, y1, x2, y2) from 'bbox' [x, y, w, h], falling back to the 8-value 'coordinates'."""
    if a.get("bbox") and len(a["bbox"]) == 4:
        x, y, w, h = [float(v) for v in a["bbox"]]
        return x, y, x + w, y + h
    c = [float(v) for v in a.get("coordinates", [])]
    if len(c) >= 4:
        xs, ys = c[0::2], c[1::2]
        return min(xs), min(ys), max(xs), max(ys)
    return None


def exif_check(paths, json_sizes, n=200):
    """Compare stored image size and EXIF rotation with the JSON width/height on a sample."""
    try:
        from PIL import Image
    except ImportError:
        return "Pillow not installed; skipped EXIF check (look at the overlays instead)."
    rotated, raw_match, turned_match, other, checked = 0, 0, 0, 0, 0
    for p, (jw, jh) in list(zip(paths, json_sizes))[:n]:
        try:
            with Image.open(p) as im:
                w, h = im.size
                orient = im.getexif().get(0x0112, 1)
        except Exception:  # noqa: BLE001
            continue
        checked += 1
        turned = orient in (5, 6, 7, 8)
        rotated += turned
        dw, dh = (h, w) if turned else (w, h)          # size after applying EXIF rotation
        if (w, h) == (jw, jh) and (dw, dh) != (jw, jh):
            raw_match += 1
        elif (dw, dh) == (jw, jh):
            turned_match += 1
        else:
            other += 1
    if checked == 0:
        return "EXIF check: no images could be opened."
    msg = f"EXIF check on {checked} images: {rotated} carry a rotation tag. "
    if rotated == 0:
        msg += "No rotation tags, so --ignore-exif makes no difference."
    elif raw_match > turned_match:
        msg += f"Boxes match the UNROTATED image ({raw_match} cases): run run_305.py with --ignore-exif."
    else:
        msg += f"Boxes match the rotated image ({turned_match} cases): run run_305.py WITHOUT --ignore-exif."
    if other:
        msg += f" {other} images match neither; check them in the overlays."
    return msg


def _zip_member_name(info: zipfile.ZipInfo) -> str:
    """Real member name. Korean zips are often cp949 without the UTF-8 flag; Python then shows cp437 mojibake."""
    name = info.filename
    if not (info.flag_bits & 0x800):
        try:
            name = name.encode("cp437").decode("cp949")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return name


def index_zips(zips):
    """Map image base name -> (zip path, ZipInfo.filename) for every image inside the zips."""
    idx = {}
    for zp in zips:
        try:
            with zipfile.ZipFile(zp) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    base = _zip_member_name(info).replace("\\", "/").rsplit("/", 1)[-1]
                    if os.path.splitext(base)[1].lower() in IMG_EXT:
                        idx[base] = (zp, info.filename)
        except (zipfile.BadZipFile, OSError) as e:
            print(f"[zip] cannot read {zp}: {e}")
    return idx


def extract_selected(needed, zip_index, extract_dir: Path):
    """Extract only the needed images (base names) from the zips. Returns base name -> path."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    found, by_zip = {}, defaultdict(list)
    for base in needed:
        target = extract_dir / base
        if target.exists():
            found[base] = str(target)
        elif base in zip_index:
            zp, member = zip_index[base]
            by_zip[zp].append((base, member))
    total = sum(len(v) for v in by_zip.values())
    done = 0
    for zp, items in by_zip.items():
        with zipfile.ZipFile(zp) as zf:
            for base, member in items:
                target = extract_dir / base
                with zf.open(member) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                found[base] = str(target)
                done += 1
                if done % 200 == 0 or done == total:
                    print(f"[zip] extracted {done}/{total} images")
    return found


def select_per_day(recs, per_day: int, target_time: str):
    """Keep `per_day` images per plant per calendar day, closest to target_time (HH:MM)."""
    if per_day <= 0:
        return recs
    th, tm = [int(v) for v in target_time.split(":")]
    df = pd.DataFrame({"i": range(len(recs)), "plant": [r["plant_id"] for r in recs],
                       "t": pd.to_datetime([r["captured_at"] for r in recs], errors="coerce")})
    df = df.dropna(subset=["t"])
    df["date"] = df["t"].dt.date
    df["gap"] = (df["t"].dt.hour * 60 + df["t"].dt.minute - (th * 60 + tm)).abs()
    keep = df.sort_values(["plant", "date", "gap"]).groupby(["plant", "date"]).head(per_day)["i"]
    return [recs[i] for i in sorted(keep)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", required=True,
                    help="folder to search (images, labels and zips); may be given more than once")
    ap.add_argument("--out", default="leaves.csv")
    ap.add_argument("--crop", default="딸기", help="crop name to keep (images.crops)")
    ap.add_argument("--min-box", type=float, default=40.0,
                    help="drop leaf boxes whose shorter side is below this many pixels")
    ap.add_argument("--per-day", type=int, default=1,
                    help="images kept per plant per day (0 = keep all)")
    ap.add_argument("--target-time", default="12:00", help="preferred time of day when choosing images")
    ap.add_argument("--extract-dir", default="images_305",
                    help="where to extract selected images that are still inside zips")
    a = ap.parse_args()

    print("[scan] indexing folders (this can take a minute)...")
    images, jsons, zips, dup_names = scan(a.root)
    print(f"[scan] {len(jsons)} JSON files, {len(images)} image files, {len(zips)} zip files")
    if dup_names:
        print(f"[scan] warning: {dup_names} image file names appear more than once; the last one found is used")
    if not jsons:
        raise SystemExit("[scan] no label JSON files found: check --root, or unzip the 라벨링데이터 zips.")

    # ---- 1. read labels (image-level records with their leaf boxes)
    recs, stats = [], Counter()
    causes, all_classes = defaultdict(Counter), Counter()
    for k, jp in enumerate(jsons):
        if (k + 1) % 2000 == 0:
            print(f"[labels] read {k + 1}/{len(jsons)}")
        try:
            with open(jp, encoding="utf-8-sig") as fh:
                d = json.load(fh)
        except (json.JSONDecodeError, UnicodeDecodeError):
            stats["json_unreadable"] += 1
            continue
        im = d.get("images")
        if not isinstance(im, dict) or "annotations" not in d:
            stats["json_other_format"] += 1
            continue
        if im.get("crops") != a.crop:
            stats["json_other_crop"] += 1
            continue
        dclass = im.get("disease_class", "")
        all_classes[dclass] += 1
        if dclass not in CLASS_MAP:
            stats["json_other_class"] += 1
            continue
        causes[dclass][im.get("disease_cause_method", "")] += 1
        W, H = float(im.get("width", 0) or 0), float(im.get("height", 0) or 0)
        cat_names = {c["id"]: c.get("name", "") for c in d.get("categories", [])}
        boxes = []
        for ann in d["annotations"]:
            if cat_names.get(ann.get("category_id")) != LEAF_CATEGORY:
                stats["box_not_leaf"] += 1
                continue
            box = box_from_annotation(ann)
            if box is None:
                stats["box_unreadable"] += 1
                continue
            x1, y1, x2, y2 = box
            if W and H:
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
            if min(x2 - x1, y2 - y1) < a.min_box:
                stats["box_too_small"] += 1
                continue
            status = str(ann.get("disease_status", "")).strip().upper()
            if status not in ("Y", "N"):
                stats["box_status_unknown"] += 1
                continue
            boxes.append((ann.get("id"), x1, y1, x2, y2, int(status == "Y")))
        if boxes:
            recs.append({"fname": im.get("fname"), "image_id": im.get("image_id"), "plant_id": im.get("crops_id"),
                         "class_name": CLASS_MAP[dclass], "captured_at": im.get("date_captured"),
                         "kind_type": im.get("kind_type"), "farm_id": im.get("farm_id"),
                         "growth_stage": im.get("growth_stage"), "cause_method": im.get("disease_cause_method"),
                         "W": W, "H": H, "boxes": boxes})
    n_all = len(recs)

    # ---- 2. one image per plant per day (closest to the target time)
    recs = select_per_day(recs, a.per_day, a.target_time)
    print(f"[select] {n_all} labelled images -> {len(recs)} kept "
          f"({'all' if a.per_day <= 0 else f'{a.per_day} per plant per day, closest to {a.target_time}'})")

    # ---- 3. find the images (folders first, then inside zips)
    missing = [r["fname"] for r in recs if r["fname"] not in images]
    if missing and zips:
        print(f"[zip] {len(missing)} selected images not found as files; looking inside {len(zips)} zip files...")
        zip_index = index_zips(zips)
        print(f"[zip] {len(zip_index)} images listed inside the zips")
        images.update(extract_selected(missing, zip_index, Path(a.extract_dir).resolve()))

    rows, exif_paths, exif_sizes = [], [], []
    for r in recs:
        p = images.get(r["fname"])
        if p is None:
            stats["image_missing"] += 1
            continue
        if len(exif_paths) < 200:
            exif_paths.append(p)
            exif_sizes.append((int(r["W"]), int(r["H"])))
        for ann_id, x1, y1, x2, y2, sym in r["boxes"]:
            rows.append({"leaf_id": f"I{r['image_id']}_A{ann_id}", "image_path": p, "plant_id": r["plant_id"],
                         "class_name": r["class_name"], "captured_at": r["captured_at"],
                         "x1": round(x1, 2), "y1": round(y1, 2), "x2": round(x2, 2), "y2": round(y2, 2),
                         "symptomatic": sym, "kind_type": r["kind_type"], "farm_id": r["farm_id"],
                         "growth_stage": r["growth_stage"], "cause_method": r["cause_method"],
                         "img_width": r["W"], "img_height": r["H"]})

    if not rows:
        raise SystemExit(f"[table] no leaf boxes kept. Counts: {dict(stats)}. Classes seen: {dict(all_classes)}")
    df = pd.DataFrame(rows)
    if df["leaf_id"].duplicated().any():                    # image_id not unique: fall back to a counter
        df["leaf_id"] = [f"{lid}_{i}" for i, lid in enumerate(df["leaf_id"])]
    df.to_csv(a.out, index=False, encoding="utf-8")

    # ---- report
    print(f"\n[table] wrote {len(df)} leaf boxes from {df['image_path'].nunique()} images -> {Path(a.out).resolve()}")
    print(f"[table] skipped: {dict(stats)}")
    if stats["image_missing"]:
        print("[table] some selected images were found neither as files nor inside the zips: "
              "check that all 원천데이터 zips are under --root.")
    print(f"[table] strawberry classes seen in the labels (all images): {dict(all_classes)}")
    print("\n[classes] induced cause (disease_cause_method) per class, all labelled images:")
    for k, c in causes.items():
        print(f"  {k} ({CLASS_MAP[k]}): {dict(c)}")
    df["captured_at"] = pd.to_datetime(df["captured_at"], errors="coerce")
    summ = df.groupby("class_name").agg(
        plants=("plant_id", "nunique"), images=("image_path", "nunique"), leaf_boxes=("leaf_id", "size"),
        Y_boxes=("symptomatic", "sum"), first_date=("captured_at", "min"), last_date=("captured_at", "max"))
    print("\n[summary]")
    print(summ.to_string())
    per_plant = df.groupby("plant_id")["class_name"].nunique()
    if (per_plant > 1).any():
        print(f"\n[warning] {int((per_plant > 1).sum())} plants appear under both classes; check plant ids.")
    if df["captured_at"].isna().any():
        print(f"[warning] {int(df['captured_at'].isna().sum())} rows have no readable capture date.")
    print("\n" + exif_check(exif_paths, exif_sizes))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Split COCO dataset (train/valid/test) into per-category folders based on _annotations.coco.json.

Usage examples:
  python split_by_category.py --root /path/to/dataset_root
  python split_by_category.py --root /path/to/dataset_root --mode symlink
  python split_by_category.py --root /path/to/dataset_root --exclude leaf,background
  python split_by_category.py --root /path/to/dataset_root --keep-unlabeled

The script expects each of {train, valid, test} to contain images and a _annotations.coco.json.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
import re

def slugify(name: str) -> str:
    # Normalize category folder names: replace spaces and non-word chars with underscores
    name = name.strip().lower()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^\w\-]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "unknown"

def load_coco(coco_json_path: Path):
    with coco_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Build maps
    cat_id_to_name = {c["id"]: c["name"] for c in data.get("categories", [])}
    img_id_to_fname = {img["id"]: img["file_name"] for img in data.get("images", [])}

    # image_id -> set of category names present
    img_id_to_cats = {img_id: set() for img_id in img_id_to_fname.keys()}
    for ann in data.get("annotations", []):
        img_id = ann.get("image_id")
        cat_id = ann.get("category_id")
        if img_id in img_id_to_cats and cat_id in cat_id_to_name:
            img_id_to_cats[img_id].add(cat_id_to_name[cat_id])

    return cat_id_to_name, img_id_to_fname, img_id_to_cats

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def place_file(src: Path, dst: Path, mode: str):
    if mode == "copy":
        if not dst.exists():
            shutil.copy2(src, dst)
    elif mode == "symlink":
        if not dst.exists():
            try:
                dst.symlink_to(src.resolve())
            except Exception:
                # Fallback to copy if symlink not permitted
                shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def process_split_folder(split_dir: Path, mode: str, exclude_cats: set, keep_unlabeled: bool, out_suffix: str):
    coco_json = split_dir / "_annotations.coco.json"
    if not coco_json.exists():
        print(f"[WARN] No _annotations.coco.json in {split_dir}, skipping")
        return

    cat_id_to_name, img_id_to_fname, img_id_to_cats = load_coco(coco_json)

    # Prepare output base: split_by_category/
    out_base = split_dir / out_suffix
    ensure_dir(out_base)

    # Build set of excluded (normalized)
    exclude_norm = {slugify(x) for x in exclude_cats}

    # Map normalized name -> original display name
    allowed_cat_names = {}
    for cat_name in set(cat_id_to_name.values()):
        norm = slugify(cat_name)
        if norm not in exclude_norm:
            allowed_cat_names[norm] = cat_name

    # Create per-category folders
    for norm, disp in allowed_cat_names.items():
        ensure_dir(out_base / norm)

    # Optional unlabeled folder
    unlabeled_dir = out_base / "_unlabeled"
    if keep_unlabeled:
        ensure_dir(unlabeled_dir)

    # Copy/symlink images into their category folders
    n_total, n_placed, n_unlabeled = 0, 0, 0
    for img_id, fname in img_id_to_fname.items():
        n_total += 1
        src = split_dir / fname
        if not src.exists():
            # Some exports place images in an "images/" subfolder; try fallback
            alt = split_dir / "images" / fname
            if alt.exists():
                src = alt
            else:
                print(f"[WARN] Missing image file: {src}")
                continue

        cats = img_id_to_cats.get(img_id, set())
        # Filter out excluded categories
        cats = {c for c in cats if slugify(c) not in exclude_norm}

        if cats:
            for cat in cats:
                norm = slugify(cat)
                dst = out_base / norm / src.name
                place_file(src, dst, mode)
                n_placed += 1
        else:
            if keep_unlabeled:
                dst = unlabeled_dir / src.name
                place_file(src, dst, mode)
                n_unlabeled += 1
            # else: skip

    print(f"[DONE] {split_dir}: images={n_total}, placed={n_placed}, unlabeled={n_unlabeled}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True,
                    help="Root folder that contains dataset(s). Each dataset should have train/, valid/, test/ with _annotations.coco.json and images.")
    ap.add_argument("--splits", type=str, default="train,valid,test",
                    help="Comma-separated list of split folder names to process (default: train,valid,test)")
    ap.add_argument("--mode", type=str, default="copy", choices=["copy", "symlink"],
                    help="Place images by copying or creating symlinks (default: copy)")
    ap.add_argument("--exclude", type=str, default="leaf",
                    help="Comma-separated category names to exclude (default: leaf)")
    ap.add_argument("--keep-unlabeled", action="store_true",
                    help="If set, images with no annotations are placed into split_by_category/_unlabeled/")
    ap.add_argument("--out-suffix", type=str, default="split_by_category",
                    help="Name of the output folder created inside each split (default: split_by_category)")
    args = ap.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    exclude_cats = {s.strip() for s in args.exclude.split(",") if s.strip()}

    root = args.root
    if not root.exists():
        print(f"[ERROR] root not found: {root}")
        sys.exit(1)

    # If root directly contains train/valid/test, process it.
    # Otherwise, iterate subdirectories of root and process each dataset found.
    candidate_datasets = []
    if all((root / s).exists() for s in splits):
        candidate_datasets.append(root)
    else:
        for ds in sorted([p for p in root.iterdir() if p.is_dir()]):
            if all((ds / s).exists() for s in splits):
                candidate_datasets.append(ds)

    if not candidate_datasets:
        print("[ERROR] No dataset with the given splits found under root.")
        sys.exit(2)

    for ds in candidate_datasets:
        print(f"[DATASET] {ds}")
        for split in splits:
            split_dir = ds / split
            process_split_folder(
                split_dir=split_dir,
                mode=args.mode,
                exclude_cats=exclude_cats,
                keep_unlabeled=args.keep_unlabeled,
                out_suffix=args.out_suffix
            )

if __name__ == "__main__":
    main()
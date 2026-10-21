"""
Check a YOLO dataset (train/valid/test, each with images/ and labels/) for
data leakage and basic problems.

What it checks:
  1. Images and labels per split, missing/empty labels, boxes per class.
  2. Roboflow copies: files exported as "<name>_png.rf.<hash>.jpg" share the
     part before ".rf.". The same original name in two splits = leakage.
  3. Near-duplicate images across splits, using a perceptual hash (catches
     copies that were renamed, resized, brightened, slightly blurred, etc.).

Usage:
  pip install pillow imagehash pyyaml
  python check_leakage.py path/to/dataset
  python check_leakage.py path/to/dataset --threshold 8   # looser matching

Writes leak_pairs.csv in the dataset folder with every suspicious pair.
"""

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    import imagehash
    from PIL import Image
except ImportError:
    sys.exit("Install the requirements first:  pip install pillow imagehash pyyaml")

SPLITS = ["train", "valid", "test"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_class_names(root: Path):
    yaml_path = root / "data.yaml"
    if not yaml_path.exists():
        return None
    try:
        import yaml
        names = yaml.safe_load(yaml_path.read_text()).get("names")
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]
        return names
    except Exception:
        return None


def original_name(filename: str) -> str:
    """Roboflow export name -> original upload name (e.g. '1594_png')."""
    stem = Path(filename).stem
    return re.split(r"\.rf\.", stem)[0]


def hash_image(path: Path):
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            return imagehash.phash(im), imagehash.dhash(im)
    except Exception as e:
        print(f"  ! could not read {path.name}: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path, help="folder containing train/ valid/ test/")
    ap.add_argument("--threshold", type=int, default=6,
                    help="max hash difference to count as near-duplicate (0-64, default 6)")
    args = ap.parse_args()
    root = args.dataset
    names = load_class_names(root)

    # ---------- 1. Basic counts ----------
    print("\n=== 1. Split summary ===")
    images = {}
    for split in SPLITS:
        img_dir, lbl_dir = root / split / "images", root / split / "labels"
        if not img_dir.exists():
            print(f"{split:5}: (folder missing)")
            images[split] = []
            continue
        imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
        images[split] = imgs
        missing, empty, class_counts = 0, 0, Counter()
        for img in imgs:
            lbl = lbl_dir / (img.stem + ".txt")
            if not lbl.exists():
                missing += 1
                continue
            lines = [l for l in lbl.read_text().splitlines() if l.strip()]
            if not lines:
                empty += 1
            for l in lines:
                class_counts[int(float(l.split()[0]))] += 1
        cls = ", ".join(
            f"{names[c] if names and c < len(names) else c}: {n}"
            for c, n in sorted(class_counts.items())
        )
        print(f"{split:5}: {len(imgs):4} images | missing labels: {missing} | "
              f"empty labels: {empty} | boxes -> {cls or 'none'}")

    # ---------- 2. Roboflow original-name check ----------
    print("\n=== 2. Same original image in more than one split ===")
    origin_splits = defaultdict(lambda: defaultdict(list))
    for split, imgs in images.items():
        for img in imgs:
            origin_splits[original_name(img.name)][split].append(img.name)
    name_leaks = {o: s for o, s in origin_splits.items() if len(s) > 1}
    aug_counts = Counter(len(s.get("train", [])) for s in origin_splits.values())
    if name_leaks:
        print(f"LEAK: {len(name_leaks)} original images appear in more than one split")
        for o, s in list(name_leaks.items())[:15]:
            print(f"  {o}: " + " | ".join(f"{k}={len(v)}" for k, v in s.items()))
        if len(name_leaks) > 15:
            print(f"  ... and {len(name_leaks) - 15} more (see leak_pairs.csv)")
    else:
        print("OK: no original filename shared across splits")
    print("Copies per original in train:",
          ", ".join(f"{k}x: {v}" for k, v in sorted(aug_counts.items()) if k))

    # ---------- 3. Near-duplicate images ----------
    print(f"\n=== 3. Near-duplicate images across splits (threshold {args.threshold}) ===")
    print("Hashing images...")
    hashes = {}
    for split, imgs in images.items():
        for img in imgs:
            h = hash_image(img)
            if h:
                hashes[(split, img.name)] = h

    pairs = []
    keys = list(hashes)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if a[0] == b[0]:
                continue  # only compare across splits
            pa, da = hashes[a]
            pb, db = hashes[b]
            dist = min(pa - pb, da - db)
            if dist <= args.threshold:
                pairs.append((a[0], a[1], b[0], b[1], dist))

    eval_imgs = {k for k in keys if k[0] in ("valid", "test")}
    leaked_eval = {(s, n) for s1, n1, s2, n2, _ in pairs
                   for s, n in ((s1, n1), (s2, n2)) if s in ("valid", "test")}

    if pairs:
        by_split = Counter(f"{min(p[0], p[2])}<->{max(p[0], p[2])}" for p in pairs)
        print(f"LEAK: {len(pairs)} near-duplicate pairs across splits "
              f"({', '.join(f'{k}: {v}' for k, v in by_split.items())})")
        print(f"{len(leaked_eval)} of {len(eval_imgs)} valid/test images "
              f"({100 * len(leaked_eval) / max(len(eval_imgs), 1):.0f}%) have a near-copy in another split")
        for p in sorted(pairs, key=lambda x: x[4])[:10]:
            print(f"  dist {p[4]:2}: {p[0]}/{p[1]}  <->  {p[2]}/{p[3]}")
    else:
        print("OK: no near-duplicates found across splits")

    out = root / "leak_pairs.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type", "split_a", "image_a", "split_b", "image_b", "hash_distance"])
        for o, s in name_leaks.items():
            w.writerow(["same_original_name", ",".join(s), o, "", "", ""])
        for p in sorted(pairs, key=lambda x: x[4]):
            w.writerow(["near_duplicate", *p])
    print(f"\nDetails saved to {out}")
    print("Tip: open a few pairs to confirm. If many valid/test images have copies in train,\n"
          "your scores are inflated: remove duplicates, re-split, then augment train only.")


if __name__ == "__main__":
    main()

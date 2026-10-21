"""
Re-split a YOLO dataset so that near-duplicate images never cross splits.

Images are grouped with perceptual hashes (pHash and dHash): any two images whose
hashes differ by <= THRESHOLD bits are put in the same group (transitively), and
whole groups are assigned to train/valid/test. Among many random assignments, the
one whose split sizes and per-class box counts are closest to the target ratios
is kept. Originals are not modified; files are copied to OUT_DIR.

Usage:
  pip install pillow imagehash
  python make_clean_split.py path/to/dataset path/to/output --threshold 8
"""

import argparse
import csv
import random
import shutil
from collections import Counter
from pathlib import Path

import imagehash
from PIL import Image

SPLITS = ["train", "valid", "test"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--threshold", type=int, default=8)
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.7, 0.2, 0.1])
    ap.add_argument("--tries", type=int, default=300)
    args = ap.parse_args()
    target = dict(zip(SPLITS, args.ratios))

    # Collect images, labels and hashes
    items = []
    for split in SPLITS:
        img_dir = args.dataset / split / "images"
        if not img_dir.exists():
            continue
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() not in IMG_EXT:
                continue
            lbl = args.dataset / split / "labels" / (p.stem + ".txt")
            classes = Counter()
            if lbl.exists():
                for line in lbl.read_text().splitlines():
                    if line.strip():
                        classes[int(float(line.split()[0]))] += 1
            with Image.open(p) as im:
                im = im.convert("RGB")
                items.append(dict(img=p, lbl=lbl, cls=classes,
                                  ph=imagehash.phash(im), dh=imagehash.dhash(im)))
    n = len(items)
    print(f"{n} images hashed")

    # Group near-duplicates (union-find)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in range(n):
        for b in range(a + 1, n):
            d = min(items[a]["ph"] - items[b]["ph"], items[a]["dh"] - items[b]["dh"])
            if d <= args.threshold:
                parent[find(a)] = find(b)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    groups = list(groups.values())
    print(f"{len(groups)} groups (largest: {max(map(len, groups))} images)")

    # Assign whole groups to splits; keep the most balanced assignment
    total = sum((it["cls"] for it in items), Counter())
    best = None
    for seed in range(args.tries):
        rng = random.Random(seed)
        gs = groups[:]
        rng.shuffle(gs)
        gs.sort(key=len, reverse=True)
        assign = {s: [] for s in SPLITS}
        for g in gs:
            s = min(SPLITS, key=lambda s: (len(assign[s]) + len(g)) / (target[s] * n))
            assign[s] += g
        err = 0.0
        for s in SPLITS:
            cc = sum((items[i]["cls"] for i in assign[s]), Counter())
            err += abs(len(assign[s]) / n - target[s]) * 5
            err += sum(abs(cc[c] / total[c] - target[s]) for c in total)
        if best is None or err < best[0]:
            best = (err, assign)

    # Copy files and write a report
    group_of = {i: k for k, g in enumerate(groups) for i in g}
    rows = []
    for s, idx in best[1].items():
        (args.out / s / "images").mkdir(parents=True, exist_ok=True)
        (args.out / s / "labels").mkdir(parents=True, exist_ok=True)
        for i in idx:
            it = items[i]
            shutil.copy2(it["img"], args.out / s / "images" / it["img"].name)
            if it["lbl"].exists():
                shutil.copy2(it["lbl"], args.out / s / "labels" / it["lbl"].name)
            rows.append([it["img"].name, it["img"].parent.parent.name, s, group_of[i]])
        cc = sum((items[i]["cls"] for i in idx), Counter())
        print(f"{s:5}: {len(idx)} images, boxes per class {dict(sorted(cc.items()))}")

    with open(args.out / "split_report.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "old_split", "new_split", "group_id"])
        w.writerows(sorted(rows, key=lambda r: (r[3], r[0])))
    src_yaml = args.dataset / "data.yaml"
    if src_yaml.exists():
        lines = [l for l in src_yaml.read_text().splitlines()
                 if not l.startswith(("train:", "val:", "test:", "path:"))]
        (args.out / "data.yaml").write_text(
            "train: train/images\nval: valid/images\ntest: test/images\n" + "\n".join(lines) + "\n")
    print(f"Written to {args.out}")


if __name__ == "__main__":
    main()

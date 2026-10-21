# Re-split a YOLO dataset so that copies of the same image stay in the same split.
# Usage: python make_clean_split.py path/to/dataset path/to/output
# Needs: pip install pillow imagehash

import random
import shutil
import sys
from pathlib import Path

import imagehash
from PIL import Image

DATASET = Path(sys.argv[1])
OUTPUT = Path(sys.argv[2])
SPLITS = ["train", "valid", "test"]
RATIOS = {"train": 0.7, "valid": 0.2, "test": 0.1}
THRESHOLD = 8  # max hash difference to count two images as copies
SEED = 0


def get_hash(path):
    img = Image.open(path).convert("RGB")
    return imagehash.phash(img), imagehash.dhash(img)


def is_copy(h1, h2):
    return min(h1[0] - h2[0], h1[1] - h2[1]) <= THRESHOLD


# 1. Collect all images from all splits
images = []
for split in SPLITS:
    for path in sorted((DATASET / split / "images").glob("*")):
        label = DATASET / split / "labels" / (path.stem + ".txt")
        images.append({"image": path, "label": label, "hash": get_hash(path)})
print(f"{len(images)} images found")

# 2. Group copies together: each image starts in its own group,
#    and two groups are merged whenever they contain a copy of each other
group_of = list(range(len(images)))
for i in range(len(images)):
    for j in range(i + 1, len(images)):
        if is_copy(images[i]["hash"], images[j]["hash"]):
            old, new = group_of[j], group_of[i]
            group_of = [new if g == old else g for g in group_of]

groups = {}
for i, g in enumerate(group_of):
    groups.setdefault(g, []).append(images[i])
groups = list(groups.values())
print(f"{len(groups)} groups of unique images")

# 3. Shuffle the groups and fill train, then valid, then test
random.seed(SEED)
random.shuffle(groups)
total = len(images)
assigned = {split: [] for split in SPLITS}
for group in groups:
    if len(assigned["train"]) < RATIOS["train"] * total:
        assigned["train"].extend(group)
    elif len(assigned["valid"]) < RATIOS["valid"] * total:
        assigned["valid"].extend(group)
    else:
        assigned["test"].extend(group)

# 4. Copy images and labels to the output folder
for split in SPLITS:
    (OUTPUT / split / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT / split / "labels").mkdir(parents=True, exist_ok=True)
    for item in assigned[split]:
        shutil.copy(item["image"], OUTPUT / split / "images" / item["image"].name)
        if item["label"].exists():
            shutil.copy(item["label"], OUTPUT / split / "labels" / item["label"].name)
    print(f"{split}: {len(assigned[split])} images")

# 5. Copy data.yaml with paths pointing to the new folders
names = [line for line in (DATASET / "data.yaml").read_text().splitlines()
         if line.startswith(("nc:", "names:"))]
(OUTPUT / "data.yaml").write_text(
    "train: train/images\nval: valid/images\ntest: test/images\n\n" + "\n".join(names) + "\n")
print(f"Done. New dataset in {OUTPUT}")

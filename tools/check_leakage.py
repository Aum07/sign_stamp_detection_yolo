# Find images that appear (as near-copies) in more than one split.
# Usage: python check_leakage.py path/to/dataset
# Needs: pip install pillow imagehash

import csv
import sys
from pathlib import Path

import imagehash
from PIL import Image

DATASET = Path(sys.argv[1])
SPLITS = ["train", "valid", "test"]
THRESHOLD = 8  # max hash difference to count two images as copies


def get_hash(path):
    img = Image.open(path).convert("RGB")
    return imagehash.phash(img), imagehash.dhash(img)


def is_copy(h1, h2):
    return min(h1[0] - h2[0], h1[1] - h2[1]) <= THRESHOLD


# Hash every image
images = []
for split in SPLITS:
    for path in sorted((DATASET / split / "images").glob("*")):
        images.append((split, path.name, get_hash(path)))
    count = sum(1 for img in images if img[0] == split)
    print(f"{split}: {count} images")

# Compare every pair of images from different splits
leaks = []
for i in range(len(images)):
    for j in range(i + 1, len(images)):
        split1, name1, hash1 = images[i]
        split2, name2, hash2 = images[j]
        if split1 != split2 and is_copy(hash1, hash2):
            leaks.append((split1, name1, split2, name2))

print(f"\nNear-duplicate pairs across splits: {len(leaks)}")
for leak in leaks[:10]:
    print(f"  {leak[0]}/{leak[1]}  <->  {leak[2]}/{leak[3]}")

with open(DATASET / "leak_pairs.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["split_a", "image_a", "split_b", "image_b"])
    writer.writerows(leaks)
print(f"All pairs saved to {DATASET / 'leak_pairs.csv'}")

import argparse
import random
import re
import shutil
from pathlib import Path


TRAIN_RATIO = 0.80
SEED = 42


def brand(name):
    return re.split(r"[_ ]", name, maxsplit=1)[0].strip()


def copy_or_move(src, dst, copy):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if copy:
        shutil.copytree(src, dst) if src.is_dir() else shutil.copy2(src, dst)
    else:
        shutil.move(str(src), dst)


def group_by_brand(src_dir, dst_dir=None, copy=True):
    src = Path(src_dir)
    dst = Path(dst_dir or src_dir)
    train_dir, test_dir = dst / "train", dst / "test"
    items = [p for p in src.iterdir() if not p.name.startswith(".") and p.resolve() != dst.resolve()]
    by_brand = {}

    for item in items:
        by_brand.setdefault(brand(item.name), []).append(item)

    rng = random.Random(SEED)
    for brand_name, brand_items in by_brand.items():
        rng.shuffle(brand_items)
        split = int(len(brand_items) * TRAIN_RATIO)
        if len(brand_items) > 1:
            split = min(max(1, split), len(brand_items) - 1)
        for subset, subset_items in [(train_dir, brand_items[:split]), (test_dir, brand_items[split:])]:
            for item in subset_items:
                copy_or_move(item, subset / brand_name / item.name, copy)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Group car images/folders by brand into train/test.")
    p.add_argument("src")
    p.add_argument("--dst")
    p.add_argument("--move", action="store_true")
    a = p.parse_args()
    group_by_brand(a.src, a.dst, copy=not a.move)

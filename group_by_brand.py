"""
KOUKOSIAS ATHANASIOS-UTH-2026

group_by_brand.py
-----------------
Reorganise a directory of car images (or model sub-folders) so that every
item is grouped under its brand.

Expected source naming convention (files OR sub-folders):
    Brand_Model[_Year[_extra]].ext
    Brand Model[_Year[_extra]].ext   (space as first separator also accepted)

Examples
    Honda_civic_2015/        →  Honda/Honda_civic_2015/
    Toyota HIACE 2000/       →  Toyota/Toyota HIACE 2000/
    BMW_3Series_2021_001.jpg →  BMW/BMW_3Series_2021_001.jpg

Usage
    python scripts/group_by_brand.py <input_dir> [--copy]

    --copy   copy items instead of moving them (original directory untouched)
"""

import argparse
import re
import shutil
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_brand(name: str) -> str:
    """Return the brand token (everything before the first '_' or space)."""
    # Split on first underscore or space, whichever comes first
    match = re.split(r"[_ ]", name, maxsplit=1)
    return match[0].strip()


def collect_items(src: Path) -> list[Path]:
    """Return direct children of *src* (files + directories, non-hidden)."""
    return [p for p in src.iterdir() if not p.name.startswith(".")]


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def group_by_brand(src_dir: Path, copy: bool = False) -> None:
    src_dir = src_dir.resolve()
    if not src_dir.is_dir():
        sys.exit(f"[ERROR] '{src_dir}' is not a valid directory.")

    items = collect_items(src_dir)
    if not items:
        print(f"[INFO] No items found in '{src_dir}'. Nothing to do.")
        return

    move_fn = shutil.copytree if copy else shutil.move  # for dirs
    op_name = "Copying" if copy else "Moving"

    moved, skipped = 0, 0

    for item in sorted(items):
        brand = extract_brand(item.name)

        if not brand:
            print(f"[SKIP] Cannot determine brand for: {item.name}")
            skipped += 1
            continue

        brand_dir = src_dir / brand

        # Don't try to move a brand folder into itself
        if item == brand_dir:
            skipped += 1
            continue

        brand_dir.mkdir(exist_ok=True)
        dest = brand_dir / item.name

        if dest.exists():
            print(f"[SKIP] Destination already exists, skipping: {dest}")
            skipped += 1
            continue

        print(f"{op_name}: {item.name}  →  {brand}/{item.name}")

        if item.is_dir():
            if copy:
                shutil.copytree(item, dest)
            else:
                shutil.move(str(item), dest)
        else:
            if copy:
                shutil.copy2(item, dest)
            else:
                shutil.move(str(item), dest)

        moved += 1

    print(f"\nDone. {op_name.lower()[:-3]}d {moved} item(s), skipped {skipped}.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Group car images/folders by brand (first name token)."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory whose contents should be grouped by brand.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        default=False,
        help="Copy items instead of moving them (default: move).",
    )
    args = parser.parse_args()
    group_by_brand(args.input_dir, copy=args.copy)


if __name__ == "__main__":
    main()

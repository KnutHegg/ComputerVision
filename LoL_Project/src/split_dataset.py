import argparse
import random
import shutil
from pathlib import Path

from common import CLASS_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create train/val/test split for YOLO dataset.")
    parser.add_argument("--images-dir", type=Path, required=True, help="Source images directory.")
    parser.add_argument("--labels-dir", type=Path, required=True, help="Source labels directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output split dataset directory.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def ensure_structure(base: Path) -> None:
    for split in ("train", "val", "test"):
        (base / "images" / split).mkdir(parents=True, exist_ok=True)
        (base / "labels" / split).mkdir(parents=True, exist_ok=True)


def copy_pair(name: str, images_dir: Path, labels_dir: Path, out_dir: Path, split: str) -> None:
    src_image = images_dir / name
    src_label = labels_dir / f"{Path(name).stem}.txt"

    if not src_label.exists():
        return

    dst_image = out_dir / "images" / split / name
    dst_label = out_dir / "labels" / split / src_label.name
    shutil.copy2(src_image, dst_image)
    shutil.copy2(src_label, dst_label)


def write_dataset_yaml(output_dir: Path, class_names: list[str]) -> None:
    yaml_path = output_dir / "dataset.yaml"
    lines = [
        "path: datasets/minimap_dataset",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {len(class_names)}",
        f"names: {class_names}",
        "",
    ]
    yaml_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    if args.train_ratio <= 0 or args.val_ratio <= 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("Use ratios so that 0 < train_ratio, val_ratio and train_ratio + val_ratio < 1.")

    images = sorted(
        p.name
        for p in args.images_dir.glob("*.jpg")
        if (args.labels_dir / f"{p.stem}.txt").exists()
    )
    if not images:
        raise RuntimeError("No image/label pairs found.")

    random.shuffle(images)

    n = len(images)
    train_end = int(n * args.train_ratio)
    val_end = train_end + int(n * args.val_ratio)

    train_files = images[:train_end]
    val_files = images[train_end:val_end]
    test_files = images[val_end:]

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    ensure_structure(args.output_dir)

    for name in train_files:
        copy_pair(name, args.images_dir, args.labels_dir, args.output_dir, "train")
    for name in val_files:
        copy_pair(name, args.images_dir, args.labels_dir, args.output_dir, "val")
    for name in test_files:
        copy_pair(name, args.images_dir, args.labels_dir, args.output_dir, "test")

    write_dataset_yaml(args.output_dir, CLASS_NAMES)

    print(f"Total: {n}")
    print(f"Train: {len(train_files)}")
    print(f"Val: {len(val_files)}")
    print(f"Test: {len(test_files)}")
    print(f"dataset.yaml: {args.output_dir / 'dataset.yaml'}")


if __name__ == "__main__":
    main()

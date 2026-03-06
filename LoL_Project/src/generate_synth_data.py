import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic minimap dataset in YOLO format.")
    parser.add_argument("--backgrounds-dir", type=Path, required=True)
    parser.add_argument("--icons-dir", type=Path, required=True)
    parser.add_argument("--coco-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--canvas-size", type=int, default=300)
    parser.add_argument("--min-icons", type=int, default=10)
    parser.add_argument("--max-icons", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_coco(coco_path: Path) -> dict:
    with coco_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_icon_pool(coco: dict, icons_dir: Path) -> tuple[list[dict], list[str]]:
    image_map = {img["id"]: img["file_name"] for img in coco["images"]}
    categories = sorted(coco["categories"], key=lambda c: c["id"])
    category_to_index = {cat["id"]: idx for idx, cat in enumerate(categories)}
    class_names = [cat["name"] for cat in categories]

    pool = []
    for ann in coco["annotations"]:
        file_name = image_map.get(ann["image_id"])
        if not file_name:
            continue

        image_path = icons_dir / file_name
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        x, y, w, h = ann["bbox"]
        x1, y1 = int(max(0, x)), int(max(0, y))
        x2, y2 = int(min(image.shape[1], x + w)), int(min(image.shape[0], y + h))
        if x2 <= x1 or y2 <= y1:
            continue

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        cls_idx = category_to_index[ann["category_id"]]
        pool.append({"icon": crop, "class_id": cls_idx})

    if not pool:
        raise RuntimeError("No valid icon crops were found in COCO annotations.")
    return pool, class_names


def non_black_mask(icon: np.ndarray) -> np.ndarray:
    # Treat near-black pixels as transparent to reduce icon square artifacts.
    gray = cv2.cvtColor(icon, cv2.COLOR_BGR2GRAY)
    return gray > 15


def paste_icon(canvas: np.ndarray, icon: np.ndarray, x: int, y: int) -> None:
    h, w = icon.shape[:2]
    roi = canvas[y:y + h, x:x + w]
    mask = non_black_mask(icon)
    roi[mask] = icon[mask]
    canvas[y:y + h, x:x + w] = roi


def yolo_box(x: int, y: int, w: int, h: int, size: int) -> tuple[float, float, float, float]:
    cx = (x + w / 2) / size
    cy = (y + h / 2) / size
    nw = w / size
    nh = h / size
    return cx, cy, nw, nh


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    backgrounds = sorted(args.backgrounds_dir.glob("*"))
    backgrounds = [p for p in backgrounds if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not backgrounds:
        raise RuntimeError(f"No backgrounds found in: {args.backgrounds_dir}")

    coco = load_coco(args.coco_json)
    icon_pool, class_names = build_icon_pool(coco, args.icons_dir)

    image_out = args.output_dir / "images"
    label_out = args.output_dir / "labels"
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    for i in range(1, args.num_samples + 1):
        bg_path = random.choice(backgrounds)
        bg = cv2.imread(str(bg_path))
        if bg is None:
            continue

        canvas = cv2.resize(bg, (args.canvas_size, args.canvas_size), interpolation=cv2.INTER_AREA)
        label_lines = []
        n_icons = random.randint(args.min_icons, args.max_icons)

        for _ in range(n_icons):
            icon_item = random.choice(icon_pool)
            icon = icon_item["icon"]
            class_id = icon_item["class_id"]

            target_size = random.randint(14, 26)
            resized = cv2.resize(icon, (target_size, target_size), interpolation=cv2.INTER_AREA)

            max_x = args.canvas_size - target_size
            max_y = args.canvas_size - target_size
            if max_x <= 0 or max_y <= 0:
                continue

            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            paste_icon(canvas, resized, x, y)

            cx, cy, bw, bh = yolo_box(x, y, target_size, target_size, args.canvas_size)
            label_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        stem = f"synth_{i:06d}"
        cv2.imwrite(str(image_out / f"{stem}.jpg"), canvas)
        (label_out / f"{stem}.txt").write_text("\n".join(label_lines), encoding="utf-8")

    print(f"Generated samples: {args.num_samples}")
    print(f"Image dir: {image_out}")
    print(f"Label dir: {label_out}")
    print("Classes:")
    for idx, name in enumerate(class_names):
        print(f"  {idx}: {name}")


if __name__ == "__main__":
    main()


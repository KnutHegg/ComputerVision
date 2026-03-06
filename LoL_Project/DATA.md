# Data Notes

## Sources
- Champion icon and minimap assets are used to create synthetic training images.
- Real minimap captures are optional and are kept out of version control.

## Dataset format
- YOLO detection format (`.jpg` + `.txt` labels).
- Split directory:
  - `images/train`, `images/val`, `images/test`
  - `labels/train`, `labels/val`, `labels/test`

## Versioning policy
- Large data files are excluded from git.
- `datasets/minimap_dataset/dataset.yaml` is tracked for reproducibility.
- Generated data should be recreated via scripts in `src/`.

## Ethics and legal
- This repository is for educational/research portfolio use.
- Ensure downstream use respects game terms, IP, and local rules.


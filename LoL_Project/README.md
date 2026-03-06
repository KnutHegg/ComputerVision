# LoL Minimap Champion Detection (Computer Vision Project)

Real-time champion detection on the League of Legends minimap using YOLO and synthetic data generation.

## What this project demonstrates
- Synthetic dataset generation for object detection.
- YOLO training and evaluation workflow.
- Real-time inference on a live game window (`dxcam` + OpenCV overlay).
- Reproducible scripts instead of notebook-only workflow.

## Current status
- Notebook experiment: [`main.ipynb`](./main.ipynb)
- Live demo script: [`src/infer_live.py`](./src/infer_live.py)
- Training outputs are intentionally excluded from Git via `.gitignore`.

## Project structure
```text
LoL_Project/
  configs/
    minimap_config.yaml
  datasets/
    minimap_dataset/
      dataset.yaml
  src/
    common.py
    capture_minimap.py
    generate_synth_data.py
    split_dataset.py
    train.py
    evaluate.py
    infer_live.py
  main.ipynb
  requirements.txt
  DATA.md
  MODEL_CARD.md
```

## Setup
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 1) Generate synthetic images
```powershell
python src/generate_synth_data.py `
  --backgrounds-dir datasets/minimap_backgrounds `
  --icons-dir datasets/icons/champs `
  --coco-json datasets/icons/champs/_annotations.coco.json `
  --output-dir datasets/synthetic_minimaps `
  --num-samples 1000
```

## 2) Create train/val/test split
```powershell
python src/split_dataset.py `
  --images-dir datasets/synthetic_minimaps/images `
  --labels-dir datasets/synthetic_minimaps/labels `
  --output-dir datasets/minimap_dataset `
  --train-ratio 0.8 `
  --val-ratio 0.1
```

## 3) Train model
```powershell
python src/train.py `
  --data datasets/minimap_dataset/dataset.yaml `
  --model yolo26n.pt `
  --epochs 100 `
  --imgsz 320 `
  --batch 32 `
  --name minimap_champion_detector
```

## 4) Evaluate model
```powershell
python src/evaluate.py `
  --model runs/detect/minimap_champion_detector/weights/best.pt `
  --data datasets/minimap_dataset/dataset.yaml `
  --split test
```

## 5) Run live detection
```powershell
python src/infer_live.py `
  --model runs/detect/minimap_champion_detector/weights/best.pt `
  --window-title "League of Legends (TM) Client"
```

## Optional: capture real minimap crops
```powershell
python src/capture_minimap.py `
  --output-dir raw_data/real_minimaps `
  --capture-interval 10 `
  --max-images 5000
```

## One-command pipeline
```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_all.ps1
```

## GitHub publishing notes
- Do not commit `runs/`, `results/`, `raw_data/`, or `.pt` files.
- Keep this repo code-first and upload large model/data artifacts as:
  - GitHub Releases, or
  - External dataset/model links (Kaggle, GDrive, etc.).
- Clear notebook outputs before pushing.
- Legacy entry points still work:
  - `python windowCapture.py` (for live inference)
  - `python raw_data/get_images.py` (for capture)

## Limitations
- The current class set is fixed to 10 champions.
- Synthetic-to-real gap may reduce robustness in real matches.
- Region crop is calibrated for one UI layout/resolution.

## License
MIT License (see [`LICENSE`](./LICENSE)).

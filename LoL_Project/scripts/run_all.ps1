param(
    [int]$NumSamples = 1000,
    [int]$Epochs = 100,
    [int]$ImgSize = 320,
    [int]$Batch = 32,
    [string]$BaseModel = "yolo26n.pt"
)

$ErrorActionPreference = "Stop"

Write-Host "1/4 Generating synthetic data..."
python src/generate_synth_data.py `
  --backgrounds-dir datasets/minimap_backgrounds `
  --icons-dir datasets/icons/champs `
  --coco-json datasets/icons/champs/_annotations.coco.json `
  --output-dir datasets/synthetic_minimaps `
  --num-samples $NumSamples

Write-Host "2/4 Splitting dataset..."
python src/split_dataset.py `
  --images-dir datasets/synthetic_minimaps/images `
  --labels-dir datasets/synthetic_minimaps/labels `
  --output-dir datasets/minimap_dataset `
  --train-ratio 0.8 `
  --val-ratio 0.1

Write-Host "3/4 Training..."
python src/train.py `
  --data datasets/minimap_dataset/dataset.yaml `
  --model $BaseModel `
  --epochs $Epochs `
  --imgsz $ImgSize `
  --batch $Batch `
  --name minimap_champion_detector

Write-Host "4/4 Evaluating..."
python src/evaluate.py `
  --model runs/detect/minimap_champion_detector/weights/best.pt `
  --data datasets/minimap_dataset/dataset.yaml `
  --split test `
  --imgsz $ImgSize

Write-Host "Pipeline complete."


# Computer Vision Modeling

This project contains CIFAR-10 classification notebooks, including a focused comparison of top models with and without augmentation.

## Main Notebook

- `best_models_classification.ipynb`

## What This Notebook Does

- Trains and evaluates 3 models:
  - `SimpleCNN` (from scratch, not pretrained)
  - `ResNet18` (pretrained on ImageNet-1K, fine-tuned on CIFAR-10)
  - `EfficientNet-B0` (pretrained on ImageNet-1K, fine-tuned on CIFAR-10)
- Runs two experiment settings:
  - Baseline (no augmentation)
  - Augmented training data
- Tracks metrics on train, validation, and test:
  - Accuracy
  - Precision (weighted)
  - Recall (weighted)
  - F1 (weighted, primary metric)
- Compares baseline vs augmented results and prints classification reports for the best model in each experiment.

## Data Augmentation Used

Augmented training pipeline includes combinations of:

- Random horizontal flip
- Random crop with padding
- Random rotation
- Random affine transforms (translate/scale/shear)
- Color jitter
- Random erasing

Validation and test sets are not augmented.

## How To Run

1. Open `best_models_classification.ipynb`.
2. Run cells from top to bottom.
3. Review:
   - `baseline_results`
   - `aug_results`
   - combined comparison tables and plots
   - final classification reports

## Runtime Notes

- Full training can take a while, especially on CPU.
- You can speed up runs by setting:
  - `TRAIN_SUBSET_SIZE`
  - `VAL_SUBSET_SIZE`
  - `TEST_SUBSET_SIZE`
- GPU is used automatically if available.

## Files

- `best_models_classification.ipynb` - top-3 model comparison, baseline vs augmentation
- `comparing_models_classification.ipynb` - broader model comparison notebook
- `CIFAR-10_classifier_conv.ipynb` - additional CIFAR-10 experiments

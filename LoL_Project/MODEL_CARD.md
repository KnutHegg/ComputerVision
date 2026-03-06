# Model Card: LoL Minimap Champion Detector

## Model details
- Task: Object detection (10 classes)
- Framework: Ultralytics YOLO
- Input: Minimap crop (default 320x320 training size)
- Output: Bounding boxes + class probabilities

## Classes
- ally_Ahri
- ally_Ashe
- ally_Darius
- ally_Hecarim
- ally_Khazix
- enemy_Nasus
- enemy_Sejuani
- enemy_Warwick
- enemy_XinZhao
- enemy_Ziggs

## Intended use
- Portfolio demonstration of CV pipeline design:
  - synthetic data generation
  - training/evaluation
  - live inference overlay

## Limitations
- Synthetic bias and domain gap to real gameplay.
- Calibrated crop assumes specific UI scale and layout.
- Small class scope (10 champions) and no temporal tracking.

## Safety and misuse
- Not for anti-cheat evasion or unfair gameplay tooling.
- Use responsibly and in line with applicable platform rules.


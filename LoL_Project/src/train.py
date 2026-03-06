import argparse
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO model for minimap champion detection.")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset.yaml")
    parser.add_argument("--model", type=str, default="yolo26n.pt", help="Base model or checkpoint path")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--name", type=str, default="minimap_champion_detector")
    parser.add_argument("--project", type=str, default="runs/detect")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        project=args.project,
    )


if __name__ == "__main__":
    main()


import argparse
from src.training.train_yolo import train_yolo_model
from src.training.train_depth import train_depth_model
from src.training.train_sam import train_sam_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=['yolo', 'depth', 'sam', 'all'], required=True)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    data_dir = "data/nyu_prompt_331"  # Custom dataset from paper

    if args.model in ['yolo', 'all']:
        train_yolo_model(f"{data_dir}/dataset.yaml", args.epochs)
    if args.model in ['depth', 'all']:
        train_depth_model(data_dir, args.epochs)
    if args.model in ['sam', 'all']:
        train_sam_model(data_dir, args.epochs)


if __name__ == "__main__":
    main()

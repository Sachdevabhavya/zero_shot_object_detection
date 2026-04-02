import argparse, yaml, os
from src.pipeline import ZeroShotPipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output_dir", default="data/output")
    parser.add_argument("--visualize", action="store_true", help="Enable 3D visualization (requires display)")
    parser.add_argument("--save_intermediates", action="store_true", help="Save detection, mask, and depth map outputs")
    args = parser.parse_args()

    with open("config/config.yaml", "r") as f: config = yaml.safe_load(f)
    with open("config/models.yaml", "r") as f: models = yaml.safe_load(f)

    cfg = {'conf': config['inference']['yolo_conf'], 'intrinsics': config['camera']['intrinsics'], 'models': models['paths']}
    
    os.makedirs(args.output_dir, exist_ok=True)
    ZeroShotPipeline(cfg).run(args.image_path, args.prompt, args.output_dir, visualize=args.visualize, save_intermediates=args.save_intermediates)

if __name__ == "__main__":
    main()
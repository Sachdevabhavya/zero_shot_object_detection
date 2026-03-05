import os
from src.modules.detector import ObjectDetector
from src.modules.segmentor import ObjectSegmentor
from src.modules.depth_estimator import DepthEstimator
from src.utils.geometry import GeometryUtils
from src.utils.file_io import load_image, save_point_cloud
from src.utils.visualization import visualize_pcd

class ZeroShotPipeline:
    def __init__(self, config):
        print("[*] Loading modules: Detection, Segmentation, Depth...")
        self.detector = ObjectDetector(config['models']['yolo_model'], config['conf'])
        self.segmentor = ObjectSegmentor(config['models']['sam_checkpoint'])
        self.depth_estimator = DepthEstimator(config['models']['depth_model'])
        self.geometry = GeometryUtils(config['intrinsics'])

    def run(self, image_path, prompt, output_dir):
        img_rgb, img_pil = load_image(image_path)
        
        # 1. Detect
        bbox = self.detector.detect(image_path, prompt)
        # 2. Segment
        mask = self.segmentor.segment(img_rgb, bbox)
        # 3. Depth
        depth_map = self.depth_estimator.estimate_metric_depth(img_pil)
        # 4. Reconstruct 
        points, colors = self.geometry.unproject_pixels(img_rgb, depth_map, mask)
        pcd = self.geometry.create_point_cloud(points, colors)
        
        out_path = os.path.join(output_dir, f"{prompt.replace(' ', '_')}.ply")
        save_point_cloud(pcd, out_path)
        visualize_pcd(pcd, f"3D Object: {prompt}")
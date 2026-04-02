import os, cv2
from src.modules.detector import ObjectDetector
from src.modules.segmentor import ObjectSegmentor
from src.modules.depth_estimator import DepthEstimator
from src.utils.geometry import GeometryUtils
from src.utils.file_io import load_image, save_point_cloud, save_image, save_depth_map, save_mask
from src.utils.visualization import visualize_pcd

class ZeroShotPipeline:
    def __init__(self, config):
        print("[*] Loading modules: Detection, Segmentation, Depth...")
        self.detector = ObjectDetector(config['models']['yolo_model'], config['conf'])
        self.segmentor = ObjectSegmentor(config['models']['sam_checkpoint'])
        self.depth_estimator = DepthEstimator(config['models']['depth_model'])
        self.geometry = GeometryUtils(config['intrinsics'])

    def run(self, image_path, prompt, output_dir, visualize=False, save_intermediates=False):
        img_rgb, img_pil = load_image(image_path)
        
        # 1. Detect
        bbox = self.detector.detect(image_path, prompt)
        if save_intermediates and bbox is not None:
            img_with_bbox = img_rgb.copy()
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img_with_bbox, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            detect_path = os.path.join(output_dir, f"{prompt.replace(' ', '_')}_01_detection.png")
            save_image(cv2.cvtColor(img_with_bbox, cv2.COLOR_RGB2BGR), detect_path)
        
        # 2. Segment
        mask = self.segmentor.segment(img_rgb, bbox)
        if save_intermediates:
            mask_path = os.path.join(output_dir, f"{prompt.replace(' ', '_')}_02_mask.png")
            save_mask(mask, mask_path)
        
        # 3. Depth
        depth_map = self.depth_estimator.estimate_metric_depth(img_pil)
        if save_intermediates:
            depth_path = os.path.join(output_dir, f"{prompt.replace(' ', '_')}_03_depth.png")
            save_depth_map(depth_map, depth_path)
        
        # 4. Reconstruct 
        points, colors = self.geometry.unproject_pixels(img_rgb, depth_map, mask)
        pcd = self.geometry.create_point_cloud(points, colors)
        
        out_path = os.path.join(output_dir, f"{prompt.replace(' ', '_')}_04_pointcloud.ply")
        save_point_cloud(pcd, out_path)
        if visualize:
            visualize_pcd(pcd, f"3D Object: {prompt}")
        else:
            print(f"[*] Visualization disabled. Point cloud saved to: {out_path}")
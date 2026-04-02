import cv2, open3d as o3d, numpy as np
from PIL import Image

def load_image(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return img, Image.fromarray(img)

def save_point_cloud(pcd, path):
    o3d.io.write_point_cloud(path, pcd)
    print(f"[*] Point cloud successfully saved to {path}")

def save_image(img, path):
    """Save RGB/BGR image to file."""
    if isinstance(img, Image.Image):
        img.save(path)
    else:
        # Assume OpenCV BGR format, convert to RGB for PIL
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
        cv2.imwrite(path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR) if len(img.shape) == 3 else img_rgb)
    print(f"[*] Image saved to {path}")

def save_depth_map(depth, path):
    """Save depth map as normalized PNG (16-bit) or numpy file."""
    if path.endswith('.npy'):
        np.save(path, depth)
    else:
        # Normalize to 0-65535 for 16-bit PNG
        depth_norm = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 65535).astype(np.uint16)
        cv2.imwrite(path, depth_norm)
    print(f"[*] Depth map saved to {path}")

def save_mask(mask, path):
    """Save segmentation mask as binary image."""
    mask_uint8 = (mask * 255).astype(np.uint8)
    cv2.imwrite(path, mask_uint8)
    print(f"[*] Mask saved to {path}")

def convert_ply_to_jpg(ply_path, jpg_path):
    """Convert PLY point cloud to JPG image using orthogonal projection."""
    try:
        # Load point cloud
        pcd = o3d.io.read_point_cloud(ply_path)
        
        # Get point cloud bounds and center
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else np.ones((len(points), 3))
        
        # Create axis-aligned bounding box
        min_bound = points.min(axis=0)
        max_bound = points.max(axis=0)
        center = (min_bound + max_bound) / 2
        
        # Normalize points to [-1, 1] range
        scale = 2 / np.linalg.norm(max_bound - min_bound)
        points_normalized = (points - center) * scale
        
        # Project onto XY plane (top-down view)
        # Map to image coordinates [0, h], [0, w]
        img_size = 512
        x_proj = points_normalized[:, 0]
        y_proj = points_normalized[:, 1]
        
        # Map to pixel coordinates
        x_pix = ((x_proj + 1) / 2 * (img_size - 1)).astype(int)
        y_pix = ((y_proj + 1) / 2 * (img_size - 1)).astype(int)
        
        # Clip to image bounds
        mask = (x_pix >= 0) & (x_pix < img_size) & (y_pix >= 0) & (y_pix < img_size)
        x_pix = x_pix[mask]
        y_pix = y_pix[mask]
        colors_masked = colors[mask]
        
        # Create image
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
        
        # Convert colors to uint8 (assuming they are in [0, 1] range)
        colors_uint8 = (colors_masked * 255).astype(np.uint8) if colors_masked.max() <= 1.0 else colors_masked.astype(np.uint8)
        
        # Draw points on image
        img[y_pix, x_pix] = colors_uint8
        
        # Save as JPG
        cv2.imwrite(jpg_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"[*] Point cloud image saved to {jpg_path}")
        
    except Exception as e:
        print(f"[!] Error converting PLY to JPG: {e}")
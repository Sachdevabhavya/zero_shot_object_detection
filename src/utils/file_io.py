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
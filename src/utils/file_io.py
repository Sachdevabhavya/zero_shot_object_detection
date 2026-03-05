import cv2, open3d as o3d
from PIL import Image

def load_image(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return img, Image.fromarray(img)

def save_point_cloud(pcd, path):
    o3d.io.write_point_cloud(path, pcd)
    print(f"[*] Point cloud successfully saved to {path}")
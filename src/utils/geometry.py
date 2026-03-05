import numpy as np
import open3d as o3d

class GeometryUtils:
    def __init__(self, K):
        self.fx, self.fy, self.cx, self.cy = K['fx'], K['fy'], K['cx'], K['cy']

    def unproject_pixels(self, rgb, depth, mask):
        # Applies Equation (2) and (3) from paper to lift 2D to 3D [cite: 17, 106]
        h, w = depth.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        valid = mask > 0
        z, u, v = depth[valid], u[valid], v[valid]
        
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        
        return np.stack((x, y, z), axis=-1), rgb[valid] / 255.0

    def create_point_cloud(self, points, colors):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd
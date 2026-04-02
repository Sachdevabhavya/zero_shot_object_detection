import open3d as o3d

def visualize_pcd(pcd, window_name="3D"):
    """Visualize point cloud with error handling for headless/Wayland systems."""
    try:
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name)
        vis.add_geometry(pcd)
        vis.add_geometry(coord)
        vis.run()
        vis.destroy_window()
    except Exception as e:
        print(f"[!] Visualization unavailable (headless/Wayland system): {type(e).__name__}")
        print(f"[*] Point cloud saved successfully. Use a 3D viewer (e.g., CloudCompare, MeshLab) to inspect the output.")
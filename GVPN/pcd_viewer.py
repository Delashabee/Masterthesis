import open3d as o3d
import numpy as np

# Read PCD file
pcd_convert = o3d.io.read_point_cloud("D:/Programfiles/Myscvp/backup/034.pcd")
pcd_mesh = o3d.io.read_point_cloud("D:/Programfiles/Myscvp/3d_models/034.pcd")

'''# Set the first point cloud to gray (RGB = 0.5, 0.5, 0.5)
gray_color = [0.5, 0.5, 0.5]
pcd_convert.colors = o3d.utility.Vector3dVector(np.tile(gray_color, (len(pcd_convert.points), 1)))

# Set multiple colors for the second point cloud
# Example: Set random colors or partition-based colors
colors = np.random.rand(len(pcd_mesh.points), 3)  # Generate random colors for each point
pcd_mesh.colors = o3d.utility.Vector3dVector(colors)
'''
# Visualize the first point cloud (gray)
o3d.visualization.draw_geometries([pcd_convert],
                                  window_name="PCD Viewer 1",
                                  width=800,
                                  height=600,
                                  left=50,
                                  top=50,
                                  point_show_normal=False,
                                  mesh_show_wireframe=False,
                                  mesh_show_back_face=False)

# Visualize the second point cloud (colored)
o3d.visualization.draw_geometries([pcd_mesh],
                                  window_name="PCD Viewer 2",
                                  width=800,
                                  height=600,
                                  left=50,
                                  top=50,
                                  point_show_normal=False,
                                  mesh_show_wireframe=False,
                                  mesh_show_back_face=False)

import CSF
import laspy
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

# Step 1: Load the LAS File
las = laspy.read('cloud_17m.las')

# Step 2: Extract point cloud data (X, Y, Z coordinates)
points = np.vstack((las.x, las.y, las.z)).transpose()

# Normalize points to set Z grounded at 0
points[:, 2] -= points[:, 2].min()

# Step 3: Extract color information (Red, Green, Blue)
colors = np.vstack((las.red, las.green, las.blue)).transpose()
colors = colors / 65535.0  # Normalize colors to [0, 1] range

# Step 4: Set up CSF for ground point classification
csf = CSF.CSF()
csf.params.bSloopSmooth = True    # Optional: Smooth slope (set based on your needs)
csf.params.cloth_resolution = 0.1 # Resolution of the cloth (higher means more detailed)
csf.params.rigidness = 3          # Rigidness of the cloth
csf.params.time_step = 0.65       # Time step for simulation

# Step 5: Load the points into CSF
csf.setPointCloud(points)

# Step 6: Perform ground filtering using CSF
ground_indexes = CSF.VecInt()  # List for ground point indexes
non_ground_indexes = CSF.VecInt()  # List for non-ground point indexes

csf.do_filtering(ground_indexes, non_ground_indexes)  # Use lists to store the indexes

# Step 7: Extract ground points and their colors using the ground indexes
ground_points = points[ground_indexes]
ground_colors = colors[ground_indexes]

# Step 8: Extract non-ground points (tree points)
tree_points = points[non_ground_indexes]
tree_colors = colors[non_ground_indexes]

# Step 9: Export tree points to a LAS file
# Create a new header for the LAS file
header = laspy.LasHeader(point_format=las.header.point_format, version=las.header.version)
header.scales = las.header.scales  # Copy the scales from the original LAS file
header.offsets = las.header.offsets  # Copy the offsets from the original LAS file

# Create a new LAS file for the segmented tree points
output_file = laspy.LasData(header)

# Set the X, Y, Z coordinates for the tree points
output_file.x = tree_points[:, 0]
output_file.y = tree_points[:, 1]
output_file.z = tree_points[:, 2]

# Set the color information
output_file.red = (tree_colors[:, 0] * 65535).astype(np.uint16)
output_file.green = (tree_colors[:, 1] * 65535).astype(np.uint16)
output_file.blue = (tree_colors[:, 2] * 65535).astype(np.uint16)

# Write the output to a new LAS file
output_file.write('output/segment_tree.las')
print("Tree points have been exported to 'output/segment_tree.las'.")

# Step 10: Visualize tree points with Open3D using smaller point size
tree_pcd = o3d.geometry.PointCloud()
tree_pcd.points = o3d.utility.Vector3dVector(tree_points)
tree_pcd.colors = o3d.utility.Vector3dVector(tree_colors)
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Tree Points Visualization")
vis.add_geometry(tree_pcd)
opt = vis.get_render_option()
opt.point_size = 1.0  # Smaller point size for rendering
vis.run()
vis.destroy_window()
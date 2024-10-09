import CSF
import laspy
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

# Step 1: Load the LAS File
las = laspy.read('cloud_sub.las')

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
output_file = laspy.create(point_format=las.point_format, file_version=las.header.version)
output_file.points = laspy.ScaleAwarePointRecord(points[non_ground_indexes], las.point_format)
output_file.write('output/segment_tree.las')
print("Tree points have been exported to 'output/segment_tree.las'.")

# Step 10: Visualize tree points with Open3D using smaller point size
tree_pcd = o3d.geometry.PointCloud()
tree_pcd.points = o3d.utility.Vector3dVector(tree_points)
tree_pcd.colors = o3d.utility.Vector3dVector(tree_colors)

# Set rendering options for smaller point size
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Tree Points Visualization")
vis.add_geometry(tree_pcd)
opt = vis.get_render_option()
opt.point_size = 1.0  # Smaller point size for rendering
vis.run()
vis.destroy_window()

# Step 11: Slice tree points by height and detect trees
height_min = 1.0
height_max = 7.0
slice_height = 1.0  # Thickness of slice

slice_results = []

# Slice by height range
current_height = height_min
while current_height < height_max:
    mask = np.logical_and(tree_points[:, 2] >= current_height, tree_points[:, 2] < current_height + slice_height)
    slice_points = tree_points[mask]
    
    if len(slice_points) > 0:
        slice_points_2d = slice_points[:, :2]  # Only take X and Y coordinates
        
        # Use DBSCAN clustering algorithm
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=0.2, min_samples=5).fit(slice_points_2d)
        labels = clustering.labels_
        
        # Get clustering results, -1 means noise
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remove noise label
        
        # Calculate centroids for each cluster
        tree_positions = []
        for label in unique_labels:
            class_member_mask = (labels == label)
            xy = slice_points_2d[class_member_mask]
            centroid = xy.mean(axis=0)
            tree_positions.append(centroid)
        
        # Add current slice results to list
        slice_results.append({
            'height_range': (current_height, current_height + slice_height),
            'tree_positions': tree_positions
        })
    
    # Update height
    current_height += slice_height

# Select a specific height range slice and visualize
selected_slice = None
for result in slice_results:
    if result['height_range'] == (1.0, 2.0):
        selected_slice = result
        break

if selected_slice is not None:
    tree_positions = np.array(selected_slice['tree_positions'])
    plt.scatter(tree_points[:, 0], tree_points[:, 1], s=1, c='gray', label='Point Cloud')
    plt.scatter(tree_positions[:, 0], tree_positions[:, 1], s=50, c='red', marker='x', label='Detected Trees')
    plt.title(f'Tree Detection at Height {selected_slice["height_range"][0]}-{selected_slice["height_range"][1]} m')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()
else:
    print('No data for the selected height range.')

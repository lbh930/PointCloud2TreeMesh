import laspy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic_2d
from skimage.feature import peak_local_max

# Step 1: Load the LAS File
# Replace 'your_pointcloud.las' with the path to your LAS file
las = laspy.read('cloud.las')

# Extract point coordinates
points = np.vstack((las.x, las.y, las.z)).transpose()

# Remove points with height lower than 180
points = points[points[:, 2] > 186]

# Step 2: Generate a Height Map
# Define grid resolution (adjust based on point cloud density)
grid_size = 0.5  # Grid cell size in units matching your data (e.g., meters)

# Compute grid boundaries
x_min, x_max = points[:, 0].min(), points[:, 0].max()
y_min, y_max = points[:, 1].min(), points[:, 1].max()

# Create grid edges
x_edges = np.arange(x_min, x_max + grid_size, grid_size)
y_edges = np.arange(y_min, y_max + grid_size, grid_size)

# Compute the height map using the maximum elevation in each grid cell
height_map, x_edges, y_edges, _ = binned_statistic_2d(
    points[:, 0], points[:, 1], points[:, 2],
    statistic='max', bins=[x_edges, y_edges]
)

# Step 3: Ground Segmentation
# Define a threshold to separate ground from vegetation
ground_threshold = 186  # Adjust based on the minimum vegetation height

# Create masks for ground and vegetation (optional)
ground_mask = np.logical_and(height_map < ground_threshold, ~np.isnan(height_map))
vegetation_mask = np.logical_and(height_map >= ground_threshold, ~np.isnan(height_map))

# Step 4: Smooth the Height Map
# Handle NaN values by replacing them with the minimum valid height
min_height = np.nanmin(height_map)
height_map_filled = np.nan_to_num(height_map, nan=min_height)

# Apply Gaussian smoothing to the height map
sigma = 1  # Standard deviation for Gaussian kernel (adjust as needed)
smoothed_height_map = gaussian_filter(height_map_filled, sigma=sigma)

# Step 5: Detect Tree Locations
# Prepare the smoothed height map for peak detection
smoothed_min = np.nanmin(smoothed_height_map)
smoothed_height_map_prepared = np.nan_to_num(smoothed_height_map, nan=smoothed_min - 1)

# Define minimum distance between tree peaks in pixels
tree_spacing_meters = 2.0  # Minimum expected distance between trees
min_distance = max(1, int(tree_spacing_meters / grid_size))

# Detect local maxima (tree tops) in the smoothed height map
coordinates = peak_local_max(
    smoothed_height_map_prepared,
    min_distance=min_distance,
    threshold_abs=ground_threshold
)
# 计算网格中心
x_centers = (x_edges[:-1] + x_edges[1:]) / 2
y_centers = (y_edges[:-1] + y_edges[1:]) / 2

# 提取树的位置（交换索引顺序）
tree_x = x_centers[coordinates[:, 0]]
tree_y = y_centers[coordinates[:, 1]]
tree_z = smoothed_height_map[coordinates[:, 0], coordinates[:, 1]]

# 调试信息
print("x_centers.shape:", x_centers.shape)
print("y_centers.shape:", y_centers.shape)
print("coordinates.shape:", coordinates.shape)
print("Max index in coordinates[:, 0]:", np.max(coordinates[:, 0]))
print("Max index in coordinates[:, 1]:", np.max(coordinates[:, 1]))


# Step 6: Visualize Intermediate Results
# Visualize the original height map
plt.figure(figsize=(10, 8))
plt.imshow(
    height_map.T,
    extent=(x_min, x_max, y_min, y_max),
    origin='lower',
    cmap='terrain'
)
plt.title('Original Height Map')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.colorbar(label='Height (m)')
plt.show()

# Visualize the smoothed height map
plt.figure(figsize=(10, 8))
plt.imshow(
    smoothed_height_map.T,
    extent=(x_min, x_max, y_min, y_max),
    origin='lower',
    cmap='terrain'
)
plt.title('Smoothed Height Map')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.colorbar(label='Height (m)')
plt.show()

# Visualize detected tree locations on the smoothed height map
plt.figure(figsize=(10, 8))
plt.imshow(
    smoothed_height_map.T,
    extent=(x_min, x_max, y_min, y_max),
    origin='lower',
    cmap='terrain'
)
plt.scatter(
    tree_x,
    tree_y,
    c='red',
    s=10,
    label='Detected Trees'
)
plt.title('Detected Tree Locations')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.colorbar(label='Height (m)')
plt.show()

# Step 7: Output Tree Locations
# Create a DataFrame with tree coordinates and heights
tree_locations = pd.DataFrame({
    'X': tree_x,
    'Y': tree_y,
    'Z': tree_z
})

# Save the tree locations to a CSV file
tree_locations.to_csv('tree_locations.csv', index=False)

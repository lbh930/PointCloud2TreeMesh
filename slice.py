import os
import sys

import laspy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN  # Import DBSCAN


def load_segmented_tree(file_path):
    """Loads the segmented tree LAS file."""
    try:
        las = laspy.read(file_path)
        print(f"Loaded LAS file: {file_path}")
    except Exception as e:
        print(f"Failed to load LAS file: {e}")
        sys.exit(1)
    
    # Extract points
    points = np.vstack((las.x, las.y, las.z)).transpose()
    
    # Check for color info
    if ('red' in las.point_format.dimension_names and
        'green' in las.point_format.dimension_names and
        'blue' in las.point_format.dimension_names):
        colors = np.vstack((las.red, las.green, las.blue)).transpose()
        colors = colors / 65535.0  # Normalize to [0,1]
    else:
        colors = np.ones((points.shape[0], 3))  # White if no color
    
    return points, colors


def print_bounding_box(points):
    """Prints the bounding box of the point cloud."""
    min_x, min_y, min_z = points.min(axis=0)
    max_x, max_y, max_z = points.max(axis=0)
    print("Bounding Box:")
    print(f"  X: {min_x} to {max_x}")
    print(f"  Y: {min_y} to {max_y}")
    print(f"  Z: {min_z} to {max_z}")


def normalize_z(points):
    """Shifts the Z-axis so that it starts at 0."""
    min_z = points[:, 2].min()
    points[:, 2] -= min_z
    print(f"Normalized Z-axis by subtracting {min_z}")
    return points


def create_output_dir(directory):
    """Creates the output directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")


def slice_and_save(points, colors, output_dir, slice_min=0.5, slice_max=15.0, slice_step=0.25):
    """
    Slices the point cloud along the Z-axis, clusters each slice using DBSCAN,
    and saves the plots with cluster centers.
    """
    from sklearn.cluster import DBSCAN  # Ensure DBSCAN is imported
    current_slice = slice_min
    while current_slice <= slice_max:
        # Get points in the current slice based on Z-axis (height)
        mask = np.logical_and(points[:, 2] >= current_slice, points[:, 2] < current_slice + slice_step)
        slice_pts = points[mask]
        slice_cols = colors[mask]
        
        if len(slice_pts) == 0:
            print(f"No points found in slice {current_slice}m to {current_slice + slice_step}m.")
            current_slice += slice_step
            continue
        
        # Project to 2D (X and Y) since Z is height
        slice_2d = slice_pts[:, [0, 1]]
        
        # Use DBSCAN to cluster the points
        # Set parameters for DBSCAN
        eps = 2  # This value may need to be adjusted based on your data
        min_samples = 10
        
        # Run DBSCAN
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(slice_2d)
        labels = db.labels_
        unique_labels = set(labels)
        
        # Number of clusters in labels, ignoring noise if present.
        n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
        print(f"Slice {current_slice}m to {current_slice + slice_step}m: Found {n_clusters} clusters")
        
        # Create a color palette
        colors_db = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        
        # Plotting
        plt.figure(figsize=(10, 8))
        for k, col in zip(unique_labels, colors_db):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]
            
            class_member_mask = (labels == k)
            xy = slice_2d[class_member_mask]
            plt.plot(xy[:, 0], xy[:, 1], '.', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=3)
            
            if k != -1:
                # Compute the centroid of the cluster
                centroid = xy.mean(axis=0)
                plt.plot(centroid[0], centroid[1], 'x', markerfacecolor=tuple(col),
                         markeredgecolor='k', markersize=12)
        
        plt.title(f'Slice Z: {current_slice}m to {current_slice + slice_step}m - {n_clusters} clusters')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')  # To maintain aspect ratio
        plt.tight_layout()
        
        # Save the plot
        filename = f'slice_{current_slice}_{current_slice + slice_step}_clusters.png'
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved clustered slice image: {save_path}")
        
        current_slice += slice_step


def main():
    # Path to the segmented trunk LAS file
    input_las = 'output/segment_trunk.las'
    
    if not os.path.isfile(input_las):
        print(f"Segmented tree LAS file does not exist: {input_las}")
        sys.exit(1)
    
    # Load the data
    points, colors = load_segmented_tree(input_las)
    
    # Sanity check: Print bounding box
    print_bounding_box(points)
    
    # Normalize Z-axis
    points = normalize_z(points)
    
    # Output directory for slice images
    output_dir = 'output/slice_vis'
    create_output_dir(output_dir)
    
    # Slice parameters
    slice_min = 2.0
    slice_max = 6
    slice_step = 0.25
    
    # Perform slicing, clustering, and save images
    slice_and_save(points, colors, output_dir, slice_min, slice_max, slice_step)
    
    print("Slicing, clustering, and mapping completed.")


if __name__ == "__main__":
    main()

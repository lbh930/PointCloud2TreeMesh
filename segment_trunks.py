import os
import sys

import laspy
import numpy as np
from matplotlib.colors import rgb_to_hsv


def load_segmented_tree(file_path):
    """
    Loads the segmented tree LAS file and extracts points and colors.

    Args:
        file_path (str): Path to the LAS file.

    Returns:
        tuple: (points, colors, las_data)
    """
    try:
        las = laspy.read(file_path)
        print(f"Successfully loaded LAS file: {file_path}")
    except Exception as e:
        print(f"Error loading LAS file: {e}")
        sys.exit(1)

    # Get XYZ coordinates
    points = np.vstack((las.x, las.y, las.z)).transpose()

    # Check if color information is present
    if all(dim in las.point_format.dimension_names for dim in ['red', 'green', 'blue']):
        colors = np.vstack((las.red, las.green, las.blue)).transpose()
        colors = colors / 65535.0  # Normalize to [0,1]
    else:
        colors = np.ones((points.shape[0], 3))  # Default to white if no color

    return points, colors, las


def print_bounding_box(points):
    """
    Prints the bounding box of the point cloud for verification.

    Args:
        points (ndarray): Array of point coordinates.
    """
    min_vals = points.min(axis=0)
    max_vals = points.max(axis=0)
    print("Point Cloud Bounding Box:")
    print(f"  X: {min_vals[0]:.2f} to {max_vals[0]:.2f}")
    print(f"  Y: {min_vals[1]:.2f} to {max_vals[1]:.2f}")
    print(f"  Z: {min_vals[2]:.2f} to {max_vals[2]:.2f}")


def normalize_z(points):
    """
    Normalizes the Z-axis so that the lowest point starts at 0.

    Args:
        points (ndarray): Array of point coordinates.

    Returns:
        ndarray: Normalized points.
    """
    min_z = points[:, 2].min()
    points[:, 2] -= min_z
    print(f"Z-axis normalized by subtracting {min_z:.2f}")
    return points


def create_output_dir(directory):
    """
    Creates the output directory if it doesn't exist.

    Args:
        directory (str): Path to the directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")


def filter_trunk(points, colors, hue_min=0.1, hue_max=0.50, saturation_min=0.15, value_min=0.1):
    """
    Filters out green and yellow-green points based on HSV color thresholds to isolate tree trunks.

    Args:
        points (ndarray): Array of point coordinates.
        colors (ndarray): Array of normalized RGB colors.
        hue_min (float): Minimum Hue value to consider as green.
        hue_max (float): Maximum Hue value to consider as green.
        saturation_min (float): Minimum Saturation to consider color valid.
        value_min (float): Minimum Value (brightness) to consider color valid.

    Returns:
        tuple: (trunk_points, trunk_colors)
    """
    # Convert RGB to HSV
    hsv = rgb_to_hsv(colors)

    # Define mask for green and yellow-green colors
    green_mask = (
        (hsv[:, 0] >= hue_min) &
        (hsv[:, 0] <= hue_max) &
        (hsv[:, 1] >= saturation_min) &
        (hsv[:, 2] >= value_min)
    )

    # Invert mask to keep trunk points
    trunk_mask = ~green_mask

    total_points = points.shape[0]
    excluded_points = np.sum(green_mask)
    trunk_points = points[trunk_mask]
    trunk_colors = colors[trunk_mask]
    trunk_count = trunk_points.shape[0]

    print(f"Total points: {total_points}")
    print(f"Excluded green points (likely leaves): {excluded_points}")
    print(f"Trunk points remaining: {trunk_count}")

    return trunk_points, trunk_colors


def save_trunk_points(trunk_points, trunk_colors, original_las, output_path):
    """
    Saves the filtered trunk points to a new LAS file.

    Args:
        trunk_points (ndarray): Array of trunk point coordinates.
        trunk_colors (ndarray): Array of trunk point colors.
        original_las (LasData): Original LAS data to preserve header information.
        output_path (str): Path to save the new LAS file.
    """
    # Create new LAS header based on original
    header = laspy.LasHeader(point_format=original_las.header.point_format, version=original_las.header.version)
    header.scales = original_las.header.scales
    header.offsets = original_las.header.offsets

    # Initialize new LAS data
    trunk_las = laspy.LasData(header)

    # Assign coordinates
    trunk_las.x = trunk_points[:, 0]
    trunk_las.y = trunk_points[:, 1]
    trunk_las.z = trunk_points[:, 2]

    # Assign colors if present
    if all(dim in original_las.point_format.dimension_names for dim in ['red', 'green', 'blue']):
        trunk_las.red = (trunk_colors[:, 0] * 65535).astype(np.uint16)
        trunk_las.green = (trunk_colors[:, 1] * 65535).astype(np.uint16)
        trunk_las.blue = (trunk_colors[:, 2] * 65535).astype(np.uint16)

    # Write to LAS file
    trunk_las.write(output_path)
    print(f"Trunk points saved to '{output_path}'.")


def main():
    # Path to the segmented tree LAS file
    input_las = 'output/segment_tree.las'

    if not os.path.isfile(input_las):
        print(f"Segmented tree LAS file does not exist: {input_las}")
        sys.exit(1)

    # Load the data
    points, colors, original_las = load_segmented_tree(input_las)

    # Print bounding box for sanity check
    print_bounding_box(points)

    # Normalize Z-axis
    points = normalize_z(points)

    # Filter out leaves based on color
    trunk_points, trunk_colors = filter_trunk(points, colors)

    # Ensure output directory exists
    output_dir = 'output'
    create_output_dir(output_dir)

    # Define output file path
    output_las = os.path.join(output_dir, 'segment_trunk.las')

    # Save the trunk points to LAS file
    save_trunk_points(trunk_points, trunk_colors, original_las, output_las)

    print("Trunk segmentation done.")


if __name__ == "__main__":
    main()

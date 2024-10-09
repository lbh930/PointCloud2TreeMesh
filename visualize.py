import argparse
import os
import sys

import laspy
import numpy as np
import open3d as o3d


def load_las_file(file_path):
    """
    Loads a LAS or LAZ file and gets the points and colors.
    If no color info, defaults to white.
    """
    try:
        las = laspy.read(file_path)
    except Exception as e:
        print(f"Couldn't read LAS file: {e}")
        sys.exit(1)

    # Get the XYZ coordinates
    points = np.vstack((las.x, las.y, las.z)).transpose()

    # Check if colors are there
    if 'red' in las.point_format.dimension_names and \
       'green' in las.point_format.dimension_names and \
       'blue' in las.point_format.dimension_names:
        colors = np.vstack((las.red, las.green, las.blue)).transpose()
        colors = colors / 65535.0  # Normalize to [0,1]
    else:
        colors = np.ones((points.shape[0], 3))  # White if no color

    return points, colors


def load_ply_file(file_path):
    """
    Loads a PLY file and gets the points and colors.
    If no color info, defaults to white.
    """
    try:
        pcd = o3d.io.read_point_cloud(file_path)
    except Exception as e:
        print(f"Couldn't read PLY file: {e}")
        sys.exit(1)

    points = np.asarray(pcd.points)
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
    else:
        colors = np.ones((points.shape[0], 3))  # White if no color

    return points, colors


def load_file(file_path):
    """
    Decides which loader to use based on file extension.
    Supports LAS, LAZ, and PLY.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.las', '.laz']:
        return load_las_file(file_path)
    elif ext == '.ply':
        return load_ply_file(file_path)
    else:
        print(f"Unsupported file format: {ext}")
        sys.exit(1)


def visualize_point_cloud(points, colors, point_size=2.0, window_name="Point Cloud"):
    """
    Shows the point cloud using Open3D with some settings.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    vis.add_geometry(pcd)

    # Messy settings for the render
    opt = vis.get_render_option()
    opt.point_size = point_size
    opt.background_color = np.asarray([0, 0, 0])  # Black background, cuz why not

    vis.run()
    vis.destroy_window()


def main():
    # Command line args setup
    parser = argparse.ArgumentParser(description="Visualize point clouds with Open3D. Supports LAS, LAZ, PLY.")
    parser.add_argument("file_path", type=str, help="Path to the point cloud file (.las, .laz, .ply)")
    parser.add_argument("--point_size", type=float, default=1.0, help="Size of points in the display (default: 1.0)")
    parser.add_argument("--window_name", type=str, default="Point Cloud Visualization", help="Name of the window")

    args = parser.parse_args()

    file_path = args.file_path

    if not os.path.isfile(file_path):
        print(f"File does not exist: {file_path}")
        sys.exit(1)

    # Load the data
    points, colors = load_file(file_path)

    print(f"Loaded point cloud: {file_path}")
    print(f"Number of points: {points.shape[0]}")

    # Visualize it
    visualize_point_cloud(points, colors, point_size=args.point_size, window_name=args.window_name)


if __name__ == "__main__":
    main()

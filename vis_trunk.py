import json
import os
import sys

import laspy
import matplotlib.pyplot as plt
import numpy as np


def visualize_trunk_points_with_point_cloud(las_file_path, trunk_points_path):
    """
    读取给定路径的 LAS 点云文件和树干点文件，在顶视图中将原始点云（保留原有颜色）和树干点进行可视化。

    参数:
    - las_file_path: str，输入 LAS 文件的路径。
    - trunk_points_path: str，树干点 JSON 文件的路径。
    """
    # 加载点云数据和颜色
    points, colors = load_point_cloud(las_file_path)

    # 归一化 Z 轴
    points = normalize_z(points)

    # 加载树干点
    trunk_points = load_trunk_points(trunk_points_path)

    # 在顶视图中可视化点云和树干点
    plt.figure(figsize=(12, 10))
    plt.scatter(points[:, 0], points[:, 1], c=colors, s=1, label='Point Cloud')

    # 绘制树干点
    if trunk_points:
        trunk_centers_array = np.array([center for center in trunk_points.values()])
        plt.scatter(trunk_centers_array[:, 0], trunk_centers_array[:, 1],
                    c='red', s=50, marker='x', label='Trunk Points')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Top View: Point Cloud with Trunk Points')
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def load_point_cloud(file_path):
    """加载 LAS 文件并返回点和颜色。"""
    try:
        las = laspy.read(file_path)
        print(f"Loaded LAS file: {file_path}")
    except Exception as e:
        print(f"Failed to load LAS file: {e}")
        sys.exit(1)

    # 提取点
    points = np.vstack((las.x, las.y, las.z)).transpose()

    # 检查颜色信息
    if ('red' in las.point_format.dimension_names and
            'green' in las.point_format.dimension_names and
            'blue' in las.point_format.dimension_names):
        colors = np.vstack((las.red, las.green, las.blue)).transpose()
        colors = colors / 65535.0  # 归一化到 [0,1]
    else:
        colors = np.ones((points.shape[0], 3))  # 如果没有颜色，默认为白色

    return points, colors


def normalize_z(points):
    """将 Z 轴归一化，使其从 0 开始。"""
    min_z = points[:, 2].min()
    points[:, 2] -= min_z
    return points


def load_trunk_points(trunk_points_path):
    """加载树干点 JSON 文件。"""
    if not os.path.isfile(trunk_points_path):
        print(f"Trunk points file does not exist: {trunk_points_path}")
        sys.exit(1)

    with open(trunk_points_path, 'r') as f:
        trunk_points = json.load(f)

    # 将列表转换回 numpy 数组
    trunk_points = {k: np.array(v) for k, v in trunk_points.items()}

    return trunk_points


if __name__ == "__main__":
    # LAS 文件路径
    las_file_path = 'output/segment_trunk.las'
    # 树干点文件路径
    trunk_points_path = 'output/trunk_points.json'

    # 检查文件是否存在
    if not os.path.isfile(las_file_path):
        print(f"LAS file does not exist: {las_file_path}")
        sys.exit(1)
    if not os.path.isfile(trunk_points_path):
        print(f"Trunk points file does not exist: {trunk_points_path}")
        sys.exit(1)

    # 调用可视化函数
    visualize_trunk_points_with_point_cloud(las_file_path, trunk_points_path)

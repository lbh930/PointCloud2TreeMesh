import os
import sys

import laspy
import matplotlib.pyplot as plt
import numpy as np


def load_segmented_tree(file_path):
    """加载分割后的树干 LAS 文件。"""
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


def print_bounding_box(points):
    """打印点云的边界框。"""
    min_x, min_y, min_z = points.min(axis=0)
    max_x, max_y, max_z = points.max(axis=0)
    print("Bounding Box:")
    print(f"  X: {min_x} to {max_x}")
    print(f"  Y: {min_y} to {max_y}")
    print(f"  Z: {min_z} to {max_z}")


def normalize_z(points):
    """将 Z 轴归一化，使其从 0 开始。"""
    min_z = points[:, 2].min()
    points[:, 2] -= min_z
    print(f"Normalized Z-axis by subtracting {min_z}")
    return points


def create_output_dir(directory):
    """如果输出目录不存在，则创建它。"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")


def slice_and_cluster(points, colors, output_dir, slice_min=0.5, slice_max=15.0, slice_step=0.25,
                      r1=0.5, b=10, T1=0.01, T2=10, T3=0.5):
    """
    沿 Z 轴对点云进行切片，并基于半径 r1 内的点密度对每个切片进行聚类。
    然后比较相邻切片的聚类中心，跟踪树干。
    """
    current_slice = slice_min
    all_cluster_centers = []  # 存储所有聚类中心
    slice_cluster_centers = []  # 存储每个切片的聚类中心列表
    trunk_id_counter = 0
    trunks = {}  # 存储每个树干的聚类中心

    previous_slice_centers = []
    previous_slice_trunk_ids = []

    while current_slice <= slice_max:
        # 获取当前切片内的点（基于 Z 轴高度）
        mask = np.logical_and(points[:, 2] >= current_slice, points[:, 2] < current_slice + slice_step)
        slice_pts = points[mask]
        slice_cols = colors[mask]

        if len(slice_pts) == 0:
            print(f"No points found in slice {current_slice}m to {current_slice + slice_step}m.")
            slice_cluster_centers.append([])
            current_slice += slice_step
            continue

        # 初始化变量
        unclustered_indices = np.arange(len(slice_pts))
        clustered_points = np.zeros(len(slice_pts), dtype=bool)
        clusters = []
        density_threshold = b  # 形成聚类的最小密度（点/立方米）
        volume = np.pi * r1 ** 2 * slice_step  # 半径为 r1、高度为 slice_step 的圆柱体积

        while True:
            if len(unclustered_indices) == 0:
                break

            # 对每个未聚类的点，计算半径 r1 内的邻居数量
            kd_tree = KDTree(slice_pts[unclustered_indices, :2])
            neighbor_indices_list = kd_tree.query_ball_point(slice_pts[unclustered_indices, :2], r=r1)
            neighbor_counts = np.array([len(indices) for indices in neighbor_indices_list])

            # 找到具有最高密度的点
            max_density_index = np.argmax(neighbor_counts)
            max_density = neighbor_counts[max_density_index]
            max_density_point_index = unclustered_indices[max_density_index]

            # 计算实际密度（点/立方米）
            actual_density = max_density / volume

            if actual_density < density_threshold:
                # 没有更多密集区域
                break

            # 初始化组，包含半径 r1 内的所有点
            group_indices = neighbor_indices_list[max_density_index]
            group_point_indices = unclustered_indices[group_indices]

            # 迭代更新组中心，直到收敛
            group_center = slice_pts[group_point_indices].mean(axis=0)
            while True:
                # 找到新组中心半径 r1 内的所有点
                distances = np.linalg.norm(slice_pts[unclustered_indices, :2] - group_center[:2], axis=1)
                new_group_indices = np.where(distances <= r1)[0]
                new_group_point_indices = unclustered_indices[new_group_indices]

                # 计算新组中心
                new_group_center = slice_pts[new_group_point_indices].mean(axis=0)

                # 检查是否收敛
                if np.linalg.norm(new_group_center - group_center) < T1:
                    break
                else:
                    group_center = new_group_center

            # 将这些点标记为已聚类
            clusters.append({
                'center': group_center,
                'indices': group_point_indices
            })
            clustered_points[group_point_indices] = True

            # 更新未聚类的索引
            unclustered_indices = np.where(~clustered_points)[0]

        # 存储当前切片的聚类中心
        centers_in_slice = [cluster['center'] for cluster in clusters]
        slice_cluster_centers.append(centers_in_slice)

        # 跟踪树干
        current_slice_trunk_ids = []
        if not previous_slice_centers:
            # 第一层，初始化树干
            for center in centers_in_slice:
                trunks[trunk_id_counter] = [center]
                current_slice_trunk_ids.append(trunk_id_counter)
                trunk_id_counter += 1
        else:
            # 比较当前切片的聚类中心与上一切片的聚类中心
            for center in centers_in_slice:
                min_tan_theta = None
                assigned_trunk_id = None
                for prev_center, trunk_id in zip(previous_slice_centers, previous_slice_trunk_ids):
                    dx = center[0] - prev_center[0]
                    dy = center[1] - prev_center[1]
                    dz = center[2] - prev_center[2]
                    horizontal_distance = np.sqrt(dx**2 + dy**2)
                    vertical_distance = np.abs(dz)
                    if vertical_distance == 0:
                        continue  # 避免除以零
                    tan_theta = horizontal_distance / vertical_distance  # 计算斜率
                    if tan_theta < T3:
                        if min_tan_theta is None or tan_theta < min_tan_theta:
                            min_tan_theta = tan_theta
                            assigned_trunk_id = trunk_id
                if assigned_trunk_id is not None:
                    # 分配到已有的树干
                    trunks[assigned_trunk_id].append(center)
                    current_slice_trunk_ids.append(assigned_trunk_id)
                else:
                    # 创建新的树干
                    trunks[trunk_id_counter] = [center]
                    current_slice_trunk_ids.append(trunk_id_counter)
                    trunk_id_counter += 1


        # 更新 previous_slice_centers 和 previous_slice_trunk_ids
        previous_slice_centers = centers_in_slice
        previous_slice_trunk_ids = current_slice_trunk_ids

        # 对当前切片进行可视化
        plt.figure(figsize=(10, 8))
        plt.scatter(slice_pts[:, 0], slice_pts[:, 1], c='lightgrey', s=5, label='Points')

        # 绘制聚类结果
        colors_cycle = plt.cm.rainbow(np.linspace(0, 1, max(len(clusters), 1)))
        for idx, cluster in enumerate(clusters):
            cluster_pts = slice_pts[cluster['indices']]
            plt.scatter(cluster_pts[:, 0], cluster_pts[:, 1], color=colors_cycle[idx % len(colors_cycle)], s=10)
            plt.plot(cluster['center'][0], cluster['center'][1], 'o', color='k', markersize=12)

        plt.title(f'Slice Z: {current_slice}m to {current_slice + slice_step}m - {len(clusters)} clusters')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')  # 保持长宽比例
        plt.tight_layout()

        # 保存绘图
        filename = f'slice_{current_slice}_{current_slice + slice_step}_density_clusters.png'
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved density-based clustered slice image: {save_path}")

        current_slice += slice_step

    # 在 3D 中可视化所有树干
    visualize_trunks(points, colors, trunks, output_dir)


def visualize_trunks(points, colors, trunks, output_dir):
    """可视化树干并与原始点云对比。"""
    from mpl_toolkits.mplot3d import Axes3D

    # 绘制原始点云
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=1, label='Point Cloud')

    # 绘制每个树干
    colors_cycle = plt.cm.rainbow(np.linspace(0, 1, max(len(trunks), 1)))
    for idx, (trunk_id, centers) in enumerate(trunks.items()):
        centers = np.array(centers)
        ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], '-o', color=colors_cycle[idx % len(colors_cycle)],
                markersize=5, label=f'Trunk {trunk_id}')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Trunks and Point Cloud')
    ax.legend()
    plt.tight_layout()

    # 保存可视化结果
    save_path = os.path.join(output_dir, 'trunks.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved trunks visualization: {save_path}")


def main():
    # 分割后的树干 LAS 文件路径
    input_las = 'output/segment_trunk.las'

    if not os.path.isfile(input_las):
        print(f"Segmented tree LAS file does not exist: {input_las}")
        sys.exit(1)

    # 加载数据
    points, colors = load_segmented_tree(input_las)

    # 检查边界框
    print_bounding_box(points)

    # 归一化 Z 轴
    points = normalize_z(points)

    # 切片图像的输出目录
    output_dir = 'output/slice_vis'
    create_output_dir(output_dir)

    # 切片参数
    slice_min = 3.0
    slice_max = 6
    slice_step = 0.1

    # 聚类参数
    r1 = 2  # 密度计算的半径（单位：米）
    b = 32    # 密度阈值（点/立方米）
    T1 = 0.01 # 质心收敛的阈值
    T3 = 0.5  # 相邻切片聚类中心距离的阈值

    # 进行切片、聚类并保存图像
    slice_and_cluster(points, colors, output_dir, slice_min, slice_max, slice_step, r1, b, T1, b, T3)

    print("Slicing, clustering, and mapping completed.")


if __name__ == "__main__":
    from scipy.spatial import KDTree
    main()
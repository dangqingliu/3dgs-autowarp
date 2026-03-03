"""
可视化 3D Gaussian Splatting PLY 文件。
使用 Open3D 将高斯中心显示为带颜色的点云。
用法: python visualize_3dgs_ply.py path/to/point_cloud.ply
"""
import argparse
import numpy as np


def visualize_3dgs_ply(
    ply_path: str,
    window_name: str = "3DGS PLY",
    show_axes: bool = True,
) -> None:
    """
    加载并可视化 3DGS 格式的 PLY 文件（将高斯中心渲染为彩色点云）。

    Args:
        ply_path: PLY 文件路径
        window_name: 显示窗口标题
        show_axes: 是否显示 XYZ 坐标轴（OpenGL/Apple 系：X红 Y绿 Z蓝）
    """
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError("请先安装: pip install open3d")

    try:
        from plyfile import PlyData
    except ImportError: 
        raise ImportError("请先安装: pip install plyfile")

    ply = PlyData.read(ply_path)
    vertex = ply["vertex"]

    x = np.asarray(vertex["x"], dtype=np.float64)
    y = np.asarray(vertex["y"], dtype=np.float64)
    z = np.asarray(vertex["z"], dtype=np.float64)
    points = np.stack([x, y, z], axis=1)
    points[:, 1] = -points[:, 1]  # 沿 XZ 平面翻转
    points[:, 2] = -points[:, 2]  # 沿 XY 平面翻转

    # 3DGS 用球谐 DC 分量存颜色: f_dc_0, f_dc_1, f_dc_2 -> R, G, B
    # 公式: RGB = 0.5 + C0 * (f_dc_0, f_dc_1, f_dc_2), C0 ≈ 0.282
    C0 = 0.28209479177387814
    if "f_dc_0" in vertex.data.dtype.names:
        r = 0.5 + C0 * np.asarray(vertex["f_dc_0"], dtype=np.float64)
        g = 0.5 + C0 * np.asarray(vertex["f_dc_1"], dtype=np.float64)
        b = 0.5 + C0 * np.asarray(vertex["f_dc_2"], dtype=np.float64)
        colors = np.stack([r, g, b], axis=1).astype(np.float64)
        np.clip(colors, 0.0, 1.0, out=colors)
    else:
        # 若无颜色属性，使用统一灰色
        colors = np.ones_like(points) * 0.7

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 加强渲染：全点云、大点径、高分辨率
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1280, height=960)
    vis.add_geometry(pcd)
    if show_axes:
        # OpenGL/Apple 右手系：X=右(红) Y=上(绿) Z=朝向观察者(蓝)
        extent = np.ptp(points, axis=0)
        axis_size = float(np.max(extent)) * 0.05 if np.max(extent) > 0 else 1.0
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=axis_size, origin=[0, 0, 0]
        )
        vis.add_geometry(coord_frame)
    ro = vis.get_render_option()
    ro.point_size = 2.0
    ro.background_color = np.array([0.1, 0.1, 0.12])
    ro.show_coordinate_frame = False
    vis.run()
    vis.destroy_window()


def load_3dgs_ply_as_pointcloud(ply_path: str):
    """
    仅加载 3DGS PLY 为 Open3D 点云，不弹窗。便于在 notebook 中配合其他方式展示。

    Args:
        ply_path: PLY 文件路径

    Returns:
        open3d.geometry.PointCloud
    """
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError("请先安装: pip install open3d")
    try:
        from plyfile import PlyData
    except ImportError:
        raise ImportError("请先安装: pip install plyfile")

    ply = PlyData.read(ply_path)
    vertex = ply["vertex"]

    x = np.asarray(vertex["x"], dtype=np.float64)
    y = np.asarray(vertex["y"], dtype=np.float64)
    z = np.asarray(vertex["z"], dtype=np.float64)
    points = np.stack([x, y, z], axis=1)
    # points[:, 2] = -points[:, 2]  # 沿 XY 平面翻转，需要时取消注释

    C0 = 0.28209479177387814
    if "f_dc_0" in vertex.data.dtype.names:
        r = 0.5 + C0 * np.asarray(vertex["f_dc_0"], dtype=np.float64)
        g = 0.5 + C0 * np.asarray(vertex["f_dc_1"], dtype=np.float64)
        b = 0.5 + C0 * np.asarray(vertex["f_dc_2"], dtype=np.float64)
        colors = np.stack([r, g, b], axis=1).astype(np.float64)
        np.clip(colors, 0.0, 1.0, out=colors)
    else:
        colors = np.ones_like(points) * 0.7

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="可视化 3DGS PLY 文件")
    parser.add_argument("ply_path", type=str, help="PLY 文件路径")
    parser.add_argument("--title", "-t", type=str, default="3DGS PLY", help="窗口标题")
    parser.add_argument("--no-axes", action="store_true", help="不显示 XYZ 坐标轴")
    args = parser.parse_args()
    visualize_3dgs_ply(
        args.ply_path,
        window_name=args.title,
        show_axes=not args.no_axes,
    )

# python visualize_3dgs_ply.py test.ply
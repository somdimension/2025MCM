import math
import numpy as np

def calculate_shadow_vertices(sunlight_dir, mirror_normal, L, W, H):
    """
    计算太阳光照射到地面上的反射镜后，在地面上投射的阴影的四个顶点坐标。

    参数:
        sunlight_dir (tuple): 太阳光方向向量 (dx, dy, dz)
        mirror_normal (tuple): 镜面法向量 (nx, ny, nz)
        L (float): 反射镜的长度（沿局部x轴方向）
        W (float): 反射镜的宽度（沿局部y轴方向）
        H (float): 安装高度（z坐标）

    返回:
        list of tuples: 四个阴影顶点的坐标（x, y）
    """
    # 将输入向量转换为单位向量
    sun_dx, sun_dy, sun_dz = [v/np.linalg.norm(sunlight_dir) for v in sunlight_dir]
    nx, ny, nz = [v/np.linalg.norm(mirror_normal) for v in mirror_normal]

    # 计算镜面坐标系（局部坐标系）
    # 局部z轴为镜面法线方向
    local_z = np.array([nx, ny, nz])
    
    # 创建局部x轴（假设初始方向为全局x轴投影到镜面）
    global_x_proj = np.array([1, 0, 0]) - np.dot([1, 0, 0], local_z) * local_z
    if np.linalg.norm(global_x_proj) < 1e-6:
        global_x_proj = np.array([0, 1, 0]) - np.dot([0, 1, 0], local_z) * local_z
    local_x = global_x_proj / np.linalg.norm(global_x_proj)
    
    # 创建局部y轴（通过叉乘）
    local_y = np.cross(local_z, local_x)

    # 计算镜面四个顶点在全局坐标系中的坐标
    vertices_local = [
        (-L/2, -W/2, 0),
        (L/2, -W/2, 0),
        (L/2, W/2, 0),
        (-L/2, W/2, 0)
    ]

    # 转换到全局坐标系并抬升到高度H
    vertices_global = []
    for v in vertices_local:
        # 局部坐标到全局坐标的转换：中心在 (0, 0, H)
        global_v = np.array([0, 0, H]) + v[0]*local_x + v[1]*local_y + v[2]*local_z
        vertices_global.append((global_v[0], global_v[1], global_v[2]))

    def project_point(x, y, z):
        """将三维点投影到地面（z=0）"""
        if math.isclose(sun_dz, 0, abs_tol=1e-9):
            raise ValueError("太阳光方向水平，无法计算阴影。")
        
        t = z / sun_dz
        return (x - t*sun_dx, y - t*sun_dy)

    # 投影四个顶点
    print([(f"{v[0]:.4f}", f"{v[1]:.4f}", f"{v[2]:.4f}") for v in vertices_global])
    return [project_point(*v) for v in vertices_global]
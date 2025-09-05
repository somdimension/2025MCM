import pandas as pd

import numpy as np
from scipy.optimize import root_scalar

def 读取excel(file_path):
    """
    读取CSV文件的前两列数据
    :param file_path: CSV文件路径
    :return: 包含前两列数据的DataFrame
    """

    # 读取CSV文件，只取前两列
    df = pd.read_csv(file_path, usecols=[0, 1])
    return df
# if __name__ == "__main__":
#     # 指定CSV文件路径
#     csv_file_path = r"C:\Users\czw17\Desktop\附件1.csv"
    
#     # 读取前两列数据
#     data = 读取excel(csv_file_path)
    
#     if data is not None:
#         print("CSV文件的前两列数据:")
#         print(data)



def calculate_refractive_index(reflectance, angle_deg, n1=1.0):
    """
    根据给定的反射率和入射角，使用菲涅尔方程计算材料的折射率 (n2)。

    此函数假设入射光为非偏振光。

    参数:
    reflectance (float): 测量得到的总反射率 (一个 0 到 1 之间的值)。
    angle_deg (float): 光线的入射角 (单位: 度)。
    n1 (float, optional): 入射介质的折射率。默认为 1.0 (空气)。

    返回:
    float: 计算得到的介质折射率 (n2)。
    
    可能引发的异常:
    ValueError: 如果输入值无效或在物理上不可能找到解。
    """

    # --- 1. 输入验证 ---
    if not 0 < reflectance < 1:
        raise ValueError("反射率必须在 (0, 1) 范围内。")
    if not 0 <= angle_deg < 90:
        raise ValueError("入射角必须在 [0, 90) 范围内。")

    # --- 2. 将角度转换为弧度 ---
    theta_i = np.deg2rad(angle_deg)

    # --- 3. 定义需要求解的菲涅尔方程 ---
    # 我们需要找到一个 n2 值，使得根据菲涅尔公式计算出的理论反射率
    # 与输入的测量反射率之间的差值为零。
    def fresnel_equation_for_solver(n2, R_measured, n1, theta_i):
        # 避免n2小于n1*sin(theta_i)导致物理上无解（全反射条件）
        if n2 <= n1 * np.sin(theta_i):
            # 返回一个很大的数，表示这个解是无效的
            return 1e9

        # 根据斯涅尔定律计算 cos(theta_t)
        # n1*sin(theta_i) = n2*sin(theta_t)
        # cos(theta_t) = sqrt(1 - sin^2(theta_t))
        cos_theta_t = np.sqrt(1 - (n1 * np.sin(theta_i) / n2)**2)
        cos_theta_i = np.cos(theta_i)

        # s-偏振反射率
        Rs_num = n1 * cos_theta_i - n2 * cos_theta_t
        Rs_den = n1 * cos_theta_i + n2 * cos_theta_t
        Rs = (Rs_num / Rs_den)**2

        # p-偏振反射率
        Rp_num = n2 * cos_theta_i - n1 * cos_theta_t
        Rp_den = n2 * cos_theta_i + n1 * cos_theta_t
        Rp = (Rp_num / Rp_den)**2
        
        # 非偏振光的理论总反射率
        R_calculated = (Rs + Rp) / 2
        
        # 返回理论值与测量值之差
        return R_calculated - R_measured

    # --- 4. 使用数值求解器寻找方程的根 ---
    # 我们在一个合理的折射率范围内 [1.0, 4.0] 寻找解。
    # 大多数常见材料的折射率都在这个范围内。
    try:
        # root_scalar 会寻找使 fresnel_equation_for_solver 函数结果为 0 的 n2 值
        solution = root_scalar(
            fresnel_equation_for_solver,
            args=(reflectance, n1, theta_i),
            bracket=[1.0, 4.0],  # 设置求解区间
            method='brentq'     # 一种高效且稳定的求解算法
        )
        
        if solution.converged:
            return solution.root
        else:
            raise ValueError("无法找到收敛的解。请检查输入的反射率和角度是否物理上可能。")

    except ValueError as e:
        # 如果求解器在初始区间找不到符号变化，会抛出 ValueError
        raise ValueError(f"求解失败: {e}。这可能意味着对于给定的输入，在[1.0, 4.0]范围内不存在有效折射率解。")

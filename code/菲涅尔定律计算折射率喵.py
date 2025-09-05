import pandas as pd
import numpy as np
from scipy.optimize import root_scalar
plt.rcParams['font.family'] = 'SimHei'
def 读取excel(file_path):
    """
    读取 CSV 文件的前两列数据
    """
    df = pd.read_csv(file_path, usecols=[0, 1])
    return df

def calculate_refractive_index(reflectance, angle_deg=10, n1=1.0):
    """
    使用菲涅尔公式计算折射率。
    :param reflectance: 0-1 之间的小数
    """
    if not 0 < reflectance < 1:
        raise ValueError("反射率必须在 (0, 1) 范围内。")
    if not 0 <= angle_deg < 90:
        raise ValueError("入射角必须在 [0, 90) 范围内。")

    theta_i = np.deg2rad(angle_deg)

    def eq(n2, R_meas, n1, theta_i):
        if n2 <= n1 * np.sin(theta_i):
            return 1e9
        cos_t = np.sqrt(1 - (n1 * np.sin(theta_i) / n2)**2)
        cos_i = np.cos(theta_i)
        Rs = ((n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t))**2
        Rp = ((n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t))**2
        return (Rs + Rp) / 2 - R_meas

    sol = root_scalar(eq,
                      args=(reflectance, n1, theta_i),
                      bracket=[1.0, 4.0],
                      method='brentq')
    if sol.converged:
        return sol.root
    else:
        raise ValueError("无法找到收敛解，请检查输入值。")

def process_and_save(input_path, output_path, angle_deg=10, n1=1.0):
    """
    读取 CSV，计算折射率，输出调试信息，保存新文件。
    """
    df = 读取excel(input_path)
    refractive_indices = []

    for i, R_percent in enumerate(df.iloc[:, 1]):
        R = R_percent / 100.0  # 百分制转小数
        try:
            n2 = calculate_refractive_index(R, angle_deg, n1)
        except Exception as e:
            n2 = np.nan
            print(f"第{i}行 转换后 R={R:.4f} 计算出错: {e}")
        refractive_indices.append(n2)

        # 调试输出前5行
        if i < 5:
            print(f"第{i}行: 原始R%={R_percent} → R={R:.4f}, 折射率={n2}")

    df['RefractiveIndex'] = refractive_indices
    df.to_csv(output_path, index=False)
    return df

if __name__ == "__main__":
    输入文件 = r"C:\Users\czw17\Desktop\反射率处理数据.csv"
    输出文件 = r"C:\Users\czw17\Desktop\输出结果.csv"
    result_df = process_and_save(输入文件, 输出文件, angle_deg=10, n1=1.0)
    print("处理完成，结果预览：")
    print(result_df.head())

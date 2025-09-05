import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.metrics import r2_score

# 1. 定义读取 CSV 的函数
def 读取excel(file_path):
    """
    读取 CSV 文件的前两列数据
    参数:
        file_path (str): CSV 文件路径
    返回:
        pandas.DataFrame: 包含第一列和第二列（x, y）的数据
    """
    df = pd.read_csv(file_path, usecols=[0, 2])
    df.columns = ['x', 'y']  # 重命名列
    return df

# 2. Sellmeier 折射率计算函数
def calculate_sic_refractive_index(wavenumber_cm):
    """
    根据给定的 Sellmeier 公式计算 SiC 的折射率
    参数:
        wavenumber_cm (array-like): 波数，单位为 cm⁻¹
    返回:
        numpy.ndarray: 对应的折射率
    """
    # 波数 (cm⁻¹) 转波长 (μm): λ = 1e4 / ν
    lambda_um = 1e4 / wavenumber_cm
    lambda_sq = lambda_um**2

    # Sellmeier 常数
    B1 = 5.58245
    C1_sq = 0.1625394**2
    B2 = 2.468516
    C2_sq = 11.35656**2

    # 计算 n^2
    term1 = (B1 * lambda_sq) / (lambda_sq - C1_sq)
    term2 = (B2 * lambda_sq) / (lambda_sq - C2_sq)
    n_squared = 1 + term1 + term2

    # 对负值自动返回 NaN
    with np.errstate(invalid='ignore'):
        n = np.sqrt(n_squared)
    return n

# 3. 主运行逻辑
if __name__ == '__main__':
    # 3.1 读取 CSV 数据
    file_path = r'C:\Users\czw17\Desktop\输出结果.csv'
    df = 读取excel(file_path)
    x_csv = df['x'].values
    y_csv = df['y'].values

    # 3.2 生成标准 Sellmeier 曲线
    #    注意原程序在奇点处分两个区间，这里直接使用整个区间或分段均可
    wavenumber = np.linspace(min(x_csv), max(x_csv), 2000)
    n_standard = calculate_sic_refractive_index(wavenumber)

    # 3.3 在 CSV x 点处插值计算对应的 Sellmeier 曲线值
    interp_func = interp1d(wavenumber, n_standard, 
                           bounds_error=False, fill_value='extrapolate')
    n_interp = interp_func(x_csv)

    # 3.4 绘图对比
    plt.figure(figsize=(10, 6))
    plt.plot(wavenumber, n_standard, color='royalblue', linewidth=2, label='Sellmeier 计算曲线')
    plt.plot(x_csv, y_csv, 'o', color='orange', label='CSV 数据点')
    plt.plot(x_csv, n_interp, '--', color='green', label='插值后的 Sellmeier 曲线')
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Refractive Index (n)')
    plt.title('SiC 折射率曲线对比')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 3.5 计算 R²（评估 CSV 曲线与 Sellmeier 曲线插值结果的相似性）
    r2 = r2_score(y_csv, n_interp)
    print(f'R² = {r2:.6f}')

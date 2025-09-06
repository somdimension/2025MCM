import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.metrics import r2_score
plt.rcParams['font.family'] = 'SimHei'
# 读取 CSV 的函数（读取第1、3列）
def 读取excel(file_path):
    """
    读取 CSV 文件的第1和第3列数据
    返回 DataFrame，列名为 ['x','y']
    """
    df = pd.read_csv(file_path, usecols=[0, 2])
    df.columns = ['x', 'y']
    return df

# Sellmeier 折射率计算
def calculate_sic_refractive_index(wavenumber_cm):
    lambda_um = 1e4 / wavenumber_cm
    lambda_sq = lambda_um**2
    B1, B2 = 5.58245, 2.468516
    C1_sq, C2_sq = 0.1625394**2, 11.35656**2
    term1 = (B1 * lambda_sq) / (lambda_sq - C1_sq)
    term2 = (B2 * lambda_sq) / (lambda_sq - C2_sq)
    n2 = 1 + term1 + term2
    with np.errstate(invalid='ignore'):
        return np.sqrt(n2)

if __name__ == '__main__':
    # 1. 读取用户 CSV
    file_path = r'C:\Users\czw17\Desktop\新输出结果2喵.csv'
    df = 读取excel(file_path)
    x_csv, y_csv = df['x'].values, df['y'].values

    # 2. 生成 Sellmeier 曲线
    wv = np.linspace(x_csv.min(), x_csv.max(), 3000)
    n_std = calculate_sic_refractive_index(wv)

    # 3. 插值到 CSV x 点
    interp = interp1d(wv, n_std, bounds_error=False, fill_value='extrapolate')
    n_interp = interp(x_csv)

    # 4. 计算两条曲线数值级相似度（R²）和趋势级相似度
    r2_value = r2_score(y_csv, n_interp)
    # 一阶差分（趋势） —— 反映曲线走向
    diff_csv   = np.diff(y_csv)
    diff_interp= np.diff(n_interp)
    # 如果长度不一致，取最小长度
    L = min(len(diff_csv), len(diff_interp))
    r2_trend = r2_score(diff_csv[:L], diff_interp[:L])

    # 5. 绘图对比
    plt.figure(figsize=(10,6))
    plt.plot(wv, n_std,   '-', color='royalblue', label='Sellmeier公式计算曲线')
    plt.plot(x_csv, y_csv,'o', color='orange',   label='CSV 数据点')
    plt.xlabel('波数 (/cm)')
    plt.ylabel('折射率')
    plt.title('SiC 折射率曲线对比')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 6. 输出 R² 指标
    print(f'整体 R²（数值相似度）: {r2_value:.6f}')
    print(f'趋势 R²（一阶差分走向相似度）: {r2_trend:.6f}')

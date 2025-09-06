import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 1. 读取数据
input_path = r"C:\Users\czw17\Desktop\附件3.csv"

try:
    df = pd.read_csv(input_path, header=0,names=['Wavenumber', 'Reflectance'])
    sigma = df['Wavenumber'].values  # 波数 (cm^-1)
    R_meas = df['Reflectance'].values / 100.0  # 反射率 (转换为小数)
     # 使用索引作为x轴（若“自变量”是时间/位置，可替换）
except FileNotFoundError:
    print(f"❌ 错误：找不到文件，请检查路径是否正确: {input_path}")
    exit()
except Exception as e:
    print(f"❌ 数据加载出错: {e}")
    exit()
# 请将 '附件3.xlsx' 替换为您的实际文件路径
# df = pd.read_excel('附件3.xlsx', header=None, names=['Wavenumber', 'Reflectance'])
# sigma = df['Wavenumber'].values  # 波数 (cm^-1)
# R_meas = df['Reflectance'].values / 100.0  # 反射率 (转换为小数)

# 2. 定义物理模型函数
# 参数: p = [A, B, d]
# 其中: n(sigma) = A + B * sigma^2
#       delta = 4 * pi * sigma * n(sigma) * d (小角度近似)
#       r1 = |(1 - n) / (1 + n)|
#       r2 = |(n - n_s) / (n + n_s)|, n_s = 3.4 (硅衬底)
def multibeam_interference(sigma, A, B, d):
    n_s = 3.4  # 硅衬底折射率
    n_film = A + B * (sigma ** 2)  # 薄膜折射率 (Cauchy模型)
    
    # 计算两个界面的反射系数 (振幅反射率)
    r1 = np.abs((1 - n_film) / (1 + n_film))
    r2 = np.abs((n_film - n_s) / (n_film + n_s))
    
    # 计算相位差 (小角度近似, cos(theta)≈1)
    delta = 4 * np.pi * sigma * n_film * d
    
    # 多光束干涉反射率公式
    numerator = r1**2 + r2**2 + 2 * r1 * r2 * np.cos(delta)
    denominator = 1 + (r1 * r2)**2 + 2 * r1 * r2 * np.cos(delta)
    R = numerator / denominator
    
    return R

# 3. 设置初始参数猜测值
# 根据对硅外延层的物理认知和数据观察
A_guess = 3.5      # 折射率常数项
B_guess = 1e-7     # 色散系数 (小正数)
d_guess = 5e-4     # 厚度 (cm), 即 5 微米 (这是一个典型猜测值)

p0 = [A_guess, B_guess, d_guess]

# 4. 进行非线性拟合
try:
    popt, pcov = curve_fit(multibeam_interference, sigma, R_meas, p0=p0, maxfev=5000)
    A_fit, B_fit, d_fit = popt
    print(f"拟合参数:")
    print(f"  A (n0) = {A_fit:.6f}")
    print(f"  B (dispersion) = {B_fit:.6e}")
    print(f"  d (thickness) = {d_fit*1e4:.4f} 微米") # 转换为微米
    print(f"  (对应物理厚度: {d_fit:.6f} cm)")
    
    # 计算拟合值
    R_fit = multibeam_interference(sigma, A_fit, B_fit, d_fit)
    
    # 5. 绘图
    plt.figure(figsize=(12, 8))
    plt.plot(sigma, R_meas * 100, 'o', label='实验数据 (附件3)', markersize=3, alpha=0.7)
    plt.plot(sigma, R_fit * 100, '-', linewidth=2, label='多光束干涉模型拟合')
    plt.xlabel('波数 (cm$^{-1}$)', fontsize=12)
    plt.ylabel('反射率 (%)', fontsize=12)
    plt.title('硅外延层多光束干涉光谱拟合 (入射角 10°)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    
    # 6. 绘制残差图
    residuals = (R_meas - R_fit) * 100 # 转换为百分比单位的残差
    plt.figure(figsize=(12, 4))
    plt.plot(sigma, residuals, 'o-', markersize=3, linewidth=1)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('波数 (cm$^{-1}$)', fontsize=12)
    plt.ylabel('残差 (%)', fontsize=12)
    plt.title('拟合残差', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"拟合失败: {e}")
    print("可能原因: 初始值不佳、模型过于复杂、数据噪声过大等。")
    # 即使拟合失败，也绘制原始数据
    plt.figure(figsize=(12, 8))
    plt.plot(sigma, R_meas * 100, 'o', label='实验数据 (附件3)', markersize=3)
    plt.xlabel('波数 (cm$^{-1}$)', fontsize=12)
    plt.ylabel('反射率 (%)', fontsize=12)
    plt.title('硅外延层红外干涉光谱 (入射角 10°)', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()
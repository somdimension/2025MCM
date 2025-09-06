import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# 重新读取csv文件（假设前面成功读取）
input_path = r"C:\Users\czw17\Desktop\附件3.csv"

try:
    data = pd.read_csv(input_path, header=0)
    data.columns = ['波数 (cm-1)', '反射率 (%)']
    time_series = data['反射率 (%)'].values  # 转为numpy数组便于计算
    x = np.arange(len(time_series))   # 使用索引作为x轴（若“自变量”是时间/位置，可替换）
    print(f"✅ 数据加载成功，共 {len(time_series)} 个点")
except FileNotFoundError:
    print(f"❌ 错误：找不到文件，请检查路径是否正确: {input_path}")
    exit()
except Exception as e:
    print(f"❌ 数据加载出错: {e}")
    exit()
wave_number = data.iloc[:, 0].values  # 波数 (cm-1)
reflectance = data.iloc[:, 1].values / 100.0  # 反射率(转换为比例)

# 多光束干涉模型定义
# 折射率n随波数w变化，假设线性函数
# 振幅随波数指数衰减

def n_model(w, n0, k):
    return n0 - k * w

def amplitude_model(w, a0, a1):
    return a0 * np.exp(-a1 * w)

# 全模型，波长为lambda = 1e4 / w
# R = A(w) * (1 + cos(2pi * n(w) * d / lambda)) + b

def reflec_model(w, n0, k, a0, a1, d, b):
    lambda_ = 1e4 / w
    n_w = n_model(w, n0, k)
    A_w = amplitude_model(w, a0, a1)
    return A_w * (1 + np.cos(2 * np.pi * n_w * d / lambda_)) + b

# 使用curve_fit拟合数据
initial_params = [3.5, 0.001, 0.2, 0.0005, 20, 0.1]
popt, pcov = curve_fit(reflec_model, wave_number, reflectance, p0=initial_params, maxfev=10000)

# 拟合结果
popt

# 生成拟合曲线数据
reflectance_fit = reflec_model(wave_number, *popt)

# 绘制原始数据和拟合曲线
plt.figure(figsize=(10, 6))
plt.plot(wave_number, reflectance, 'b-', label='原始数据', alpha=0.7)
plt.plot(wave_number, reflectance_fit, 'r--', label='拟合曲线', linewidth=2)
plt.xlabel('波数 (cm⁻¹)')
plt.ylabel('反射率')
plt.title('多光束干涉模型拟合结果')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
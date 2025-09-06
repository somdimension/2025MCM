import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

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
# 读取附件3数据

# 附件3数据范围约3000-3700 cm^-1，切片数据用于拟合
data = data[(data['波数 (cm-1)'] >= 500) & (data['波数 (cm-1)'] <= 2000)]

# 定义拟合函数
# 参数为波数sigma，参数A, B, d
def reflectance_func(sigma, A, B, d):
    n = A + B * sigma**2
    r1 = np.abs((1 - n) / (1 + n))
    r2 = np.abs((n - 3.4) / (n + 3.4))
    delta = 4 * np.pi * sigma * n * d
    numerator = r1**2 + r2**2 + 2 * r1 * r2 * np.cos(delta)
    denominator = 1 + (r1**2) * (r2**2) + 2 * r1 * r2 * np.cos(delta)
    return numerator / denominator * 100  # 反射率单位为%

# 取波数和反射率数据
xdata = data['波数 (cm-1)'].values
ydata = data['反射率 (%)'].values

# 初始参数猜测
initial_guess = [3.4, -1e-7, 0.002]

# 曲线拟合
params, params_covariance = curve_fit(reflectance_func, xdata, ydata, p0=initial_guess, maxfev=20000)

# 拟合参数
A_fit, B_fit, d_fit = params

# 计算拟合值
y_fit = reflectance_func(xdata, *params)

# 绘图展示拟合效果
plt.figure(figsize=(10, 6))
plt.plot(xdata, ydata, label='Experimental data')
plt.plot(xdata, y_fit, label='Fitted curve', linestyle='--')
plt.xlabel('波数 (cm⁻¹)')
plt.ylabel('反射率 (%)')
plt.title('多光束干涉模型拟合曲线 - 附件3数据')
plt.legend()
plt.show()
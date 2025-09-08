import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

input_path = r"C:\Users\czw17\Desktop\扰动信号数据2.csv"

try:
    df = pd.read_csv(input_path, header = 0)#有headers
    df.columns = ['自变量', '因变量']
    time_series = df['因变量'].values
    x_values = df['自变量'].values
except FileNotFoundError:
    print(f"错误：找不到文件，请检查路径是否正确: {input_path}")
    exit()

# 数据平滑处理
window_length = 51  # 窗口长度，必须为奇数
polyorder = 3       # 多项式阶数
smoothed_data = savgol_filter(time_series, window_length, polyorder)

# 计算一阶导数
first_derivative = savgol_filter(time_series, window_length, polyorder, deriv=1, delta=1.0)

# 计算二阶导数
second_derivative = savgol_filter(time_series, window_length, polyorder, deriv=2, delta=1.0)

# 计算二阶导数关于x轴的对称性
# 对称性分析：计算二阶导数与其关于x轴对称的差异
second_derivative_symmetric = -second_derivative
# 计算对称性度量（均方根误差）
symmetry_metric = np.sqrt(np.mean((second_derivative - second_derivative_symmetric)**2))

# 可视化结果
plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
plt.plot(x_values, time_series, label='原始数据')
plt.plot(x_values, smoothed_data, label='平滑数据')
plt.legend()
plt.title('原始数据与平滑数据')

plt.subplot(4, 1, 2)
plt.plot(x_values, second_derivative, label='二阶导数', color='green')
plt.legend()
plt.title('二阶导数')

plt.legend()
plt.title(f'二阶导数及其关于x轴的对称性 (对称性度量: {symmetry_metric:.4f})')

plt.tight_layout()
plt.show()

# 输出对称性分析结果
print(f"二阶导数关于x轴的对称性度量: {symmetry_metric:.4f}")
if symmetry_metric < 0.1:
    print("二阶导数具有良好的关于x轴的对称性")
elif symmetry_metric < 0.5:
    print("二阶导数具有中等程度的关于x轴的对称性")
else:
    print("二阶导数关于x轴的对称性较差")
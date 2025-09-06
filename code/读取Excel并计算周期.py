import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# 1. 读取 Excel 文件
file_path = r'C:\Users\czw17\Desktop\附件1_STL分解结果_自动周期.xlsx'
data = pd.read_excel(file_path)

# 2. 提取第一列（自变量 x）和第三列（信号 y）
x = data.iloc[:, 0]          # 第一列
signal = data.iloc[:, 3]     # 第三列

# 3. 对信号进行高斯平滑处理，过滤噪声
sigma = 40                   # 高斯滤波标准差，可根据噪声情况调整
smoothed_signal = gaussian_filter1d(signal, sigma)

# 4. 检测平滑后信号的波峰
#    这里的 find_peaks 会检测所有局部峰值，若只需关注最大波动趋势，可通过‘height’或‘prominence’参数筛选
peaks, properties = find_peaks(
    smoothed_signal,
    prominence=(None, None),  # 可设定最小 prominence 过滤微小波峰
    distance=1                # 波峰间最小间隔，可根据 x 的单位调整
)

# 如果噪声仍多，可进一步过滤：
# peaks, properties = find_peaks(
#     smoothed_signal,
#     prominence=0.5 * np.max(properties['prominences'])  # 只保留较高显著波峰
# )

# 5. 计算相邻波峰在 x 轴上的距离，并求平均值（周期）
peak_distances = np.diff(x.iloc[peaks])
mean_period = np.mean(peak_distances)

# 6. 绘图展示
plt.figure(figsize=(12, 6))
plt.plot(x, signal, label='原始信号', alpha=0.5)
plt.plot(x, smoothed_signal, label='高斯平滑信号', linewidth=2)
plt.plot(x.iloc[peaks], smoothed_signal[peaks], 'ro', label='检测到的波峰')
plt.xlabel('自变量 x (第一列)')
plt.ylabel('信号 y (第三列)')
plt.title(f'信号高斯平滑与波峰检测 — 平均周期: {mean_period:.3f}')
plt.legend()
plt.tight_layout()
plt.show()

print(f'检测到 {len(peaks)} 个主要波峰')
print(f'平均周期: {mean_period:.3f}')
# 已获取波峰索引 peaks 和自变量 x

# 计算相邻波峰之间的距离
peak_distances = np.diff(x.iloc[peaks])

# 打印每个波峰之间的距离
print("每个波峰之间的距离（周期）：")
for i, d in enumerate(peak_distances, start=1):
    print(f"峰值 {i} 到 峰值 {i+1} 的距离: {d:.3f}")

# 打印平均周期
mean_period = np.mean(peak_distances)
print(f"\n平均周期: {mean_period:.3f}")

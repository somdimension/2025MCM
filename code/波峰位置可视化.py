import matplotlib.pyplot as plt
import numpy as np

# 波峰位置数据
peak_positions = [417.03, 749.69, 1107.42, 1510.95, 1936.66, 2378.28, 2781.33, 3208.49]
peak_indices = range(1, len(peak_positions) + 1)

# 创建图表
plt.figure(figsize=(10, 6))
plt.plot(peak_indices, peak_positions, 'bo-', linewidth=2, markersize=8)
plt.xlabel('波峰序号')
plt.ylabel('波数 (cm⁻¹)')
plt.title('波峰位置分布图')
plt.grid(True, alpha=0.3)

# 添加数值标签
for i, (idx, pos) in enumerate(zip(peak_indices, peak_positions)):
    plt.annotate(f'{pos:.2f}', (idx, pos), textcoords="offset points", xytext=(0,10), ha='center')

plt.tight_layout()
plt.show()

# 计算相邻波峰之间的间距
spacings = np.diff(peak_positions)
print("相邻波峰之间的间距:")
for i, spacing in enumerate(spacings):
    print(f"波峰 {i+1} 到波峰 {i+2}: {spacing:.2f} cm⁻¹")

# 计算平均间距
avg_spacing = np.mean(spacings)
print(f"\n平均间距: {avg_spacing:.2f} cm⁻¹")
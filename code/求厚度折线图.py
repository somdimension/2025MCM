import boss二阶段
peaks=[ 223,  742, 1261, 1774, 2300, 2828, 3358, 3891, 4427, 4967]

# 计算每两个相邻peak之间的中点值
midpoints = []
for i in range(len(peaks) - 1):
    midpoint = (peaks[i] + peaks[i+1]) / 2
    midpoints.append(midpoint)

print("相邻peak之间的中点值列表：", midpoints)

import matplotlib.pyplot as plt

# 用户指定的周期值列表
周期列表 = [250.219, 250.218, 247.326, 253.594, 254.557, 255.522, 256.968, 258.415, 260.343]

# 存储计算结果
厚度结果 = []
周期结果 = []

# 对每个周期值计算对应的厚度
for 周期 in 周期列表:
    print(f"\n计算周期 {周期} 对应的厚度:")
    厚度 = boss二阶段.寻找目标周期厚度(目标周期=周期, 厚度范围=(1, 20), 精度=0.01, 最大迭代次数=1000)
    if 厚度 is not None:
        厚度结果.append(厚度)
        周期结果.append(周期)
        print(f"周期 {周期} 对应的厚度为: {厚度:.4f} 微米")
    else:
        print(f"无法计算周期 {周期} 对应的厚度")

# 绘制厚度与周期的关系图
plt.figure(figsize=(10, 6))
plt.plot(周期结果, 厚度结果, 'o-', linewidth=2, markersize=6)
plt.xlabel('周期')
plt.ylabel('厚度 (微米)')
plt.title('厚度与周期的关系')
plt.grid(True)
plt.show()
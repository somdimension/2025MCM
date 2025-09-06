import numpy as np
import 光程差
import 计算折射率喵1
import 计算折射率喵2
from scipy.signal import find_peaks
file_path = r"C:\Users\czw17\Desktop\新输出结果喵.csv"
厚度 = 7.5#微米

def 周期函数(波数,厚度):

    波长 = 1/波数*10**4#微米
    周期 =  np.abs(np.cos(np.pi*光程差.计算光程差(计算折射率喵1.计算折射率(波数, file_path),厚度,10)/波长))
    # print(波长)  # 注释掉这行以减少输出
    return 周期

def 计算厚度周期关系(厚度值):
    """计算给定厚度下的周期信息"""
    波数 = np.linspace(1500,2700, 20000)
    周期 = 周期函数(波数, 厚度值)
    
    # 使用find_peaks找到所有峰值
    peaks, _ = find_peaks(周期, height=0)
    
    # 计算相邻峰值波数的差值作为周期
    if len(peaks) > 1:
        峰值波数 = 波数[peaks]
        周期差值 = np.diff(峰值波数)
        平均周期 = np.mean(np.abs(周期差值))
        #print(f'厚度 {厚度值} 对应的平均周期: {平均周期:.2f}')
        return 平均周期
    else:
        print(f'厚度 {厚度值} 下未找到足够的峰值')
        return None
#绘制周期函数
import matplotlib.pyplot as plt
波数 = np.linspace(1500, 4000, 20000)
周期 = 周期函数(波数,厚度)

# plt.plot(波数, 周期)
# plt.xlabel('波数')
# plt.ylabel('周期')
# plt.title('周期函数')

# 使用find_peaks找到所有峰值
peaks, _ = find_peaks(周期, height=0)  # 可以根据需要调整height参数来过滤较小的峰值

# 在图上标注峰值点
plt.plot(波数[peaks], 周期[peaks], 'ro', markersize=4, label='Peaks')

# 输出所有峰值信息
print("所有峰值信息：")
for i, peak_index in enumerate(peaks):
    print(f'峰值 {i+1}: 周期值 = {周期[peak_index]:.4f}, 对应波数 = {波数[peak_index]:.2f}')

# 计算并输出相邻峰值波数差值
if len(peaks) > 1:
    峰值波数 = 波数[peaks]
    周期差值 = np.diff(峰值波数)
    print("\n相邻峰值波数差值（周期）：")
    for i, diff in enumerate(周期差值):
        print(f'周期 {i+1}: {diff:.2f}')

# 找到最大值及其对应的波数
max_value = np.max(周期)
max_index = np.argmax(周期)
max_wavenumber = 波数[max_index]

print(f'\n最大值: {max_value}')
print(f'对应的波数: {max_wavenumber}')

plt.legend()
plt.show()

# 调用新函数计算厚度与周期的关系
print("\n计算厚度与周期的关系：")
计算厚度周期关系(厚度)
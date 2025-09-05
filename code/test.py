import numpy as np
import matplotlib.pyplot as plt

def calculate_sic_refractive_index(wavenumber_cm):
    """
    根据给定的Sellmeier公式计算SiC的折射率。
    
    参数:
    wavenumber_cm (array-like): 波数，单位为 cm⁻¹
    
    返回:
    n (numpy.ndarray): 对应的折射率
    """
    # 将波数 (cm⁻¹) 转换为波长 (μm)
    # λ(μm) = 10^4 / ν(cm⁻¹)
    lambda_um = 1e4 / wavenumber_cm
    lambda_sq = lambda_um**2

    # Sellmeier 公式中的常数
    B1 = 5.58245
    C1_sq = 0.1625394**2
    B2 = 2.468516
    C2_sq = 11.35656**2

    # 计算 n²
    # n² = 1 + (B1 * λ²) / (λ² - C1²) + (B2 * λ²) / (λ² - C2²)
    term1 = (B1 * lambda_sq) / (lambda_sq - C1_sq)
    term2 = (B2 * lambda_sq) / (lambda_sq - C2_sq)
    n_squared = 1 + term1 + term2

    # 在n²为负的区域（物理上无意义），结果会是NaN
    # np.sqrt会自动处理这些情况
    with np.errstate(invalid='ignore'): # 忽略因对负数开方产生的警告
        n = np.sqrt(n_squared)
        
    return n

# 定义绘图的波数范围
# 公式在 λ = 11.35656 μm (约 880.6 cm⁻¹) 处有奇点，因此我们在此处断开范围以获得更好的可视化效果
wavenumber_range1 = np.linspace(400, 875, 500)
wavenumber_range2 = np.linspace(885, 5000, 1000)

# 计算对应的折射率
n1 = calculate_sic_refractive_index(wavenumber_range1)
n2 = calculate_sic_refractive_index(wavenumber_range2)

# 绘图
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(wavenumber_range1, n1, color='royalblue', linewidth=2.5)
ax.plot(wavenumber_range2, n2, color='royalblue', linewidth=2.5)

# 设置图表标题和坐标轴标签
ax.set_title('SiC Refractive Index vs. Wavenumber', fontsize=16)
ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12)
ax.set_ylabel('Refractive Index (n)', fontsize=12)

# 设置坐标轴范围，以更好地显示曲线
ax.set_ylim(2.2, 2.55)
ax.set_xlim(1500, 4000)

# 显示图表
plt.show()
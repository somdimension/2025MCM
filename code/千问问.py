import numpy as np
import matplotlib.pyplot as plt

# ==================== 用户自定义参数 ====================
# 三束光的光强 (任意单位，如 W/m²)
intensities = [1.0, 0.5, 0.2]  # [I1, I2, I3]

# 三束光的初始相位（弧度），第一束光相位为0，其余是相对相位差
phases = [0.0, 0, 0]  # [φ1, φ2, φ3]

# 波数范围 (cm⁻¹)
k_start = 400
k_end = 40000
k_step = 0.1  # 步长，越小越精细

# =======================================================

# 生成波数数组
k_values = np.arange(k_start, k_end + k_step/2, k_step)  # 包含 end

# 假设三束光传播路径完全重合（x=0），则相位仅由初始相位决定
# 总电场复振幅：E_total = Σ √I_i * exp(1j * φ_i)
# 注意：这里没有路径差导致的相位项（比如 k*Δx），因为题目未提及空间分离
# 如果你希望加入路径差，请扩展此模型

sqrt_I = np.sqrt(intensities)  # 电场振幅正比于 sqrt(光强)

# 计算每个波数下的总光强（由于没有路径差，其实与k无关？）
# 但！—— 如果你想模拟“由于波长变化导致固定路径差引起的相位差变化”，
# 那么需要引入路径差 Δx，此时相位差 = k * Δx

# 🔺 重要修正：通常干涉实验中，相位差来源于路径差 Δx，而 Δx 是固定的，
# 波数 k 变化 → 相位差变化 → 干涉条纹变化。
# 因此，我们应允许用户设置路径差！

# ==================== 增强版：支持路径差设置 ====================

# 设置三束光相对于参考点的路径差（单位：cm）
# 例如：path_diffs = [0.0, 0.001, -0.0005] 表示第二束光多走 10 微米，第三束少走 5 微米
path_diffs = [0.0, 0.002, 0.004]  # 单位：厘米（cm），因为k单位是 cm⁻¹

# 总相位 = 初始相位 + k * 路径差
# E_i = √I_i * exp(1j * (φ_i + k * Δx_i))

total_intensity = []

for k in k_values:
    E_total = 0j
    for i in range(3):
        phase_total = phases[i] + k * path_diffs[i]
        E_i = sqrt_I[i] * np.exp(1j * phase_total)
        E_total += E_i
    I_total = np.abs(E_total)**2
    total_intensity.append(I_total)

total_intensity = np.array(total_intensity)

# ==================== 绘图 ====================
plt.figure(figsize=(12, 6))
plt.plot(k_values, total_intensity, 'b-', linewidth=1)
plt.title('三束光干涉总光强 vs 波数', fontsize=16)
plt.xlabel('波数 (cm$^{-1}$)', fontsize=14)
plt.ylabel('总光强 (a.u.)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(k_start, k_end)
plt.tight_layout()
plt.show()

# 可选：打印一些统计信息
print(f"最小光强: {np.min(total_intensity):.4f}")
print(f"最大光强: {np.max(total_intensity):.4f}")
print(f"平均光强: {np.mean(total_intensity):.4f}")
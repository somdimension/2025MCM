import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# ========== 1. 加载你的数据 ==========
input_path = r"C:\Users\czw17\Desktop\附件4.csv"

try:
    df = pd.read_csv(input_path, header=0)
    df.columns = ['自变量', '因变量']
    time_series = df['因变量'].values  # 转为numpy数组便于计算
    x = df['自变量'].values  # 使用实际的波数作为x轴
    print(f"✅ 数据加载成功，共 {len(time_series)} 个点")
except FileNotFoundError:
    print(f"❌ 错误：找不到文件，请检查路径是否正确: {input_path}")
    exit()
except Exception as e:
    print(f"❌ 数据加载出错: {e}")
    exit()

# ========== 2. 提取趋势 —— 高斯滤波器 ==========
# 使用高斯滤波器进行平滑处理，sigma值控制平滑程度
sigma = len(time_series) // 5  # 根据数据长度自动调整sigma值
sigma = max(1, sigma)  # 确保sigma至少为1

try:
    trend_est = gaussian_filter1d(time_series, sigma=sigma)
except Exception as e:
    print(f"⚠️ 高斯滤波失败: {e}")
    exit()

disturbance_est = time_series - trend_est

# ========== 2.1 对扰动项进行高斯滤波 ==========
# 使用高斯滤波器进一步平滑扰动项，sigma值控制平滑程度
sigma_dist = max(1, len(time_series) // 1000)  # 根据数据长度自动调整sigma值

try:
    disturbance_est = gaussian_filter1d(disturbance_est, sigma=sigma_dist)
except Exception as e:
    print(f"⚠️ 扰动项高斯滤波失败: {e}")
    exit()

# ========== 3. 可视化原始信号、趋势、扰动 ==========
plt.figure(figsize=(14, 10))

plt.subplot(3, 1, 1)
plt.plot(x, time_series, label='原始信号', color='blue', alpha=0.8)
plt.title('原始信号')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(x, time_series, label='原始信号', color='blue', alpha=0.5)
plt.plot(x, trend_est, 'r-', linewidth=2, label='估计趋势', alpha=0.9)
plt.title('趋势项提取')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(x, disturbance_est, 'g-', label='估计扰动', alpha=0.8)
plt.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.6)
plt.title('扰动项（周期性尖峰）')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

# ========== 4. 检测扰动周期（通过波谷） ==========
# 设置合理的波谷检测参数
# 通过检测负值来找到波谷
valley_height_threshold = np.mean(disturbance_est) - 0.05 * np.std(disturbance_est)  # 低于均值-0.5标准差
distance = max(10, len(time_series) // 50)  # 波谷最小间隔，避免过密

valleys, properties = find_peaks(-disturbance_est, height=-valley_height_threshold, distance=distance)

print(f"\n🔍 检测到 {len(valleys)} 个扰动波谷")

# 打印每个波谷对应的波数
for i, valley_index in enumerate(valleys):
    print(f"第 {i+1} 个波谷位置: {x[valley_index]:.2f} 波数")

if len(valleys) > 1:
    # 计算波数单位的周期
    periods = np.diff(x[valleys])
    avg_period = np.mean(periods)
    std_period = np.std(periods)
    print(f"📈 平均周期: {avg_period:.2f} 波数单位 (cm⁻¹)")
    print(f"📉 周期标准差: {std_period:.2f} → {'稳定' if std_period < avg_period*0.3 else '波动较大'}")

    # 创建表格数据
    valley_data = []
    for i, valley_index in enumerate(valleys):
        data_row = {
            '波谷序号': i + 1,
            '索引位置': valley_index,
            '波数位置': x[valley_index]
        }
        
        # 添加周期信息（除了最后一个波谷）
        if i < len(periods):
            data_row['到下一波谷周期'] = periods[i]
            
        valley_data.append(data_row)
    
    # 转换为DataFrame
    valley_df = pd.DataFrame(valley_data)
    
    # 保存到CSV文件
    output_path = "扰动周期数据.csv"
    valley_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 扰动周期数据已保存至: {output_path}")
    print("\n📋 扰动周期数据表格:")
    print(valley_df.to_string(index=False))

    # 绘制扰动与波谷
    plt.figure(figsize=(12, 5))
    plt.plot(x, disturbance_est, 'g-', label='扰动信号', alpha=0.7)
    plt.plot(x[valleys], disturbance_est[valleys], "rx", markersize=8, label='检测到的波谷')
    plt.title('扰动信号中的周期性波谷')
    plt.xlabel('波数 (cm⁻¹)')
    plt.ylabel('扰动幅值')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
else:
    print("⚠️ 未检测到足够峰值，可能扰动不明显或参数需调整")

# ========== 5. （可选）尝试拟合扰动形状：(ax+f) / (cos(b*x + c) + d) + e ==========
def pulse_model(x, a, f, b, c, d, e):
    # 避免分母接近0导致爆炸 → clip分母下限
    denominator = np.cos(b * x + c) + d
    denominator = np.clip(denominator, 0.1, None)  # 防止除零
    return -(a * x + f) / denominator + e

if len(valleys) >= 3:
    try:
        # 初始参数猜测
        a_guess = -0.2  # 对于(ax+f)形式，a的初始值设为0
        f_guess = np.min(disturbance_est[valleys]) - np.mean(disturbance_est)
        b_guess = 2 * np.pi / avg_period  # 根据平均周期估算角频率
        c_guess = 0.0
        d_guess = 1.1  # 避免分母为0
        e_guess = np.mean(disturbance_est)

        p0 = [a_guess, f_guess, b_guess, c_guess, d_guess, e_guess]

        # 局部拟合：只拟合前几个周期提高稳定性
        # 使用波数单位计算拟合范围
        fit_range_points = min(len(x), int(avg_period * 5 / np.mean(np.diff(x))))  # 拟合前5个周期
        popt, pcov = curve_fit(
            pulse_model,
            x[:fit_range_points],
            disturbance_est[:fit_range_points],
            p0=p0,
            maxfev=10000,
            bounds=([-10, -10, 0, -np.pi, 0.5, -np.inf], [-0.1, np.inf, 2*np.pi, np.pi, 3.0, np.inf])
        )

        fitted_disturbance = pulse_model(x, *popt)
        residuals = disturbance_est - fitted_disturbance
        rmse = np.sqrt(np.mean(residuals**2))
        print(f"\n🎯 扰动形状拟合成功！")
        param_names = ['a (斜率)', 'f (截距)', 'b (角频率)', 'c (相位)', 'd (偏移)', 'e (垂直偏移)']
        for name, val in zip(param_names, popt):
            print(f"   {name}: {val:.4f}")
        print(f"   拟合误差 (RMSE): {rmse:.4f}")

        # 绘图对比
        plt.figure(figsize=(12, 5))
        plt.plot(x, disturbance_est, 'g-', alpha=0.6, label='实际扰动')
        plt.plot(x, fitted_disturbance, 'r--', linewidth=2, label='拟合模型')
        plt.title('扰动形状拟合结果对比')
        plt.xlabel('波数 (cm⁻¹)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    except Exception as e:
        print(f"\n⚠️ 形状拟合失败: {e}")
        print("👉 可能原因：噪声大、形状不符、初值不准。可尝试手动调整 p0 或拟合区间。")

# ========== 6. 输出扰动信号到高分辨率表格 ==========
# 创建包含波数和扰动值的数据框
high_res_data = pd.DataFrame({
    '波数 (cm⁻¹)': x,
    '扰动值': disturbance_est
})

# 保存到CSV文件
output_path_high_res = "扰动信号数据.csv"
high_res_data.to_csv(output_path_high_res, index=False, encoding='utf-8-sig')
print(f"\n💾 扰动信号数据已保存至: {output_path_high_res}")
print("\n📋 扰动信号数据表格 (前10行):")
print(high_res_data.head(10).to_string(index=False))
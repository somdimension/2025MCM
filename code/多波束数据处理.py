import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# ========== 1. 加载你的数据 ==========
input_path = r"C:\Users\czw17\Desktop\附件3.csv"

try:
    df = pd.read_csv(input_path, header=0)
    df.columns = ['自变量', '因变量']
    time_series = df['因变量'].values  # 转为numpy数组便于计算
    x = np.arange(len(time_series))   # 使用索引作为x轴（若“自变量”是时间/位置，可替换）
    print(f"✅ 数据加载成功，共 {len(time_series)} 个点")
except FileNotFoundError:
    print(f"❌ 错误：找不到文件，请检查路径是否正确: {input_path}")
    exit()
except Exception as e:
    print(f"❌ 数据加载出错: {e}")
    exit()

# ========== 2. 提取趋势 —— Savitzky-Golay滤波器 ==========
# 窗口长度选择：建议覆盖2~5个扰动周期。初始设为总长度的1/10，必须是奇数
window_length = len(time_series) // 4
print(window_length)

if window_length % 2 == 0:
    window_length += 1
window_length = max(51, window_length)  # 至少51保证平滑效果

polyorder = 3
try:
    trend_est = savgol_filter(time_series, window_length, polyorder)
except ValueError:
    # 如果窗口太大，自动缩小
    window_length = min(99, len(time_series) - 1 if len(time_series) % 2 == 0 else len(time_series) - 2)
    if window_length < 5:
        print("⚠️ 数据太短，无法进行滤波")
        exit()
    trend_est = savgol_filter(time_series, window_length, polyorder)
    print(f"⚠️ 自动调整窗口长度为: {window_length}")

disturbance_est = time_series - trend_est

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

# ========== 4. 检测扰动周期（通过峰值） ==========
# 设置合理的峰值检测参数
height_threshold = np.mean(disturbance_est) + 0.5 * np.std(disturbance_est)  # 高于均值+0.5标准差
distance = max(10, len(time_series) // 50)  # 峰值最小间隔，避免过密

peaks, properties = find_peaks(disturbance_est, height=height_threshold, distance=distance)

print(f"\n🔍 检测到 {len(peaks)} 个扰动峰值")
if len(peaks) > 1:
    periods = np.diff(peaks)
    avg_period = np.mean(periods)
    std_period = np.std(periods)
    print(f"📈 平均周期: {avg_period:.2f} 个采样点")
    print(f"📉 周期标准差: {std_period:.2f} → {'稳定' if std_period < avg_period*0.3 else '波动较大'}")

    # 绘制扰动与峰值
    plt.figure(figsize=(12, 5))
    plt.plot(x, disturbance_est, 'g-', label='扰动信号', alpha=0.7)
    plt.plot(x[peaks], disturbance_est[peaks], "rx", markersize=8, label='检测到的峰值')
    plt.title('扰动信号中的周期性峰值')
    plt.xlabel('采样点索引')
    plt.ylabel('扰动幅值')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
else:
    print("⚠️ 未检测到足够峰值，可能扰动不明显或参数需调整")

# ========== 5. （可选）尝试拟合扰动形状：a / (cos(b*x + c) + d) + e ==========
def pulse_model(x, a, b, c, d, e):
    # 避免分母接近0导致爆炸 → clip分母下限
    denominator = np.cos(b * x + c) + d
    denominator = np.clip(denominator, 0.1, None)  # 防止除零
    return a / denominator + e

if len(peaks) >= 3:
    try:
        # 初始参数猜测
        a_guess = np.max(disturbance_est[peaks]) - np.mean(disturbance_est)
        b_guess = 2 * np.pi / avg_period  # 根据平均周期估算角频率
        c_guess = 0.0
        d_guess = 1.1  # 避免分母为0
        e_guess = np.mean(disturbance_est)

        p0 = [a_guess, b_guess, c_guess, d_guess, e_guess]

        # 局部拟合：只拟合前几个周期提高稳定性
        fit_range = min(len(x), int(avg_period * 5))  # 拟合前5个周期
        popt, pcov = curve_fit(
            pulse_model,
            x[:fit_range],
            disturbance_est[:fit_range],
            p0=p0,
            maxfev=10000,
            bounds=([0, 0, -np.pi, 0.5, -np.inf], [np.inf, 2*np.pi, np.pi, 3.0, np.inf])
        )

        fitted_disturbance = pulse_model(x, *popt)
        residuals = disturbance_est - fitted_disturbance
        rmse = np.sqrt(np.mean(residuals**2))
        print(f"\n🎯 扰动形状拟合成功！")
        param_names = ['a (幅度)', 'b (角频率)', 'c (相位)', 'd (偏移)', 'e (垂直偏移)']
        for name, val in zip(param_names, popt):
            print(f"   {name}: {val:.4f}")
        print(f"   拟合误差 (RMSE): {rmse:.4f}")

        # 绘图对比
        plt.figure(figsize=(12, 5))
        plt.plot(x, disturbance_est, 'g-', alpha=0.6, label='实际扰动')
        plt.plot(x, fitted_disturbance, 'r--', linewidth=2, label='拟合模型')
        plt.title('扰动形状拟合结果对比')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    except Exception as e:
        print(f"\n⚠️ 形状拟合失败: {e}")
        print("👉 可能原因：噪声大、形状不符、初值不准。可尝试手动调整 p0 或拟合区间。")
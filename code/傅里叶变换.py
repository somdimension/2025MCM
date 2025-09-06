import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
import statsmodels.tsa.stattools as sm
from scipy.fft import rfft, rfftfreq
from scipy import signal

# --- 1. 读取数据 ---
input_path = r"C:\Users\czw17\Desktop\附件2.csv"

try:
    df = pd.read_csv(input_path, header = 0)#有headers
    df.columns = ['自变量', '因变量']
    time_series = df['因变量']
except FileNotFoundError:
    print(f"错误：找不到文件，请检查路径是否正确: {input_path}")
    exit()

# --- 2. 预处理：去趋势 ---
# 周期性分析在平稳（或近似平稳）的数据上效果最好。
# 原始数据有明显上升趋势，会干扰周期判断，因此先移除线性趋势。
detrended_series = signal.detrend(time_series.values)

print("--- 周期自动检测 ---")

# --- 3. 方法一：使用傅里叶变换(FFT)检测周期 ---
N = len(detrended_series)
# 进行实数傅里叶变换
yf = rfft(detrended_series)
xf = rfftfreq(N, 1) # 假设采样间隔为1

# 找到频谱中能量最大的点的索引 (忽略索引0，因为它是直流分量/均值)
idx = np.argmax(np.abs(yf[1:])) + 1
# 根据索引找到对应的频率
dominant_freq = xf[idx]
# 周期 = 1 / 频率
if dominant_freq > 0:
    fft_period = int(round(1 / dominant_freq))
    print(f"方法一 (FFT) 检测到的主周期大约是: {fft_period}")
else:
    fft_period = None
    print("方法一 (FFT) 未能检测到明确的周期。")


# --- 4. 方法二：使用自相关函数(ACF)检测周期 ---
# 计算自相关函数，nlags表示最大延迟期数
# 我们观察一半数据长度的延迟就足够了
acf_vals = sm.acf(detrended_series, nlags=N//2, fft=True)

# 寻找ACF中的峰值
# find_peaks会返回峰值的索引
peaks, _ = signal.find_peaks(acf_vals, height=0.1) # height可以过滤掉较小的噪声峰值

if len(peaks) > 0:
    # 第一个峰值通常对应着主周期
    acf_period = peaks[0]
    print(f"方法二 (ACF) 检测到的主周期大约是: {acf_period}")
else:
    acf_period = None
    print("方法二 (ACF) 未能检测到明确的周期。")
    
print("--------------------")


# --- 5. 使用检测到的周期进行STL分解 ---
# 决策：选择一个估算出的周期用于STL。通常两者结果会很接近。
# 如果两者差异很大，建议您画出原始数据图，根据肉眼观察来辅助判断。
# 这里我们优先使用FFT的结果，如果FFT没有结果，则使用ACF的结果。
detected_period = fft_period if fft_period is not None else acf_period

if detected_period is None or detected_period <= 1:
    print("\n未能检测到有效的周期，无法执行STL分解。")
    print("建议：请检查您的数据是否真的具有周期性，或尝试调整检测参数。")
    exit()

print(f"\n将使用检测到的周期(period={detected_period})进行STL分解...")

# 使用检测到的周期进行STL
stl = STL(time_series, period=detected_period, robust=True)
result = stl.fit()

# --- 6. 整合、输出并可视化 ---
output_df = pd.DataFrame({
    '原始自变量': df['自变量'],
    '原始因变量': df['因变量'],
    '趋势分量(Trend)': result.trend,
    '季节性分量(Seasonal)': result.seasonal,
    '残差分量(Residual)': result.resid
})

output_path = r"C:\Users\czw17\Desktop\附件2_STL分解结果_自动周期.xlsx"
output_df.to_excel(output_path, index=False)
print(f"分解完成！详细信息已保存到: {output_path}")

# 可视化
fig = result.plot()
plt.suptitle(f'STL Decomposition (Detected Period = {detected_period})', y=1.02)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.tight_layout(rect=[0, 0, 1, 0.96])

plot_path = r"C:\Users\czw17\Desktop\附件1_STL分解图_自动周期.png"
plt.savefig(plot_path)
print(f"分解结果图已保存到: {plot_path}")
plt.show()
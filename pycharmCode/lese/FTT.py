# @Version : 1.0
# @Author  : 亥子曜
# @File    : FTT.py
# @Time    : 2025/9/5 11:27
# 多正弦拟合（FFT 初筛 + 线性最小二乘）
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from numpy.linalg import lstsq

# 读取数据（按你的文件）
df = pd.read_csv("../question1/dealfile1/附件1.csv", encoding='utf-8')  # 或改为 gbk
x = df["波数 (cm-1)"].values.astype(float)
y = df["反射率 (%)"].values.astype(float)

# 保证 x 单调并排序
idx = np.argsort(x)
x = x[idx]
y = y[idx]

# 插值到等间距网格用于 FFT（频谱更可靠）
dx = np.median(np.diff(x))
x_uniform = np.arange(x.min(), x.max()+dx, dx)
y_uniform = np.interp(x_uniform, x, y)

# FFT 找主频
Y = rfft(y_uniform - np.mean(y_uniform))
freqs = rfftfreq(len(y_uniform), d=dx)   # 单位：cycles per (波数 unit)
mag = np.abs(Y)
mag[0] = 0  # 忽略直流
# 取前 K 峰（按幅值排序）
K = 2
peak_indices = np.argsort(mag)[-K:][::-1]  # 从大到小
dominant_freqs = freqs[peak_indices]       # 频率（cycles per x-unit）
# 转换为周期 P_i（单位与 x 相同，P = 1/f）
P_list = [1.0/f if f>0 else np.inf for f in dominant_freqs]

print("Detected frequencies (cycles per unit):", dominant_freqs)
print("Estimated periods P (same units as x):", P_list)

# 构建设计矩阵：列为 [x, 1, sin(2π x / P1), cos(...), sin(...P2...), cos(...P3...)]
cols = [x, np.ones_like(x)]
for P in P_list:
    if np.isfinite(P):
        cols.append(np.sin(2*np.pi*x / P))
        cols.append(np.cos(2*np.pi*x / P))

A = np.column_stack(cols)   # shape (n_samples, 2+2K)

# 最小二乘求解
coef, *_ = lstsq(A, y, rcond=None)   # coef shape (2+2K,)
# 拆解参数
a = coef[0]
b = coef[1]
CD = coef[2:]
amps = []
phases = []
for i in range(K):
    C = CD[2*i]
    D = CD[2*i+1]
    A_i = np.hypot(C, D)           # 振幅
    phi_i = np.arctan2(D, C)       # 注意 arctan2 的顺序 (cos->D, sin->C mapping)
    amps.append(A_i)
    phases.append(phi_i)

# 重构拟合值
y_fit = A.dot(coef)
resid = y - y_fit

print("Linear slope a =", a)
print("Intercept b =", b)

# 构造函数字符串
terms = [f"{a:.6f} * x", f"{b:.6f}"]
for i, P in enumerate(P_list):
    if np.isfinite(P):
        Ai = amps[i]
        phii = phases[i]
        terms.append(f"{Ai:.6f} * sin(2π * x / {P:.6f} + {phii:.6f})")

full_expr = "y(x) ≈ " + " + ".join(terms)
print("\nFitted function:")
print(full_expr)


plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(x, y, '.', markersize=3, label='data')
plt.plot(x, y_fit, '-', label='multi-sin fit (linear LS)')
plt.gca().invert_xaxis()
plt.legend()
plt.ylabel("Reflectance (%)")

plt.subplot(2,1,2)
plt.plot(x, resid, '.', markersize=3)
plt.axhline(0, color='k', lw=0.6)
plt.gca().invert_xaxis()
plt.xlabel("Wavenumber (cm^-1)")
plt.ylabel("Residual")
plt.tight_layout()
plt.show()

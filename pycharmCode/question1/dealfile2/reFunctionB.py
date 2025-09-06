# @Version : 5.2
# @Author  : 亥子曜
# @File    : reFunctionFitLinear3Sine_excel.py
# @Time    : 2025/9/5
# 功能: 拟合 f(x) = g(x) + 3*sin(x)，其中 g(x) = c0 + c1*x，输入为 Excel 文件

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.fft import rfft, rfftfreq

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# === 1. 读取数据（Excel 文件） ===
df = pd.read_excel("附件2.xlsx", engine="openpyxl")
df = df[df['波数 (cm-1)'] > 1500].copy()

# 提取 x 和 y
x = df["波数 (cm-1)"].values
y = df["反射率 (%)"].values.astype(float)

# === 2. FFT 提取主频 ===
dx = np.median(np.diff(x))
x_u = np.arange(x.min(), x.max()+dx, dx)
y_u = np.interp(x_u, x, y)
Y = rfft(y_u - np.mean(y_u))
freqs = rfftfreq(len(y_u), d=dx)
mag = np.abs(Y); mag[0] = 0

# 过滤掉过低频
valid = freqs > 1/x.ptp()
idx_peaks = np.argsort(mag[valid])[-3:][::-1]
P0_list = [1.0/freqs[valid][i] for i in idx_peaks]

# === 3. 定义模型 ===
def model(x, c0, c1,
          A1, P1, phi1,
          A2, P2, phi2,
          A3, P3, phi3):
    g = c0 + c1*x
    s1 = A1*np.sin(2*np.pi*x/P1 + phi1)
    s2 = A2*np.sin(2*np.pi*x/P2 + phi2)
    s3 = A3*np.sin(2*np.pi*x/P3 + phi3)
    return g + s1 + s2 + s3

# === 4. 初始参数 ===
c1_init, c0_init = np.polyfit(x, y, 1)
p0 = [
    c0_init, c1_init,
    0.5*np.ptp(y), P0_list[0], 0.0,
    0.3*np.ptp(y), P0_list[1], 0.0,
    0.2*np.ptp(y), P0_list[2], 0.0
]

bounds = ([-np.inf, -np.inf,
           0, 0.1, -2*np.pi,
           0, 0.1, -2*np.pi,
           0, 0.1, -2*np.pi],
          [np.inf, np.inf,
           np.inf, np.inf, 2*np.pi,
           np.inf, np.inf, 2*np.pi,
           np.inf, np.inf, 2*np.pi])

# === 5. 拟合 ===
popt, pcov = curve_fit(model, x, y, p0=p0, bounds=bounds, maxfev=200000)
(c0, c1,
 A1, P1, phi1,
 A2, P2, phi2,
 A3, P3, phi3) = popt

# === 6. 结果计算 ===
f_fit = model(x, *popt)          # f(x)
g_fit = c0 + c1*x                # g(x)
error = y - f_fit                # 残差

# === 7. 打印公式 ===
print("最终拟合结果：")
print(f"f(x) = ({c0:.6f} + {c1:.6f}*x) "
      f"+ {A1:.6f}*sin(2πx/{P1:.6f} + {phi1:.6f}) "
      f"+ {A2:.6f}*sin(2πx/{P2:.6f} + {phi2:.6f}) "
      f"+ {A3:.6f}*sin(2πx/{P3:.6f} + {phi3:.6f})")
print(f"g(x) = {c0:.6f} + {c1:.6f}*x")

# === 8. 保存结果 ===
out = pd.DataFrame({
    df.columns[0]: x,
    "y_original": y,
    "f_fit": f_fit,
    "g_fit": g_fit,
    "error": error
})
out.to_excel("fit_result_linear3sine_attachment2.xlsx", index=False)
print("结果已保存到 fit_result_linear3sine_attachment2.xlsx")

# === 9. 作图 ===
plt.figure(figsize=(12,10))

plt.subplot(4,1,1)
plt.plot(x, y, 'k.', label="原始数据")
plt.plot(x, f_fit, 'r-', label="拟合 f(x)")
plt.legend()
plt.title("原始数据 vs f(x)")

plt.subplot(4,1,2)
plt.plot(x, f_fit, 'r-', label="f(x)")
plt.plot(x, g_fit, 'b--', label="g(x)")
plt.legend()
plt.title("f(x) 与 g(x)")

plt.subplot(4,1,3)
plt.plot(x, y, 'k.', label="原始数据")
plt.plot(x, g_fit, 'b-', label="g(x)")
plt.legend()
plt.title("原始数据 vs g(x)")

plt.subplot(4,1,4)
plt.plot(x, error, 'g.', label="残差 y - f(x)")
plt.axhline(0, color="gray", linestyle="--")
plt.legend()
plt.title("残差")

plt.tight_layout()
plt.show()

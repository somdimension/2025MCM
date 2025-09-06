# @Version : 1.1
# @Author  : 亥子曜
# @File    : peekRefunction.py
# @Time    : 2025/9/5
# 功能: 已知 f(x) = g(x)+sin1+sin2+sin3, 分步画7个图

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# === 1. 读取数据 ===
df = pd.read_excel("附件2.xlsx", engine="openpyxl")  # 使用 openpyxl 读取 Excel
df = df[df['波数 (cm-1)'] > 1500].copy()  # 过滤波数 > 1500 的数据
x = df["波数 (cm-1)"].values
y = df["反射率 (%)"].values.astype(float)

# === 2. 已知拟合参数 ===
def g(x):
    return 15.070904 + 0.001245*x

def sin1(x):
    return 0.303434*np.sin(2*np.pi*x/2067.675079 + 6.283185)

def sin2(x):
    return 0.492687*np.sin(2*np.pi*x/251.676557 - 0.873362)

def sin3(x):
    return 0.142390*np.sin(2*np.pi*x/768.384168 - 1.865889)

# === 3. 组合函数 ===
f1 = g(x) + sin1(x)
f2 = g(x) + sin2(x)
f3 = g(x) + sin3(x)
f12 = g(x) + sin1(x) + sin2(x)
f13 = g(x) + sin1(x) + sin3(x)
f23 = g(x) + sin2(x) + sin3(x)
f123 = g(x) + sin1(x) + sin2(x) + sin3(x)

# === 4. 作图 ===
fits = [f1, f2, f3, f12, f13, f23, f123]
titles = [
    "原始数据 vs f1(x) = g(x)+sin1(x)",
    "原始数据 vs f2(x) = g(x)+sin2(x)",
    "原始数据 vs f3(x) = g(x)+sin3(x)",
    "原始数据 vs f12(x) = g(x)+sin1(x)+sin2(x)",
    "原始数据 vs f13(x) = g(x)+sin1(x)+sin3(x)",
    "原始数据 vs f23(x) = g(x)+sin2(x)+sin3(x)",
    "原始数据 vs f123(x) = g(x)+sin1(x)+sin2(x)+sin3(x)"
]

plt.figure(figsize=(14,18))
for i, (fit, title) in enumerate(zip(fits, titles), start=1):
    plt.subplot(4,2,i)
    plt.plot(x, y, 'k.', markersize=2, label="原始数据")
    plt.plot(x, fit, 'r-', linewidth=1, label=title.split(" vs ")[1])
    plt.title(title)
    plt.legend()

plt.tight_layout()
plt.show()

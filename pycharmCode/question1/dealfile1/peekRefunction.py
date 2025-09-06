# @Version : 1.0
# @Author  : 亥子曜
# @File    : peekRefunction.py
# @Time    : 2025/9/5 13:25
# @Version : 1.0
# @Author  : 亥子曜
# @File    : plot_stepwise_fits.py
# @Time    : 2025/9/5
# 功能: 已知 f(x) = g(x)+sin1+sin2+sin3, 分步画7个图

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# === 1. 读取数据 ===
df = pd.read_csv("附件1.csv", encoding="utf-8")
x = df.iloc[:,0].values.astype(float)
y = df.iloc[:,1].values.astype(float)

# === 2. 已知拟合参数 ===
def g(x):
    return -7.785946 + 0.008737*x

def sin1(x):
    return 10.258256*np.sin(2*np.pi*x/4934.555255 + 5.688461)

def sin2(x):
    return 2.309663*np.sin(2*np.pi*x/2888.850278 + 6.282987)

def sin3(x):
    return 0.509514*np.sin(2*np.pi*x/250.651096 - 0.460126)

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

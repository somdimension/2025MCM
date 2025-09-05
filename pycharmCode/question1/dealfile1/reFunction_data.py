# @Version : 1.0
# @Author  : 亥子曜
# @File    : reFunction_data.py
# @Time    : 2025/9/5 13:30
# @Version : 1.0
# @Author  : 亥子曜
# @File    : f12_curve.py
# @Time    : 2025/9/5

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']   # 支持中文
plt.rcParams['axes.unicode_minus'] = False

a = 0.008737
b = -7.785946

A1, P1, phi1 = 10.258256, 4934.555255, 5.688461
A2, P2, phi2 = 2.309663, 2888.850278, 6.282987
A3, P3, phi3 = 0.509514, 250.651096, -0.460126

df = pd.read_csv("附件1.csv", encoding="utf-8")
x = df.iloc[:, 0].values.astype(float)
y = df.iloc[:, 1].values.astype(float)

def g(x):
    return a * x + b
def f12(x):
    return g(x) + A1 * np.sin(2*np.pi*x/P1 + phi1) + A2 * np.sin(2*np.pi*x/P2 + phi2)

y_f12 = f12(x)

out = pd.DataFrame({df.columns[0]: x, "f12(x)": y_f12})
out.to_csv("f12_curve.csv", index=False, encoding="utf-8-sig")
print("结果已保存到 f12_curve.csv")

plt.figure(figsize=(10,6))
plt.plot(x, y, ".", label="原始数据")
plt.plot(x, y_f12, "-", label="f12(x) = g(x)+sin1+sin2")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("原始数据 vs f12(x)")
plt.show()

# @Version : 1.0
# @Author  : 亥子曜
# @File    : reFunction_data.py
# @Time    : 2025/9/5 13:30

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False

# === 已知拟合参数 ===
# 你可以根据实际需求修改这些参数
A1, P1, phi1 = 0.303434, 2067.675079, 6.283185
A2, P2, phi2 = 0.142390, 768.384168, -1.865889
c1 = 15.070904
c2 = 0.001245

# === 读取附件2的数据 ===
df = pd.read_excel("附件2.xlsx", engine="openpyxl")  # 使用 openpyxl 引擎读取 Excel
df = df[df['波数 (cm-1)'] > 1500].copy()  # 只保留 波数 > 1500 的数据
x = df["波数 (cm-1)"].values
y = df["反射率 (%)"].values.astype(float)

# === 定义 f(x) ===
def f(x):
    return (c1 + c2 * x) + A1 * np.sin(2 * np.pi * x / P1 + phi1) + A2 * np.sin(2 * np.pi * x / P2 + phi2)

# === 计算 f(x) ===
y_f = f(x)

# === 保存结果 ===
out = pd.DataFrame({df.columns[0]: x, "f(x)": y_f})
out.to_csv("f_curve_attachment2.csv", index=False, encoding="utf-8-sig")
print("结果已保存到 f13_curve.csv")

# === 绘图 ===
plt.figure(figsize=(10, 6))
plt.plot(x, y, ".", label="原始数据")
plt.plot(x, y_f, "-", label="f(x) = c1 + c2 * x + sin1 + sin2")
plt.legend()
plt.xlabel("波数 (cm$^{-1}$)")
plt.ylabel("反射率 (%)")
plt.title("原始数据 vs f(x)")
plt.show()

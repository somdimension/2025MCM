import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 读取附件3数据 (Fu-Jian-3.csv)
input_path = r"C:\Users\czw17\Desktop\附件3.csv"

try:
    data = pd.read_csv(input_path, header=0)
    data.columns = ['波数 (cm-1)', '反射率 (%)']
    time_series = data['反射率 (%)'].values  # 转为numpy数组便于计算
    x = np.arange(len(time_series))   # 使用索引作为x轴（若“自变量”是时间/位置，可替换）
    print(f"✅ 数据加载成功，共 {len(time_series)} 个点")
except FileNotFoundError:
    print(f"❌ 错误：找不到文件，请检查路径是否正确: {input_path}")
    exit()
except Exception as e:
    print(f"❌ 数据加载出错: {e}")
    exit()
# 读取附件3数据

# 数据预览
wave_number = data.iloc[:, 0].values  # 波数 (cm-1)
reflectance = data.iloc[:, 1].values / 100.0  # 反射率(转换为比例)

# 多光束干涉理论基础:
# 反射率R由薄膜厚度d、折射率n(波数依赖性)、入射角等决定。
# 反射率计算中折射率n=n(wave_number)，导致振幅随波数变化。

# 根据多光束干涉干涉条纹模型提取函数形式（简化模型的反射率公式）：
# R = A * (1 + cos(2 * pi * n(w) * d / lambda)) + B
# 其中折射率n随波数变化，振幅A随波数变化。
# 为了拟合反射率和波数函数，先定义n(wave_number)的经验模型和A(wave_number)的振幅模型。

# 将波数转换为波长(lambda, 单位微米)，lambda = 1e4 / wave_number
wavelength = 1e4 / wave_number

# 经验(假定)折射率随波长变化模型，例如线性衰减n(wavelength) = n0 - k * wavelength
# 经验振幅随波数的衰减模型，采用简单指数或多项式模型

def n_model(w, n0, k):
    return n0 - k * w

def amplitude_model(w, a0, a1):
    return a0 * np.exp(-a1 * w)

# 综合多光束干涉反射率模型
# R_fit = amplitude_model(wave_number, a0, a1) * (1 + np.cos(2 * np.pi * n_model(wave_number, n0, k) * d * wave_number)) + b
# 其中浓度d和b为其他拟合参数

def reflec_model(w, n0, k, a0, a1, d, b):
    n_w = n_model(w, n0, k)
    A_w = amplitude_model(w, a0, a1)
    return A_w * (1 + np.cos(2 * np.pi * n_w * d / (1e4 / w))) + b

# 提供初始猜测值
initial_params = [3.5, 0.001, 0.2, 0.0005, 20, 0.1]

# 拟合函数
popt, pcov = curve_fit(reflec_model, wave_number, reflectance, p0=initial_params, maxfev=10000)

# 拟合参数
popt

# 使用拟合参数绘制拟合曲线
reflectance_fit = reflec_model(wave_number, *popt)

# 绘制原始数据和拟合数据
plt.figure(figsize=(10, 6))
plt.plot(wave_number, reflectance, label='原始反射率')
plt.plot(wave_number, reflectance_fit, label='拟合反射率', linestyle='--')
plt.xlabel('波数 (cm-1)')
plt.ylabel('反射率')
plt.title('多光束干涉反射率拟合模型')
plt.legend()
plt.grid(True)
plt.show()
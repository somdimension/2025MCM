import numpy as np
import matplotlib.pyplot as plt
import 计算折射率喵3

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def 生成模拟函数(波数,折射率,厚度):
    入射角 = np.radians(10) 
    折射角 = np.arcsin(np.sin(入射角)/折射率)
    波长 = 1/波数*10**4
    光程差 = 2*厚度*折射率*np.cos(折射角)
    
    模拟函数 = np.cos(2*np.pi*光程差/波长)
    return 模拟函数

def 计算模拟函数周期(波数,折射率,厚度):
    周期 = 2*厚度*折射率*np.cos(np.radians(10))/波数
    return 周期



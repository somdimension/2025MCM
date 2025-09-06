import pandas as pd
from scipy.interpolate import interp1d
import numpy as np
file_path = r"C:\Users\czw17\Desktop\硅 折射率.csv"

def 读取csv指定列(file_path):
    """
    读取CSV文件的前几列数据
    :param file_path: CSV文件路径
    :return: 包含前几列数据的DataFrame
    """
    # 读取CSV文件，只取前几列(这里取前2列)
    df = pd.read_csv(file_path, usecols=[0, 1])
    return df

def 创建插值函数(file_path):
    """
    根据CSV文件创建插值函数
    :param file_path: CSV文件路径
    :return: 插值函数
    """
    df = 读取csv指定列(file_path)
    # 使用scipy的interp1d创建插值函数，第一列为x值(波数)，第三列为y值(折射率)
    f = interp1d(df.iloc[:, 0], df.iloc[:, 1], kind='linear', fill_value='extrapolate')
    return f
  

def 计算折射率(波数, file_path):
    """
    根据波数计算折射率
    :param 波数: 输入的波数
    :param file_path: CSV文件路径
    :return: 对应的折射率
    """
    波长 = 1/波数*10**4

    插值函数 = 创建插值函数(file_path)
    折射率 = 插值函数(波长)
    return 折射率


# 使用示例
# 调用函数并打印结果
# df = 读取csv指定列(file_path)
# print("前几列数据：")
# print(df.head())

# # 测试插值函数
# print("测试插值函数，波长为2.5时的折射率：", 计算折射率(2.5, file_path))
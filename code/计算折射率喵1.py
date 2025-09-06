import pandas as pd
from scipy.interpolate import interp1d


def 读取csv指定列(file_path):
    """
    读取CSV文件的第一列和第三列数据
    :param file_path: CSV文件路径
    :return: 包含第一列和第三列数据的DataFrame
    """
    # 读取CSV文件，只取第一列(索引0)和第三列(索引2)
    df = pd.read_csv(file_path, usecols=[0, 2])
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
    插值函数 = 创建插值函数(file_path)
    折射率 = 插值函数(波数)
    return 折射率


# 使用示例
if __name__ == "__main__":
    csv_file_path = r"C:\Users\czw17\Desktop\新输出结果喵.csv"
    df = 读取csv指定列(csv_file_path)
    # print("前5行数据:")
    # print(df.head())
    
    # 测试插值函数
    波数 = 2000  # 示例波数
    折射率 = 计算折射率(波数, csv_file_path)
    print(f"波数 {波数} 对应的折射率为: {折射率}")
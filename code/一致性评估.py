import numpy as np
from typing import List, Union, Dict, Any

def analyze_data_consistency(data: List[Union[int, float]]) -> Dict[str, Any]:
    """
    分析数据列表的一致性，计算平均值、方差、标准差、极差、变异系数等参数
    
    参数:
        data: 包含数值的列表
    
    返回:
        包含各种统计参数的字典
    """
    if len(data) == 0:
        return {"error": "数据列表为空"}
    
    # 转换为numpy数组便于计算
    arr = np.array(data)
    
    # 基本统计量
    mean = np.mean(arr)  # 平均值
    variance = np.var(arr, ddof=0)  # 总体方差
    sample_variance = np.var(arr, ddof=1)  # 样本方差
    std_dev = np.std(arr, ddof=0)  # 标准差
    sample_std_dev = np.std(arr, ddof=1)  # 样本标准差
    data_range = np.max(arr) - np.min(arr)  # 极差
    cv = (std_dev / mean * 100) if mean != 0 else 0  # 变异系数(百分比)
    
    # 其他反映一致性的指标
    median = np.median(arr)  # 中位数
    q1 = np.percentile(arr, 25)  # 第一四分位数
    q3 = np.percentile(arr, 75)  # 第三四分位数
    iqr = q3 - q1  # 四分位距
    
    # 偏度和峰度（反映数据分布形态）
    skewness = np.mean(((arr - mean) / std_dev) ** 3) if std_dev != 0 else 0
    kurtosis = np.mean(((arr - mean) / std_dev) ** 4) - 3 if std_dev != 0 else 0
    
    results = {
        "数据个数": len(data),
        "平均值": round(mean, 6),
        "中位数": round(median, 6),
        "总体方差": round(variance, 6),
        "样本方差": round(sample_variance, 6),
        "总体标准差": round(std_dev, 6),
        "样本标准差": round(sample_std_dev, 6),
        "极差": round(data_range, 6),
        "变异系数(%)": round(cv, 6),
        "第一四分位数(Q1)": round(q1, 6),
        "第三四分位数(Q3)": round(q3, 6),
        "四分位距(IQR)": round(iqr, 6),
        "偏度": round(skewness, 6),
        "峰度": round(kurtosis, 6)
    }
    
    return results

def print_analysis(data: List[Union[int, float]]) -> None:
    """
    打印数据分析结果
    """
    print("=" * 50)
    print("数据一致性分析报告")
    print("=" * 50)
    print(f"原始数据: {data}")
    print("-" * 50)
    
    results = analyze_data_consistency(data)
    
    if "error" in results:
        print(f"错误: {results['error']}")
        return
    
    for key, value in results.items():
        print(f"{key}: {value}")
    
    # 添加一致性评估
    print("-" * 50)
    cv = results["变异系数(%)"]
    if cv < 10:
        print("一致性评估: 数据高度一致")
    elif cv < 30:
        print("一致性评估: 数据一致性较好")
    elif cv < 50:
        print("一致性评估: 数据一致性一般")
    else:
        print("一致性评估: 数据一致性较差")

# 主程序
if __name__ == "__main__":
    # 示例数据
    sample_data = [10, 12, 11, 13, 9, 14, 10, 11, 12, 13]
    
    print("示例1: 使用预设数据")
    print_analysis(sample_data)
    
    print("\n" + "=" * 50)
    print("示例2: 请输入自定义数据")
    print("=" * 50)
    
    try:
        user_input = input("请输入数据(用空格分隔，如: 1 2 3 4 5): ")
        if user_input.strip():
            user_data = [float(x) for x in user_input.split()]
            print_analysis(user_data)
        else:
            print("未输入数据，使用默认示例数据")
            print_analysis(sample_data)
    except ValueError:
        print("输入格式错误，请输入数字，用空格分隔")
    except Exception as e:
        print(f"发生错误: {e}")

    print("\n" + "=" * 50)
    print("提示: 变异系数(CV)是衡量数据相对离散程度的重要指标")
    print("CV < 10%: 高度一致")
    print("10% ≤ CV < 30%: 一致性较好") 
    print("30% ≤ CV < 50%: 一致性一般")
    print("CV ≥ 50%: 一致性较差")
    print("=" * 50)
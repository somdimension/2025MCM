import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体（避免中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def input_coordinates(prompt):
    """让用户输入一组坐标，格式为：x1,y1 x2,y2 x3,y3 ..."""
    print(prompt)
    raw_input = input("请输入坐标点（格式如：100,1.2 200,1.5 300,1.3，空格分隔）: ").strip()
    if not raw_input:
        return []
    points = []
    for pair in raw_input.split():
        try:
            x, y = map(float, pair.split(','))
            points.append((x, y))
        except ValueError:
            print(f"⚠️  坐标格式错误，跳过: {pair}")
            continue
    return points

def plot_coordinates(group1, group2):
    """绘制两组坐标点和y值平均线"""
    if not group1 and not group2:
        print("❌ 没有输入任何有效数据，无法绘图。")
        return

    # 合并所有点用于计算平均值
    all_points = group1 + group2
    all_y = [p[1] for p in all_points]
    y_mean = np.mean(all_y)

    # 拆分x和y用于绘图
    if group1:
        x1, y1 = zip(*group1)
        plt.scatter(x1, y1, color='red', label='第一组数据', s=50, alpha=0.7)
    
    if group2:
        x2, y2 = zip(*group2)
        plt.scatter(x2, y2, color='blue', label='第二组数据', s=50, alpha=0.7)

    # 绘制平均值线
    x_min = min(p[0] for p in all_points)
    x_max = max(p[0] for p in all_points)
    plt.axhline(y=y_mean, color='black', linestyle='--', linewidth=2, label=f'平均厚度: {y_mean:.4f} μm')

    # 设置坐标轴标签和标题
    plt.xlabel('波数(/cm)', fontsize=12)
    plt.ylabel('厚度(μm)', fontsize=12)
    plt.title('两组坐标数据对比图', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# 主程序
if __name__ == "__main__":
    print("📊 坐标数据绘图工具")
    print("=" * 50)

    # 输入两组数据
    group1 = input_coordinates("🔴 请输入第一组坐标（红色点）：")
    group2 = input_coordinates("🔵 请输入第二组坐标（蓝色点）：")

    # 绘图
    plot_coordinates(group1, group2)

    print("\n✅ 绘图完成！")
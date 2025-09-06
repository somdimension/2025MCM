import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 忽略拟合过程中的警告（如除零、溢出），以便程序能继续运行
warnings.filterwarnings('ignore')
input_path = r"C:\Users\czw17\Desktop\附件3.csv"

try:
    df = pd.read_csv(input_path, header = 0, names=['Wavenumber', 'Reflectance'])#有headers

    df.columns = ['Wavenumber', 'Reflectance']
    time_series = df['Reflectance']
except FileNotFoundError:
    print(f"错误：找不到文件，请检查路径是否正确: {input_path}")
    exit()


# 1. 读取数据
# df = pd.read_excel('附件3.xlsx', header=None, names=['Wavenumber', 'Reflectance'])
sigma = df['Wavenumber'].values  # 波数 (cm^-1)
R_meas = df['Reflectance'].values / 100.0  # 反射率 (转换为小数)

# 2. 定义物理模型函数
def multibeam_interference(sigma, A, B, d):
    n_s = 3.4  # 硅衬底折射率
    n_film = A + B * (sigma ** 2)  # 薄膜折射率 (Cauchy模型)
    
    # 避免出现负折射率或零折射率，增加数值稳定性
    n_film = np.maximum(n_film, 0.1)
    
    # 计算两个界面的反射系数 (振幅反射率)
    r1 = np.abs((1 - n_film) / (1 + n_film))
    r2 = np.abs((n_film - n_s) / (n_film + n_s))
    
    # 计算相位差 (小角度近似, cos(theta)≈1)
    delta = 4 * np.pi * sigma * n_film * d
    
    # 多光束干涉反射率公式
    numerator = r1**2 + r2**2 + 2 * r1 * r2 * np.cos(delta)
    denominator = 1 + (r1 * r2)**2 + 2 * r1 * r2 * np.cos(delta)
    R = numerator / denominator
    
    return R

# 3. 定义网格搜索函数
def grid_search_initial_values(sigma, R_meas, A_range, B_range, d_range, maxfev=5000):
    """
    在给定的参数范围内进行网格搜索，找到使RSS最小的最佳初始值组合。
    
    返回: 最佳参数, 最小RSS, 所有结果的DataFrame
    """
    results = []
    total_combinations = len(A_range) * len(B_range) * len(d_range)
    print(f"开始网格搜索，总共 {total_combinations} 种组合...")
    
    counter = 0
    for A_guess in A_range:
        for B_guess in B_range:
            for d_guess in d_range:
                counter += 1
                if counter % 10 == 0:
                    print(f"  已尝试 {counter}/{total_combinations} 种组合...")
                
                p0 = [A_guess, B_guess, d_guess]
                try:
                    popt, pcov = curve_fit(multibeam_interference, sigma, R_meas, p0=p0, maxfev=maxfev)
                    R_fit = multibeam_interference(sigma, *popt)
                    rss = np.sum((R_meas - R_fit) ** 2)  # 残差平方和
                    success = True
                    final_A, final_B, final_d = popt
                except:
                    # 如果拟合失败，记录一个很大的RSS
                    rss = np.inf
                    success = False
                    final_A, final_B, final_d = np.nan, np.nan, np.nan
                
                results.append({
                    'A_guess': A_guess,
                    'B_guess': B_guess,
                    'd_guess': d_guess,
                    'rss': rss,
                    'success': success,
                    'final_A': final_A,
                    'final_B': final_B,
                    'final_d': final_d
                })
    
    results_df = pd.DataFrame(results)
    # 找到RSS最小且拟合成功的行
    best_row = results_df[(results_df['success'] == True) & (results_df['rss'] < np.inf)].sort_values('rss').iloc[0]
    
    return best_row, results_df

# 4. 设置参数搜索范围
# 根据物理意义和数据特点调整这些范围
A_range = np.linspace(3.4, 3.6, 5)  # 折射率常数项 (5个点)
B_range = np.logspace(-8, -6, 5)    # 色散系数 (对数间隔, 5个点: 1e-8, 3.16e-8, 1e-7, 3.16e-7, 1e-6)
d_range = np.logspace(-5, -3, 10)   # 厚度 (cm), 对数间隔 (10个点: 0.1微米 到 100微米)

# 5. 执行网格搜索
best_result, all_results = grid_search_initial_values(sigma, R_meas, A_range, B_range, d_range)

print("\n" + "="*50)
print("网格搜索完成！")
print("="*50)
print(f"最佳初始值组合:")
print(f"  A_guess = {best_result['A_guess']:.4f}")
print(f"  B_guess = {best_result['B_guess']:.4e}")
print(f"  d_guess = {best_result['d_guess']:.4e} cm")
print(f"对应的最终拟合参数:")
print(f"  A_fit = {best_result['final_A']:.6f}")
print(f"  B_fit = {best_result['final_B']:.6e}")
print(f"  d_fit = {best_result['final_d']*1e4:.4f} 微米")
print(f"最小残差平方和 (RSS): {best_result['rss']:.6e}")

# 6. 使用最佳结果绘制拟合图
A_fit = best_result['final_A']
B_fit = best_result['final_B']
d_fit = best_result['final_d']
R_fit_best = multibeam_interference(sigma, A_fit, B_fit, d_fit)

plt.figure(figsize=(14, 10))

# 子图1: 数据与拟合曲线
plt.subplot(2, 1, 1)
plt.plot(sigma, R_meas * 100, 'o', label='实验数据 (附件3)', markersize=3, alpha=0.7)
plt.plot(sigma, R_fit_best * 100, '-', linewidth=2.5, label='最佳多光束干涉模型拟合', color='red')
plt.xlabel('波数 (cm$^{-1}$)', fontsize=12)
plt.ylabel('反射率 (%)', fontsize=12)
plt.title('硅外延层多光束干涉光谱拟合 (入射角 10°) - 最佳拟合', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# 子图2: 残差
plt.subplot(2, 1, 2)
residuals = (R_meas - R_fit_best) * 100
plt.plot(sigma, residuals, 'o-', markersize=3, linewidth=1, label='残差')
plt.axhline(y=0, color='red', linestyle='--', label='零线')
plt.xlabel('波数 (cm$^{-1}$)', fontsize=12)
plt.ylabel('残差 (%)', fontsize=12)
plt.title('拟合残差', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

# 7. (可选) 绘制RSS热力图 (固定一个参数，看另外两个)
# 例如，固定 A_guess 为最佳值，看 B_guess 和 d_guess 的组合效果
best_A_guess = best_result['A_guess']
subset = all_results[(all_results['A_guess'] == best_A_guess) & (all_results['success'] == True)]

if not subset.empty:
    plt.figure(figsize=(10, 8))
    # 创建 B_guess 和 d_guess 的网格
    B_vals = subset['B_guess'].unique()
    d_vals = subset['d_guess'].unique()
    RSS_matrix = np.full((len(d_vals), len(B_vals)), np.nan)
    
    for i, d_val in enumerate(d_vals):
        for j, B_val in enumerate(B_vals):
            rss_val = subset[(subset['B_guess'] == B_val) & (subset['d_guess'] == d_val)]['rss'].values
            if len(rss_val) > 0:
                RSS_matrix[i, j] = rss_val[0]
    
    # 绘制热力图
    im = plt.imshow(RSS_matrix, extent=[np.min(B_vals), np.max(B_vals), np.min(d_vals), np.max(d_vals)],
                    aspect='auto', origin='lower', cmap='viridis_r') # cmap_r 表示小值颜色亮
    plt.colorbar(im, label='残差平方和 (RSS)')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('B_guess (色散系数)', fontsize=12)
    plt.ylabel('d_guess (厚度, cm)', fontsize=12)
    plt.title(f'RSS 热力图 (A_guess 固定为 {best_A_guess:.4f})', fontsize=14)
    plt.scatter(best_result['B_guess'], best_result['d_guess'], color='red', s=100, marker='x', label='最佳点')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.show()
else:
    print("没有成功拟合的结果来绘制热力图。")
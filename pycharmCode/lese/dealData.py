import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pygad


df = pd.read_csv("../question1/dealfile1/附件1.csv")
x = df["波数 (cm-1)"].values
y = df["反射率 (%)"].values
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
def model(params, x):
    a, b, A, P, phi = params
    return a*x + b + A*np.sin(2*np.pi*x/P + phi)
def fitness_func(ga_instance, solution, solution_idx):
    y_pred = model(solution, x)
    mse = np.mean((y - y_pred)**2)
    return 1.0 / (mse + 1e-6)

gene_space = [
    np.linspace(-1, 1, 50),       # a
    np.linspace(0, 50, 50),       # b
    np.linspace(0, 50, 50),       # A
    np.linspace(100, 5000, 200),  # P
    np.linspace(-np.pi, np.pi, 100) # phi
]
ga_instance = pygad.GA(
    num_generations=200,
    num_parents_mating=20,
    fitness_func=fitness_func,
    sol_per_pop=50,
    num_genes=5,
    gene_space=gene_space,
    mutation_percent_genes=40,
    parent_selection_type="tournament",
    crossover_type="single_point",
    mutation_type="random"
)
ga_instance.run()
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("最优参数:", solution)
print("最佳适应度:", solution_fitness)


y_fit = model(solution, x)

plt.figure(figsize=(10,6))
plt.plot(x, y, 'k.', markersize=2, label="原始数据")
plt.plot(x, y_fit, 'r-', label="遗传算法拟合")
plt.xlabel("波数 (cm-1)")
plt.ylabel("反射率 (%)")
plt.legend()
plt.show()

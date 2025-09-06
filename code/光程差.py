import numpy as np

def 计算光程差(折射率, 厚度, 入射角):
    # 入射角是角度制不是弧度制
    入射角 = np.radians(入射角)

    折射角 = np.arcsin(np.sin(入射角) / 折射率)
    
    # 计算光程差
    光程差 = 2 * 厚度 * (折射率/np.cos(折射角)-np.sin(折射角)*np.sin(入射角))
    
    return 光程差

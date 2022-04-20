import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import spatial, integrate, interpolate, optimize
import sys

sys.path.append("Lib/scikit")
from sko.GA import GA  # 项目内置库
from sko.SA import SA  # 项目内置库
from sko.PSO import PSO  # 项目内置库
import Lib.myFunctions as my  # 项目内置库

# 测量4距离误差之和函数
def disErr_scene1(tag_loc, distance4):
    tag_loc = np.array(tag_loc)
    A0 = np.array([0, 0, 1300])
    A1 = np.array([5000, 0, 1700])
    A2 = np.array([0, 5000, 1700])
    A3 = np.array([5000, 5000, 1300])

    dis0 = np.linalg.norm(tag_loc - A0)
    dis1 = np.linalg.norm(tag_loc - A1)
    dis2 = np.linalg.norm(tag_loc - A2)
    dis3 = np.linalg.norm(tag_loc - A3)

    err = np.fabs(dis0 - distance4[0]) + np.fabs(dis1 - distance4[1]) + np.fabs(dis2 - distance4[2]) + np.fabs(dis3 - distance4[3])
    return err

fit_coefficient = np.array([[0.97058616, 145.00860654], [0.98017754, 110.75448702], [0.97744475, 135.73833774], [0.97150634, 161.11584952]])
length4 = np.array([2480, 3530, 4180, 5070])

for t in range(4):
    length4[t] = length4[t] * fit_coefficient[t, 0] + fit_coefficient[t, 1]

res = disErr_scene1([1792.57359328, 1266.62037837, 2454.36934799], [5095, 2229.66, 4990.71, 815.8 ])
print(res)





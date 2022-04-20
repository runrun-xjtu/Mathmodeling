import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import spatial, integrate, interpolate, optimize
import sys
sys.path.append("Lib/scikit")
from sko.GA import GA          # 项目内置库
from sko.SA import SA          # 项目内置库
from sko.PSO import PSO        # 项目内置库
import Lib.myFunctions as my   # 项目内置库

""" ---------------------- 第一问 ---------------------- """

# 筛选差值不等于0的点
"""
df_zheng.loc[:,'diff'] = df_zheng.loc[:,'length'] - df_zheng.loc[:,'length2']
df_zheng_diff0 = df_zheng.loc[df_zheng['diff'] != 0,:]
if df_zheng_diff0.shape[0] != 0:
    print(df_zheng_diff0.shape)
"""
# 统计异常维度不等于2的数量
"""
for t in range(4):
    if (df_W_p_std[t] < df_R_p_std[t]*2):
        nomal_num += 1
if nomal_num != 2:
    not2D_num += 1
"""

#绘制分布
"""
plt.figure()
plt.scatter(df_zheng_0['length'], df_zheng_1['length'])
plt.figure()
plt.scatter(df_zheng_2['length'], df_zheng_3['length'])
plt.show()
"""
"""
    fig = plt.figure()
    ax1 = plt.axes(projection='3d')
    plt.xlim(0, 5000)
    plt.ylim(0, 5000)
    ax1.set_zlim3d(zmin=0, zmax=5000)
    ax1.scatter(df_W_p[0]['length'], df_W_p[1]['length'], df_W_p[2]['length'])

    fig = plt.figure()
    ax1 = plt.axes(projection='3d')
    plt.xlim(0, 5000)
    plt.ylim(0, 5000)
    ax1.set_zlim3d(zmin=0, zmax=5000)
    ax1.scatter(df_W_p[0]['length'], df_W_p[1]['length'], df_W_p[3]['length'])

    fig = plt.figure()
    ax1 = plt.axes(projection='3d')
    plt.xlim(0, 5000)
    plt.ylim(0, 5000)
    ax1.set_zlim3d(zmin=0, zmax=5000)
    ax1.scatter(df_W_p[0]['length'], df_W_p[2]['length'], df_W_p[3]['length'])

    fig = plt.figure()
    ax1 = plt.axes(projection='3d')
    plt.xlim(0, 5000)
    plt.ylim(0, 5000)
    ax1.set_zlim3d(zmin=0, zmax=5000)
    ax1.scatter(df_W_p[1]['length'], df_W_p[2]['length'], df_W_p[3]['length'])
"""

# 场景1，2坐标
"""
    A0 = np.array([0, 0, 1300])
    A1 = np.array([5000, 0, 1700])
    A2 = np.array([0, 5000, 1700])
    A3 = np.array([5000, 5000, 1300])

    A0 = np.array([0, 0, 1200])
    A1 = np.array([5000, 0, 1600])
    A2 = np.array([0, 3000, 1600])
    A3 = np.array([5000, 3000, 1200])
"""
"""
    # 统计异常维度数目
    err_num, err_index_list = 0, []
    sort_mean_R = np.argsort(np.array(df_W_p_std))

    for t in range(4):
        if(np.fabs(df_R_p_mean[t] - df_W_p_mean[t]) > 20):
            err_num += 1
            err_index_list.append(t)
"""


""" ---------------------- 第二问 ---------------------- """
# y = ax 拟合
"""
def func_ax(x,a):
    return a*x
for j in range(4):
    popt, pcov = optimize.curve_fit(func_ax, A1234_array[:, j], real_dis_array[:, j])
    fit_list.append(popt)
print(fit_list)
"""
#遗传算法
"""
ga = GA(func=optfunc, n_dim=3, size_pop=50, max_iter=800, prob_mut=0.001, lb=[0, 0, 0], ub=[5000, 5000, 3000],
        precision=1e-7)
best_x, best_y = ga.run()
"""

# 神经网络
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

df_data = pd.read_excel('./输出/输出均值Excel.xlsx')
X = np.array(df_data.iloc[:, 1:5])
Y = np.array(df_data.iloc[:, 13])
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(40, 5), random_state=1)
clf.fit(X, Y)
MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(5, 2), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)
for i in range(15, 25):
       print(clf.predict([[df_data.iloc[i, 1], df_data.iloc[i, 2], df_data.iloc[i, 3], df_data.iloc[i, 4]], ]))
#print(clf.predict_proba([[2., 2.], [-1.,- 2.]]))
"""














































































import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import spatial, integrate, interpolate, optimize
import sys
from scipy.interpolate import interpn

sys.path.append("Lib/scikit")
from sko.GA import GA  # 项目内置库
from sko.SA import SA  # 项目内置库
from sko.PSO import PSO  # 项目内置库
import Lib.myFunctions as my  # 项目内置库

""" matplotlib 字体设置 """
config = {"font.family": 'serif', "font.size": 20, "mathtext.fontset": 'stix', "font.serif": ['SimSun'], }
plt.rcParams.update(config)
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

""" 第四问 """
df_gen_err = pd.read_excel('./输出/第二问遗传算法分类.xlsx')
df_gen_err2 = pd.read_excel('./输出/第二问遗传算法分类-新修正.xlsx')
df_gen_err3 = pd.read_excel('./输出/第二问遗传算法分类-sanbian.xlsx')

R_err_mean = df_gen_err['R'].mean()
R_err_std = df_gen_err['R'].std()

W1_err_mean = df_gen_err['W1'].mean()
W1_err_std = df_gen_err['W1'].std()
W2_err_mean = df_gen_err['W2'].mean()
W2_err_std = df_gen_err['W2'].std()

"""
plt.figure()
plt.xlim(0, 800)
plt.ylim(0, 800)
plt.scatter(df_gen_err['R'], df_gen_err['W1'], color='k', lw=0.5)

plt.figure()
plt.xlim(0, 800)
plt.ylim(0, 800)
plt.scatter(df_gen_err['R'], df_gen_err['W2'], color='r', lw=0.5)
"""
plt.figure()
plt.xlim(0, 600)
plt.ylim(0, 600)
plt.xlabel(r'正常数据的距离误差之和 $h$ $\rm(mm)$', fontsize=30)
plt.ylabel(r'两类异常数据的距离误差之和平均值 $h$ $\rm(mm)$', fontsize=30)
plt.xticks(np.around(np.linspace(0, 600, 7), decimals=0), fontproperties='Times New Roman', size=26)
plt.yticks(np.around(np.linspace(0, 600, 7), decimals=0), fontproperties='Times New Roman', size=26)
plt.scatter(df_gen_err['R'], (df_gen_err['W2']+df_gen_err['W1'])/2, color='k', lw=0.5)
plt.plot([0, 600], [0, 600], 'k-')
#plt.show()

df_gen_err_sort = df_gen_err.sort_values(by='R')
df_gen_err_sort.index = range(df_gen_err_sort.shape[0])
#df_gen_err_sort.to_excel('1.xlsx')
#print(df_gen_err_sort)

def error_num(x):
    lower_W1 = df_gen_err.loc[df_gen_err['W2'] < x,:]
    lower_W2 = df_gen_err.loc[df_gen_err['W1'] < x, :]
    upper_R = df_gen_err.loc[df_gen_err['R'] > x, :]
    n = lower_W1.shape[0] + lower_W2.shape[0] + upper_R.shape[0]
    #print(lower_W1.shape[0], lower_W2.shape[0], upper_R.shape[0])
    return n


i_list, f_list, ratio_list = [], [], []
for i in np.arange(1, 600, 1):
    i_list.append(i)
    f_list.append(error_num(i))
    ratio_list.append(100 * (648+324 - error_num(i))/(648+324))

#min_index = np.argmin(f_list)
#print(min_index, i_list[min_index], f_list[min_index], ratio_list[min_index])

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(i_list, f_list, 'k-', lw = 1.5)
ax2.plot(i_list, ratio_list, 'r-.', lw = 1.5)
ax1.set_xlabel(r'区分干扰点与非干扰点的 $h$ 临界值 $\rm(mm)$', fontsize=30)
ax1.set_ylabel(r'判断错误数', fontsize=30)
ax2.set_ylabel(r'判断正确率 $\rm(\%)$', color='red', fontsize = 30)
ax2.set_xlim(0, 600)
plt.xticks(np.around(np.linspace(0, 600, 7), decimals=0), fontproperties='Times New Roman', size=26)

ax2.set_ylim(0, 100)

ax2.tick_params(axis='y',colors='red')                      #坐标轴颜色
plt.show()

"""
plt.figure()
plt.xlim(0, 800)
plt.ylim(0, 800)
plt.scatter(df_gen_err2['R'], df_gen_err2['W1'], color='k', lw=0.5)

plt.figure()
plt.xlim(0, 800)
plt.ylim(0, 800)
plt.scatter(df_gen_err2['R'], df_gen_err2['W2'], color='r', lw=0.5)

plt.figure()
plt.xlim(0, 800)
plt.ylim(0, 800)
plt.scatter(df_gen_err2['R'], (df_gen_err2['W2']+df_gen_err2['W1'])/2, color='b', lw=0.5)
plt.show()

plt.figure()
plt.xlim(0, 800)
plt.ylim(0, 800)
plt.scatter(df_gen_err3['R'], df_gen_err2['W1'], color='k', lw=0.5)
plt.figure()
plt.xlim(0, 800)
plt.ylim(0, 800)
plt.scatter(df_gen_err3['R'], df_gen_err2['W2'], color='r', lw=0.5)
plt.figure()
plt.xlim(0, 800)
plt.ylim(0, 800)
plt.scatter(df_gen_err3['R'], (df_gen_err2['W2']+df_gen_err2['W1'])/2, color='b', lw=0.5)
plt.show()
"""


""" 第五问 """
df_path = pd.read_excel('./输出/第五问轨迹点坐标.xlsx')
#df_path = pd.read_excel('./输出/第五问轨迹点坐标——最终遗传.xlsx')

fig = plt.figure()
ax1 = plt.axes(projection='3d')
ax1.plot3D(df_path['x'], df_path['y'], df_path['z'], 'k-', lw=0.5)
ax1.scatter(df_path['x'], df_path['y'], df_path['z'], 'k-', lw=0.5)

fig = plt.figure()
plt.plot(range(df_path['z'].shape[0]), df_path['z'], 'k-')

plt.show()


import pylab
df_path = pd.read_excel('./输出/第五问轨迹点坐标.xlsx')
def KalmanFilter(z, n_iter=20):
    # 这里是假设A=1，H=1的情况

    # intial parameters

    sz = (n_iter,)  # size of array

    # Q = 1e-5 # process variance
    Q = 1e-6  # process variance
    # allocate space for arrays
    xhat = np.zeros(sz)  # a posteri estimate of x
    P = np.zeros(sz)  # a posteri error estimate
    xhatminus = np.zeros(sz)  # a priori estimate of x
    Pminus = np.zeros(sz)  # a priori error estimate
    K = np.zeros(sz)  # gain or blending factor

    R = 0.1 ** 2  # estimate of measurement variance, change to see effect

    # intial guesses
    xhat[0] = 0.0
    P[0] = 1.0
    A = 1
    H = 1

    for k in range(1, n_iter):
        # time update
        xhatminus[k] = A * xhat[k - 1]  # X(k|k-1) = AX(k-1|k-1) + BU(k) + W(k),A=1,BU(k) = 0
        Pminus[k] = A * P[k - 1] + Q  # P(k|k-1) = AP(k-1|k-1)A' + Q(k) ,A=1

        # measurement update
        K[k] = Pminus[k] / (Pminus[k] + R)  # Kg(k)=P(k|k-1)H'/[HP(k|k-1)H' + R],H=1
        xhat[k] = xhatminus[k] + K[k] * (z[k] - H * xhatminus[k])  # X(k|k) = X(k|k-1) + Kg(k)[Z(k) - HX(k|k-1)], H=1
        P[k] = (1 - K[k] * H) * Pminus[k]  # P(k|k) = (1 - Kg(k)H)P(k|k-1), H=1
    return xhat


if __name__ == '__main__':
    raw_data = np.array(df_path['z'].values)

    xhat = KalmanFilter(raw_data, n_iter=len(raw_data))

    pylab.plot(raw_data, 'k-', label='raw measurement')  # 测量值
    pylab.plot(xhat, 'b-', label='Kalman estimate')  # 过滤后的值
    pylab.legend()
    pylab.xlabel('Iteration')
    pylab.ylabel('ADC reading')

    xhat = np.array(xhat)[8:]
    df_path = df_path.iloc[8:,:]

    """ matplotlib 字体设置 """
    config = {"font.family": 'serif', "font.size": 20, "mathtext.fontset": 'stix', "font.serif": ['SimSun'], }
    plt.rcParams.update(config)
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    plt.figure()
    plt.xlim(-25, 575)
    plt.ylim(0, 3000)
    plt.xlabel(r'采集数据点编号', fontsize=30)
    plt.ylabel(r'Z坐标值 $\rm(mm)$', fontsize=30)
    plt.xticks(np.around(np.linspace(0, 500, 6), decimals=0), fontproperties='Times New Roman', size=26)
    plt.yticks(np.around(np.linspace(0000, 3000, 7), decimals=0), fontproperties='Times New Roman' , size=26)
    l1, = plt.plot(range(raw_data.shape[0]), raw_data, 'k-', lw=1.5)
    l2, = plt.plot(range(xhat.shape[0]),xhat,'b-', lw=1.5)
    plt.legend(handles=[l1, l2], labels=[r'原始数据', r'Kalman滤波'], loc='upper right', fontsize=18)
    plt.show()

    fig = plt.figure()
    ax1 = plt.axes(projection='3d')
    ax1.set_xlabel(r'$x$ $\rm(mm)$', fontsize=15)
    ax1.set_ylabel(r'$y$ $\rm(mm)$', fontsize=15)
    ax1.set_zlabel(r'$z$ $\rm(mm)$', fontsize=15)
    plt.xlim(0, 5000)
    plt.ylim(0, 5000)
    ax1.set_zlim3d(zmin=500, zmax=2500)
    ax1.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax1.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax1.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    plt.xticks(np.around(np.linspace(0, 5000, 6), decimals=0), fontproperties='Times New Roman', rotation=-0, size=16)
    plt.yticks(np.around(np.linspace(0, 5000, 6), decimals=0), fontproperties='Times New Roman', rotation=0, size=16)
    ax1.set_zticklabels(np.round(np.linspace(500, 2500, 5), decimals=0), fontproperties='Times New Roman', fontsize=16)
    ax1.plot3D(df_path['x'], df_path['y'], xhat, 'k-', lw=0.5)
    ax1.scatter(df_path['x'], df_path['y'], xhat, 'k-', lw=0.5)

    plt.show()
    pylab.show()

plt.figure()
plt.xlim(0, 5000)
plt.ylim(0, 5000)
plt.xlabel(r'$x$ $\rm(mm)$', fontsize=30)
plt.ylabel(r'$y$ $\rm(mm)$', fontsize=30)
plt.xticks(np.around(np.linspace(0, 5000, 6), decimals=0), fontproperties='Times New Roman', size=26)
plt.yticks(np.around(np.linspace(0, 5000, 6), decimals=0), fontproperties='Times New Roman', size=26)
plt.plot(df_path['x'], df_path['y'], 'o-', color='#005AB5', lw=0.5)
plt.show()
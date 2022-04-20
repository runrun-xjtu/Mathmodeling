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

""" 第三问 """
df_test_er = pd.read_excel('./输出/2-测试集.xlsx')
df_test_san = pd.read_excel('./输出/3-测试集.xlsx')


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
ax1.set_zticklabels(np.round(np.linspace(500, 2500, 5), decimals=0), fontproperties='Times New Roman',fontsize=16)

l1 = ax1.scatter(df_test_er.loc[range(0, 5),'x'], df_test_er.loc[range(0, 5),'y'], df_test_er.loc[range(0, 5),'z'], color='k', lw=0.5)
l2 = ax1.scatter(df_test_er.loc[range(5, 10),'x'], df_test_er.loc[range(5, 10),'y'], df_test_er.loc[range(5, 10),'z'], color='b', lw=0.5)

A0_A4 = np.array([[0, 0, 1300], [5000, 0, 1700], [0, 5000, 1700], [5000, 5000, 1300]])
l3 = ax1.scatter(A0_A4[:, 0], A0_A4[:, 1], A0_A4[:, 2], color='r', lw=5)
plt.legend(handles=[l1, l2, l3], labels=[r'无干扰锚点', r'有干扰锚点', r'锚点'], loc='upper right', fontsize=16, ncol=1)
plt.show()


fig = plt.figure()
ax1 = plt.axes(projection='3d')
ax1.set_xlabel(r'$x$ $\rm(mm)$', fontsize=15)
ax1.set_ylabel(r'$y$ $\rm(mm)$', fontsize=15)
ax1.set_zlabel(r'$z$ $\rm(mm)$', fontsize=15)
plt.xlim(0, 5000)
plt.ylim(0, 3000)
ax1.set_zlim3d(zmin=0, zmax=2500)
ax1.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax1.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax1.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
plt.xticks(np.around(np.linspace(0, 5000, 6), decimals=0), fontproperties='Times New Roman', rotation=-0, size=16)
plt.yticks(np.around(np.linspace(0, 3000, 6), decimals=0), fontproperties='Times New Roman', rotation=0, size=16)
ax1.set_zticklabels(np.round(np.linspace(0, 2500, 6), decimals=0), fontproperties='Times New Roman',fontsize=16)

l1 = ax1.scatter(df_test_san.loc[range(0, 5),'x'], df_test_san.loc[range(0, 5),'y'], df_test_san.loc[range(0, 5),'z'], color='k', lw=0.5)
l2 = ax1.scatter(df_test_san.loc[range(5, 10),'x'], df_test_san.loc[range(5, 10),'y'], df_test_san.loc[range(5, 10),'z'], color='b', lw=0.5)

A0_A4 = np.array([[0, 0, 1200], [5000, 0, 1600], [0, 3000, 1600], [5000, 3000, 1200]])
l3 = ax1.scatter(A0_A4[:, 0], A0_A4[:, 1], A0_A4[:, 2], color='r', lw=5)
plt.legend(handles=[l1, l2, l3], labels=[r'无干扰锚点', r'有干扰锚点', r'锚点'], loc='upper right', fontsize=16, ncol=1)
plt.show()

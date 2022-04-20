import sympy
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from scipy import spatial, integrate, interpolate, optimize
import sys
from scipy.interpolate import interpn
from mpl_toolkits.mplot3d import Axes3D  # 3D 绘图

df_Zerr = pd.read_excel('./输出/2-输出xyz坐标和Z误差.xlsx')
df_Zerr_zequal = [0, 0, 0, 0]
df_Zerr_zequal_mean, df_Zerr_zequal_std = [0, 0, 0, 0], [0, 0, 0, 0]

fig = plt.figure()
ax1 = plt.axes(projection='3d')
# 分类
for i in np.arange(4):
    zlist = [880, 1300, 1700, 2000]
    df_Zerr_zequal[i] = df_Zerr.loc[df_Zerr['rz'] == zlist[i], :]
    df_Zerr_zequal_mean[i] = df_Zerr_zequal[i].loc[:, 'z'].mean()
    df_Zerr_zequal_std[i] = df_Zerr_zequal[i].loc[:, 'z'].std()
# 剔除3sigema
for i in np.arange(4):
    sigema3_min = df_Zerr_zequal_mean[i] - 3 * df_Zerr_zequal_std[i]
    sigema3_max = df_Zerr_zequal_mean[i] + 3 * df_Zerr_zequal_std[i]
    df_Zerr_zequal[i] = df_Zerr_zequal[i].loc[(df_Zerr_zequal[i]['z'] > sigema3_min) & (df_Zerr_zequal[i]['z'] < sigema3_max), :]
# 剔除2
for i in np.arange(4):
    df_Zerr_zequal[i].loc[:, 'pianli'] = np.fabs(df_Zerr_zequal[i].loc[:, 'z'] - df_Zerr_zequal_mean[i])
    df_Zerr_zequal[i] = df_Zerr_zequal[i].sort_values(by='pianli')
    df_Zerr_zequal[i] = df_Zerr_zequal[i].iloc[:-15, :]

for i in np.arange(4):
    color_list = ['k', 'r', 'b', 'g']
    ax1.scatter(df_Zerr_zequal[i]['x'], df_Zerr_zequal[i]['y'], df_Zerr_zequal[i]['z'],  color=color_list[i], lw=0.5)
    ax1.plot_trisurf(df_Zerr_zequal[i]['x'], df_Zerr_zequal[i]['y'], df_Zerr_zequal[i]['z'],  color=color_list[i], lw=0.5)

ax1.set_xlabel(r'x', fontsize=13)
ax1.set_ylabel(r'y', fontsize=13)
ax1.set_zlabel(r'Z', fontsize=13)
A0_A4 = np.array([[0, 0, 1300], [5000, 0, 1700], [0, 5000, 1700], [5000, 5000, 1300]])
ax1.scatter(A0_A4[:, 0], A0_A4[:, 1], A0_A4[:, 2], color='r', lw=5)
plt.show()

x_merge_array = np.hstack((np.array(df_Zerr_zequal[0]['x'].values), np.array(df_Zerr_zequal[1]['x'].values), np.array(df_Zerr_zequal[2]['x'].values), np.array(df_Zerr_zequal[3]['x'].values)))
y_merge_array = np.hstack((np.array(df_Zerr_zequal[0]['y'].values), np.array(df_Zerr_zequal[1]['y'].values), np.array(df_Zerr_zequal[2]['y'].values), np.array(df_Zerr_zequal[3]['y'].values)))
z_merge_array = np.hstack((np.array(df_Zerr_zequal[0]['z'].values), np.array(df_Zerr_zequal[1]['z'].values), np.array(df_Zerr_zequal[2]['z'].values), np.array(df_Zerr_zequal[3]['z'].values)))
rz_merge_array = np.hstack((np.array(df_Zerr_zequal[0]['rz'].values), np.array(df_Zerr_zequal[1]['rz'].values), np.array(df_Zerr_zequal[2]['rz'].values), np.array(df_Zerr_zequal[3]['rz'].values)))

def func(x, y, z, p):
       """ 数据拟合所用的函数：z=ax+by
       :param x: 自变量 x
       :param y: 自变量 y
       :param p: 拟合参数 a, b
       """
       a, b, c, d, e, g, k = p
       return g + k * (a * x**2 + b * y**2 + c*x + d*y + e*x*y)

def residuals(p, f, x, y, z):
       """ 得到数据 z 和拟合函数之间的差
       """
       #print(p)
       return f - func(x, y, z, p)

x = x_merge_array[-60:]
y = y_merge_array[-60:]
z = z_merge_array[-60:]
f = rz_merge_array[-60:]

plsq = optimize.leastsq(residuals, np.array([0, 0, 0, 0, 0, 0, 0]), args=(f, x, y, z))  # 最小二乘法拟合 [0, 0] 为参数 a, b 初始值

a, b, c, d, e, g, k = plsq[0]  # 获得拟合结果
print("拟合结果:\na = {}".format(a))
print("b = {}".format(b))
print("c = {}".format(c))
print("d = {}".format(d))
print("g = {}".format(g))
print("k = {}".format(k))

# 绘图
xp = np.linspace(0, 5000, 20)
yp = np.linspace(0, 5000, 20)
zp = 2000

X, Y = np.meshgrid(xp, yp)
Z = X.copy() * 0 + zp

F = func(X, Y, Z, [a, b, c, d, e, g, k])  # 带入拟合得到的 a, b, c

fig = plt.figure(figsize=(8, 6))
ax = Axes3D(fig)  # 3D 绘图

ax.plot_surface(X, Y, F, alpha=0.5)
for i in np.arange(4):
    color_list = ['k', 'r', 'b', 'g']
    ax.scatter(df_Zerr_zequal[i]['x'], df_Zerr_zequal[i]['y'], df_Zerr_zequal[i]['z'],  color=color_list[i], lw=0.5)
A0_A4 = np.array([[0, 0, 1300], [5000, 0, 1700], [0, 5000, 1700], [5000, 5000, 1300]])
ax1.scatter(A0_A4[:, 0], A0_A4[:, 1], A0_A4[:, 2], color='r', lw=5)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f")
plt.show()


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

""" 第二问 """

# 1.正常、异常均值
"""
df_even = pd.read_excel('./输出/输出均值Excel.xlsx')
plt.figure()
plt.xlim(-25, 350)
plt.ylim(0, 7000)
plt.xlabel(r'采集数据点编号', fontsize=30)
plt.ylabel(r'测量距离平均值 $\rm(mm)$', fontsize=30)
plt.xticks(np.around(np.linspace(0, 300, 7), decimals=0), fontproperties='Times New Roman', size=26)
plt.yticks(np.around(np.linspace(0000, 7000, 8), decimals=0), fontproperties='Times New Roman', size=26)
l1 = plt.scatter(range(df_even.shape[0]), df_even['R1'], color='k', lw=0.5)
l2 = plt.scatter(range(df_even.shape[0]), df_even['R2'], color='r', lw=0.5)
l3 = plt.scatter(range(df_even.shape[0]), df_even['R3'], color='b', lw=0.5)
l4 = plt.scatter(range(df_even.shape[0]), df_even['R4'], color='g', lw=0.5)
plt.legend(handles=[l1, l2, l3, l4], labels=[r'A0', r'A1', r'A2', r'A3'], loc='upper right', fontsize=18, ncol=1)
plt.show()
"""
"""
columns_R = ['R1', 'R2', 'R3', 'R4']
columns_e1 = ['W11', 'W12', 'W13', 'W14']
columns_e2 = ['W21', 'W22', 'W23', 'W24']
color_list = ['k', 'r', 'b', 'g']

df_even = pd.read_excel('./输出/输出均值Excel.xlsx')

df_even_W11_y = df_even.loc[df_even['eW11'] > 100, 'W11']
df_even_W11_z = df_even.loc[df_even['eW11'] <= 100, 'W11']
df_even_W12_y = df_even.loc[df_even['eW12'] > 100, 'W12']
df_even_W12_z = df_even.loc[df_even['eW12'] <= 100, 'W12']
df_even_W13_y = df_even.loc[df_even['eW13'] > 100, 'W13']
df_even_W13_z = df_even.loc[df_even['eW13'] <= 100, 'W13']
df_even_W14_y = df_even.loc[df_even['eW14'] > 100, 'W14']
df_even_W14_z = df_even.loc[df_even['eW14'] <= 100, 'W14']

df_even_W21_y = df_even.loc[df_even['eW21'] > 100, 'W21']
df_even_W21_z = df_even.loc[df_even['eW21'] <= 100, 'W21']
df_even_W22_y = df_even.loc[df_even['eW22'] > 100, 'W22']
df_even_W22_z = df_even.loc[df_even['eW22'] <= 100, 'W22']
df_even_W23_y = df_even.loc[df_even['eW23'] > 100, 'W23']
df_even_W23_z = df_even.loc[df_even['eW23'] <= 100, 'W23']
df_even_W24_y = df_even.loc[df_even['eW24'] > 100, 'W24']
df_even_W24_z = df_even.loc[df_even['eW24'] <= 100, 'W24']

plt.xlabel(r'采集数据点编号', fontsize=30)
plt.ylabel(r'测量距离平均值 $\rm(mm)$', fontsize=30)
plt.xlim(-25, 350)
plt.ylim(0, 7500)
plt.xticks(np.around(np.linspace(0, 300, 7), decimals=0), fontproperties='Times New Roman', size=26)
plt.yticks(np.around(np.linspace(0000, 7000, 8), decimals=0), fontproperties='Times New Roman', size=26)

l2 = plt.scatter(df_even_W11_y.index.to_numpy(), df_even_W11_y,marker='*', color='k', lw=0.5)
l1 = plt.scatter(df_even_W11_z.index.to_numpy(), df_even_W11_z, color='k', lw=0.5)
l4 = plt.scatter(df_even_W12_y.index.to_numpy(), df_even_W12_y,marker='*', color='r', lw=0.5)
l3 = plt.scatter(df_even_W12_z.index.to_numpy(), df_even_W12_z, color='r', lw=0.5)
l6 = plt.scatter(df_even_W13_y.index.to_numpy(), df_even_W13_y,marker='*', color='b', lw=0.5)
l5 = plt.scatter(df_even_W13_z.index.to_numpy(), df_even_W13_z, color='b', lw=0.5)
l8 = plt.scatter(df_even_W14_y.index.to_numpy(), df_even_W14_y,marker='*', color='g', lw=0.5)
l7 = plt.scatter(df_even_W14_z.index.to_numpy(), df_even_W14_z, color='g', lw=0.5)
plt.legend(handles=[l1, l2, l3, l4, l5, l6, l7, l8], labels=[r'A0 无干扰', r'A0 有干扰', r'A1 无干扰', r'A1 有干扰', r'A2 无干扰', r'A2 有干扰', r'A3 无干扰', r'A3 有干扰'], loc='upper right', fontsize=18, ncol=1)

plt.scatter(df_even_W21_y.index.to_numpy(), df_even_W21_y,marker='*', color='k', lw=0.5)
plt.scatter(df_even_W21_z.index.to_numpy(), df_even_W21_z, color='k', lw=0.5)
plt.scatter(df_even_W22_y.index.to_numpy(), df_even_W22_y,marker='*', color='r', lw=0.5)
plt.scatter(df_even_W22_z.index.to_numpy(), df_even_W22_z, color='r', lw=0.5)
plt.scatter(df_even_W23_y.index.to_numpy(), df_even_W23_y,marker='*', color='b', lw=0.5)
plt.scatter(df_even_W23_z.index.to_numpy(), df_even_W23_z, color='b', lw=0.5)
plt.scatter(df_even_W24_y.index.to_numpy(), df_even_W24_y,marker='*', color='g', lw=0.5)
plt.scatter(df_even_W24_z.index.to_numpy(), df_even_W24_z, color='g', lw=0.5)
plt.show()
"""

# 展示 2-输出xyz坐标和Z误差.xlsx

df_Zerr = pd.read_excel('./输出/2-输出xyz坐标和Z误差.xlsx')
#df_Zerr = pd.read_excel('./输出/2-输出xyz坐标和Z误差-最终遗传算法.xlsx')

"""
fig = plt.figure()
ax1 = plt.axes(projection='3d')

df_Zerr_xequal = [0, 0, 0, 0, 0, 0, 0, 0, 0]
for i in np.arange(9):
    x = 500 * i
    df_Zerr_xequal[i] = df_Zerr.loc[df_Zerr['rx'] == x, :]
    ax1.plot3D(df_Zerr['x'], df_Zerr['y'], df_Zerr['z'],  color='k', lw=0.5)
plt.show()
"""

df_Zerr_zequal = [0, 0, 0, 0]
df_Zerr_zequal_mean, df_Zerr_zequal_std = [0, 0, 0, 0], [0, 0, 0, 0]


# 分类
for i in np.arange(4):
    zlist = [880, 1300, 1700, 2000]
    df_Zerr_zequal[i] = df_Zerr.loc[df_Zerr['rz'] == zlist[i], :]
    df_Zerr_zequal_mean[i] = df_Zerr_zequal[i].loc[:, 'z'].mean()
    df_Zerr_zequal_std[i] = df_Zerr_zequal[i].loc[:, 'z'].std()

# 剔除前的图
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
l = [0, 0, 0, 0, 0]
for i in np.arange(4):
    color_list = ['k', 'r', 'b', 'g']
    l[i] = ax1.scatter(df_Zerr_zequal[i]['x'], df_Zerr_zequal[i]['y'], df_Zerr_zequal[i]['z'],  color=color_list[i], lw=0.5)
    #ax1.plot_trisurf(df_Zerr_zequal[i]['x'], df_Zerr_zequal[i]['y'], df_Zerr_zequal[i]['z'],  color=color_list[i], lw=0.5)
A0_A4 = np.array([[0, 0, 1300], [5000, 0, 1700], [0, 5000, 1700], [5000, 5000, 1300]])
l[4] = ax1.scatter(A0_A4[:, 0], A0_A4[:, 1], A0_A4[:, 2], color='r', lw=5)
plt.legend(handles=[l[0], l[1], l[2], l[3], l[4]], labels=[r'真实值：$Z=880$', r'真实值：$Z=1300$', r'真实值：$Z=1700$', r'真实值：$Z=2000$', r'锚点'], loc='upper right',labelspacing=0, fontsize=16, ncol=1)
plt.show()

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
l = [0, 0, 0, 0, 0]
for i in np.arange(4):
    color_list = ['k', 'r', 'b', 'g']
    l[i] = ax1.scatter(df_Zerr_zequal[i]['x'], df_Zerr_zequal[i]['y'], df_Zerr_zequal[i]['z'],  color=color_list[i], lw=0.5)
    #ax1.plot_trisurf(df_Zerr_zequal[i]['x'], df_Zerr_zequal[i]['y'], df_Zerr_zequal[i]['z'],  color=color_list[i], lw=0.5)

A0_A4 = np.array([[0, 0, 1300], [5000, 0, 1700], [0, 5000, 1700], [5000, 5000, 1300]])
l[4] = ax1.scatter(A0_A4[:, 0], A0_A4[:, 1], A0_A4[:, 2], color='r', lw=5)
plt.legend(handles=[l[0], l[1], l[2], l[3], l[4]], labels=[r'真实值：$Z=880$', r'真实值：$Z=1300$', r'真实值：$Z=1700$', r'真实值：$Z=2000$', r'锚点'], loc='upper right',labelspacing=0, fontsize=16, ncol=1)
plt.show()

x_merge_array = np.hstack((np.array(df_Zerr_zequal[0]['x'].values), np.array(df_Zerr_zequal[1]['x'].values), np.array(df_Zerr_zequal[2]['x'].values), np.array(df_Zerr_zequal[3]['x'].values)))
y_merge_array = np.hstack((np.array(df_Zerr_zequal[0]['y'].values), np.array(df_Zerr_zequal[1]['y'].values), np.array(df_Zerr_zequal[2]['y'].values), np.array(df_Zerr_zequal[3]['y'].values)))
z_merge_array = np.hstack((np.array(df_Zerr_zequal[0]['z'].values), np.array(df_Zerr_zequal[1]['z'].values), np.array(df_Zerr_zequal[2]['z'].values), np.array(df_Zerr_zequal[3]['z'].values)))
rz_merge_array = np.hstack((np.array(df_Zerr_zequal[0]['rz'].values), np.array(df_Zerr_zequal[1]['rz'].values), np.array(df_Zerr_zequal[2]['rz'].values), np.array(df_Zerr_zequal[3]['rz'].values)))
data = {'x':x_merge_array, 'y':y_merge_array, 'z':z_merge_array, 'err':rz_merge_array}
df_outdata = pd.DataFrame(data)
#df_outdata.to_excel('./输出/2-输出xyz坐标和Z误差-筛选.xlsx')

"""
from scipy.interpolate import griddata
points = np.array([x_merge_array, y_merge_array, z_merge_array]).T
values = np.array(rz_merge_array)
#res = interpn(points, rz_merge_array, [500, 500, 1000], fill_value=True, bounds_error=False, method='linear')
res = griddata(points, values, [500, 500, 1700], method='linear')


fig = plt.figure()
ax1 = plt.axes(projection='3d')
ax1.scatter(df_Zerr['x'], df_Zerr['y'], df_Zerr['z'],  color='k', lw=0.5)
ax1.set_xlabel(r'x', fontsize=13)
ax1.set_ylabel(r'y', fontsize=13)
ax1.set_zlabel(r'Z', fontsize=13)


fig = plt.figure()
ax1 = plt.axes(projection='3d')
ax1.scatter(df_Zerr['x'], df_Zerr['y'], df_Zerr['errZ'],  color='r', lw=0.5)
ax1.set_xlabel(r'x', fontsize=13)
ax1.set_ylabel(r'y', fontsize=13)
ax1.set_zlabel(r'errZ', fontsize=13)
plt.show()
"""
"""
fig = plt.figure()
ax1 = plt.axes(projection='3d')
ax1.scatter(range(df_Zerr.shape[0]), df_Zerr['y'], df_Zerr['errZ'],  color='r', lw=0.5)
ax1.set_xlabel(r'NO.324', fontsize=13)
ax1.set_ylabel(r'y', fontsize=13)
ax1.set_zlabel(r'errZ', fontsize=13)
plt.show()
"""

"""
plt.figure()
plt.xlim(-25, 350)
plt.ylim(0, 5000)
plt.xlabel(r'采集数据点编号', fontsize=30)
plt.ylabel(r'预测点 $x$ 坐标 $\rm(mm)$', fontsize=30)
plt.xticks(np.around(np.linspace(0, 300, 7), decimals=0), fontproperties='Times New Roman', size=26)
plt.yticks(np.around(np.linspace(0, 5000, 6), decimals=0), fontproperties='Times New Roman', size=26)
plt.scatter(range(df_Zerr.shape[0]), df_Zerr['x'], color='k', lw=0.5)
#plt.scatter(range(df_Zerr.shape[0]), df_Zerr['errZ'], color='g', lw=0.5)

plt.figure()
plt.xlim(-25, 350)
plt.ylim(0, 5000)
plt.xlabel(r'采集数据点编号', fontsize=30)
plt.ylabel(r'预测点 $y$ 坐标 $\rm(mm)$', fontsize=30)
plt.xticks(np.around(np.linspace(0, 300, 7), decimals=0), fontproperties='Times New Roman', size=26)
plt.yticks(np.around(np.linspace(0, 5000, 6), decimals=0), fontproperties='Times New Roman', size=26)
plt.scatter(range(df_Zerr.shape[0]), df_Zerr['y'], color='r', lw=0.5)
#plt.scatter(range(df_Zerr.shape[0]), df_Zerr['errZ'], color='g', lw=0.5)

plt.figure()
plt.xlim(-25, 350)
plt.ylim(0, 2500)
plt.xlabel(r'采集数据点编号', fontsize=30)
plt.ylabel(r'预测点 $z$ 坐标 $\rm(mm)$', fontsize=30)
plt.xticks(np.around(np.linspace(0, 300, 7), decimals=0), fontproperties='Times New Roman', size=26)
plt.yticks(np.around(np.linspace(0, 2500, 6), decimals=0), fontproperties='Times New Roman', size=26)
plt.scatter(range(df_Zerr.shape[0]), df_Zerr['z'], color='b', lw=0.5)

plt.plot([0, 80], [880, 880],marker='s', color='#0080FF', lw=1.5)
plt.plot([81, 161], [1300, 1300],marker='s', color='#0080FF', lw=1.5)
plt.plot([162, 262], [1700, 1700],marker='s', color='#0080FF', lw=1.5)
l1, = plt.plot([263, 323], [2000, 2000],marker='s', color='#0080FF', lw=1.5)
plt.legend(handles=[l1], labels=[r'真实值'], loc='upper right', fontsize=16, ncol=1)
#plt.scatter(range(df_Zerr.shape[0]), df_Zerr['errZ'], color='g', lw=0.5)
plt.show()
"""
"""
plt.figure()
plt.scatter(df_Zerr['rx'], df_Zerr['errZ'], color='k', lw=0.5)
plt.figure()
plt.scatter(df_Zerr['ry'], df_Zerr['errZ'], color='r', lw=0.5)
plt.figure()
plt.scatter(df_Zerr['rz'], df_Zerr['errZ'], color='b', lw=0.5)
plt.show()
"""

# 2.神经网络修正
"""
df_Zerr_Rmodify = pd.read_excel('./输出/2.神经网络修正结果-正常.xlsx')

df_Zerr_Rmodify_zequal = [0, 0, 0, 0]
# 分类
for i in np.arange(4):
    zlist = [880, 1300, 1700, 2000]
    df_Zerr_Rmodify_zequal[i] = df_Zerr_Rmodify.loc[df_Zerr_Rmodify['真实z'] == zlist[i], :]

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
l = [0, 0, 0, 0, 0]

color_list = ['k', 'r', 'b', 'g']
for i in range(df_Zerr_Rmodify.shape[0]):
    if df_Zerr_Rmodify.loc[i, '真实z'] == 880:
        color = 'k'
    elif df_Zerr_Rmodify.loc[i, '真实z'] == 1300:
        color = 'r'
    elif df_Zerr_Rmodify.loc[i, '真实z'] == 1700:
        color = 'b'
    elif df_Zerr_Rmodify.loc[i, '真实z'] == 2000:
        color = 'g'
    l[0] = ax1.scatter(df_Zerr_Rmodify.loc[i, 'x'], df_Zerr_Rmodify.loc[i, 'y'], df_Zerr_Rmodify.loc[i, 'z'], color='k', lw=1)
    l[1] = ax1.scatter(df_Zerr_Rmodify.loc[i, 'x'],df_Zerr_Rmodify.loc[i, 'y'],df_Zerr_Rmodify.loc[i, '预测修正后z值'],marker='D', color='r', lw=1)
    ax1.plot3D([df_Zerr_Rmodify.loc[i, 'x'], df_Zerr_Rmodify.loc[i, 'x']], [df_Zerr_Rmodify.loc[i, 'y'], df_Zerr_Rmodify.loc[i, 'y']], [df_Zerr_Rmodify.loc[i, 'z'], df_Zerr_Rmodify.loc[i, '预测修正后z值']],'--',  color='r', lw=1)

xp = np.linspace(0, 5000, 11)
yp = np.linspace(0, 5000, 11)
X, Y = np.meshgrid(xp, yp)
Z1 = X.copy() * 0 + 880
Z2 = X.copy() * 0 + 1300
Z3 = X.copy() * 0 + 1700
Z4 = X.copy() * 0 + 2000

ax1.plot_surface(X, Y, Z1, alpha=0.5)
ax1.plot_surface(X, Y, Z2, alpha=0.5)
ax1.plot_surface(X, Y, Z3, alpha=0.5)
ax1.plot_surface(X, Y, Z4, alpha=0.5)

A0_A4 = np.array([[0, 0, 1300], [5000, 0, 1700], [0, 5000, 1700], [5000, 5000, 1300]])
l[2] = ax1.scatter(A0_A4[:, 0], A0_A4[:, 1], A0_A4[:, 2], color='b', lw=5)
plt.legend(handles=[l[0], l[1], l[2]], labels=[r'修正前靶点', r'修正后靶点', r'锚点'], loc='upper right', fontsize=16, ncol=1)
plt.show()
"""

# 异常数据预测 xyz坐标图

df_Zerr_e1 = pd.read_excel('./输出/2-输出xyz坐标和Z误差(异常1).xlsx')
df_Zerr_e2 = pd.read_excel('./输出/2-输出xyz坐标和Z误差(异常2).xlsx')

df_Zerr_zequal_e1, df_Zerr_zequal_e2 = [0, 0, 0, 0], [0, 0, 0, 0]
df_Zerr_zequal_mean_e1, df_Zerr_zequal_std_e1 = [0, 0, 0, 0], [0, 0, 0, 0]
df_Zerr_zequal_mean_e2, df_Zerr_zequal_std_e2 = [0, 0, 0, 0], [0, 0, 0, 0]

# 分类
for i in np.arange(4):
    zlist = [880, 1300, 1700, 2000]
    df_Zerr_zequal_e1[i] = df_Zerr_e1.loc[df_Zerr_e1['rz'] == zlist[i], :]
    df_Zerr_zequal_e2[i] = df_Zerr_e2.loc[df_Zerr_e2['rz'] == zlist[i], :]
    df_Zerr_zequal_mean_e1[i] = df_Zerr_zequal_e1[i].loc[:, 'z'].mean()
    df_Zerr_zequal_std_e1[i] = df_Zerr_zequal_e1[i].loc[:, 'z'].std()
    df_Zerr_zequal_mean_e2[i] = df_Zerr_zequal_e2[i].loc[:, 'z'].mean()
    df_Zerr_zequal_std_e2[i] = df_Zerr_zequal_e2[i].loc[:, 'z'].std()

# 剔除前的图
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
l = [0, 0, 0, 0, 0]
for i in np.arange(4):
    color_list = ['k', 'r', 'b', 'g']
    l[i] = ax1.scatter(df_Zerr_zequal_e1[i]['x'], df_Zerr_zequal_e1[i]['y'], df_Zerr_zequal_e1[i]['z'],  color=color_list[i], lw=0.5)

A0_A4 = np.array([[0, 0, 1300], [5000, 0, 1700], [0, 5000, 1700], [5000, 5000, 1300]])
l[4] = ax1.scatter(A0_A4[:, 0], A0_A4[:, 1], A0_A4[:, 2], color='r', lw=5)
plt.legend(handles=[l[0], l[1], l[2], l[3], l[4]], labels=[r'真实值：$Z=880$', r'真实值：$Z=1300$', r'真实值：$Z=1700$', r'真实值：$Z=2000$', r'靶点'], loc='upper right',labelspacing=0, fontsize=16, ncol=1)
#plt.show()

# 剔除前的图
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
l = [0, 0, 0, 0, 0]
for i in np.arange(4):
    color_list = ['k', 'r', 'b', 'g']
    l[i] = ax1.scatter(df_Zerr_zequal_e2[i]['x'], df_Zerr_zequal_e2[i]['y'], df_Zerr_zequal_e2[i]['z'],  color=color_list[i], lw=0.5)

A0_A4 = np.array([[0, 0, 1300], [5000, 0, 1700], [0, 5000, 1700], [5000, 5000, 1300]])
l[4] = ax1.scatter(A0_A4[:, 0], A0_A4[:, 1], A0_A4[:, 2], color='r', lw=5)
plt.legend(handles=[l[0], l[1], l[2], l[3], l[4]], labels=[r'真实值：$Z=880$', r'真实值：$Z=1300$', r'真实值：$Z=1700$', r'真实值：$Z=2000$', r'靶点'], loc='upper right',labelspacing=0, fontsize=16, ncol=1)
#plt.show()


# 异常出图 精度计算
df_right1 = df_Zerr_e1.loc[df_Zerr_e1['errZABS'] < 500, :]
df_right2 = df_Zerr_e2.loc[df_Zerr_e2['errZABS'] < 500, :]

num_all = df_right1.shape[0] + df_right2.shape[0]
errXall = df_right1.loc[:,'errXABS'].sum() + df_right2.loc[:,'errXABS'].sum()
errYall = df_right1.loc[:,'errYABS'].sum() + df_right2.loc[:,'errYABS'].sum()
errZall = df_right1.loc[:,'errZABS'].sum() + df_right2.loc[:,'errZABS'].sum()
err_mean = np.array([errXall, errYall, errZall]) / num_all

errX = np.hstack((df_right1.loc[:,'errXABS'].values, df_right2.loc[:,'errXABS'].values))
errY = np.hstack((df_right1.loc[:,'errYABS'].values, df_right2.loc[:,'errYABS'].values))
errZ = np.hstack((df_right1.loc[:,'errZABS'].values, df_right2.loc[:,'errZABS'].values))
errX_2, errY_2, errZ_2 = np.square(errX), np.square(errY), np.square(errZ)
XY_dis = np.sqrt(errX_2 + errY_2)
XYZ_dis = np.sqrt(errX_2 + errY_2 + errZ_2)
XY_dis_mean = np.mean(XY_dis)
XYZ_dis_mean = np.mean(XYZ_dis)
X2_mean_sqrt, Y2_mean_sqrt, Z2_mean_sqrt = np.sqrt(np.mean(errX_2)), np.sqrt(np.mean(errY_2)), np.sqrt(np.mean(errZ_2))
XY2_mean_sqrt, XYZ2_mean_sqrt = np.sqrt(np.mean(errX_2 + errY_2)), np.sqrt(np.mean(errX_2 + errY_2 + errZ_2))

plt.figure()
plt.xlim(-25, 250)
plt.ylim(0, 5000)
plt.xlabel(r'采集数据点编号', fontsize=30)
plt.ylabel(r'预测点 $x$ 坐标 $\rm(mm)$', fontsize=30)
plt.xticks(np.around(np.linspace(0, 250, 5), decimals=0),['0', '150', '300', '450', '600'] ,  fontproperties='Times New Roman', size=26)
plt.yticks(np.around(np.linspace(0, 5000, 6), decimals=0), fontproperties='Times New Roman', size=26)
plt.scatter(range(df_right1.shape[0]), df_right1['x'], color='k', lw=0.5)
plt.scatter(range(df_right2.shape[0]), df_right2['x'], color='k', lw=0.5)
#plt.scatter(range(df_Zerr.shape[0]), df_Zerr['errZ'], color='g', lw=0.5)

plt.figure()
plt.xlim(-25, 250)
plt.ylim(0, 5000)
plt.xlabel(r'采集数据点编号', fontsize=30)
plt.ylabel(r'预测点 $y$ 坐标 $\rm(mm)$', fontsize=30)
plt.xticks(np.around(np.linspace(0, 250, 5), decimals=0), ['0', '150', '300', '450', '600'] , fontproperties='Times New Roman', size=26)
plt.yticks(np.around(np.linspace(0, 5000, 6), decimals=0), fontproperties='Times New Roman', size=26)
plt.scatter(range(df_right1.shape[0]), df_right1['y'], color='r', lw=0.5)
plt.scatter(range(df_right2.shape[0]), df_right2['y'], color='r', lw=0.5)
#plt.scatter(range(df_Zerr.shape[0]), df_Zerr['errZ'], color='g', lw=0.5)

plt.figure()
plt.xlim(-25, 250)
plt.ylim(0, 2500)
plt.xlabel(r'采集数据点编号', fontsize=30)
plt.ylabel(r'预测点 $z$ 坐标 $\rm(mm)$', fontsize=30)
plt.xticks(np.around(np.linspace(0, 250, 5), decimals=0), ['0', '150', '300', '450', '600'], fontproperties='Times New Roman', size=26)
plt.yticks(np.around(np.linspace(0, 2500, 6), decimals=0), fontproperties='Times New Roman', size=26)
plt.scatter(range(df_right1.shape[0]), df_right1['z'], color='b', lw=0.5)
plt.scatter(range(df_right2.shape[0]), df_right2['z'], color='b', lw=0.5)

plt.plot([0, 50], [880, 880],marker='s', color='#0080FF', lw=1.5)
plt.plot([50, 100], [1300, 1300],marker='s', color='#0080FF', lw=1.5)
plt.plot([100, 150], [1700, 1700],marker='s', color='#0080FF', lw=1.5)
l1, = plt.plot([150, 200], [2000, 2000],marker='s', color='#0080FF', lw=1.5)
plt.legend(handles=[l1], labels=[r'真实值'], loc='upper right', fontsize=16, ncol=1)
#plt.scatter(range(df_Zerr.shape[0]), df_Zerr['errZ'], color='g', lw=0.5)
plt.show()



df_Zerr_e1 = pd.read_excel('./输出/2-输出xyz坐标和Z误差(异常1).xlsx')
df_Zerr_e2 = pd.read_excel('./输出/2-输出xyz坐标和Z误差(异常2).xlsx')

z_list = []
for i in range(323):
    z1 = df_Zerr_e1.loc[i, 'z']
    z2 = df_Zerr_e2.loc[i, 'z']
    z_list.append(z1)
    z_list.append(z2)

z_list = np.array(z_list)

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
    raw_data = np.array(z_list)
    plt.ylim(500, 2500)
    plt.scatter(range(raw_data.shape[0]), z_list)
    plt.show()

    xhat = KalmanFilter(raw_data, n_iter=len(raw_data))

    pylab.plot(raw_data, 'k-', label='raw measurement')  # 测量值
    pylab.plot(xhat, 'b-', label='Kalman estimate')  # 过滤后的值
    pylab.legend()
    pylab.xlabel('Iteration')
    pylab.ylabel('ADC reading')

    xhat = np.array(xhat)[8:]
    pylab.show()
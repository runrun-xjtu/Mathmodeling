import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Lib.myFunctions as my  # 项目内置库

""" matplotlib 字体设置 """
config = {"font.family": 'serif', "font.size": 20, "mathtext.fontset": 'stix', "font.serif": ['SimSun'], }
plt.rcParams.update(config)
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 2.3D预测点分布图 —— 展示分布情况2
"""
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
plt.legend(handles=[l[0], l[1], l[2], l[3], l[4]], labels=[r'真实值：$Z=880$', r'真实值：$Z=1300$', r'真实值：$Z=1700$', r'真实值：$Z=2000$', r'锚点'], loc='upper right', fontsize=16, ncol=1)
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
plt.legend(handles=[l[0], l[1], l[2], l[3], l[4]], labels=[r'真实值：$Z=880$', r'真实值：$Z=1300$', r'真实值：$Z=1700$', r'真实值：$Z=2000$', r'锚点'], loc='upper right', fontsize=16, ncol=1)
plt.show()
"""
# 预测点 xyz 坐标 —— 展示分布情况2
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
plt.legend(handles=[l[0], l[1], l[2]], labels=[r'修正前锚点', r'修正后锚点', r'靶点'], loc='upper right', fontsize=16, ncol=1)
plt.show()
"""
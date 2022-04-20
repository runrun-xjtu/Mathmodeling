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

""" matplotlib 字体设置 """
config = {"font.family": 'serif', "font.size": 20, "mathtext.fontset": 'stix', "font.serif": ['SimSun'], }
plt.rcParams.update(config)
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

""" 第一问 """
def distance4(tag_loc):
    tag_loc = np.array(tag_loc)
    A0 = np.array([0, 0, 1300])
    A1 = np.array([5000, 0, 1700])
    A2 = np.array([0, 5000, 1700])
    A3 = np.array([5000, 5000, 1300])
    return [np.linalg.norm(tag_loc - A0), np.linalg.norm(tag_loc - A1), np.linalg.norm(tag_loc - A2), np.linalg.norm(tag_loc - A3)]

df_tag = pd.read_excel('./Data/Tag坐标信息.xlsx')
df_W_deletenum = pd.read_table('./输出/异常数据删除数.txt', sep=' ', names=['x1', 'x2', 'x3', 'num'])
df_W_deletenum_sort = df_W_deletenum.sort_values(by='num', ascending=False)
top10_index = df_W_deletenum_sort.index[:10]

""" 1.tag轨迹图 """

A0_A4 = np.array([[0, 0, 1300], [5000, 0, 1700], [0, 5000, 1700], [5000, 5000, 1300]])
fig = plt.figure()
ax1 = plt.axes(projection='3d')
ax1.set_xlabel(r'$x$ $\rm(mm)$', fontsize=16)
ax1.set_ylabel(r'$y$ $\rm(mm)$', fontsize=16)
ax1.set_zlabel(r'$z$ $\rm(mm)$', fontsize=16)
plt.xlim(0, 5000)
plt.ylim(0, 5000)
ax1.set_zlim3d(zmin=800, zmax=2000)
ax1.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax1.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax1.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
plt.xticks(np.around(np.linspace(0, 5000, 6), decimals=0), fontproperties='Times New Roman', rotation=-0, size=16)
plt.yticks(np.around(np.linspace(0, 5000, 6), decimals=0), fontproperties='Times New Roman', rotation=0, size=16)
ax1.set_zticklabels(np.round(np.linspace(800, 2000, 7), decimals=2), fontproperties='Times New Roman',fontsize=16)

l1 = ax1.scatter(df_tag.loc[0,'x'], df_tag.loc[0,'y'], df_tag.loc[0,'z'], color='b', lw=2.5)
ax1.scatter(df_tag.loc[323,'x'], df_tag.loc[323,'y'], df_tag.loc[323,'z'], color='b', lw=2.5)
ax1.plot3D(df_tag['x'], df_tag['y'], df_tag['z'], 'ok-')
l2 = ax1.scatter(df_tag.loc[32,'x'], df_tag.loc[32,'y'], df_tag.loc[32,'z'], color='k', lw=2.5)
l3 = ax1.scatter(A0_A4[:, 0], A0_A4[:, 1], A0_A4[:, 2], color='r', lw=2.5)
plt.legend(handles=[l1, l2, l3], labels=[r'起点与终点', r'运动轨迹', r'锚点'],
           loc='upper right', fontsize=18, ncol=1)
plt.show()

A1, A2, A3, A4 = [], [], [], []

# np.arange(0, 20) [0, 23, 99, 108]
for i in [0, 23, 99, 108]:
    filename_R = './Data/正常数据/' + str(i+1) + '.正常.txt'
    filename_W = './Data/异常数据/' + str(i+1) + '.异常.txt'
    print(filename_R, filename_W)

    filename_R_qu = './输出/第一问excel/' + str(i+1) + '.正常.xlsx'
    filename_W_qu = './输出/第一问excel/' + str(i+1) + '.异常.xlsx'

    df_R_qu = pd.read_excel(filename_R_qu)
    df_W_qu = pd.read_excel(filename_W_qu)

    df_R = pd.read_table(filename_R, sep=':', skiprows=1, names=['x1', 'time', 'x2', 'x3', 'type', 'length', 'length2', 'x4', 'No.'])
    df_W = pd.read_table(filename_W, sep=':', skiprows=1, names=['x1', 'time', 'x2', 'x3', 'type', 'length', 'length2', 'x4', 'No.'])
    df_R = df_R.iloc[:, [1, 4, 5, 8]].astype(float)
    df_W = df_W.iloc[:, [1, 4, 5, 8]].astype(float)

    # 按照点的类型分类
    df_R_p, df_W_p = [0, 0, 0, 0], [0, 0, 0, 0]
    for t in range(4):
        df_R_p[t] = df_R.loc[df_R['type'] == t, :]
        df_R_p[t].index = np.arange(df_R_p[t].shape[0])
        df_W_p[t] = df_W.loc[df_W['type'] == t, :]
        df_W_p[t].index = np.arange(df_W_p[t].shape[0])

    # 检查是否有重复点
    """
    for x in range(df_R_p[t].shape[0]-1):
        if (np.all(df_R_p[0].loc[x,:] == df_R_p[0].loc[x+1,:])) and (np.all(df_R_p[1].loc[x,:] == df_R_p[1].loc[x+1,:])) and (np.all(df_R_p[2].loc[x,:] == df_R_p[2].loc[x+1,:])) and (np.all(df_R_p[3].loc[x,:] == df_R_p[3].loc[x+1,:])):
            print('same')
    """

    # 求测量与实际准确的偏差
    x0, y0, z0 = df_tag.loc[i, 'x'], df_tag.loc[i, 'y'], df_tag.loc[i, 'z']
    real_dis = np.array(distance4([x0, y0, z0]))

    # 求平均值, 标准差
    df_R_p_mean, df_W_p_mean, df_R_p_std, df_W_p_std = [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]
    for t in range(4):
        df_R_p_mean[t] = df_R_p[t].loc[:,'length'].mean()
        df_W_p_mean[t] = df_W_p[t].loc[:,'length'].mean()
        df_R_p_std[t] = df_R_p[t].loc[:,'length'].std()
        df_W_p_std[t] = df_W_p[t].loc[:,'length'].std()
    # 统计异常维度数目:使用方差排序取较大两个
    err_num, nomal_num = 2, 0
    sort_mean_R = np.argsort(np.array(df_W_p_std))
    err_index_list = sort_mean_R[[-1, -2]]

    # 求324x4个距离
    A1.append(df_R_p_mean[0])
    A2.append(df_R_p_mean[1])
    A3.append(df_R_p_mean[2])
    A4.append(df_R_p_mean[3])
    #print(err_dis)

    # 1.正常清理前
    """ 1.正常清理前 """
    """
    fig = plt.figure()
    if i==23:
        plt.ylim(0, 150)
        plt.ylim(0, 5000)
        plt.xticks(np.around(np.linspace(0, 150, 6), decimals=0), fontproperties='Times New Roman', size=26)
        plt.yticks(np.around(np.linspace(0, 5000, 6), decimals=0), fontproperties='Times New Roman', size=26)
    if i == 108:
        plt.ylim(0, 300)
        plt.ylim(0, 5500)
        plt.xticks(np.around(np.linspace(0, 300, 6), decimals=0), fontproperties='Times New Roman', size=26)
        plt.yticks(np.around(np.linspace(0, 5000, 6), decimals=0), fontproperties='Times New Roman', size=26)
    plt.xlabel(r'采集数据点编号', fontsize=30)
    plt.ylabel(r'测量距离 $\rm(mm)$', fontsize=30)

    l1 = plt.scatter(np.arange(df_R_p[0].shape[0]), df_R_p[0]['length'], color='k', lw=0.5)
    l2 = plt.scatter(np.arange(df_R_p[1].shape[0]), df_R_p[1]['length'], color='r', lw=0.5)
    l3 = plt.scatter(np.arange(df_R_p[2].shape[0]), df_R_p[2]['length'], color='b', lw=0.5)
    l4 = plt.scatter(np.arange(df_R_p[3].shape[0]), df_R_p[3]['length'], color='g', lw=0.5)

    #plt.plot(np.arange(df_R_p[0].shape[0]), real_dis[0] * np.ones(df_R_p[0].shape[0]), color='k', lw=1)
    #plt.plot(np.arange(df_R_p[0].shape[0]), real_dis[1] * np.ones(df_R_p[0].shape[0]), color='r', lw=1)
    #plt.plot(np.arange(df_R_p[0].shape[0]), real_dis[2] * np.ones(df_R_p[0].shape[0]), color='b', lw=1)
    #plt.plot(np.arange(df_R_p[0].shape[0]), real_dis[3] * np.ones(df_R_p[0].shape[0]), color='g', lw=1)
    if i==23 or 108:
        plt.legend(handles=[l1, l2, l3, l4], labels=[r'A0', r'A1', r'A2', r'A3'], loc='lower right', fontsize=26, ncol=1)
    plt.show()
    """
    # 1.异常清理前
    """ 1.异常清理前 """
    """
    fig = plt.figure()
    if i==0:
        plt.xlim(0, 400)
        plt.ylim(0, 6500)
        plt.xticks(np.around(np.linspace(0, 400, 6), decimals=0), fontproperties='Times New Roman', size=26)
        plt.yticks(np.around(np.linspace(0, 6000, 7), decimals=0), fontproperties='Times New Roman', size=26)
    if i == 99:
        plt.ylim(0, 300)
        plt.ylim(0, 6000)
        plt.xticks(np.around(np.linspace(0, 300, 6), decimals=0), fontproperties='Times New Roman', size=26)
        plt.yticks(np.around(np.linspace(0, 6000, 7), decimals=0), fontproperties='Times New Roman', size=26)
    plt.xlabel(r'采集数据点编号', fontsize=30)
    plt.ylabel(r'测量距离 $\rm(mm)$', fontsize=30)

    l1 = plt.scatter(np.arange(df_W_p[0].shape[0]), df_W_p[0]['length'], color='k', lw=0.5)
    l2 = plt.scatter(np.arange(df_W_p[1].shape[0]), df_W_p[1]['length'], color='r', lw=0.5)
    l3 = plt.scatter(np.arange(df_W_p[2].shape[0]), df_W_p[2]['length'], color='b', lw=0.5)
    l4 = plt.scatter(np.arange(df_W_p[3].shape[0]), df_W_p[3]['length'], color='g', lw=0.5)

    #plt.plot(np.arange(df_R_p[0].shape[0]), real_dis[0] * np.ones(df_R_p[0].shape[0]), color='k', lw=1)
    #plt.plot(np.arange(df_R_p[0].shape[0]), real_dis[1] * np.ones(df_R_p[0].shape[0]), color='r', lw=1)
    #plt.plot(np.arange(df_R_p[0].shape[0]), real_dis[2] * np.ones(df_R_p[0].shape[0]), color='b', lw=1)
    #plt.plot(np.arange(df_R_p[0].shape[0]), real_dis[3] * np.ones(df_R_p[0].shape[0]), color='g', lw=1)
    if i==0:
        plt.legend(handles=[l1, l2, l3, l4], labels=[r'A0', r'A1', r'A2', r'A3'], loc='center right', fontsize=26, ncol=1)
    if i == 99:
        plt.legend(handles=[l1, l2, l3, l4], labels=[r'A0', r'A1', r'A2', r'A3'], loc='lower right', fontsize=26, ncol=1)
    plt.show()
    """
    """
    # 1.正常清理后
    fig = plt.figure()
    if i==23:
        plt.ylim(0, 150)
        plt.ylim(1000, 5000)
        plt.xticks(np.around(np.linspace(0, 150, 6), decimals=0), fontproperties='Times New Roman', size=26)
        plt.yticks(np.around(np.linspace(1000, 5000, 6), decimals=0), fontproperties='Times New Roman', size=26)
    if i == 108:
        plt.ylim(0, 300)
        plt.ylim(0, 5500)
        plt.xticks(np.around(np.linspace(0, 300, 6), decimals=0), fontproperties='Times New Roman', size=26)
        plt.yticks(np.around(np.linspace(0, 5000, 6), decimals=0), fontproperties='Times New Roman', size=26)
    plt.xlabel(r'采集数据点编号', fontsize=30)
    plt.ylabel(r'测量距离 $\rm(mm)$', fontsize=30)
    l1 = plt.scatter(np.arange(df_R_qu.shape[0]), df_R_qu['A0'], color='k', lw=0.5)
    l2 = plt.scatter(np.arange(df_R_qu.shape[0]), df_R_qu['A1'], color='r', lw=0.5)
    l3 = plt.scatter(np.arange(df_R_qu.shape[0]), df_R_qu['A2'], color='b', lw=0.5)
    l4 = plt.scatter(np.arange(df_R_qu.shape[0]), df_R_qu['A3'], color='g', lw=0.5)
    if i == 23 or 108:
        plt.legend(handles=[l1, l2, l3, l4], labels=[r'A0', r'A1', r'A2', r'A3'], loc='lower right', fontsize=26,ncol=1)
    plt.show()
    """
    # 舍弃
    fig = plt.figure()
    """
    plt.xlabel(r'采集数据点编号', fontsize=30)
    plt.ylabel(r'测量距离 $\rm(mm)$', fontsize=30)

    plt.scatter(np.arange(df_W_qu.shape[0]), df_W_qu['A0'], color='k', lw=0.5)
    plt.scatter(np.arange(df_W_qu.shape[0]), df_W_qu['A1'], color='r', lw=0.5)
    plt.scatter(np.arange(df_W_qu.shape[0]), df_W_qu['A2'], color='b', lw=0.5)
    plt.scatter(np.arange(df_W_qu.shape[0]), df_W_qu['A3'], color='g', lw=0.5)

    plt.plot(np.arange(df_R_qu.shape[0]), df_R_qu['A0'], color='k', lw=0.5)
    plt.plot(np.arange(df_R_qu.shape[0]), df_R_qu['A1'], color='r', lw=0.5)
    plt.plot(np.arange(df_R_qu.shape[0]), df_R_qu['A2'], color='b', lw=0.5)
    plt.plot(np.arange(df_R_qu.shape[0]), df_R_qu['A3'], color='g', lw=0.5)
    
    plt.show()
    """
A1 = np.array(A1)
A2 = np.array(A2)
A3 = np.array(A3)
A4 = np.array(A4)

# x-正常数据324点；y-4个距离变化.png
print('x-正常数据324点；y-4个距离变化.png')
plt.figure()
plt.plot(np.arange(A1.shape[0]), A1, color='k', lw=0.5)
plt.plot(np.arange(A2.shape[0]), A2, color='r', lw=0.5)
plt.plot(np.arange(A3.shape[0]), A3, color='b', lw=0.5)
plt.plot(np.arange(A4.shape[0]), A4, color='g', lw=0.5)
plt.show()







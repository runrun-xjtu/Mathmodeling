import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import spatial, integrate, interpolate, optimize
import sys
sys.path.append("Lib/scikit")
from sko.GA import GA  # 用户自定义库
from sko.SA import SA  # 项目内置库
from sko.PSO import PSO  # 项目内置库
import Lib.myFunctions as my  # 项目内置库

""" matplotlib 字体设置 """
config = {"font.family": 'serif', "font.size": 20, "mathtext.fontset": 'stix', "font.serif": ['SimSun'], }
plt.rcParams.update(config)
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def distance4(tag_loc):
    tag_loc = np.array(tag_loc)
    A0 = np.array([0, 0, 1300])
    A1 = np.array([5000, 0, 1700])
    A2 = np.array([0, 5000, 1700])
    A3 = np.array([5000, 5000, 1300])
    return [np.linalg.norm(tag_loc - A0), np.linalg.norm(tag_loc - A1), np.linalg.norm(tag_loc - A2), np.linalg.norm(tag_loc - A3)]

df_tag = pd.read_excel('./Data/Tag坐标信息.xlsx')

""" 第一问 """
not2D_num = 0
sigema3_R_min = np.zeros((324, 4))
sigema3_R_max = np.zeros((324, 4))
sigema3_W_e1_min = np.zeros((324, 4))
sigema3_W_e1_max = np.zeros((324, 4))
sigema3_W_e2_min = np.zeros((324, 4))
sigema3_W_e2_max = np.zeros((324, 4))

A1_e1, A2_e1, A3_e1, A4_e1 = [], [], [], []
A1_e2, A2_e2, A3_e2, A4_e2 = [], [], [], []

#[0, 23, 99, 108] range(324) [308, 65, 279, 199, 250, 101, 106, 263, 112, 128]
for i in range(324):
    filename_R = './Data/正常数据/' + str(i+1) + '.正常.txt'
    filename_W = './Data/异常数据/' + str(i+1) + '.异常.txt'

    df_R = pd.read_table(filename_R, sep=':', skiprows=1, names=['x1', 'time', 'x2', 'tagID', 'type', 'length', 'length2', 'Serial No.', 'No.'])
    df_W = pd.read_table(filename_W, sep=':', skiprows=1, names=['x1', 'time', 'x2', 'tagID', 'type', 'length', 'length2', 'Serial No.', 'No.'])
    df_R = df_R.iloc[:, [1, 3, 4, 5, 7, 8]].astype(float)
    df_W = df_W.iloc[:, [1, 3, 4, 5, 7, 8]].astype(float)

    # 按照点的类型分类
    df_R_p, df_W_p = [0, 0, 0, 0], [0, 0, 0, 0]
    for t in range(4):
        df_R_p[t] = df_R.loc[df_R['type'] == t, :]
        df_R_p[t].index = np.arange(df_R_p[t].shape[0])
        df_W_p[t] = df_W.loc[df_W['type'] == t, :]
        df_W_p[t].index = np.arange(df_W_p[t].shape[0])

    # 求平均值, 标准差
    df_R_p_mean, df_W_p_mean, df_R_p_std, df_W_p_std = [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]
    for t in range(4):
        df_R_p_mean[t] = df_R_p[t].loc[:,'length'].mean()
        df_W_p_mean[t] = df_W_p[t].loc[:,'length'].mean()
        df_R_p_std[t] = df_R_p[t].loc[:,'length'].std()
        df_W_p_std[t] = df_W_p[t].loc[:,'length'].std()

    # 求测量与实际准确的偏差
    x0, y0, z0 = df_tag.loc[i, 'x'], df_tag.loc[i, 'y'], df_tag.loc[i, 'z']
    real_dis = np.array(distance4([x0, y0, z0]))
    measure_dis = np.array([df_R_p_mean[0], df_R_p_mean[1], df_R_p_mean[2], df_R_p_mean[3]])
    err_dis = np.round((measure_dis - real_dis), 2)
    #print(err_dis)

    """  ----- 正常数据处理 -----  """
    # 求正常数据.txt 3sigema 内的保留数据
    cut_array_R = np.array([])
    for t in range(4):
        sigema3_R_min[i,t] = df_R_p_mean[t] - 3 * df_R_p_std[t]
        sigema3_R_max[i,t] = df_R_p_mean[t] + 3 * df_R_p_std[t]
        out3sigema_R_index = df_R_p[t].loc[(df_R_p[t]['length'] < sigema3_R_min[i,t]) | (df_R_p[t]['length'] > sigema3_R_max[i,t]), 'length'].index.to_numpy()
        cut_array_R = np.append(cut_array_R, out3sigema_R_index)
    cut_array_R = np.unique(cut_array_R)
    left_array_R = np.setdiff1d(df_R_p[0].index.to_numpy(), cut_array_R)
    df_R_left = [0, 0, 0, 0]
    for t in range(4):
        df_R_left[t] = df_R_p[t].loc[left_array_R, :]
    #print('编号', i+1, '正常数据删除数：', -(df_R_left[t].shape[0] - df_R_p[t].shape[0]))
    #print('编号', i + 1, '正常数据中保留个数：', df_R_left[t].shape[0])
    df_R_final = df_R_left[0].loc[:, ['tagID', 'Serial No.', 'No.']]
    df_R_final.loc[:, 'A0'] = df_R_left[0]['length']
    df_R_final.loc[:, 'A1'] = df_R_left[1]['length']
    df_R_final.loc[:, 'A2'] = df_R_left[2]['length']
    df_R_final.loc[:, 'A3'] = df_R_left[3]['length']

    """  ----- 异常数据处理 -----  """
    # 统计异常维度数目:使用方差排序取较大两个
    err_num, nomal_num = 2, 0
    sort_mean_R = np.argsort(np.array(df_W_p_std))
    err_index_list = sort_mean_R[[-1, -2]]

    # 统计异常维度下，去除正常数据标准 3sigema 内数据后的点个数
    out3sigema_array_W, in3sigema_array_W = [0, 0, 0, 0], [0, 0, 0, 0]
    out3sigema_array_W_numlist, in3sigema_array_W_numlist, err_pointnum_list = [], [], []
    for t in err_index_list:
        out3sigema_W_index = df_W_p[t].loc[(df_W_p[t]['length'] < (df_R_p_mean[t] - 3 * df_R_p_std[t])) | (df_W_p[t]['length'] > (df_R_p_mean[t] + 3 * df_R_p_std[t])), 'length'].index.to_numpy()
        out3sigema_array_W[t] = out3sigema_W_index                                             # 3σ外
        in3sigema_array_W[t] = np.setdiff1d(df_W_p[t].index.to_numpy(), out3sigema_array_W[t]) # 3σ内
        out3sigema_array_W_numlist.append(out3sigema_array_W[t].shape[0])                      # 3σ外计数
        in3sigema_array_W_numlist.append(in3sigema_array_W[t].shape[0])                        # 3σ内计数
        err_pointnum_list.append(df_W_p[t].shape[0])                                           # 异常维度的点总数计数
    double_point_num = out3sigema_array_W_numlist[0] + out3sigema_array_W_numlist[1] - err_pointnum_list[0]

    e1, e2 = err_index_list[0], err_index_list[1]
    if out3sigema_array_W_numlist[0] < 40:
        out3sigema_array_W[e1] = in3sigema_array_W[e2]
        out3sigema_array_W_numlist[0] = out3sigema_array_W[e1].shape[0]
        double_point_num = out3sigema_array_W_numlist[0] + out3sigema_array_W_numlist[1] - err_pointnum_list[0]
    if out3sigema_array_W_numlist[1] < 40:
        out3sigema_array_W[e2] = in3sigema_array_W[e1]
        out3sigema_array_W_numlist[1] = out3sigema_array_W[e2].shape[0]
        double_point_num = out3sigema_array_W_numlist[0] + out3sigema_array_W_numlist[1] - err_pointnum_list[0]

    # 异常点中去除同时异常点，然后计算平均值与标准差，3σ筛选
    e1, e2 = err_index_list[0], err_index_list[1]
    jiaoji = np.intersect1d(out3sigema_array_W[e1], out3sigema_array_W[e2])
    out3sigema_array_W[e1] = np.setdiff1d(out3sigema_array_W[e1], jiaoji)
    out3sigema_array_W[e2] = np.setdiff1d(out3sigema_array_W[e2], jiaoji)
    df_W_p_e1, df_W_p_e2 = [0, 0, 0, 0], [0, 0, 0, 0]
    for t in range(4):
        df_W_p_e1[t] = df_W_p[t].loc[out3sigema_array_W[e1], :]
        df_W_p_e2[t] = df_W_p[t].loc[out3sigema_array_W[e2], :]

    df_W_p_e1_mean, df_W_p_e2_mean, df_W_p_e1_std, df_W_p_e2_std = [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]
    for t in range(4):
        df_W_p_e1_mean[t] = df_W_p_e1[t].loc[:,'length'].mean()
        df_W_p_e2_mean[t] = df_W_p_e2[t].loc[:, 'length'].mean()
        df_W_p_e1_std[t] = df_W_p_e1[t].loc[:,'length'].std()
        df_W_p_e2_std[t] = df_W_p_e2[t].loc[:, 'length'].std()

    cut_array_W_e1, cut_array_W_e2 = np.array([]), np.array([])
    for t in range(4):
        sigema3_W_e1_min[i, t] = df_W_p_e1_mean[t] - 3 * df_W_p_e1_std[t]
        sigema3_W_e1_max[i, t] = df_W_p_e1_mean[t] + 3 * df_W_p_e1_std[t]
        sigema3_W_e2_min[i, t] = df_W_p_e2_mean[t] - 3 * df_W_p_e2_std[t]
        sigema3_W_e2_max[i, t] = df_W_p_e2_mean[t] + 3 * df_W_p_e2_std[t]
        out3sigema_W_e1_index = df_W_p_e1[t].loc[(df_W_p_e1[t]['length'] < sigema3_W_e1_min[i, t]) | (df_W_p_e1[t]['length'] > sigema3_W_e1_max[i, t]), 'length'].index.to_numpy()
        out3sigema_W_e2_index = df_W_p_e2[t].loc[(df_W_p_e2[t]['length'] < sigema3_W_e2_min[i, t]) | (df_W_p_e2[t]['length'] > sigema3_W_e2_max[i, t]), 'length'].index.to_numpy()
        cut_array_W_e1 = np.append(cut_array_W_e1, out3sigema_W_e1_index)
        cut_array_W_e2 = np.append(cut_array_W_e2, out3sigema_W_e2_index)
    cut_array_W_e1 = np.unique(cut_array_W_e1)
    cut_array_W_e2 = np.unique(cut_array_W_e2)

    cut_array_W = np.union1d(cut_array_W_e1, cut_array_W_e2)
    left_array_W_e1 = np.setdiff1d(out3sigema_array_W[e1], cut_array_W)
    left_array_W_e2 = np.setdiff1d(out3sigema_array_W[e2], cut_array_W)
    left_array_W = np.union1d(left_array_W_e1, left_array_W_e2)

    #print(cut_array_W_e1, cut_array_W_e2)
    df_W_left = [0, 0, 0, 0]
    for t in range(4):
        df_W_left[t] = df_W_p[t].loc[left_array_W, :]
    df_W_left_e1, df_W_left_e2 = [0, 0, 0, 0], [0, 0, 0, 0]
    for t in range(4):
        df_W_left_e1[t] = df_W_p[t].loc[left_array_W_e1, :]
        df_W_left_e2[t] = df_W_p[t].loc[left_array_W_e2, :]
    #print('编号', i+1, '异常数据删除数：', -(df_W_left[t].shape[0] - df_W_p[t].shape[0]))
    #print('编号', i+1, '异常数据中保留个数：', df_W_left[t].shape[0])
    df_W_final = df_W_left[0].loc[:, ['tagID', 'Serial No.', 'No.']]
    df_W_final.loc[:, 'A0'] = df_W_left[0]['length']
    df_W_final.loc[:, 'A1'] = df_W_left[1]['length']
    df_W_final.loc[:, 'A2'] = df_W_left[2]['length']
    df_W_final.loc[:, 'A3'] = df_W_left[3]['length']

    df_W_final_e1 = df_W_left_e1[0].loc[:, ['tagID', 'Serial No.', 'No.']]
    df_W_final_e1.loc[:, 'A0'] = df_W_left_e1[0]['length']
    df_W_final_e1.loc[:, 'A1'] = df_W_left_e1[1]['length']
    df_W_final_e1.loc[:, 'A2'] = df_W_left_e1[2]['length']
    df_W_final_e1.loc[:, 'A3'] = df_W_left_e1[3]['length']
    df_W_final_e2 = df_W_left_e2[0].loc[:, ['tagID', 'Serial No.', 'No.']]
    df_W_final_e2.loc[:, 'A0'] = df_W_left_e2[0]['length']
    df_W_final_e2.loc[:, 'A1'] = df_W_left_e2[1]['length']
    df_W_final_e2.loc[:, 'A2'] = df_W_left_e2[2]['length']
    df_W_final_e2.loc[:, 'A3'] = df_W_left_e2[3]['length']

    # 求324x4个距离
    A1_e1.append(df_W_p_e1[0]['length'].mean())
    A2_e1.append(df_W_p_e1[1]['length'].mean())
    A3_e1.append(df_W_p_e1[2]['length'].mean())
    A4_e1.append(df_W_p_e1[3]['length'].mean())
    A1_e2.append(df_W_p_e2[0]['length'].mean())
    A2_e2.append(df_W_p_e2[1]['length'].mean())
    A3_e2.append(df_W_p_e2[2]['length'].mean())
    A4_e2.append(df_W_p_e2[3]['length'].mean())

    """ 总结 """
    delete_R_list = np.setdiff1d(df_R_p[0].index.to_numpy(), left_array_R)
    for j in delete_R_list:
        print('--------------------- ', j+1, ' ---------------------')
        for t in range(4):
            print(df_R_p[t].loc[j, :])
    delete_W_list = np.setdiff1d(df_W_p[0].index.to_numpy(), left_array_W)
    for j in delete_W_list:
        print('--------------------- ', j+1, ' ---------------------')
        for t in range(4):
            print(df_W_p[t].loc[j, :])

    #print('第', i,'个位置:', '异常维度数:', err_num, '异常维度:', err_index_list, '异常维度点数:', err_pointnum_list, '异常小于3σ数:', in3sigema_array_W_numlist, '异常大于3σ数:', out3sigema_array_W_numlist, '2点同时异常数', double_point_num)
    print('第', i+1,'个位置:', '正常保留点数:', left_array_W.shape[0], '正常删除点数:',df_R.shape[0] / 4 - left_array_R.shape[0], '删除点:', delete_R_list + 1)
    print('第', i+1,'个位置:', '异常保留点数:',[left_array_W_e1.shape[0], left_array_W_e2.shape[0]], '异常删除点数:', df_W.shape[0]/4-left_array_W.shape[0], '删除点:', delete_W_list+1)
    print('异常2 A2 的平均值与标准差', df_W_p_e1_mean[2], df_W_p_e1_std[2])

    fig = plt.figure()
    if i==0:
        plt.xlim(-20, 420)
        plt.ylim(0, 10000)
        plt.xticks(np.around(np.linspace(0, 400, 6), decimals=0), fontproperties='Times New Roman', size=26)
        plt.yticks(np.around(np.linspace(0, 10000, 6), decimals=0), fontproperties='Times New Roman', size=26)
        plt.text(50, 8100, "第一类异常点", fontsize=26, color="y")
        plt.text(240, 8100, "第二类异常点", fontsize=26, color="b")
    if i == 99:
        plt.xlim(0, 300)
        plt.ylim(0, 10000)
        plt.xticks(np.around(np.linspace(0, 300, 6), decimals=0), fontproperties='Times New Roman', size=26)
        plt.yticks(np.around(np.linspace(0, 10000, 6), decimals=0), fontproperties='Times New Roman', size=26)
        plt.text(50, 8100, "第一类异常点", fontsize=26, color="y")
        plt.text(190, 8100, "第二类异常点", fontsize=26, color="b")
        plt.text(120, 6500, r"第二类异常点均值与3$σ$线", fontsize=20, color="#000079")
        plt.plot([np.min(left_array_W_e1), np.max(left_array_W_e1)], [df_W_p_e1_mean[2] - 3*df_W_p_e1_std[2], df_W_p_e1_mean[2] - 3*df_W_p_e1_std[2]], '*-.', color='#46A3FF', lw=1)
        plt.plot([np.min(left_array_W_e1), np.max(left_array_W_e1)], [df_W_p_e1_mean[2], df_W_p_e1_mean[2]], '*-.', color='#000079', lw=1)
        plt.plot([np.min(left_array_W_e1), np.max(left_array_W_e1)], [df_W_p_e1_mean[2] + 3*df_W_p_e1_std[2], df_W_p_e1_mean[2] + 3*df_W_p_e1_std[2]], '*-.', color='#46A3FF', lw=1)
    plt.xlabel(r'采集数据点编号', fontsize=30)
    plt.ylabel(r'测量距离 $\rm(mm)$', fontsize=30)

    plt.xlim(0, df_W.shape[0]/4)
    l1 = plt.scatter(np.arange(left_array_W.shape[0]), df_W_p[0].loc[left_array_W, 'length'], color='k', lw=0.1)
    l2 = plt.scatter(np.arange(left_array_W.shape[0]), df_W_p[1].loc[left_array_W, 'length'], color='r', lw=0.1)
    l3 = plt.scatter(np.arange(left_array_W.shape[0]), df_W_p[2].loc[left_array_W, 'length'], color='b', lw=0.1)
    l4 = plt.scatter(np.arange(left_array_W.shape[0]), df_W_p[3].loc[left_array_W, 'length'], color='g', lw=0.1)

    plt.plot([np.min(left_array_W_e1), np.max(left_array_W_e1)], [9000, 9000], 'sb-.', lw=1)
    plt.plot([np.min(left_array_W_e2), np.max(left_array_W_e2)], [9000, 9000], 'sy-.', lw=1)
    if i == 0 or 99:
        plt.legend(handles=[l1, l2, l3, l4], labels=[r'A0', r'A1', r'A2', r'A3'], loc='lower right', fontsize=26, ncol=1)

    #plt.plot(np.arange(df_R_p[0].shape[0]), df_R_p[0]['length'], color='k', lw=1)
    #plt.plot(np.arange(df_R_p[0].shape[0]), df_R_p[1]['length'], color='r', lw=1)
    #plt.plot(np.arange(df_R_p[0].shape[0]), df_R_p[2]['length'], color='b', lw=1)
    #plt.plot(np.arange(df_R_p[0].shape[0]), df_R_p[3]['length'], color='g', lw=1)
    
    plt.show()

    """ 输出"""
    #df_R_final.to_excel('./输出/第一问excel/' + str(i+1) + '.正常.xlsx', index = True)
    #df_W_final.to_excel('./输出/第一问excel/' + str(i+1) + '.异常.xlsx', index = True)
    #df_W_final_e1.to_excel('./输出/第一问excel-异常分两类/' + str(i + 1) + '.异常-1.xlsx', index=True)
    #df_W_final_e2.to_excel('./输出/第一问excel-异常分两类/' + str(i + 1) + '.异常-2.xlsx', index=True)

"""
print('x-异常数据324点；y-4个距离变化.png')
plt.figure()
plt.scatter(np.arange(np.array(A1_e1).shape[0]), A1_e1, color='k', lw=0.5)
plt.scatter(np.arange(np.array(A2_e1).shape[0]), A2_e1, color='r', lw=0.5)
plt.scatter(np.arange(np.array(A3_e1).shape[0]), A3_e1, color='b', lw=0.5)
plt.scatter(np.arange(np.array(A4_e1).shape[0]), A4_e1, color='g', lw=0.5)
plt.scatter(np.arange(np.array(A1_e2).shape[0]), A1_e2, color='k', lw=0.5)
plt.scatter(np.arange(np.array(A2_e2).shape[0]), A2_e2, color='r', lw=0.5)
plt.scatter(np.arange(np.array(A3_e2).shape[0]), A3_e2, color='b', lw=0.5)
plt.scatter(np.arange(np.array(A4_e2).shape[0]), A4_e2, color='g', lw=0.5)
plt.show()
"""




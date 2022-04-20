import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Lib.myFunctions as my  # 项目内置库

""" matplotlib 字体设置 """
config = {"font.family": 'serif', "font.size": 20, "mathtext.fontset": 'stix', "font.serif": ['SimSun'], }
plt.rcParams.update(config)
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 1.tag轨迹图——展示分布情况1
"""
df_tag = pd.read_excel('./Data/Tag坐标信息.xlsx')
A0_A4 = np.array([[0, 0, 1300], [5000, 0, 1700], [0, 5000, 1700], [5000, 5000, 1300]])
fig = plt.figure()
ax1 = plt.axes(projection='3d')
ax1.set_xlabel(r'$x$ $\rm(mm)$', fontsize=12)
ax1.set_ylabel(r'$y$ $\rm(mm)$', fontsize=12)
ax1.set_zlabel(r'$z$ $\rm(mm)$', fontsize=12)
plt.xlim(0, 5000)
plt.ylim(0, 5000)
ax1.set_zlim3d(zmin=800, zmax=2000)
ax1.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax1.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax1.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
plt.xticks(np.around(np.linspace(0, 5000, 6), decimals=0), fontproperties='Times New Roman', rotation=-0, size=12)
plt.yticks(np.around(np.linspace(0, 5000, 6), decimals=0), fontproperties='Times New Roman', rotation=0, size=12)
ax1.set_zticklabels(np.round(np.linspace(800, 2000, 7), decimals=2), fontproperties='Times New Roman',fontsize=12)

l1 = ax1.scatter(df_tag.loc[0,'x'], df_tag.loc[0,'y'], df_tag.loc[0,'z'], color='b', lw=2.5)
ax1.scatter(df_tag.loc[323,'x'], df_tag.loc[323,'y'], df_tag.loc[323,'z'], color='b', lw=2.5)
ax1.plot3D(df_tag['x'], df_tag['y'], df_tag['z'], 'ok-')
l2 = ax1.scatter(df_tag.loc[32,'x'], df_tag.loc[32,'y'], df_tag.loc[32,'z'], color='k', lw=2.5)
l3 = ax1.scatter(A0_A4[:, 0], A0_A4[:, 1], A0_A4[:, 2], color='r', lw=2.5)
plt.legend(handles=[l1, l2, l3], labels=[r'起点与终点', r'运动轨迹', r'锚点'],
           loc='upper right', fontsize=18, ncol=1)
plt.show()
"""

# 1.正常清理前——展示分布情况1
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
# 1.异常清理前——展示分布情况1
""" 1.异常清理前 """
"""
fig = plt.figure()
if i==0:
    plt.ylim(0, 400)
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

# 2.异常清理后——main1
"""
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

"""
# 1.正常、异常均值——展示分布情况2
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

# 1.田字格3-A2误差校正——main2
"""
err_fit1 = np.polyfit(mean_A1_list, err_A1_list, 1)
err_fit2 = np.polyfit(mean_A2_list, err_A2_list, 1)
err_fit3 = np.polyfit(mean_A3_list, err_A3_list, 1)
err_fit4 = np.polyfit(mean_A4_list, err_A4_list, 1)
err_fit_curve1 = np.polyval(err_fit1, np.linspace(np.min(mean_A1_list), np.max(mean_A1_list),10))
err_fit_curve2 = np.polyval(err_fit2, np.linspace(np.min(mean_A2_list), np.max(mean_A2_list),10))
err_fit_curve3 = np.polyval(err_fit3, np.linspace(np.min(mean_A3_list), np.max(mean_A3_list),10))
err_fit_curve4 = np.polyval(err_fit4, np.linspace(np.min(mean_A4_list), np.max(mean_A4_list),10))
print(err_fit1, err_fit2, err_fit3, err_fit4)

fig = plt.figure()
#plt.plot(np.arange(np.array(err_A1_list).shape[0]), err_A1_list, color='k', lw=0.5)
#plt.plot(np.arange(np.array(err_A1_list).shape[0]), err_A2_list, color='r', lw=0.5)
#plt.plot(np.arange(np.array(err_A1_list).shape[0]), err_A3_list, color='b', lw=0.5)
#plt.plot(np.arange(np.array(err_A1_list).shape[0]), err_A4_list, color='g', lw=0.5)

plt.xlabel(r'A0测量距离平均值 $\rm(mm)$', fontsize=30)
plt.ylabel(r'测量距离平均值与真实值误差 $\rm(mm)$', fontsize=30)
plt.xlim(0, 7000)
plt.ylim(-200, 200)
plt.xticks(np.around(np.linspace(0, 7000, 8), decimals=0), fontproperties='Times New Roman', size=26)
plt.yticks(np.around(np.linspace(-200, 200, 9), decimals=0), fontproperties='Times New Roman', size=26)

plt.scatter(mean_A1_list, err_A1_list, color='k', lw=0.5)
plt.plot(np.linspace(np.min(mean_A1_list), np.max(mean_A1_list),10), err_fit_curve1, 'k-', lw=1)
fig = plt.figure()
plt.xlabel(r'A1测量距离平均值 $\rm(mm)$', fontsize=30)
plt.ylabel(r'测量距离平均值与真实值误差 $\rm(mm)$', fontsize=30)
plt.xlim(0, 7000)
plt.ylim(-200, 150)
plt.xticks(np.around(np.linspace(0, 7000, 8), decimals=0), fontproperties='Times New Roman', size=26)
plt.yticks(np.around(np.linspace(-200, 150, 8), decimals=0), fontproperties='Times New Roman', size=26)
plt.scatter(mean_A2_list, err_A2_list, color='r', lw=0.5)
plt.plot(np.linspace(np.min(mean_A2_list), np.max(mean_A2_list),10), err_fit_curve2, 'r-', lw=1)
fig = plt.figure()
plt.xlabel(r'A2测量距离平均值 $\rm(mm)$', fontsize=30)
plt.ylabel(r'测量距离平均值与真实值误差 $\rm(mm)$', fontsize=30)
plt.xlim(0, 7000)
plt.ylim(-250, 100)
plt.scatter(mean_A3_list, err_A3_list, color='b', lw=0.5)
plt.plot(np.linspace(np.min(mean_A3_list), np.max(mean_A3_list),10), err_fit_curve3, 'b-', lw=1)
plt.xticks(np.around(np.linspace(0, 7000, 8), decimals=0), fontproperties='Times New Roman', size=26)
plt.yticks(np.around(np.linspace(-250, 100, 8), decimals=0), fontproperties='Times New Roman', size=26)
fig = plt.figure()
plt.xlabel(r'A3测量距离平均值 $\rm(mm)$', fontsize=30)
plt.ylabel(r'测量距离平均值与真实值误差 $\rm(mm)$', fontsize=30)
plt.xlim(0, 7000)
plt.ylim(-250, 150)
plt.xticks(np.around(np.linspace(0, 7000, 8), decimals=0), fontproperties='Times New Roman', size=26)
plt.yticks(np.around(np.linspace(-250, 150, 9), decimals=0), fontproperties='Times New Roman', size=26)
plt.scatter(mean_A4_list, err_A4_list, color='g', lw=0.5)
plt.plot(np.linspace(np.min(mean_A4_list), np.max(mean_A4_list),10), err_fit_curve4, 'g-', lw=1)
plt.show()
"""
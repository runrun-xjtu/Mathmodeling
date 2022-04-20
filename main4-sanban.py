import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import spatial, integrate, interpolate, optimize
import sys
import sympy
import math

sys.path.append("Lib/scikit")
from sko.GA import GA  # 项目内置库
from sko.SA import SA  # 项目内置库
from sko.PSO import PSO  # 项目内置库
import Lib.myFunctions as my  # 项目内置库

"""
场景一
    A0 = np.array([0, 0, 1300])
    A1 = np.array([5000, 0, 1700])
    A2 = np.array([0, 5000, 1700])
    A3 = np.array([5000, 5000, 1300])
场景二
A0 = np.array([0, 0, 1200])
A1 = np.array([5000, 0, 1600])
A2 = np.array([0, 3000, 1600])
A3 = np.array([5000, 3000, 1200])
"""

A0 = np.array([0, 0, 1300])
A1 = np.array([5000, 0, 1700])
A2 = np.array([0, 5000, 1700])
A3 = np.array([5000, 5000, 1300])

# 计算定位坐标
def triposition(A0, da, A1, db, A2, dc, A4, dd):
    xa, ya, za = A0[0], A0[1], A0[2]
    xb, yb, zb = A1[0], A1[1], A1[2]
    xc, yc, zc = A2[0], A2[1], A2[2]
    xd, yd, zd = A3[0], A3[1], A3[2]

    x, y, z = sympy.symbols("x, y, z")
    f1 = 2 * x * (xa - xd) + np.square(xd) - np.square(xa) + 2 * y * (ya - yd) + np.square(yd) - np.square(ya) + 2 * z * (za - zd) + np.square(zd) - np.square(za) - (np.square(dd) - np.square(da))
    f2 = 2 * x * (xb - xd) + np.square(xd) - np.square(xb) + 2 * y * (yb - yd) + np.square(yd) - np.square(yb) + 2 * z * (zb - zd) + np.square(zd) - np.square(zb) - (np.square(dd) - np.square(db))
    f3 = 2 * x * (xc - xd) + np.square(xd) - np.square(xc) + 2 * y * (yc - yd) + np.square(yd) - np.square(yc) + 2 * z * (zc - zd) + np.square(zd) - np.square(zc) - (np.square(dd) - np.square(dc))
    result = sympy.solve([f1, f2, f3], [x, y, z])

    locx, locy, locz = result[x], result[y], result[z]
    return [locx, locy, locz]

def disErr_scene1(tag_loc, distance4):
    tag_loc = np.array(tag_loc).astype(float)
    A0_ = np.array(A0)
    A1_ = np.array(A1)
    A2_ = np.array(A2)
    A3_ = np.array(A3)
    dis0 = np.linalg.norm(tag_loc - A0_)
    dis1 = np.linalg.norm(tag_loc - A1_)
    dis2 = np.linalg.norm(tag_loc - A2_)
    dis3 = np.linalg.norm(tag_loc - A3_)

    err = np.fabs(dis0 - distance4[0]) + np.fabs(dis1 - distance4[1]) + np.fabs(dis2 - distance4[2]) + np.fabs(dis3 - distance4[3])
    return err

def distance4(tag_loc):
    tag_loc = np.array(tag_loc)
    return [np.linalg.norm(tag_loc - A0), np.linalg.norm(tag_loc - A1), np.linalg.norm(tag_loc - A2), np.linalg.norm(tag_loc - A3)]

def distance3(tag_loc, i1, i2, i3):
    i1, i2, i3 = int(i1), int(i2), int(i3)
    tag_loc = np.array(tag_loc)
    A_all = np.array([A0, A1, A2, A3])
    return [np.linalg.norm(tag_loc - A_all[i1, :]), np.linalg.norm(tag_loc - A_all[i2, :]), np.linalg.norm(tag_loc - A_all[i3, :])]

df_tag = pd.read_excel('./Data/Tag坐标信息.xlsx')
df_tag_xyz = np.array([df_tag['x'], df_tag['y'], df_tag['z']])

""" 第二问 """
A_dis = [0, 0, 0, 0]
i1, i2, i3 = 0, 0, 0
real_dis_list, A1234_list = [], []
def optfunc4(x):
    dis4 = np.array(distance4(x))
    dis4 = np.fabs(dis4 - np.array([A_dis[0], A_dis[1], A_dis[2], A_dis[3]]))
    return dis4[0] + dis4[1] + dis4[2] + dis4[3]
def optfunc3(x):
    dis3 = np.array(distance3(x, i1, i2, i3))
    dis3 = np.fabs(dis3 - np.array([A_dis[i1], A_dis[i2], A_dis[i3]]))
    return dis3[0] + dis3[1] + dis3[2]

# 校正4个距离
err_A1_list, err_A2_list, err_A3_list, err_A4_list = [], [], [], []
mean_A1_list, mean_A2_list, mean_A3_list, mean_A4_list = [], [], [], []
""" 距离校正 """
"""
for i in range(324):
    filename_R = './输出/第一问excel/' + str(i + 1) + '.正常.xlsx'
    filename_W = './输出/第一问excel/' + str(i + 1) + '.异常.xlsx'
    df_R = pd.read_excel(filename_R)
    df_W = pd.read_excel(filename_W)
    #平均值
    df_R_mean, df_W_mean, df_R_std, df_W_std = [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]
    df_R_mean[0] = df_R.loc[:, 'A0'].mean()
    df_R_mean[1] = df_R.loc[:, 'A1'].mean()
    df_R_mean[2] = df_R.loc[:, 'A2'].mean()
    df_R_mean[3] = df_R.loc[:, 'A3'].mean()

    mean_A1_list.append(df_R_mean[0])
    mean_A2_list.append(df_R_mean[1])
    mean_A3_list.append(df_R_mean[2])
    mean_A4_list.append(df_R_mean[3])

    A1234 = np.array([df_R_mean[0], df_R_mean[1], df_R_mean[2], df_R_mean[3]])
    tag_xyz = [df_tag.loc[i,'x'], df_tag.loc[i,'y'], df_tag.loc[i,'z']]
    real_dis = np.array(distance4(tag_xyz))

    A1234_list.append(A1234)
    real_dis_list.append(real_dis)

    err_A1, err_A2, err_A3, err_A4 = A1234 - real_dis
    err_A1_list.append(err_A1)
    err_A2_list.append(err_A2)
    err_A3_list.append(err_A3)
    err_A4_list.append(err_A4)

    #print(err_A1, err_A2, err_A3, err_A4)

# 误差变化散点图、拟合

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
plt.scatter(mean_A1_list, err_A1_list, color='k', lw=0.5)
plt.plot(np.linspace(np.min(mean_A1_list), np.max(mean_A1_list),10), err_fit_curve1, 'k-', lw=0.5)
fig = plt.figure()
plt.scatter(mean_A2_list, err_A2_list, color='r', lw=0.5)
plt.plot(np.linspace(np.min(mean_A2_list), np.max(mean_A2_list),10), err_fit_curve2, 'k-', lw=0.5)
fig = plt.figure()
plt.scatter(mean_A3_list, err_A3_list, color='b', lw=0.5)
plt.plot(np.linspace(np.min(mean_A3_list), np.max(mean_A3_list),10), err_fit_curve3, 'k-', lw=0.5)
fig = plt.figure()
plt.scatter(mean_A4_list, err_A4_list, color='g', lw=0.5)
plt.plot(np.linspace(np.min(mean_A4_list), np.max(mean_A4_list),10), err_fit_curve4, 'k-', lw=0.5)
plt.show()
"""
# 真实/测量拟合图
"""
A1234_array = np.array(A1234_list)
real_dis_array = np.array(real_dis_list)
fig = plt.figure()
plt.scatter(A1234_array[:, 0], real_dis_array[:, 0], color='k', lw=0.2)
plt.scatter(A1234_array[:, 1], real_dis_array[:, 1], color='r', lw=0.2)
plt.scatter(A1234_array[:, 2], real_dis_array[:, 2], color='b', lw=0.2)
plt.scatter(A1234_array[:, 3], real_dis_array[:, 3], color='g', lw=0.2)
plt.show()

fit_list = []
for j in range(4):
    fit = np.polyfit(A1234_array[:, j], real_dis_array[:, j], 1)
    fit_list.append(fit)
print(fit_list)
"""

""" 遗传算法 """
def modify(dis4):
    # 使用误差-距离 一次函数拟合公式修正
    dis_mod1 = np.polyval(np.array([2.94138393e-02, -1.45008607e+02]), dis4[0])
    dis_mod2 = np.polyval(np.array([1.98224645e-02, -1.10754487e+02]), dis4[0])
    dis_mod3 = np.polyval(np.array([2.25552478e-02, -1.35738338e+02]), dis4[0])
    dis_mod4 = np.polyval(np.array([2.84936591e-02, -1.61115850e+02]), dis4[0])
    return [dis_mod1, dis_mod2, dis_mod3, dis_mod4]

fit_coefficient = np.ones((4, 2))
fit_coefficient = np.array([[0.97058616, 145.00860654], [0.98017754, 110.75448702], [0.97744475, 135.73833774], [0.97150634, 161.11584952]])
global_best_x_list_R, global_best_x_list_W_e1, global_best_x_list_W_e2, global_best_y_list_R, global_best_y_list_W_e1, global_best_y_list_W_e2 = [], [], [], [], [], []

#range(324)
for i in range(324):
    filename_R = './输出/第一问excel/' + str(i+1) + '.正常.xlsx'
    filename_W_e1 = './输出/第一问excel-异常分两类/' + str(i+1) + '.异常-1.xlsx'
    filename_W_e2 = './输出/第一问excel-异常分两类/' + str(i+1) + '.异常-2.xlsx'
    filename_T = './Data/附件2：测试集（实验场景1）.txt'
    filename_T3 = './Data/附件3：测试集（实验场景2）.txt'
    filename_T4 = './Data/附件4：测试集（实验场景1）.txt'

    df_R = pd.read_excel(filename_R)
    df_W_e1 = pd.read_excel(filename_W_e1)
    df_W_e2 = pd.read_excel(filename_W_e2)
    df_T = pd.read_table(filename_T, sep=':', names=['x1', 'time', 'x2', 'tagID', 'type', 'length', 'length2', 'Serial No.', 'No.'])
    df_T = df_T.iloc[:, [1, 3, 4, 5, 7, 8]].astype(float)
    df_T3 = pd.read_table(filename_T3, sep=':', names=['x1', 'time', 'x2', 'tagID', 'type', 'length', 'length2', 'Serial No.', 'No.'])
    df_T3 = df_T3.iloc[:, [1, 3, 4, 5, 7, 8]].astype(float)
    df_T4 = pd.read_table(filename_T4, sep=':', names=['x1', 'time', 'x2', 'tagID', 'type', 'length', 'length2', 'Serial No.', 'No.'])
    df_T4 = df_T4.iloc[:, [1, 3, 4, 5, 7, 8]].astype(float)


    # 按照类型分类测试集
    df_T_p = [0, 0, 0, 0]
    df_T3_p = [0, 0, 0, 0]
    df_T4_p = [0, 0, 0, 0]
    for t in range(4):
        df_T_p[t] = df_T.loc[df_T['type'] == t, :]
        df_T_p[t].index = np.arange(df_T_p[t].shape[0])
        df_T3_p[t] = df_T3.loc[df_T3['type'] == t, :]
        df_T3_p[t].index = np.arange(df_T3_p[t].shape[0])
        df_T4_p[t] = df_T4.loc[df_T4['type'] == t, :]
        df_T4_p[t].index = np.arange(df_T4_p[t].shape[0])

    #正常平均值
    df_R_mean, df_R_std = [0, 0, 0, 0], [0, 0, 0, 0]
    df_R_mean[0] = df_R.loc[:, 'A0'].mean()
    df_R_mean[1] = df_R.loc[:, 'A1'].mean()
    df_R_mean[2] = df_R.loc[:, 'A2'].mean()
    df_R_mean[3] = df_R.loc[:, 'A3'].mean()

    # 异常平均值, 标准差, 统计异常维度数目:使用方差排序取较大两个
    df_W_mean_e1, df_W_mean_e2, df_W_std_e1, df_W_std_e2 = [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]
    df_W_mean_e1[0] = df_W_e1.loc[:, 'A0'].mean()
    df_W_mean_e1[1] = df_W_e1.loc[:, 'A1'].mean()
    df_W_mean_e1[2] = df_W_e1.loc[:, 'A2'].mean()
    df_W_mean_e1[3] = df_W_e1.loc[:, 'A3'].mean()
    df_W_std_e1[0] = df_W_e1.loc[:, 'A0'].std()
    df_W_std_e1[1] = df_W_e1.loc[:, 'A1'].std()
    df_W_std_e1[2] = df_W_e1.loc[:, 'A2'].std()
    df_W_std_e1[3] = df_W_e1.loc[:, 'A3'].std()

    df_W_mean_e2[0] = df_W_e2.loc[:, 'A0'].mean()
    df_W_mean_e2[1] = df_W_e2.loc[:, 'A1'].mean()
    df_W_mean_e2[2] = df_W_e2.loc[:, 'A2'].mean()
    df_W_mean_e2[3] = df_W_e2.loc[:, 'A3'].mean()
    df_W_std_e2[0] = df_W_e2.loc[:, 'A0'].std()
    df_W_std_e2[1] = df_W_e2.loc[:, 'A1'].std()
    df_W_std_e2[2] = df_W_e2.loc[:, 'A2'].std()
    df_W_std_e2[3] = df_W_e2.loc[:, 'A3'].std()

    # GA 遗传算法, 3距离
    """
    selcet_list = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]
    for k in range(4):
        i1, i2, i3 = selcet_list[k][0], selcet_list[k][1], selcet_list[k][2]
        global_best_y = 1e8
        for j in range(10):
            pso = PSO(func=optfunc3, n_dim=3, pop=50, max_iter=800, lb=[0, 0, 0], ub=[5000, 5000, 3000], w=0.8, c1=0.5, c2=0.5)
            pso.run()
            best_x, best_y = pso.gbest_x, pso.gbest_y
            if best_y < global_best_y:
                global_best_x = best_x
                global_best_y = best_y
            if global_best_y < 100:
                print(i, k, 'GA: best_x:', global_best_x, 'best_y:', global_best_y)
                global_best_x_list.append(global_best_x)
                break
    """

    # GA 遗传算法, 测试集合（修改 上面4点坐标、范围、A_dis[j] = df_T3_p[j].loc[k, 'length']）

    for k in range(10):
        for j in range(4):
            A_dis[j] = df_T4_p[j].loc[k, 'length']
            #A_dis[j] = fit_coefficient[j, 0] * A_dis[j] + fit_coefficient[j, 1]
        A_dis = A_dis - np.array(modify(A_dis))

        res = triposition(A0, A_dis[0], A1, A_dis[1], A2, A_dis[2], A3, A_dis[3])
        err = disErr_scene1(res, A_dis)

        global_best_x_list_R.append(res)
        global_best_y_list_R.append(err)
        print(k, '测试集: best_x:', res, 'best_y:', err)

    # 利用三个距离求出的四个点 与 一点分布
    """
    global_best_x_array = np.array(global_best_x_list)
    fig = plt.figure()
    ax1 = plt.axes(projection='3d')
    ax1.scatter(global_best_x_array[:4, 0], global_best_x_array[:4, 1], global_best_x_array[:4, 2], color='k', lw=0.5)
    ax1.scatter(global_best_x_array[-1, 0], global_best_x_array[-1, 1], global_best_x_array[-1, 2], color='r', lw=0.5)
    plt.show()
    """
# 输出Excel

global_best_y_list_R = np.array(global_best_y_list_R).flatten()
global_best_y_list_W_e1 = np.array(global_best_y_list_W_e1).flatten()
global_best_y_list_W_e2 = np.array(global_best_y_list_W_e2).flatten()
df_out_data = {'R':global_best_y_list_R, 'W1':global_best_y_list_W_e1, 'W2':global_best_y_list_W_e2}
df_out = pd.DataFrame(df_out_data)
#df_out.to_excel('./输出/第二问遗传算法分类.xlsx')
df_out.to_excel('./输出/第二问遗传算法分类-新修正.xlsx')

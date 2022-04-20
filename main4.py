import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import spatial, integrate, interpolate, optimize
import sys
import sympy

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

A_dis = [0, 0, 0, 0]
i1, i2, i3 = 0, 0, 0
real_dis_list, A1234_list = [], []
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
    A0 = np.array([0, 0, 1300])
    A1 = np.array([5000, 0, 1700])
    A2 = np.array([0, 5000, 1700])
    A3 = np.array([5000, 5000, 1300])
    return [np.linalg.norm(tag_loc - A0), np.linalg.norm(tag_loc - A1), np.linalg.norm(tag_loc - A2), np.linalg.norm(tag_loc - A3)]

def distance3(tag_loc, i1, i2, i3):
    i1, i2, i3 = int(i1), int(i2), int(i3)
    tag_loc = np.array(tag_loc)
    A0 = np.array([0, 0, 1300])
    A1 = np.array([5000, 0, 1700])
    A2 = np.array([0, 5000, 1700])
    A3 = np.array([5000, 5000, 1300])
    A_all = np.array([A0, A1, A2, A3])
    return [np.linalg.norm(tag_loc - A_all[i1, :]), np.linalg.norm(tag_loc - A_all[i2, :]), np.linalg.norm(tag_loc - A_all[i3, :])]

df_tag = pd.read_excel('./Data/Tag坐标信息.xlsx')
df_tag_xyz = np.array([df_tag['x'], df_tag['y'], df_tag['z']])

""" 第二问 """
def optfunc4(x):
    dis4 = np.array(distance4(x))
    dis4 = np.fabs(dis4 - np.array([A_dis[0], A_dis[1], A_dis[2], A_dis[3]]))
    return dis4[0] + dis4[1] + dis4[2] + dis4[3]
def optfunc4_print(x):
    dis4 = np.array(distance4(x))
    dis4 = np.fabs(dis4 - np.array([A_dis[0], A_dis[1], A_dis[2], A_dis[3]]))
    dis4 = np.round(dis4, 2)
    print(dis4[0], dis4[1], dis4[2], dis4[3])
def optfunc3(x):
    dis3 = np.array(distance3(x, i1, i2, i3))
    dis3 = np.fabs(dis3 - np.array([A_dis[i1], A_dis[i2], A_dis[i3]]))
    return dis3[0] + dis3[1] + dis3[2]

# 校正4个距离
err_A1_list, err_A2_list, err_A3_list, err_A4_list = [], [], [], []
mean_A1_list, mean_A2_list, mean_A3_list, mean_A4_list = [], [], [], []

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
Rmean1, Rmean2, Rmean3, Rmean4, W1mean1, W1mean2, W1mean3, W1mean4, W2mean1, W2mean2, W2mean3, W2mean4 = [], [], [], [], [], [], [], [], [], [], [], []

"""  三个测试集验证  """
filename_T = './Data/附件2：测试集（实验场景1）.txt'
filename_T3 = './Data/附件3：测试集（实验场景2）.txt'
filename_T4 = './Data/附件4：测试集（实验场景1）.txt'

df_T = pd.read_table(filename_T, sep=':',
                     names=['x1', 'time', 'x2', 'tagID', 'type', 'length', 'length2', 'Serial No.', 'No.'])
df_T = df_T.iloc[:, [1, 3, 4, 5, 7, 8]].astype(float)
df_T3 = pd.read_table(filename_T3, sep=':',
                      names=['x1', 'time', 'x2', 'tagID', 'type', 'length', 'length2', 'Serial No.', 'No.'])
df_T3 = df_T3.iloc[:, [1, 3, 4, 5, 7, 8]].astype(float)
df_T4 = pd.read_table(filename_T4, sep=':',
                      names=['x1', 'time', 'x2', 'tagID', 'type', 'length', 'length2', 'Serial No.', 'No.'])
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

# 三边判据
"""
limit = 87.6

for i in range(10):
    for j in range(4):
        A_dis[j] = df_T_p[j].loc[i, 'length']
        #A_dis[j] = fit_coefficient[j, 0] * A_dis[j] + fit_coefficient[j, 1]
    A_dis = A_dis - np.array(modify(A_dis))

    # 2. 三边初始值计算
    res, err = 0, 0
    res = triposition(A0, A_dis[0], A1, A_dis[1], A2, A_dis[2], A3, A_dis[3])
    err = disErr_scene1(res, A_dis)
    if err > limit:
        print(i, '测试集 2 计算结果: best_x:', res, 'best_y:', err, '四个距离', np.round(A_dis, 2), '大于上限，干扰点')
    else:
        print(i, '测试集 2 计算结果: best_x:', res, 'best_y:', err, '四个距离', np.round(A_dis, 2), '小于上限，非干扰点')

for i in range(10):
    A0 = np.array([0, 0, 1200])
    A1 = np.array([5000, 0, 1600])
    A2 = np.array([0, 3000, 1600])
    A3 = np.array([5000, 3000, 1200])
    for j in range(4):
        A_dis[j] = df_T3_p[j].loc[i, 'length']
        #A_dis[j] = fit_coefficient[j, 0] * A_dis[j] + fit_coefficient[j, 1]
    A_dis = A_dis - np.array(modify(A_dis))

    # 2. 三边初始值计算
    res, err = 0, 0
    res = triposition(A0, A_dis[0], A1, A_dis[1], A2, A_dis[2], A3, A_dis[3])
    err = disErr_scene1(res, A_dis)
    if err > limit:
        print(i, '测试集 3 计算结果: best_x:', res, 'best_y:', err, '四个距离', np.round(A_dis, 2), '大于上限，干扰点')
    else:
        print(i, '测试集 3 计算结果: best_x:', res, 'best_y:', err, '四个距离', np.round(A_dis, 2), '小于上限，非干扰点')

for i in range(10):
    A0 = np.array([0, 0, 1300])
    A1 = np.array([5000, 0, 1700])
    A2 = np.array([0, 5000, 1700])
    A3 = np.array([5000, 5000, 1300])

    for j in range(4):
        A_dis[j] = df_T4_p[j].loc[i, 'length']
        #A_dis[j] = fit_coefficient[j, 0] * A_dis[j] + fit_coefficient[j, 1]
    A_dis = A_dis - np.array(modify(A_dis))

    # 2. 三边初始值计算
    res, err = 0, 0
    res = triposition(A0, A_dis[0], A1, A_dis[1], A2, A_dis[2], A3, A_dis[3])
    err = disErr_scene1(res, A_dis)
    if err > limit:
        print(i, '测试集 4 计算结果: best_x:', res, 'best_y:', err, '四个距离', np.round(A_dis, 2), '大于上限，干扰点')
    else:
        print(i, '测试集 4 计算结果: best_x:', res, 'best_y:', err, '四个距离', np.round(A_dis, 2), '小于上限，非干扰点')
"""

# 遗传判据
n = 2
limit = 87.6
for i in range(10):
    for j in range(4):
        A_dis[j] = df_T_p[j].loc[i, 'length']
        #A_dis[j] = fit_coefficient[j, 0] * A_dis[j] + fit_coefficient[j, 1]
    A_dis = A_dis - np.array(modify(A_dis))

    global_best_y = 1e8
    for j in range(n):
        ga = GA(func=optfunc4, n_dim=3, size_pop=100, max_iter=200, prob_mut=0.001, lb=[0, 0, 0], ub=[5000, 5000, 3000], precision=1e-7)
        best_x, best_y = ga.run()
        # pso = PSO(func=optfunc4, n_dim=3, pop=pop, max_iter=max_iter, lb=lb, ub=ub, w=w, c1=c1, c2=c2)
        # pso.run()
        # best_x, best_y = pso.gbest_x, pso.gbest_y

        if best_y < global_best_y:
            global_best_x = best_x
            global_best_y = best_y
        if global_best_y < 1:
            break

    if global_best_y > limit:
        print(i, '测试集 2 计算结果: best_x:', global_best_x, 'best_y:', global_best_y, '大于上限，干扰点')
    else:
        print(i, '测试集 2 计算结果: best_x:', global_best_x, 'best_y:', best_y, '小于上限，非干扰点')

for i in range(10):
    for j in range(4):
        A_dis[j] = df_T4_p[j].loc[i, 'length']
        #A_dis[j] = fit_coefficient[j, 0] * A_dis[j] + fit_coefficient[j, 1]
    A_dis = A_dis - np.array(modify(A_dis))

    global_best_y = 1e8
    for j in range(10):
        ga = GA(func=optfunc4, n_dim=3, size_pop=100, max_iter=200, prob_mut=0.001, lb=[0, 0, 0], ub=[5000, 5000, 3000], precision=1e-7)
        best_x, best_y = ga.run()
        # pso = PSO(func=optfunc4, n_dim=3, pop=pop, max_iter=max_iter, lb=lb, ub=ub, w=w, c1=c1, c2=c2)
        # pso.run()
        # best_x, best_y = pso.gbest_x, pso.gbest_y

        if best_y < global_best_y:
            global_best_x = best_x
            global_best_y = best_y
        if global_best_y < 1:
            break

    if global_best_y > limit:
        print(i, '测试集 4 计算结果: best_x:', global_best_x, 'best_y:', global_best_y, '大于上限，干扰点')
    else:
        print(i, '测试集 4 计算结果: best_x:', global_best_x, 'best_y:', global_best_y, '小于上限，非干扰点')




""" -------------------后面暂时无用 ----------------------"""





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

    Rmean1.append(df_R_mean[0]); Rmean2.append(df_R_mean[1]); Rmean3.append(df_R_mean[2]); Rmean4.append(df_R_mean[3]);
    W1mean1.append(df_W_mean_e1[0]); W1mean2.append(df_W_mean_e1[1]); W1mean3.append(df_W_mean_e1[2]); W1mean4.append(df_W_mean_e1[3]);
    W2mean1.append(df_W_mean_e2[0]); W2mean2.append(df_W_mean_e2[1]); W2mean3.append(df_W_mean_e2[2]); W2mean4.append(df_W_mean_e2[3]);

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
    exit_error = 1
    pop = 50; max_iter = 50; lb = [0, 0, 0]; w = 0.8; c1 = 0.5; c2 = 0.5

    ub = [5000, 5000, 3000];
    # GA 遗传算法, 测试集合（修改 上面4点坐标、范围、A_dis[j] = df_T3_p[j].loc[k, 'length']）
    """
    for k in range(10):
        for j in range(4):
            A_dis[j] = df_T4_p[j].loc[k, 'length']
            #A_dis[j] = fit_coefficient[j, 0] * A_dis[j] + fit_coefficient[j, 1]
        A_dis = A_dis - np.array(modify(A_dis))
        global_best_y = 1e8
        for j in range(10):
            
            #ga = GA(func=optfunc4, n_dim=3, size_pop=50, max_iter=100, prob_mut=0.001, lb=lb, ub=ub,precision=1e-7)
            #best_x, best_y = ga.run()
            
            pso = PSO(func=optfunc4, n_dim=3, pop=pop, max_iter=max_iter, lb=lb, ub=ub, w=w, c1=c1, c2=c2)
            pso.run()
            best_x, best_y = pso.gbest_x, pso.gbest_y

            if best_y < global_best_y:
                global_best_x = best_x
                global_best_y = best_y
            if global_best_y < exit_error:
                break
        global_best_x_list_R.append(global_best_x)
        global_best_y_list_R.append(global_best_y)

        #optfunc4_print(global_best_x)
        print(k, '测试集: best_x:', global_best_x, 'best_y:', global_best_y)
    """

    # GA 遗传算法, 4距离, 正常

    for j in range(4):
        A_dis[j] = df_R_mean[j]
        A_dis[j] = fit_coefficient[j, 0] * A_dis[j] + fit_coefficient[j, 1]
    global_best_y = 1e8
    for j in range(10):
        pso = PSO(func=optfunc4, n_dim=3, pop=pop, max_iter=max_iter, lb=lb, ub=ub, w=w, c1=c1, c2=c2)
        pso.run()
        best_x, best_y = pso.gbest_x, pso.gbest_y
        if best_y < global_best_y:
            global_best_x = best_x
            global_best_y = best_y
        if global_best_y < exit_error:
            break
    global_best_x_list_R.append(global_best_x)
    global_best_y_list_R.append(global_best_y)
    print(i, '正常: best_x:', global_best_x, 'best_y:', global_best_y)

    # GA 遗传算法, 4距离, 异常-1
    for j in range(4):
        A_dis[j] = df_W_mean_e1[j]
        A_dis[j] = fit_coefficient[j, 0] * A_dis[j] + fit_coefficient[j, 1]
    global_best_y = 1e8
    for j in range(10):
        pso = PSO(func=optfunc4, n_dim=3, pop=pop, max_iter=max_iter, lb=lb, ub=ub, w=w, c1=c1, c2=c2)
        pso.run()
        best_x, best_y = pso.gbest_x, pso.gbest_y
        if best_y < global_best_y:
            global_best_x = best_x
            global_best_y = best_y
        if global_best_y < exit_error:
            break
    global_best_x_list_W_e1.append(global_best_x)
    global_best_y_list_W_e1.append(global_best_y)
    print(i, '异常-1: best_x:', global_best_x, 'best_y:', global_best_y)

    # GA 遗传算法, 4距离, 异常-2
    for j in range(4):
        A_dis[j] = df_W_mean_e2[j]
        A_dis[j] = fit_coefficient[j, 0] * A_dis[j] + fit_coefficient[j, 1]
    global_best_y = 1e8
    for j in range(10):
        pso = PSO(func=optfunc4, n_dim=3, pop=pop, max_iter=max_iter, lb=lb, ub=ub, w=w, c1=c1, c2=c2)
        pso.run()
        best_x, best_y = pso.gbest_x, pso.gbest_y
        if best_y < global_best_y:
            global_best_x = best_x
            global_best_y = best_y
        if global_best_y < exit_error:
            break
    global_best_x_list_W_e2.append(global_best_x)
    global_best_y_list_W_e2.append(global_best_y)
    print(i, '异常-2: best_x:', global_best_x, 'best_y:', global_best_y)

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
"""
global_best_y_list_R = np.array(global_best_y_list_R).flatten()
global_best_y_list_W_e1 = np.array(global_best_y_list_W_e1).flatten()
global_best_y_list_W_e2 = np.array(global_best_y_list_W_e2).flatten()
df_out_data = {'R':global_best_y_list_R, 'W1':global_best_y_list_W_e1, 'W2':global_best_y_list_W_e2}
df_out = pd.DataFrame(df_out_data)
df_out.to_excel('./输出/第二问遗传算法分类.xlsx')
"""
# 输出均值Excel
"""
df_outeven = {'R1':Rmean1, 'R2':Rmean2, 'R3':Rmean3, 'R4':Rmean4, 'W11':W1mean1, 'W12':W1mean2, 'W13':W1mean3, 'W14':W1mean4, 'W21':W2mean1, 'W22':W2mean2, 'W23':W2mean3, 'W24':W2mean4, 'x':df_tag['x'], 'y':df_tag['y'], 'z':df_tag['z']}
df_outeven = pd.DataFrame(df_outeven)
df_outeven.to_excel('./输出均值Excel.xlsx')
"""
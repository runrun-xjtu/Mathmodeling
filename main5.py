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

A_dis = [0, 0, 0, 0]
i1, i2, i3 = 0, 0, 0
real_dis_list, A1234_list = [], []
minus_object = 0

def distance4(tag_loc):
    tag_loc = np.array(tag_loc)
    A0 = np.array([0, 0, 1300])
    A1 = np.array([5000, 0, 1700])
    A2 = np.array([0, 5000, 1700])
    A3 = np.array([5000, 5000, 1300])
    return [np.linalg.norm(tag_loc - A0), np.linalg.norm(tag_loc - A1), np.linalg.norm(tag_loc - A2), np.linalg.norm(tag_loc - A3)]

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

def sanbian_gene_opFunc(x):
    A_m = A_dis.copy()
    A_m[minus_object] = A_m[minus_object] - x
    res1 = triposition(A0, A_m[0], A1, A_m[1], A2, A_m[2], A3, A_m[3])
    err1 = disErr_scene1(res1, A_m)
    return err1

df_tag = pd.read_excel('./Data/Tag坐标信息.xlsx')
df_tag_xyz = np.array([df_tag['x'], df_tag['y'], df_tag['z']])

""" 第二问 """
def optfunc4(x):
    dis4 = np.array(distance4(x))
    dis4 = np.fabs(dis4 - np.array([A_dis[0], A_dis[1], A_dis[2], A_dis[3]]))
    return dis4[0] + dis4[1] + dis4[2] + dis4[3]

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
global_best_x_list, global_best_y_list = [], []
location_list = []
""" 三边算法测试集 """
filename_T5 = './Data/附件5：动态轨迹数据.txt'
df_T5 = pd.read_table(filename_T5, sep=':', names=['x1', 'time', 'x2', 'tagID', 'type', 'length', 'length2', 'Serial No.', 'No.'])
df_T5 = df_T5.iloc[:, [1, 3, 4, 5, 7, 8]].astype(float)

# 按照类型分类测试集
df_T5_p = [0, 0, 0, 0]
for t in range(4):
    df_T5_p[t] = df_T5.loc[df_T5['type'] == t, :]
    df_T5_p[t].index = np.arange(df_T5_p[t].shape[0])

# GA 遗传算法, 测试集合（修改 上面4点坐标、范围、A_dis[j] = df_T3_p[j].loc[k, 'length']）
limit, exit_error = 100, 1

for k in range(df_T5_p[0].shape[0]):
    # 1. 先修正
    for j in range(4):
        A_dis[j] = df_T5_p[j].loc[k, 'length']
        # A_dis[j] = fit_coefficient[j, 0] * A_dis[j] + fit_coefficient[j, 1]
    A_dis = A_dis - np.array(modify(A_dis))

    # 2. 三边初始值计算
    res, err = 0, 0
    res = triposition(A0, A_dis[0], A1, A_dis[1], A2, A_dis[2], A3, A_dis[3])
    err = disErr_scene1(res, A_dis)
    if err > limit:
        print(k, '测试集初始计算结果结果: best_x:', res, 'best_y:', err, '四个距离', np.round(A_dis, 2), '异常点')
    else:
        print(k, '测试集初始计算结果结果: best_x:', res, 'best_y:', err, '四个距离', np.round(A_dis, 2), '正常点')

    #正常数据遗传算法
    if err <= limit:
        pop = 50; max_iter = 200; lb = [0, 0, 0]; w = 0.8; c1 = 0.5; c2 = 0.5;ub = [5000, 5000, 3000]
        global_best_y = 1e8
        for j in range(10):

            ga = GA(func=optfunc4, n_dim=3, size_pop=100, max_iter=200, prob_mut=0.001, lb=[0, 0, 0],
                    ub=[5000, 3000, 3000], precision=1e-7)
            best_x, best_y = ga.run()

            """
            pso = PSO(func=optfunc4, n_dim=3, pop=pop, max_iter=max_iter, lb=lb, ub=ub, w=w, c1=c1, c2=c2)
            pso.run()
            best_x, best_y = pso.gbest_x, pso.gbest_y
            """

            if best_y < global_best_y:
                global_best_x = best_x
                global_best_y = best_y
            if global_best_y < exit_error:
                break
        #global_best_x_list.append(global_best_x)
        global_best_y_list.append(global_best_y)
        location_list.append(global_best_x)
        print(k, '轨迹点正常结果: best_x:', global_best_x, 'best_y:', global_best_y, '四个距离', np.round(A_dis, 2))

    # 3. 如果异常则遗传优化
    final_choose = -1
    global_best_x, global_best_y = 0, err
    if err > limit:
        for j in range(4):
            minus_object = j # 修改优化对象

            """
            ga = GA(func=sanbian_gene_opFunc, n_dim=1, size_pop=20, max_iter=10, prob_mut=0.001, lb=0, ub=600,
                    precision=1e-7)
            best_x, best_y = ga.run()
            """
            for minus_list in np.arange(10, 601, 10):
                minus_value = minus_list
                err_gen = sanbian_gene_opFunc(minus_value)
                best_x, best_y = minus_value, err_gen

                if best_y < global_best_y:
                    final_choose = j
                    global_best_x = best_x
                    global_best_y = best_y
                if global_best_y < exit_error:
                    break

        global_best_x_list.append(global_best_x)
        global_best_y_list.append(global_best_y)

    # 2. 三边重新计算
    if err > limit:
        if final_choose != -1:
            res, err = 0, 0
            A_dis[final_choose] = A_dis[final_choose] - global_best_x
            res = triposition(A0, A_dis[0], A1, A_dis[1], A2, A_dis[2], A3, A_dis[3])
            err = disErr_scene1(res, A_dis)
            location_list.append(res)
            print(k, '轨迹点修正结果: 修正后坐标:', res, 'best_y:', global_best_y, '干扰点', final_choose, 'best_x 修正值', global_best_x, '四个距离', np.round(A_dis, 2))
        else:
            print(k, '轨迹点修正结果:失败')

# 输出Excel
location_list = np.array(location_list)
global_best_y_list = np.array(global_best_y_list)

df_out_data = {'x':location_list[:, 0], 'y':location_list[:, 1], 'z':location_list[:, 2], 'err':global_best_y_list}
df_out = pd.DataFrame(df_out_data)
df_out.to_excel('./输出/第五问轨迹点坐标——最终遗传算法.xlsx')



import os
import warnings
import random

import numpy as np
import pandas as pd
import xgboost as xgb
import pickle

from sklearn.cluster import KMeans
from smt.surrogate_models import KRG, RBF, QP
from smt.sampling_methods import LHS
import matplotlib.pyplot as plt

# import其他py文件
from pyswarms.single import GlobalBestPSO

import model_v3_3 as mt

# Seed settings
seed = int(np.random.rand(1) * 5e3)
print(seed)

# Meta model settings
parameters = {'seed': 100, 'nthread': -1, 'gamma': 0, 'lambda': 6,
              'max_depth': 3, 'eta': 0.15, 'objective': 'reg:squarederror'}
num_boost_round = 250

# Base model settings
enabled_model = np.array([
    'GP',
    'RBF',
    'Polynomial',
])

base_model_weight = np.ones(len(enabled_model)).ravel() / len(enabled_model)

# Test problem settings
problem_param = {

    'dimension': 5,
    'range': [[0.55, 0.2, 0.2, 0.02, 8], [0.95, 0.3, 0.32, 0.04, 12]],
    'column_name': ['x1', 'x2', 'x3', 'x4', 'x5', 'y']

}

# EA Algorithm settings
EA_param = {
    'n_particles': 300,
    'iters': 100,
    'local_search': True,
    # 'local_search_num': int(problem_param['dimension'] * 1),
    'local_search_num': int(problem_param['dimension'] * 1),
    'local_shrink_scale': 1,
    'global_clusters': 6,
    'explore': False,
    'explore_prob_offset': 0.7,
}

options = {
    'c1': 0.2,
    'c2': 0.2,
    'w': 0.8
}

# 计算资源相关参数
# computations = int(problem_param['dimension']) * (5 + 10)
computations = 100
sample_init = 20
generations = computations - sample_init
current_generation = 0


def warpSimulation(design_para):
    # 输入要求为一个二维向量，本方法会自动展为一维向量
    design_para = design_para.ravel()
    design_csv = np.savetxt('design_para.csv', design_para)
    cmd_cae = os.system('abaqus cae noGUI=D:\MLproject\BSSO_Chippackaging\\5_variables_result\chip_simulation.py')
    with open('D:\MLproject\BSSO_Chippackaging\\5_variables_result\\result_miao\chip_warpage.csv') as f:
        warpageData = np.loadtxt(f)
    warpage = warpageData[0]
    return warpage


# Latin Hypercube Sampling
def latin_hypercube_sampling(num_samples):
    x1_l = problem_param['range'][0][0]
    x1_u = problem_param['range'][1][0]
    x2_l = problem_param['range'][0][1]
    x2_u = problem_param['range'][1][1]
    x3_l = problem_param['range'][0][2]
    x3_u = problem_param['range'][1][2]
    x4_l = problem_param['range'][0][3]
    x4_u = problem_param['range'][1][3]
    x5_l = problem_param['range'][0][4]
    x5_u = problem_param['range'][1][4]
    x_lim = np.array(([x1_l, x1_u], [x2_l, x2_u], [x3_l, x3_u], [x4_l, x4_u], [x5_l, x5_u]))
    sampling = LHS(xlimits=x_lim, criterion='cm')
    x = sampling(num_samples)
    return x

def globalCluster(pso_global_pos, pso_global_cost):
    n_clusters = EA_param['global_clusters']
    select_pos = []
    select_cost = []
    input = np.hstack((pso_global_pos, pso_global_cost.reshape(-1, 1)))
    kms = KMeans(init='k-means++', n_clusters=n_clusters, random_state=1, tol=1e-4)
    pred = kms.fit_predict(input)
    for i in range(n_clusters):
        cluster = input[pred[0:] == i, :]
        cluster = cluster[np.argsort(cluster[:, -1]), :]
        select_pos.append(cluster[0, :-1])
        select_cost.append(cluster[0, -1])
    choice = int(np.random.randint(0, 1, 1))
    pso_cluster_pos = np.array(select_pos[choice])
    pso_cluster_cost = np.array(select_cost[choice])
    return pso_cluster_pos.reshape(1, -1), pso_cluster_cost

def explore_prob():
    if EA_param['explore']:
        gen = int(current_generation / generations * 100)
        rand = np.random.rand()
        order = 0.15
        scale = 0.2
        x_offset = 0.5
        y_offset = EA_param['explore_prob_offset']
        y = 1 - 1.0 / (1 + np.exp(-(gen - 100 * x_offset)) ** order)
        y = y * scale + y_offset
        return rand > y
    else:
        return False

def explore(sample_point, model_num, weight, base_model_weight):
    dim = problem_param['dimension']
    xl = problem_param['range'][0]
    xu = problem_param['range'][1]
    repeat = 1e4
    explore_dim = np.random.randint(0, dim, 1)[0]
    sequence = np.arange(xl, xu, 1 / repeat)
    explore_mat = sample_point.reshape(-1, 1).T.repeat((xu - xl) * repeat, axis=0)
    explore_mat[:, explore_dim] = sequence
    explore_eval = popEvaluate(explore_mat, model_num, weight, base_model_weight)
    explore_optimal_point = explore_mat[np.argmin(explore_eval), :]
    explore_optimal_cost = explore_eval.min()
    return explore_optimal_point.reshape(1, -1), explore_optimal_cost


def popEvaluate(population, model_num, weight, base_model_weight):
    """
    对种群内所有个体的适应度进行评估
    :param population: 需要进行适应度评估的种群
    :return: /
    """
    model_num = model_num
    base_model_weight = base_model_weight
    if model_num == 4:
        pop_array = np.vstack((population[0:]))
        # dtest = xgb.DMatrix(pop_array)
        Pred_gp = np.empty((len(population), 1))
        Pred_poly = np.empty((len(population), 1))
        Pred_rbf = np.empty((len(population), 1))

        for i in range(5):
            # gp/krg base model
            model_gp = mt.base_model[model_num][0]
            gp_pred = model_gp.predict_values(pop_array)
            pop_pred_gp = gp_pred * weight[0][i]
            pop_pred_gp = pop_pred_gp.reshape(-1, 1)

            # poly base model
            model_poly = mt.base_model[model_num][1]
            poly_pred = model_poly.predict_values(pop_array)
            pop_pred_poly = poly_pred * weight[1][i]
            pop_pred_poly = pop_pred_poly.reshape(-1, 1)

            # rbf base model
            model_rbf = mt.base_model[model_num][2]
            rbf_pred = model_rbf.predict_values(pop_array)
            pop_pred_rbf = rbf_pred * weight[2][i]
            pop_pred_rbf = pop_pred_rbf.reshape(-1, 1)

            Pred_gp = np.hstack((Pred_gp, pop_pred_gp))
            Pred_poly = np.hstack((Pred_poly, pop_pred_poly))
            Pred_rbf = np.hstack((Pred_rbf, pop_pred_rbf.reshape(-1, 1)))

        Pred_gp = Pred_gp[:, 1:]
        Pred_poly = Pred_poly[:, 1:]
        Pred_rbf = Pred_rbf[:, 1:]

        Pred_gp = np.sum(Pred_gp, axis=1).reshape(-1, 1)
        Pred_poly = np.sum(Pred_poly, axis=1).reshape(-1, 1)
        Pred_rbf = np.sum(Pred_rbf, axis=1).reshape(-1, 1)

        # Meta model here
        meta_input = np.hstack((Pred_gp, Pred_poly, Pred_rbf))
        meta = xgb.Booster()
        meta.load_model('meta_xgb.model')
        dmeta = xgb.DMatrix(meta_input)
        meta_pred = meta.predict(dmeta)
        meta_pred = meta_pred.ravel()

        return meta_pred
    else:
        pop_array = np.vstack((population[0:]))

        # gp/krg base model
        model_gp = mt.base_model[model_num][0]
        gp_pred = model_gp.predict_values(pop_array)
        pop_pred_gp = gp_pred.reshape(-1, 1)

        # poly base model
        model_poly = mt.base_model[model_num][1]
        poly_pred = model_poly.predict_values(pop_array)
        pop_pred_poly = poly_pred.reshape(-1, 1)

        # rbf base model
        model_rbf = mt.base_model[model_num][2]
        rbf_pred = model_rbf.predict_values(pop_array)
        pop_pred_rbf = rbf_pred.reshape(-1, 1)

        # 各base model赋权重
        pop_pred = base_model_weight[0] * pop_pred_gp + base_model_weight[1] * pop_pred_poly + base_model_weight[
            2] * pop_pred_rbf
        pop_pred = pop_pred.ravel()

        return pop_pred


def updateRegion(Sample_X, Sample_Warp, bounds):
    k_fold = 5
    num_local_search = EA_param['local_search_num']
    if current_generation % k_fold == (k_fold - 1):
        exact_sample = Sample_X
        exact_value = Sample_Warp
        exact_sample = exact_sample[np.argsort(exact_value.ravel()), :]
        dist = exact_sample[:, 0:-1] - exact_sample[:, 0].reshape(-1, 1)
        dist = np.linalg.norm(dist, axis=1)
        dist_index = np.argsort(dist)
        dist_index = dist_index[dist_index < num_local_search]
        near_optimal = exact_sample[dist_index, :]
        range_max = np.max(near_optimal, axis=0)
        range_min = np.min(near_optimal, axis=0)
        range_mean = np.mean(near_optimal, axis=0)
        range_lb = (range_min * int(100 * EA_param['local_shrink_scale']) + range_mean * int(
            100 * (1 - EA_param['local_shrink_scale']))) / 100
        range_ub = (range_max * int(100 * EA_param['local_shrink_scale']) + range_mean * int(
            100 * (1 - EA_param['local_shrink_scale']))) / 100
        updated_bounds = (range_lb, range_ub)
        return updated_bounds
    else:
        return bounds


def PSO(Sample_X, Sample_Warp, model_num, weight, base_model_weight):
    dim = problem_param['dimension']
    x_min = problem_param['range'][0] * np.ones(dim)
    x_max = problem_param['range'][1] * np.ones(dim)
    bounds = (x_min, x_max)
    if EA_param['local_search']:
        bounds = updateRegion(Sample_X, Sample_Warp, bounds)
    # PSO参数设置
    optimizer = GlobalBestPSO(
        n_particles=EA_param['n_particles'],
        dimensions=dim,
        options=options,
        bounds=bounds)
    # PSO迭代
    cost, pos = optimizer.optimize(
        popEvaluate,
        iters=EA_param['iters'],
        model_num=model_num,
        weight=weight,
        base_model_weight=base_model_weight)
    cost = np.array(cost).reshape(1, -1)
    pos = pos.reshape(1, -1)

    # 局部搜索：选择最优点
    prob = explore_prob()
    if prob or (not EA_param['explore']):
        if current_generation % 5 == (5 - 1):
            pso_pos = pos
            pso_cost = cost
        else:
            pso_global_pos = optimizer.swarm.position
            pso_global_cost = optimizer.swarm.current_cost
            pso_pos, pso_cost = globalCluster(pso_global_pos, pso_global_cost)
    else:
        pso_pos, pso_cost = explore(pos, model_num, weight, base_model_weight)

    return pso_pos, pso_cost.reshape(-1, 1)


def SAiterate(Sample_Init, Sample_Points, num_boost_round):
    global current_generation
    mt.paraInit()

    Sample_X = Sample_Init
    Sample_Warp_Init = []
    for sample in Sample_X:
        warpage = warpSimulation(sample)
        Sample_Warp_Init.append(warpage)
    Sample_Warp_Init = np.array(Sample_Warp_Init).reshape(-1, 1)

    Sample_Warp = Sample_Warp_Init
    Sample_Warp_Select = []
    for generation in range(0, generations):
        current_generation = generation
        # 采样点训练代理模型
        base_model, meta_model, model_num, weight, base_model_weight = mt.modelTrain(Sample_X, Sample_Warp, parameters,
                                                                                     num_boost_round, generation)
        # PSO
        sample, pred = PSO(Sample_X, Sample_Warp, model_num, weight, base_model_weight)

        # 仿真评估所选点
        # 本方法输入值要求为一维数组
        warpage = warpSimulation(sample)
        Sample_X = np.append(Sample_X, sample, axis=0)
        Sample_Warp = np.append(Sample_Warp, [[warpage]], axis=0)
        Sample_Warp_Select.append(warpage)

    Sample_Warp_Select = np.array(Sample_Warp_Select).reshape(-1, 1)
    mt.paraInit()
    resultDisp(Sample_Warp_Select, Sample_Warp_Init)

    Sample_Array = np.hstack((Sample_X, Sample_Warp))
    Sample_Array = pd.DataFrame(Sample_Array, columns=problem_param['column_name'])
    Sample_Array.sort_values(by='y', axis=0, ascending=True)
    Sample_Optimum = np.array(Sample_Array.iloc[0, :])

    return Sample_Array, Sample_Optimum


def resultDisp(Sample_Warp_Select, Sample_Warp_Init):

    Min = []  # 每一代时所有采样点的最小值
    min = np.min(Sample_Warp_Init)  # 截止该次代时的最优解

    for i in range(0, len(Sample_Warp_Select)):
        if Sample_Warp_Select[i] < min:
            Min.append(float(Sample_Warp_Select[i]))
            min = float(Sample_Warp_Select[i])
        else:
            Min.append(min)

    meta_values = []
    for index in range(len(Sample_Warp_Select)):
        if index % 5 == 4:
            meta_values.append(Sample_Warp_Select[index])
    meta_values = np.array(meta_values).reshape(-1, 1)

    global computations
    global sample_init
    g = range(computations - sample_init)

    meta_index = np.arange(4, computations - sample_init, 5).reshape(-1, 1)

    plt.figure(figsize=(20, 10))
    plt.xlabel('Generations')
    plt.ylabel('Test Values')
    plt.legend("Select Points", loc='lower right')
    plt.title('Generations vs TestValues')
    plt.plot(g, Sample_Warp_Select, 'r-', lw=1)
    plt.scatter(g, Sample_Warp_Select, alpha=1)
    plt.scatter(meta_index, meta_values, alpha=1, s=80, c='r')

    plt.legend("Convergence Curve", loc='lower right')
    plt.plot(g, Min, 'b-', lw=2)
    plt.show()



def deleteModels():
    path = './'
    for foldName, subfolders, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('.model'):
                os.remove(os.path.join(path, filename))


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    # deleteModels()
    # 初始化采样点并训练代理模型
    Sample_Init = latin_hypercube_sampling(sample_init)  # 初始化的训练采样点
    Optimum = []
    Resdisp = []

    for i in range(1):
        Sample_Points = np.empty((0, problem_param['dimension']))
        Sample_Array, Sample_Optimum = SAiterate(Sample_Init, Sample_Points, num_boost_round)
        Optimum.append(Sample_Optimum)
        Resdisp.append(Sample_Optimum[-1])
        print("Minimum Test Value = ", Sample_Optimum[-1])
        print(i)

    print("Mean value:", np.mean(Resdisp))
    print("Variance value:" + str(np.sqrt(np.var(Resdisp))) + "^2")

    optimum_array = pd.DataFrame(data=Optimum, columns=problem_param['column_name'])  # 数据有三列，列名分别为one,two,three
    csvname = './' + str(seed) + '_' + str(problem_param['name']) + '_' + str(problem_param['dimension']) + 'd_' \
              + str(options['c1']) + '_' + str(options['c2']) + '_' + str(options['w']) + '_' \
              + str(enabled_model[0]) + '_' + str(enabled_model[1]) + '_' + str(enabled_model[2]) \
              + '.csv'
    optimum_array.to_csv(csvname, encoding='gbk')
    # deleteModels()
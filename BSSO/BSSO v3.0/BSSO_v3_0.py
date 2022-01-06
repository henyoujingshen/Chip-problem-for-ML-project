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

import model_v3_0 as mt

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
    # 'name': 'rosenbrock',
    # 'dimension': 6,
    # 'range': [-5, 10],

    # 'name': 'rastrigin',
    # 'dimension': 20,
    # 'range': [-5, 5],

    # 'name': 'griewank',
    # 'dimension': 20,
    # 'range': [-600, 600],

    # 'name': 'ellipsoid',
    # 'dimension': 20,
    # 'range': [-5.12, 5.12],

    # 'name': 'ackley',
    # 'dimension': 10,
    # 'range': [-32.768, 32.768],

    # 'name': 'shcb',
    # 'dimension': 2,
    # 'range': [[-3, -2], [3, 2]],

    # 'name': 'goldstein_price',
    # 'dimension': 2,
    # 'range': [-2, 2],

    # 'name': 'hartman3',
    # 'dimension': 3,
    # 'range': [0, 1],

    # 'name': 'alpine',
    # 'dimension': 2,
    # 'range': [-10, 10],

    # 'name': 'hartman6',
    # 'dimension': 6,
    # 'range': [0, 1],

    # 'name': 'easom',
    # 'dimension': 2,
    # 'range': [-100, 100],

    'name': 'shekel',
    'dimension': 4,
    'range': [0, 10],

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
computations = int(problem_param['dimension']) * (5 + 11)
sample_init = int(problem_param['dimension']) * 5
generations = computations - sample_init
current_generation = 0


def testFunc(sample_array):
    """
    在csv数据集中找到采样点对应的翘曲率(CAE过程模拟)，避免使用for循环，加速计算
    :param population:传参为所需要获取翘曲率的采样点array
    :return: 分别为特征X和翘曲率y，array
    """
    X = sample_array
    if problem_param['name'] == 'rosenbrock':
        result = np.sum(100 * np.square(X[:, 1:] - np.square(X[:, :-1])) + np.square(X[:, -1] - 1).reshape(-1, 1),
                        axis=1)
        if problem_param['dimension'] == 10:
            problem_param['column_name'] = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'y']
        elif problem_param['dimension'] == 20:
            problem_param['column_name'] = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10',
                                            'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'y']
        else:
            problem_param['column_name'] = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10',
                                            'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20',
                                            'x21', 'x22', 'x23', 'x24', 'x25', 'x26', 'x27', 'x28', 'x29', 'x30', 'y']
    elif problem_param['name'] == 'rastrigin':
        result = 10 * problem_param['dimension'] + np.sum(np.square(X) - 10 * np.cos(2 * np.pi * X), axis=1)
        if problem_param['dimension'] == 10:
            problem_param['column_name'] = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'y']
        elif problem_param['dimension'] == 20:
            problem_param['column_name'] = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10',
                                            'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'y']
        else:
            problem_param['column_name'] = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10',
                                            'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20',
                                            'x21', 'x22', 'x23', 'x24', 'x25', 'x26', 'x27', 'x28', 'x29', 'x30', 'y']
    elif problem_param['name'] == 'griewank':
        den = 1 / np.sqrt(np.arange(1, problem_param['dimension'] + 1))
        result = np.sum(np.square(X), axis=1) / 4e3 - np.prod(np.cos(np.multiply(X, den)), axis=1) + 1
        if problem_param['dimension'] == 10:
            problem_param['column_name'] = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'y']
        elif problem_param['dimension'] == 20:
            problem_param['column_name'] = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10',
                                            'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'y']
        else:
            problem_param['column_name'] = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10',
                                            'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20',
                                            'x21', 'x22', 'x23', 'x24', 'x25', 'x26', 'x27', 'x28', 'x29', 'x30', 'y']
    elif problem_param['name'] == 'ellipsoid':
        i = np.arange(1, problem_param['dimension'] + 1)
        result = np.sum(np.square(X) * i, axis=1)
        if problem_param['dimension'] == 10:
            problem_param['column_name'] = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'y']
        elif problem_param['dimension'] == 20:
            problem_param['column_name'] = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10',
                                            'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'y']
        else:
            problem_param['column_name'] = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10',
                                            'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20',
                                            'x21', 'x22', 'x23', 'x24', 'x25', 'x26', 'x27', 'x28', 'x29', 'x30', 'y']
    elif problem_param['name'] == 'goldstein_price':
        x1, x2 = X[:, 0], X[:, 1]
        result_a = 1 + (np.square(x1 + x2 + 1)) * \
                   (19 - 14 * x1 + 3 * np.square(x1)
                    - 14 * x2 + 6 * x1 * x2 + 3 * np.square(x2))
        result_b = 30 + (np.square(2 * x1 - 3 * x2)) * \
                   (18 - 32 * x1 + 12 * np.square(x1)
                    + 48 * x2 - 36 * x1 * x2 + 27 * np.square(x2))
        result = result_a * result_b
        problem_param['column_name'] = ['x1', 'x2', 'y']
    elif problem_param['name'] == 'hartman3':
        ALPHA = np.array([[1.0], [1.2], [3.0], [3.2]])
        A = np.array([[3.0, 10, 30],
                      [0.1, 10, 35],
                      [3.0, 10, 30],
                      [0.1, 10, 35]]).repeat(X.shape[0], axis=0)
        P = 0.0001 * np.array([
            [3689, 1170, 2673],
            [4699, 4387, 7470],
            [1091, 8732, 5547],
            [381, 5743, 8828]]).repeat(X.shape[0], axis=0)
        X_trans = np.tile(X, (4, 1))
        inner_sum = np.sum(np.multiply(A, np.square(X_trans - P)), axis=1).reshape(4, -1)
        result = - np.sum(ALPHA * np.exp(-inner_sum), axis=0)
        problem_param['column_name'] = ['x1', 'x2', 'x3', 'y']
    elif problem_param['name'] == 'hartman6':
        ALPHA = np.array([[1.0], [1.2], [3.0], [3.2]])
        A = np.array([
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],
        ]).repeat(X.shape[0], axis=0)
        P = 0.0001 * np.array([
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ]).repeat(X.shape[0], axis=0)
        X_trans = np.tile(X, (4, 1))
        inner_sum = np.sum(np.multiply(A, np.square(X_trans - P)), axis=1).reshape(4, -1)
        result = - np.sum(ALPHA * np.exp(-inner_sum), axis=0)
        problem_param['column_name'] = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'y']
    elif problem_param['name'] == 'ackley':
        result = -20 * np.exp(-0.2 * np.sqrt(np.sum(np.square(X), axis=1) / problem_param['dimension'])) - np.exp(
            np.sum(np.cos(2 * np.pi * X), axis=1) / problem_param['dimension']) + 20 + np.exp(1)
        if problem_param['dimension'] == 10:
            problem_param['column_name'] = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'y']
        elif problem_param['dimension'] == 20:
            problem_param['column_name'] = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10',
                                            'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'y']
        else:
            problem_param['column_name'] = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10',
                                            'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20',
                                            'x21', 'x22', 'x23', 'x24', 'x25', 'x26', 'x27', 'x28', 'x29', 'x30', 'y']
    elif problem_param['name'] == 'shcb':
        x1, x2 = X[:, 0], X[:, 1]
        a = x1 * x2
        result = (4 - 2.1 * np.square(x1) + np.power(x1, 4) / 3) * np.square(x1) + x1 * x2 + (
                    -4 + 4 * np.square(x2)) * np.square(x2)
        problem_param['column_name'] = ['x1', 'x2', 'y']
    elif problem_param['name'] == 'easom':
        x1, x2 = X[:, 0], X[:, 1]
        result = -np.cos(x1) * np.cos(x2) * np.exp(-np.square(x1 - np.pi) - np.square(x2 - np.pi))
        problem_param['column_name'] = ['x1', 'x2', 'y']
    elif problem_param['name'] == 'alpine':
        x1, x2 = X[:, 0], X[:, 1]
        result = np.abs(x1 * np.sin(x1) + 0.1 * x1) + np.abs(x2 * np.sin(x2) + 0.1 * x2)
        problem_param['column_name'] = ['x1', 'x2', 'y']
    elif problem_param['name'] == 'shekel':
        m = 5
        C = np.array([
            [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
            [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
            [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
            [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6]
        ])
        beta = 0.1 * np.array([[1, 2, 2, 4, 4, 6, 3, 7, 5, 5]])
        C = np.tile(C[:, :m].T, (X.shape[0], 1))
        X_trans = X.repeat(m, axis=0)
        result_p = np.sum(np.square(X_trans - C), axis=1).reshape(-1, m) + beta[0, :m]
        result = - np.sum(1 / result_p, axis=1)
        problem_param['column_name'] = ['x1', 'x2', 'x3', 'x4', 'y']

    else:
        print("New test function")
        result = None
    y = result.reshape(-1, 1)
    return X, y


# Latin Hypercube Sampling
def latin_hypercube_sampling(num_samples):
    if type(problem_param['range'][0]) != int:
        x1_l = problem_param['range'][0][0]
        x1_u = problem_param['range'][1][0]
        x2_l = problem_param['range'][0][1]
        x2_u = problem_param['range'][1][1]
        x_lim = np.array(([x1_l, x1_u], [x2_l, x2_u]))
    else:
        x_lim = np.array(problem_param['range']).reshape(-1, 1)
        x_lim = x_lim.repeat(problem_param['dimension'], axis=1).T
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


def updateRegion(bounds):
    k_fold = 5
    num_local_search = EA_param['local_search_num']
    if current_generation % k_fold == (k_fold - 1):
        exact_sample = Sample_Train
        _, exact_value = testFunc(Sample_Train)
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


def PSO(model_num, weight, base_model_weight):
    dim = problem_param['dimension']
    x_min = problem_param['range'][0] * np.ones(dim)
    x_max = problem_param['range'][1] * np.ones(dim)
    bounds = (x_min, x_max)
    if EA_param['local_search']:
        bounds = updateRegion(bounds)
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

    # 种群中选点测试，不计入评估
    _, sample_true = testFunc(pso_pos)
    pso_cost = pso_cost.reshape(-1, 1)
    sample_disp = np.hstack((pso_pos, pso_cost))
    sample_disp = np.hstack((sample_disp, sample_true))

    # 粒子群测试，不计入评估
    _, pso_true = testFunc(optimizer.swarm.position)
    pso_disp = np.hstack((optimizer.swarm.position, optimizer.swarm.current_cost.reshape(-1, 1)))
    pso_disp = np.hstack((pso_disp, pso_true.reshape(-1, 1)))

    return pso_pos, pso_cost.reshape(-1, 1)


def SAiterate(Sample_Train, Sample_Points, num_boost_round):
    global current_generation
    mt.paraInit()

    for generation in range(0, generations):
        current_generation = generation
        # 训练: 采样点训练XGB代理模型
        base_model, meta_model, model_num, weight, base_model_weight = mt.modelTrain(Sample_Train, parameters,
                                                                                     num_boost_round, generation)
        # PSO
        sample, cost = PSO(model_num, weight, base_model_weight)
        # pso_best, pso_swarm = PSO(model_num, weight, base_model_weight)
        #
        # pso_array = pso_swarm[:, 0:-1]
        #
        # # pso_pred = pso_swarm[:, -1].reshape(-1, 1)
        #
        # # 采样，选择种群中的最佳点作为下一个CAE仿真采样点
        # # 大坑注意：Best是选最小，Worst是选最大(因为初始化时优化问题设置的是最小为优)
        # sample = pso_best[:, 0:-1]
        #
        # # 种群中选点测试，不计入评估
        # _, sample_true = testFunc(sample)
        # sample_disp = np.hstack((sample, sample_true))
        #
        # # 粒子群测试，不计入评估
        # _, pso_true = testFunc(pso_array)
        # pso_disp = np.hstack((pso_swarm, pso_true.reshape(-1, 1)))

        # 迭代过程中选择的点 及 包含初始选点的总训练集
        Sample_Points = np.vstack((Sample_Points, sample))
        Sample_Train = np.vstack((Sample_Train, sample))

        _, exact_samples = testFunc(Sample_Train)

    mt.paraInit()
    result, optimum = resultDisp(Sample_Points, Sample_Init, generations)

    X, y = testFunc(Sample_Train)
    Samples_Disp = np.hstack((X, y))
    Samples_Disp = pd.DataFrame(data=Samples_Disp)
    Samples_Disp.to_csv('./Samples.csv', encoding='gbk')

    return result, optimum


def resultDisp(Sample_Points, Sample_Init, generations):
    """
    在本轮求解中，最佳个体表现，及其真实排名
    绘制每个采样点真实值与迭代次数的图
    绘制每次迭代时最佳值的变化图
    :param Sample_Points: 迭代后所有的采样点
    :param Sample_Init: 初始生成的采样点
    :param generations: 总共的迭代次数
    :return: /
    """
    _, SampleValues = testFunc(Sample_Points)  # 迭代时，每一个采样点的取值
    Min = []  # 每一代时所有采样点的最小值
    _, InitValues = testFunc(Sample_Init)
    min = np.min(InitValues)  # 截止该次代时的最优解
    for i in range(0, len(SampleValues)):
        if SampleValues[i] < min:
            Min.append(float(SampleValues[i]))
            min = float(SampleValues[i])
        else:
            Min.append(min)

    result = np.hstack((Sample_Points, SampleValues))

    result = pd.DataFrame(result, columns=problem_param['column_name'])

    meta_values = []
    for index in range(len(SampleValues)):
        if index % 5 == 4:
            meta_values.append(SampleValues[index])
    meta_values = np.array(meta_values).reshape(-1, 1)

    a = 1
    global computations
    global sample_init
    g = range(computations - sample_init)

    meta_index = np.arange(4, computations - sample_init, 5).reshape(-1, 1)

    plt.figure(figsize=(20, 10))
    plt.xlabel('Generations')
    plt.ylabel('Test Values')
    plt.legend("Select Points", loc='lower right')
    plt.title('Generations vs TestValues')
    plt.plot(g, SampleValues, 'r-', lw=1)
    plt.scatter(g, SampleValues, alpha=1)
    plt.scatter(meta_index, meta_values, alpha=1, s=80, c='r')

    plt.legend("Convergence Curve", loc='lower right')
    plt.plot(g, Min, 'b-', lw=2)
    plt.show()

    result = result.sort_values(by='y', axis=0, ascending=True)
    optimum = np.array(result.iloc[0, :])
    return result, optimum


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

    for i in range(2):
        Sample_Train = Sample_Init  # 用于训练模型的种群定义为Sample_Train
        Sample_Points = np.empty((0, problem_param['dimension']))
        result, optimum = SAiterate(Sample_Train, Sample_Points, num_boost_round)
        Optimum.append(optimum)
        Resdisp.append(optimum[-1])
        print("Minimum Test Value = ", optimum[-1])
        print(i)

    print("Mean value:", np.mean(Resdisp))
    print("Variance value:" + str(np.sqrt(np.var(Resdisp))) + "^2")
    test = pd.DataFrame(data=Optimum, columns=result.columns.values.tolist())  # 数据有三列，列名分别为one,two,three
    csvname = './' + str(seed) + '_' + str(problem_param['name']) + '_' + str(problem_param['dimension']) + 'd_' \
              + str(options['c1']) + '_' + str(options['c2']) + '_' + str(options['w']) + '_' \
              + str(enabled_model[0]) + '_' + str(enabled_model[1]) + '_' + str(enabled_model[2]) \
              + '.csv'
    test.to_csv(csvname, encoding='gbk')
    # deleteModels()
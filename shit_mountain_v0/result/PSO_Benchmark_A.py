import random
import warnings

# import deap
import cma
import xgboost as xgb
import pandas as pd
# from deap import base, creator, tools
import numpy as np
from matplotlib import pyplot as plt
import lhsmdu
from pyDOE import lhs
import pickle
import pyswarms as ps

# import其他py文件
from pyswarms.single import GlobalBestPSO

import PSO_model_A as mt


# 超参数定义
from sklearn.cluster import KMeans


num_cluster = [1, 1]  # 种群聚类个数
threshold = [60, 100]

# retention_rate = 0.8  # 繁育出子代的保留率
seed = int(np.random.rand(1) * 2e3)
print(seed)

parameters = {'seed': 100, 'nthread': -1, 'gamma': 0, 'lambda': 6,
              'max_depth': 3, 'eta': 0.15, 'objective': 'reg:squarederror'}
num_boost_round = 250

# Base model settings
enabled_model = np.array([
    'XGBoost',
    # 'Polynomial',
    'RandomForest',
    'AdaBoost',
])

base_model_weight = np.ones(len(enabled_model)).ravel() / len(enabled_model)

base_model_cache = []

# EA Algorithm settings
options = {
    'c1': 0.2,
    'c2': 0.2,
    'w': 0.8
}

# Test problem settings
problem_param = {
    # 'name': 'rosenbrock',
    # 'dimension': 6,
    # 'range': [-5, 10],

    'name': 'rastrigin',
    'dimension': 8,
    'range': [-5, 5],

    # 'name': 'griewank'
    # 'dimension': 10,
    # 'range': [-600, 600],
}

# 计算资源相关参数
computations = int(problem_param['dimension']) * (5 + 10)
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
        problem_param['column_name'] = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'y']
    elif problem_param['name'] == 'rastrigin':
        result = 10 * problem_param['dimension'] + np.sum(np.square(X) - 10 * np.cos(2 * np.pi * X), axis=1)
        problem_param['column_name'] = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'y']
    elif problem_param['name'] == 'griewank':
        den = 1 / np.sqrt(np.arange(1, problem_param['dimension'] + 1))
        result = np.sum(np.square(X), axis=1) / 4e3 - np.prod(np.cos(np.multiply(X, den))) + 1
        problem_param['column_name'] = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'y']
    else:
        print("New test function")
        result = None
    y = result.reshape(-1, 1)
    return X, y


# Latin Hypercube Sampling
def latin_hypercube_sampling(num_samples):
    global seed
    num_dimensions = problem_param['dimension']
    sample_matrix = lhs(n=num_dimensions,
                        samples=num_samples,
                        criterion="centermaximin",
                        iterations=5,
                        ).T
    # sample_matrix = lhsmdu.sample(numDimensions=num_dimensions,
    #                               numSamples=num_samples,
    #                               randomSeed=seed)
    scale = (problem_param['range'][1] - problem_param['range'][0]) * np.ones((num_dimensions, 1))
    offset = problem_param['range'][0]
    # res = np.array(sample_matrix) * scale + offset
    res = np.multiply(np.array(sample_matrix), scale) + offset
    return res.T


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
    SampleValues = np.abs(SampleValues)
    Min = []  # 每一代时所有采样点的最小值
    _, InitValues = testFunc(Sample_Init)
    InitValues = np.abs(InitValues)
    min = np.min(InitValues)  # 截止该次代时的最优解
    for i in range(0, len(SampleValues)):
        if SampleValues[i] < min:
            Min.append(float(SampleValues[i]))
            min = float(SampleValues[i])
        else:
            Min.append(min)

    result = np.hstack((Sample_Points, SampleValues))

    result = pd.DataFrame(result, columns=problem_param['column_name'])

    global computations
    global sample_init
    g = range(computations - sample_init)
    plt.figure(figsize=(20, 10))
    plt.xlabel('Generations')
    plt.ylabel('Test Values')
    plt.legend("Select Points", loc='lower right')
    plt.title('Generations vs TestValues')
    plt.plot(g, SampleValues, 'r-', lw=1)
    plt.scatter(g, SampleValues, alpha=1)

    plt.legend("Convergence Curve", loc='lower right')
    plt.plot(g, Min, 'b-', lw=2)
    plt.show()

    result = result.sort_values(by='y', axis=0, ascending=True)
    optimum = np.array(result.iloc[0, :])
    return result, optimum

# def popEvaluate(x):
#     global base_model_weight
#     global base_model_cache
#     # base_model_list = base_model_cache
#     weight = base_model_weight
#     pop_pred = []
#
#     batch_size = 5
#     model_list_index = 0
#     model_num = current_generation % 5
#
#     if model_num != 4:
#         batch_size = 1
#
#     for i in range(batch_size):
#         if 'XGBoost' in enabled_model:
#             # model_xgb = base_model_cache[model_list_index, i]
#             model_xgb = base_model_cache[model_list_index]
#             pop_pred_xgb = model_xgb.predict(xgb.DMatrix(x.reshape(1, -1)))
#             pop_pred.append(pop_pred_xgb)
#             model_list_index += 1
#
#         if 'Polynomial' in enabled_model:
#             # model_poly = base_model_cache[model_list_index, i]
#             model_poly = base_model_cache[model_list_index]
#             pop_pred_poly = np.abs(model_poly.predict(x))
#             pop_pred.append(pop_pred_poly)
#             model_list_index += 1
#
#         if 'AdaBoost' in enabled_model:
#             # model_ada = base_model_cache[model_list_index, i]
#             model_ada = base_model_cache[model_list_index]
#             pop_pred_ada = np.abs(model_ada.predict(x))
#             pop_pred.append(pop_pred_ada)
#             model_list_index += 1
#
#         if 'RandomForest' in enabled_model:
#             # model_rf = base_model_cache[model_list_index, i]
#             model_rf = base_model_cache[model_list_index]
#             pop_pred_rf = np.abs(model_rf.predict(x.reshape(1, -1)))
#             pop_pred.append(pop_pred_rf)
#             model_list_index += 1
#
#     if model_num < 4:
#         pop_pred_weighted = np.array(pop_pred).reshape((int(len(pop_pred) / batch_size), batch_size))
#         pop_pred_weighted = pop_pred_weighted * weight
#         pop_pred_weighted = np.sum(pop_pred_weighted) / len(weight)
#         return pop_pred_weighted
#     else:
#         pop_pred_weighted = np.array(pop_pred).reshape((int(len(pop_pred) / batch_size), batch_size))
#         pop_pred_weighted = pop_pred_weighted * weight.reshape(-1, 1)
#
#         # Meta model here
#         meta_input = np.sum(pop_pred_weighted, axis=1)
#         meta = xgb.Booster()
#         meta.load_model('meta_xgb.model')
#         dmeta = xgb.DMatrix(meta_input.reshape(1, -1))
#         meta_pred = np.abs(meta.predict(dmeta))
#
#         return meta_pred

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
        dtest = xgb.DMatrix(pop_array)
        Pred_xgb = np.empty((len(population), 1))
        Pred_poly = np.empty((len(population), 1))
        Pred_knn = np.empty((len(population), 1))

        for i in range(5):
            # xgb base model
            model_xgb = xgb.Booster()
            model_xgb.load_model('xgb' + str(i) + '.model')
            pop_pred_xgb = model_xgb.predict(dtest).reshape(-1, 1)
            pop_pred_xgb = pop_pred_xgb * weight[0][i]
            # poly base model
            with open('rf' + str(i) + '.model', 'rb') as file:
                model_poly = pickle.load(file)
            pop_pred_poly = np.abs(model_poly.predict(pop_array)) * weight[1][i]
            pop_pred_poly = pop_pred_poly.reshape(-1, 1)
            # knn base model
            with open('ada' + str(i) + '.model', 'rb') as file:
                model_knn = pickle.load(file)
            pop_pred_knn = np.abs(model_knn.predict(pop_array)) * weight[2][i]
            pop_pred_knn = pop_pred_knn.reshape(-1, 1)

            Pred_xgb = np.hstack((Pred_xgb, pop_pred_xgb))
            Pred_poly = np.hstack((Pred_poly, pop_pred_poly))
            Pred_knn = np.hstack((Pred_knn, pop_pred_knn.reshape(-1, 1)))
        Pred_xgb = Pred_xgb[:, 1:]
        Pred_poly = Pred_poly[:, 1:]
        Pred_knn = Pred_knn[:, 1:]

        Pred_xgb = np.sum(Pred_xgb, axis=1).reshape(-1, 1)
        Pred_poly = np.sum(Pred_poly, axis=1).reshape(-1, 1)
        Pred_knn = np.sum(Pred_knn, axis=1).reshape(-1, 1)

        # Meta model here
        meta_input = np.hstack((Pred_xgb, Pred_poly, Pred_knn))
        meta = xgb.Booster()
        meta.load_model('meta_xgb.model')
        dmeta = xgb.DMatrix(meta_input)
        meta_pred = np.abs(meta.predict(dmeta))
        meta_pred = meta_pred.ravel()

        return meta_pred
    else:
        pop_array = np.vstack((population[0:]))

        # xgb base model
        model_xgb = xgb.Booster()
        model_xgb.load_model('xgb' + str(model_num) + '.model')
        dtest = xgb.DMatrix(pop_array)
        pop_pred_xgb = np.abs(model_xgb.predict(dtest)).reshape(-1, 1)

        # poly base model
        with open('rf' + str(model_num) + '.model', 'rb') as file:
            model_poly = pickle.load(file)
        pop_pred_poly = np.abs(model_poly.predict(pop_array)).reshape(-1, 1)

        # ada base model
        with open('ada' + str(model_num) + '.model', 'rb') as file:
            model_knn = pickle.load(file)
        pop_pred_knn = np.abs(model_knn.predict(pop_array)).reshape(-1, 1)

        # 各base model赋权重
        pop_pred = base_model_weight[0] * pop_pred_xgb + base_model_weight[1] * pop_pred_poly + base_model_weight[
            2] * pop_pred_knn
        pop_pred = pop_pred.ravel()

        return pop_pred


def PSO(model_num, weight, base_model_weight):

    dim = problem_param['dimension']
    x_min = problem_param['range'][0] * np.ones(dim)
    x_max = problem_param['range'][1] * np.ones(dim)
    bounds = (x_min, x_max)
    # PSO参数设置
    optimizer = GlobalBestPSO(n_particles=200, dimensions=dim, options=options, bounds=bounds)
    # PSO迭代
    cost, pos = optimizer.optimize(popEvaluate, iters=100, model_num=model_num, weight=weight, base_model_weight=base_model_weight)
    cost = np.array(cost).reshape(1, -1)
    pos = pos.reshape(1, -1)

    pso_best = np.hstack((pos, cost))

    pso = optimizer.swarm.pbest_pos
    pso_pred = optimizer.swarm.pbest_cost.reshape(-1, 1)
    pso_swarm = np.hstack((pso, pso_pred))
    return pso_best, pso_swarm


def SAiterate(Sample_Train, Sample_Points, num_boost_round):
    global current_generation
    mt.paraInit()

    for generation in range(0, generations):
        current_generation = generation
        # 训练: 采样点训练XGB代理模型
        base_model, meta_model, model_num, weight, base_model_weight = mt.modelTrain(Sample_Train, parameters,
                                                                                     num_boost_round, generation)
        # PSO
        pso_best, pso_swarm = PSO(model_num, weight, base_model_weight)

        pso_array = pso_swarm[:, 0:-1]
        # pso_pred = pso_swarm[:, -1].reshape(-1, 1)

        # 粒子群测试，不计入评估
        _, pso_true = testFunc(pso_array)
        pso_true = pso_true.reshape(-1, 1)
        pso_disp = np.hstack((pso_swarm, pso_true))

        # 采样，选择种群中的最佳点作为下一个CAE仿真采样点
        # 大坑注意：Best是选最小，Worst是选最大(因为初始化时优化问题设置的是最小为优)
        sample = pso_best[:, 0:-1]

        # 种群中选点测试，不计入评估
        _, sample_true = testFunc(sample)
        sample_disp = np.hstack((sample, sample_true))

        # 迭代过程中选择的点 及 包含初始选点的总训练集
        Sample_Points = np.vstack((Sample_Points, sample))
        Sample_Train = np.vstack((Sample_Train, sample))

    mt.paraInit()
    result, optimum = resultDisp(Sample_Points, Sample_Init, generations)

    X, y = testFunc(Sample_Train)
    Samples_Disp = np.hstack((X, y))
    Samples_Disp = pd.DataFrame(data=Samples_Disp)
    Samples_Disp.to_csv('./Samples.csv', encoding='gbk')

    return result, optimum


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    # 初始化采样点并训练代理模型
    Sample_Init = latin_hypercube_sampling(sample_init)  # 初始化的训练采样点
    Sample_Train = Sample_Init  # 用于训练模型的种群定义为Sample_Train
    Sample_Points = np.empty((0, problem_param['dimension']))

    Optimum = []
    Resdisp = []
    for i in range(10):
        result, optimum = SAiterate(Sample_Train, Sample_Points, num_boost_round)
        Optimum.append(optimum)
        Resdisp.append(optimum[-1])
        print("Minimum Test Value = ", optimum[-1])
        print(i)

    print("Mean value:", np.mean(Resdisp))
    print("Variance value:" + str(np.sqrt(np.var(Resdisp))) + "^2")
    test = pd.DataFrame(data=Optimum, columns=result.columns.values.tolist())  # 数据有三列，列名分别为one,two,three
    test.to_csv('./Opt.csv', encoding='gbk')

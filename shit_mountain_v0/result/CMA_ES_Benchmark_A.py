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

# import其他py文件
import CMA_ES_model_A as mt
# import DEAPchange

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

base_model_weight = np.ones(len(enabled_model)) / len(enabled_model)

base_model_cache = []

# Test problem settings
problem_param = {
    'name': 'rosenbrock',
    'dimension': 6,
    'range': [-5, 10],

    # 'name': 'rastrigin',
    # 'dimension': 8,
    # 'range': [-5, 5],

    # 'name': 'griewank'
    # 'dimension': 10,
    # 'range': [-600, 600],
}

# EA Algorithm settings
CMA_ES_param = {
    'xl': problem_param['range'][0],
    'xu': problem_param['range'][1],
    'popsize': 5 * problem_param['dimension'],
    'sigma': 0.4,
    'bounds': [[problem_param['range'][0]] * problem_param['dimension'],
               [problem_param['range'][1]] * problem_param['dimension']],
    'tolfun': 1e-6,
    'maxfevals': 1000,
    'verb_disp': 0
}


# Selection settings
selection_param = {
    'ts': 2,
}

# 种群大小参数
pop_init = int(problem_param['dimension'])  # 初始种群大小
select_num = int(pop_init * 0.7 / 2) * 2  # 从育种个体中挑选的个体数量
reserve_num = pop_init - select_num

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

cnt_p = 0  # 在重复生成个体的过程中，从拉丁超立方矩阵中提取元素的index

def indInit():
    global cnt_p
    global init_matrix
    res = init_matrix[cnt_p, :].tolist()
    cnt_p = cnt_p + 1
    return res


def latin_hypercube_sampling(num_samples):
    global seed
    num_dimensions = problem_param['dimension']
    sample_matrix = lhs(n=num_dimensions,
                        samples=num_samples,
                        criterion="maximin",
                        iterations=100,
                        ).T
    # sample_matrix = lhsmdu.sample(numDimensions=num_dimensions,
    #                               numSamples=num_samples,
    #                               randomSeed=seed)
    scale = (problem_param['range'][1] - problem_param['range'][0]) * np.ones((num_dimensions, 1))
    offset = problem_param['range'][0]
    # res = np.array(sample_matrix) * scale + offset
    res = np.multiply(np.array(sample_matrix), scale) + offset
    return res.T


def monte_carlo_sampling(num_samples):
    num_dimensions = problem_param['dimension']
    sample_matrix = lhsmdu.createRandomStandardUniformMatrix(numDimensions=num_dimensions,
                                                             numRealizations=num_samples)
    scale = (problem_param['range'][1] - problem_param['range'][0]) * np.ones((num_dimensions, 1))
    offset = problem_param['range'][0]
    res = np.array(sample_matrix) * scale + offset
    return res.T


def updateBaseModel():
    global base_model_cache
    batch_size = 5
    model_list = []
    model_num = current_generation % 5
    if model_num < 4:
        batch_size = 1
    for i in range(batch_size):
        if 'XGBoost' in enabled_model:
            model_xgb = xgb.Booster()
            model_xgb.load_model('xgb' + str(i) + '.model')
            model_list.append(model_xgb)

        if 'Polynomial' in enabled_model:
            with open('poly' + str(i) + '.model', 'rb') as file:
                model_poly = pickle.load(file)
            model_list.append(model_poly)

        if 'AdaBoost' in enabled_model:
            with open('ada' + str(i) + '.model', 'rb') as file:
                model_ada = pickle.load(file)
            model_list.append(model_ada)

        if 'RandomForest' in enabled_model:
            with open('rf' + str(i) + '.model', 'rb') as file:
                model_rf = pickle.load(file)
            model_list.append(model_rf)

    # base_model_cache = np.array(model_list).reshape((int(len(model_list) / batch_size), batch_size))
    base_model_cache = model_list

    return base_model_cache


def sampleSelect(pop_array, pop_pred, sample_Train):
    """
    从现存种群中选出下一个采样点，本函数和贝叶斯优化的采集函数具有类似的功能，需要对贪心和探索进行平衡。
    目前的思路是随着训练模型的样本量逐渐增加，选择更相信模型预测结果。
    :param candidate_population: 候选采样点种群
    :param train_population: 训练代理模型的种群
    :return: 下一个采样点
    """
    if pop_array.shape[0] == pop_pred.shape[0]:
        if len(pop_pred.shape) == 1:
            pop_pred = pop_pred.reshape(-1, 1)
        POP = np.hstack((pop_array, pop_pred))  # 适应度和特征拼起来的array
    else:
        print("Array size mismatched")

    sample_num = len(sample_Train)
    if sample_num < threshold[0]:
        cluster = num_cluster[0]
        select_index = np.arange(0, cluster)
        resample_round = 1
    else:
        cluster = num_cluster[1]
        select_index = np.arange(0, cluster)
        resample_round = len(select_index)

    sample_select = np.empty((0, problem_param['dimension']))
    Clusters = []
    Cluster_Mean = []
    kms = KMeans(init='k-means++', n_clusters=cluster, random_state=1, tol=1e-3)
    pred = kms.fit_predict(POP)
    for i in range(cluster):
        c = POP[pred[0:] == i, :]
        Clusters.append(c)
        Cluster_Mean.append(np.mean(c))
    c_index = Cluster_Mean.index(np.min(Cluster_Mean))
    cluster_select = Clusters[c_index]
    index = np.argsort(cluster_select[:, -1])
    C = cluster_select[index, :]
    select = C[select_index, 0:-1]

    # 样本点去重：如果重复则选择次优的点
    for k in range(resample_round):
        default_select = 0
        if resample_round == 1:
            sub_select = select
        else:
            sub_select = select[k, :]
        while ~np.any(sample_Train - sub_select, axis=1).all():
            default_select += 1
            if default_select < C.shape[0]:
                sub_select = C[default_select, :-1]
                if resample_round == 1:
                    select = sub_select
                else:
                    select[k, :] = sub_select
            else:
                print("Too many repeated sample attempts, fail to select a new point.")
                select[k] = C[0, :-1]
                break

    sample_select = np.vstack((sample_select, select))
    return sample_select


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


def popEvaluate(x):
    global base_model_weight
    global base_model_cache
    # base_model_list = base_model_cache
    weight = base_model_weight
    pop_pred = []

    batch_size = 5
    model_list_index = 0
    model_num = current_generation % 5

    if model_num != 4:
        batch_size = 1

    for i in range(batch_size):
        if 'XGBoost' in enabled_model:
            # model_xgb = base_model_cache[model_list_index, i]
            model_xgb = base_model_cache[model_list_index]
            pop_pred_xgb = model_xgb.predict(xgb.DMatrix(x.reshape(1, -1)))
            pop_pred.append(pop_pred_xgb)
            model_list_index += 1

        if 'Polynomial' in enabled_model:
            # model_poly = base_model_cache[model_list_index, i]
            model_poly = base_model_cache[model_list_index]
            pop_pred_poly = np.abs(model_poly.predict(x.reshape(1, -1)))
            pop_pred.append(pop_pred_poly)
            model_list_index += 1

        if 'AdaBoost' in enabled_model:
            # model_ada = base_model_cache[model_list_index, i]
            model_ada = base_model_cache[model_list_index]
            pop_pred_ada = np.abs(model_ada.predict(x.reshape(1, -1)))
            pop_pred.append(pop_pred_ada)
            model_list_index += 1

        if 'RandomForest' in enabled_model:
            # model_rf = base_model_cache[model_list_index, i]
            model_rf = base_model_cache[model_list_index]
            pop_pred_rf = np.abs(model_rf.predict(x.reshape(1, -1)))
            pop_pred.append(pop_pred_rf)
            model_list_index += 1

    if model_num < 4:
        pop_pred_weighted = np.array(pop_pred).reshape((int(len(pop_pred) / batch_size), batch_size))
        pop_pred_weighted = pop_pred_weighted * weight
        pop_pred_weighted = np.sum(pop_pred_weighted) / len(weight)
        return pop_pred_weighted
    else:
        pop_pred_weighted = np.array(pop_pred).reshape((int(len(pop_pred) / batch_size), batch_size))
        pop_pred_weighted = pop_pred_weighted * weight.reshape(-1, 1)

        # Meta model here
        meta_input = np.sum(pop_pred_weighted, axis=1)
        meta = xgb.Booster()
        meta.load_model('meta_xgb.model')
        dmeta = xgb.DMatrix(meta_input.reshape(1, -1))
        meta_pred = np.abs(meta.predict(dmeta))

        return meta_pred


def CMA_ES(pop_array):
    """
        CMA-ES Evolutionary Algorithm
        -----------------------------
        fun: fitness function
        x0: initial solution/points
        sigma0: initial standard deviation to sample new solutions
        :return: /
    """

    pop_children = []
    pop_best = []
    pred_best = []

    pred_initial = []
    for j in range(pop_array.shape[0]):
        pred_initial.append(popEvaluate(pop_array[j, :]))
    pred_initial = np.array(pred_initial).reshape(-1, 1)
    init_index = np.argmin(pred_initial)

    fun = popEvaluate
    # fun = cma.ff.rastrigin

    x0 = pop_array[init_index, :]
    x_best, es = cma.fmin2(fun, x0, CMA_ES_param['sigma'],
                           {'popsize': CMA_ES_param['popsize'],
                            'bounds': CMA_ES_param['bounds'],
                            'tolfun': CMA_ES_param['tolfun'],
                            'maxfevals': CMA_ES_param['maxfevals'],
                            'verb_disp': CMA_ES_param['verb_disp'],
                            })
    pop_best.append(x_best)
    pred_best.append(es.result.fbest)
    pop_children.append(es.pop_sorted)

    pop_children = np.array(pop_children[0])
    pop_children = np.clip(pop_children, a_min=CMA_ES_param['xl'], a_max=CMA_ES_param['xu'
                                                                                      ''])
    pred_children = []
    for j in range(pop_children.shape[0]):
        pred_children.append(popEvaluate(pop_children[j, :]))
    pred_children = np.array(pred_children).reshape(-1, 1)

    return pop_children, np.array(pred_children), np.array(pop_best), np.array(pred_best)


def SAiterate(Sample_Train, Sample_Points, num_boost_round):
    global current_generation
    mt.paraInit()

    for generation in range(0, generations):
        current_generation = generation
        # 初始化
        pop_f = latin_hypercube_sampling(pop_init ** 2)
        # 训练: 采样点训练XGB代理模型
        base_model, meta_model, model_num, weight, base_model_weight = mt.modelTrain(Sample_Train, parameters,
                                                                                     num_boost_round, generation)
        # 加载储存的模型
        base_model_cache = updateBaseModel()
        # 演化算法产生候选点：CMA_ES
        pop_children, pred_children, pop_best, pred_best = CMA_ES(pop_f)


        # EAgenerations = (generation + 1) * 50
        # pop_array, pop_pred = EAiterate(base_model, meta_model, model_num, weight, base_model_weight, EAgenerations)

        # 种群测试，不计入评估
        _, pop_true = testFunc(pop_children)
        pop_true = pop_true.reshape(-1, 1)

        pop_disp = np.hstack((pop_children, pred_children, pop_true))

        # 采样，选择种群中的最佳点作为下一个CAE仿真采样点
        # 大坑注意：Best是选最小，Worst是选最大(因为初始化时优化问题设置的是最小为优)
        sample = sampleSelect(pop_children, pred_children, Sample_Train)

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
    # 使用拉丁超立方，或蒙特卡洛，或随机抽样方法生成初始种群
    init_matrix = latin_hypercube_sampling(pop_init)

    # 初始化采样点并训练代理模型
    Sample_Init = latin_hypercube_sampling(sample_init)  # 初始化的训练采样点
    Sample_Train = Sample_Init  # 用于训练模型的种群定义为Sample_Train
    Sample_Points = np.empty((0, problem_param['dimension']))

    Optimum = []
    Resdisp = []
    for i in range(5):
        result, optimum = SAiterate(Sample_Train, Sample_Points, num_boost_round)
        Optimum.append(optimum)
        Resdisp.append(optimum[-1])
        print("Minimum Test Value = ", optimum[-1])
        print(i)

    print("Mean value:", np.mean(Resdisp))
    print("Variance value:" + str(np.sqrt(np.var(Resdisp))) + "^2")
    test = pd.DataFrame(data=Optimum, columns=result.columns.values.tolist())  # 数据有三列，列名分别为one,two,three
    test.to_csv('./Opt.csv', encoding='gbk')
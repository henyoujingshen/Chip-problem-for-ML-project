import warnings

import deap
import xgboost as xgb
import pandas as pd
from deap import base, creator, tools
import numpy as np
from matplotlib import pyplot as plt
import lhsmdu
import pickle

# import其他py文件
import stacking_model_v7 as mt
import DEAPchange

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

# Mutation settings
mutation_param = {
    'dim': problem_param['dimension'],
    'mutation_prob': 0.7,
    'eta': 2,
    'mu': 0,
    'sigma': 2,
    'xl': problem_param['range'][0],
    'xu': problem_param['range'][1],
}
# Crossover settings
crossover_param = {
    'crossover_prob': 0.5,
}
# Selection settings
selection_param = {
    'ts': 2,
}
# 种群大小参数
pop_init = int(problem_param['dimension']) * 30  # 初始种群大小
select_num = int(pop_init * 0.7)  # 从育种个体中挑选的个体数量
reserve_num = pop_init - select_num
# 计算资源相关参数
computations = int(problem_param['dimension']) * 10
sample_init = int(problem_param['dimension']) * 4
generations = computations - sample_init

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

# def mutation(individual):
#     """
#     定义个体突变函数，在toolbox中注册为Mutation函数
#     polynomial mutation
#     :param individual: 进行突变操作的个体
#     :return: 完成突变操作后的个体
#     """
#
#     eta = mutation_param['eta']
#     xl = mutation_param['xl']
#     xu = mutation_param['xu']
#     xp = np.array(individual[0])  # x parents
#     u = np.random.rand(1)
#     prob = np.random.rand(1)
#
#     if prob > mutation_param['mutation_prob']:
#         return individual
#     else:
#         if u <= 0.5:
#             sigma = 2 * u + (1 - 2 * u) * np.power(1 - (xp - xl) / (xu - xl), eta + 1) - 1
#             Sigma = np.sign(sigma) * np.power(np.abs(sigma), 1 / (eta + 1))
#         else:
#             sigma = 1 - np.power(2 * (1 - u) + (2 * u - 1) * (1 - (xu - xp) / (xu - xl)), eta + 1)
#             Sigma = np.sign(sigma) * np.power(np.abs(sigma), 1 / (eta + 1))
#
#         xo = xp + Sigma * (xu - xl)
#         xo = np.clip(xo, a_min=xl, a_max=xu)
#         individual[0] = xo.tolist()
#         if np.isnan(xo).any():
#             print("failed")
#         return individual




# def crossover(ind1, ind2, pb):
#     rand = np.random.rand(1)
#     if rand <= pb:
#         x1 = 0.5 * ((1 + pb) * np.array(ind1[0]) + (1 - pb) * np.array(ind2[0]))
#         x2 = 0.5 * ((1 - pb) * np.array(ind1[0]) + (1 + pb) * np.array(ind2[0]))
#         ind1[0] = x1.tolist()
#         ind2[0] = x2.tolist()
#         if (np.isnan(x1).any() or np.isnan(x2).any()):
#             print("failed")
#     return ind1, ind2


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
    sample_matrix = lhsmdu.sample(numDimensions=num_dimensions,
                                  numSamples=num_samples,
                                  randomSeed=seed)
    scale = (problem_param['range'][1] - problem_param['range'][0]) * np.ones((num_dimensions, 1))
    offset = problem_param['range'][0]
    res = np.array(sample_matrix) * scale + offset
    return res.T

def monte_carlo_sampling(num_samples):
    num_dimensions = problem_param['dimension']
    sample_matrix = lhsmdu.createRandomStandardUniformMatrix(numDimensions=num_dimensions,
                                                             numRealizations=num_samples)
    scale = (problem_param['range'][1] - problem_param['range'][0]) * np.ones((num_dimensions, 1))
    offset = problem_param['range'][0]
    res = np.array(sample_matrix) * scale + offset
    return res.T


def popEvaluate(base_model, meta_model, model_num, weight, base_model_weight, population):
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
            with open('poly' + str(i) + '.model', 'rb') as file:
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

        for i, individual in zip(range(len(population)), population):
            individual.fitness.values = np.array(meta_pred[i]).ravel()
        return pop_array, meta_pred
    else:
        pop_array = np.vstack((population[0:]))

        # xgb base model
        model_xgb = xgb.Booster()
        model_xgb.load_model('xgb' + str(model_num) + '.model')
        dtest = xgb.DMatrix(pop_array)
        pop_pred_xgb = np.abs(model_xgb.predict(dtest)).reshape(-1, 1)

        # poly base model
        with open('poly' + str(model_num) + '.model', 'rb') as file:
            model_poly = pickle.load(file)
        pop_pred_poly = np.abs(model_poly.predict(pop_array)).reshape(-1, 1)

        # ada base model
        with open('ada' + str(model_num) + '.model', 'rb') as file:
            model_knn = pickle.load(file)
        pop_pred_knn = np.abs(model_knn.predict(pop_array)).reshape(-1, 1)

        # 各base model赋权重
        pop_pred = base_model_weight[0] * pop_pred_xgb + base_model_weight[1] * pop_pred_poly + base_model_weight[
            2] * pop_pred_knn

        for i, individual in zip(range(len(population)), population):
            individual.fitness.values = np.array(pop_pred[i]).ravel()
        return pop_array, pop_pred

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

def EAiterate(base_model, meta_model, model_num, weight, base_model_weight, EAgenerations):
    # 初始化种群，同时将种群采样index重置为0
    pop = toolbox.Population(n=pop_init)  # 种群定义为pop
    global cnt_p
    cnt_p = 0
    # 对初始化的种群进行个体适应度评估
    popEvaluate(base_model=base_model, meta_model=meta_model, model_num=model_num, weight=weight,
                base_model_weight=base_model_weight, population=pop)
    # 演化算法迭代
    for g in range(EAgenerations):
        # 从种群中选出部分优势个体作为育种父代
        pop_f = deap.tools.selStochasticUniversalSampling(pop, k=select_num, fit_attr='fitness')
        # 育种父代混合交叉
        pop_cross = [toolbox.clone(individual) for individual in pop_f]



        for i in range(0, len(pop_cross) - len(pop_cross) % 2, 2):
            toolbox.Crossover(ind1=pop_cross[i], ind2=pop_cross[i + 1], alpha=crossover_param['crossover_prob'],
                              param=problem_param)
        # 交叉后的个体突变
        pop_mutation = [toolbox.clone(individual) for individual in pop_cross]
        for ind in pop_mutation:
            toolbox.Mutation(individual=ind, param=mutation_param)
        # 评估新产生的个体
        popEvaluate(base_model=base_model, meta_model=meta_model, model_num=model_num, weight=weight,
                                          base_model_weight=base_model_weight, population=pop_mutation)
        # 环境适应度选择子代
        pop_children = deap.tools.selStochasticUniversalSampling(pop_mutation, k=select_num, fit_attr='fitness')
        # pop_select = deap.tools.selTournament(pop_children, k=select_num, tournsize=3, fit_attr='fitness')

        # 保留父代中的优势个体,与子代合并形成下一代种群(Elitist Reinsertion)
        pop_reserve = tools.selBest(pop_f, k=reserve_num)
        pop = pop_reserve + pop_children
        pop_array, pop_pred = popEvaluate(base_model=base_model, meta_model=meta_model, model_num=model_num, weight=weight,
                    base_model_weight=base_model_weight, population=pop)
        pop_pred = pop_pred.reshape(-1, 1)

        # 子代和下一代种群个体和真实值查看
        # child_array = np.vstack((pop_children[0:]))
        # _, child_true = testFunc(child_array)
        # child_disp = np.hstack((child_array, child_true))
        # _, pop_true = testFunc(pop_array)
        # pop_true = pop_true.reshape(-1, 1)
        # pop_disp = np.hstack((pop_array, pop_pred, pop_true))

        a = 1

    return pop_array, pop_pred


def SAiterate(Sample_Train, Sample_Points, num_boost_round):
    # 初始化种群
    pop_f = toolbox.Population(n=pop_init)  # 种群定义为pop
    global cnt_p
    cnt_p = 0
    mt.paraInit()
    for generation in range(0, generations):
        # 训练
        # 采样点训练XGB代理模型
        # print(generation)
        base_model, meta_model, model_num, weight, base_model_weight = mt.modelTrain(Sample_Train, parameters,
                                                                                     num_boost_round, generation)
        # 演化算法种群迭代
        EAgenerations = (generation + 1) * 20
        pop_array, pop_pred = EAiterate(base_model, meta_model, model_num, weight, base_model_weight, EAgenerations)

        _, pop_true = testFunc(pop_array)
        pop_true = pop_true.reshape(-1, 1)
        pop_disp = np.hstack((pop_array, pop_pred, pop_true))
        # 采样，选择种群中的最佳点作为下一个CAE仿真采样点
        # 大坑注意：Best是选最小，Worst是选最大(因为初始化时优化问题设置的是最小为优)
        sample = sampleSelect(pop_array, pop_pred, Sample_Train)  # 选择下一个采样点
        # 种群选点情况测试
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

# def iterate(Sample_Train, Sample_Points, num_boost_round):
#     # 初始化种群
#     pop_f = toolbox.Population(n=pop_init)  # 种群定义为pop
#     global cnt_p
#     cnt_p = 0
#     mt.paraInit()
#     for generation in range(0, generations):
#         # 训练
#         # 采样点训练XGB代理模型
#         # print(generation)
#         base_model, meta_model, model_num, weight, base_model_weight = mt.modelTrain(Sample_Train, parameters,
#                                                                                      num_boost_round, generation)
#
#         # 评估
#         # 种群个体fitness评估
#         popEvaluate(base_model=base_model, meta_model=meta_model, model_num=model_num, weight=weight,
#                     base_model_weight=base_model_weight, population=pop_f)
#         # toolbox.Evaluate(population=pop)
#         # 均匀交叉
#         pop_cross = [toolbox.clone(individual) for individual in pop_f]
#         for i in range(0, len(pop_cross) - len(pop_cross) % 2, 2):
#             toolbox.Crossover(ind1=pop_cross[i], ind2=pop_cross[i + 1], alpha=crossover_param['crossover_prob'], param=problem_param)
#         # 变异
#         pop_mutation = [toolbox.clone(individual) for individual in pop_cross]
#         # pop_mutation = pop_cross
#         for ind in pop_mutation:
#             toolbox.Mutation(individual=ind, param=mutation_param)
#         # 评估新产生的个体
#         pop_children = pop_mutation
#         pop_array, pop_pred = popEvaluate(base_model=base_model, meta_model=meta_model, model_num=model_num, weight=weight,
#                     base_model_weight=base_model_weight, population=pop_children)
#         pop_pred = pop_pred.reshape(-1, 1)
#         # 选择
#         # 一般用轮盘赌或锦标赛，但是此问题fitness会为负，故选用锦标赛，tournsize个体选1，跑k次
#         # pop_select = deap.tools.selTournament(pop_children, k=select_num, tournsize=3, fit_attr='fitness')
#         # pop_select = deap.tools.selStochasticUniversalSampling(pop_children, k=select_num, fit_attr='fitness')
#         # pop_select = deap.tools.selLexicase(pop_children, k=select_num)
#         pop_select = deap.tools.selStochasticUniversalSampling(pop_children, k=select_num, fit_attr='fitness')
#         # 采样
#         # 选择种群中的最佳点作为下一个CAE仿真采样点
#         # 大坑注意：Best是选最小，Worst是选最大(因为初始化时优化问题设置的是最小为优)
#         sample = sampleSelect(pop_array, pop_pred, Sample_Train)  # 选择下一个采样点
#         # 种群选点情况测试
#         _, pop_true = testFunc(pop_array)
#         _, sample_true = testFunc(sample)
#         child_array = np.vstack((pop_select[0:]))
#         _, child_true = testFunc(child_array)
#         sample_disp = np.hstack((sample, sample_true))
#         pop_disp = np.hstack((pop_array, pop_pred, pop_true))
#         child_disp = np.hstack((child_array, child_true))
#
#         Sample_Points = np.vstack((Sample_Points, sample))
#         Sample_Train = np.vstack((Sample_Train, sample))
#         # 保留父代中的优势个体
#         pop_f = tools.selBest(pop_f, k=reserve_num)
#         pop_f = pop_f + pop_select
#     mt.paraInit()
#     result, optimum = resultDisp(Sample_Points, Sample_Init, generations)
#
#     X, y = testFunc(Sample_Train)
#     Samples_Disp = np.hstack((X, y))
#     Samples_Disp = pd.DataFrame(data=Samples_Disp)
#     Samples_Disp.to_csv('./Samples.csv', encoding='gbk')
#
#     return result, optimum


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    # 使用拉丁超立方，或蒙特卡洛，或随机抽样方法生成初始种群
    init_matrix = latin_hypercube_sampling(pop_init)
    # 定义单目标/多目标优化问题，weights表示每个目标的权重，weight取1表示最大化，-1表示最小化
    creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
    # 创建Individual类，继承list
    creator.create('Individual', list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    # 定义个体生成函数
    # 函数别名为Individual，真实调用的是tool.initRepeat函数，该函数的传参为container, func, n，container用的是Individual类，func生成个体调用的是之前定义的randomX函数
    # Latin Hypercube Sampling
    toolbox.register('Individual', tools.initRepeat, creator.Individual, indInit, n=1)
    # 定义种群生成函数Population
    toolbox.register('Population', tools.initRepeat, list, toolbox.Individual)
    # 定义突变函数
    toolbox.register("Mutation", DEAPchange.mutPoly)
    # 定义交叉函数
    toolbox.register("Crossover", DEAPchange.cxBlend)

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
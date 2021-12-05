import codecs
import csv

import deap
import xgboost as xgb
import pandas as pd
from deap import base, creator, tools
import numpy as np
from matplotlib import pyplot as plt
import lhsmdu

# 超参数定义
from sklearn.cluster import KMeans

sample_init = 20  # 初始代理模型样本点大小
pop_init = 200  # 初始种群大小
computations = 100  # 总共的计算资源(CAE)
num_cluster = [4, 1]  # 种群聚类个数
threshold = [60, 100]
generations = int((threshold[0] - sample_init)/num_cluster[0] + (threshold[1] - threshold[0])/num_cluster[1])
# generations = int((computations - sample_init) / cluster)  # 迭代次数
cross_pb = 0.7  # 交叉概率
mutation_pb = 0.4  # 突变概率
select_num = 150  # 从父代中选择出的育种个体数量

ts = 2  # 锦标赛一次选出ts个个体
# retention_rate = 0.8  # 繁育出子代的保留率
np.random.seed(666)  # 随机数种子
seed = int(np.random.rand(1) * 2e3)
print(seed)
LHS = True
parameters = {'seed': 100, 'nthread': -1, 'gamma': 0, 'lambda': 6,
                  'max_depth': 6, 'eta': 0.15, 'objective': 'reg:squarederror'}
num_boost_round = 250


def custom_function(iter, maxiter):
    rang = [0.01, 0.99]
    x = (rang[1] * iter + rang[0] * (maxiter - iter) - (rang[0] + rang[1]) * maxiter / 2) / maxiter * 10 / (
                rang[1] - rang[0])
    fx = 1.0 / (1 + np.exp(-x))
    return fx

def mutation(individual, pb):
    """
    定义个体突变函数，在toolbox中注册为Mutation函数
    :param individual: 进行突变操作的个体
    :param pb: 个体每个维度发生突变的概率
    :return: 完成突变操作后的个体
    """
    t0 = np.random.choice([0, 1], replace=False, p=[1 - pb, pb])
    t1 = np.random.choice([0, 1], replace=False, p=[1 - pb, pb])
    t2 = np.random.choice([0, 1], replace=False, p=[1 - pb, pb])
    t3 = np.random.choice([0, 1], replace=False, p=[1 - pb, pb])
    t4 = np.random.choice([0, 1], replace=False, p=[1 - pb, pb])
    prob = [0.2, 0.2, 0.2, 0.2, 0.2]
    if t0 == 1:
        a = np.random.choice([200, 230, 260, 290, 320], replace=True, p=prob)
        individual[0][0] = a
    if t1 == 1:
        b = np.random.choice([20, 25, 30, 35, 40], replace=True, p=prob)
        individual[0][1] = b
    if t2 == 1:
        c = np.random.choice([200, 225, 250, 275, 300], replace=True, p=prob)
        individual[0][2] = c
    if t3 == 1:
        d = np.random.choice([550, 650, 750, 850, 950], replace=True, p=prob)
        individual[0][3] = d
    if t4 == 1:
        e = np.random.choice([8, 9, 10, 11, 12], replace=True, p=prob)
        individual[0][4] = e
    return individual

# Latin Hypercube Sampling
cnt_p = 0
def pop_lhs_init():
    global cnt_p
    global k_p
    res = k_p[cnt_p, :].tolist()
    cnt_p = cnt_p + 1
    return res

def lhs(num_samples):
    global seed
    num_dimensions = 5
    k = np.array(lhsmdu.sample(numDimensions=num_dimensions,
                               numSamples=num_samples,
                               randomSeed=seed))
    scale = np.array([30, 5, 25, 100, 1]).reshape(-1, 1)
    offset = np.array([200, 20, 200, 550, 8]).reshape(-1, 1)
    res = (np.multiply(scale, np.rint((num_dimensions - 1) * k)) + offset).T
    return res

def findSamples_nofor(sample_array, Data):
    """
    在csv数据集中找到采样点对应的翘曲率(CAE过程模拟)，避免使用for循环，加速计算
    :param population:传参为所需要获取翘曲率的采样点array
    :return: 分别为特征X和翘曲率y，array
    """
    scale = np.array([30, 5, 25, 100, 1])
    offset = np.array([200, 20, 200, 550, 8])
    sample_index = sample_array - offset
    sample_index = sample_index / scale
    index = np.multiply(np.array([5 ** 4, 5 ** 3, 5 ** 2, 5 ** 1, 5 ** 0]), sample_index)
    index = np.sum(index, axis=1)
    Samples = Data[index.astype(int), :6]
    X = Samples[:, :-1]
    y = Samples[:, -1] * 1000
    return X, y

def modelTrain(Sample_Train, Data, parameters, num_boost_round):
    """
    根据训练种群，训练XGBoost代理模型，同时将模型保存为xgb.model文件
    :param population: 用于训练代理模型的种群，为初始种群加每代选取的采样点个体
    :return: /
    """
    X, y = findSamples_nofor(Sample_Train, Data)
    y = y.ravel()
    # 训练XGBoost代理模型 作为evaluate function
    dtrain = xgb.DMatrix(X, label=y)
    model = xgb.train(parameters, dtrain, num_boost_round=num_boost_round)
    # 保存训练好的xgb模型
    model.save_model('xgb.model')

def popEvaluate(population):
    """
    对种群内所有个体的适应度进行评估
    :param population: 需要进行适应度评估的种群
    :return: /
    """
    model = xgb.Booster()
    model.load_model('xgb.model')

    pop_array = np.vstack((population[0:]))
    dtest = xgb.DMatrix(pop_array)
    pop_pred = np.abs(model.predict(dtest))
    for i, individual in zip(range(len(population)), population):
        individual.fitness.values = np.array(pop_pred[i]).ravel()
    return pop_array, pop_pred

def sampleSelect(candidate_population, sample_num):
    """
    从现存种群中选出下一个采样点，本函数和贝叶斯优化的采集函数具有类似的功能，需要对贪心和探索进行平衡。
    目前的思路是随着训练模型的样本量逐渐增加，选择更相信模型预测结果。样本量小时在前5名
    :param candidate_population: 候选采样点种群
    :param train_population: 训练代理模型的种群
    :return: 下一个采样点
    """
    global num_cluster
    global threshold
    pop = candidate_population

    pop_array, pop_pred = popEvaluate(pop)
    POP = np.hstack((pop_array, pop_pred.reshape(-1, 1)))  # 适应度和特征拼起来的array

    if sample_num < threshold[0]:
        sample_select = np.empty((0, 5))
        Clusters = []
        Cluster_Mean = []
        cluster = num_cluster[0]
        kms = KMeans(init='k-means++', n_clusters=cluster, random_state=1, tol=1e-3)
        pred = kms.fit_predict(POP)
        for i in range(cluster):
            c = POP[pred[0:] == i, 0:6]
            Clusters.append(c)
            Cluster_Mean.append(np.mean(c))
        c_index = Cluster_Mean.index(np.min(Cluster_Mean))
        cluster_select = Clusters[c_index]
        index = np.argsort(cluster_select[:, 5])
        C = cluster_select[index, :]
        select = C[0:4, 0:5]
        select.reshape(1, -1)
        sample_select = np.vstack((sample_select, select))
        return sample_select
    elif sample_num <= threshold[1]:
        sample_select = np.empty((0, 5))
        Clusters = []
        Cluster_Mean = []
        cluster = num_cluster[1]
        kms = KMeans(init='k-means++', n_clusters=cluster, random_state=1, tol=1e-3)
        pred = kms.fit_predict(POP)
        for i in range(cluster):
            c = POP[pred[0:] == i, 0:6]
            Clusters.append(c)
            Cluster_Mean.append(np.mean(c))
        c_index = Cluster_Mean.index(np.min(Cluster_Mean))
        cluster_select = Clusters[c_index]
        index = np.argsort(cluster_select[:, 5])
        C = cluster_select[index, :]
        select = C[0, 0:5]
        select.reshape(1, -1)
        sample_select = np.vstack((sample_select, select))
        return sample_select

def ranking(result):
    """
    获取某个体适应度整个真实搜索域中的排名
    :param result: 某个体适应度
    :return: 排名
    """
    benchmark = [0, 0.0168, 0.0239, 0.057, 0.059, 0.0689, 0.0729, 0.0809, 0.0858, 0.115013, 0.130164, 0.142919,
                 0.147871, 0.166071, 0.167726, 0.180223, 0.195995, 0.196783, 0.217258, 0.239674, 0.255737, 0.270209,
                 0.289178, 0.298477, 0.309586, 0.322446, 0.337163, 0.339115, 0.349386, 0.373878, 0.374401, 0.379241,
                 0.385782, 0.392128, 0.413967, 0.438757, 0.453112, 0.466661, 0.474914, 0.477166, 0.485028, 0.517961,
                 0.519821, 0.538555, 0.545344, 0.566704, 0.567065, 0.57607, 0.580767, 0.58662, 0.588015, 0.59527,
                 0.598695, 0.61892, 0.629343, 0.630311, 0.631692, 0.655743, 0.656904, 0.657748, 0.662486, 0.665945,
                 0.668199, 0.674794, 0.697299, 0.697769, 0.70639, 0.712995, 0.732316, 0.733463, 0.736107, 0.736155,
                 0.768301, 0.780585, 0.811329, 0.812936, 0.821925, 0.887417, 0.895446, 0.928705, 0.938831, 0.959684,
                 0.964033, 0.974505, 1.016448, 1.02202, 1.025155, 1.042937, 1.044545, 1.058703, 1.08692, 1.10015,
                 1.103519, 1.110195, 1.111611, 1.143553, 1.149645, 1.158169, 1.16672, 1.196585]
    new_arr = np.sort(np.append(np.array(benchmark), result))
    index = np.where(new_arr == result)
    return int(np.min(index))

def resultDisp(Sample_Points, Sample_Init, generations, Data):
    """
    在本轮求解中，最佳个体表现，及其真实排名
    绘制每个采样点真实值与迭代次数的图
    绘制每次迭代时最佳值的变化图
    :param Sample_Points: 迭代后所有的采样点
    :param Sample_Init: 初始生成的采样点
    :param generations: 总共的迭代次数
    :return: /
    """
    _, SampleValues = findSamples_nofor(Sample_Points, Data)  # 迭代时，每一个采样点的取值
    SampleValues = np.abs(SampleValues)
    Min = []  # 每一代时所有采样点的最小值
    _, InitValues = findSamples_nofor(Sample_Init, Data)
    InitValues = np.abs(InitValues)
    min = np.min(InitValues)  # 截止该次代时的最优解
    for i in range(0, len(SampleValues)):
        if SampleValues[i] < min:
            Min.append(float(SampleValues[i]))
            min = float(SampleValues[i])
        else:
            Min.append(min)

    global computations
    global sample_init
    g = range(computations - sample_init)
    plt.figure(figsize=(20, 10))
    plt.xlabel('Generations')
    plt.ylabel('Sample Values')
    plt.legend('EA-XGB', loc='lower right')
    plt.title('EA&XGB - Generations vs SampleValues')
    plt.plot(g, SampleValues, 'r-', lw=1)
    plt.scatter(g, SampleValues, alpha=1)

    # plt.figure(figsize=(20, 10))
    # plt.xlabel('Generations')
    # plt.ylabel('BestWrapRate')
    plt.legend('EA-XGB', loc='lower right')
    # plt.title('EA&XGB - Generations vs BestWrapRate')
    plt.plot(g, Min, 'b-', lw=2)
    plt.show()
    optimum = np.min(Min)
    print("Minimum Wrap Rate = ", optimum)
    rank = ranking(optimum)
    print("rank = ", rank)
    return rank

def iterate(Sample_Train, Sample_Points, Data, num_boost_round):
    # 初始化种群
    pop = toolbox.Population(n=pop_init)  # 种群定义为pop
    global cnt_p
    cnt_p = 0
    for generation in range(0, generations):
        # print(custom_function(generation, generations))
        # parameters['eta'] = custom_function(generation, generations)
        # 评估
        # 种群个体fitness评估
        popEvaluate(population=pop)
        # toolbox.Evaluate(population=pop)
        # 均匀交叉
        pop_cross = [toolbox.clone(individual) for individual in pop]
        for i in range(0, len(pop_cross) - len(pop_cross) % 2, 2):
            deap.tools.cxUniform(ind1=pop_cross[i], ind2=pop_cross[i + 1], indpb=cross_pb)
            # tools.cxTwoPoint(a, b)
            # tools.cxOnePoint(a, b)
        # 变异
        pop_mutation = [toolbox.clone(individual) for individual in pop_cross]
        for ind in pop_mutation:
            toolbox.Mutation(individual=ind, pb=mutation_pb)
        # 评估新产生的个体
        pop_children = pop_mutation
        popEvaluate(population=pop_children)
        # toolbox.Evaluate(population=pop_children)
        # 选择
        # 一般用轮盘赌或锦标赛，但是此问题fitness会为负，故选用锦标赛，tournsize个体选1，跑k次
        pop_select = deap.tools.selTournament(pop_children, k=select_num, tournsize=ts, fit_attr='fitness')
        # 采样
        # 选择种群中的最佳点作为下一个CAE仿真采样点
        # 大坑注意：Best是选最小，Worst是选最大(因为初始化时优化问题设置的是最小为优)
        sample = sampleSelect(pop, len(Sample_Train))  # 选择下一个采样点
        Sample_Points = np.vstack((Sample_Points, sample))
        Sample_Train = np.vstack((Sample_Train, sample))
        # 训练
        # 依据新增加的采样点训练XGB代理模型
        modelTrain(Sample_Train, Data, parameters, num_boost_round)
        # 筛选繁育出的个体，生成新的种群
        pop_f = tools.selBest(pop, k=pop_init - select_num)
        pop = pop_f + pop_select
        # print(generation)
    rank = resultDisp(Sample_Points, Sample_Init, generations, Data)
    return rank

if __name__ == '__main__':
    if LHS:
        k_p = lhs(pop_init)
    path = '3output_5d.csv'
    Data = np.array(pd.read_csv(path, header=None))
    # 定义单目标/多目标优化问题，weights表示每个目标的权重，weight取1表示最大化，-1表示最小化
    creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
    # 创建Individual类，继承list
    creator.create('Individual', list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    # 定义个体生成函数
    # 函数别名为Individual，真实调用的是tool.initRepeat函数，该函数的传参为container, func, n，container用的是Individual类，func生成个体调用的是之前定义的randomX函数
    # toolbox.register('Individual', tools.initRepeat, creator.Individual, randomX, n=1)
    # Latin Hypercube Sampling
    toolbox.register('Individual', tools.initRepeat, creator.Individual, pop_lhs_init, n=1)
    # 定义种群生成函数Population
    toolbox.register('Population', tools.initRepeat, list, toolbox.Individual)
    # 定义突变函数
    toolbox.register("Mutation", mutation)

    # 初始化采样点并训练代理模型
    Sample_Init = lhs(sample_init)  # 初始化的训练采样点
    Sample_Train = Sample_Init  # 用于训练模型的种群定义为Sample_Train
    Sample_Points = np.empty((0, 5))
    modelTrain(Sample_Train, Data, parameters, num_boost_round)  # 每次迭代所选择的采样点

    Rank = []
    for i in range(10):
        rank = iterate(Sample_Train, Sample_Points, Data, num_boost_round)
        Rank.append(rank)
        print(i)
    print(Rank)
    # print("Rank List for 10 cycles:", Rank)
    print("Mean value of rank:", np.mean(Rank))
    name = ['rank']
    test = pd.DataFrame(columns=name, data=Rank)  # 数据有三列，列名分别为one,two,three
    test.to_csv('./rank1.csv', encoding='gbk')
import warnings

import deap
import xgboost as xgb
import pandas as pd
from deap import base, creator, tools
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import lhsmdu


def randomX():
  prob = [0.2, 0.2, 0.2, 0.2, 0.2]
  # 按概率采样，size参数表示采样几次，生成多大的array，replace=True表示可以对一个点重复采样，p表示每个点的概率
  a = np.random.choice([200, 230, 260, 290, 320], replace=True, p=prob)
  b = np.random.choice([20, 25, 30, 35, 40], replace=True, p=prob)
  c = np.random.choice([200, 225, 250, 275, 300], replace=True, p=prob)
  d = np.random.choice([550, 650, 750, 850, 950], replace=True, p=prob)
  e = np.random.choice([8, 9, 10, 11, 12], replace=True, p=prob)
  individual = [a, b, c, d, e]
  return individual

# Latin Hypercube Sampling
cnt = 0
def lhs_init():
    global cnt
    global POP_init
    num_dimensions = 5
    num_samples = POP_init
    k = np.array(lhsmdu.sample(numDimensions=num_dimensions,
                               numSamples=num_samples,
                               randomSeed=666))
    scale = np.array([30, 5, 25, 100, 1]).reshape(-1, 1)
    offset = np.array([200, 20, 200, 550, 8]).reshape(-1, 1)
    k = (np.multiply(scale, np.rint((num_dimensions - 1) * k)) + offset).T
    res = k[cnt, :].tolist()
    cnt = cnt + 1
    return res

# 定义个体评价函数
def evaluate(model, individual):
    dtest = xgb.DMatrix(np.array(individual))
    pred = model.predict(dtest)
    return pred, #注意这个逗号，即使是单变量优化问题，也需要返回tuple

# XGBoost模型训练与保存
def modelTrain(population):
    # 传参：种群population
    # 训练模型并保存模型
    path = 'full_scan_5d.csv'
    D = np.array(pd.read_csv(path))
    length = len(D[1])
    Train = np.empty((0, 6))
    for ind in population:
        d = D
        for i in range(0, 5):
            d = d[d[:, i] == ind[0][i], :]
        Train = np.vstack((Train, d))
    # if best != 0:
    #     for ind in best:
    #         d = D
    #         for i in range(0, 5):
    #             d = d[d[:, i] == ind[0][i], :]
    #         Train = np.vstack((Train, d))

    X = Train[:, 0:length - 1]
    y = Train[:, length - 1:length]*1000
    y = -np.abs(y)
    y = y.ravel()

    # XGBoost代理模型 作为evaluate function
    parameters = {'seed': 100, 'nthread': 4, 'gamma': 0, 'lambda': 0.1,
                  'max_depth': 10, 'eta': 0.1, 'objective': 'reg:squarederror'}
    dtrain = xgb.DMatrix(X, label=y)
    model = xgb.train(parameters, dtrain, num_boost_round=200)
    model.save_model('xgb.model')

def truth(population):
    # 传参：种群population
    # 训练模型并保存模型
    path = 'full_scan_5d.csv'
    D = np.array(pd.read_csv(path))
    length = len(D[1])
    Data = np.empty((0, 6))
    for ind in population:
        d = D
        for i in range(0, 5):
            d = d[d[:, i] == ind[0][i], :]
        Data = np.vstack((Data, d))

    X = Data[:, 0:length - 1]
    y = Data[:, length - 1:length]*1000
    y = np.abs(y)
    return y

# 评估种群内每个个体的fitness
def popEvaluate(population):
    model = xgb.Booster()
    model.load_model('xgb.model')
    for individual in population:
        individual.fitness.values = evaluate(model, individual)

# 个体基因突变函数
def mutation(individual, pb):
    prob = [0.2, 0.2, 0.2, 0.2, 0.2]
    a = np.random.choice([200, 230, 260, 290, 320], replace=True, p=prob)
    b = np.random.choice([20, 25, 30, 35, 40], replace=True, p=prob)
    c = np.random.choice([200, 225, 250, 275, 300], replace=True, p=prob)
    d = np.random.choice([550, 650, 750, 850, 950], replace=True, p=prob)
    e = np.random.choice([8, 9, 10, 11, 12], replace=True, p=prob)

    t0 = np.random.choice([0, 1], replace=False, p=[1-pb, pb])
    t1 = np.random.choice([0, 1], replace=False, p=[1-pb, pb])
    t2 = np.random.choice([0, 1], replace=False, p=[1-pb, pb])
    t3 = np.random.choice([0, 1], replace=False, p=[1-pb, pb])
    t4 = np.random.choice([0, 1], replace=False, p=[1-pb, pb])
    if t0 == 1:
        individual[0][0] = a
    if t1 == 1:
        individual[0][1] = b
    if t2 == 1:
        individual[0][2] = c
    if t3 == 1:
        individual[0][3] = d
    if t4 == 1:
        individual[0][4] = e
    return individual

def sampleSelect(candidate_population, train_population):
    # 从种群中选择下一个采样点
    # 和贝叶斯优化的采集函数具有相同功能，平衡贪心和探索
    l = len(train_population)
    # 在模型较小时选择一定程度不相信模型预测，模型较大时相信
    if l < 50:
        samples = tools.selBest(candidate_population, k=8, fit_attr="fitness")
        sample = tools.selRandom(samples, 1)
    elif l < 80:
        samples = tools.selBest(candidate_population, k=3, fit_attr="fitness")
        sample = tools.selRandom(samples, 1)
    else:
        sample = tools.selBest(candidate_population, k=1, fit_attr="fitness")
    return sample


def ranking(result):
    benchmark = [0, 0.0168, 0.0239, 0.057, 0.059, 0.0689, 0.0729, 0.0809, 0.0858, 0.115013, 0.130164, 0.142919, 0.147871, 0.166071, 0.167726, 0.180223, 0.195995, 0.196783, 0.217258, 0.239674, 0.255737, 0.270209, 0.289178, 0.298477, 0.309586, 0.322446, 0.337163, 0.339115, 0.349386, 0.373878, 0.374401, 0.379241, 0.385782, 0.392128, 0.413967, 0.438757, 0.453112, 0.466661, 0.474914, 0.477166, 0.485028, 0.517961, 0.519821, 0.538555, 0.545344, 0.566704, 0.567065, 0.57607, 0.580767, 0.58662, 0.588015, 0.59527, 0.598695, 0.61892, 0.629343, 0.630311, 0.631692, 0.655743, 0.656904, 0.657748, 0.662486, 0.665945, 0.668199, 0.674794, 0.697299, 0.697769, 0.70639, 0.712995, 0.732316, 0.733463, 0.736107, 0.736155, 0.768301, 0.780585, 0.811329, 0.812936, 0.821925, 0.887417, 0.895446, 0.928705, 0.938831, 0.959684, 0.964033, 0.974505, 1.016448, 1.02202, 1.025155, 1.042937, 1.044545, 1.058703, 1.08692, 1.10015, 1.103519, 1.110195, 1.111611, 1.143553, 1.149645, 1.158169, 1.16672, 1.196585]
    new_arr = np.sort(np.append(np.array(benchmark), result))
    index = np.where(new_arr == result)
    return np.min(index)

if __name__ == '__main__':
    # 超参数定义

    # 定义单目标/多目标优化问题，weights表示每个目标的权重，weight取1表示最大化，-1表示最小化
    creator.create('FitnessMin', base.Fitness, weights=(1.0,))
    # 创建Individual类，继承list
    creator.create('Individual', list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    # 定义个体生成函数
    # 函数别名为Individual，真实调用的是tool.initRepeat函数，该函数的传参为container, func, n，container用的是Individual类，func生成个体调用的是之前定义的randomX函数
    # toolbox.register('Individual', tools.initRepeat, creator.Individual, randomX, n=1)
    # Latin Hypercube Sampling
    toolbox.register('Individual', tools.initRepeat, creator.Individual, lhs_init, n=1)
    # 定义种群生成函数Population
    toolbox.register('Population', tools.initRepeat, list, toolbox.Individual)
    # 定义种群中个体fitness评估执行函数
    toolbox.register("Evaluate", popEvaluate)
    # 定义突变函数
    toolbox.register("Mutation", mutation)

    # 初始化种群
    POP_init = 30
    pop = toolbox.Population(n=POP_init)
    pop0 = pop
    modelTrain(pop0)

    BestGeneration = []
    POP_Train = []
    POP_Train += pop0
    for generation in range(0, 70):
        # 评估
        # 种群个体fitness评估
        toolbox.Evaluate(population=pop)

        # 选择
        # 一般用轮盘赌或锦标赛，但是此问题fitness会为负，故选用锦标赛
        # tournsize个体选1，跑k次
        k = 10
        if len(pop) < 20:
            k = round(len(pop) * 2)
        elif len(pop) < 50:
            k = round(len(pop) * 1.5)
        elif len(pop) < 100:
            k = round(len(pop) * 1.2)
        pop_select = deap.tools.selTournament(pop, k=k, tournsize=2, fit_attr='fitness')

        # 均匀交叉
        pop_cross = [toolbox.clone(individual) for individual in pop_select]
        for i in range(0, len(pop_cross)-len(pop_cross)%2, 2):
            deap.tools.cxUniform(ind1=pop_cross[i], ind2=pop_cross[i+1], indpb=0.6)
            # tools.cxTwoPoint(a, b)
            # tools.cxOnePoint(a, b)

        # 个体突变
        pop_mutation = [toolbox.clone(individual) for individual in pop_cross]
        for ind in pop_mutation:
            toolbox.Mutation(individual=ind, pb=0.2)

        # 评估新产生的个体
        pop_children = pop_mutation
        toolbox.Evaluate(population=pop_children)

        # 组合
        # pop = pop + pop_children
        pop = pop_children
        pop = tools.selBest(pop, k=round(len(pop)*0.8), fit_attr="fitness")

        # 采样
        # 选择种群中的最佳点作为下一个CAE仿真采样点
        # 大坑注意：Best是选最大，Worst是选最小！！！！
        sample = sampleSelect(pop, POP_Train)
        BestGeneration.append(sample[0])
        POP_Train.append(sample[0])
        # 训练
        # 依据新增加的采样点训练XGB代理模型
        modelTrain(POP_Train)

    SamplePoints = truth(BestGeneration)

    Min = []
    min0 = np.min(truth(pop0))
    P = truth(POP_Train)
    min = min0  # 截止目前的最小值
    for i in range(30, len(P)):
        if P[i] < min:
            Min.append(float(P[i]))
            min = float(P[i])
        else:
            Min.append(min)
    Generation = range(0, 70)
    plt.figure(figsize=(20, 10))
    plt.xlabel('Generations')
    plt.ylabel('WrapRate')
    plt.legend('EA-XGB', loc='lower right')
    plt.title('EA&XGB - Generations vs WrapRate')
    plt.plot(Generation, Min, 'g-', lw=2)
    plt.show()
    optimum = np.min(Min)
    print("Minimum Wrap Rate is: ", optimum)
    print("rank=" + str(ranking(optimum)))


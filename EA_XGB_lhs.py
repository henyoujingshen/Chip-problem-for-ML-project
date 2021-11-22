import warnings

import deap
import xgboost as xgb
import pandas as pd
from deap import base, creator, tools
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

import lhsmdu

cnt = 0

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
def lhs_init():
    global cnt
    num_dimensions = 5
    num_samples = 10
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
    return pred,  # 注意这个逗号，即使是单变量优化问题，也需要返回tuple


# XGBoost模型训练与保存
def modelTrain(population, best):
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
    if best != 0:
        for ind in best:
            d = D
            for i in range(0, 5):
                d = d[d[:, i] == ind[0][i], :]
            Train = np.vstack((Train, d))

    X = Train[:, 0:length - 1]
    y = Train[:, length - 1:length]
    y = -np.abs(y).ravel()

    # XGBoost代理模型 作为evaluate function
    parameters = {'seed': 666, 'nthread': 4, 'gamma': 0, 'lambda': 0.1,
                  'max_depth': 5, 'eta': 0.9, 'objective': 'reg:squarederror'}
    # 参数优化器
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
    y = Data[:, length - 1:length]
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

    t0 = np.random.choice([0, 1], replace=False, p=[1 - pb, pb])
    t1 = np.random.choice([0, 1], replace=False, p=[1 - pb, pb])
    t2 = np.random.choice([0, 1], replace=False, p=[1 - pb, pb])
    t3 = np.random.choice([0, 1], replace=False, p=[1 - pb, pb])
    t4 = np.random.choice([0, 1], replace=False, p=[1 - pb, pb])
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
    POP_init = 10
    pop = toolbox.Population(POP_init)
    pop0 = pop
    modelTrain(pop0, 0)

    BestGeneration = []
    for generation in range(0, 100):
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
        for i in range(0, len(pop_cross) - len(pop_cross) % 2, 2):
            deap.tools.cxUniform(ind1=pop_cross[i], ind2=pop_cross[i + 1], indpb=0.6)
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
        pop = tools.selBest(pop, k=round(len(pop) * 0.8), fit_attr="fitness")

        # 采样
        # 选择种群中的最佳点作为下一个CAE仿真采样点
        # 大坑注意：Best是选最大，Worst是选最小！！！！
        best = tools.selBest(pop, k=1, fit_attr="fitness")
        BestGeneration.append(best[0])
        # 训练
        # 依据新种群训练XGB代理模型，并使用代理模型评估个体fitness
        modelTrain(pop0, best)
    Generation = range(0, 100)
    Optimal = truth(BestGeneration)

    plt.figure(figsize=(20, 10))
    plt.xlabel('Generations')
    plt.ylabel('WrapRate')
    plt.legend('EA-XGB', loc='lower right')
    plt.title('EA&XGB - Generations vs WrapRate')
    plt.plot(Generation, Optimal, 'g-', lw=2)
    plt.show()

    print(1000 * np.min(Optimal))

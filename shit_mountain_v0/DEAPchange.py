import random
import numpy as np


def cxBlend(ind1, ind2, alpha, param):

    xl = param['range'][0]
    xu = param['range'][1]
    for i, (x1, x2) in enumerate(zip(ind1[0], ind2[0])):
        gamma = (1. + 2. * alpha) * random.random() - alpha
        ind1[0][i] = np.clip((1. - gamma) * x1 + gamma * x2, a_min=xl, a_max=xu)
        ind2[0][i] = np.clip(gamma * x1 + (1. - gamma) * x2, a_min=xl, a_max=xu)

    return ind1, ind2


def mutGaussian(individual, param):

    mu = param['mu']
    sigma = param['sigma']
    xl = param['range'][0]
    xu = param['range'][1]
    dim = param['dimension']
    indpb = param['mutaion_prob']
    rand = np.random.rand(dim) - indpb
    rint = np.ceil(rand)
    ind = np.clip(np.random.normal(mu, sigma, dim), a_min=xl, a_max=xu) * rint
    individual[0] = (np.array(individual[0]) + ind).tolist()

    return individual


def mutPoly(individual, param):
    """
    定义个体突变函数，在toolbox中注册为Mutation函数
    polynomial mutation
    :param individual: 进行突变操作的个体
    :return: 完成突变操作后的个体
    """

    eta = param['eta']
    xl = param['xl']
    xu = param['xu']
    dim = param['dim']

    xp = np.array(individual[0])  # x parents
    # rand = np.random.rand(dim)
    rand = np.random.rand(1) * np.ones(dim)
    prob = np.random.rand(1)

    if prob <= param['mutation_prob']:
        sigma = np.zeros(2 * dim)
        delta_1 = (xp - xl) / (xu - xl)
        delta_2 = (xu - xp) / (xu - xl)

        val_1 = 2.0 * rand + (1.0 - 2.0 * rand) * np.power(1 - delta_1, eta + 1)
        # sigma[:dim] = np.power(val_1, (1.0 / (eta + 1))) - 1.0
        sigma[:dim] = np.sign(val_1) * np.power(np.abs(val_1), 1.0 / (eta + 1)) - 1.0
        val_2 = 2.0 * (1.0 - rand) + (2.0 * rand - 1.0) * np.power(1 - delta_2, eta + 1)
        # sigma[dim:] = 1.0 - np.power(val_2, (1.0 / (eta + 1)))
        sigma[dim:] = 1.0 - np.sign(val_2) * np.power(np.abs(val_2), 1.0 / (eta + 1))

        index = np.rint(rand) + np.arange(0, 2 * dim, 2)
        sigma = sigma[index.astype(int)]

        xo = xp + sigma * (xu - xl)

        xo = np.clip(xo, a_min=xl, a_max=xu)
        individual[0] = xo.tolist()
        if np.isnan(xo).any():
            print("failed")
    return individual
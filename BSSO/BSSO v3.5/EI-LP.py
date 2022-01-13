import GPyOpt

# Base model settings
import numpy as np
from smt.sampling_methods import LHS

# Test problem settings
problem_param = {
    # 'name': 'rosenbrock',
    # 'dimension': 10,
    # 'range': [-1, 1],
    # 'global_min_pos': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    # 'min': 0,

    # 'name': 'rastrigin',
    # 'dimension': 20,
    # 'range': [-5, 5],
    # 'global_min_pos': [0, ... 0],
    # 'min': 0,

    # 'name': 'griewank',
    # 'dimension': 20,
    # 'range': [-600, 600],
    # 'global_min_pos': [0, ... 0],
    # 'min': 0,

    # 'name': 'ellipsoid',
    # 'dimension': 20,
    # 'range': [-5.12, 5.12],
    # 'global_min_pos': [0, ... 0],
    # 'min': 0,

    # 'name': 'ackley',
    # 'dimension': 10,
    # 'range': [-32, 32],
    # 'global_min_pos': [0, ... 0],
    # 'min': 0,

    # 'name': 'shcb',
    # 'dimension': 2,
    # 'range': [[-3, -2], [3, 2]],
    # 'min': -1.0316,
    # 'global_min_pos': [0.0898, -0.7126],
    # or [-0.0898, 0.7126]

    # 'name': 'goldstein_price',
    # 'dimension': 2,
    # 'range': [-2, 2],
    # 'min': 3,
    # 'global_min_pos': [0, -1],

    # 'name': 'hartman3',
    # 'dimension': 3,
    # 'range': [0, 1],
    # 'min': -3.86278,
    # 'global_min_pos': [0.114614, 0.555649, 0.852547],

    # 'name': 'alpine',
    # 'dimension': 5,
    # 'range': [-10, 10],
    # 'min': 0,
    # 'global_min_pos': [0, 0, 0, 0, 0],

    # 'name': 'hartman6',
    # 'dimension': 6,
    # 'range': [0, 1],
    # 'min': -3.32237,
    # 'global_min_pos': [0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573],

    # 'name': 'easom',
    # 'dimension': 2,
    # 'range': [-10, 10],
    # 'min': -1,
    # 'global_min_pos': [np.pi, np.pi],

    # 'name': 'shekel',
    # 'dimension': 4,
    # 'range': [0, 10],
    # 'min': -10.1532,
    # 'global_min_pos': [4, 4, 4, 4],

    'name': 'eggholder',
    'dimension': 2,
    'range': [-512, 512],
    'min': -959.6407,
    'global_min_pos': [512, 404.2319],

    # 'name': 'branin',
    # 'dimension': 2,
    # 'range': [[-5, 0], [10, 15]],
    # 'min': 0.397887,
    # 'global_min_pos': [9.42478, 2.475],
    # or [-np.pi, 12.275], [np.pi, 2.275]

    # 'name': 'chip',
    # 'dimension': 5,
    # 'range': [[0.55, 0.2, 0.2, 0.02, 8], [0.95, 0.3, 0.32, 0.04, 12]],
    # 'min': 0,

}

Optimization_param = {
    'sample_init_num': 20,
    'generations_num': 45,
    # 'sample_init_num': 20,
    # 'generations_num': 45*4,
    'runs_num': 10,
    'init_seed': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'fix_seed': int(np.random.rand(1)*1e3),
    'current_generation': 0,
}

# Latin Hypercube Sampling
def latin_hypercube_sampling(num_samples):
    if type(problem_param['range'][0]) != int:
        X_min = problem_param['range'][0]
        X_max = problem_param['range'][1]
        X_range = []
        for i in range(problem_param['dimension']):
            X_range.append([X_min[i], X_max[i]])
        X_range = np.array(X_range)
        sampling = LHS(xlimits=X_range, criterion='cm', random_state=Optimization_param['init_seed'][run])
        x = sampling(num_samples)
    else:
        x_lim = np.array(problem_param['range']).reshape(-1, 1)
        x_lim = x_lim.repeat(problem_param['dimension'], axis=1).T
        sampling = LHS(xlimits=x_lim, criterion='cm', random_state=Optimization_param['init_seed'][run])
        x = sampling(num_samples)
    return x

def evaluateFunc(sample_array):
    """
    Expensive optimization of test functions and chip packaging problem
    :param sample_array: sample points; 2D numpy array
    :return X: the same as sample_array
    :return y: corresponding values
    """
    X = sample_array
    result = None
    if problem_param['name'] == 'rosenbrock':
        result = np.sum(100 * np.square(X[:, 1:] - np.square(X[:, :-1])) + np.square(X[:, -1] - 1).reshape(-1, 1),
                        axis=1)
        if problem_param['dimension'] == 10:
            problem_param['column_name'] = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'y']
        elif problem_param['dimension'] == 20:
            problem_param['column_name'] = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10',
                                            'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'y']
        elif problem_param['dimension'] == 2:
            problem_param['column_name'] = ['x1', 'x2', 'y']
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
        elif problem_param['dimension'] == 2:
            problem_param['column_name'] = ['x1', 'x2', 'y']
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
        if problem_param['dimension'] == 2:
            x1, x2 = X[:, 0], X[:, 1]
            result = np.abs(x1 * np.sin(x1) + 0.1 * x1) + np.abs(x2 * np.sin(x2) + 0.1 * x2)
            problem_param['column_name'] = ['x1', 'x2', 'y']
        elif problem_param['dimension'] == 5:
            result = np.sum(np.abs(X * np.sin(X) + 0.1 * X), axis=1)
            problem_param['column_name'] = ['x1', 'x2', 'x3', 'x4', 'x5', 'y']
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
    elif problem_param['name'] == 'eggholder':
        x1, x2 = X[:, 0], X[:, 1]
        result = -(x2 + 47) * np.sin(np.sqrt(np.abs(x2 + x1 / 2 + 47))) - x1 * np.sin(np.sqrt(np.abs(x1 - x2 - 47)))
        problem_param['column_name'] = ['x1', 'x2', 'y']
    elif problem_param['name'] == 'branin':
        x1, x2 = X[:, 0], X[:, 1]
        a = 1
        b = 5.1 / (4 * np.square(np.pi))
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)
        result = a * np.square(x2 - b * np.square(x1) + c * x1 - r) + s * (1 - t) * np.cos(x1) + s
        problem_param['column_name'] = ['x1', 'x2', 'y']
    else:
        print("New test function")
    y = result.reshape(-1, 1)
    return y



for run in range(Optimization_param['runs_num']):
    # --- Objective function
    dim = 2
    objective_true = GPyOpt.objective_examples.experiments2d.eggholder()               # true function
    # objective_noisy = GPyOpt.objective_examples.experimentsNd.ackley(input_dim=dim, sd = 0.1)         # noisy version
    bounds = objective_true.bounds
    if dim == 3:
        domain = [{'name': 'var_1', 'type': 'continuous', 'domain': bounds[0]}, ## use default bounds
              {'name': 'var_2', 'type': 'continuous', 'domain': bounds[1]},
              {'name': 'var_3', 'type': 'continuous', 'domain': bounds[2]}]
    elif dim == 2:
        domain = [{'name': 'var_1', 'type': 'continuous', 'domain': bounds[0]},  ## use default bounds
                  {'name': 'var_2', 'type': 'continuous', 'domain': bounds[1]},]
    elif dim == 4:
        domain = [{'name': 'var_1', 'type': 'continuous', 'domain': bounds[0]},  ## use default bounds
                  {'name': 'var_2', 'type': 'continuous', 'domain': bounds[1]},
                  {'name': 'var_3', 'type': 'continuous', 'domain': bounds[2]},
                  {'name': 'var_3', 'type': 'continuous', 'domain': bounds[3]}]
    elif dim == 6:
        domain = [{'name': 'var_1', 'type': 'continuous', 'domain': bounds[0]},  ## use default bounds
                  {'name': 'var_2', 'type': 'continuous', 'domain': bounds[1]},
                  {'name': 'var_3', 'type': 'continuous', 'domain': bounds[2]},
                  {'name': 'var_3', 'type': 'continuous', 'domain': bounds[3]},
                  {'name': 'var_3', 'type': 'continuous', 'domain': bounds[4]},
                  {'name': 'var_3', 'type': 'continuous', 'domain': bounds[5]}]

    # objective_true.plot()
    batch_size = 1
    num_cores = 6
    from numpy.random import seed
    seed(123)
    X = latin_hypercube_sampling(Optimization_param['sample_init_num'])
    Y = evaluateFunc(X)

    BO_demo_parallel = GPyOpt.methods.BayesianOptimization(f=objective_true.f,
                                                           X = X,
                                                           Y = Y,
                                                domain = domain,
                                                acquisition_type = 'EI',
                                                normalize_Y = True,
                                                # evaluator_type = 'local_penalization',
                                                batch_size = batch_size,
                                                num_cores = num_cores,
                                                acquisition_jitter = 0)
    # --- Run the optimization for 10 iterations
    max_iter = 45
    BO_demo_parallel.run_optimization(max_iter)
    print(BO_demo_parallel.fx_opt)
    # BO_demo_parallel.plot_acquisition()
    BO_demo_parallel.plot_convergence()
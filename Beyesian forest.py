import numpy as np
import matplotlib.pyplot as plt
import xlrd
from skopt.plots import plot_convergence


path='full_scan_5d.xls'  # 源数据的路径
wb = xlrd.open_workbook(path)
sh = wb.sheet_by_name("full_scan_5d")
nrows = sh.nrows  # 获取行数


def f(x:np.int8):
    D = [0.55,0.65, 0.75, 0.85, 0.95]
    C = [0.2, 0.225, 0.25, 0.275, 0.3]
    A = [0.2, 0.23, 0.26, 0.29, 0.32]
    B = [0.02, 0.025, 0.03, 0.035, 0.04]
    E = [8, 9, 10, 11, 12]
    a=A[x[0]]*1000
    b=B[x[1]]*1000
    c=C[x[2]]*1000
    d=D[x[3]]*1000
    e=E[x[4]]
    for i in range(nrows):
        if sh.cell_value(i, 0) == a and sh.cell_value(i, 1) == b and sh.cell_value(i, 2) == c and sh.cell_value(i,
                                                                                                                3) == d and sh.cell_value(i, 4) == e:
            m=abs(sh.cell_value(i, 5))*1000

    return m

from skopt import forest_minimize
#极端随机森林
res = forest_minimize(f, # the function to minimize
                  [(0, 4),(0, 4),(0, 4),(0, 4),(0, 4)],  # the bounds on each dimension of x
                  acq_func="EI",  # the acquisition function
                  n_calls=100,  # the number of evaluations of f
                  n_random_starts=5,  # the number of random initialization points
                  random_state=1234)  # the random seed

#"x^*=%.4f, f(x^*)=%.4f" % (res.x[0], res.fun)
print(res)
plot_convergence(res)
plt.show()

#普通随机森林
from sklearn.ensemble import RandomForestRegressor
res = forest_minimize(f,  # the function to minimize
RandomForestRegressor
                  [(0, 4),(0, 4),(0, 4),(0, 4),(0, 4)],  # the bounds on each dimension of x
                  acq_func="EI",  # the acquisition function
                  n_calls=100,  # the number of evaluations of f
                  n_random_starts=5,  # the number of random initialization points
                  random_state=1234)  # the random seed

#"x^*=%.4f, f(x^*)=%.4f" % (res.x[0], res.fun)
print(res)
plot_convergence(res)
plt.show()

#调参方法：数的个数，最大深度，。。。
res = forest_minimize(f,  # the function to minimize
                       RandomForestRegressor(n_estimators=100,min_samples_split=2, min_samples_leaf=1,  min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None)
                  [(0, 4),(0, 4),(0, 4),(0, 4),(0, 4)],  # the bounds on each dimension of x
                  acq_func="EI",  # the acquisition function
                  n_calls=100,  # the number of evaluations of f
                  n_random_starts=5,  # the number of random initialization points
                  random_state=1234)  # the random seed

#"x^*=%.4f, f(x^*)=%.4f" % (res.x[0], res.fun)
print(res)
plot_convergence(res)
plt.show()
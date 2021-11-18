import datetime
import time
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import plot_importance
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def predict_plot(y_test, y_pred):
    # 编号 预测值绘图
    plt.figure(figsize=(100, 40), facecolor='w')
    ln_x_test = range(len(y_test))
    plt.plot(ln_x_test, y_test, 'r-', lw=2, label='实际值')
    plt.plot(ln_x_test, y_pred, 'g-', lw=4, label='XGBoost模型')
    plt.xlabel('数据编码')
    plt.ylabel('翘曲率')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.title('芯片封装翘曲数据预测')
    plt.show()

def iter_plot(parameters):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=666)
    y_test = y_test.ravel()
    y_train = y_train.ravel()
    dtest = xgb.DMatrix(X_test, label=y_test)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    Error = []
    Time = []
    Iters = range(100, 500, 20)
    for i in Iters:
        num_boost_round = i
        start_time = time.perf_counter()
        model = xgb.train(parameters, dtrain, num_boost_round)
        y_pred = model.predict(dtest)
        end_time = time.perf_counter()
        compute_time = end_time - start_time
        error = mean_squared_error(y_test, y_pred)
        Error.append(error)
        Time.append(compute_time)

    plt.figure(figsize=(20, 10))
    plt.xlabel('Iterations')
    plt.ylabel('MSError')
    plt.legend('MSE', loc='lower right')
    plt.title('Iters vs MSE')
    plt.plot(Iters, Error, 'r-', lw=2)
    plt.figure(figsize=(20, 10))
    plt.xlabel('Iterations')
    plt.ylabel('CompTime')
    plt.legend('Time', loc='lower right')
    plt.title('Iters vs Time')
    plt.plot(Iters, Time, 'g-', lw=2)
    plt.show()

def trainSize_plot(parameters):
    Error = []
    Time = []
    TrainSize = range(10, 300)
    for i in TrainSize:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=i, test_size=20, random_state=666)
        y_test = y_test.ravel()
        y_train = y_train.ravel()
        dtest = xgb.DMatrix(X_test, label=y_test)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        num_boost_round = 200
        start_time = time.perf_counter()
        model = xgb.train(parameters, dtrain, num_boost_round)
        y_pred = model.predict(dtest)
        end_time = time.perf_counter()
        compute_time = end_time - start_time
        error = mean_absolute_percentage_error(y_test, y_pred)
        Error.append(error)
        Time.append(compute_time)

    plt.figure(figsize=(20, 10))
    plt.xlabel('DataSize')
    plt.ylabel('MAPError')
    plt.legend('MAPE', loc='lower right')
    plt.title('XGB - zDataSize vs MAPE')
    plt.plot(TrainSize, Error, 'r-', lw=2)
    plt.figure(figsize=(20, 10))
    plt.xlabel('DataSize')
    plt.ylabel('CompTime')
    plt.legend('Time', loc='lower right')
    plt.title('XGB - DataSize vs Time')
    plt.plot(TrainSize, Time, 'g-', lw=2)
    plt.show()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    path = 'full_scan_5d.csv'
    D = np.array(pd.read_csv(path))
    length = len(D[1])
    X = D[:, 0:length - 1]
    y = D[:, length - 1:length]

    parameters = {'seed': 100, 'nthread': 4, 'gamma': 0, 'lambda': 0.1,
                  'max_depth': 10, 'eta': 0.1, 'verbose_eval': False, 'objective': 'reg:linear'}

    trainSize_plot(parameters)
    # iter_plot(parameters, dtrain)
    # predict_plot(y_test, y_pred)

    # xgb.plot_importance(model)
    # xgb.plot_tree(model, num_trees=2)
    # plt.show()

    # 两特征三维绘图
    # x1 = X_test[:, 0]
    # x2 = X_test[:, 1]
    # fig1 = plt.figure(figsize=(8, 6))
    # ax1 = Axes3D(fig1)
    # # ax.plot_trisurf(x1, x2, np.abs(y1), alpha=0.3, cmap='winter')
    # ax1.plot_trisurf(x1, x2, np.abs(y_pred), alpha=0.3, cmap='winter')
    # fig2 = plt.figure(figsize=(8, 6))
    # ax2 = Axes3D(fig2)
    # ax2.plot_trisurf(x1, x2, np.abs(y_test), alpha=0.3, cmap='winter')
    # plt.show()




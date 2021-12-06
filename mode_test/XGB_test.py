import datetime
import time
import warnings

import lhsmdu
import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import plot_importance
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_squared_log_error
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def lhs(num):
    num_dimensions = 5
    num_samples = num
    k = np.array(lhsmdu.sample(numDimensions=num_dimensions,
                               numSamples=num_samples,
                               randomSeed=666))
    index = np.rint((num_dimensions - 1) * k).T
    index = np.multiply(np.array([5 ** 4, 5 ** 3, 5 ** 2, 5 ** 1, 5 ** 0]), index)
    index = np.sum(index, axis=1)
    scale = np.array([30, 5, 25, 100, 1]).reshape(-1, 1)
    offset = np.array([200, 20, 200, 550, 8]).reshape(-1, 1)
    k = (np.multiply(scale, np.rint((num_dimensions - 1) * k)) + offset).T
    return index, k

def predict_plot(y_test, y_pred):
    # 编号 预测值绘图
    plt.figure(figsize=(15, 10), facecolor='w')
    ln_x_test = range(len(y_test))
    plt.plot(ln_x_test, y_test, 'r-', lw=2, label='True Value')
    plt.plot(ln_x_test, y_pred, 'g-', lw=2, label='XGBoost Prediction')
    plt.xlabel('Index')
    plt.ylabel('Wrap Rate')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.title('Wrap Rate Prediction')
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
        y_train = y_train * 1000
        y_test = y_test * 1000
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
    # warnings.filterwarnings("ignore")
    # with open('./3output_5d.csv') as f:
    #     full_scan = np.loadtxt(f, delimiter=',')
    #
    # objective = -3
    # X_full = full_scan[:, :5]
    # y_full = full_scan[:, objective]
    #
    parameters = {'seed': 100, 'nthread': -1, 'gamma': 0, 'lambda': 6,
                  'max_depth': 2, 'eta': 0.35, 'objective': 'reg:squarederror'}
    num_boost_round = 400

    #
    # lhs_index, lhs_k = lhs(100)
    # X, y = full_scan[lhs_index.astype(int), :5].astype(int), \
    #                    full_scan[lhs_index.astype(int), objective]
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20, random_state=0)
    #
    # y_train = (y_train * 1000).ravel()
    # y_test = (y_test * 1000).ravel()

    train_path = 'model_train_data.csv'
    test_path = 'model_test_data.csv'
    Train = np.array(pd.read_csv(train_path, header=None))
    Test = np.array(pd.read_csv(test_path, header=None))
    X_train = Train[:, 0:-1]
    y_train = Train[:, -1]
    X_test = Test[:, 0:-1]
    y_test = Test[:, -1]


    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    model = xgb.train(parameters, dtrain, num_boost_round)

    y_train_pred = model.predict(dtrain)
    y_test_pred = model.predict(dtest)

    y_train_pred = np.abs(y_train_pred)
    y_test_pred = np.abs(y_test_pred)

    predict_plot(np.abs(y_train), y_train_pred)
    predict_plot(np.abs(y_test), y_test_pred)

    error0 = mean_squared_error(np.abs(y_test), y_test_pred)
    print("mean squared error:", error0)
    error1 = mean_absolute_percentage_error(np.abs(y_test), y_test_pred)
    print("mean absolute percentage error:", error1)
    error2 = mean_squared_log_error(np.abs(y_test), y_test_pred)
    print("mean squared log error:", error2)


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




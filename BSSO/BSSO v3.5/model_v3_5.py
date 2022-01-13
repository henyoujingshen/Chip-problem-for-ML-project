import torch
import numpy as np
import pandas as pd
import xgboost as xgb

from matplotlib import pyplot as plt
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import pickle

from smt.sampling_methods import LHS
from smt.surrogate_models import KRG, RBF, QP

from sklearn.tree import DecisionTreeRegressor

from BSSO_v3_5 import enabled_model, problem_param, evaluateFunc, Optimization_param, plot_param

base_model = [0, 0, 0, 0, 0]

Valid_Pred = [np.empty((0, 1)), np.empty((0, 1)), np.empty((0, 1))]
Valid_Y = np.empty((0, 1))
Valid_Error = [np.empty((0, 1)), np.empty((0, 1)), np.empty((0, 1))]
Base_Round = pd.DataFrame(columns=['err_m1', 'err_m2', 'err_m3', 'y_valid'])
Meta_Round = pd.DataFrame(columns=['err_meta', 'meta_valid'])
base_model_weight = np.array([1 / 3, 1 / 3, 1 / 3])

DST_G = []
DST_ERROR = []
SL_G = []
SL_ERROR = []


def lhsMin(num_samples):
    if problem_param['name'] != 'chip':
        global_min_pos = problem_param['global_min_pos']
        if type(problem_param['range'][0]) != int:
            X_min = problem_param['range'][0]  # 每个维度x的最小值
            X_max = problem_param['range'][1]  # 每个维度x的最大值
            X_range = (np.array(X_max) - np.array(X_min)).tolist()  # 每个维度x从最小到最大的跨度
            X_test = []  # 每个维度最优点附近的搜索域
            for i in range(problem_param['dimension']):
                X_test.append([global_min_pos[i] - X_range[i] * 1/5, global_min_pos[i] + X_range[i] * 1/5])
                if X_test[i][0] < X_min[i]:
                    X_test[i][0] = X_min[i]
                if X_test[i][1] > X_max[i]:
                    X_test[i][1] = X_max[i]
            X_test = np.array(X_test)
            sampling = LHS(xlimits=X_test, criterion='cm', random_state=Optimization_param['fix_seed'])
            x = sampling(num_samples)
        else:
            x_min = problem_param['range'][0]  # 每个维度x的最小值
            x_max = problem_param['range'][1]  # 每个维度x的最大值
            X_min = np.array(x_min).repeat(problem_param['dimension'])
            X_max = np.array(x_max).repeat(problem_param['dimension'])

            X_range = (X_max - X_min).tolist()  # 每个维度x从最小到最大的跨度
            X_test = []  # 每个维度最优点附近的搜索域
            for i in range(problem_param['dimension']):
                X_test.append([global_min_pos[i] - X_range[i] * 1 / 5, global_min_pos[i] + X_range[i] * 1 / 5])
                if X_test[i][0] < X_min[i]:
                    X_test[i][0] = X_min[i]
                if X_test[i][1] > X_max[i]:
                    X_test[i][1] = X_max[i]
            X_test = np.array(X_test)
            sampling = LHS(xlimits=X_test, criterion='cm', random_state=Optimization_param['fix_seed'])
            x = sampling(num_samples)
    else:
        x = np.zeros((1, 1))
        print('Error: optimal of chip packaging design is unknown.')
    return x

if problem_param['name'] != 'chip':
    X_test = lhsMin(problem_param['dimension']*50)
    y_test = evaluateFunc(X_test)
else:
    X_test = None
    y_test = None

def paraInit():
    global Valid_Pred
    global Valid_Y
    global Valid_Error
    global Base_Round
    global Meta_Round
    global base_model_weight
    Valid_Pred = [np.empty((0, 1)), np.empty((0, 1)), np.empty((0, 1))]
    Valid_Y = np.empty((0, 1))
    Valid_Error = [np.empty((0, 1)), np.empty((0, 1)), np.empty((0, 1))]
    Base_Round = pd.DataFrame(columns=['err_m1', 'err_m2', 'err_m3', 'y_valid'])
    Meta_Round = pd.DataFrame(columns=['err_meta', 'meta_valid'])
    base_model_weight = np.array([1 / 3, 1 / 3, 1 / 3]).reshape(-1, 1)


def perfTest(X_train, y_train, X_valid, y_valid):
    # 不同base model的尝试
    # xgboost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    parameters = {'seed': Optimization_param['fix_seed'], 'nthread': -1, 'gamma': 0, 'lambda': 6,
                  'max_depth': 6, 'eta': 0.15, 'objective': 'reg:squarederror'}
    model_xgb = xgb.train(parameters, dtrain, num_boost_round=200)
    dvalid = xgb.DMatrix(X_valid)
    pred_xgb = model_xgb.predict(dvalid).reshape(-1, 1)
    error_xgb = mean_absolute_error(y_valid, pred_xgb).reshape(-1, 1)

    # poly2
    model_poly2 = Pipeline(steps=[
        ('preprocessor', PolynomialFeatures(degree=2, include_bias=False)),
        ('estimator', Ridge(alpha=1))
    ])
    model_poly2.fit(X_train, y_train)
    pred_poly2 = model_poly2.predict(X_valid).reshape(-1, 1)
    error_poly2 = mean_absolute_error(y_valid, pred_poly2).reshape(-1, 1)

    # poly3
    model_poly3 = Pipeline(steps=[
        ('preprocessor', PolynomialFeatures(degree=3, include_bias=False)),
        ('estimator', Ridge(alpha=1))
    ])
    model_poly3.fit(X_train, y_train)
    pred_poly3 = model_poly3.predict(X_valid).reshape(-1, 1)
    error_poly3 = mean_absolute_error(y_valid, pred_poly3).reshape(-1, 1)

    # knn
    model_knn = KNeighborsRegressor(n_neighbors=6)
    model_knn.fit(X_train, y_train)
    pred_knn = model_knn.predict(X_valid).reshape(-1, 1)
    error_knn = mean_absolute_error(y_valid, pred_knn).reshape(-1, 1)

    # decesion tree
    model_dt = DecisionTreeRegressor(random_state=Optimization_param['fix_seed'])
    model_dt.fit(X_train, y_train)
    pred_dt = model_dt.predict(X_valid).reshape(-1, 1)
    error_dt = mean_absolute_error(y_valid, pred_dt).reshape(-1, 1)

    # adaboost
    model_ada = AdaBoostRegressor(random_state=Optimization_param['fix_seed'], n_estimators=200, loss='exponential', learning_rate=0.5)
    model_ada.fit(X_train, y_train)
    pred_ada = model_ada.predict(X_valid).reshape(-1, 1)
    error_ada = mean_absolute_error(y_valid, pred_ada).reshape(-1, 1)

    # ramdom forest
    model_rf = RandomForestRegressor(max_depth=3, random_state=Optimization_param['fix_seed'], n_jobs=-1, ccp_alpha=0)
    model_rf.fit(X_train, y_train)
    pred_rf = model_rf.predict(X_valid).reshape(-1, 1)
    error_rf = mean_absolute_error(y_valid, pred_rf).reshape(-1, 1)

    Pred = np.hstack((pred_xgb, pred_ada, pred_rf, pred_poly3, pred_poly2, pred_dt, pred_knn))
    Error = np.hstack((error_xgb, error_ada, error_rf, error_poly3, error_poly2, error_dt, error_knn))

    # return correlation coefficient and p values
    r = np.corrcoef(Pred, rowvar=False)
    r = pd.DataFrame(data=r, columns=['xgb', 'ada', 'rf', 'poly3', 'poly2', 'dt', 'knn'],
                     index=['xgb', 'ada', 'rf', 'poly3', 'poly2', 'dt', 'knn'])
    return Pred, Error, r


def baseModel(Sample_X, Sample_y, parameters, num_boost_round, index):

    global base_model

    k = 5
    X = Sample_X
    y = Sample_y
    y = y.reshape(-1, 1)
    batch = (len(X) - index) / k + 1
    batch = int(batch)
    interval_start = batch * index
    interval_end = batch * (index + 1)
    X_valid = X[interval_start:interval_end, :]
    y_valid = y[interval_start:interval_end]
    X_train = np.delete(X, np.s_[interval_start:interval_end], axis=0)
    y_train = np.delete(y, np.s_[interval_start:interval_end], axis=0)

    # base model list
    model = []

    # GP base model
    if 'GP' in enabled_model:
        model_gp = KRG(theta0=[1e-2], nugget=1e-3, print_global=False)
        model_gp.set_training_values(X_train, y_train)
        model_gp.train()
        # with open('gp' + str(index) + '.model', 'wb') as file:
        #     pickle.dump(model_gp, file)
        model.append(model_gp)

    if 'RBF' in enabled_model:
        model_rbf = RBF(d0=5, print_global=False)
        model_rbf.set_training_values(X_train, y_train)
        model_rbf.train()
        a = model_rbf.predict_values(X_train)
        e = mean_absolute_error(y_train, a)
        # with open('rbf' + str(index) + '.model', 'wb') as file:
        #     pickle.dump(model_rbf, file)
        model.append(model_rbf)

    # poly base model
    if 'Polynomial' in enabled_model:
        model_poly = QP(print_global=False)
        model_poly.set_training_values(X_train, y_train)
        model_poly.train()
        # with open('poly' + str(index) + '.model', 'wb') as file:
        #     pickle.dump(model_poly, file)
        model.append(model_poly)


    # 在validation data上进行预测，同时返回预测值和真实值
    valid_pred = []
    valid_error = []
    for i in range(len(enabled_model)):
        if enabled_model[i] == 'GP':
            pred = model_gp.predict_values(X_valid).reshape(-1, 1)
            error = np.abs(y_valid - pred).reshape(-1, 1)
            valid_pred.append(pred)
            valid_error.append(error)
        elif enabled_model[i] == 'RBF':
            pred = model_rbf.predict_values(X_valid).reshape(-1, 1)
            error = np.abs(y_valid - pred).reshape(-1, 1)
            valid_pred.append(pred)
            valid_error.append(error)
        else:
            pred = model_poly.predict_values(X_valid).reshape(-1, 1)
            error = np.abs(y_valid - pred).reshape(-1, 1)
            valid_pred.append(pred)
            valid_error.append(error)

    base_round = np.hstack((valid_error[0], valid_error[1], valid_error[2], y_valid))
    base_round = pd.DataFrame(data=base_round, columns=['err_m1', 'err_m2', 'err_m3', 'y_valid'])

    base_model[index] = [model_gp, model_rbf, model_poly]

    return model, base_round, valid_pred, y_valid, valid_error


def metaModel(base_model_weight, Valid_Pred, Valid_Y):

    X_meta = Valid_Pred[0].reshape(-1, 1)
    y_meta = Valid_Y
    for i in range(1, len(Valid_Pred)):
        X_meta = np.hstack((X_meta, Valid_Pred[i]))
    Data = np.hstack((X_meta, y_meta))
    Data_train, Data_valid = train_test_split(Data, train_size=0.8)
    X_train = Data_train[:, :3]
    y_train = Data_train[:, -1].reshape(-1, 1)
    X_valid = Data_valid[:, :3]
    y_valid = Data_valid[:, -1].reshape(-1, 1)

    dmeta = xgb.DMatrix(X_train, label=y_train)
    meta_parameters = {'booster': 'gbtree', 'seed': Optimization_param['fix_seed'], 'nthread': -1, 'gamma': 0, 'lambda': 4,
                       'max_depth': 4, 'eta': 0.14, 'objective': 'reg:squarederror'}
    meta_num_boost_round = 270
    # meta_parameters = {'booster': 'gblinear', 'seed': 100, 'nthread': -1, 'eta': 0.65, 'objective': 'reg:squarederror', 'lambda': 0.05}
    # meta_num_boost_round = 600

    meta = xgb.train(meta_parameters, dmeta, num_boost_round=meta_num_boost_round)
    meta.save_model('meta_xgb.model')

    # 线性加权在validation集上的效果
    pred_linear = np.sum(X_valid, axis=1).reshape(-1, 1)
    error_linear = np.abs(pred_linear - y_valid)

    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    pred_valid = meta.predict(dvalid).reshape(-1, 1)
    pred_train = meta.predict(dmeta).reshape(-1, 1)

    error = np.abs(pred_valid - y_valid)
    meta_round = np.hstack((error, y_valid))
    meta_round = pd.DataFrame(data=meta_round, columns=['err_meta', 'meta_valid'])
    #
    if plot_param['train_test_error']:
        g_train = range(len(pred_train))
        plt.figure(figsize=(14.40, 9.00))
        plt.xlabel('Train Samples')
        plt.ylabel('Train Pred')
        plt.plot(g_train, pred_train, 'b-', lw=2)
        plt.plot(g_train, y_train, 'r-', lw=2)
        plt.show()

        g_valid = range(len(pred_valid))
        plt.figure(figsize=(14.40, 9.00))
        plt.xlabel('Valid Samples')
        plt.ylabel('Valid Pred')
        plt.plot(g_valid, pred_valid, 'b-', lw=2)
        plt.plot(g_valid, y_valid, 'r-', lw=2)
        plt.show()
    return meta, meta_round


def DSTweight(Valid_Y, Valid_Pred):
    DST_MASS = np.zeros((3, 3))

    Valid_Pred_DST = np.squeeze(np.array(Valid_Pred)).T

    # Step-01 construct DST matrix
    for i in range(3):
        DST_MASS[i, 0] = mean_absolute_error(Valid_Y, Valid_Pred_DST[:, i])
        DST_MASS[i, 1] = mean_absolute_percentage_error(Valid_Y, Valid_Pred_DST[:, i])
        DST_MASS[i, 2] = mean_squared_error(Valid_Y, Valid_Pred_DST[:, i])
    DST_MASS = 1 / DST_MASS

    # Step-02 normalize DST matrix
    DST_colsum = np.sum(DST_MASS, axis=0)
    DST_MASS_TRANSFORMED = DST_MASS / DST_colsum

    # Step-03 calculate the sum of row prod
    DST_rowprod = np.prod(DST_MASS_TRANSFORMED, axis=1)
    base_model_weight = DST_rowprod / np.sum(DST_rowprod)

    return base_model_weight


def naiveWeight(Valid_Error):
    base_model_weight = np.ones((3, 1))
    error_xgb = np.sum(Valid_Error[0])
    error_poly = np.sum(Valid_Error[1])
    error_knn = np.sum(Valid_Error[2])
    error_model = [error_xgb, error_poly, error_knn]
    base_model_weight_coeff = []
    for i in range(3):
        base_model_weight_coeff.append((np.max(error_model) + np.min(error_model) - error_model[i]))
    for i in range(3):
        base_model_weight[i] = base_model_weight_coeff[i] / np.sum(base_model_weight_coeff)

    return base_model_weight


def modelTrain(Sample_X, Sample_y, parameters, num_boost_round, generation):
    """
    k-fold要求初始采样点是几十一个
    根据训练种群，训练XGBoost代理模型，同时将模型保存为xgb.model文件
    :param generation: 正在进行的迭代次数
    :return: /
    """
    global Valid_Pred
    global Valid_Y
    global Valid_Error
    global Base_Round
    global Meta_Round
    global base_model_weight

    k = 5  # k-fold
    index = generation % k
    meta_model = None

    if index == 4:
        base_model, base_round, valid_pred, y_valid, valid_error = baseModel(Sample_X, Sample_y, parameters, num_boost_round, index)
        Base_Round = pd.concat([Base_Round, base_round], axis=0)
        # 每五轮的矩阵拼接
        Valid_Y = np.vstack((Valid_Y, y_valid))
        for i in range(3):
            Valid_Pred[i] = np.vstack((Valid_Pred[i], valid_pred[i]))
            Valid_Error[i] = np.vstack((Valid_Error[i], valid_error[i]))


        base_model_weight = DSTweight(Valid_Y, Valid_Pred)

        meta_model, meta_round = metaModel(base_model_weight, Valid_Pred, Valid_Y)
        Meta_Round = pd.concat([Meta_Round, meta_round], axis=0)
        Valid_Pred = [np.empty((0, 1)), np.empty((0, 1)), np.empty((0, 1))]
        Valid_Y = np.empty((0, 1))
        Valid_Error = [np.empty((0, 1)), np.empty((0, 1)), np.empty((0, 1))]
        print(str(len(Sample_X)) + str(base_model_weight))

        # test dataset error
        base_model.append(meta_model)
        second_layer_error = errorTest(base_model)
        SL_ERROR.append(second_layer_error)
        SL_G.append(generation)

    else:
        base_model, base_round, valid_pred, y_valid, valid_error = baseModel(Sample_X, Sample_y, parameters, num_boost_round, index)
        Base_Round = pd.concat([Base_Round, base_round], axis=0)
        # 每五轮的矩阵拼接
        Valid_Y = np.vstack((Valid_Y, y_valid))
        for i in range(3):
            Valid_Pred[i] = np.vstack((Valid_Pred[i], valid_pred[i]))
            Valid_Error[i] = np.vstack((Valid_Error[i], valid_error[i]))
        base_model_weight = DSTweight(Valid_Y, Valid_Pred)
        print(str(len(Sample_X)) + str(base_model_weight))

        # test dataset error
        dst_error = errorTest(base_model)
        DST_ERROR.append(dst_error)
        DST_G.append(generation)

    if generation == Optimization_param['generations_num'] - 1 and plot_param['error_plot']:

        plt.figure(figsize=(14.40, 9.00))
        plt.xlabel('Generations')
        plt.ylabel('Test Error')
        plt.legend("Select Points", loc='lower right')
        plt.title('Generations vs Test Error')
        plt.scatter(DST_G, DST_ERROR, alpha=1)
        plt.scatter(SL_G, SL_ERROR, alpha=1, s=80, c='r')
        plt.plot(DST_G, DST_ERROR, 'b-', lw=2)
        plt.plot(SL_G, SL_ERROR, 'r-', lw=2)
        plt.show()

        DST_G.clear()
        SL_G.clear()
        DST_ERROR.clear()
        SL_ERROR.clear()

    return base_model, meta_model, index, base_model_weight


def errorTest(model):
    if len(model) == 3:
        if problem_param['name'] != 'chip':
            gp_test = model[0].predict_values(X_test).reshape(-1, 1)
            rbf_test = model[1].predict_values(X_test).reshape(-1, 1)
            poly_test = model[2].predict_values(X_test).reshape(-1, 1)
            Y_pred = np.hstack((gp_test, rbf_test, poly_test))
            dst_pred = np.matmul(Y_pred, base_model_weight)
            dst_error = mean_squared_error(y_test, dst_pred)
            return dst_error
    elif len(model) == 4:
        if problem_param['name'] != 'chip':
            gp_test = model[0].predict_values(X_test).reshape(-1, 1)
            rbf_test = model[1].predict_values(X_test).reshape(-1, 1)
            poly_test = model[2].predict_values(X_test).reshape(-1, 1)
            Y_pred = np.hstack((gp_test, rbf_test, poly_test))
            second_layer_model = model[3]
            second_layer_pred = second_layer_model.predict(xgb.DMatrix(Y_pred)).ravel()
            second_layer_error = mean_squared_error(y_test, second_layer_pred)
            return second_layer_error

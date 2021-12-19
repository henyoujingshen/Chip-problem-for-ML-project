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
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.tree import DecisionTreeRegressor

from EA_testfunc_v6 import enabled_model, testFunc



xgboost_parameters = {
    'seed': 100,
    'nthread': -1,
    'gamma': 0,
    'lambda': 6,
    'max_depth': 6,
    'eta': 0.15,
    'objective': 'reg:squarederror'
}

poly_parameters = {
    'degree': 2,
    'include_bias': False,
    'alpha': 1
}

adaboost_parameters = {
    'n_estimators': 280,
    'loss': 'exponential',
    'learning_rate': 0.5
}

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
    parameters = {'seed': 100, 'nthread': -1, 'gamma': 0, 'lambda': 6,
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
    model_dt = DecisionTreeRegressor(random_state=0)
    model_dt.fit(X_train, y_train)
    pred_dt = model_dt.predict(X_valid).reshape(-1, 1)
    error_dt = mean_absolute_error(y_valid, pred_dt).reshape(-1, 1)

    # adaboost
    model_ada = AdaBoostRegressor(random_state=0, n_estimators=200, loss='exponential', learning_rate=0.5)
    model_ada.fit(X_train, y_train)
    pred_ada = model_ada.predict(X_valid).reshape(-1, 1)
    error_ada = mean_absolute_error(y_valid, pred_ada).reshape(-1, 1)

    # ramdom forest
    model_rf = RandomForestRegressor(max_depth=3, random_state=0, n_jobs=-1, ccp_alpha=0)
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

def baseModel(Sample_Train, parameters, num_boost_round, index):

    k = 5
    X, y = testFunc(Sample_Train)
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
    # XGBoost base model
    if 'XGBoost' in enabled_model:
        dtrain = xgb.DMatrix(X_train, label=y_train)
        model_xgb = xgb.train(parameters, dtrain, num_boost_round=num_boost_round)
        model.append(model_xgb)
        model_xgb.save_model('xgb' + str(index) + '.model')

    # poly base model
    if 'Polynomial' in enabled_model:
        model_poly = Pipeline(steps=[
            ('preprocessor', PolynomialFeatures(degree=2, include_bias=False)),
            ('estimator', Ridge(alpha=1))
        ])
        model_poly.fit(X_train, y_train)
        model.append(model_poly)
        with open('poly' + str(index) + '.model', 'wb') as file:
            pickle.dump(model_poly, file)

    # rf base model
    if 'RandomForest' in enabled_model:
        model_rf = RandomForestRegressor(max_depth=3, random_state=0, n_jobs=-1, ccp_alpha=0)
        model_rf.fit(X_train, y_train)
        model.append(model_rf)
        with open('poly' + str(index) + '.model', 'wb') as file:
            pickle.dump(model_rf, file)

    # AdaBoost base model
    if 'AdaBoost' in enabled_model:
        model_ada = AdaBoostRegressor(random_state=0, n_estimators=200, loss='exponential', learning_rate=0.8)
        model_ada.fit(X_train, y_train)
        model.append(model_ada)
        with open('ada' + str(index) + '.model', 'wb') as file:
            pickle.dump(model_ada, file)

    # 在validation data上进行预测，同时返回预测值和真实值
    valid_pred = []
    valid_error = []
    for i in range(3):
        if i == 0:
            dvalid = xgb.DMatrix(X_valid)
            pred = model[i].predict(dvalid).reshape(-1, 1)
            # error = mean_absolute_error(y_valid, pred).reshape(-1, 1)
            error = np.abs(y_valid - pred).reshape(-1, 1)
            valid_pred.append(pred)
            valid_error.append(error)
        else:
            pred = model[i].predict(X_valid).reshape(-1, 1)
            # error = mean_absolute_error(y_valid, pred).reshape(-1, 1)
            error = np.abs(y_valid - pred).reshape(-1, 1)
            valid_pred.append(pred)
            valid_error.append(error)

    base_round = np.hstack((valid_error[0], valid_error[1], valid_error[2], y_valid))
    base_round = pd.DataFrame(data=base_round, columns=['err_m1', 'err_m2', 'err_m3', 'y_valid'])

    return model, base_round, valid_pred, y_valid, valid_error

def metaModel(base_model_weight, Valid_Pred, Valid_Y):
    # 给base model在validation上的预测值赋予权重
    for i in range(3):
        Valid_Pred[i] = Valid_Pred[i] * base_model_weight[i]

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
    meta_parameters = {'booster': 'gbtree', 'seed': 100, 'nthread': -1, 'gamma': 0, 'lambda': 3,
                       'max_depth': 4, 'eta': 0.12, 'objective': 'reg:squarederror'}
    meta_num_boost_round = 220
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

    g_train = range(len(pred_train))
    plt.figure(figsize=(20, 10))
    plt.xlabel('Train Samples')
    plt.ylabel('Train Pred')
    plt.plot(g_train, pred_train, 'b-', lw=2)
    plt.plot(g_train, y_train, 'r-', lw=2)
    plt.show()

    g_valid = range(len(pred_valid))
    plt.figure(figsize=(20, 10))
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


Valid_Pred = [np.empty((0, 1)), np.empty((0, 1)), np.empty((0, 1))]
Valid_Y = np.empty((0, 1))
Valid_Error = [np.empty((0, 1)), np.empty((0, 1)), np.empty((0, 1))]
Base_Round = pd.DataFrame(columns=['err_m1', 'err_m2', 'err_m3', 'y_valid'])
Meta_Round = pd.DataFrame(columns=['err_meta', 'meta_valid'])
base_model_weight = np.array([1 / 3, 1 / 3, 1 / 3]).reshape(-1, 1)

def modelTrain(Sample_Train, parameters, num_boost_round, generation):
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

    Weight = [np.ones((5, 1)), np.ones((5, 1)), np.ones((5, 1))]
    meta_model = None

    if index == 4:
        base_model, base_round, valid_pred, y_valid, valid_error = baseModel(Sample_Train, parameters, num_boost_round, index)
        Base_Round = pd.concat([Base_Round, base_round], axis=0)
        a = Base_Round
        # 每五轮的矩阵拼接
        Valid_Y = np.vstack((Valid_Y, y_valid))
        for i in range(3):
            Valid_Pred[i] = np.vstack((Valid_Pred[i], valid_pred[i]))
            Valid_Error[i] = np.vstack((Valid_Error[i], valid_error[i]))
        for i in range(3):
            # k-fold权重
            w = np.ones((5, 1))
            w = w * 0.2
            Weight[i] = w

        base_model_weight = DSTweight(Valid_Y, Valid_Pred)
        # base_model_weight = naiveWeight(Valid_Error)
        # base_model_weight = np.array([1, 1, 1]).reshape(-1, 1)

        meta_model, meta_round = metaModel(base_model_weight, Valid_Pred, Valid_Y)
        Meta_Round = pd.concat([Meta_Round, meta_round], axis=0)
        b = Meta_Round

        Valid_Pred = [np.empty((0, 1)), np.empty((0, 1)), np.empty((0, 1))]
        Valid_Y = np.empty((0, 1))
        Valid_Error = [np.empty((0, 1)), np.empty((0, 1)), np.empty((0, 1))]
        print(str(len(Sample_Train)) + str(base_model_weight))
    else:
        base_model, base_round, valid_pred, y_valid, valid_error = baseModel(Sample_Train, parameters, num_boost_round, index)
        Base_Round = pd.concat([Base_Round, base_round], axis=0)
        a = Base_Round

        # 每五轮的矩阵拼接
        Valid_Y = np.vstack((Valid_Y, y_valid))
        for i in range(3):
            Valid_Pred[i] = np.vstack((Valid_Pred[i], valid_pred[i]))
            Valid_Error[i] = np.vstack((Valid_Error[i], valid_error[i]))
        # base_model_weight = DSTweight(Valid_Y, Valid_Pred)
        print(str(len(Sample_Train)) + str(base_model_weight))

    return base_model, meta_model, index, Weight, base_model_weight
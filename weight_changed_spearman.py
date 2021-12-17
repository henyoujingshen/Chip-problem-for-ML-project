import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsRegressor

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import pickle
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import max_error

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

def perfTest(X_train, y_train, X_valid, y_valid):
    # 不同base model的尝试
    # xgboost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    parameters = {'seed': 100, 'nthread': -1, 'gamma': 0, 'lambda': 6,
                  'max_depth': 6, 'eta': 0.15, 'objective': 'reg:squarederror'}
    model_xgb = xgb.train(parameters, dtrain, num_boost_round=200)
    dvalid = xgb.DMatrix(X_valid)
    pred_xgb = model_xgb.predict(dvalid).reshape(-1, 1)
    error_xgb = mean_absolute_error(pred_xgb, y_valid).reshape(-1, 1)
    # poly2
    model_poly2 = Pipeline(steps=[
        ('preprocessor', PolynomialFeatures(degree=2, include_bias=False)),
        ('estimator', Ridge(alpha=1))
    ])
    model_poly2.fit(X_train, y_train)
    pred_poly2 = model_poly2.predict(X_valid).reshape(-1, 1)
    error_poly2 = mean_absolute_error(pred_poly2, y_valid).reshape(-1, 1)
    # poly3
    model_poly3 = Pipeline(steps=[
        ('preprocessor', PolynomialFeatures(degree=3, include_bias=False)),
        ('estimator', Ridge(alpha=1))
    ])
    model_poly3.fit(X_train, y_train)
    pred_poly3 = model_poly3.predict(X_valid).reshape(-1, 1)
    error_poly3 = mean_absolute_error(pred_poly3, y_valid).reshape(-1, 1)
    # knn
    model_knn = KNeighborsRegressor(n_neighbors=6)
    model_knn.fit(X_train, y_train)
    pred_knn = model_knn.predict(X_valid).reshape(-1, 1)
    error_knn = mean_absolute_error(pred_knn, y_valid).reshape(-1, 1)
    # decesion tree
    model_dt = DecisionTreeRegressor(random_state=0)
    model_dt.fit(X_train, y_train)
    pred_dt = model_dt.predict(X_valid).reshape(-1, 1)
    error_dt = mean_absolute_error(pred_dt, y_valid).reshape(-1, 1)
    # adaboost
    model_ada = AdaBoostRegressor(random_state=0, n_estimators=200, loss='exponential', learning_rate=0.5)
    model_ada.fit(X_train, y_train)
    pred_ada = model_ada.predict(X_valid).reshape(-1, 1)
    error_ada = mean_absolute_error(pred_ada, y_valid).reshape(-1, 1)
    # ramdom forest
    model_rf = RandomForestRegressor(max_depth=3, random_state=0, n_jobs=-1, ccp_alpha=0)
    model_rf.fit(X_train, y_train)
    pred_rf = model_rf.predict(X_valid).reshape(-1, 1)
    error_rf = mean_absolute_error(pred_rf, y_valid).reshape(-1, 1)

    Pred = np.hstack((pred_xgb, pred_ada, pred_rf, pred_poly3, pred_poly2, pred_dt, pred_knn))
    Error = np.hstack((error_xgb, error_ada, error_rf, error_poly3, error_poly2, error_dt, error_knn))

    # return correlation coefficient and p values
    r = np.corrcoef(Pred, rowvar=False)
    r = pd.DataFrame(data=r, columns=['xgb', 'ada', 'rf', 'poly3', 'poly2', 'dt', 'knn'],
                     index=['xgb', 'ada', 'rf', 'poly3', 'poly2', 'dt', 'knn'])
    return Pred, Error, r


def baseModel(Sample_Train, Data, parameters, num_boost_round, index):
    model1 = 1
    model2 = 1
    model3 = 1
    activate_sign = [model1, model2, model3]

    k = 5
    X, y = findSamples_nofor(Sample_Train, Data)
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
    if activate_sign[0] == 1:
        dtrain = xgb.DMatrix(X_train, label=y_train)
        model_xgb = xgb.train(parameters, dtrain, num_boost_round=num_boost_round)
        model.append(model_xgb)
        model_xgb.save_model('xgb' + str(index) + '.model')
    # poly base model
    if activate_sign[1] == 1:
        model_poly = Pipeline(steps=[
            ('preprocessor', PolynomialFeatures(degree=2, include_bias=False)),
            ('estimator', Ridge(alpha=1))
        ])
        model_poly.fit(X_train, y_train)
        model.append(model_poly)
        with open('poly' + str(index) + '.model', 'wb') as file:
            pickle.dump(model_poly, file)
    # KNN base model
    if activate_sign[2] == 1:
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
            if activate_sign[0] == 1:
                dvalid = xgb.DMatrix(X_valid)
                pred = model[i].predict(dvalid).reshape(-1, 1)
                error = mean_absolute_error(pred, y_valid).reshape(-1, 1)
                valid_pred.append(pred)
                valid_error.append(error)
        else:
            if activate_sign[i] == 1:
                pred = model[i].predict(X_valid).reshape(-1, 1)
                error = mean_absolute_error(pred, y_valid).reshape(-1, 1)
                valid_pred.append(pred)
                valid_error.append(error)

    if len(Sample_Train) == 90:
        Pred, Err, Corr = perfTest(X_train, y_train, X_valid, y_valid)
        a = 1
    return model, valid_pred, y_valid, valid_error

def metaModel(Valid_Pred, Valid_Y):
    X_meta = Valid_Pred[0]
    y_meta = Valid_Y
    for i in range(1, 3):
        X_meta = np.hstack((X_meta, Valid_Pred[i]))
    dmeta = xgb.DMatrix(X_meta, label=y_meta)
    meta_parameters = {'booster': 'gbtree', 'seed': 100, 'nthread': -1, 'gamma': 0, 'lambda': 6,
                       'max_depth': 3, 'eta': 0.15, 'objective': 'reg:squarederror'}
    meta_num_boost_round = 100
    meta = xgb.train(meta_parameters, dmeta, num_boost_round=meta_num_boost_round)
    meta.save_model('meta_xgb.model')

    # train = meta.predict(dmeta)
    # g = range(len(X_meta))
    # plt.figure(figsize=(20, 10))
    # plt.xlabel('Samples')
    # plt.ylabel('Train Pred')
    # plt.plot(g, train, 'b-', lw=2)
    # plt.plot(g, y_meta, 'r-', lw=2)
    # plt.show()
    return meta

Valid_Pred = [np.empty((0, 1)), np.empty((0, 1)), np.empty((0, 1))]
Valid_Y = np.empty((0, 1))
Valid_Error = [np.empty((0, 1)), np.empty((0, 1)), np.empty((0, 1))]

def modelTrain(Sample_Train, Data, parameters, num_boost_round, generation):
    """
    k-fold要求初始采样点是几十一个
    根据训练种群，训练XGBoost代理模型，同时将模型保存为xgb.model文件
    :param generation: 正在进行的迭代次数
    :return: /
    """
    global Valid_Pred
    global Valid_Y
    global Valid_Error
    k = 5  # k-fold
    index = generation % k
    base_model, valid_pred, y_valid, valid_error = baseModel(Sample_Train, Data, parameters, num_boost_round, index)
    b=valid_error
    a=y_valid
    c=valid_pred
    # 每五轮的矩阵拼接
    Valid_Y = np.vstack((Valid_Y, y_valid))
    for i in range(3):
        Valid_Pred[i] = np.vstack((Valid_Pred[i], valid_pred[i]))
        Valid_Error[i] = np.vstack((Valid_Error[i], valid_error[i]))
    Weight = [np.ones((5, 1)), np.ones((5, 1)), np.ones((5, 1))]
    base_model_weight = np.ones((3, 1))
    meta_model = None
    if index == 4:
        # plt.figure(figsize=(20, 10))
        # plt.xlabel('Validation Prediction')
        # plt.ylabel('Validation Label')
        # plt.title('Pred vs Label')
        # plt.scatter(Valid_Pred, Y_Valid, alpha=1)
        # plt.show()
        # print("one round completed")
        for i in range(3):
            weight_coeff = (np.max(Valid_Error[i]) + np.min(Valid_Error[i]) - Valid_Error[i])
            Weight[i] = weight_coeff / np.sum(weight_coeff)
        m=Valid_Error
        n=Valid_Y
        q=Valid_Pred
        DST_MASS=np.ones((3,3))
        for i in range(3):
            vc = q[i]
            vb = n
            print(np.mean(np.multiply((vc - np.mean(vc)), (vb - np.mean(vb)))) / (np.std(vb) * np.std(vc)))
        # corrcoef得到相关系数矩阵（向量的相似程度）
            DST_MASS[i-1,0]=np.mean(np.multiply((vc - np.mean(vc)), (vb - np.mean(vb)))) / (np.std(vb) * np.std(vc))

        for i in range(3):
            y_pred = q[i]
            y_true = n
            DST_MASS[i-1, 1] =1/ max_error(y_true, y_pred)

        for i in range(3):
            y_pred = q[i]
            y_true = n
            DST_MASS[i-1, 2] = 1/mean_squared_error(y_true, y_pred)

        DST_MASS_TRANSFORMED=np.ones((3,3))
        for i in range(3):
            for j in range(4):
                DST_MASS_TRANSFORMED[i-1, j-1]=DST_MASS[i-1, j-1]/np.sum(DST_MASS[:, j-1])
        K=0
        for i in range(3):
            K=K+DST_MASS_TRANSFORMED[i-1,:].prod()
        for i in range(3):
            base_model_weight[i-1]=DST_MASS_TRANSFORMED[i-1,:].prod()/K

        meta_model = metaModel(Valid_Pred, Valid_Y)
        print(str(len(Sample_Train)) + str(base_model_weight))
    return base_model, meta_model, index, Weight, base_model_weight


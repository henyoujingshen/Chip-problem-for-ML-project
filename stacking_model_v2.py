import numpy as np
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import StratifiedGroupKFold

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import pickle

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


def baseModel(Sample_Train, Data, parameters, num_boost_round, index):
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

    # # 训练XGBoost作为base model
    # dtrain = xgb.DMatrix(X_train, label=y_train)
    # dvalid = xgb.DMatrix(X_valid, label=y_valid)
    # model = xgb.train(parameters, dtrain, num_boost_round=num_boost_round)
    # # 保存训练好的xgb模型
    # model.save_model('xgb' + str(index) + '.model')

    model = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('preprocessor', PolynomialFeatures(degree=3, include_bias=False)),
        ('estimator', Ridge(alpha=1))
    ])

    # fit the pipeline
    model.fit(X, y)

    with open('poly' + str(index) + '.model', 'wb') as file:
        pickle.dump(model, file)

    # 在validation data上进行预测，同时返回预测值和真实值
    valid_pred = model.predict(X_valid).reshape(-1, 1)
    valid_error = mean_squared_error(valid_pred, y_valid).reshape(-1, 1)
    return valid_pred, y_valid, valid_error


def metaModel(Valid_Pred, Valid_Y):
    dmeta = xgb.DMatrix(Valid_Pred, label=Valid_Y)
    meta_parameters = {'booster': 'gbtree', 'seed': 100, 'nthread': -1, 'gamma': 0, 'lambda': 6,
                       'max_depth': 3, 'eta': 0.15, 'objective': 'reg:squarederror'}
    meta_num_boost_round = 50
    meta = xgb.train(meta_parameters, dmeta, num_boost_round=meta_num_boost_round)
    meta.save_model('meta_xgb.model')


Valid_Pred = np.empty((0, 1))
Valid_Y = np.empty((0, 1))
Valid_Error = np.empty((0, 1))

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
    valid_pred, y_valid, valid_error = baseModel(Sample_Train, Data, parameters, num_boost_round, index)
    # 每五轮的矩阵拼接
    Valid_Pred = np.vstack((Valid_Pred, valid_pred))
    Valid_Y = np.vstack((Valid_Y, y_valid))
    Valid_Error = np.vstack((Valid_Error, valid_error))
    Weight = np.ones((5, 1))
    if index == 4:
        # plt.figure(figsize=(20, 10))
        # plt.xlabel('Validation Prediction')
        # plt.ylabel('Validation Label')
        # plt.title('Pred vs Label')
        # plt.scatter(Valid_Pred, Y_Valid, alpha=1)
        # plt.show()
        # print("one round completed")
        weight_coeff = (np.max(Valid_Error) + np.min(Valid_Error) - Valid_Error)
        Weight = weight_coeff / np.sum(weight_coeff)
        metaModel(Valid_Pred, Valid_Y)
        Valid_Pred = np.empty((0, 1))
        Valid_Y = np.empty((0, 1))
        Valid_Error = np.empty((0, 1))
    return index, Weight
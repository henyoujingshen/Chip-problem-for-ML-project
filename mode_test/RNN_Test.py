import warnings
import lhsmdu
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_squared_log_error
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor


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


if __name__ == '__main__':
    train_path = 'model_train_data.csv'
    test_path = 'model_test_data.csv'
    Train = np.array(pd.read_csv(train_path, header=None))
    Test = np.array(pd.read_csv(test_path, header=None))
    X_train = Train[:, 0:-1]
    y_train = Train[:, -1]
    X_test = Test[:, 0:-1]
    y_test = Test[:, -1]

    rnn = RadiusNeighborsRegressor(radius=45.0)
    rnn.fit(X_train, y_train)
    y_train_pred = rnn.predict(X_train)
    y_test_pred = rnn.predict(X_test)

    y_train_pred = np.abs(y_train_pred)
    y_test_pred = np.abs(y_test_pred)

    # 绘制训练集、测试集的拟合效果
    predict_plot(np.abs(y_train), y_train_pred)
    predict_plot(np.abs(y_test), y_test_pred)


    error0 = mean_squared_error(np.abs(y_test), y_test_pred)
    print("mean squared error:", error0)
    error1 = mean_absolute_percentage_error(np.abs(y_test), y_test_pred)
    print("mean absolute percentage error:", error1)
    error2 = mean_squared_log_error(np.abs(y_test), y_test_pred)
    print("mean squared log error:", error2)




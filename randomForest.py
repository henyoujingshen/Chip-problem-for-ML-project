import time
import warnings

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    path = 'full_scan_5d.csv'
    D = np.array(pd.read_csv(path))
    length = len(D[1])
    X = D[:, 0:length-1]
    y = D[:, length-1:length]

    Error = []
    Time = []
    TrainSize = range(10, 300)
    for i in TrainSize:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=i, test_size=20, random_state=666)
        y_test = y_test.ravel()
        y_train = y_train.ravel()

        start_time = time.perf_counter()
        # 建立随机森林
        rfc = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=1, min_samples_split=2,
                                    criterion='squared_error', random_state=1)
        rfc.fit(X_train, y_train)
        y_pred = rfc.predict(X_test)

        end_time = time.perf_counter()
        compute_time = end_time - start_time
        error = mean_absolute_percentage_error(y_test, y_pred)
        Error.append(error)
        Time.append(compute_time)

    plt.figure(figsize=(20, 10))
    plt.xlabel('DataSize')
    plt.ylabel('MAPError')
    plt.legend('MAPE', loc='lower right')
    plt.title('RF - DataSize vs MAPE')
    plt.plot(TrainSize, Error, 'r-', lw=2)
    plt.figure(figsize=(20, 10))
    plt.xlabel('DataSize')
    plt.ylabel('CompTime')
    plt.legend('Time', loc='lower right')
    plt.title('RF - DataSize vs Time')
    plt.plot(TrainSize, Time, 'g-', lw=2)
    plt.show()
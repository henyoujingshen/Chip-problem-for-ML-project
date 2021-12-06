# %%
# Import Library
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_squared_log_error

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline

print('Libraries imported')

def predict_plot(y_test, y_pred):
    # 编号 预测值绘图
    plt.figure(figsize=(19.20, 10.80), facecolor='w')
    ln_x_test = range(len(y_test))
    plt.plot(ln_x_test, y_test, 'r-', lw=2, label='Real')
    plt.plot(ln_x_test, y_pred, 'g-', lw=2, label='Model')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.title('Prediction')
    plt.show()

# load data
with open('./model_train_data.csv') as f:
    data_train = np.loadtxt(f, delimiter=',')

with open('./model_test_data.csv') as f:
    data_test = np.loadtxt(f, delimiter=',')

X_train, y_train = data_train[:, 0:-1], np.abs(data_train[:, -1])
X_test, y_test = data_test[:, 0:-1], np.abs(data_test[:, -1])

# define the model
model = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('estimator', SVR(C=1.0, kernel="poly", degree=3, coef0=0, gamma='auto', tol=1e-3))
])

# fit the model
model.fit(X_train, y_train)

# predict the curve
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

predict_plot(y_test, y_test_pred)
predict_plot(y_train, y_train_pred)

error0 = mean_squared_error(y_test, y_test_pred)
print("mean squared error:", error0)
error1 = mean_absolute_percentage_error(y_test, y_test_pred)
print("mean absolute percentage error:", error1)
error2 = mean_squared_log_error(y_test, y_test_pred)
print("mean squared log error:", error2)
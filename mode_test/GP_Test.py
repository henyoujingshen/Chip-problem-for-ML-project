# %%
# Import Library
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from botorch.models import FixedNoiseGP

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_squared_log_error

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

X_train, y_train = data_train[:, 0:-1], np.abs(data_train[:, -1]).reshape(-1, 1)
X_test, y_test = data_test[:, 0:-1], np.abs(data_test[:, -1]).reshape(-1, 1)

data_train[:, 0] = (data_train[:, 0] - data_train[:, 0].min()) / (data_train[:, 0].max() - data_train[:, 0].min())
data_train[:, 1] = (data_train[:, 1] - data_train[:, 1].min()) / (data_train[:, 1].max() - data_train[:, 1].min())
data_train[:, 2] = (data_train[:, 2] - data_train[:, 2].min()) / (data_train[:, 2].max() - data_train[:, 2].min())
data_train[:, 3] = (data_train[:, 3] - data_train[:, 3].min()) / (data_train[:, 3].max() - data_train[:, 3].min())
data_train[:, 4] = (data_train[:, 4] - data_train[:, 4].min()) / (data_train[:, 4].max() - data_train[:, 4].min())

data_test[:, 0] = (data_test[:, 0] - data_test[:, 0].min()) / (data_test[:, 0].max() - data_test[:, 0].min())
data_test[:, 1] = (data_test[:, 1] - data_test[:, 1].min()) / (data_test[:, 1].max() - data_test[:, 1].min())
data_test[:, 2] = (data_test[:, 2] - data_test[:, 2].min()) / (data_test[:, 2].max() - data_test[:, 2].min())
data_test[:, 3] = (data_test[:, 3] - data_test[:, 3].min()) / (data_test[:, 3].max() - data_test[:, 3].min())
data_test[:, 4] = (data_test[:, 4] - data_test[:, 4].min()) / (data_test[:, 4].max() - data_test[:, 4].min())

# %%
# Implement PyTorch and Bayesian loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

train_x, train_y = torch.as_tensor(X_train, device=device, dtype=dtype),\
                   torch.as_tensor(y_train, device=device, dtype=dtype)

train_mu = train_y.mean()
train_sig = train_y.std()
model = FixedNoiseGP(train_x, -(train_y - train_mu) / train_sig, train_Yvar=torch.full_like(train_y, 1e-6)).to(
    train_x)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_model(mll)

design_domain = torch.as_tensor(X_train, device=device, dtype=dtype)
mu = model.posterior(design_domain).mean.detach().squeeze(-1).cpu().numpy()
sigma = model.posterior(design_domain).variance.detach().squeeze(-1).cpu().numpy()

y_train_pred = np.abs(-mu * train_sig.cpu().numpy() + train_mu.cpu().numpy())

design_domain = torch.as_tensor(X_test, device=device, dtype=dtype)
mu = model.posterior(design_domain).mean.detach().squeeze(-1).cpu().numpy()
sigma = model.posterior(design_domain).variance.detach().squeeze(-1).cpu().numpy()

y_test_pred = np.abs(-mu * train_sig.cpu().numpy() + train_mu.cpu().numpy())

predict_plot(y_train, y_train_pred)
predict_plot(y_test, y_test_pred)

error0 = mean_squared_error(y_test, y_test_pred)
print("mean squared error:", error0)
error1 = mean_absolute_percentage_error(y_test, y_test_pred)
print("mean absolute percentage error:", error1)
error2 = mean_squared_log_error(y_test, y_test_pred)
print("mean squared log error:", error2)
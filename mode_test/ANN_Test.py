import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_squared_log_error
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional
from torch.optim import Adam
import torch.optim as optim
import torch

class ANN(nn.Module):
    def __init__(self, input, hidden1, hidden2):
        super(ANN, self).__init__()  # 对继承自父类的属性进行初始化
        self.hidden_1 = nn.Linear(in_features=input, out_features=hidden1, bias=True)
        self.reLU_1 = nn.ReLU()
        self.hidden_2 = nn.Linear(in_features=hidden1, out_features=hidden2, bias=True)
        self.reLU_2 = nn.ReLU()

    def init_normal(self):
        if type(self) == nn.Linear:
            nn.init.normal_(self.weight, mean=0, std=0.1)
            nn.init.zeros_(self.bias)

    def forward(self, x_input):
        z_1 = self.hidden_1(x_input)
        a_1 = self.reLU_1(z_1)
        z_2 = self.hidden_2(a_1)
        return z_2

def training(X_train, model, optimizer):
    losses = []
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        # 将模型设置为训练模式
        model.train()
        y_pred = model(X_train)
        # 计算分类的准确率,找到概率最大的下标
        _, pred = y_pred.max(dim=1)
        loss = criterion(y_pred, y_train)
        # 反向传播
        # optimizer.zero_grad()意思是把梯度置零，也就是把loss关于weight的导数变成0
        optimizer.zero_grad()
        loss.backward()
        # 这个方法会更新所有的参数，一旦梯度被如backward()之类的函数计算好后，我们就可以调用这个函数
        optimizer.step()
        # 记录误差
        losses.append(loss.item())
    return model, losses

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
    X_test = Test[:, 0:-1].ravel()
    y_test = Test[:, -1].ravel()

    y_train = np.abs(y_train)
    y_test = np.abs(y_test)

    # 模型训练和预测

    # 定义超参数
    learning_rate = 0.005  # 学习率
    epochs = 100  # 迭代次数
    # 实例化神经网络
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ANN(5, 20, 1)
    # 定义损失函数和优化器
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # 动量因子的作用
    X_train, X_test = torch.tensor(X_train).float(), torch.tensor(X_test).float()
    y_train, y_test = torch.tensor(y_train).long(), torch.tensor(y_test).long()

    model, cost = training(X_train, model, optimizer)

    _, y_train_pred = np.mean(model(X_train))
    _, y_test_pred = np.mean(model(X_test))

    # 绘制训练集、测试集的拟合效果
    predict_plot(np.abs(y_train), y_train_pred)
    predict_plot(np.abs(y_test), y_test_pred)

    error0 = mean_squared_error(np.abs(y_test), y_test_pred)
    print("mean squared error:", error0)
    error1 = mean_absolute_percentage_error(np.abs(y_test), y_test_pred)
    print("mean absolute percentage error:", error1)
    error2 = mean_squared_log_error(np.abs(y_test), y_test_pred)
    print("mean squared log error:", error2)




# %%
# Import Botorch library
import torch
import os
import numpy as np
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition.analytic import PosteriorMean
from botorch.acquisition.analytic import ProbabilityOfImprovement
from botorch.acquisition.analytic import UpperConfidenceBound
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.models import FixedNoiseGP
from botorch.models import HeteroskedasticSingleTaskGP
import matplotlib.pyplot as plt
from matplotlib import cm, ticker, colors

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import pandas as pd

print('Libraries imported')

# %%
# Load data
import random

n_h_emc = 5
n_cte_emc = 5
n_h_sub = 5
n_h_adhesive = 5
n_h_die = 5
n_variables = 5

def ranking(result):
    """
    获取某个体适应度整个真实搜索域中的排名
    :param result: 某个体适应度
    :return: 排名
    """
    benchmark = [0, 0.0168, 0.0239, 0.057, 0.059, 0.0689, 0.0729, 0.0809, 0.0858, 0.115013, 0.130164, 0.142919, 0.147871, 0.166071, 0.167726, 0.180223, 0.195995, 0.196783, 0.217258, 0.239674, 0.255737, 0.270209, 0.289178, 0.298477, 0.309586, 0.322446, 0.337163, 0.339115, 0.349386, 0.373878, 0.374401, 0.379241, 0.385782, 0.392128, 0.413967, 0.438757, 0.453112, 0.466661, 0.474914, 0.477166, 0.485028, 0.517961, 0.519821, 0.538555, 0.545344, 0.566704, 0.567065, 0.57607, 0.580767, 0.58662, 0.588015, 0.59527, 0.598695, 0.61892, 0.629343, 0.630311, 0.631692, 0.655743, 0.656904, 0.657748, 0.662486, 0.665945, 0.668199, 0.674794, 0.697299, 0.697769, 0.70639, 0.712995, 0.732316, 0.733463, 0.736107, 0.736155, 0.768301, 0.780585, 0.811329, 0.812936, 0.821925, 0.887417, 0.895446, 0.928705, 0.938831, 0.959684, 0.964033, 0.974505, 1.016448, 1.02202, 1.025155, 1.042937, 1.044545, 1.058703, 1.08692, 1.10015, 1.103519, 1.110195, 1.111611, 1.143553, 1.149645, 1.158169, 1.16672, 1.196585]
    new_arr = np.sort(np.append(np.array(benchmark), result))
    index = np.where(new_arr == result)
    return int(np.min(index))


with open('./3output_5d.csv') as f:
    full_scan = np.loadtxt(f, delimiter=',')

# Dataset structure:
# | 0-h_emc | 1-cte_emc | 2-h_sub | 3-h_adhesive | 4-h_die | 5-corner_warpage | 6-eq_mises_stress | 7-shear_stress |
full_scan[:, 0] = (full_scan[:, 0] - full_scan[:, 0].min()) / (full_scan[:, 0].max() - full_scan[:, 0].min())
full_scan[:, 1] = (full_scan[:, 1] - full_scan[:, 1].min()) / (full_scan[:, 1].max() - full_scan[:, 1].min())
full_scan[:, 2] = (full_scan[:, 2] - full_scan[:, 2].min()) / (full_scan[:, 2].max() - full_scan[:, 2].min())
full_scan[:, 3] = (full_scan[:, 3] - full_scan[:, 3].min()) / (full_scan[:, 3].max() - full_scan[:, 3].min())
full_scan[:, 4] = (full_scan[:, 4] - full_scan[:, 4].min()) / (full_scan[:, 4].max() - full_scan[:, 4].min())

# Set objective variable;
# | corner_warpage = -3 | eq_mises_stress = -2 | shear_stress = -1|
objective = -3
result = full_scan[:, objective]

n_train_init = 2  # Number of initial data
# Ramdon selection
train_data = full_scan[random.sample(range(0, len(result)), n_train_init)]
# fig, ax = plt.subplots(ncols=2)
# ax[0].hist(result, bins=25)
# ax[1].hist(train_data[:,-1], bins=n_train_init)
# Specific selection
# x1, x2, x3, x4, x5 = np.mgrid[0:n_variables-1:2j, 0:n_variables-1:2j, 0:n_variables-1:2j, 0:n_variables-1:2j, 0:n_variables-1:2j]
# points = np.hstack((x1.reshape(32, 1), x2.reshape(32, 1), x3.reshape(32, 1), x4.reshape(32, 1), x5.reshape(32, 1))).transpose()
# train_data_32 = full_scan[points.astype(int)[0], points.astype(int)[1], points.astype(int)[2], points.astype(int)[3], points.astype(int)[4]]
# train_data_4 = train_data_32[np.random.randint(32, size=4)]

print('Initial data loaded')

# %%
# Implement PyTorch and Bayesian loop
# acqf_para_list = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# directory_list = ['001', '01', '02', '03', '04', '05', '06', '07', '08', '09', '1']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
dtype = torch.double
design_domain = torch.as_tensor(full_scan[:, 0:n_variables], device=device, dtype=dtype)
# %%

# for j in range(len(acqf_para_list)):

# Create and change directory accordingly
# fr_path_c = 'Random_selection\\1_init\\ucb_'+directory_list[j]+'\\'
# if not os.path.exists(f_path_c+fr_path_c):
#     os.makedirs(f_path_c+fr_path_c)
# fr_path = 'Random_selection/1_init/ucb_'+directory_list[j]+'/'
# print(f_path+fr_path)

Round = 100

value = []

for r in range(Round):
    acqf_para = 10.0
    print('Acquisition parameter = ', acqf_para)
    acqf_para_list = []

    train_x = torch.as_tensor(train_data[:, 0:objective], dtype=dtype, device=device)
    train_x_total = torch.as_tensor(full_scan[:, 0:objective], dtype=dtype, device=device)
    train_y_origin = torch.as_tensor(train_data[:, objective], dtype=dtype, device=device).unsqueeze(1)
    train_y = train_y_origin ** 2
    best_observed_value = torch.sqrt(train_y.min())
    verbose = False

    # Bayesian loop
    Trials = 100
    for trial in range(1, Trials + 1):

        # print(f"\nTrial {trial:>2} of {Trials} ", end="\n")
        # print(f"Current best: {best_observed_value} ", end="\n")
        print(acqf_para)
        acqf_para_list.append(acqf_para)
        # fit the model
        train_mu = train_y.mean()
        train_sig = train_y.std()
        model = FixedNoiseGP(train_x, -(train_y - train_mu) / train_sig, train_Yvar=torch.full_like(train_y, 1e-6)).to(
            train_x)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)
        # fig, ax = plt.subplots(ncols=3)
        # ax[0].hist(model.posterior(train_x_total).variance.detach().numpy())
        # ax[1].hist(model.posterior(train_x_total).mean.detach().numpy())
        # for best_f, we use the best observed values as an approximation
        # EI = ExpectedImprovement(model = model, best_f = -train_y.min() - acqf_para,)
        UCB = UpperConfidenceBound(model=model, beta=acqf_para)
        # PI = ProbabilityOfImprovement(model = model, best_f = -train_y.min() - acqf_para,)
        # Evaluate acquisition function over the discrete domain of parameters
        acqf = UCB(design_domain.unsqueeze(-2))
        # ax[2].hist(acqf.detach().numpy())
        # plt.show()
        # np.savetxt(f_path+fr_path+'Acqf_matrix_' + str(trial) + '.csv', acqf.detach().cpu().numpy().flatten(), delimiter=',')
        acqf_sorted = torch.argsort(acqf, descending=True)
        acqf_max = acqf_sorted[0].unsqueeze(0)
        for j in range(1, 10):
            if acqf[acqf_max[0]] > acqf[acqf_sorted[j]]:
                break
            else:
                acqf_max = torch.cat((acqf_max, acqf_sorted[j].unsqueeze(0)))
        # for j in range(1, 10):
        #     if acqf[acqf_max[0]//(n_h_adhesive*n_h_sub*n_h_emc*n_cte_emc),
        #             acqf_max[0]%(n_h_adhesive*n_h_sub*n_h_emc*n_cte_emc)//(n_h_sub*n_h_emc*n_cte_emc),
        #             acqf_max[0]%(n_h_adhesive*n_h_sub*n_h_emc*n_cte_emc)%(n_h_sub*n_h_emc*n_cte_emc)//(n_h_emc*n_cte_emc),
        #             acqf_max[0]%(n_h_adhesive*n_h_sub*n_h_emc*n_cte_emc)%(n_h_sub*n_h_emc*n_cte_emc)%(n_h_emc*n_cte_emc)//n_cte_emc,
        #             acqf_max[0]%(n_h_adhesive*n_h_sub*n_h_emc*n_cte_emc)%(n_h_sub*n_h_emc*n_cte_emc)%(n_h_emc*n_cte_emc)%n_cte_emc] > \
        #         acqf[acqf_sorted[j]//(n_h_adhesive*n_h_sub*n_h_emc*n_cte_emc),
        #              acqf_sorted[j]%(n_h_adhesive*n_h_sub*n_h_emc*n_cte_emc)//(n_h_sub*n_h_emc*n_cte_emc),
        #              acqf_sorted[j]%(n_h_adhesive*n_h_sub*n_h_emc*n_cte_emc)%(n_h_sub*n_h_emc*n_cte_emc)//(n_h_emc*n_cte_emc),
        #              acqf_sorted[j]%(n_h_adhesive*n_h_sub*n_h_emc*n_cte_emc)%(n_h_sub*n_h_emc*n_cte_emc)%(n_h_emc*n_cte_emc)//n_cte_emc,
        #              acqf_sorted[j]%(n_h_adhesive*n_h_sub*n_h_emc*n_cte_emc)%(n_h_sub*n_h_emc*n_cte_emc)%(n_h_emc*n_cte_emc)%n_cte_emc]:
        #         break
        #     else:
        #         acqf_max = torch.cat((acqf_max, acqf_sorted[j].unsqueeze(0)))
        print(acqf_max.cpu().numpy())
        candidate_id = acqf_max[torch.randint(len(acqf_max), size=(1,))]
        candidate = design_domain[candidate_id]
        # print(candidate.tolist())

        mu = model.posterior(design_domain).mean.detach().squeeze(-1).cpu().numpy()
        sigma = model.posterior(design_domain).variance.detach().squeeze(-1).cpu().numpy()

        new_y = result[candidate_id]
        # print(new_y)
        train_new_y_origin = torch.as_tensor([[new_y]], dtype=dtype, device=device)
        train_new_y = train_new_y_origin ** 2
        # update training points
        train_x = torch.cat([train_x, candidate])
        train_y = torch.cat([train_y, train_new_y])
        train_y_origin = torch.cat([train_y_origin, train_new_y_origin])

        train_delta = train_new_y - best_observed_value
        TA = (Trials - trial) / Trials
        if TA <= torch.rand(1) and train_delta <= 0:
            acqf_para = acqf_para * 0.5
        if TA <= torch.rand(1) and train_delta.item() > 0:
            acqf_para = acqf_para * 0.8

        # train_delta = train_new_y - best_observed_value
        # if train_delta < 0:
        #     best_observed_value = train_new_y.item()
        #     if torch.exp(-train_delta/acqf_para) < torch.rand(1):
        #         acqf_para = acqf_para * 0.9
        #     else:
        #         acqf_para = acqf_para * 1.1
        # else:
        #     if torch.exp(-train_delta/acqf_para) > torch.rand(1):
        #         acqf_para = acqf_para * 0.9
        #     else:
        #         acqf_para = acqf_para * 1.1

        current_value = train_new_y.item()
        best_observed_value = train_y.min().item()

        if False:
            print(
                f"\nTrial {trial:>2}: current_value = "
                f"{current_value}, "
                f"best_value = "
                f"{best_observed_value} ", end=".\n"
            )
        else:
            print(".", end="")

    # %%
    optim_result = torch.cat([train_x, train_y_origin, train_y], 1)
    # np.savetxt(f_path+fr_path+'Optimization_loop.csv', optim_result.cpu().numpy(), delimiter=',')
    # print('Bayesian loop completed')
    # fig1, ax1 = plt.subplots()
    y_plot = abs(train_y_origin.cpu().numpy())
    # l1 = ax1.plot(y_plot * 1000, marker='.')
    # ax1.set_yscale('log')
    # ax1.set_xlabel('Loop iteration', fontsize='large')
    # ax1.set_ylabel('Absolute warpage (um)', fontsize='large')
    # plt.show()
    # plt.close()
    print("Min value="+str(np.min(y_plot) * 1000))
    value.append(ranking(np.min(y_plot) * 1000))
    # # fig1.savefig(f_path+fr_path+'Warpage_reduction.jpg', dpi=600)
    #
    # # %%
    # fig2, ax2 = plt.subplots()
    # l2 = ax2.plot(acqf_para_list, marker='*', c='g')
    # # ax2.set_yscale('log')
    # ax2.set_xlabel('Loop iteration', fontsize='large')
    # ax2.set_ylabel('Acquisition function parameter')
    # plt.show()
    # plt.close()
    # # %%

    # y_pred_all = np.sqrt(-mu * train_sig.cpu().numpy() + train_mu.cpu().numpy())
    # y_true_all = np.abs(result)
    # print(mean_absolute_percentage_error(y_true_all, y_pred_all))
    # print(mean_squared_error(y_true_all, y_pred_all))
    # print("done")



print(value)
print(np.mean(value))
name = ['rank']
test = pd.DataFrame(columns=name, data=value)  # 数据有三列，列名分别为one,two,three
test.to_csv('./rank_ref.csv', encoding='gbk')
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
import matplotlib.pyplot as plt
from matplotlib import cm, ticker, colors
from mpl_toolkits.mplot3d import axes3d

print('Libraries imported')

# %%
# Load data
import random

n_h_emc = 21
n_cte_emc = 21
n_variables = 2
# f_path = 'C:/Users/weijing/Documents/Nutstore/Abaqus_warpage_result/2_variables_result/'
with open('./full_scan_2d.csv') as f:
    full_scan = np.loadtxt(f, delimiter=',').reshape(n_h_emc, n_cte_emc, n_variables + 1)
n_train_init = 2
h_emc_max = full_scan[:, :, 0].max()
h_emc_min = full_scan[:, :, 0].min()
cte_emc_max = full_scan[:, :, 1].max()
cte_emc_min = full_scan[:, :, 1].min()
full_scan_norm = full_scan.copy()
full_scan_norm[:, :, 0] = (full_scan_norm[:, :, 0] - h_emc_min) / (h_emc_max - h_emc_min)
full_scan_norm[:, :, 1] = (full_scan_norm[:, :, 1] - cte_emc_min) / (cte_emc_max - cte_emc_min)

# %%
# Ramdon selection
train_data = full_scan_norm[(random.randint(0, n_h_emc - 1), random.randint(0, n_cte_emc - 1))][None]
while len(train_data) < n_train_init:
    temp = full_scan_norm[(random.randint(0, n_h_emc - 1), random.randint(0, n_cte_emc - 1))]
    if temp not in train_data:
        train_data = np.insert(train_data, -1, temp, axis=0)

# Specific selection
# train_data = full_scan[(0, 20, 10, 0, 20), (20, 0, 10, 0, 20)]

cte_emc, h_emc = np.meshgrid(full_scan[0, :, 1] / 10, full_scan[:, 0, 0])
fig1, ax1 = plt.subplots()
cs1 = ax1.contourf(cte_emc, h_emc, abs(full_scan[:, :, -1]) * 1000, levels=np.power(10, np.linspace(-1.5, 1.5, 31)),
                   locator=ticker.LogLocator(), cmap=cm.PuBu)
cbar1 = fig1.colorbar(cs1)
cbar1.set_ticks([0.01, 0.1, 1., 10., 100])
cbar1.set_label('Absolute warpage (um)')
ax1.set_title('Absolute warpage mapping (um)')
ax1.set_xlabel('EMC CTE (ppm)')
ax1.set_ylabel('EMC thickness (um)')
ax1.scatter((train_data[:, 1] * (cte_emc_max - cte_emc_min) + cte_emc_min) / 10,
            train_data[:, 0] * (h_emc_max - h_emc_min) + h_emc_min,
            marker='.', c='green', s=100)
ax1.set_xlim(7.95, 12.05)
ax1.set_ylim(545, 955)
fig2, ax2 = plt.subplots()
cs2 = ax2.contourf(cte_emc, h_emc, full_scan[:, :, -1] ** 2, levels=np.power(10, np.linspace(-7, -3, 31)),
                   locator=ticker.LogLocator(), cmap=cm.PuBu)
cbar2 = fig2.colorbar(cs2)
cbar2.set_ticks([0.0000001, 0.000001, 0.00001, 0.0001, 0.001])
cbar2.set_label('Squared warpage')
ax2.set_title('Squared warpage mapping')
ax2.set_xlabel('EMC CTE (ppm)')
ax2.set_ylabel('EMC thickness (um)')
ax2.scatter((train_data[:, 1] * (cte_emc_max - cte_emc_min) + cte_emc_min) / 10,
            train_data[:, 0] * (h_emc_max - h_emc_min) + h_emc_min,
            marker='.', c='green', s=100)
ax2.set_xlim(7.95, 12.05)
ax2.set_ylim(545, 955)
fig3 = plt.figure(figsize=(10, 7))
ax3 = fig3.gca(projection='3d', azim=-60, elev=25)
ax3.plot_surface(cte_emc, h_emc, abs(full_scan[:, :, -1]) ** 2, alpha=0.5)
print('Initial data loaded')

# %%
# Implement PyTorch and Bayesian loop
# acqf_para_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# directory_list = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '1']

# Create and change directory accordingly
# fr_path = 'Specific_selection/Right_bottom_two_diagonal/ucb_'+directory_list[j]+'/'
# print(f_path+fr_path)

acqf_para = 10.0
print('Acquisition parameter = ', acqf_para)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
dtype = torch.double
design_domain = torch.as_tensor(full_scan_norm[:, :, 0:n_variables], device=device, dtype=dtype)
train_x = torch.as_tensor(train_data[:, 0:-1], dtype=dtype, device=device)
train_y_origin = torch.as_tensor(train_data[:, -1], dtype=dtype, device=device).unsqueeze(1) * 1000
train_y = train_y_origin ** 2
best_observed_value = train_y.min()

verbose = False
# %%
# Bayesian loop
Trials = 25
for trial in range(1, Trials + 1):

    print(f"\nTrial {trial:>2} of {Trials} ", end="\n")
    print(f"Current best: {best_observed_value} ", end="\n")
    print(f"Current acquisition parameter: {acqf_para} ", end="\n")

    # fit the model
    # model = SingleTaskGP(train_x, -train_y).to(train_x)
    train_mu = train_y.mean()
    train_sig = train_y.std()
    model = FixedNoiseGP(train_x, -(train_y - train_mu) / train_sig, train_Yvar=torch.full_like(train_y, 1e-6)).to(
        train_x)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)

    # for best_f, we use the best observed values as an approximation
    # EI = ExpectedImprovement(model = model, best_f = -train_y.min(),)
    UCB = UpperConfidenceBound(model=model, beta=acqf_para)
    # PI = ProbabilityOfImprovement(model = model, best_f = -train_y.min(),)

    # Evaluate acquisition function over the discrete domain of parameters
    acqf = UCB(design_domain.unsqueeze(-2))
    # np.savetxt(f_path+fr_path+'Acqf_matrix_' + str(trial) + '.csv', acqf.detach().cpu().numpy(), delimiter=',')
    acqf_sorted = torch.argsort(acqf.flatten(), descending=True)
    acqf_max = acqf_sorted[0].unsqueeze(0)
    j = 1
    while acqf[acqf_max[0] // n_h_emc, acqf_max[0] % n_cte_emc] == acqf[
        acqf_sorted[j] // n_h_emc, acqf_sorted[j] % n_cte_emc]:
        acqf_max = torch.cat((acqf_max, acqf_sorted[j].unsqueeze(0)))
        j += 1
    candidate_id = acqf_max[torch.randint(len(acqf_max), size=(1,))]
    candidate = design_domain[candidate_id // n_h_emc, candidate_id % n_cte_emc]
    fig = plt.figure(figsize=(10, 7))
    ax = fig.gca(projection='3d', azim=-60, elev=30)
    mu = model.posterior(design_domain).mean.detach().squeeze(-1).numpy()
    sigma = model.posterior(design_domain).variance.detach().squeeze(-1).numpy()
    s1 = ax.plot_surface(cte_emc, h_emc, -mu * train_sig.numpy() + train_mu.numpy(),
                         alpha=0.5)
    # s2 = ax.plot_surface(cte_emc, h_emc, (-mu + 1.96 * sigma) * train_sig.numpy() + train_mu.numpy(),
    #                      color='orange', alpha=0.5)
    # s3 = ax.plot_surface(cte_emc, h_emc, (-mu - 1.96 * sigma) * train_sig.numpy() + train_mu.numpy(),
    #                      color='orange', alpha=0.5)
    s4 = ax.plot_surface(cte_emc, h_emc, (abs(full_scan[:, :, -1]) * 1000) ** 2,
                         color='g', alpha=0.5)
    # ax.plot(((candidate[:, 1] * (cte_emc_max - cte_emc_min) + cte_emc_min) / 10,
    #          (candidate[:, 1] * (cte_emc_max - cte_emc_min) + cte_emc_min) / 10),
    #         (candidate[:, 0] * (h_emc_max - h_emc_min) + h_emc_min,
    #          candidate[:, 0] * (h_emc_max - h_emc_min) + h_emc_min),
    #         ((-mu - 1.96 * sigma).min() * train_sig.numpy() + train_mu.numpy(),
    #          (-mu + 1.96 * sigma).max() * train_sig.numpy() + train_mu.numpy()), color='r')
    ax.scatter((train_x[:, 1].numpy() * (cte_emc_max - cte_emc_min) + cte_emc_min) / 10,
               train_x[:, 0].numpy() * (h_emc_max - h_emc_min) + h_emc_min,
               train_y.numpy().flatten(), color='b', s=40, alpha=1)
    ax.set_title('Iteration step ' + str(trial))
    ax.set_xlabel("EMC CTE (ppm)")
    ax.set_ylabel("EMC thickness (um)")
    ax.set_zlabel("Squared warpage value")
    plt.show()
    plt.close()
    new_y = full_scan[candidate_id // n_h_emc, candidate_id % n_cte_emc, -1]
    train_new_y_origin = torch.as_tensor([[new_y]], dtype=dtype, device=device) * 1000
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
    # else:
    #     if trial/Trials > torch.rand(1):
    #         acqf_para = acqf_para * 1.05

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
print('Bayesian loop completed')

fig, ax = plt.subplots()
cs = ax.contourf(cte_emc, h_emc, abs(full_scan[:, :, -1]) * 1000, levels=np.power(10, np.linspace(-1.5, 1.5, 31)),
                 locator=ticker.LogLocator(), cmap=cm.PuBu)
cbar1 = fig.colorbar(cs)
cbar1.set_ticks([0.01, 0.1, 1., 10., 100])
cbar1.set_label('Absolute warpage (um)')
ax.set_title('Absolute warpage mapping (um)')
ax.set_xlabel('EMC CTE (ppm)')
ax.set_ylabel('EMC thickness (um)')
ax.scatter((train_data[:, 1] * (full_scan[:, :, 1].max() - full_scan[:, :, 1].min()) + full_scan[:, :, 1].min()) / 10,
           train_data[:, 0] * (full_scan[:, :, 0].max() - full_scan[:, :, 0].min()) + full_scan[:, :, 0].min(),
           marker='.', c='green', s=100)
x1_plot = (train_x.numpy()[len(train_data):, 1] * (full_scan[:, :, 1].max() - full_scan[:, :, 1].min()) + full_scan[:,
                                                                                                          :,
                                                                                                          1].min()) / 10
x2_plot = train_x.numpy()[len(train_data):, 0] * (full_scan[:, :, 0].max() - full_scan[:, :, 0].min()) + full_scan[:, :,
                                                                                                         0].min()
c_plot = np.arange(1, len(x1_plot) + 1)
s1 = ax.scatter(x1_plot, x2_plot, marker='*', c=c_plot, cmap=cm.plasma, alpha=0.5, s=c_plot)
cbar2 = fig.colorbar(s1, orientation="horizontal")
cbar2.set_ticks([1, 20, 40, 60, 80, 100])
cbar2.set_label('Iteration')
ax.set_xlim(7.95, 12.05)
ax.set_ylim(545, 955)
plt.show()
# fig.savefig(f_path+fr_path+'Optimization_loop.jpg', dpi=600)

fig1, ax1 = plt.subplots()
y_plot = abs(train_y_origin.numpy())
l1 = ax1.plot(y_plot, marker='.')
ax1.set_yscale('log')
# ax1.set_ylim(10**-(3.7), 10**-(1.6))
ax1.set_xlabel('Loop iteration', fontsize='large')
ax1.set_ylabel('Absolute warpage (um)', fontsize='large')
# fig1.savefig(f_path+fr_path+'Warpage_reduction.jpg', dpi=600)


# %%

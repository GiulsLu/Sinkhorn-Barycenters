
import torch

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import time
import pickle

import os
import sys


# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '..' + os.path.sep + 'common')

script_path = os.path.dirname(os.path.abspath(__file__))
# data_path = os.path.join(script_path, '..', 'data', 'ellipses', 'ellipses.pckl')

sys.path.append(os.path.join(script_path,'..'))

from utils.plot_utils import plot
from Distribution.Distribution import Distribution
from Barycenter.GridBarycenter import GridBarycenter




data_path = os.path.join(script_path,'..','data','ellipses','ellipses.pckl')
save_path = os.path.join(script_path,'..','out','ellipses')


# Load the file containing the ellipses support and weights
pkl_file = open(data_path, 'rb')
D = pickle.load(pkl_file)
pkl_file.close()
pre_supp = D['support']
pre_weights = D['weights']
scale = 1
X = torch.linspace(0, scale * 1, 50)
Y = torch.linspace(0, scale * 1, 50)
X, Y = torch.meshgrid(X, Y)
X1 = X.reshape(X.shape[0] ** 2)
Y1 = Y.reshape(Y.shape[0] ** 2)

distributions = []
supp_meas = []
weights_meas = []
for i in range(len(pre_supp)):
    supp = torch.zeros((pre_supp[i].shape[0], 2))
    supp[:, 0] = X1[pre_supp[i]]
    supp[:, 1] = Y1[pre_supp[i]]
    supp_meas.append(supp)
    weights = (1 / pre_supp[i].shape[0]) * torch.ones(pre_supp[i].shape[0], 1)
    weights_meas.append(weights)
    distributions.append(Distribution(supp,weights))


init_bary = Distribution(torch.rand(10, 2)).normalize()


total_iter = 500
eps = 0.001

grid_step = 50

support_budget = total_iter + 100

bary = GridBarycenter(distributions, init_bary, support_budget = support_budget,\
                      grid_step = grid_step, eps=eps,\
                      sinkhorn_n_itr=100,sinkhorn_tol=1e-3)

save_every_n_iter = 1
num_fw_steps = 10
num_meta_fw_steps = int(total_iter/num_fw_steps)

for i in range(num_meta_fw_steps):
    t1 = time.time()
    bary.performFrankWolfe(num_fw_steps)
    t1 = time.time() - t1

    print('Iter:',(i+1)*num_fw_steps,'/',total_iter,'  Time for ',num_fw_steps,' FW iterations:', t1)

    if i % save_every_n_iter == 0:
        plot(bary.bary.support, bary.bary.weights)

        plt.savefig(os.path.join(save_path,'barycenter_{}.png'.format(i*num_fw_steps)))

        try:
            plt.pause(0.01)
        finally:
            pass


plot(bary.bary.support, bary.bary.weights)
plt.savefig(os.path.join(save_path, 'barycenter_end.png'))

try:
    plt.pause(0.01)
finally:
    pass


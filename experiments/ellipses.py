import torch
import time
import pickle

import os.path as op
import matplotlib

from otbar.utils import plot
from otbar import Distribution, GridBarycenter

matplotlib.use("TkAgg")


from matplotlib import pyplot as plt

torch.set_default_tensor_type(torch.DoubleTensor)

data_path = 'data/ellipses'
save_path = op.join(data_path, 'output')
ellipses_fname = op.join(data_path, 'Ellipses.pkl')


# Load the file containing the ellipses support and weights
pkl_file = open(ellipses_fname, 'rb')
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
    distributions.append(Distribution(supp, weights))


init_bary = Distribution(torch.rand(10, 2)).normalize()


total_iter = 500
eps = 0.001

grid_step = 50

support_budget = total_iter + 100

bary = GridBarycenter(distributions, init_bary, support_budget=support_budget,
                      grid_step=grid_step, eps=eps)

num_fw_steps = 10
num_meta_fw_steps = int(total_iter / num_fw_steps)

for i in range(num_meta_fw_steps):
    bary.performFrankWolfe(num_fw_steps)
    plot(bary.bary.support, bary.bary.weights)


plot(bary.bary.support, bary.bary.weights)
plt.savefig(op.join(save_path, 'barycenter_end.png'))

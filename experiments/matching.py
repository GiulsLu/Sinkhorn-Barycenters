




import torch
torch.set_default_tensor_type(torch.DoubleTensor)

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

from PIL import Image

import time
import pickle

import numpy as np


import os
import sys

script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_path,'..'))

from utils.plot_utils import plot
from Distribution.Distribution import Distribution
from Barycenter.GridBarycenter import GridBarycenter


data_path = os.path.join(script_path,'..','data','matching','cheetah.jpg')
save_path = os.path.join(script_path,'..','out','matching')


im_size = 100

# load and resize image
img = Image.open(data_path)
img.thumbnail((im_size,im_size), Image.ANTIALIAS)  # resizes image in-place
# imgplot = plt.imshow(img)

pix = np.array(img)
min_side = np.min(pix[:, :, 0].shape)
pix = 255 - pix[0:min_side, 0:min_side]


# visualize and save the resized original image
try:
    imgplot = plt.imshow(img)
    plt.savefig(os.path.join(save_path,'original.png'))
    plt.pause(0.1)
finally:
    pass


# create a meshgrid and interpret the image as a probability distribution on it
x = torch.linspace(0, 1, steps=pix.shape[0])
y = torch.linspace(0, 1, steps=pix.shape[0])
X, Y = torch.meshgrid(x, y)
X1 = X.reshape(X.shape[0] ** 2)
Y1 = Y.reshape(Y.shape[0] ** 2)
n = X.shape[0] ** 2
y1 = []

MX = max(X1)

weights = []
pix_arr = pix[:, :, 0].reshape(pix.shape[0] ** 2)
for i in range(n):
    if pix_arr[i] > 50:
        y1.append(torch.tensor([Y1[i],MX- X1[i]]))
        weights.append(torch.tensor(pix_arr[i], dtype=torch.float32))

nu1t = torch.stack(y1)
w1 = torch.stack(weights).reshape((len(weights), 1))
w1 = w1 / (torch.sum(w1, dim=0)[0])
supp_meas = [nu1t]
weights_meas = [w1]


# create the list of "distributions" of which we will compute the barycenter
distributions = [Distribution(nu1t,w1)]


# barycenter initialization
init = torch.Tensor([0.5, 0.5]).view(1,-1)
# init = Distribution(torch.rand(100,2)).normalize()

init_bary = Distribution(init).normalize()

total_iter = 20000
eps = 0.005

support_budget = total_iter + 100
grid_step = 100

grid_step = min(grid_step,im_size)

bary = GridBarycenter(distributions, init_bary, support_budget = support_budget,\
                      grid_step = grid_step, eps=eps)


n_iter_per_loop = 200
num_meta_fw_steps = int(total_iter/n_iter_per_loop)

try:
    plt.figure()
finally:
    pass

print('starting FW iterations')
for i in range(num_meta_fw_steps):
    t1 = time.time()
    bary.performFrankWolfe(n_iter_per_loop)
    t1 = time.time() - t1


    ### DEBUG FOR PRINTING
    print('Iter: ',(i+1)*n_iter_per_loop,'/',total_iter)
    print('n support points', bary.bary.support_size)
    print('Time for ', n_iter_per_loop,' FW iterations:', t1)
    print('')



    plot(bary.bary.support.cpu(), bary.bary.weights.cpu(),\
         bins=im_size, thresh=bary.bary.weights.min().item())
    try:
        plt.pause(0.1)
    finally:
        pass

    plt.savefig(os.path.join(save_path, 'barycenter_{}.png'.format(i)))


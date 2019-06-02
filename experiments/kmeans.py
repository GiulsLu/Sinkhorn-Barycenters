



import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import time

import torch
torch.set_default_tensor_type(torch.DoubleTensor)

import os
import sys

script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_path,'..'))


from Distribution.Distribution import Distribution
from Barycenter.GridBarycenter import GridBarycenter

import utils.mnist_utils as mnist
from utils.plot_utils import *
from utils.kmeans_utils import *

from utils.sinkhorn_utils import sinkhorn_divergence



# if we are asked to use cuda
if '--cuda' in sys.argv:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


def initial_plus(chosen_random, data, num_groups, reg):
    centroids = []
    centroids.append(chosen_random)
    for k in range(num_groups - 1):
        distances = np.zeros(len(data))
        for i in range(len(data)):
            distances[i] = sinkhorn_divergence(chosen_random.weights, \
                                               chosen_random.support, \
                                               data[i].weights, data[i].support, eps=reg)[0]

        probs = distances ** 2
        index = np.random.choice(np.arange(0, len(data)), p=probs / sum(probs))
        centroids.append(data[index])
        chosen_random = data[index]
    return centroids




save_path = os.path.join(script_path,'..','out','kmeans')


images = mnist.train_images()
labels = mnist.train_labels()

num_images = 500
images = images[0:num_images]
num_groups = 20


# create dataset
distrib = []
rescale = 1 / 28
for i in range(images.shape[0]):
    distrib.append(from_image_to_distribution(images[i], rescale))

reg = 0.001

ind_rand = np.random.randint(0, num_images)
first_centroid = from_image_to_distribution(images[ind_rand], rescale)
centroids_distrib = initial_plus(first_centroid,distrib, num_groups,reg)

# Frank-Wolfe stuff
grid_step = 30
fw_iter = 1500
support_budget = fw_iter + 100
init = torch.Tensor([0.5, 0.5])
init_bary = Distribution(init.view(1, -1)).normalize()
# -----

kmeans_iteration = 0
kmeans_iteration_max = 1000
while kmeans_iteration < kmeans_iteration_max:
    tic = time.time()
    print('\nK-Means Iteration N:',kmeans_iteration)
    t_group = time.time()

    num_groups = len(centroids_distrib)

    print("Total number of groups: ",num_groups)

    groups = partition_into_groups(distrib, centroids_distrib, num_groups,reg,rescale)
    centroids_distrib = []
    t_group_end = time.time()
    print('Time for group assignment:', t_group_end - t_group)

    for i in range(num_groups):

        if groups[i]==[]:
            continue

        bary = GridBarycenter(groups[i], init_bary, support_budget=support_budget, \
                              grid_step=grid_step, eps=reg)

        t = time.time()
        bary.performFrankWolfe(fw_iter)
        t2 = time.time()

        print('[Group',i,'] Time for',fw_iter,'FW iterations:', t2 - t)

        centroids_distrib.append(bary.bary)



    kmeans_iteration = kmeans_iteration + 1

    for idx in range(len(centroids_distrib)):
        plot(centroids_distrib[idx].support.cpu(),centroids_distrib[idx].weights.cpu())
        plt.savefig(os.path.join(save_path,'centroid_{}_at_iter_{}.png'.format(idx, kmeans_iteration)))

        try:
            plt.pause(0.1)
        finally:
            pass


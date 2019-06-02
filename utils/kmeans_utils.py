

import utils.mnist_utils as mnist
import scipy.misc
import torch
import numpy as np
from Distribution.Distribution import Distribution
import matplotlib.pyplot as plt
# from utils_glob_divergences.sinkhorn_balanced import sink
from utils.sinkhorn_utils import sinkhorn_divergence
import torch


def from_image_to_distribution(image, rescale):
    loc = np.argwhere(image > 0)
    weights = image[loc[:, 0], loc[:, 1]]
    weights = weights / sum(weights)
    loc = rescale * loc
    distrib = Distribution(torch.tensor(loc), torch.tensor(weights))
    return distrib


def partition_into_groups(data, centroids, num_groups, reg, rescale):
    groups = [[] for i in range(num_groups)]
    for i in range(len(data)):
        min_dist = 100
        for k in range(len(centroids)):
            dist = sinkhorn_divergence(centroids[k].weights, centroids[k].support, \
                                       data[i].weights, data[i].support, eps=reg)[0]

            if dist < min_dist:
                tmp_c = k
                min_dist = dist

        groups[tmp_c].append(data[i])
    return groups
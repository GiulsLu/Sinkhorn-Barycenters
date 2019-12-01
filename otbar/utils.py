

import torch
import numpy as np
from .distribution import Distribution
from .sinkhorn import sinkhorn_divergence
from matplotlib import pyplot as plt


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
            dist = sinkhorn_divergence(centroids[k].weights,
                                       centroids[k].support,
                                       data[i].weights, data[i].support,
                                       eps=reg)[0]

            if dist < min_dist:
                tmp_c = k
                min_dist = dist

        groups[tmp_c].append(data[i])
    return groups


def dist_matrix(x, y, p=2):
    x_y = x.unsqueeze(1) - y.unsqueeze(0)
    if p == 1:
        return x_y.norm(dim=2)
    elif p == 2:
        return (x_y ** 2).sum(2)
    else:
        return x_y.norm(dim=2)**(p/2)


def pre_histogram(points, _weights, thresh=0):
    weights = torch.clone(_weights)
    weights[weights <= thresh] = 0

    n = points.shape[0]
    x = []
    y = []
    for i in range(n):
        x.append(points[i][0])
        y.append(points[i][1])
    return np.array(x), np.array(y), weights.reshape(weights.shape[0])


def plot(point, weights, bins=50, thresh=0):
    xb, yb, wb = pre_histogram(point, weights, thresh=thresh)
    plt.hist2d(xb, yb, bins=[bins, bins], weights=wb)
    return xb, yb


def plotd(d, bins=50, thresh=0):
    return plot(d.support, d.weights, bins=bins, thresh=thresh)

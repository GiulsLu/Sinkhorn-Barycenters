
import torch
import matplotlib.pyplot as plt
import numpy as np


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
    return plot(d.support,d.weights,bins=bins,thresh=thresh)


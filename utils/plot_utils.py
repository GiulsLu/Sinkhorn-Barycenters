
import torch
import matplotlib.pyplot as plt
import numpy as np


def pre_histogram(points, weights):
    n = points.shape[0]
    x = []
    y = []
    for i in range(n):
        x.append(points[i][0])
        y.append(points[i][1])
    return np.array(x), np.array(y), weights.reshape(weights.shape[0])


def plot(point, wei, bins=50, thresh=0):
    tmp_plot = torch.clone(wei)
    tmp_plot[tmp_plot<=thresh] = 0

    xb, yb, wb = pre_histogram(point, tmp_plot)
    # plt.figure()
    plt.hist2d(xb, yb, bins=[bins, bins], weights=wb)
    return xb, yb



def plotd(d, bins=50, thresh=0):
    return plot(d.support,d.weights,bins=bins,thresh=thresh)


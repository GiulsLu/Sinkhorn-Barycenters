


import torch

import numpy as np
import matplotlib
# matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from utils_glob_divergences.sinkhorn_balanced_simple import *
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from utils.plot_utils import plot, plotd

from Distribution.Distribution import Distribution

from Barycenter.GridBarycenter import GridBarycenter

import time

import pickle
from PIL import Image


from utils.sinkhorn_divergence_full import sink, sym_sink



def test_repeat_support():

    n = 10
    m = 20

    x = torch.randn(n,2)
    y = torch.randn(m,2)

    w1a = torch.rand(n)
    w1b = torch.rand(n)

    wsum = (w1a+w1b).sum()
    w1a.mul_(1/wsum)
    w1b.mul_(1/wsum)

    w2 = torch.rand(m)

    d1 = Distribution(torch.cat([x,x]),torch.cat([w1a,w1b],0))
    d1small = Distribution(x,w1a+w1b)
    d2 = Distribution(y,w2).normalize()

    eps = 0.08

    a_y, b_x, _ = sink(d1.weights,d1.support,d2.weights,d2.support,eps=eps,nits=1000,tol=1e-9)
    a_ysmall, b_xsmall, _ = sink(d1small.weights,d1small.support,d2.weights,d2.support,eps=eps,nits=1000,tol=1e-9)

    val = torch.dot(a_y, d2.weights.squeeze())
    valsmall = torch.dot(a_y, d2.weights.squeeze())

    print('val1 =', val, 'valsmall1 = ', valsmall)

    val = val + torch.dot(b_x, d1.weights.squeeze())
    valsmall = valsmall + torch.dot(b_xsmall, d1small.weights.squeeze())

    print('val =', val, 'valsmall = ', valsmall)

    ciao = 3




def compare_same_support_increase():


    n = 1

    m = 20

    y = 0.05*torch.randn(m, 2) + 0.1
    w2 = torch.rand(m)
    d2 = Distribution(y, w2).normalize()


    x = 0.05*torch.randn(n,2)

    w = torch.rand(n)

    d_rep = Distribution(x,w,do_replace=True).normalize()
    d_norep = Distribution(x,w,do_replace=False).normalize()

    n_new = 200

    precision = 1e-2

    for i in range(n_new):

        new_point = 0.05*torch.randn(2)
        new_point = new_point.mul(1/precision).floor().mul(precision)

        d_rep.convexAddSupportPoint(new_point,1/(1+i))
        d_norep.convexAddSupportPoint(new_point,1/(1+i))


    plotd(d_rep)
    plt.show()
    plotd(d_norep)
    plt.show()



    eps = 0.001


    _, a_rep, _ = sym_sink(d_rep.weights,d_rep.support,eps=eps,nits=1000,tol=1e-9)
    _, a_norep, _ = sym_sink(d_norep.weights,d_norep.support,eps=eps,nits=1000,tol=1e-9)


    print(torch.dot(a_rep,d_rep.weights.squeeze()))
    print(torch.dot(a_norep,d_norep.weights.squeeze()))


    print(d_rep.support_size)
    print(d_norep.support_size)






    a_y, b_x, _ = sink(d_rep.weights,d_rep.support,d2.weights,d2.support,eps=eps,nits=1000,tol=1e-9)
    a_ysmall, b_xsmall, _ = sink(d_norep.weights,d_norep.support,d2.weights,d2.support,eps=eps,nits=1000,tol=1e-9)

    val = torch.dot(a_y, d2.weights.squeeze())
    valsmall = torch.dot(a_y, d2.weights.squeeze())

    print('val1 =', val, 'valsmall1 = ', valsmall)

    val = val + torch.dot(b_x, d_rep.weights.squeeze())
    valsmall = valsmall + torch.dot(b_xsmall, d_norep.weights.squeeze())

    print('val =', val, 'valsmall = ', valsmall)

    ciao = 3



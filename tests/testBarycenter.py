



import torch

import numpy as np
import matplotlib
# matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from utils_glob_divergences.sinkhorn_balanced_simple import *
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from utils.plot_utils import plot

from Distribution.Distribution import Distribution

from Barycenter.Barycenter import Barycenter

import time

import pickle
from PIL import Image


def testBarycenter():

	# generate a distribution with three points
	mu_support = torch.tensor([[1., 2.], [-3., 14.], [5., 9.]])
	mu_support2 = torch.tensor([[11., -2.], [53., 4.], [21., 0.]])
	mu_support3 = torch.tensor([[3., 4.], [83., 7.], [3., 3.]])
	mu_weights = torch.tensor([0.3, 1.0, 0.8])
	mu_weights2 = torch.tensor([0.3, 1.0, 0.8]).unsqueeze(1)

	mu0 = Distribution(mu_support)
	mu1 = Distribution(mu_support2, mu_weights)
	mu2 = Distribution(mu_support3, mu_weights2)

	mu0.normalize()
	mu1.normalize()
	mu2.normalize()

	bary_init_support = torch.tensor([[11.,12.],[8.,10.]])
	bary_init = Distribution(bary_init_support)
	bary_init.normalize()

	bary = Barycenter([mu0,mu1,mu2],bary=bary_init)

	# TODO compare with true sinkhorn
	bary._computeSinkhorn()



def compareSinks():

	eps = 0.1

	sizes = [10,20,14]
	# sizes = [10]
	nus = [Distribution(torch.randn(s,2),torch.rand(s,1)).normalize() for s in sizes]

	init_size = 7
	init_bary = Distribution(torch.randn(init_size,2),torch.rand(init_size,1)).normalize()

	bary = Barycenter(nus,init_bary,eps=eps,sinkhorn_tol=1e-3)

	b,a = bary._computeSinkhorn()
	bary._computeSymSinkhorn()



	# TODO capire perche' viene un numero diverso?
	a_s = [None]*len(sizes)
	b_s = [None]*len(sizes)
	for k in range(len(sizes)):
		a_s[k],b_s[k],_ = sink(nus[k].weights,nus[k].support,\
							   init_bary.weights,init_bary.support,\
							   p=2,eps = eps,tol=1e-3)


	a_s_best = [None]*len(sizes)
	b_s_best = [None]*len(sizes)
	for k in range(len(sizes)):
		a_s_best[k],b_s_best[k],_ = sink(nus[k].weights,nus[k].support,\
							   init_bary.weights,init_bary.support,\
							   p=2,eps = eps,tol=0,nits=1000)


	abest = torch.cat(a_s_best)
	bbest = torch.cat(b_s_best)

	a1 = torch.cat(a_s)
	b1 = torch.cat(b_s)

	print((a1-a.t()).norm())
	print((b1-b.t()).norm())

	print(torch.dot(a1.view(-1),bary.bary.weights.repeat(bary.num_distributions, 1).view(-1)) + \
		  torch.dot(b1.view(-1),bary.full_weights.view(-1)))

	print(torch.dot(a.view(-1),bary.bary.weights.repeat(bary.num_distributions, 1).view(-1)) + \
		  torch.dot(b.view(-1),bary.full_weights.view(-1)))


	print('best')

	print((a1-abest).norm())
	print((b1-bbest).norm())

	print((abest-a.t()).norm())
	print((bbest-b.t()).norm())

	print(torch.dot(abest.view(-1),bary.bary.weights.repeat(bary.num_distributions, 1).view(-1)) + \
		  torch.dot(bbest.view(-1),bary.full_weights.view(-1)))

	asym = bary.potential_bary_sym
	_,asym_s,_ = sym_sink(init_bary.weights,init_bary.support,p=2,eps = eps,tol=1e-3)

	print((asym.t()-asym_s).norm())

	ciao = 3



def performFWTest():

	sizes = [10,20,14]

	nus = [Distribution(torch.randn(s,2),torch.rand(s,1)).normalize() for s in sizes]

	init_size = 3
	init_bary = Distribution(torch.randn(init_size,2),torch.rand(init_size,1)).normalize()

	bary = Barycenter(nus,init_bary,support_budget=init_size+2)



	bary.performFrankWolfe(4)

	ciao = 3

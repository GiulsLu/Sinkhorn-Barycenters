




import torch

import numpy as np
from matplotlib import pyplot as plt
from utils.plot_utils import plot

from Distribution.Distribution import Distribution
from Barycenter.GridBarycenter import GridBarycenter

import time

import pickle
from PIL import Image

def testGridBarycenter():

	sizes = [10,20,14]

	nus = [Distribution(torch.randn(s,2),torch.rand(s,1)).normalize() for s in sizes]

	init_size = 20
	init_bary = Distribution(torch.randn(init_size,2),torch.rand(init_size,1)).normalize()

	bary = GridBarycenter(nus,init_bary,support_budget=init_size+2)



	bary.performFrankWolfe(4)

	print(bary.grid_step)

	ciao = 3




def testFirstFW():

	nu1 = Distribution(torch.tensor([0., 0.]).view(1,-1)).normalize()
	nu2 = Distribution(torch.tensor([1., 1.]).view(1,-1)).normalize()
	nus = [nu1,nu2]

	init_bary = Distribution(torch.randn(1,2)).normalize()

	bary = GridBarycenter(nus, init_bary, support_budget=200, grid_step = 200,eps=0.1)

	bary.performFrankWolfe(150)

	print(bary.bary.support)



	print(bary.grid_step)

	bary.performFrankWolfe(1)
	ciao = 3






def testDelteDiverse():
	d = 2
	n = 4
	m = 5
	y1 = torch.Tensor([[0.05, 0.2], [0.05, 0.7], [0.05, 0.8], [0.05, 0.9]])
	y1 = torch.reshape(y1, (n, d))

	y2 = torch.Tensor([[0.6, 0.25], [0.8, 0.1], [0.8, 0.23], [0.8, 0.61], [1., 0.21]])
	y2 = torch.reshape(y2, (m, d))

	eps = 0.001
	niter = 1000

	init = torch.Tensor([0.5, 0.5])

	nu1 = Distribution(y1).normalize()
	nu2 = Distribution(y2).normalize()
	nus = [nu1,nu2]

	mix_weighs = [0.5,0.5]


	init_bary = Distribution(init.view(1,-1)).normalize()

	# init_bary = Distribution(torch.rand(100,2)).normalize()

	# support_budget = niter + 100
	support_budget = 100

	bary = GridBarycenter(nus, init_bary, support_budget = support_budget,\
						  grid_step = 100, eps=eps,\
						  mixing_weights=mix_weighs,\
						  sinkhorn_n_itr=100,sinkhorn_tol=1e-3)


	for i in range(10):
		t1 = time.time()
		bary.performFrankWolfe(100)
		t1 = time.time() - t1

		print('Time for 100 FW iterations:', t1)
		### DEBUG FOR PRINTING

		print(min(bary.func_val) / 2)

		plot(bary.bary.support, bary.bary.weights)

		plt.figure()
		plt.plot(bary.func_val[30:])
		plt.show()

		ciao = 3

	# print(bary.bary.support)



def testGaussiane():

	eps = 0.01
	niter = 1000

	init = torch.Tensor([0.1, 0.1])

	nu1 = Distribution(0.08*torch.randn(40,2)+torch.tensor([1.0,1.0]).unsqueeze(1).t()).normalize()
	nu2 = Distribution(0.08*torch.randn(40,2)).normalize()
	nus = [nu1, nu2]

	init_bary = Distribution(init.view(1, -1)).normalize()

	# init_bary = Distribution(torch.rand(100,2)).normalize()

	bary = GridBarycenter(nus, init_bary, support_budget=niter + 100, \
						  grid_step=50, eps=eps,\
						  sinkhorn_tol=1e-3)



	for i in range(10):
		bary.performFrankWolfe(100)
		### DEBUG FOR PRINTING

		print(min(bary.func_val) / 2)

		plot(bary.bary.support, bary.bary.weights)

		plt.figure()
		plt.plot(bary.func_val[30:])
		plt.show()

		ciao = 3



	print(bary.bary.support)



def testTwoEllipses():
	# TESTS
	pkl_file = open('Ellipses.pckl', 'rb')
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


	init = torch.Tensor([0.5, 0.5])

	# init_bary = Distribution(init.view(1,-1)).normalize()



	init_bary = Distribution(torch.rand(100,2)).normalize()

	# f = open('./data_bary.pckl', 'rb')
	# x = pickle.load(f)
	# w = pickle.load(f)
	# f.close()

	# init_bary = Distribution(x, w)


	distributions = distributions[0:2]
	mixing_weights = [0.1,0.9]

	niter = 1000
	eps = 0.01

	support_budget = niter + 100

	bary = GridBarycenter(distributions, init_bary, support_budget = support_budget,\
						  grid_step = 100, eps=eps,\
						  mixing_weights=mixing_weights,\
						  sinkhorn_n_itr=100,sinkhorn_tol=1e-3)


	bary_list = []

	total_iter = 5000
	num_fw_steps = 10
	num_meta_fw_steps = int(total_iter/num_fw_steps)

	for i in range(num_meta_fw_steps):
		t1 = time.time()
		bary.performFrankWolfe(num_fw_steps)
		t1 = time.time() - t1

		print('Time for ',num_fw_steps,' FW iterations:', t1)
		### DEBUG FOR PRINTING

		print(min(bary.func_val))
		print(min(bary.func_val)/30)


		plt.plot(bary.func_val[30:])
		# plot(bary.bary.support, bary.bary.weights)
		plot(bary.best_bary.support, bary.best_bary.weights)
		plt.show()

		bary_list.append((bary.best_bary.support,bary.best_bary.weights))

		ciao = 3




def testEllipses():
	# TESTS
	pkl_file = open('Ellipses.pckl', 'rb')
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


	init = torch.Tensor([0.5, 0.5])

	# init_bary = Distribution(init.view(1,-1)).normalize()



	init_bary = Distribution(torch.rand(100,2)).normalize()

	# f = open('./data_bary.pckl', 'rb')
	# x = pickle.load(f)
	# w = pickle.load(f)
	# f.close()

	# init_bary = Distribution(x, w)

	niter = 1000
	eps = 0.001

	grid_step = 50

	support_budget = niter + 100

	bary = GridBarycenter(distributions, init_bary, support_budget = support_budget,\
						  grid_step = grid_step, eps=eps,\
						  sinkhorn_n_itr=100,sinkhorn_tol=1e-3)


	bary_list = []
	best_bary_list = []

	total_iter = 5000
	num_fw_steps = 10
	num_meta_fw_steps = int(total_iter/num_fw_steps)

	for i in range(num_meta_fw_steps):
		t1 = time.time()
		bary.performFrankWolfe(num_fw_steps)
		t1 = time.time() - t1

		print('Time for ',num_fw_steps,' FW iterations:', t1)
		### DEBUG FOR PRINTING

		print(min(bary.func_val))
		print(min(bary.func_val)/30)


		plt.plot(bary.func_val[30:])
		# plot(bary.bary.support, bary.bary.weights)
		plot(bary.bary.support, bary.bary.weights)
		plt.show()

		bary_list.append((bary.bary.support,bary.bary.weights))
		best_bary_list.append((bary.best_bary.support,bary.best_bary.weights))


		if i % 20 == 0:
			pickle.dump(bary_list, open("bary3.pckl", "wb"))
			pickle.dump(bary_list, open("best_bary3.pckl", "wb"))

		ciao = 3

	pickle.dump(bary_list, open("bary3.pckl", "wb"))
	pickle.dump(best_bary_list, open("best_bary3.pckl", "wb"))


def testMatchSimple():

	d = 2
	n = 4
	m = 5
	y1 = torch.Tensor([[0.05, 0.2], [0.05, 0.7], [0.05, 0.8], [0.05, 0.9]])
	y1 = torch.reshape(y1, (n, d))

	y2 = torch.Tensor([[0.6, 0.25], [0.8, 0.1], [0.8, 0.23], [0.8, 0.61], [1., 0.21]])
	y2 = torch.reshape(y2, (m, d))

	eps = 0.01
	# eps = 1
	niter = 1000


	init = torch.Tensor([0.5, 0.5])

	nu1 = Distribution(y1).normalize()
	nu2 = Distribution(y2).normalize()
	# nus = [nu1,nu2]

	# matching rather than barycenter!
	# nus = [nu2,nu2]
	nus = [nu2]

	init_bary = Distribution(init.view(1,-1)).normalize()

	# init_bary = Distribution(torch.rand(100,2)).normalize()

	# support_budget = niter + 100
	support_budget = 1000

	bary = GridBarycenter(nus, init_bary, support_budget = support_budget,\
						  grid_step = 100, eps=eps,\
						  sinkhorn_n_itr=100,sinkhorn_tol=1e-9)


	for i in range(20):
		t1 = time.time()
		bary.performFrankWolfe(100)
		t1 = time.time() - t1

		print('Time for 100 FW iterations:', t1)
		### DEBUG FOR PRINTING

		print(min(bary.func_val) / 2)

		plt.figure()
		plt.plot(bary.func_val[30:])
		plt.show()

		plot(bary.best_bary.support, bary.best_bary.weights)
		plt.show()

		plot(bary.bary.support, bary.bary.weights)
		plt.show()


		ciao = 3

	# print(bary.bary.support)




def testMatchEllipses():
	# TESTS
	pkl_file = open('Ellipses.pckl', 'rb')
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


	init = torch.Tensor([0.5, 0.5])

	init_bary = Distribution(init.view(1,-1)).normalize()

	# init_bary = Distribution(torch.rand(100,2)).normalize()

	niter = 2000
	eps = 0.001

	support_budget = niter + 100


	# select a single ellipse
	idx_selected = 0
	selected_distribution = [distributions[idx_selected]]

	plot(selected_distribution[0].support,selected_distribution[0].weights)
	plt.show()


	bary = GridBarycenter(selected_distribution, init_bary, support_budget = support_budget,\
						  grid_step = 100, eps=eps,\
						  sinkhorn_n_itr=100,sinkhorn_tol=1e-3)

	n_steps_per_plot = 100
	total_n_plots = int(niter/n_steps_per_plot)

	for i in range(total_n_plots):
		t1 = time.time()
		bary.performFrankWolfe(n_steps_per_plot)
		t1 = time.time() - t1

		print('Time for ', n_steps_per_plot, ' FW iterations:', t1)
		### DEBUG FOR PRINTING

		print(min(bary.func_val) / 2)

		plt.figure()
		plt.plot(bary.func_val[30:])
		plt.show()

		plot(bary.best_bary.support, bary.best_bary.weights)
		plt.show()

		plot(bary.bary.support, bary.bary.weights)
		plt.show()


		ciao = 3






def testMatch():
	##########IMAGE TEST

	im_size = 100

	img = Image.open(r"cheeta2.jpg")
	I = np.asarray(Image.open(r"cheeta2.jpg"))
	img.thumbnail((im_size,im_size), Image.ANTIALIAS)  # resizes image in-place
	imgplot = plt.imshow(img)
	plt.show()
	pix = np.array(img)
	min_side = np.min(pix[:, :, 0].shape)
	# pix = pix[2:,:,0]/sum(sum(pix[2:,:,0]))
	pix = 255 - pix[0:min_side, 0:min_side]
	# x=torch.linspace(0,1,steps=62)
	# y=torch.linspace(0,1,steps=62)
	# X, Y = torch.meshgrid(x, y)
	# X1 = X.reshape(X.shape[0]**2)
	# Y1 = Y.reshape(Y.shape[0] ** 2)

	imgplot = plt.imshow(img)
	# pix = pix/sum(sum(pix))
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
			# y1.append(torch.tensor([X1[i], Y1[i]]))
			weights.append(torch.tensor(pix_arr[i], dtype=torch.float32))

	nu1t = torch.stack(y1)
	w1 = torch.stack(weights).reshape((len(weights), 1))
	w1 = w1 / (torch.sum(w1, dim=0)[0])
	supp_meas = [nu1t]
	weights_meas = [w1]


	distributions = [Distribution(nu1t,w1)]



	init = torch.Tensor([0.5, 0.5])

	init_bary = Distribution(init.view(1,-1)).normalize()

	# init_bary = Distribution(torch.rand(100,2)).normalize()

	# init_bary = distributions[0]


	niter = 10000
	eps = 0.01

	support_budget = niter + 100
	grid_step = 100

	grid_step = min(grid_step,im_size)

	bary = GridBarycenter(distributions, init_bary, support_budget = support_budget,\
						  grid_step = grid_step, eps=eps,\
						  sinkhorn_n_itr=100,sinkhorn_tol=1e-3)


	plot(distributions[0].support,distributions[0].weights,bins=im_size)
	plt.show()


	n_iter_per_loop = 200

	print('starting FW iterations')
	for i in range(100):
		t1 = time.time()
		bary.performFrankWolfe(n_iter_per_loop)
		t1 = time.time() - t1

		print('Time for ', n_iter_per_loop,' FW iterations:', t1)
		### DEBUG FOR PRINTING
		print('n iterations = ',(i+1)*n_iter_per_loop, 'n support points', bary.bary.support_size)


		print(min(bary.func_val))


		plt.figure()
		plt.plot(bary.func_val[30:])
		# plot(bary.bary.support, bary.bary.weights,bins=grid_step)
		plt.show()

		plot(bary.best_bary.support, bary.best_bary.weights, bins=im_size)
		plt.show()

		plot(bary.best_bary.support, bary.best_bary.weights,\
			 bins=im_size, thresh=bary.best_bary.weights.min().item())
		plt.show()

		plot(bary.bary.support, bary.bary.weights,bins=im_size)
		plt.show()

		ciao = 3


def testProvideGridBarycenter():
	d = 2
	n = 4
	m = 5
	y1 = torch.Tensor([[0.05, 0.2], [0.05, 0.7], [0.05, 0.8], [0.05, 0.9]])
	y1 = torch.reshape(y1, (n, d))

	y2 = torch.Tensor([[0.6, 0.25], [0.8, 0.1], [0.8, 0.23], [0.8, 0.61], [1., 0.21]])
	y2 = torch.reshape(y2, (m, d))

	eps = 0.01
	niter = 1000

	init = torch.Tensor([0.5, 0.5])

	nu1 = Distribution(y1).normalize()
	nu2 = Distribution(y2).normalize()
	nus = [nu1, nu2]

	init_bary = Distribution(init.view(1, -1)).normalize()

	# init_bary = Distribution(torch.rand(100,2)).normalize()

	# support_budget = niter + 100
	support_budget = 100



	# create grid
	grid_step = 50

	min_max_range = torch.tensor([[0.0500, 0.1000],\
								  [1.0000, 0.9000]])

	margin_percentage = 0.05
	margin = (min_max_range[0, :] - min_max_range[1, :]).abs() * margin_percentage

	tmp_ranges = [torch.arange(min_max_range[0, i] - margin[i], min_max_range[1, i] + margin[i], \
							   ((min_max_range[1, i] - min_max_range[0, i]).abs() + 2 * margin[
								   i]) / grid_step) \
				  for i in range(d)]

	tmp_meshgrid = torch.meshgrid(*tmp_ranges)

	grid = torch.cat([mesh_column.reshape(-1, 1) for mesh_column in tmp_meshgrid], dim=1)
	# created grid

	bary = GridBarycenter(nus, init_bary, support_budget=support_budget, \
						  grid = grid, eps=eps, \
						  sinkhorn_n_itr=100, sinkhorn_tol=1e-3)

	for i in range(10):
		t1 = time.time()
		bary.performFrankWolfe(100)
		t1 = time.time() - t1

		print('Time for 100 FW iterations:', t1)
		### DEBUG FOR PRINTING

		print(min(bary.func_val) / 2)

		plot(bary.bary.support, bary.bary.weights)

		plt.figure()
		plt.plot(bary.func_val[30:])
		plt.show()

		ciao = 3

		# print(bary.bary.support)

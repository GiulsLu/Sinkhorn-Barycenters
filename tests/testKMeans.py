import matplotlib.pyplot as plt
import mnist.mnist as mnist
from utils.utils_kmeans import *
import os
import pickle

from Distribution.Distribution import Distribution
from Barycenter.GridBarycenter import GridBarycenter
import scipy.misc
import torch
from utils.plot_utils import *
import time

# torch.set_default_tensor_type('torch.cuda.FloatTensor')
print('miao')


def plot2(point, wei, bins=50, thresh=0):
	tmp_plot = torch.clone(wei)
	tmp_plot[tmp_plot <= thresh] = 0

	xb, yb, wb = pre_histogram(point, tmp_plot)
	# plt.figure()

	return xb, yb, wb


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


import matplotlib.pyplot as plt
# from utils_glob_divergences.sinkhorn_balanced import *
from utils.sinkhorn_divergence_full import sinkhorn_divergence as sink


def testKMeans():

	images = mnist.train_images()
	labels = mnist.train_labels()

	num_images = 500
	images = images[0:num_images]
	num_groups = 10
	# create dataset
	distrib = []
	rescale = 1 / 28
	for i in range(images.shape[0]):
		distrib.append(from_image_to_distribution(images[i], rescale))

	reg = 0.001
	# centroids_images = [images[i] for i in range(num_groups)]
	ind_rand = np.random.randint(0, num_images)
	first_centroid = from_image_to_distribution(images[ind_rand], rescale)
	centroids_distrib = initial_plus(first_centroid,distrib, num_groups,reg)

	# centroids_distrib = [Distribution(from_image_to_distribution(centroids_images[i],rescale)) for i in range(num_groups)]
	# groups = [[] for i in range(10)]
	# for j in range(num_images):
	# 	groups[labels[j]].append(from_image_to_distribution(images[j], rescale))

	err = 10
	tresh = 1e-4
	iter = 0
	n_iter_max = 1000
	while iter < n_iter_max:
		tic = time.time()
		print(iter)
		t_group = time.time()

		num_groups = len(centroids_distrib)

		print("num groups: ",num_groups)

		#for j in centroids_distrib:
		#	plotd(j, bins=28)
		#	plt.show()

		groups = partition_into_groups(distrib, centroids_distrib, num_groups,reg,rescale)
		centroids_distrib = []
		t_group_end = time.time()
		print('time_group', t_group_end - t_group)
		grid_step = 30
		niter = 1
		n_iter_per_loop = 1
		support_budget = niter + 100
		init = torch.Tensor([0.5, 0.5])

		init_bary = Distribution(init.view(1, -1)).normalize()
		for i in range(num_groups):

			if groups[i]==[]:
				continue

			# groups = partition_into_groups(distrib, centroids_distrib, num_groups, reg, rescale)
			bary = GridBarycenter(groups[i], init_bary, support_budget=support_budget, \
								  grid_step=grid_step, eps=reg, \
								  sinkhorn_n_itr=100, sinkhorn_tol=1e-3)

			t = time.time()
			for j in range(int(niter / n_iter_per_loop)):
				bary.performFrankWolfe(n_iter_per_loop)
			t2 = time.time()
			print('time',niter,'iter', t2 - t)
			centroids_distrib.append(bary.bary)
			#plotd(bary.bary, bins=28)

		iter = iter + 1
		directory = 'Kmeans2'
		if not os.path.exists(directory):
			os.makedirs(directory)
		if iter % 5 == 0:
			# for i in range(num_groups):
			#     xb,yb,wb = plot2(centroids_distrib[i].support.cpu(),centroids_distrib[i].weights.cpu(),bins=30)
			#     plt.hist2d(xb, yb, bins=[30,30], weights=wb)
			#     plt.savefig('./' + directory + '/' + 'centroid_{}_at_iter{}.png'.format(i, iter))
			for idx in range(num_groups):
				f = open('./' + directory + '/' + 'centroid_{}_at_iter_{}_reg0.001'.format(idx, iter) + '.pckl', 'ab')
				pickle.dump(centroids_distrib[idx].support.cpu(), f)
				pickle.dump(centroids_distrib[idx].weights.cpu(), f)
				f.close()
			groups_cpu = [[] for i in range(num_groups)]
			for i in range(len(groups)):
				for j in range(len(groups[i])):
					groups_cpu[i].append(Distribution(groups[i][j].support.cpu(),groups[i][j].weights.cpu()))
			f = open('./' + directory + '/' + 'centroid_{}_at_iter_{}_reg0.001'.format(idx, iter) + '.pckl', 'ab')
			pickle.dump(groups_cpu, f)
			f.close()

	print('end')
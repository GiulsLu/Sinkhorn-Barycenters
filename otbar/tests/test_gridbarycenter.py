import torch

# import numpy as np
# from matplotlib import pyplot as plt
# from .otbar.utils import plot

from otbar import Distribution, GridBarycenter

import time

# from PIL import Image


def test_gridbarycenter():

    sizes = [10, 20, 14]

    nus = [Distribution(torch.randn(s, 2), torch.rand(s, 1)).normalize()
           for s in sizes]

    init_size = 20
    init_bary = Distribution(torch.randn(init_size, 2),
                             torch.rand(init_size, 1)).normalize()

    bary = GridBarycenter(nus, init_bary,
                          support_budget=init_size+2)

    bary.performFrankWolfe(4)


def test_firstfW():

    nu1 = Distribution(torch.tensor([0., 0.]).view(1, -1)).normalize()
    nu2 = Distribution(torch.tensor([1., 1.]).view(1, -1)).normalize()
    nus = [nu1, nu2]

    init_bary = Distribution(torch.randn(1, 2)).normalize()

    bary = GridBarycenter(nus, init_bary, support_budget=200, grid_step=200,
                          eps=0.1)

    bary.performFrankWolfe(150)

    bary.performFrankWolfe(1)

#
# def test_match():
#     # IMAGE TEST
#
#     im_size = 100
#
#     img = Image.open(r"cheeta2.jpg")
#     img.thumbnail((im_size, im_size), Image.ANTIALIAS)
#     # resizes image in-place
#     # plt.imshow(img)
#     # plt.show()
#     pix = np.array(img)
#     min_side = np.min(pix[:, :, 0].shape)
#     # pix = pix[2:,:,0]/sum(sum(pix[2:,:,0]))
#     pix = 255 - pix[0:min_side, 0:min_side]
#     # x=torch.linspace(0,1,steps=62)
#     # y=torch.linspace(0,1,steps=62)
#     # X, Y = torch.meshgrid(x, y)
#     # X1 = X.reshape(X.shape[0]**2)
#     # Y1 = Y.reshape(Y.shape[0] ** 2)
#
#     # plt.imshow(img)
#     # pix = pix/sum(sum(pix))
#     x = torch.linspace(0, 1, steps=pix.shape[0])
#     y = torch.linspace(0, 1, steps=pix.shape[0])
#     X, Y = torch.meshgrid(x, y)
#     X1 = X.reshape(X.shape[0] ** 2)
#     Y1 = Y.reshape(Y.shape[0] ** 2)
#     n = X.shape[0] ** 2
#     y1 = []
#
#     MX = max(X1)
#
#     weights = []
#     pix_arr = pix[:, :, 0].reshape(pix.shape[0] ** 2)
#     for i in range(n):
#         if pix_arr[i] > 50:
#             y1.append(torch.tensor([Y1[i], MX - X1[i]]))
#             # y1.append(torch.tensor([X1[i], Y1[i]]))
#             weights.append(torch.tensor(pix_arr[i], dtype=torch.float32))
#
#     nu1t = torch.stack(y1)
#     w1 = torch.stack(weights).reshape((len(weights), 1))
#     w1 = w1 / (torch.sum(w1, dim=0)[0])
#     distributions = [Distribution(nu1t, w1)]
#     init = torch.Tensor([0.5, 0.5])
#
#     init_bary = Distribution(init.view(1, -1)).normalize()
#
#     # init_bary = Distribution(torch.rand(100,2)).normalize()
#
#     # init_bary = distributions[0]
#
#     niter = 10000
#     eps = 0.01
#
#     support_budget = niter + 100
#     grid_step = 100
#
#     grid_step = min(grid_step, im_size)
#
#     bary = GridBarycenter(distributions, init_bary,
#                           support_budget=support_budget,
#                           grid_step=grid_step, eps=eps,
#                           sinkhorn_n_itr=100, sinkhorn_tol=1e-3)
#
#     # plot(distributions[0].support, distributions[0].weights, bins=im_size)
#     # plt.show()
#
#     n_iter_per_loop = 200
#
#     print('starting FW iterations')
#     for i in range(100):
#         t1 = time.time()
#         bary.performFrankWolfe(n_iter_per_loop)
#         t1 = time.time() - t1
#
#         print('Time for ', n_iter_per_loop, 'FW iterations:', t1)
#         # DEBUG FOR PRINTING
#         print('n iterations = ', (i + 1) * n_iter_per_loop,
#               'n support points', bary.bary.support_size)
#
#         print(min(bary.func_val))
#
#         # plt.figure()
#         # plt.plot(bary.func_val[30:])
#         # # plot(bary.bary.support, bary.bary.weights,bins=grid_step)
#         # plt.show()
#
#         # plot(bary.best_bary.support, bary.best_bary.weights, bins=im_size)
#         # plt.show()
#
#         # plot(bary.best_bary.support, bary.best_bary.weights,
#         #      bins=im_size, thresh=bary.best_bary.weights.min().item())
#         # plt.show()
#         #
#         # plot(bary.bary.support, bary.bary.weights, bins=im_size)
#         # plt.show()


def test_provide_gridbarycenter():
    d = 2
    n = 4
    m = 5
    y1 = torch.Tensor([[0.05, 0.2], [0.05, 0.7], [0.05, 0.8], [0.05, 0.9]])
    y1 = torch.reshape(y1, (n, d))

    y2 = torch.Tensor([[0.6, 0.25], [0.8, 0.1], [0.8, 0.23], [0.8, 0.61],
                      [1., 0.21]])
    y2 = torch.reshape(y2, (m, d))

    eps = 0.01

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

    min_max_range = torch.tensor([[0.0500, 0.1000],
                                  [1.0000, 0.9000]])

    margin_percentage = 0.05
    margin = (min_max_range[0, :] - min_max_range[1, :]).abs()
    margin *= margin_percentage

    tmp_ranges = [torch.arange(min_max_range[0, i] - margin[i],
                  min_max_range[1, i] + margin[i],
                  ((min_max_range[1, i] - min_max_range[0, i]).abs() +
                  2 * margin[i]) / grid_step) for i in range(d)]

    tmp_meshgrid = torch.meshgrid(*tmp_ranges)

    grid = torch.cat([mesh_column.reshape(-1, 1)
                     for mesh_column in tmp_meshgrid], dim=1)
    # created grid

    bary = GridBarycenter(nus, init_bary, support_budget=support_budget,
                          grid=grid, eps=eps,
                          sinkhorn_n_itr=100,
                          sinkhorn_tol=1e-3)

    for i in range(10):
        t1 = time.time()
        bary.performFrankWolfe(100)
        t1 = time.time() - t1

        print('Time for 100 FW iterations:', t1)
        # DEBUG FOR PRINTING

        print(min(bary.func_val) / 2)

        # plot(bary.bary.support, bary.bary.weights)
        #
        # plt.figure()
        # plt.plot(bary.func_val[30:])
        # plt.show()

        print("Support of barycenter: ", bary.bary.support)

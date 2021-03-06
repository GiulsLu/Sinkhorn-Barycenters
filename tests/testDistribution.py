


import torch

from Distribution.Distribution import Distribution


def testDistributions():

    # generate a distribution with three points
    mu_support = torch.tensor([[1.,2.],[-3.,4.],[5.,9.]])
    mu0 = Distribution(mu_support)
    new_point = torch.tensor([9.,8.])

    rho = 0.1
    mu0.convexAddSupportPoint(new_point,rho)


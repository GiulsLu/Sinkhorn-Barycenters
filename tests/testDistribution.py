


import torch

from Distribution.Distribution import Distribution


def testDistributions():

    # generate a distribution with three points
    mu_support = torch.tensor([[1.,2.],[-3.,4.],[5.,9.]])
    mu_weights = torch.tensor([0.3,1.0,0.8])

    mu_weights2 = torch.tensor([0.3,1.0,0.8]).unsqueeze(1)


    mu0 = Distribution(mu_support)
    mu0limited = Distribution(mu_support,max_support_size=3)
    mu1 = Distribution(mu_support,mu_weights)
    mu2 = Distribution(mu_support,mu_weights2)

    new_point = torch.tensor([9.,8.])
    new_point2 = torch.tensor([9.,8.,0.])

    rho = 0.1

    mu0.convexAddSupportPoint(new_point,rho)

    # the next two should throw two different warnings
    mu0limited.convexAddSupportPoint(new_point,rho)
    mu0.convexAddSupportPoint(new_point2,rho)

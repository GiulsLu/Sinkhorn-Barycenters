import torch

from otbar import Distribution
from otbar import Barycenter


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

    bary_init_support = torch.tensor([[11., 12.], [8., 10.]])
    bary_init = Distribution(bary_init_support)
    bary_init.normalize()

    bary = Barycenter([mu0, mu1, mu2], bary=bary_init)

    bary._computeSinkhorn()


def testFW():

    sizes = [10, 20, 14]

    nus = [Distribution(torch.randn(s, 2), torch.rand(s, 1)).normalize()
           for s in sizes]

    init_size = 3
    init_bary = Distribution(torch.randn(init_size, 2),
                             torch.rand(init_size, 1)).normalize()

    bary = Barycenter(nus, init_bary, support_budget=init_size + 2)

    try:
        bary.performFrankWolfe(4)
    except Exception as msg:
        if msg.args[0] != "You are using a 'virtual' argminGradient method. This is doing nothing":
            raise Exception('Error in testFW')

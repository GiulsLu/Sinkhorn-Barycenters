import torch

from otbar import Distribution


torch.set_default_tensor_type(torch.DoubleTensor)


def test_distributions():

    # generate a distribution with three points
    mu_support = torch.tensor([[1., 2.], [-3., 4.], [5., 9.]])
    mu0 = Distribution(mu_support)
    new_point = torch.tensor([9., 8.])

    rho = 0.1
    mu0.convexAddSupportPoint(new_point, rho)

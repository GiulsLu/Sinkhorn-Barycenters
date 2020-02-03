import warnings

import torch

from .distribution import Distribution
from .utils import dist_matrix
from .sinkhorn import lse


# TODO random generation in the range
def _startingBary(distributions, min_max_range):
    pass


class Barycenter:
    """Abstract class for barycenter computation."""

    def __init__(self, distributions, bary=None,
                 eps=0.1, mixing_weights=None,
                 support_budget=100, sinkhorn_tol=1e-3,
                 sinkhorn_n_itr=100, **params):

        self.device = distributions[0].support.device

        # basic variables
        self.eps = eps
        self.support_budget = support_budget

        self.sinkhorn_tol = sinkhorn_tol
        self.sinkhorn_n_itr = sinkhorn_n_itr

        self.func_val = []

        # current iteration of FW
        self.current_iteration = 1

        if mixing_weights is None:
            mixing_weights = torch.tensor([0.], device=self.device)
        # the weights for each distribution
        self.mixing_weights = mixing_weights.clone().detach()
        # store the information about the distributions
        self._storeDistributions(distributions)
        if bary is None:
            bary = torch.tensor([0.], device=self.device)

        # initialize the barycenter
        self._initBary(bary)

        # customizable initialization function
        # it is performed *before* the creation of the distance matrices
        self._inner_init(**params)

        # now that we have both the starting barycenter and the distributions
        # compute the distance matrices of the corresponding supports
        self._initDistanceMatrices()

    # "virtual" method for custom initialization of child Barycenter classes
    def _inner_init(self, **params):
        pass

    # Store information about the distributions
    def _storeDistributions(self, distributions):

        # number of distributions
        self.num_distributions = len(distributions)

        # dimension of the ambient space
        self.d = distributions[0].support.size(1)

        # save a tensor filled with all support points of all distributions
        self.full_support = torch.cat([nu.support for nu in distributions],
                                      dim=0)
        self.full_weights = torch.cat([nu.weights for nu in distributions],
                                      dim=0)

        # if the weights for each distribution have not been provided,
        # assign one to each one
        # then normalize to have all the weights sum to one
        w_size = self.mixing_weights.size(0)
        if (w_size == 0 or w_size != self.num_distributions):
            self.mixing_weights = torch.ones(self.num_distributions,
                                             device=self.device) * \
                    (1.0/self.num_distributions)

        # if some weight is negative, throw an error
        if (self.mixing_weights < 0).sum() > 0:
            warnings.warn("Warning! Negative weights assigned to barycenter "
                          "distributions!")

        # smallest cube containing all distributions (hence also the
        # barycenter)
        # oriented as a 2 x d vector (first row "min" second row "max")
        row_min = self.full_support.min(0)[0].view(-1, 1)
        row_max = self.full_support.max(0)[0].view(-1, 1)
        self.min_max_range = torch.cat((row_min, row_max), dim=1).t()

        # list of support sizes
        support_pts = torch.tensor([nu.support_size for nu in distributions],
                                   device=self.device)
        self.support_number_points = support_pts

        # indices to recover the support points (we start with a leading 0 for
        # the first position)
        support_indx = torch.cat([torch.tensor([0], device=self.device),
                                 self.support_number_points.cumsum(dim=0)],)
        self.support_location_indices = support_indx

        self.potential_distributions = torch.zeros_like(self.full_weights)

    # Initialize the barycenter with the one provided or with a random point
    # in the distributions range
    def _initBary(self, bary):
        if not isinstance(bary, Distribution):
            # sample uniformly x in [0,1] and then x -> x*(max-min) + max
            # which corresponds to sampling x in [max,min]
            diff_max_min = self.min_max_range[1] - self.min_max_range[0]

            # Distribution class wants support tensors n x d
            bary = torch.rand(1, self.d, device=self.device)
            bary = bary * diff_max_min + self.min_max_range[0]

            bary_full_size = self.support_budget
        else:
            # the potential support of the barycenter distribution
            # is the max support plus the FW iterations
            # bary_full_size = bary.support_size + self.niter
            bary_full_size = max(bary.support_size, self.support_budget)

        self.bary = Distribution(bary, max_support_size=bary_full_size,
                                 device=self.device)

        self.best_bary = Distribution(bary, device=self.device)
        self.best_func_val = -1

        # we store the potentials for all sinkhorn computations of
        # all distributions against the current barycenter estimate
        bary_size = self.bary.support_size * self.num_distributions
        self.potential_bary = torch.zeros(bary_size, 1)

        # the potential for the OT(alpha,alpha)
        self.potential_bary_sym = torch.zeros((self.bary.support_size, 1),
                                              device=self.device)

    # initializes the big matrix containing all distances
    def _initDistanceMatrices(self):

        x_max_size = self.bary.max_support_size

        # big matirx containing support_bary x full_support + support_bary
        self._bigC = torch.empty((x_max_size,
                                  self.full_support.size(0) + x_max_size),
                                 device=self.device)
        self._updateDistanceMatrices(self.bary.support, 0)

    # updates the pointers (views) of all distribution-vs-bary distances
    # on the big matrix of all distances
    def _updateDistanceMatrices(self, x_new, idx):

        x_size = self.bary.support_size
        y_size = self.full_support.size(0)

        # update the pointers
        self.Cxy = self._bigC[:x_size, :y_size]
        self.Cxx = self._bigC[:x_size, y_size:(y_size+x_size)]

        sl_idx = self.support_location_indices
        self.Cxy_list = [self.Cxy[:, sl_idx[i]:sl_idx[i + 1]]
                         for i in range(self.num_distributions)]

        # current support of the barycenter
        bary_supp = self.bary.support

        C_full = dist_matrix(x_new, self.full_support)
        C_bary = dist_matrix(x_new, bary_supp)
        self.Cxy[idx:idx+x_new.size(0), :].copy_(C_full / self.eps)
        self.Cxx[idx:idx + x_new.size(0), :].copy_(C_bary / self.eps)

        # if we are not giving exactly a new support, then update both
        # corresponding columns and rows
        if x_new.size(0) != self.bary.support_size:
            C_bary_t = dist_matrix(x_new, bary_supp).t()
            self.Cxx[:, idx:idx + x_new.size(0)].copy_(C_bary_t / self.eps)

    # whenever a new point is added to the bary support,
    # we need to update the shape of the tensors containing the potentials
    # here we keep track of the previous potential as starting point for the
    # next Sinkhorn computation. We add a zero in the new position added
    def _updatePotentialsContainers(self):

        # make the potentials larger only if the bary has not yet reached its
        # maximum size (budget)
        if self.bary.support_size > self.potential_bary_sym.size(0):
            bary_size = self.bary.support_size * self.num_distributions
            tmp_potential_bary = torch.zeros(bary_size, 1)
            for k in range(self.num_distributions):
                idx_pre = self.bary.support_size * k
                idx_next = self.bary.support_size * (k+1) - 1

                idx_pre_old = (self.bary.support_size-1) * k
                idx_next_old = (self.bary.support_size-1) * (k + 1)

                _pot_bary = self.potential_bary[idx_pre_old:idx_next_old]
                tmp_potential_bary[idx_pre:idx_next].copy_(_pot_bary)

            self.potential_bary = tmp_potential_bary

            # the potential for the OT(alpha,alpha)
            tmp_potential_bary_sym = torch.empty(self.bary.support_size, 1,
                                                 device=self.device)
            tmp_potential_bary_sym[:-1].copy_(self.potential_bary_sym)
            tmp_potential_bary_sym[-1] = 0
            self.potential_bary_sym = tmp_potential_bary_sym

    # evaluate sinkhorn for the current barycenter and all distributions
    # code adapted from https://github.com/jeanfeydy/global-divergences
    def _computeSinkhorn(self):

        # we repeat the weight vector for the barycenter for each distribution
        α_log = self.bary.weights.log()
        β_log = self.full_weights.log()

        A = self.potential_distributions
        B = self.potential_bary

        # the iterations are performed for the potentials u/eps v/eps
        # we will multiply it back at the end of the Sinkhorn computation
        A.mul_(1 / self.eps)
        B.mul_(1 / self.eps)

        A_prev = torch.empty_like(A)

        # create list of pointers
        B_list = [B[(i * self.bary.support_size):((i + 1) *
                    self.bary.support_size), :]
                  for i in range(self.num_distributions)]

        Cxy = self.Cxy
        Cxy_list = self.Cxy_list
        tmpM = torch.empty_like(Cxy)

        # create list of pointers to the temporary matrix M
        sl_idx = self.support_location_indices
        tmpM_list = [tmpM[:, sl_idx[i]:sl_idx[i + 1]]
                     for i in range(self.num_distributions)]

        perform_last_step = False

        for idx_itr in range(self.sinkhorn_n_itr):

            A_prev.copy_(A)

            tmpM.copy_((A + β_log).view(1, -1) - Cxy)

            for idx_nu in range(self.num_distributions):
                B_list[idx_nu].copy_(-lse(tmpM_list[idx_nu]))

            # add alpha log (in place)
            for idx_nu in range(self.num_distributions):
                tmpM_list[idx_nu].copy_(B_list[idx_nu] +
                                        α_log - Cxy_list[idx_nu])

            A.copy_(-lse(tmpM.t()))

            if perform_last_step:
                break

            err = self.eps * (A - A_prev).abs().mean()
            # Stopping criterion: L1 norm of the updates
            if self.num_distributions*err.item() < self.sinkhorn_tol:
                perform_last_step = True

        A.mul_(self.eps)
        B.mul_(self.eps)

        # compute the sinkhorn functional OTe(alpha,beta)
        tmp_func_val = 0
        for idx_nu in range(self.num_distributions):

            inner_tmp_func_val = 0

            a = A[sl_idx[idx_nu]:sl_idx[idx_nu+1], :].view(-1)
            s_a = self.full_weights[sl_idx[idx_nu]:sl_idx[idx_nu + 1], :]
            s_a = s_a.view(-1)
            inner_tmp_func_val = inner_tmp_func_val + torch.dot(a, s_a)

            b = B_list[idx_nu].view(-1)

            inner_tmp_func_val = inner_tmp_func_val + \
                torch.dot(b, self.bary.weights.view(-1))

            tmp_func_val = tmp_func_val + \
                self.mixing_weights[idx_nu] * inner_tmp_func_val

        self.func_val.append(tmp_func_val.item())

        return A, B

    # Compute OTe(alpha,alpha)
    # code adapted from https://github.com/jeanfeydy/global-divergences
    def _computeSymSinkhorn(self):

        α_log = self.bary.weights.log()
        A = self.potential_bary_sym

        # the iterations are performed for the potentials u/eps v/eps
        # we will multiply it back at the end of the Sinkhorn computation
        A.mul_(1 / self.eps)

        A_prev = torch.empty_like(A)

        for idx_itr in range(self.sinkhorn_n_itr):

            A_prev.copy_(A)

            A.copy_(0.5 * (A - lse((A + α_log).view(1, -1) - self.Cxx)))
            # a(x)/ε = .5*(a(x)/ε + Smin_ε,y~α [ C(x,y) - a(y) ] / ε)

            err = self.eps * (A - A_prev).abs().mean()
            # Stopping criterion: L1 norm of the updates
            if err.item() < self.sinkhorn_tol:
                break

        A.copy_(- lse((A + α_log).view(1, -1) - self.Cxx))
        # a(x) = Smin_e,z~α [ C(x,z) - a(z) ]

        A.mul_(self.eps)

        tmp_func_val = self.mixing_weights.sum() *\
            torch.dot(A.view(-1), self.bary.weights.view(-1))
        self.func_val[-1] = self.func_val[-1] - tmp_func_val.item()

        return A

    # "Virtual" method for gradient minimization w.r.t. z
    def _argminGradient(self, potentials_distributions, potentials_bary_sym):
        # warnings.warn("You are using a 'virtual' argminGradient method. "
        #               "This is doing nothing")
        raise Exception("You are using a 'virtual' argminGradient method. "
                        "This is doing nothing")
        # return torch.randn(self.full_support[0,:].size(0),1)
        return None

    def currentRho(self):
        rho = torch.tensor([float(self.current_iteration)]).pow(1).item()
        rho = rho.to(self.device)
        return 1 / (1 + rho)

    # perform a single step of the Frank Wolfe algorithm
    def performFrankWolfe(self, num_itr=1):

        for idx_itr in range(num_itr):
            potentials_distributions, _ = self._computeSinkhorn()
            potentials_bary_sym = self._computeSymSinkhorn()

            new_support_point = self._argminGradient(potentials_distributions,
                                                     potentials_bary_sym)

            rho = self.currentRho()

            # add the new point to the support of the barycenter
            # perform self = (1-rho) * self + rho * other
            self.bary.convexAddSupportPoint(new_support_point, rho)

            # update the distance matrices with the new point (needs to be a 1
            # x d tensor)
            self._updateDistanceMatrices(new_support_point.view(1, -1),
                                         self.bary.last_updated_idx)

            # update the tensors containing the Sinkhorn potentials
            self._updatePotentialsContainers()

            # update the iteration counter
            self.current_iteration = self.current_iteration + 1

            # if the current barycenter is the best one so far...
            is_better = self.best_func_val > self.func_val[-1]
            if self.best_func_val < 0 or is_better:
                self.best_bary = Distribution(self.bary)
                self.best_func_val = self.func_val[-1]

    # perform one single step of FW
    def performFrankWolfeStep(self):
        self.performFrankWolfe(1)

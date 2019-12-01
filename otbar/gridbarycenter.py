import torch

from .barycenter import Barycenter
from .utils import dist_matrix


class GridBarycenter(Barycenter):

    # create a meshgrid to be used to evalute the Sinkhorn potential and
    # take the max
    def _inner_init(self, grid=None, grid_step=50,
                    grid_margin_percentage=0.05, **params):
        if grid is None:
            margin = (self.min_max_range[0, :] -
                      self.min_max_range[1, :]).abs()
            margin *= grid_margin_percentage

            tmp_ranges = [torch.arange(self.min_max_range[0, i]-margin[i],
                                       self.min_max_range[1, i]+margin[i],
                                       ((self.min_max_range[1, i] -
                                         self.min_max_range[0, i]).abs() +
                                        2 * margin[i]) / grid_step)
                          for i in range(self.d)]

            tmp_meshgrid = torch.meshgrid(*tmp_ranges)

            self.meshgrid = torch.cat([mesh_column.reshape(-1, 1) for
                                      mesh_column in tmp_meshgrid], dim=1)

        else:
            self.meshgrid = grid

    # overload of the init distance matrices method of standard barycenter
    # to account for the meshgrid
    def _initDistanceMatrices(self):

        x_max_size = self.bary.max_support_size
        grid_size = self.meshgrid.size(0)

        # Commented out since never used ?
        # y_size = self.full_support.size(0)

        # big matirx containing bary vs grid distances
        self._bigGridC = torch.empty(x_max_size, grid_size)

        # big matirx containing grid vs full support distances
        self.Cgridy = dist_matrix(self.meshgrid, self.full_support) / self.eps
        sl_idx = self.support_location_indices
        self.Cgridy_list = [self.Cgridy[:, sl_idx[i]:sl_idx[i+1]]
                            for i in range(self.num_distributions)]

        # first initialize the default matrices
        super(GridBarycenter, self)._initDistanceMatrices()

    # updates the pointers (views) of all distribution-vs-bary distances
    # on the big matrix of all distances
    def _updateDistanceMatrices(self, x_new, idx):

        # first initialize the default matrices
        super(GridBarycenter, self)._updateDistanceMatrices(x_new, idx)

        x_size = self.bary.support_size

        # update the pointers
        self.Cxgrid = self._bigGridC[:x_size, :]

        Cx = dist_matrix(x_new, self.meshgrid)
        self.Cxgrid[idx:idx+x_new.size(0), :].copy_(Cx / self.eps)

    # overload of the _argminGradient virtual method for the class Barycenter
    # solve the inner FW problem
    def _argminGradient(self, potentials_distributions, potentials_bary_sym):

        # create list of pointers to corresponding potentials and weights
        sl_idx = self.support_location_indices

        tmpM = (potentials_distributions / self.eps +
                self.full_weights.log()).view(1, -1) - self.Cgridy
        tmpM_list = [tmpM[:, sl_idx[k]:sl_idx[k + 1]]
                     for k in range(self.num_distributions)]

        V = torch.ones(self.meshgrid.size(0), 1)
        Vmax = torch.zeros(self.meshgrid.size(0), 1)
        for k in range(self.num_distributions):

            Vmax_k = torch.max(tmpM_list[k], 1)[0].view(-1, 1)

            lse_arg = (tmpM_list[k] - Vmax_k).exp().sum(1)
            lse_arg = lse_arg.pow(self.mixing_weights[k]).view(-1, 1)
            V.mul_(lse_arg)
            Vmax_k.mul_(self.mixing_weights[k])

            Vmax.add_(Vmax_k)

        tmpN = (potentials_bary_sym / self.eps +
                self.bary.weights.log()).view(1, -1) - self.Cxgrid.t()

        U = (tmpN - Vmax).exp().sum(1).view(-1, 1)

        feval = V.mul(U.reciprocal())

        idx_max = feval.argmax()

        out = torch.clone(self.meshgrid[idx_max, :])
        return out

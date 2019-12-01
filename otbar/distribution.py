import torch

import warnings


class Distribution:
    """Distribution object.

    Attributes:
    -----------
    support: matrix m x d of m support points in R^d

    weights: vector (m x 1) of the associated weights

    max_support_size: maximum number of support points allowed (default 1000)

    prune_close_flag:	a flag checking whether to verify if a point is
        already in the support (default False)
    prune_close_accuracy:  the accuracy at which to perform the pruning of
        close points (default 0)
    prune_weights_flag:	 a flag to prune points that have weight under a
        given threshold (default False)
    prune_weights_threshold: the threshold to prune support points
        with small weight (default 0.001)

    """
    def __init__(self, support, weights=torch.tensor([]),
                 max_support_size=1000, has_budget=True,
                 do_replace=True):

        # get pointers to the actual support and weights torch tensors
        # if the variable support is actually a Distribution, we copy it
        if isinstance(support, Distribution):
            tmp_support = support.support
            tmp_weights = support.weights
        else:
            tmp_support = support
            tmp_weights = weights.view(-1, 1)
            # make sure to "see" the tensor as a d x 1 vector

        # get the maximum allowed support size
        self.max_support_size = max(tmp_support.size(0), max_support_size)

        # create the tensors for the full support and weights
        self._full_support = torch.empty(self.max_support_size,
                                         tmp_support.size(1))
        self._full_weights = torch.zeros(self.max_support_size, 1)

        # the current support index identifies up to where we have filled
        # the support tensor
        self.support_size = tmp_support.size(0)

        # create the pointers to the current support and weights
        self.support = self._full_support[:self.support_size]
        self.weights = self._full_weights[:self.support_size]

        # copy the support tensor
        self.support.copy_(tmp_support)

        # if we do not provide a weight vector just give uniform weight = 1
        if tmp_weights.size(0) == 0:
            self.weights[:] = 1
        else:
            self.weights.copy_(tmp_weights[:self.support_size])

        # the current index starts at the last element of the support
        self._current_index = self.support_size-1

        # if has_budget is true, budget corresponds to max_size
        # whenever a new point is added, if it is over the budget the
        # current_index just goes back to 0
        self.has_budget = has_budget

        # if a point is already in the support, just update
        # the weight but not increase the support
        self.do_replace = do_replace

        # if the bary has still room to grow its support then, set this flag
        # to true
        self.is_growing = self._current_index < self.max_support_size-1

        # keep track of the last index that has been updated
        self.last_updated_idx = self._current_index

    # normalize the distribution to sum up to one
    def normalize(self):
        self.weights.mul_(1 / self.weights.sum())
        return self

    # perform self = (1-rho) * self + rho * other
    def convexAddSupportPointNoReplace(self, new_point, rho):

        # update the current index
        self._current_index = self._current_index + 1

        # if we went over the allowed size
        if self._current_index >= self.max_support_size:

            # set the growing flag to false
            self.is_growing = False

            # if it does have budget just reset the current index
            # otherwise throw a warning and do nothing
            if self.has_budget:
                self._current_index = 0
            else:
                warnings.warn("reached the maximum number of support points "
                              "limit")
                return

        if new_point.size(0) != self.support.size(1):
            warnings.warn("wrong dimensions!")
            return

        # add the new point to the full support
        self._full_support[self._current_index] = new_point.view(1, -1)

        # if we are updating a point (if we are using budget)
        # spread the weight across all other support points and then add the
        # new weight
        if self._full_weights[self._current_index] > 0:
            self._full_weights.add_(self._full_weights[self._current_index] /
                                    (self.support_size - 1))

        # multiply the current weights by 1-rho
        self.weights.mul_(1 - rho)

        # set the weight of the new point to rho
        self._full_weights[self._current_index] = rho

        # update the support_size if it is still smaller than the maximum
        # budget
        if self.support_size < self.max_support_size:
            self.support_size = self.support_size + 1

        # attach the pointers to the (possibly increased) current support and
        # weights
        self.support = self._full_support[:self.support_size]
        self.weights = self._full_weights[:self.support_size]

    def convexAddSupportPointReplace(self, new_point, rho):

        # update the current index
        self._current_index = self._current_index + 1

        # if we went over the allowed size
        if self._current_index >= self.max_support_size:

            # set the growing flag to false
            self.is_growing = False

            # if it does have budget just reset the current index
            # otherwise throw a warning and do nothing
            if self.has_budget:
                self._current_index = 0
            else:
                warnings.warn("reached the maximum number of support points "
                              "limit")
                return

        if new_point.size(0) != self.support.size(1):
            warnings.warn("wrong dimensions!")
            return

        # if we are updating a point (if we are using budget)
        # spread the weight across all other support points and then add the
        # new weight
        if self._full_weights[self._current_index] > 0:
            self._full_weights.add_(self._full_weights[self._current_index] /
                                    (self.support_size-1))

        # multiply the current weights by 1-rho
        self.weights.mul_(1 - rho)

        # check if the new point is already in the support
        tmp_min = (self.support - new_point).abs().sum(1).min(0)

        if tmp_min[0] > 0:

            # add the new point to the full support
            self._full_support[self._current_index] = new_point.view(1, -1)

            # set the weight of the new point to rho
            self._full_weights[self._current_index] = rho

            # update the support_size if it is still smaller than the maximum
            # budget
            if self.support_size < self.max_support_size:
                self.support_size = self.support_size + 1

                # attach the pointers to the (possibly increased) current
                # support and weights
                self.support = self._full_support[:self.support_size]
                self.weights = self._full_weights[:self.support_size]

            # keep track of the last index that has been updated
            self.last_updated_idx = self._current_index

        else:
            self.weights[tmp_min[1]] = self.weights[tmp_min[1]] + rho
            self._current_index = self._current_index - 1

            # keep track of the last index that has been updated
            self.last_updated_idx = tmp_min[1]

    def convexAddSupportPoint(self, new_point, rho):

        if self.do_replace:
            self.convexAddSupportPointReplace(new_point, rho)
        else:
            self.convexAddSupportPointNoReplace(new_point, rho)

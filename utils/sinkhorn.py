

import torch

# sinkhorn utility. From https://github.com/jeanfeydy/global-divergences
def lse(v_ij):
    V_i = torch.max(v_ij, 1)[0].view(-1, 1)
    return V_i + (v_ij - V_i).exp().sum(1).log().view(-1, 1)
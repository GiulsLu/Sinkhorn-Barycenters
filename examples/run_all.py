

# Only used to check if everything is working correctly

from run_gridbarycenter import *
from run_barycenter import *
from run_distribution import *


import torch
torch.set_default_tensor_type(torch.DoubleTensor)

testDistributions()
testBarycenter()
testFW()

testGridBarycenter()
testFirstFW()


print('Everything running smoothly!')



# Only used to check if everything is working correctly

import os
import sys

script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_path,'..'))


from testDistribution import *
from testBarycenter import *
from testGridBarycenter import *


import torch
torch.set_default_tensor_type(torch.DoubleTensor)

testDistributions()
testBarycenter()
testFW()

testGridBarycenter()
testFirstFW()


print('Everything running smoothly!')


from tests.testDistribution import *
from tests.testBarycenter import *
from tests.testGridBarycenter import *

#from tests.test_sinkhorn import test_repeat_support, compare_same_support_increase
#from testPropagator import *
# from testPropagator_k_neighbour_per_vertex import *

# from utils.load_meteodata_debug import *

# from tests.testKMeans import testKMeans
import torch
torch.set_default_tensor_type(torch.DoubleTensor)

testDistributions()
testBarycenter()
testFW()

testGridBarycenter()
testFirstFW()

#
# compareSinks()
#
# performFWTest()
#
# testGridBarycenter()

# testFirstFW()

# testDelteDiverse()

# testTwoEllipses()

#testEllipses()

# testMatch()

# testMatchSimple()

# testMatchEllipses()

# testGaussiane()

# testPropagator()

# testPropagatorMap()

# testSpeedPropagator()

# testKMeans()

# testProvideGridBarycenter()

# test_repeat_support()

# compare_same_support_increase()


#load_meteodata_debug()
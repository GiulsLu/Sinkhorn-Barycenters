"""Free support Sinkhorn barycenters via Frank-Wolf's algorithm."""
from .barycenter import Barycenter
from .gridbarycenter import GridBarycenter

from .distribution import Distribution
from .sinkhorn import sinkhorn_divergence
from . import utils


__all__ = ["Barycenter", "GridBarycenter", "Distribution",
           "sinkhorn_divergence", "utils"]

__version__ = '0.0.1dev'

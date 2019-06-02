
# (Mini) Documentation for Free-Support Sinkhorn Barycenters

### Main classes
- [`Distribution`](#distribution): stores a distribution with finite support. Its main elements are a matrix of `support` points and a vector of `weights`.
- [`Barycenter`](#barycenter): interface for different barycenter classes. At the moment they are:
    - [`GridBarycenter`](#grid-barycenter): constraining the barycenter to have support on a pre-specified grid (useful when dealing with images). 
    - [`SciPyBarycenter`](#scipy-barycenter): this class leverages a [SciPy Solver](link) to perform the inner minimization of the Frank-Wolfe algorithm.  

- [`Propagator`](#propagator): solver for the Sinkhorn propagation problem described [here]() and inspired by [(Solomon et al. 2014)](link)


<a name='barycenter'></a>
## Distribution
It takes the following input parameters:

- `support` **(mandatory)**: a torch n x d tensor storing the n d-dimensional support points of the distribution. 
- `weights` (default empty): a torch n-dimensional tensor containing the weights associated to the support points. Defaults to the vector of uniform weights.  

It currently implements only the method:
- `normalize()` which normalizes the vector of `weights`. 



<a name='barycenter'></a>
## Barycenter

This is an interface for actual barycenter computation. All children to this class have the following input parameters:

- `distributions` **(mandatory)**: the list of `Distribution`s we want to compute the barycenter of. 
- `bary` (default empty): initialization of the barycenter. Requires a `Distribution`
- `eps` (default 0.1): entropic regularization parametr for the Sinkhorn divergence. 
- `mixing_weights` (default uniform): vector of weights associated to the distributions to compute the barycenter. If not provided, the algorithm assumes same weight for each input distribution. 
 - `support_budget` (default 100): how many iterations of FW are we expecting to perform (it pre-allocates the memory required for the corresponding support)
 - `sinkhorn_tol` (default 1e-3): the tolerance for stopping the Sinkhorn-Knopp algorithm. 
 - `sinkhorn_n_itr` (default 100): the total number of iterations when performing the Sinkhorn-Knopp algorithm.

Additional input parameters are discussed for each child class independently. 

The main method is:
- `performFrankWolfe(num_itr = 1)` which performs `num_itr` iterations of FW on the distributions provided in input.  

The `Barycenter` class is only an interface to perform Sinkhorn barycenters and should not be used directly. Children of this class should overload the "virtual" method `_argminGradient`, performing the inner minimization required by the Frank-Wolfe algorithm proposed in [Sinkhorn Barycenters with Free Support via Frank-Wolfe Algorithm](link). Currently two such implementations are available. See below. 


<a name='grid-barycenter'></a>
### Grid Barycenter

This class takes as an additional input a grid from the user. If none is provided, it takes the smallest l_infty ball containing the union of the support sets of the input distributions and builds creates a grid on it. 

The following additional parameters are available:
- `grid` (default empty): grid provided in input by the user
- `grid_step` (default 50): the number of bins (per dimension) in which divide the l_infty ball. 
- `grid_margin_percentage` (default 0.05): consider a slightly larger l_infty ball than the one containing the union of the input distributions' supports. 

<a name='scipy-barycenter'></a>
### SciPy Barycenter (Coming Soon :wrench:)


<a name='propagator'></a>
## Propagator (Coming Soon :wrench:)

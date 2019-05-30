# Free-Support Sinkhorn Barycenters

This repository complements the paper [Sinkhorn Barycenters with Free Support via Frank-Wolfe Algorithm](link-to-arxiv-paper) with an implementation of the proposed algorithm to compute the Barycenter of multiple probability measures with respect to the [Sinkhorn Divergence](genevay).

We provide the code to reproduce most of the experiments in the paper. We recommend running all experiments on GPU.  

**If you are interested in using the proposed algorithm in your projects please refer to the instructions [here](link-to-other-readme).**

### Dependencies
The only core dependencies are:
- [PyTorch](link-to-pytorch)
- [Matplotlib](link-to-matplotlib)

For some experiments we also have the following additional dependencies:
- [Folio](link)
- [MNIST](link)


### List of Experiments
Below we describe how to reproduce the experiments in the original paper:
- [Nested Ellipses](#ellipses)
- [Barycenter of Continuous Measures](#continuous-measures)
- [Distribution Matching](#matching)
- [k-Means](#k-means)
- [Sinkhorn Propagation](#propagation)


## A Classic: Barycenter of Nested Ellipes

We compute the barycenter of 30 randomly generated nested ellipses on a 50 × 50 pixels image, similarly to [(Cuturi and Doucet 2014)](link). We interpret each image as a probability distribution in 2D. The cost matrix is given by the squared Euclidean distances between pixels. The fiture reports 8 samples of the input ellipses (all examples can be found in the folder `data/ellipses` and the barycenter obtained with the proposed algorithm in the middle. It shows qualitatively that our approach captures key geometric properties of the input measures.

Run: 
```sh
$ python experiments/ellipses.py
```

Output in folder `out/ellipses`



## Continuous Measures: Barycenter of Gaussian Distributions 

We compute the barycenter of 5 Gaussian distributions with mean and covariance matrix randomly generated. We apply to empirical measures obtained by sampling n = 500 points from each one. Since the (Wasserstein) barycenter of Gaussian distributions can be estimated accurately (see [(A)](link)), in the figure we report both the output of the proposed algorithm (as a scatter plot) and the true Wasserstein barycenter (as level sets of its density). We observe that our estimator recovers both the mean and covariance of the target barycenter. 

Run: 
```sh
$ python experiments/gaussians.py
```

Output in folder `out/gaussians`

Instructions for additional experiments and parameters can be found directly in the file `experiments/gaussians.py`.


## Distribution Matching

Similarly to [(Claici??)](link), we test the proposed algorithm in the special case where we are computing the “barycenter” of a single measure (rather than multiple ones). While the solution of this problem is the input distribution itself, we can interpret the intermediate iterates the proposed algorithm as compressed version of the original measure. In this sense the iteration number `k` represents the level of compression since the corresponding barycenter estimate is supported on at most `k` points. The figure (Right) reports iteration `k` = 5000 for the proposed algorithm applied to the 140 × 140 image in (Left) interpreted as a probability measure in 2D. We note that the number of points in the support is ∼3900: the most relevant support points are selected multiple times to accumulate the right amount of mass on each of them (darker color = higher weight). This shows that the proposed approach tends to greedily search for the most relevant support points, prioritizing those with higher weight. 

Run: 
```sh
$ python experiments/matching.py
```

Output in folder `out/matching`

The code can be run with any image by passing the path to the desired image as additional argument. 


## Sinkhorn k-Means Clustering 

We test the proposed algorithm on a k-means clustering experiment. We consider a subset of 500 random images from the [MNIST dataset](link-to-mnist). Each image is suitably normalized to be interpreted as a probability distribution on the grid of 28 × 28 pixels with values scaled between 0 and 1. We initialize 20 centroids according to the [k-means++](link) strategy. The figure deipcts the corresponding 20 centroids obtained throughout this process. We see that the structure of the digits is successfully detected, recovering also minor details (e.g. note the difference between the 2 centroids).

Run: 
```sh
$ python experiments/kmeans.py
```

Output in folder `out/kmeans`


## Sinkhorn Propagation

We consider the problem of Sinkhorn propagation similar to the Wasserstein propagation in [(Solomon et al. 2014)](link). The goal is to predict the distribution of missing measurements for weather stations in the state of Texas, US (data from [website](link))) by “propagating” measurements from neighboring stations in the network. The problem can be formulated as minimizing the functional `missing` over the set `missing` with: `missing` the subset of stations with missing measurements, `missing` the whole graph of the stations network, `missing` a weight inversely proportional to the geographical distance between two vertices/stations `missing`. The variable `missing` denotes the distribution of measurements at station `missing` of daily temperature and atmospheric pressure over one year. This is a generalization of the barycenter problem. From the total `missing`, we randomly select 10%, 20% or 30% to be available stations, and use the proposed algorithm to propagate their measurements to the remaining “missing” ones. We compare our approach (FW) with the Dirichlet (DR) baseline in [(Solomon et al)](link) in terms of the error `missing` between the covariance matrix `missing` of the  ground truth distribution and that of the predicted one. Here `missing` is the geodesic distance on the cone of positive definite matrices. In the figure we qualitatively report the improvement `missing` of our method on individual stations: a higher color intensity corresponds to a wider gap in our favor between prediction errors, from light green `missing` to red `missing`. Our approach tends to propagate the distributions to missing locations with higher accuracy.

Run: 
```sh
$ python experiments/propagation.py
```

Output in folder `out/propagation`




## References

- [our](our)
- [feydy](feydy)
- [genevay16](genevay16)
- [cuturi14](cuturi)

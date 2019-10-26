# Free-Support Sinkhorn Barycenters

This repository complements the paper [Sinkhorn Barycenters with Free Support via Frank-Wolfe Algorithm](https://arxiv.org/pdf/1905.13194.pdf) with an implementation of the proposed algorithm to compute the Barycenter of multiple probability measures with respect to the [Sinkhorn Divergence](https://arxiv.org/pdf/1706.00292.pdf).

**If you are interested in using the proposed algorithm in your projects please refer to the instructions [here](./documentation.md).**

Below we provide the code and instructions to reproduce most of the experiments in the paper. We recommend running all experiments on GPU.  




### List of Experiments
- [Nested Ellipses](#ellipses)
- [Barycenter of Continuous Measures](#continuous-measures)(Coming soon :wrench:)
- [Distribution Matching](#matching)
- [k-Means](#k-means)
- [Sinkhorn Propagation](#propagation)(Coming soon :wrench:)


### Dependencies
The core dependencies are:
- [PyTorch](https://pytorch.org/)
- [Matplotlib](https://matplotlib.org/)

For some experiments we also have the following additional dependencies:
- [NumPy](https://www.numpy.org/)
- [Pillow](https://pillow.readthedocs.io/en/stable/index.html)
- [Folium](https://python-visualization.github.io/folium/)


<a name='ellipses'></a>
## A Classic: Barycenter of Nested Ellipes


<img align='right' style='' src="https://giulslu.github.io/Sinkhorn-Barycenters/git_data/git_images/ellipses.png" width="20%">


We compute the barycenter of 30 randomly generated nested ellipses on a 50 × 50 pixels image, similarly to [(Cuturi and Doucet 2014)](https://arxiv.org/pdf/1310.4375.pdf). We interpret each image as a probability distribution in 2D. The cost matrix is given by the squared Euclidean distances between pixels. The fiture reports 8 samples of the input ellipses (all examples can be found in the folder `data/ellipses` and the barycenter obtained with the proposed algorithm in the middle. It shows qualitatively that our approach captures key geometric properties of the input measures.

<img align='right' style='' src="https://giulslu.github.io/Sinkhorn-Barycenters/git_data/git_images/nested_ellipses.gif" width="20%">

**Run:** 
```sh
$ python experiments/ellipses.py
```

**Output** in folder `out/ellipses`



## Continuous Measures: Barycenter of Gaussian Distributions (Coming Soon) 

<p>
<img align='left' style='' src="https://giulslu.github.io/Sinkhorn-Barycenters/git_data/git_images/gauss1.png" width="20%">
<p>
<p>
<img align='left' style='' src="https://giulslu.github.io/Sinkhorn-Barycenters/git_data/git_images/gauss2.png" width="20%">
</p>

We compute the barycenter of 5 Gaussian distributions with mean and covariance matrix randomly generated. We apply to empirical measures obtained by sampling n = 500 points from each one. Since the (Wasserstein) barycenter of Gaussian distributions can be estimated accurately [(Cuturi and Doucet 2014)](https://arxiv.org/pdf/1310.4375.pdf) (see [(Agueh and Carlier 2011)](https://www.ceremade.dauphine.fr/~carlier/AC_bary_Aug11_10.pdf)), in the figure we report both the output of the proposed algorithm (as a scatter plot) and the true Wasserstein barycenter (as level sets of its density). We observe that our estimator recovers both the mean and covariance of the target barycenter. 



**Run:**
```sh
$ python experiments/gaussians.py
```

**Output** in folder `out/gaussians`

Instructions for additional experiments and parameters can be found directly in the file `experiments/gaussians.py`.

<a name='matching'></a>
## Distribution Matching


<img align='right' style='' src="https://giulslu.github.io/Sinkhorn-Barycenters/git_data/git_images/cheetah.gif" width="30%">
<img align='right' style='' src="https://giulslu.github.io/Sinkhorn-Barycenters/git_data/git_images/cheetah_orig.png" width="20%">



Similarly to [(Claici et al. 2018)](https://arxiv.org/pdf/1802.05757.pdf), we test the proposed algorithm in the special case where we are computing the “barycenter” of a single measure (rather than multiple ones). While the solution of this problem is the input distribution itself, we can interpret the intermediate iterates the proposed algorithm as compressed version of the original measure. In this sense the iteration number `k` represents the level of compression since the corresponding barycenter estimate is supported on at most `k` points. The figure (Right) reports iteration `k` = 5000 for the proposed algorithm applied to the 140 × 140 image in (Left) interpreted as a probability measure in 2D. We note that the number of points in the support is ∼3900: the most relevant support points are selected multiple times to accumulate the right amount of mass on each of them (darker color = higher weight). This shows that the proposed approach tends to greedily search for the most relevant support points, prioritizing those with higher weight. 

**Run:**
```sh
$ python experiments/matching.py
```

**Output** in folder `out/matching`

The code can be run with any image by passing the path to the desired image as additional argument. 


<a name='k-means'></a>
## Sinkhorn k-Means Clustering 

<img align='left' style='' src="https://giulslu.github.io/Sinkhorn-Barycenters/git_data/git_images/kmeans.png" width="30%">

We test the proposed algorithm on a k-means clustering experiment. We consider a subset of 500 random images from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). Each image is suitably normalized to be interpreted as a probability distribution on the grid of 28 × 28 pixels with values scaled between 0 and 1. We initialize 20 centroids according to the [k-means++](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf) strategy. The figure deipcts the corresponding 20 centroids obtained throughout this process. We see that the structure of the digits is successfully detected, recovering also minor details (e.g. note the difference between the 2 centroids).

**Run:**
```sh
$ python experiments/kmeans.py
```

**Output** in folder `out/kmeans`



<a name='propagation'></a>
## Sinkhorn Propagation (Coming Soon :wrench:)

<p align='center'>
<img style='' src="https://giulslu.github.io/Sinkhorn-Barycenters/git_data/git_images/propagation-10.png" width="25%">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img style='' src="https://giulslu.github.io/Sinkhorn-Barycenters/git_data/git_images/propagation-20.png" width="25%">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img style='' src="https://giulslu.github.io/Sinkhorn-Barycenters/git_data/git_images/propagation-30.png" width="25%">
</p>

We consider the problem of Sinkhorn propagation similar to the Wasserstein propagation in [(Solomon et al. 2014)](http://proceedings.mlr.press/v32/solomon14.pdf). The goal is to predict the distribution of missing measurements for weather stations in the state of Texas, US (data from [National Climatic Weather Data](http://tiny.cc/0p2o7y)) by “propagating” measurements from neighboring stations in the network. The problem can be formulated as minimizing the functional 

<p align="center"><img alt="$$&#10;\sum_{(v,u)\in\mathcal{V}} \omega_{uv}~\mathsf{S}_\varepsilon(\rho_v,\rho_u)&#10;$$" src="https://giulslu.github.io/Sinkhorn-Barycenters/git_data/svgs/aa468d0e2371e0652a7391f0c74cac50.svg" valign=0.0px width="150.73620044999998pt" height="40.548151049999994pt"/></p>

over the set <img alt="$\{\rho_v\in\mathcal{M}_1^+(\mathbb{R}^2) | v\in\mathcal{V}_0\}$" src="https://giulslu.github.io/Sinkhorn-Barycenters/git_data/svgs/391a77d97e59f4dd9addb721b100aafd.svg" valign=-4.383446099999994px width="166.18759905pt" height="18.538702049999998pt"/> with: <img alt="$\mathcal{V}_0\subset\mathcal{V}$" src="https://giulslu.github.io/Sinkhorn-Barycenters/git_data/svgs/bfb0700e6838201d41a064d16059b2ad.svg" valign=-2.4657286499999893px width="50.789958449999986pt" height="13.698590399999999pt"/> the subset of stations with missing measurements, <img alt="$G = (\mathcal{V},\mathcal{E})$" src="https://giulslu.github.io/Sinkhorn-Barycenters/git_data/svgs/b2b3927a88f310b4bfe12b3ba87cc443.svg" valign=-4.109589000000009px width="76.50444284999999pt" height="16.438356pt"/> the whole graph of the stations network, <img alt="$\omega_{uv}$" src="https://giulslu.github.io/Sinkhorn-Barycenters/git_data/svgs/0f8cb2a808663d1801aeb4dae8ebe347.svg" valign=-2.4657286499999893px width="24.992543399999988pt" height="9.54335085pt"/> a weight inversely proportional to the geographical distance between two vertices/stations <img alt="$u,v\in\mathcal{V}$" src="https://giulslu.github.io/Sinkhorn-Barycenters/git_data/svgs/4e1127b951d3a091bef7c89c4eed81ae.svg" valign=-3.196350299999991px width="56.789866649999986pt" height="14.42921205pt"/>. The variable <img alt="$\rho_v\in\mathcal{M}_1^+(\mathbb{R}^2)$" src="https://giulslu.github.io/Sinkhorn-Barycenters/git_data/svgs/36cba46e9101d17a14f9e8de4fe7774b.svg" valign=-4.383446099999994px width="99.08638079999999pt" height="18.538702049999998pt"/> denotes the distribution of measurements at station <img alt="$v$" src="https://giulslu.github.io/Sinkhorn-Barycenters/git_data/svgs/6c4adbc36120d62b98deef2a20d5d303.svg" valign=0.0px width="8.55786029999999pt" height="7.0776222pt"/> of daily temperature and atmospheric pressure over one year. This is a generalization of the barycenter problem. From the total <img alt="$|\mathcal{V}|$" src="https://giulslu.github.io/Sinkhorn-Barycenters/git_data/svgs/d63ce73b5e521a58add11bec71b45364.svg" valign=-4.109589000000009px width="20.55717839999999pt" height="16.438356pt"/>=115, we randomly select 10%, 20% or 30% to be available stations, and use the proposed algorithm to propagate their measurements to the remaining “missing” ones. We compare our approach (FW) with the Dirichlet (DR) baseline in [(Solomon et al. 2014)](http://proceedings.mlr.press/v32/solomon14.pdf) in terms of the error <img alt="$d(C_T,\hat C)$" src="https://giulslu.github.io/Sinkhorn-Barycenters/git_data/svgs/41743dfa72e5c4b72ebc8451ff35d059.svg" valign=-4.109589000000006px width="63.67635944999999pt" height="19.68035685pt"/> between the covariance matrix <img alt="$C_T$" src="https://giulslu.github.io/Sinkhorn-Barycenters/git_data/svgs/7e0cae7fd38b427ac706ff18bbb4ecee.svg" valign=-2.4657286499999893px width="21.28256789999999pt" height="13.698590399999999pt"/> of the  ground truth distribution and that of the predicted one. Here <img alt="$d(A,B) = \|\log(A^{-1/2} B A^{-1/2})\|$" src="https://giulslu.github.io/Sinkhorn-Barycenters/git_data/svgs/5cc0d96242fd7f04e8f753a57fb0bb15.svg" valign=-4.109588999999997px width="229.20687569999996pt" height="18.7050765pt"/> is the geodesic distance on the cone of positive definite matrices. In the figures above we qualitatively report the improvement <img alt="$\Delta = d(C_T,C_{DR}) - d(C_T,C_{FW})$" src="https://giulslu.github.io/Sinkhorn-Barycenters/git_data/svgs/f68b383e11f7d98852f8dfe0e354edd2.svg" valign=-4.109589000000009px width="227.65502264999995pt" height="16.438356pt"/> of our method on individual stations: a higher color intensity corresponds to a wider gap in our favor between prediction errors, from light green <img alt="$(\Delta\sim 0)$" src="https://giulslu.github.io/Sinkhorn-Barycenters/git_data/svgs/46134ebe19fd6d202e5952c6474c0c9b.svg" valign=-4.109589000000009px width="56.62094624999999pt" height="16.438356pt"/> to red <img alt="$(\Delta\sim 2)$" src="https://giulslu.github.io/Sinkhorn-Barycenters/git_data/svgs/97ca6c52e2d299e85d1ff0f791376b63.svg" valign=-4.109589000000009px width="56.62094624999999pt" height="16.438356pt"/>. Our approach tends to propagate the distributions to missing locations with higher accuracy.

**Run:**
```sh
$ python experiments/propagation.py
```

**Output** in folder `out/propagation`



## References

- **(this work)** G. Luise, S. Salzo, M. Pontil, C. Ciliberto. [_Sinkhorn Barycenters with Free Support via Frank-Wolfe Algorithm_](https://arxiv.org/pdf/1905.13194.pdf) arXiv preprint arxiv:1905.13194, 2019
- J. Feydy, T. Séjourné, F.X. Vialard, S.I. Amari, A. Trouvé, G. Peyré. [_Interpolating between optimal transport and mmd using sinkhorn divergences._](https://arxiv.org/pdf/1810.08278.pdf) International Conference on Artificial Intelligence and Statistics (AIStats), 2019.
- A. Genevay, G. Peyré, M. Cuturi. [_Learning generative models with sinkhorndivergences._](https://arxiv.org/pdf/1706.00292.pdf) International Conference on Artificial Intelligence and Statistics (AIStats), 2018.

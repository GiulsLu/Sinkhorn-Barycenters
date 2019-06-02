import torch
#import pyro
import seaborn as sns
import numpy as np
import time
import scipy
import scipy.linalg as sla
from utils import *
#from plot_utils import *
torch.set_default_tensor_type(torch.DoubleTensor)
from optimizers import *
from utils_glob_divergences.sinkhorn_balanced import *

functional_barycenter2 = lambda _xx, _weights, _supp_meas, _weights_meas : sum([w[j]*sinkhorn_divergence(_weights, _xx, _weights_meas[j].view(-1,1), _supp_meas[j],p=2, eps= eps) for j in range(len(supp_meas))])

def pre_histogram(points, weights):
    n = points.shape[0]
    x = []
    y = []
    for i in range(n):
        x.append(points[i][0])
        y.append(points[i][1])
    return np.array(x), np.array(y), weights.reshape(weights.shape[0])

def plot(point, wei, bins=100, thresh=0):
    tmp_plot = torch.clone(wei)
    tmp_plot[tmp_plot<=thresh] = 0

    xb, yb, wb = pre_histogram(point, tmp_plot)
    # plt.figure()
    HIST = plt.hist2d(xb, yb, bins=[bins, bins], weights=wb)
    return xb, yb, wb, HIST

def scatterplot(p, w):
    HIST = plot(p, w)[3]
    tmp_meshgrid = torch.meshgrid(torch.tensor(HIST[1][1:]), torch.tensor(HIST[2][1:]))

    grid = torch.cat([mesh_column.reshape(-1, 1) for mesh_column in tmp_meshgrid], dim=1)

    c = HIST[0].reshape(HIST[1][1:].shape[0]**2)
    plt.scatter(grid[:,0],grid[:,1],c=c,cmap='Blues',s=5)
    plt.show()

def covariance_bary(n_its, covariances,weights):
    num_meas = weights.shape[0]
    cov = np.random.rand(2, 2)
    i=0
    thresh=1e-7
    err = 10
    while i < n_its and err> thresh:
        prev_cov = cov
        sqrtm_cov = sla.sqrtm(cov)
        inv_sqrtm_cov = np.linalg.inv(sla.sqrtm(cov))
        A = np.zeros((2,2))
        for j in range(num_meas):
            A = A + weights[j]*sla.sqrtm(sqrtm_cov@covariances[j]@sqrtm_cov)
        cov = inv_sqrtm_cov @ (A@A) @ inv_sqrtm_cov
        err= np.linalg.norm(prev_cov-cov)
        i=i+1
    return cov

def sample_quick(means, var, K, rel_weights, ns, d):
    N = torch.zeros(ns,d)
    m = torch.distributions.normal.Normal(torch.tensor([0.]), torch.tensor([0.08]), validate_args=None)
    N[:,0] = m.sample(sample_shape=torch.Size([ns])).squeeze(1)
    N[:, 1] = m.sample(sample_shape=torch.Size([ns])).squeeze(1)
    choices = torch.tensor(np.random.choice(K, ns, p = rel_weights), dtype = torch.uint8)
    sample = torch.zeros(ns,d)
    for i in range(K):
        sample[choices ==i, :] = N[choices == i,:] + means[i,:].unsqueeze(0)
    return sample


def sample(distrib, size, d):
    sample = torch.zeros(size,d)
    for i in range(size):
        sample[i,:] = distrib.sample()

    return sample

def dist_matrix(x_i, y_j, p,ε) :
    x_y = x_i.unsqueeze(1) - y_j.unsqueeze(0)
    if   p == 1 : return   x_y.norm(dim=2)  / ε
    elif p == 2 :
        #print( ( x_y ** 2).sum(2) / ε)
        return ( x_y ** 2).sum(2) / ε
    else : return   x_y.norm(dim=2)**(p/2) / ε


def barycenters_sample_dens(measures, eps, niter, h, w,d):
    p_wass = 2
    weights = torch.zeros((niter + h,1))
    xx = torch.zeros((niter + h,d))

    ### UNIFORM INITIALIZATION ###
    uni1 = torch.distributions.uniform.Uniform(0, 1)
    x_init = uni1.sample(sample_shape=torch.Size([h]))
    y_init = uni1.sample(sample_shape=torch.Size([h]))
    init = torch.zeros((h, d))
    init[:, 0] = x_init
    if d==2:
        init[:, 1] = y_init
    xx[0:h,:] = init
    xx[0:h, :] = torch.tensor([0.3, 0.3])
    weights[0:h] = 1/h
    # 'tnp' means Total Number of Points
    # 'npem' means Number of Points of Each Measure

    ### COMPUTE ALL DISTANCE MATRICES on the GRID ###
    #x_grid, y_grid, C_grid_y, pre_C_grid_x,  all_points, tnp, npem, array_grid  = kernel_distance_on_grid(supp_meas, xx[0:h], start, end, steps, 2, eps)
    C_xx = torch.zeros((niter + h, niter + h))
    C_xx[0:h , 0:h] = dist_matrix(xx[0:h].reshape(h , d), xx[0:h].reshape(h, d), 2, eps)
    time1=time.time()
    for k in np.arange(1,niter-1):

        #### SAMPLE FROM THE MEASURES ####
        num_meas = len(measures)
        supp_meas = []
        weights_meas = []
        ns = 500
        t1= time.time()
        func=[]
        for j in range(num_meas):
            supp_meas.append(sample(measures[j], ns,d))
            weights_meas.append(torch.ones(ns)/ns)
        t2 = time.time()
        print('time camp', t2-t1)
        num_points_supp_each_meas = []
        for i in range(len(supp_meas)):
            num_points_supp_each_meas.append(supp_meas[i].shape[0])

        tot_num_points = sum(num_points_supp_each_meas)
        all_points = torch.cat([supp_meas[i] for i in range(len(supp_meas))])
        npem = torch.zeros(len(num_points_supp_each_meas) + 1, dtype=torch.int32)
        npem[1:] = torch.tensor(num_points_supp_each_meas)
        tnp = sum(npem)
        # wm = torch.stack(weights_meas)
        wm = torch.cat([weights_meas[i] for i in range(len(supp_meas))])

        wm = wm.reshape(tnp)




        ### START FILLING IN THE MATRIX WITH CURRENT DATA
        C_xy = dist_matrix(xx[0:h+k-1].reshape(h+k-1, d), all_points.reshape(tnp, d), 2, eps)

        print(k)
        t1 = time.time()
        autocorr, v = grad_of_functional_givenC2(weights[0:h+k-1],  weights_meas, C_xy, C_xx[0:h+k-1,0:h+k-1], npem, tnp, d, eps)

        t2 = time.time()
        print('time_potentials', t2-t1)

        x, wei, gamma = franke_wolfe_step_scipy(xx[0:h + k - 1, :], weights[0:h + k - 1],tnp, npem, autocorr, v,all_points, wm,
                                               k, eps, w, d, 20)



                #### update the barycenter with the new point and weights
        xx[h+k-1] = x
        print(xx[h+k-1])
        weights[0:h+k-1] = wei
        weights[h+k-1] = gamma

                #### now that we have a new point, update all the distance matrices

        C_xx[0:h + k-1 ,h + k-1 ] = dist_matrix(xx[h+k-1].reshape(1,d),xx[0:h+k-1].reshape((h+k-1,d)) , 2, eps)
        C_xx[h + k-1 , 0:h + k -1] = C_xx[0:h + k-1 , h + k-1 ]
        print('next')
        func.append(print(functional_barycenter2(xx[0:h+k-1], weights[0:h+k-1], supp_meas, weights_meas)))
        if k%100==0:
            # w = w.reshape(2500)
            # idx = w.nonzero()[0]
            # c = w[idx]
            # plt.scatter(grid[idx,0],grid[idx,1],c=c, cmap = 'Blues', s=5)

            plt.figure()
            plot(xx,weights, bins = 100)
        # dictionary with weighted points
    time2 = time.time()
    print('elapsed time', time2-time1)
    print(functional_barycenter2(xx, weights, supp_meas, weights_meas))
    return xx, weights


def barycenters_sample_dens2(means, variances, rel_weights, K, eps, niter, h, w,d):
    p_wass = 2
    weights = torch.zeros((niter + h,1))
    xx = torch.zeros((niter + h,d))

    ### UNIFORM INITIALIZATION ###
    uni1 = torch.distributions.uniform.Uniform(0, 1)
    x_init = uni1.sample(sample_shape=torch.Size([h]))
    y_init = uni1.sample(sample_shape=torch.Size([h]))
    init = torch.zeros((h, d))
    init[:, 0] = x_init
    if d==2:
        init[:, 1] = y_init
    xx[0:h,:] = init
    xx[0:h, :] = torch.tensor([0.3, 0.3])
    weights[0:h] = 1/h
    # 'tnp' means Total Number of Points
    # 'npem' means Number of Points of Each Measure

    ### COMPUTE ALL DISTANCE MATRICES on the GRID ###
    #x_grid, y_grid, C_grid_y, pre_C_grid_x,  all_points, tnp, npem, array_grid  = kernel_distance_on_grid(supp_meas, xx[0:h], start, end, steps, 2, eps)
    C_xx = torch.zeros((niter + h, niter + h))
    C_xx[0:h , 0:h] = dist_matrix(xx[0:h].reshape(h , d), xx[0:h].reshape(h, d), 2, eps)
    time1=time.time()
    func=[]

    functional_barycenter2 = lambda _xx, _weights, _supp_meas, _weights_meas: sum(
        [w[j] * sinkhorn_divergence(_weights, _xx, _weights_meas[j].view(-1, 1), _supp_meas[j], p=2, eps=eps) for j
         in range(len(supp_meas))])


    for k in np.arange(1,niter-1):

        #### SAMPLE FROM THE MEASURES ####
        num_meas = len(means)
        supp_meas = []
        weights_meas = []
        ns = 500
        t1= time.time()
        for j in range(num_meas):
            sample = np.random.multivariate_normal(means[j], variances[j],ns)
            supp_meas.append(torch.tensor(sample, dtype=torch.float64))
            weights_meas.append(torch.ones(ns)/ns)
        t2 = time.time()
        print('time camp', t2-t1)
        #########

        # xb, yb, wb = pre_histogram(supp_meas[3], weights_meas[3])
        # # Basic 2D density plot
        # sns.set_style("white")
        # # sns.kdeplot(xb, yb)
        #
        # # Custom it with the same argument as 1D density plot
        # sns.kdeplot(xb, yb, cmap="Reds", shade=True, bw=.15)
        # plt.show()
        # # Some features are characteristic of 2D: color palette and wether or not color the lowest range
        # sns.kdeplot(xb, yb, cmap="Blues", shade=True, shade_lowest=True, )
        ##########
        #scatterplot(supp_meas[0],weights_meas[0])
        num_points_supp_each_meas = []
        for i in range(len(supp_meas)):
            num_points_supp_each_meas.append(supp_meas[i].shape[0])

        tot_num_points = sum(num_points_supp_each_meas)
        all_points = torch.cat([supp_meas[i] for i in range(len(supp_meas))])
        npem = torch.zeros(len(num_points_supp_each_meas) + 1, dtype=torch.int32)
        npem[1:] = torch.tensor(num_points_supp_each_meas)
        tnp = sum(npem)
        # wm = torch.stack(weights_meas)
        wm = torch.cat([weights_meas[i] for i in range(len(supp_meas))])

        wm = wm.reshape(tnp)


        ### START FILLING IN THE MATRIX WITH CURRENT DATA
        C_xy = dist_matrix(xx[0:h+k-1].reshape(h+k-1, d), all_points.reshape(tnp, d), 2, eps)

        print(k)
        t1 = time.time()
        autocorr, v = grad_of_functional_givenC2(weights[0:h+k-1],  weights_meas, C_xy, C_xx[0:h+k-1,0:h+k-1], npem, tnp, d, eps)

        t2 = time.time()
        print('time_potentials', t2-t1)

        x, wei, gamma = franke_wolfe_step_scipy(xx[0:h + k - 1, :], weights[0:h + k - 1],tnp, npem, autocorr, v,all_points, wm,
                                               k, eps, w, d, 10)



                #### update the barycenter with the new point and weights
        xx[h+k-1] = x
        print(xx[h+k-1])
        weights[0:h+k-1] = wei
        weights[h+k-1] = gamma

                #### now that we have a new point, update all the distance matrices

        C_xx[0:h + k-1 ,h + k-1 ] = dist_matrix(xx[h+k-1].reshape(1,d),xx[0:h+k-1].reshape((h+k-1,d)) , 2, eps)
        C_xx[h + k-1 , 0:h + k -1] = C_xx[0:h + k-1 , h + k-1 ]
        print('next')




        if k%500==0:
            #plt.plot(func)
            #plot(xx,weights, bins = 100)
            idx = weights[:,0].nonzero()
            idx = idx.reshape(idx.shape[0])
            c = weights[idx][:,0]
            plt.scatter(xx[idx, 0], xx[idx, 1], c=c, cmap='Blues', s=5)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.show()
        t = time.time()
        func.append(functional_barycenter2(xx[0:h + k - 1], weights[0:h + k - 1], supp_meas, weights_meas))
        t2 = time.time()
        print('time func', t2-t)
        # dictionary with weighted points
    time2 = time.time()
    print('elapsed time', time2-time1)
    print(functional_barycenter2(xx, weights, supp_meas, weights_meas))
    return xx, weights



num_meas = 5
measures = []
means = []
variances = []
rel_weights = []
for j in range(num_meas):
    means.append(torch.tensor([j * 0.5, 0.5]))
    var = 0.08*np.eye(2)
    var = var @ var.T
    variances.append(var)

rel_weights = torch.ones(num_meas)
rel_weights = rel_weights/sum(rel_weights)

#mix_gauss = pyro.distributions.MixtureOfDiagNormalsSharedCovariance(mean_coo, var, weights_mix)

eps = 0.005
niter = 3000
w = torch.ones(num_meas)/num_meas
d=2
#barycenters_sample_dens(measures, eps, niter, 1,w,d)
K=2
means = []
variances = []
#barycenters_sample_dens2(means,variances, rel_weights,K, eps, niter, 1,w,d)
for j in range(num_meas):
    means.append(torch.rand(2))
    var = torch.rand(2,2)/10
    var = var @ var.transpose(0,1)
    variances.append(var.numpy())


barycenters_sample_dens2(means,variances, rel_weights,K, eps, niter, 1,w,d)
means_bary = np.zeros(2)
for i in range(num_meas):
    means_bary = means_bary + w[i].numpy()*means[i].numpy()

cov_bary = covariance_bary(100,variances, w.numpy())

sample = np.random.multivariate_normal(means_bary, cov_bary, 500)
xb_t,yb_t,wb_t = pre_histogram(sample, torch.ones(500)/500)
sns.kdeplot(xb_t,yb_t, cmap="Blues", shade=True, shade_lowest=True, )
plt.xlim(0,1)
plt.ylim(0,1)
print('hello')
#plt.scatter(supp_meas[4][:,0].numpy(), supp_meas[4][:,1])
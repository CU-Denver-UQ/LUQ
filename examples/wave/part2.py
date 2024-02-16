import numpy as np
from luq.luq import *
from scipy.stats import norm, beta
from scipy.stats import gaussian_kde as GKDE
from scipy.integrate import quadrature
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
import ipywidgets as wd

# colorblind friendly color palette
c = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
# setup fontsizes for plots
plt_params = {'legend.fontsize': 14,
          'figure.figsize': (6.4, 4.8),
         'axes.labelsize': 16,
         'axes.titlesize': 16,
         'xtick.labelsize': 14,
         'ytick.labelsize': 14}
plt.rcParams.update(plt_params)


# generating samples

# parameter samples for construction of pi_obs
num_obs_samples = 200
np.random.seed(12345678)
params_obs = np.vstack([2 * np.random.beta(a=2, b=5, size=num_obs_samples) + 1,
                         np.random.normal(loc=2.5, scale=0.5, size=num_obs_samples)]) # unknown data-generated parameters corresponding to observed samples

# parameter samples of pi_init
num_samples = int(1E3)
np.random.seed(123456)
params = np.random.uniform(low=0.0,high=5.0,size=(2,num_samples)) # uniformly distributed parameter samples

# finite-difference scheme

# defining model solve function
dx = 0.05
dy = 0.05
dt = 0.005 # satifies CFL condition

xn = np.linspace(0,5.0,101) # 101 = length in x / dx
ym = np.linspace(0,5.0,101)
tk = np.linspace(0,7.0,1401) # 1401 = length in t / dt

# defining model solve on 101x101 uniform mesh of [0,5]^2 for t = 0 to t = 7 with dt = 0.005
def M(a,b):
    # initializing the model solution
    # using Dirichlet boundary conditions,so initializing with zeros means boundary values are set
    u = np.zeros((101,101,1401))
    
    # iterate through times; t here is equivalent to time and time index
    for t in range(1401):
        
        # if t = 0, use initial condition modeling wave droplet
        if t == 0:
            mesh = np.meshgrid(xn[1:-1],ym[1:-1])
            u[1:-1,1:-1,t] = 0.2*np.exp(-10*((mesh[0].T-a)**2+(mesh[1].T-b)**2))
        
        # else solve model using finite-difference scheme
        else:
            u[1:-1,1:-1,t] = 2 * u[1:-1,1:-1,t-1] - u[1:-1,1:-1,max(0,t-2)] \
                + dt**2 / dx**2 * (u[2:,1:-1,t-1] - 2 * u[1:-1,1:-1,t-1] + u[:-2,1:-1,t-1]) \
                + dt**2 / dy**2 * (u[1:-1,2:,t-1] - 2 * u[1:-1,1:-1,t-1] + u[1:-1,:-2,t-1])
    return u

# indexing for extracting data on different grid sizes

# indexing function for flattening data
def idx_at(x,y):
    idx = []
    idx.append((x / dx).astype(int))
    idx.append((y / dy).astype(int))
    return idx

# using indexing function to extract data on uniformly-spaced mesh given by delta
def create_idx(delta):
    N = (5-delta)/delta 
    # note: only delta such that (5-delta)/delta is int can be used (or does not change value when cast as int) 
    # any other delta value requires extrapolation
    pts = np.linspace(delta,5-delta,int(N))
    grid_pts = np.meshgrid(pts,pts)
    idx = idx_at(grid_pts[0],grid_pts[1])
    return [idx[0].flatten(), idx[1].flatten()]

# loading pre-computed observed data
obs = np.load('dg_samples/obs_clean', allow_pickle=True)

# predicted data samples on 9x9 grid, 0.5 mesh size
pred = np.zeros((num_samples,9**2,14))
idx = create_idx(0.5)
for i in range(num_samples):
    tmp = M(params[0,i], params[1,i])
    pred[i,:,:] = tmp[idx[0],idx[1],100::100]
    print(f'Predicted sample {i} done.')

# extracting time series data
obs_time_series = obs[:,16,:]
pred_time_series = pred[:,16,:]


# learning QoI map using LUQ

# instantiating luq
learn = LUQ(pred_time_series,
             obs_time_series)

# learning 2 QoI's from data using kernel pca and transforming the data into QoI samples
pred_maps, obs_maps = learn.learn_qois_and_transform(num_qoi=2)


# computing a DCI solution

# generate kernel density estimates on new QoI and calculate new weights
pi_predict_kdes = []
pi_obs_kdes = []
r_vals = []
r_means = []
for i in range(learn.num_clusters):
    pi_predict_kdes.append(GKDE(learn.predict_maps[i].T))
    pi_obs_kdes.append(GKDE(learn.obs_maps[i].T))
    r_vals.append(
        np.divide(
            pi_obs_kdes[i](
                learn.predict_maps[i].T), 
            pi_predict_kdes[i](
                learn.predict_maps[i].T)))
    r_means.append(np.mean(r_vals[i]))
    
print(f'Diagnostics: {r_means}')


# visualing SVD spectral gap

# # plotting spectral gap for kernel PCA
# %reset -f out
# def plot_gap(all_eig_vals, n, cluster):
#     fig = plt.figure()
#     fig.clear()
#     #Plotting until maximum number of knots
#     eig_vals = all_eig_vals[cluster].eigenvalues_[0:10]
#     plt.semilogy(np.arange(np.size(eig_vals))+1,eig_vals/np.sum(eig_vals)*100, marker='.', markersize=20, linestyle='')
#     plt.semilogy(np.arange(np.size(eig_vals))+1,eig_vals[n]/np.sum(eig_vals)*100*np.ones(np.size(eig_vals)), 'k--')
#     plt.semilogy(np.arange(np.size(eig_vals))+1,eig_vals[n+1]/np.sum(eig_vals)*100*np.ones(np.size(eig_vals)), 'r--')
#     plt.text(n+1, eig_vals[n]/np.sum(eig_vals)*150, 
#              r'%2.3f' %(np.sum(eig_vals[0:n+1])/np.sum(eig_vals)*100) + '% of variation explained by first ' + '%1d' %(n+1) + ' PCs.', 
#                                                                {'color': 'k', 'fontsize': 14})
#     plt.text(n+2, eig_vals[n+1]/np.sum(eig_vals)*150, 
#              r'Order of magnitude of gap is %4.2f.' %(np.log10(eig_vals[n])-np.log10(eig_vals[n+1])), 
#                                                                {'color': 'r', 'fontsize': 14})
#     s = 'Determining QoI for cluster #%1d' %(cluster+1)
#     plt.title(s)
#     plt.xlabel('Principal Component #')
#     plt.ylabel('% of Variation')
#     plt.xlim([0.1, np.size(eig_vals)+1])
#     plt.ylim([1e-5,500])


# wd.interact(plot_gap, all_eig_vals=wd.fixed(learn.kpcas),
#             n = wd.IntSlider(value=0, min=0, max=5),
#             cluster = wd.IntSlider(value=0, min=0, max=learn.num_clusters-1))


# visualizing results

# defining uniform distribution for initial density 
def unif_dist(x, p_range):
    y = np.zeros(x.shape)
    val = 1.0/(p_range[1] - p_range[0])
    for i, xi in enumerate(x):
        if xi < p_range[0] or xi >  p_range[1]:
            y[i] = 0
        else:
            y[i] = val
    return y

# calculating eact data-generating marginals
exact_param_marginals = [lambda x : beta.pdf((x-1)/2,2,5)/2,
                         lambda x : norm.pdf(x,2.5,0.5)]

# calculating exact data-generating joint
np.random.seed(1234) # for reproducibility
params_graphing = np.random.uniform(low=0.0,high=5.0,size=(2,10000)) # large number of uniform parameter samples for graphing

exact_dg = lambda x, y : exact_param_marginals[0](x)*exact_param_marginals[1](y)
exact_dg = exact_dg(params_graphing[0,:],params_graphing[1,:])
kde_dg = GKDE(params_obs)(params_graphing)

# KDEs of true marginals
kde_param_marginals = []
for i in range(params.shape[0]):
        kde_param_marginals.append(GKDE(params_obs[i,:]))

# constructing and plotting updated marginals
x_min = 0.0
x_max = 5.0
delta = 0.25*(x_max - x_min)
x = np.linspace(x_min-delta, x_max+delta, 100)
param_labels = [r'$a$', r'$b$']

param_marginals = []
param_str = ['a', 'b']
for i in range(params.shape[0]):
    plt.figure()
    plt.plot(x, unif_dist(x,[0.0,5.0]), label='Initial', linewidth=2, c=c[0])
    param_marginals.append(GKDE(params[i,:], weights=r_vals[0]))
    mar = param_marginals[i](x)
    plt.plot(x, mar, label = 'Updated', linewidth=4, linestyle='dashed', c=c[1])
    plt.plot(x, exact_param_marginals[i](x), label='Data-generating', linewidth=4, linestyle='dotted', c=c[2])
    plt.title('Densities for parameter '+param_labels[i])
    plt.xlabel(param_labels[i])
    plt.legend()
    plt.tight_layout()
    fn = 'plots/wave_marginal_' + param_str[i] + '_2.png'
    plt.savefig(fn, bbox_inches='tight')
    plt.show()

# color plot of updated density
pi_update = GKDE(params, weights=r_vals[0])(params_graphing)
plt.figure()
plt.scatter(params_graphing[0,:], params_graphing[1,:], c=pi_update)
plt.scatter(params_obs[0,:], params_obs[1,:], c='xkcd:black', s=10, label='data-generating samples')
plt.legend()
plt.xlabel(param_labels[0])
plt.ylabel(param_labels[1])
plt.title(f'Color plot of updated density')
plt.colorbar(label='density')
plt.tight_layout()
plt.savefig('plots/wave_joint_2.png', bbox_inches='tight')
plt.show()


# computing TV metrics

# calculating TV metric between updated and exact joint distributions
TV = np.abs(pi_update-exact_dg)/2
# TV = np.abs(pi_update-kde_dg)/2
TV = np.mean(TV)*25
print(f'TV metric between pi_update and data-generating joint distribution: {TV}')

marginal_TVs = []
for i in range(params.shape[0]):
    diff = lambda x : np.abs(param_marginals[i](x)-exact_param_marginals[i](x))
    # diff = lambda x : np.abs(param_marginals[i](x)-kde_param_marginals[i](x))
    TV, _ = quadrature(diff, 0.0, 5.0, tol=1e-2)
    marginal_TVs.append(TV/2)
print(f'TV metric between pi_update marginals and DG marginals: {marginal_TVs}')
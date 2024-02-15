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


# loading data

# parameter samples for construction of pi_obs

num_obs_samples = 200

params_obs = np.load('data/params_obs', allow_pickle=True)
obs = np.load('data/obs', allow_pickle=True)

obs_time_series = obs[:,16,:]

# parameter samples of pi_init

num_samples = int(1E3)

params = np.load('data/params', allow_pickle=True)
pred = np.load('data/pred_9x9', allow_pickle=True)

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
import numpy as np
from luq.luq import *
from scipy.stats import norm, beta
from scipy.stats import gaussian_kde as GKDE
from scipy.integrate import quadrature
import matplotlib.pyplot as plt
import matplotlib.tri as tri
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


# loading samples

# parameter samples for construction of pi_obs

num_obs_samples = 200

params_obs = np.load('data/params_obs', allow_pickle=True)
obs = np.load('data/obs', allow_pickle=True)

obs_data = obs[:,:,4]

# parameter samples of pi_init

num_samples = int(1E3)

params = np.load('data/params', allow_pickle=True)
pred = np.load('data/pred_49x49', allow_pickle=True)

pred_data = pred[:,:,4]


# filtering data

# # instantiating luq

# learn = LUQ(pred_data,
#              obs_data)

# # filtering observed data using sum of Gaussians

# # defining data coordiates where observed data is taken from
# grid_size = 9
# delta = 5 / (grid_size + 1)
# X, Y = np.meshgrid(range(grid_size),range(grid_size))
# X = X / grid_size * (5 - delta) + delta
# Y = Y / grid_size * (5 - delta) + delta
# data_coordinates = np.vstack([X.flatten(), Y.flatten()]).T

# set seed for reproducibility
# np.random.seed(333)

# # filter data; code takes a long time (maybe up to an hour), so luq instance is saved after to be loaded for future runs
# learn.filter_data(filter_method='rbfs',
#                    filtered_data_coordinates=data_coordinates,
#                    num_rbf_list=range(1,8),
#                    initializer='kmeans',
#                    max_opt_count=10,
#                    filter_predictions=False,
#                    verbose=True)

# learn.save_instance('instances/part3')

# loading pre-computed data

import pickle

pf = open('instances/part3','rb')
learn = pickle.load(pf)
pf.close()

# re-evaluating observed data on finer mesh

grid_size = 49
delta = 5 / (grid_size + 1)
X, Y = np.meshgrid(range(grid_size),range(grid_size))
X = X / grid_size * (5 - delta) + delta
Y = Y / grid_size * (5 - delta) + delta
data_coordinates = np.vstack([X.flatten(), Y.flatten()]).T

learn.filtered_predictions = pred[:,:,4] # updating predictions to use those on 49x49 grid

learn.new_data_coordinates(data_coordinates, 
                           recalc_pred=False)

print(f'Predicted data shape: {learn.filtered_predictions.shape}')
print(f'Filtered observed data shape: {learn.filtered_obs.shape}')

# visualizing filtered surfaces

# sample = np.random.randint(0, obs.shape[0])
sample = 20 # sample used in paper
num_rbfs = learn.filtered_obs_params[sample]['weights'].shape[0]

grid_size_orig = 9
delta_orig = 5 / (grid_size_orig + 1)
X_orig, Y_orig = np.meshgrid(range(grid_size_orig),range(grid_size_orig))
X_orig = X_orig / grid_size_orig * (5 - delta_orig) + delta_orig
Y_orig = Y_orig / grid_size_orig * (5 - delta_orig) + delta_orig
data_coordinates_orig = np.vstack([X_orig.flatten(), Y_orig.flatten()]).T

grid_size = 49
fitted_mesh = np.zeros((grid_size,grid_size))
i = -1
for k in range(grid_size**2):
    j = k % grid_size
    if j == 0:
        i += 1
    fitted_mesh[i,j] = learn.filtered_obs[sample,k]
    
orig = np.zeros((grid_size_orig,grid_size_orig))
i = -1
for k in range(grid_size_orig**2):
    j = k % grid_size_orig
    if j == 0:
        i += 1
    orig[i,j] = obs[sample,k,4]

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig, computed_zorder=False, auto_add_to_figure=False)
ax = fig.add_axes(ax)
ax.scatter(X_orig, Y_orig, orig, c=c[1], s=50.0, label='true data points')
ax.plot_surface(X, 
                Y, 
                fitted_mesh, 
                rstride=1, 
                cstride=1,
                cmap=cm.Blues, 
                edgecolor='none',
                alpha=0.6)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_zlabel('wave height')
fig.savefig('plots/filter_step.png', bbox_inches='tight')
plt.show()


# learning QoI map

# learning 2 QoI's from data using kernel pca and transforming the data into QoI samples

pred_maps, obs_maps = learn.learn_qois_and_transform(num_qoi=2)


# computing DCI solution

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
param_str = ['a', 'b']
param_marginals = []
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
    fn = 'plots/wave_marginal_' + param_str[i] + '_3.png'
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
plt.savefig('plots/wave_joint_3.png', bbox_inches='tight')
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
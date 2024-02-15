import numpy as np
from luq.luq import *
from scipy.stats import norm, beta
from scipy.stats import gaussian_kde as GKDE
from scipy.integrate import quadrature
import matplotlib.pyplot as plt
# plt.rcParams.update({'font.size': 12})
import matplotlib.tri as tri

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
obs = np.load('data/obs_clean', allow_pickle=True)

# extracting observed values at (4.0,1.0) when t=2.5 which represent the observed QoI samples
obs_qoi = obs[:,16,4]

# parameter samples of pi_init

num_samples = int(1E3)

params = np.load('data/params', allow_pickle=True)
pred = np.load('data/pred_9x9', allow_pickle=True)

# extracting predicted values at (4.0,1.0) when t=2.5 which represent the predicted QoI samples
pred_qoi = pred[:,16,4]


# visualizing contour structure of QoI map

# contour plot

xi = np.linspace(0.0, 5.0, 100)
yi = np.linspace(0.0, 5.0, 100)

triang = tri.Triangulation(params[0,:],params[1,:])
interpolator = tri.LinearTriInterpolator(triang, pred_qoi)
Xi, Yi = np.meshgrid(xi, yi)
zi = interpolator(Xi, Yi)

fig, ax = plt.subplots()

ax.contour(xi, yi, zi, levels=14, linewidths=0.5, colors='k')
cntr = ax.contourf(xi, yi, zi, levels=14, cmap="RdBu_r")

fig.colorbar(cntr, ax=ax).set_label(f'Q', fontsize=14)
ax.set(xlim=(0, 5), ylim=(0, 5))
plt.title('Contour plot of QoI map')
plt.xlabel('a')
plt.ylabel('b')
plt.tight_layout()
plt.savefig('plots/wave_contour1.png', bbox_inches='tight')
plt.show()


# computing DCI solution

# Generate kernel density estimates on specified QoI

pi_predict_kde = GKDE(pred_qoi.T)
pi_obs_kde = GKDE(obs_qoi.T)
r_vals = np.divide(pi_obs_kde(pred_qoi.T),
                   pi_predict_kde(pred_qoi.T))
r_mean = np.mean(r_vals)
print(f'Diagnostic: {r_mean}')


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
    param_marginals.append(GKDE(params[i,:], weights=r_vals))
    mar = param_marginals[i](x)
    plt.plot(x, mar, label = 'Updated', linewidth=4, linestyle='dashed', c=c[1])
    plt.plot(x, exact_param_marginals[i](x), label='Data-generating', linewidth=4, linestyle='dotted', c=c[2])
    plt.title('Densities for parameter '+param_labels[i])
    plt.xlabel(param_labels[i])
    plt.legend()
    plt.tight_layout()
    fn = 'plots/wave_marginal_' + param_str[i] + '_1.png'
    plt.savefig(fn, bbox_inches='tight')
    plt.show()

# color plot of updated density

pi_update = GKDE(params, weights=r_vals)(params_graphing)
plt.figure()
plt.scatter(params_graphing[0,:], params_graphing[1,:], c=pi_update)
plt.scatter(params_obs[0,:], params_obs[1,:], c='xkcd:black', s=10, label='data-generating samples')
plt.legend()
plt.xlabel(param_labels[0])
plt.ylabel(param_labels[1])
plt.title(f'Color plot of updated density')
plt.colorbar(label='density')
plt.tight_layout()
plt.savefig('plots/wave_joint_1.png', bbox_inches='tight')
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
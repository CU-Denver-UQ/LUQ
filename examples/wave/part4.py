import numpy as np
from luq.luq import *
from scipy.stats import norm, beta # for data-generating distributions
from scipy.stats import gaussian_kde as GKDE
from scipy.integrate import quadrature # for calculating 1-D TV metrics
from tabulate import tabulate # for presenting results for iterative approach
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
import matplotlib.tri as tri
import ipywidgets as wd

# color palette
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

params_obs = np.load('data/params_obs', allow_pickle=True) # unknown data-generated parameters corresponding to observed samples
obs = np.load('data/obs', allow_pickle=True) # noisy observed samples

# parameter samples of pi_init

num_samples = int(1E3)

params = np.load('data/params', allow_pickle=True) # uniformly distributed parameter samples
pred = np.load('data/pred_49x49', allow_pickle=True) # predicted data 


# filtering data

# # setup for part IV of example using LUQ

# # code used to filter data on 9x9 grid of spatial locations

# # instantiate LUQ at each time step
# learn = []

# for i in range(obs.shape[-1]): # observed data has shape (num_samples,num_spatial_locations,num_time_steps)
#     pred_data = pred[:,:,i]
#     obs_data = obs[:,:,i]
#     learn.append(LUQ(predicted_data=pred_data,
#                       observed_data=obs_data))

# # filtering observed data at each time step using sum of Gaussians

# grid_size = 9
# delta = 5 / (grid_size + 1)
# X, Y = np.meshgrid(range(grid_size),range(grid_size))
# X = X / grid_size * (5 - delta) + delta
# Y = Y / grid_size * (5 - delta) + delta
# data_coordinates = np.vstack([X.flatten(), Y.flatten()]).T

# # loading luq instance for part 3 to not redo filtering to same data
# import pickle
# pf3 = open('instances/part3','rb')
# learn3 = pickle.load(pf3)
# pf3.close()

# np.random.seed(44444)

# # iterate over time steps
# for i in range(obs.shape[-1]):
#     if i == 4: 
#         learn[i] = learn3 # using filtered data from part 3 for time index 4 corresponding to t=2.5
#     else:
#         learn[i].filter_data(filter_method='rbfs',
#                               filtered_data_coordinates=data_coordinates,
#                               num_rbf_list=range(1,8),
#                               initializer='kmeans',
#                               max_opt_count=10,
#                               filter_predictions=False,
#                               verbose=True)
#     fn = 'instances/part4/part4_' + str(i)
#     learn[i].save_instance(fn) # saving LUQ instance 

# # loading pre-computed data

import pickle

learn = []
for i in range(obs.shape[-1]):
    fn = 'instances/part4/part4_' + str(i)
    pf = open(fn,'rb')
    learn.append(pickle.load(pf))
    pf.close()

# re-evaluating filtered observations on same grid as predictions

grid_size = 49
delta = 5 / (grid_size + 1)
X, Y = np.meshgrid(range(grid_size),range(grid_size))
X = X / grid_size * (5 - delta) + delta
Y = Y / grid_size * (5 - delta) + delta
data_coordinates = np.vstack([X.flatten(), Y.flatten()]).T

for t in range(obs.shape[-1]):
    learn[t].new_data_coordinates(data_coordinates, 
                                  recalc_pred=False)
    
    print(f'Predicted data shape: {learn[t].predicted_data.shape}')
    print(f'Filtered observed data shape: {learn[t].filtered_obs.shape}')
    print()


# learning QoI map
    
# learning 2 QoI's from data using kernel pca and transforming the data into QoI samples

for t in range(obs.shape[-1]):
    learn[t].learn_qois_and_transform(num_qoi=2)


# computing iterative DCI solution
    
# generate kernel density estimates on new QoI and calculate new weights

num_qoi = 2
tol = 0.095
r_vals = []
r_means = []

times_used = []
num_qoi_used = []

for alg in [0,1]: # computing solution without and with optional quality check
    r_vals.append([])
    r_means.append([])
    times_used.append([])
    num_qoi_used.append([])
    for t in range(obs.shape[-1]): # iterate over time steps
        for n in reversed(range(1,num_qoi+1)): # iterate over number of QoI from max 2 down to 0
            if len(r_vals[alg]) == 0:
                weights = np.ones(learn[t].predict_maps[0][:,:n].shape[0]) # weights of all ones for initial iteration
            else:
                weights = np.prod(r_vals[alg], axis=0) # weights from previous time step for non-initial iterations
            pi_pred = GKDE(learn[t].predict_maps[0][:,:n].T, weights=weights) # computing new predicted density on learned QoI
            pi_obs = GKDE(learn[t].obs_maps[0][:,:n].T) # computing observed density
            # computing new r-values associated with learned QoI
            r_vals[alg].append(
                np.divide(
                    pi_obs(
                        learn[t].predict_maps[0][:,:n].T), 
                    pi_pred(
                        learn[t].predict_maps[0][:,:n].T))) 
            r = weights*r_vals[alg][-1] # new proposed r-samples
            r_means[alg].append(np.mean(r))
            if np.abs(1-r_means[alg][-1]) <= tol: # checking DCI diagnostic
                if alg == 0 or np.mean(r**2) > np.mean(weights**2): # alg=0 will use data; alg=1 will only use data if optional quality check holds
                    times_used[alg].append(0.5*(t+1))               
                    num_qoi_used[alg].append(n)
                else:
                    r_vals[alg].pop()
                    r_means[alg].pop()        
                break # breaks if DCI diagnostic is within tolerance; otherwise, r is re-computed using fewer QoI components with previous results removed
            else:
                r_vals[alg].pop() 
                r_means[alg].pop()

# presenting results for both algorithms
tables = []
for alg in [0,1]:
    tables.append([])
    for i in range(len(r_vals[alg])):
        r = np.prod(r_vals[alg][:i+1],axis=0) # r samples at each iteration
        tables[alg].append([i+1,times_used[alg][i],num_qoi_used[alg][i],r_means[alg][i],np.mean(r**2)])
    print(f'Results for algorithm {alg}:')
    print(tabulate(tables[alg], headers=['iteration','time','number of QoIs used','E_init(r)','E_update(r)']))
    print()


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

for alg in [0,1]:
    param_marginals.append([])
    for i in range(params.shape[0]):
        plt.figure()
        plt.plot(x, unif_dist(x,[0.0,5.0]), label='Initial', linewidth=2, c=c[0])
        param_marginals[alg].append(GKDE(params[i,:], weights=np.prod(r_vals[alg], axis=0)))
        mar = param_marginals[alg][i](x)
        plt.plot(x, mar, label = 'Updated', linewidth=4, linestyle='dashed', c=c[1])
        plt.plot(x, exact_param_marginals[i](x), label='Data-generating', linewidth=4, linestyle='dotted', c=c[2])
        plt.title('Densities for parameter '+param_labels[i]+f' using algorithm {alg+1}')
        plt.xlabel(param_labels[i])
        plt.legend()
        plt.tight_layout()
        if alg == 0:
            fn = 'plots/wave_marginal_' + param_str[i] + '_4.png'
        else:
            fn = 'plots/wave_marginal_' + param_str[i] + '_4_r2.png'
        plt.savefig(fn, bbox_inches='tight')
        plt.show()
        
# plotting updated marginals at each iteration

param_marginals = []

for alg in [0,1]:
    param_marginals.append([[],[]])
    for i in range(params.shape[0]):
        plt.figure()
        plt.plot(x, unif_dist(x,[0.0,5.0]), label='Initial', linewidth=2, c=c[2])
        plt.plot(x, exact_param_marginals[i](x), label='Data-generating', linewidth=2, linestyle='dotted', c=c[1])
        # plt.plot(x, kde_param_marginals[i](x), label='KDE', linewidth=4, c=c[1])
        for j in range(len(r_vals[alg])):
            param_marginals[alg][i].append(GKDE(params[i,:], weights=np.prod(r_vals[alg][:j+1],axis=0)))
            if j == len(r_vals)-1:
                plt.plot(x, param_marginals[alg][i][j](x), label=f'final iteration', linewidth=2, c=c[0], alpha=(j+1)/len(r_vals[alg]))
            else:
                plt.plot(x, param_marginals[alg][i][j](x), label=f'iteration {j+1}', linewidth=2, linestyle='dashed', c=c[0], alpha=(j+1)/len(r_vals[alg]))
        plt.title('Updated densities for parameter '+param_labels[i]+f' using algorithm {alg+1}')
        plt.xlabel(param_labels[i])
        plt.legend()
        plt.tight_layout()
        if alg == 0:
            fn = 'plots/wave_marginal_' + param_str[i] + '_iter.png'
        else:
            fn = 'plots/wave_marginal_' + param_str[i] + '_iter_r2.png'
        plt.savefig(fn, bbox_inches='tight')
        plt.show()
        
# color plot of updated density

pi_updates = []
for alg in [0,1]:
    pi_updates.append(GKDE(params, weights=np.prod(r_vals[alg], axis=0))(params_graphing))
    plt.figure()
    plt.scatter(params_graphing[0,:], params_graphing[1,:], c=pi_updates[alg])
    plt.scatter(params_obs[0,:], params_obs[1,:], c='xkcd:black', s=10, label='data-generating samples')
    plt.legend()
    plt.xlabel(param_labels[0])
    plt.ylabel(param_labels[1])
    plt.title(f'Color plot of updated density using algorithm {alg+1}')
    plt.colorbar(label='density')
    plt.tight_layout()
    if alg == 0:
        fn = 'plots/wave_joint_4.png'
    else:
        fn = 'plots/wave_joint_4_r2.png'
    plt.savefig(fn, bbox_inches='tight')
    plt.show()


# computing TV metrics
    
# calculating TV metric between updated and exact joint distributions

for alg in [0,1]:
    TV_final = np.abs(pi_updates[alg]-exact_dg)/2
    # TV = np.abs(pi_updates[alg]-kde_dg)/2
    TV_final = np.mean(TV_final)*25
    print(f'TV metric between pi_update and data-generating joint distribution for algorithm {alg+1}: {TV_final}')

# calculating TVs for marginals at each iteration

marginal_TVs = []
TVs = []
for alg in [0,1]:
    marginal_TVs.append([[],[]])
    for i in range(params.shape[0]):
        # diff = lambda x : np.abs(unif_dist(x,[0.0,5.0])-exact_param_marginals[i](x))
        diff = lambda x : np.abs(unif_dist(x,[0.0,5.0])-kde_param_marginals[i](x))
        TV, _ = quadrature(diff, 0.0, 5.0, tol=1e-2)
        marginal_TVs[alg][i].append(TV/2)
    for i in range(params.shape[0]):
        for j in range(len(r_vals[alg])):
            # diff = lambda x : np.abs(param_marginals[alg][i][j](x)-exact_param_marginals[i](x))
            diff = lambda x : np.abs(param_marginals[alg][i][j](x)-kde_param_marginals[i](x))
            TV, _ = quadrature(diff, 0.0, 5.0, tol=1e-2)
            marginal_TVs[alg][i].append(TV/2)

    TVs.append([])
    pi_init = lambda x, y : unif_dist(x,[0.0,5.0]) * unif_dist(y,[0.0,5.0])
    pi_init = pi_init(params_graphing[0,:], params_graphing[1,:])
    # TV = np.abs(pi_init-exact_dg)/2
    TV = np.abs(pi_init-kde_dg)/2
    TV = np.mean(TV)*25
    TVs[alg].append(TV)
    for j in range(len(r_vals[alg])):
        pi_update = GKDE(params, weights=np.prod(r_vals[alg][:j+1], axis=0))(params_graphing)
        # TV = np.abs(pi_update-exact_dg)/2
        TV = np.abs(pi_update-kde_dg)/2
        TV = np.mean(TV)*25
        TVs[alg].append(TV)

min_marginal_TVs = []
for i in range(params.shape[0]):
    diff = lambda x : np.abs(exact_param_marginals[i](x)-kde_param_marginals[i](x))
    TV, _ = quadrature(diff, 0.0, 5.0, tol=1e-2)
    min_marginal_TVs.append(TV/2)

min_TV = np.abs(kde_dg-exact_dg)/2
min_TV = np.mean(min_TV)*25

# printing TVs at final iteration

for alg in [0,1]:
    for i in range(params.shape[0]):
        print(f'TV metric for algorithm {alg+1} between final iteration and DG marginals for {param_labels[i]}: {str(marginal_TVs[alg][i][-1])}')

# plotting TV per iteration
        
# reset legend fontsize for TV plots
plt.rcParams.update({'legend.fontsize': 10})

for alg in [0,1]:
    n = range(len(TVs[alg]))
    plt.figure()
    plt.scatter(n,TVs[alg],c=c[0])
    plt.plot(n,TVs[alg],c=c[0],label='TV between updated and exact densities')
    plt.hlines(min_TV,0,len(TVs[alg])-1,linestyles='dashed',colors=c[0],label='TV between exact and KDE of data-generating densities')
    plt.xlabel('Iteration')
    plt.ylabel('TV')
    plt.ylim([0,1])
    # plt.title(f'TV metric between updated and exact joint parameter densities for algorithm {alg+1}')
    plt.legend()
    plt.tight_layout()

    for i, t in enumerate(times_used[alg]):
        txt = 't=' + str(t)
        plt.annotate(txt, (n[i+1]+0.1, TVs[alg][i+1]+0.02), size=12)

    if alg == 0:
        fn = 'plots/wave_TV_joint.png'
    else:
        fn = 'plots/wave_TV_joint_r2.png'
    plt.savefig(fn, bbox_inches='tight')
    plt.show()
    
# plotting marginal TVs
        
for alg in [0,1]:
    n = range(len(marginal_TVs[alg][0]))
    plt.figure()
    for i in range(params.shape[0]):
        plt.scatter(n,marginal_TVs[alg][i], c=c[i], label='TV between updated and exact marginals for '+param_labels[i])
        plt.plot(n,marginal_TVs[alg][i], c=c[i])
        plt.hlines(min_marginal_TVs[i], 0, len(marginal_TVs[alg][0])-1, linestyles='dashed', colors=c[i],label='TV between exact and KDE of data-generating marginals for '+param_labels[i])
    plt.legend()
    plt.xlabel('iteration')
    plt.ylabel('TV')
    plt.ylim([0,1])
    # plt.title(f'TV metric between updated and exact marignal parameter densities for algorithm {alg+1}')
    plt.tight_layout()

    for i, t in enumerate(times_used[alg]):
        txt = 't=' + str(t)
        plt.annotate(txt, (n[i+1]+0.1, np.max(marginal_TVs[alg],axis=0)[i+1]+0.02), size=12)
  
    if alg == 0:
        fn = 'plots/wave_TV_marginals.png'
    else:
        fn = 'plots/wave_TV_marginals_r2.png'
    plt.savefig(fn, bbox_inches='tight')
    plt.show()
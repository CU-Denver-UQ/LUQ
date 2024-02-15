# mat73 required to read data; run pip install line if not installed

# !pip install mat73  # For reading in Matlab 7.3 files
import mat73 as mat73


# The libraries we will use
import numpy as np
import scipy.io as sio

# importing LUQ
from luq.luq import *

# distributions for data-generating samples and comparing approx vs true solutions
from scipy.stats import norm, beta

# Gaussian KDE 
from scipy.stats import gaussian_kde as GKDE

# quadrautre for TV metric
from scipy.integrate import quadrature

# plotting
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import ipywidgets as wd

# colorblind friendly color palette
c = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

# Set up fontsizes for plots
plt_params = {'legend.fontsize': 14,
          'figure.figsize': (6.4, 4.8),
         'axes.labelsize': 16,
         'axes.titlesize': 16,
         'xtick.labelsize': 14,
         'ytick.labelsize': 14}
plt.rcParams.update(plt_params)

np.random.seed(123456)


# Load the initial dataset containing both model data (used to generate predicted data) and parameter samples from initial distribution

init_data_set = mat73.loadmat('../SteelDrums/1DCase1-Feb22-2023/1DCase1/Prior/prior.mat')

# init_data_set keys are a0, a1, xs, and ys

xs = init_data_set['xs']
ys = init_data_set['ys']

data_init = ys

a0_init = init_data_set['a0']  # initial samples of first parameter
a1_init = init_data_set['a1']  # initial samples of second parameter


# Now load/analyze the data-generating dataset

obs_data_set = mat73.loadmat('../SteelDrums/1DCase1-Feb22-2023/1DCase1/Observed/observed.mat')

# obs_data_set keys are a0, a1, xs, and ys

data_obs = obs_data_set['ys']

a0_obs = obs_data_set['a0']  # samples of first parameter responsible for observed data
a1_obs = obs_data_set['a1']  # samples of second parameter responsible for observed data


# What if we used all sensor data (with NO noise), 
# built a classifier from *known* labels of the initial data 
# to determine whether observed data belonged to elliptic or 
# hyperbolic parameter types, and then pieced together a 
# global inverse solution from local inverse solutions?
# This is the most ideal scenario to determine the best 
# possible case for our inverse solution.

# First get known labels from initial dataset

idx_hyperbolic = np.where(a1_init>0)[0]
idx_elliptic = np.where(a1_init<0)[0]

init_labels = np.zeros(len(a1_init))
init_labels[idx_hyperbolic] = 1

# Define the observable data function that transforms
# the model data into data values at sensor points

def generate_observable_data(data, xs, n_sensors):
    x_sensors = np.linspace(xs[0], xs[-1], n_sensors)
    return np.interp(x_sensors, xs, data)

n_sensors = int(1001)

x_sensors = np.linspace(xs[0], xs[-1], n_sensors)

num_init = int(1e4)

data_init_sensors = np.zeros((num_init, n_sensors))
for i in range(num_init):
    data_init_sensors[i,:] = generate_observable_data(data_init[i,:], xs, n_sensors)

num_obs = int(3e3)

data_obs_sensors = np.zeros((num_obs, n_sensors))
for i in range(num_obs):
    data_obs_sensors[i,:] = generate_observable_data(data_obs[i,:], xs, n_sensors)

xs_include = list(np.arange(50,n_sensors-50))

learn = LUQ(predicted_data = data_init_sensors[:,xs_include],
            observed_data = data_obs_sensors[:,xs_include])

# learn.dynamics takes the longest to run because of the size of the datasets.
# Expect to wait a couple of minutes.
# One could use a higher C value, but this will take longer to run and did not result 
# in better results than those obtained with C=1e2. 

learn.dynamics(custom_labels=init_labels,
               proposals=({'kernel': 'linear', 'C': 1e1},
                          {'kernel': 'linear', 'C': 1e2}),
               relabel_predictions=False)

pred_maps, obs_maps = learn.learn_qois_and_transform(num_qoi=2)

# Generate kernel density estimates on new QoI and calculate new weights
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

params = np.vstack((a0_init, a1_init)).T

params_obs = np.vstack((a0_obs, a1_obs)).T

param_marginals = []
true_param_marginals = []
lam_ptr = []
cluster_weights = []
param_marginals_modified = []
modified_r_values = np.zeros(len(a0_init))

for i in range(learn.num_clusters):
    lam_ptr.append(np.where(learn.predict_labels == i)[0])
    cluster_weights.append(len(np.where(learn.obs_labels == i)[0]) / num_obs)
    modified_r_values[lam_ptr[i]] = r_vals[i]*cluster_weights[i]  
    
for i in range(params.shape[1]):
    true_param_marginals.append(GKDE(params_obs[:,i]))
    param_marginals_modified.append(GKDE(params[:,i], weights=modified_r_values))
    param_marginals.append([])
    for k in range(learn.num_clusters):
        param_marginals[i].append(GKDE(params[lam_ptr[k], i], weights=r_vals[k]))

print(f'Cluster weights: {cluster_weights}')

param_labels = ['$a_0$', '$a_1$']

for i in range(params.shape[1]):
    if i==0:
        x = np.linspace(0.8,1.2,100)
    else:
        x = np.linspace(-0.2,0.2,100)
        
    fig = plt.figure()
    fig.clear()
    
    plt.plot(x, true_param_marginals[i](x), label = 'KDE of DG; full set', 
             linewidth=2, linestyle='dashed')
    
    plt.plot(x,1/0.4*np.ones(len(x)), linewidth=1)
    
    plt.plot(x, param_marginals_modified[i](x), 
             label = r'Update; full set', linewidth=2, linestyle='dashdot')
    
    if i==1:
        plt.axvline(0, color='k', lw=2, ls='-.')
    plt.title('Densities for parameter ' + param_labels[i])
    plt.legend()
    plt.tight_layout()
    plt.show()

# Now add noise to observations, use less initial 
# and observed samples, use LUQ to filter based on 
# known labels of reduced initial sample set, classify, 
# and learn QoI.
# This will produce the operational case studies for 
# limited datasets that better reflect real-world conditions 
# that are then compared to the ideal case above.
    
num_init = int(2E2)

data_init_subset_sensors = np.zeros((num_init, n_sensors))
for i in range(num_init):
    data_init_subset_sensors[i,:] = generate_observable_data(data_init[i,:], xs, n_sensors)

# Generate 3 different observation datasets

num_obs = int(7.5e1)

start_idx = [750, 1500, 2250]

num_sets = len(start_idx)

data_obs_subset_sensors = np.zeros((num_sets, num_obs, n_sensors))
for j in range(num_sets):
    for i in range(num_obs):
        data_obs_subset_sensors[j,i,:] = generate_observable_data(data_obs[start_idx[j]+i,:], xs, n_sensors)

SNR_obs = 5

var_noise = np.var(data_init_subset_sensors, axis=0) / SNR_obs

noisy_obs_subset_sensors = data_obs_subset_sensors +\
                            np.random.randn(
                                np.shape(data_obs_subset_sensors)[0], 
                                np.shape(data_obs_subset_sensors)[1],
                                np.shape(data_obs_subset_sensors)[2]) * np.sqrt(var_noise)

SNR_init = 10

var_noise = np.var(data_init_subset_sensors, axis=0) / SNR_init

noisy_data_init_subset_sensors = data_init_subset_sensors +\
            np.random.randn(np.shape(data_init_subset_sensors)[0], 
                            np.shape(data_init_subset_sensors)[1]) * np.sqrt(var_noise)


learn_base = LUQ(noisy_data_init_subset_sensors[:num_init, xs_include])

num_filtered_obs = 60  # Using this many filtered data to learn QoI 

filtered_data_coordinates = np.linspace(x_sensors[xs_include[0]],
                                        x_sensors[xs_include[-1]],
                                        num_filtered_obs)

predicted_data_coordinates = x_sensors[xs_include]

learn_base.filter_data(filter_method='splines',
                       tol=1e-2, 
                       min_knots=6, 
                       max_knots=10,
                       filtered_data_coordinates = filtered_data_coordinates, 
                       predicted_data_coordinates = predicted_data_coordinates
                       )

cluster_labels = np.zeros(num_init, dtype=int)

idx_hyperbolic = np.where(a1_init[:num_init]>0)[0]

cluster_labels[idx_hyperbolic] = 1

learn_base.dynamics(custom_labels=cluster_labels,
                   proposals=({'kernel': 'linear', 'C': 1e3},
                              {'kernel': 'linear', 'C': 1e4},
                              {'kernel': 'linear', 'C': 1e5},
                              {'kernel': 'linear', 'C': 1e6}),
                   relabel_predictions=False)

learn_base.learn_qois_and_transform(num_qoi=2)


# Compare to 30 filtered observations for sufficiency

learn_base_test = LUQ(noisy_data_init_subset_sensors[:num_init, xs_include])

num_filtered_obs_test = 30  # Using this many filtered data to learn QoI 

filtered_data_coordinates = np.linspace(x_sensors[xs_include[0]],
                                        x_sensors[xs_include[-1]],
                                        num_filtered_obs_test)

predicted_data_coordinates = x_sensors[xs_include]

learn_base_test.filter_data(filter_method='splines',
                       tol=1e-2, 
                       min_knots=6, 
                       max_knots=10,
                       filtered_data_coordinates = filtered_data_coordinates, 
                       predicted_data_coordinates = predicted_data_coordinates
                       )

num_filtered_obs_test = 30  # Using this many filtered data to learn QoI 

filtered_data_coordinates = np.linspace(x_sensors[xs_include[0]],
                                        x_sensors[xs_include[-1]],
                                        num_filtered_obs_test)

predicted_data_coordinates = x_sensors[xs_include]

learn_base_test.filter_data(filter_method='splines',
                       tol=1e-2, 
                       min_knots=6, 
                       max_knots=10,
                       filtered_data_coordinates = filtered_data_coordinates, 
                       predicted_data_coordinates = predicted_data_coordinates
                       )

cluster_labels = np.zeros(num_init, dtype=int)

idx_hyperbolic = np.where(a1_init[:num_init]>0)[0]

cluster_labels[idx_hyperbolic] = 1

learn_base_test.dynamics(custom_labels=cluster_labels,
                   proposals=({'kernel': 'linear', 'C': 1e3},
                              {'kernel': 'linear', 'C': 1e4},
                              {'kernel': 'linear', 'C': 1e5},
                              {'kernel': 'linear', 'C': 1e6}),
                   relabel_predictions=False)

learn_base_test.learn_qois_and_transform(num_qoi=2)

LUQs = [learn_base_test, learn_base]

# normalize eigenvectors
unit_alphas = []
for i in range(2):  # Loop through each of the filtering data options
    unit_alphas.append([])
    for k in range(2):  # Loop through each of the clusters
        unit_alphas[i].append([])
        for j in range(2):
            unit_alphas[i][k].append(LUQs[i].kpcas[k].eigenvectors_[:,j] / np.linalg.norm(LUQs[i].kpcas[k].eigenvectors_[:,j], ord=2))

# applying linear regression for each grid pair
from scipy.linalg import lstsq

As = []
ms = []
bs = []
R_squared = []
for i in range(1):
    As.append([])
    ms.append([])
    bs.append([])
    R_squared.append([])
    for k in range(2):
        As[i].append([])
        ms[i].append([])
        bs[i].append([])
        R_squared[i].append([])
        for j in range(2):
            As[i][k].append(np.ones((len(unit_alphas[i][k][j]),2)))
            As[i][k][j][:,1] = unit_alphas[-(i+1)][k][j]
            coeffs, res, _, _ = lstsq(As[i][k][j], unit_alphas[-(i+2)][k][j])
            ms[i][k].append(coeffs[1])
            bs[i][k].append(coeffs[0])
            SS_tot = np.sum((unit_alphas[-(i+1)][k][j] - np.mean(unit_alphas[-(i+2)][k][j]))**2)
            R_squared[i][k].append(1-res/SS_tot)

print(f'ms: {ms}')
print('~'*100)
print(f'R^2 : {R_squared}')

learn_base.save_instance('luq_base')

import pickle

learn_list = []
for j in range(num_sets):
    LUQ_base_file = open("luq_base", "rb")
    learn_list.append(pickle.load(LUQ_base_file))
    LUQ_base_file.close()

for j in range(num_sets):
    learn_list[j].set_observations(noisy_obs_subset_sensors[j, :num_obs, xs_include].T)

plt.figure()

plt.scatter(x_sensors[xs_include], noisy_data_init_subset_sensors[0,xs_include].T, 
            s=5, marker='s', c='b', alpha=0.25, label='Noisy Predicted Data')
plt.plot(x_sensors[xs_include], data_init_subset_sensors[0,xs_include].T, 
         lw=2, c='b', ls='--',
         label='Filtered Predicted Response')
plt.scatter(x_sensors[xs_include], noisy_obs_subset_sensors[0,2,xs_include].T,
            s=10, marker='x', c='r', alpha=0.25, label='Noisy Observed Data')
plt.plot(x_sensors[xs_include], data_obs_subset_sensors[0,2,xs_include].T, 
         lw=2, c='r', ls='-.',
         label='Filtered Observed Response')
plt.xlabel('$x$-coordinate')
plt.ylabel('Displacement [dimensionless]')
plt.legend()
plt.tight_layout()
# plt.title('Responses and Noisy Data', fontsize=16)
plt.show()

for j in range(num_sets):
    learn_list[j].filter_observations(observed_data_coordinates=predicted_data_coordinates)

for j in range(num_sets):
    obs_maps.append(learn_list[j].classify_and_transform_observations())

pred_maps, obs_maps = [], []
for j in range(num_sets):
    obs_maps_temp = learn_list[j].transform_observations()
    pred_maps.append(learn_list[j].predict_maps)
    obs_maps.append(obs_maps_temp)

# Generate kernel density estimates on new QoI and calculate new weights
pi_predict_kdes = []
pi_obs_kdes = []
r_vals = []
r_means = []
for j in range(num_sets):
    pi_predict_kdes.append([])
    pi_obs_kdes.append([])
    r_vals.append([])
    r_means.append([])
    for i in range(learn_list[j].num_clusters):
        pi_predict_kdes[j].append(GKDE(learn_list[j].predict_maps[i].T))
        pi_obs_kdes[j].append(GKDE(learn_list[j].obs_maps[i].T))
        r_vals[j].append(
                    np.divide(
                        pi_obs_kdes[j][i](
                        learn_list[j].predict_maps[i].T), 
                        pi_predict_kdes[j][i](
                        learn_list[j].predict_maps[i].T)))
        r_means[j].append(np.mean(r_vals[j][i]))
print(f'Diagnostics: {r_means}')

params = np.vstack((a0_init[:num_init], a1_init[:num_init])).T

params_obs = np.vstack((a0_obs, a1_obs)).T

param_marginals = []
param_marginals_modified = []
ic_marginals = []
true_param_marginals = []
true_ic_marginals = []
lam_ptr = []
cluster_weights = []
modified_r_values = []

for j in range(num_sets):
    cluster_weights.append([])
    lam_ptr.append([])
    modified_r_values.append(np.zeros(num_init))
    for i in range(learn_list[j].num_clusters):
        lam_ptr[j].append(np.where(learn_list[j].predict_labels == i)[0])
        cluster_weights[j].append(len(np.where(learn_list[j].obs_labels == i)[0]) / num_obs)
        modified_r_values[j][lam_ptr[j][i]] = r_vals[j][i]*cluster_weights[j][i]
        
for j in range(num_sets):
    param_marginals.append([])
    param_marginals_modified.append([])
    true_param_marginals.append([])
    for i in range(params.shape[1]):
        true_param_marginals[j].append(GKDE(params_obs[start_idx[j]:start_idx[j]+num_obs,i]))
        param_marginals_modified[j].append(GKDE(params[:,i], weights=modified_r_values[j]))
        param_marginals[j].append([])
        for k in range(learn.num_clusters):
            param_marginals[j][i].append(GKDE(params[lam_ptr[j][k], i], weights=r_vals[j][k]))

print(f'Cluster weights: {cluster_weights}')

param_labels = ['$a_0$', '$a_1$']

param_0_y_limits = [-0.25, 8]
param_1_y_limits = [-0.25, 6.5]

for i in range(params.shape[1]):
    if i==0:
        x = np.linspace(0.8,1.2,100)
    else:
        x = np.linspace(-0.2,0.2,100)
        
    for k in range(num_sets):
        fig = plt.figure()
        fig.clear()
        
        plt.plot(x, true_param_marginals[k][i](x), label = 'KDE of DG; set {}'.format(k+1), 
                 linewidth=2, linestyle='dashed')
        
        plt.plot(x,1/0.4*np.ones(len(x)), linewidth=1)
        
        plt.plot(x, param_marginals_modified[k][i](x), 
                 label = r'Update; set {}'.format(k+1), linewidth=2, linestyle='dashdot')
        
        if i==1:
            plt.axvline(0, color='k', lw=2, ls='-.')
        plt.title('Densities for ' + param_labels[i])
        plt.ylim(eval('param_{}_y_limits'.format(i)))
        plt.legend()
        plt.tight_layout()
        plt.show()
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


# Load the initial dataset containing both model data (used to generate predicted data) 
# and parameter samples from initial distribution

# !pip install mat73  # For reading in Matlab 7.3 files

import mat73 as mat73

init_data_set = mat73.loadmat('../SteelDrums/2DCase2-Feb24-2023/Prior/prior.mat')

# init_data_set keys are a1, a2, a3, data, xs, and ys

plt.figure(1)
plt.scatter(init_data_set['xs'], init_data_set['ys'], s=100, 
            c=np.std(init_data_set['data'][:,:,2], axis=0),
                     marker='s')
plt.colorbar()
plt.show()

data_init = init_data_set['data']

a0_init = init_data_set['a1']  # initial samples of first parameter
a1_init = init_data_set['a2']  # initial samples of second parameter
a2_init = init_data_set['a3']

# Now load/analyze the data-generating dataset

obs_data_set = mat73.loadmat('../SteelDrums/2DCase2-Feb24-2023/Observed/observed.mat')

# obs_data_set keys are a1, a2, a3, data, xs, and ys

data_obs = obs_data_set['data']

a0_obs = obs_data_set['a1']  # samples of first parameter responsible for observed data
a1_obs = obs_data_set['a2']  # samples of second parameter responsible for observed data
a2_obs = obs_data_set['a3']


# Set the precision of observable data and number of QoI to learn from 
# each simulated experiment

# print(np.var(data_init[:,:,0]))  # 2nd most variation
# print(np.var(data_init[:,:,1]))  # 1st most variation
# print(np.var(data_init[:,:,2]))  # 3rd most variation

predicted_precision = 2
num_predicted = 500

observed_precision = 2
num_obs = 50

num_qoi = [2, 3, 1]

learn_list = []
num_sets = 3

set_order = [0, 1, 2]

for j in range(num_sets):
    predicted_data = np.around(data_init[:num_predicted, :, set_order[j]], predicted_precision)
    learn_list.append(LUQ(predicted_data))
    learn_list[j].num_clusters = None
    
    learn_list[j].learn_qois_and_transform(num_qoi=num_qoi[j])
    
    observed_data = np.around(data_obs[:num_obs, :, set_order[j]], observed_precision)
    learn_list[j].set_observations(observed_data)
    
    learn_list[j].classify_and_transform_observations()

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
        if j==0:
            pi_predict_kdes[j].append(GKDE(learn_list[j].predict_maps[i].T))
            pi_obs_kdes[j].append(GKDE(learn_list[j].obs_maps[i].T))
        elif j==1:
            pi_predict_kdes[j].append(GKDE(learn_list[j].predict_maps[i].T, 
                                      weights = r_vals[j-1][i]))
            pi_obs_kdes[j].append(GKDE(learn_list[j].obs_maps[i].T))
        else:
            r1 = r_vals[j-2][i]
            r2 = r_vals[j-1][i]
            pi_predict_kdes[j].append(GKDE(learn_list[j].predict_maps[i].T, 
                                      weights = r1*r2))
            pi_obs_kdes[j].append(GKDE(learn_list[j].obs_maps[i].T))
            
        r_vals[j].append(
                    np.divide(
                        pi_obs_kdes[j][i](
                        learn_list[j].predict_maps[i].T), 
                        pi_predict_kdes[j][i](
                        learn_list[j].predict_maps[i].T)))
        r_means[j].append(np.mean(r_vals[j][i]))
print(f'Diagnostics: {r_means}')

print(f'Second iteration diagnostic: {np.mean(r1*r2)}')

print(f'Third iteration diagnostic: {np.mean(r1*r2*r_vals[-1][0])}')


# Construct the iterated updated marginals

params = np.vstack((a0_init[:num_predicted], 
                    a1_init[:num_predicted], 
                    a2_init[:num_predicted])).T

params_obs = np.vstack((a0_obs[:num_obs], 
                        a1_obs[:num_obs], 
                        a2_obs[:num_obs])).T

param_marginals = []
true_param_marginals = []
lam_ptr = []

def compute_iter_r(r_list, j):
    r_iter = r_list[0][0]
    for j in range(1,j+1):
        r_iter = r_iter * r_list[j][0]
    return r_iter

for j in range(num_sets):
    r_iter = compute_iter_r(r_vals, j)
    # print(r_iter[:10])
    true_param_marginals.append([])
    param_marginals.append([])
    for i in range(params.shape[1]):
        true_param_marginals[j].append(GKDE(params_obs[:,i]))
        param_marginals[j].append([])
        for k in range(learn_list[j].num_clusters):
            param_marginals[j][i].append(GKDE(params[:, i], weights=r_iter))

x = np.linspace(0.64-0.064,0.64+0.064,20)

plt.figure(4)
plt.clf()
plt.plot(x,true_param_marginals[j][0](x), linestyle='dashed', linewidth=2, label='KDE of DG')
plt.plot(x,1/(x.max()-x.min())*np.ones(len(x)), linewidth=1)

markers = ['o', '+', 's']
for j in range(num_sets):
    plt.plot(x,param_marginals[j][0][0](x), linestyle='dashdot', 
             linewidth=2, marker=markers[j], 
             label='Update; iter='+str(j))
    
plt.legend()
plt.title('Estimated and Exact Variation in $a_0$')    
plt.tight_layout()
plt.show()

x = np.linspace(0.8-0.08,0.8+0.08,20)

plt.figure(5)
plt.clf()
plt.plot(x,true_param_marginals[j][1](x), linestyle='dashed', linewidth=2, label='KDE of DG')
plt.plot(x,1/0.16*np.ones(len(x)), linewidth=1)

markers = ['o', '+', 's']
for j in range(num_sets):
    plt.plot(x,param_marginals[j][1][0](x), 
             linestyle='dashdot', linewidth=2, marker=markers[j],
             label='Update; iter='+str(j))
    
plt.legend()
plt.title('Estimated and Exact Variation in $a_1$')
plt.tight_layout()
plt.show()
# plt.savefig('prelimresults-a0-pdfs.png', bbox_inches='tight')

x = np.linspace(1-0.1,1+0.1,20)

plt.figure(6)
plt.clf()
plt.plot(x,true_param_marginals[j][2](x), linestyle='dashed', linewidth=2, label='KDE of DG')
plt.plot(x,1/0.2*np.ones(len(x)), linewidth=2)

markers = ['o', '+', 's']
for j in range(num_sets):
    plt.plot(x,param_marginals[j][2][0](x), 
             linestyle='dashdot', linewidth=2, marker=markers[j],
             label='Update; iter='+str(j))

plt.legend()
plt.title('Estimated and Exact Variation in $a_2$',)
plt.tight_layout()
plt.show()
# plt.savefig('prelimresults-a0-pdfs.png', bbox_inches='tight')

from scipy.integrate import quad as quad

xmins = [0.64-0.064, 0.8-0.08, 1-0.1]
xmaxs = [0.64+0.064, 0.8+0.08, 1+0.1]

for i in range(3):
    print('~'*100)
    print('a'+str(i)+' TV metrics')
    print('~'*100)
    for j in range(3):
        print('iter = ' + str(j) + ', TV metric = {:.3f}'.format(
            quad(lambda x: 0.5*np.abs(true_param_marginals[j][i](x) - param_marginals[j][i][0](x)),
                 xmins[i], xmaxs[i],
                 full_output=1)[0]))
        

# Copyright 2019 Steven Mattis and Troy Butler

import numpy as np
import dynamical_systems as ds
from luq import *

# Uniformly sample the parameter samples to form a "prediction" or "test" set
num_samples = int(500)

params = np.random.uniform(size=(num_samples, 2))
params[:, 0] = 0.1 * params[:, 0] + 0.05
params[:, 1] = 0.95 * params[:, 1] + 0.05
ics = 2.0 * np.random.uniform(size=(num_samples, 2))



# Construct the predicted time series data

num_time_preds = int(1000)  # number of predictions (uniformly space) between [time_start,time_end]
time_start = 0.5
time_end = 40.0
times = np.linspace(time_start, time_end, num_time_preds)

phys = ds.Selkov()
predicted_time_series = phys.solve(ics=ics, params=params, t_eval=times)

num_obs = int(100)

true_a = 2
true_b = 2

params_obs = np.random.beta(size=(num_obs, 2), a=true_a, b=true_b)
params_obs[:, 0] = 0.1 * params_obs[:, 0] + 0.05
params_obs[:, 1] = 0.95 * params_obs[:, 1] + 0.05
ics_obs = 2.0 * np.random.beta(size=(num_obs, 2), a=true_a, b=true_b)


observed_time_series = phys.solve(ics=ics_obs, params=params_obs, t_eval=times)

# Add noise if desired
with_noise = False
noise_stdev = 0.05

if with_noise:
    observed_time_series += noise_stdev * np.random.randn(num_obs)

# Use LUQ to learn dynamics and QoIs
learn = LUQ(predicted_time_series, observed_time_series, times)

# time array indices over which to use
time_start_idx = 500
time_end_idx = 999

# Clean data
learn.clean_data(time_start_idx=time_start_idx, time_end_idx=time_end_idx,
                 num_clean_obs=100, tol=3.0e-2, min_knots=15, max_knots=40)
learn.dynamics(cluster_method='kmeans', kwargs={'n_clusters': 2, 'n_init': 10})

import matplotlib.pyplot as plt

for j in range(learn.num_clusters):
    plt.figure()
    for i in range(num_samples):
        if learn.predict_labels[i] == j:
            plt.plot(times, predicted_time_series[i, :])
    plt.show()

learn.learn_qois_and_transform()


import pdb
pdb.set_trace()
# Copyright 2019 Steven Mattis and Troy Butler

import matplotlib.pyplot as plt
import numpy as np
from luq import *
import luq.dynamical_systems as ds

# Uniformly sample the parameter samples to form a "prediction" or "test" set
num_samples = int(500)

params = np.random.uniform(size=(num_samples, 1))
ics = 0.25 * np.random.uniform(size=(num_samples, 2)) + 0.25

#ics = 2.0 * ics - 1.0
params = params - 0.5

# Construct the predicted time series data

# number of predictions (uniformly space) between [time_start,time_end]
num_time_preds = int(500)
time_start = 0.5
time_end = 40.0
times = np.linspace(time_start, time_end, num_time_preds)

phys = ds.Lienard()
predicted_time_series = phys.solve(ics=ics, params=params, t_eval=times)

num_obs = int(100)

true_a = 2
true_b = 2

params_obs = np.random.beta(size=(num_obs, 1), a=true_a, b=true_b)
ics_obs = 0.25 * np.random.beta(size=(num_obs, 2), a=true_a, b=true_b) + 0.25

#ics_obs = 2.0 * ics_obs - 1.0
params_obs = params_obs - 0.5

observed_time_series = phys.solve(ics=ics_obs, params=params_obs, t_eval=times)

# Add noise if desired
with_noise = False
noise_stdev = 0.05

if with_noise:
    observed_time_series += noise_stdev * np.random.randn(num_obs)

# import matplotlib.pyplot as plt
# plt.figure()
# for i in range(num_samples):
#     plt.plot(times, predicted_time_series[i,:])
# plt.show()
#
# plt.figure()
# for i in range(num_obs):
#     plt.plot(times, observed_time_series[i,:])
# plt.show()

# Use LUQ to learn dynamics and QoIs
learn = luq.LUQ(predicted_time_series, observed_time_series, times)

# time array indices over which to use
time_start_idx = 350
time_end_idx = 499

# Filter data
learn.filter_data(time_start_idx=time_start_idx, time_end_idx=time_end_idx,
                  num_filtered_obs=50, tol=3.0e-2, min_knots=15, max_knots=40)
learn.dynamics(
    cluster_method='spectral',
    kwargs={
        'n_clusters': 2,
        'n_init': 10})


for j in range(learn.num_clusters):
    plt.figure()
    pos = 0
    neg = 0
    for i in range(num_samples):
        if learn.predict_labels[i] == j:
            if params[i] < 0.0:
                neg += 1
            else:
                pos += 1
            plt.plot(times, predicted_time_series[i, :])
    print(pos / (pos + neg), 'positive nu ratio for cluster', j)
    plt.show()

learn.learn_qois_and_transform()

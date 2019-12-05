# Copyright 2019 Steven Mattis and Troy Butler
import numpy as np
import dynamical_systems as ds
from luq import *

# A ***true*** distribution of $c, \omega_0, a$, and $b$ are defined by (non-uniform)
# Beta distributions and used to generate a set of time series data.
#
# An ***initial*** uniform distribution is assumed and updated by the true time series data.

# ### Time series data appended with differences and sums
#
# We initially assume no errors in the time series data, i.e., the observations are $y(t)$ at some finite set of times
# $\{t_i\}_{i=1}^N$, with $0\leq t_1 < t_2 < \cdots < t_N$.
#
# We also take differences and summations of the time series (to extract derivative and integral type information) and
# append to the data to determine if new/dominant features (i.e., principal components) are found.

# Uniformly sample the parameter samples to form a "prediction" or "test" set
num_samples = int(1E3)

params = np.random.uniform(size=(num_samples, 2))
ics = np.random.uniform(size=(num_samples, 2))

params[:, 0] = 0.1 + 0.4 * params[:, 0]  # c
params[:, 1] = 0.5 + 1.5 * params[:, 1]  # omega_0
ics[:, 0] = 1 + 1 * ics[:, 0]  # a
ics[:, 1] = -1 + 1 * ics[:, 1]   # b

# Construct the predicted time series data

num_time_preds = int(50)  # number of predictions (uniformly space) between [time_start,time_end]
time_start = 0.5
time_end = 3.5
times = np.linspace(time_start, time_end, num_time_preds)

phys = ds.HarmonicOscillator()
predicted_time_series = phys.solve(ics=ics, params=params, t_eval=times)


# Simulate an observed Beta distribution of time series data

num_obs = int(1E3)

true_a = 2
true_b = 2

params_obs = np.random.beta(size=(num_obs, 2), a=true_a, b=true_b)
ics_obs = np.random.beta(size=(num_obs, 2), a=true_a, b=true_b)

params_obs[:, 0] = 0.1 + 0.4 * params_obs[:, 0]  # c
params_obs[:, 1] = 0.5 + 1.5 * params_obs[:, 1]  # omega_0
ics_obs[:, 0] = 1 + 1 * ics_obs[:, 0]  # a
ics_obs[:, 1] = -1 + 1 * ics_obs[:, 1]   # b

observed_time_series = phys.solve(ics=ics_obs, params=params_obs, t_eval=times)


# Add noise if desired
with_noise = False
noise_stdev = 0.05

if with_noise:
    observed_time_series += noise_stdev * np.random.randn(num_obs)


# Use LUQ to learn dynamics and QoIs
learn = LUQ(predicted_time_series, observed_time_series, times)

# time array indices over which to use
time_start_idx = 20
time_end_idx = 49

# Clean data
learn.clean_data(time_start_idx=time_start_idx, time_end_idx=time_end_idx,
                     num_clean_obs=50, tol=1.0e-2, min_knots=5, max_knots=15)
learn.dynamics()
learn.learn_qois_and_transform()


#!/usr/bin/env python

# Copyright 2019 Steven Mattis and Troy Butler
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde as GKDE
from scipy.stats import beta
import scipy.integrate.quadrature as quad
from luq.luq import *
from luq.dynamical_systems import HarmonicOscillator

plt.rcParams.update({'font.size': 22})
plt.rcParams.update({'axes.linewidth': 2})

np.random.seed(123456)


# Model is for harmonic motion
# $$y''(t) + 2cy'(t) + \omega_0^2 y = f(t)$$
# with damping constant
# $$c \in [0.1,1]$$
# and natural frequency
# $$\omega_0\in[0.5,1]$$
# and forcing term initially taken to be zero.
#
# Note that with the ranges of $c$ and $\omega_0$ above, it is possible for the system
# to either be under-, over-, or critically damped (and since $c\geq 0.1$ it is never undamped,
# which is almost always physical nonsense).
#
# The roots to the characteristic equation are given by
# $$ r_{1,2} = -c\pm \sqrt{c^2-\omega_0^2}.$$
#
# When the system is under-damped, the solution is given by
# $$ y(t) = e^{-ct}[C_1\cos(\omega t) + C_2\sin(\omega t)], \ \omega=\sqrt{\omega_0^2-c^2}. $$
#
#
# When the system is over-damped, the solution is given by
# $$ y(t) = C_1 e^{r_1t}+C_2 e^{r_2t}. $$
#
# And, finally, when the system is critically damped, the solution is given by
# $$ y(t) = C_1e^{-ct} + C_2 te^{-ct}. $$
#
# However, we never expect the system to be critically damped in practice since this is "too fine-tuned" of a scenario.
#
# The constants $C_1$ and $C_2$ are determined by the initial conditions, which we assume to be given by
# $$ y(0)=a, y'(0) = b $$
# where
# $$ a\in[1,2] $$
# and
# $$ b\in[-1,0] $$.
#
# In the under-damped case,
# $$ C_1 = a, \ \text{and } \ C_2 = \frac{b+ca}{\omega}. $$
#
# In the over-damped case,
# $$ C_1 = \frac{b-ar_2}{r_1-r_2}, \ \text{and } \ C_2 = \frac{b-r_1a}{r_2-r_1} $$
#
# A ***true*** distribution of $c, \omega_0, a$, and $b$ are defined by (non-uniform)
# Beta distributions and used to generate a set of time series data.
#
# An ***initial*** uniform distribution is assumed and updated by the true
# time series data.


# Uniformly sample the parameter samples to form a "prediction" or "test" set
num_samples = int(2E3)

params = np.random.uniform(size=(num_samples, 2))
ics = np.random.uniform(size=(num_samples, 2))

param_range = np.array([[0.1, 1.0],  # c
                        [0.5, 1.0]])  # omega_0
ic_range = np.array([[3.0, 3.0],  # a
                     [0.0, 0.0]])  # b
params = param_range[:, 0] + (param_range[:, 1] - param_range[:, 0]) * params
ics = ic_range[:, 0] + (ic_range[:, 1] - ic_range[:, 0]) * ics
param_labels = [r'$c$', r'$\omega_0$']
ic_labels = [r'$a$', r'$b$']

# Construct the predicted time series data

# number of predictions (uniformly space) between [time_start,time_end]
num_time_preds = int(501)
time_start = 1.0
time_end = 6.0
times = np.linspace(time_start, time_end, num_time_preds)

phys = HarmonicOscillator()
predicted_time_series = phys.solve(ics=ics, params=params, t_eval=times)


# Simulate an observed Beta distribution of time series data

num_obs = int(3E2)

true_a = 2
true_b = 2

params_obs = np.random.beta(size=(num_obs, 2), a=true_a, b=true_b)
ics_obs = np.random.beta(size=(num_obs, 2), a=true_a, b=true_b)
params_obs = param_range[:, 0] + \
    (param_range[:, 1] - param_range[:, 0]) * params_obs
ics_obs = ic_range[:, 0] + (ic_range[:, 1] - ic_range[:, 0]) * ics_obs

observed_time_series = phys.solve(ics=ics_obs, params=params_obs, t_eval=times)

# Add noise if desired
with_noise = True
noise_stdev = 0.25

if with_noise:
    observed_time_series += noise_stdev * \
        np.random.randn(num_obs, num_time_preds)


# Use LUQ to learn dynamics and QoIs
learn = LUQ(predicted_time_series, observed_time_series, times)

# time array indices over which to use
time_start_idx = 0
time_end_idx = num_time_preds - 1

num_filtered_obs = 16

# Filter data with piecewise linear splines
learn.filter_data(
    time_start_idx=time_start_idx,
    time_end_idx=time_end_idx,
    num_filtered_obs=num_filtered_obs,
    tol=5.0e-2,
    min_knots=3,
    max_knots=10)

# learn and classify dynamics
# learn.dynamics(cluster_method='gmm', kwargs={'n_components': 3})
learn.dynamics(kwargs={'n_clusters': 3, 'n_init': 10})


fig = plt.figure(figsize=(10, 8))

chosen_obs = [0, 8, 10]
colors = ['r', 'g', 'b']

for i, c in zip(chosen_obs, colors):
    plt.plot(learn.times[time_start_idx:time_end_idx],
             learn.observed_time_series[i,
                                        time_start_idx:time_end_idx],
             color=c,
             linestyle='none',
             marker='.',
             markersize=10,
             alpha=0.25)

for i in chosen_obs:
    num_i_knots = int(0.5 * (2 + len(learn.obs_knots[i])))
    knots = np.copy(learn.obs_knots[i][num_i_knots:])
    knots = np.insert(knots, 0, learn.times[time_start_idx])
    knots = np.append(knots, learn.times[time_end_idx])
    plt.plot(knots,
             learn.obs_knots[i][:num_i_knots],
             'k',
             linestyle='dashed',
             markersize=15,
             marker='o',
             linewidth=2)

plt.xlabel('$t$')
plt.ylabel('$y(t)$')
plt.title('Approximating Dynamics')
plt.show()


fig = plt.figure(figsize=(10, 8))

for i, c in zip(chosen_obs, colors):
    plt.plot(learn.times[time_start_idx:time_end_idx],
             learn.observed_time_series[i,
                                        time_start_idx:time_end_idx],
             color=c,
             linestyle='none',
             marker='.',
             markersize=10,
             alpha=0.25)

for i in chosen_obs:
    plt.plot(learn.filtered_times,
             learn.filtered_obs[i, :],
             'k',
             linestyle='none',
             marker='s',
             markersize=12)

plt.xlabel('$t$')
plt.ylabel('$y(t)$')
plt.title('Generating Filtered Data')
plt.show()


# # Plot clusters of predicted time series

for j in range(learn.num_clusters):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(
        24, 8), gridspec_kw={'width_ratios': [1, 1]})
    ax1.scatter(
        np.tile(
            learn.filtered_times,
            num_samples).reshape(
            num_samples,
            num_filtered_obs),
        learn.filtered_predictions,
        50,
        c='gray',
        marker='.',
        alpha=0.2)
    idx = np.where(learn.predict_labels == j)[0]
    ax1.scatter(np.tile(learn.filtered_times,
                        len(idx)).reshape(len(idx),
                                          num_filtered_obs),
                learn.filtered_predictions[idx,
                                        :],
                50,
                c='b',
                marker='o',
                alpha=0.2)
    ax1.set(title='Cluster ' + str(j + 1) + ' in data')
    ax1.set_xlabel('$t$')
    ax1.set_ylabel('$y(t)$')

    ax2.scatter(params[:, 0], params[:, 1], 30,
                c='gray', marker='.', alpha=0.2)
    ax2.scatter(params[idx, 0], params[idx, 1], 50, c='blue', marker='o')
    ax2.set(title='Cluster ' + str(j + 1) + ' in parameters')
    ax2.set_ylabel(r'$\omega_0$')
    ax2.set_xlabel('$c$')
    fig.tight_layout
    plt.show()


# Plot oberved and predicted clusters

for j in range(learn.num_clusters):
    fig = plt.figure(figsize=(10, 8))
    plt.scatter(
        np.tile(
            learn.filtered_times,
            num_samples).reshape(
            num_samples,
            num_filtered_obs),
        learn.filtered_predictions,
        10,
        c='gray',
        marker='.',
        alpha=0.2)
    idx = np.where(learn.predict_labels == j)[0]
    plt.scatter(np.tile(learn.filtered_times,
                        len(idx)).reshape(len(idx),
                                          num_filtered_obs),
                learn.filtered_predictions[idx,
                                        :],
                20,
                c='b',
                marker='o',
                alpha=0.3)
    idx = np.where(learn.obs_labels == j)[0]
    plt.scatter(np.tile(learn.filtered_times, len(idx)).reshape(len(idx), num_filtered_obs),
                learn.filtered_obs[idx, :], 50, c='r', marker='s', alpha=0.2)
    plt.title('Classifying filtered observations')
    plt.xlabel('$t$')
    plt.ylabel('$y(t)$')
    bottom, top = plt.gca().get_ylim()
    props = dict(boxstyle='round', facecolor='gray', alpha=0.2)
    plt.text(1, (top - bottom) * 0.1 + bottom,
             'Cluster ' + str(j + 1),
             {'color': 'k', 'fontsize': 20},
             bbox=props)
    plt.text
    fig.tight_layout
    plt.show()


# Find best KPCA transformation for given number of QoI and transform time
# series data.
predict_map, obs_map = learn.learn_qois_and_transform(num_qoi=2)


def plot_gap(all_eig_vals, n, cluster):
    fig = plt.figure(figsize=(10, 10))
    fig.clear()
    # Plotting until maximum number of knots
    eig_vals = all_eig_vals[cluster].lambdas_[0:10]
    plt.semilogy(
        np.arange(
            np.size(eig_vals)) +
        1,
        eig_vals /
        np.sum(eig_vals) *
        100,
        Marker='.',
        MarkerSize=20,
        linestyle='')
    plt.semilogy(
        np.arange(
            np.size(eig_vals)) +
        1,
        eig_vals[n] /
        np.sum(eig_vals) *
        100 *
        np.ones(
            np.size(eig_vals)),
        'k--')
    plt.semilogy(np.arange(np.size(eig_vals)) +
                 1, eig_vals[n +
                             1] /
                 np.sum(eig_vals) *
                 100 *
                 np.ones(np.size(eig_vals)), 'r--')
    plt.text(n + 1, eig_vals[n] / np.sum(eig_vals) * 150,
             r'%2.3f' % (np.sum(eig_vals[0:n + 1]) / np.sum(eig_vals) * 100) +
             '% of variation explained by first ' + '%1d' % (n + 1) + ' PCs.',
             {'color': 'k', 'fontsize': 20})
    plt.text(n +
             2, eig_vals[n +
                         1] /
             np.sum(eig_vals) *
             150, r'Order of magnitude of gap is %4.2f.' %
             (np.log10(eig_vals[n]) -
              np.log10(eig_vals[n +
                                1])), {'color': 'r', 'fontsize': 20})
    s = 'Determining QoI for cluster #%1d' % (cluster + 1)
    plt.title(s)
    plt.xlabel('Principal Component #')
    plt.ylabel('% of Variation')
    plt.xlim([0.1, np.size(eig_vals) + 1])
    plt.ylim([0, 500])
    plt.show()


plot_gap(all_eig_vals=learn.kpcas, n=1, cluster=0)
plot_gap(all_eig_vals=learn.kpcas, n=1, cluster=1)
plot_gap(all_eig_vals=learn.kpcas, n=1, cluster=2)

# Generate kernel density estimates on new QoI
learn.generate_kdes()
# Calculate rejection rates for each cluster and print averages.
r_vals = learn.compute_r()

param_marginals = []
ic_marginals = []
true_param_marginals = []
true_ic_marginals = []
lam_ptr = []
cluster_weights = []
for i in range(learn.num_clusters):
    lam_ptr.append(np.where(learn.predict_labels == i)[0])
    cluster_weights.append(len(np.where(learn.obs_labels == i)[0]) / num_obs)

for i in range(params.shape[1]):
    true_param_marginals.append(GKDE(params_obs[:, i]))
    param_marginals.append([])
    for j in range(learn.num_clusters):
        param_marginals[i].append(
            GKDE(params[lam_ptr[j], i], weights=learn.r[j]))


def unif_dist(x, p_range):
    y = np.zeros(x.shape)
    val = 1.0 / (p_range[1] - p_range[0])
    for i, xi in enumerate(x):
        if xi < p_range[0] or xi > p_range[1]:
            y[i] = 0
        else:
            y[i] = val
    return y


for i in range(params.shape[1]):
    fig = plt.figure(figsize=(10, 10))
    fig.clear()
    x_min = min(min(params[:, i]), min(params_obs[:, i]))
    x_max = max(max(params[:, i]), max(params_obs[:, i]))
    delt = 0.25 * (x_max - x_min)
    x = np.linspace(x_min - delt, x_max + delt, 100)
    plt.plot(x, unif_dist(x, param_range[i, :]),
             label='Initial', linewidth=2)
    mar = np.zeros(x.shape)
    for j in range(learn.num_clusters):
        mar += param_marginals[i][j](x) * cluster_weights[j]
    plt.plot(x, mar, label='Updated', linewidth=4, linestyle='dashed')
    plt.plot(
        x,
        true_param_marginals[i](x),
        label='Data-generating',
        linewidth=4,
        linestyle='dotted')
    plt.title('Densities for parameter ' + param_labels[i], fontsize=20)
    plt.legend(fontsize=20)
    plt.show()

# ### Compute TV metric between densities


def param_init_error(x):
    return np.abs(unif_dist(
        x, param_range[param_num, :]) - true_param_marginals[param_num](x))


for i in range(params.shape[1]):
    param_num = i
    TV_metric = quad(param_init_error,
                     param_range[i, 0], param_range[i, 1], maxiter=1000)
    print(TV_metric)


def param_update_KDE_error(x):
    mar = np.zeros(x.shape)
    for j in range(learn.num_clusters):
        mar += param_marginals[param_num][j](x) * cluster_weights[j]
    return np.abs(mar - true_param_marginals[param_num](x))


for i in range(params.shape[1]):
    param_num = i
    TV_metric = quad(param_update_KDE_error,
                     param_range[i, 0], param_range[i, 1], maxiter=1000)
    print(TV_metric)


def KDE_error(x):
    true_beta = beta(a=true_a,
                     b=true_b,
                     loc=param_range[i,
                                     0],
                     scale=param_range[i,
                                       1] - param_range[i,
                                                        0])
    return np.abs(true_beta.pdf(x) - true_param_marginals[param_num](x))


for i in range(params.shape[1]):
    param_num = i
    TV_metric = quad(
        KDE_error, param_range[i, 0], param_range[i, 1], maxiter=1000)
    print(TV_metric)

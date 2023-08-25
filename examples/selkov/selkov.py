#!/usr/bin/env python

# Copyright 2019-2020 Steven Mattis and Troy Butler

import matplotlib.pyplot as plt
from scipy.integrate import quadrature as quad
from scipy.stats import beta
from scipy.stats import gaussian_kde as GKDE
from luq.luq import *
from luq.dynamical_systems import Selkov


plt.rcParams.update({'font.size': 22})
plt.rcParams.update({'axes.linewidth': 2})

np.random.seed(123456)


# The model is the Sel'kov model for glycolysis, a process by which living cells breakdown sugar to obtain energy:
# $$x' = -(x+b) + a \left(y + \frac{b}{a+b^2} \right) + (x+b)^2 \left(y + \frac{b}{a+b^2}\right)$$
# $$y' = b-a\left(y+ \frac{b}{a+b^2}\right) - (x+b)^2 \left(y + \frac{b}{a+b^2}\right), $$
# where $x$ and $y$ represent concentrations of ADP and F6P, respectively, and $a,b>0$.
# The initial conditions are $x(0) = x_0 \in \mathbb{R}$ and $y(0) = y_0 \in \mathbb{R}$.
#
# The system has Hopf Bifurcations at
# $$b = b_1(a) = \sqrt{(1-\sqrt{1-8a}-2a)/2}$$
# and
# $$b = b_2(a) = \sqrt{(1+\sqrt{1-8a}-2a)/2}.$$
# If $b<b_1$, the origin is a stable focus. If $b_1 < b < b_2$, there is a stable periodic orbit.
# If $b > b_2$ the origin is a stable focus.
#
# The system is solved numerically using the RK45 method.
#
# A ***true*** distribution of $a, b,  x_0$, and $y_0$ are defined by (non-uniform)
# Beta distributions and used to generate a set of time series data.
#
# An ***initial*** uniform distribution is assumed and updated by the true
# time series data.


# Uniformly sample the parameter samples to form a "prediction" or "test" set
num_samples = int(3E3)

param_range = np.array([[0.01, 0.124],  # a
                        [0.05, 1.5]])  # b
ic_range = np.array([[1.0, 1.0],  # y_0
                     [1.0, 1.0]])  # x_0

params = np.random.uniform(size=(num_samples, 2))
params = param_range[:, 0] + (param_range[:, 1] - param_range[:, 0]) * params

ics = np.random.uniform(size=(num_samples, 2))
ics = ic_range[:, 0] + (ic_range[:, 1] - ic_range[:, 0]) * ics

# labels
param_labels = [r'$a$', r'$b$']
ic_labels = [r'$x_0$', r'$y_0$']

# Construct the predicted time series data
time_start = 2.0  # 0.5
time_end = 6.5  # 40.0
# number of predictions (uniformly space) between [time_start,time_end]
num_time_preds = int((time_end - time_start) * 100)
times = np.linspace(time_start, time_end, num_time_preds)

# Solve systems
phys = Selkov()
predicted_time_series = phys.solve(ics=ics, params=params, t_eval=times)


# Simulate an observed Beta distribution of time series data

num_obs = int(5E2)

true_a = 2
true_b = 2

params_obs = np.random.beta(size=(num_obs, 2), a=true_a, b=true_b)
params_obs = param_range[:, 0] + \
    (param_range[:, 1] - param_range[:, 0]) * params_obs

ics_obs = np.random.beta(size=(num_obs, 2), a=true_a, b=true_b)
ics_obs = ic_range[:, 0] + (ic_range[:, 1] - ic_range[:, 0]) * ics_obs

# Solve system
observed_time_series = phys.solve(ics=ics_obs, params=params_obs, t_eval=times)

# Add noise if desired
with_noise = True
noise_stdev = 0.0125

if with_noise:
    observed_time_series += noise_stdev * np.random.randn(num_obs, times.shape[0])

# Use LUQ to learn dynamics and QoIs
learn = LUQ(predicted_data=predicted_time_series, 
            observed_data=observed_time_series)

# time array indices over which to use
time_start_idx = 0
time_end_idx = len(times) - 1  # 150 #120

num_filtered_obs = 20

filtered_times = np.linspace(times[time_start_idx],
                             times[time_end_idx],
                             num_filtered_obs)

# Filter data with piecewise linear splines
learn.filter_data(filter_method='splines',
                  predicted_data_coordinates=times,
                  observed_data_coordinates=times,
                  filtered_data_coordinates=filtered_times,
                  tol=5.0e-2, 
                  min_knots=3, 
                  max_knots=12)

# Learn and classify dynamics
learn.dynamics(cluster_method='kmeans', kwargs={'n_clusters': 3, 'n_init': 10})


fig = plt.figure(figsize=(10, 8))

chosen_obs = [0, 1, 499]
colors = ['r', 'g', 'b']

for i, c in zip(chosen_obs, colors):
    plt.plot(learn.observed_data_coordinates[time_start_idx:time_end_idx + 1],
             learn.observed_data[i,
                                 time_start_idx:time_end_idx + 1],
             color=c,
             linestyle='none',
             marker='.',
             markersize=10,
             alpha=0.25)

for i in chosen_obs:
    num_i_knots = int(0.5 * (2 + len(learn.obs_knots[i])))
    knots = np.copy(learn.obs_knots[i][num_i_knots:])
    knots = np.insert(knots, 0, learn.filtered_data_coordinates[0])
    knots = np.append(knots, learn.filtered_data_coordinates[-1])
    plt.plot(knots,
             learn.obs_knots[i][:num_i_knots],
             'k',
             linestyle='dashed',
             markersize=15,
             marker='o',
             linewidth=2)

plt.xlabel('$t$')
plt.ylabel('$x(t)$')
plt.title('Approximating Dynamics')
plt.show()

fig = plt.figure(figsize=(10, 8))

for i, c in zip(chosen_obs, colors):
    plt.plot(learn.observed_data_coordinates[time_start_idx:time_end_idx + 1],
             learn.observed_data[i,
                                 time_start_idx:time_end_idx + 1],
             color=c,
             linestyle='none',
             marker='.',
             markersize=10,
             alpha=0.25)

for i in chosen_obs:
    plt.plot(learn.filtered_data_coordinates,
             learn.filtered_obs[i, :],
             'k',
             linestyle='none',
             marker='s',
             markersize=12)

plt.xlabel('$t$')
plt.ylabel('$x(t)$')
plt.title('Generating Filtered Data')
plt.show()

# # Plot clusters of predicted time series
num_filtered_obs = learn.filtered_data_coordinates.shape[0]
for j in range(learn.num_clusters):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(
        24, 8), gridspec_kw={'width_ratios': [1, 1]})
    ax1.scatter(
        np.tile(
            learn.filtered_data_coordinates,
            num_samples).reshape(
            num_samples,
            num_filtered_obs),
        learn.filtered_predictions,
        50,
        c='gray',
        marker='.',
        alpha=0.2)
    idx = np.where(learn.predict_labels == j)[0]
    ax1.scatter(np.tile(learn.filtered_data_coordinates,
                        len(idx)).reshape(len(idx),
                                          num_filtered_obs),
                learn.filtered_predictions[idx, :],
                50,
                c='b',
                marker='o',
                alpha=0.2)
    idx2 = np.where(learn.obs_labels == j)[0]
    ax1.scatter(np.tile(learn.filtered_data_coordinates, len(idx2)).reshape(len(idx2), num_filtered_obs),
                learn.filtered_obs[idx2, :], 50, c='r', marker='s', alpha=0.2)
    ax1.set(title='Cluster ' + str(j + 1) + ' in data')
    ax1.set_xlabel('$t$')
    ax1.set_ylabel('$x(t)$')

    ax2.scatter(params[:, 0], params[:, 1], 30,
                c='gray', marker='.', alpha=0.2)
    ax2.scatter(params[idx, 0], params[idx, 1], 50, c='blue', marker='o')
    ax2.set(title='Cluster ' + str(j + 1) + ' in parameters')
    ax2.set_ylabel(param_labels[1])
    ax2.set_xlabel(param_labels[0])
    xs = np.linspace(param_range[0, 0], param_range[0, 1], 100)
    ys1 = np.sqrt(0.5 * (1.0 - np.sqrt(1.0 - 8.0 * xs) - 2.0 * xs))
    ys2 = np.sqrt(0.5 * (1.0 + np.sqrt(1.0 - 8.0 * xs) - 2.0 * xs))
    ax2.plot(xs, ys1, 'r-', linewidth=3)
    ax2.plot(xs, ys2, 'r-', linewidth=3)
    fig.tight_layout
    plt.show()

# Plot oberved and predicted clusters

for j in range(learn.num_clusters):
    fig = plt.figure(figsize=(10, 8))
    plt.scatter(
        np.tile(
            learn.filtered_data_coordinates,
            num_samples).reshape(
            num_samples,
            num_filtered_obs),
        learn.filtered_predictions,
        10,
        c='gray',
        marker='.',
        alpha=0.2)
    idx = np.where(learn.predict_labels == j)[0]
    plt.scatter(np.tile(learn.filtered_data_coordinates,
                        len(idx)).reshape(len(idx),
                                          num_filtered_obs),
                learn.filtered_predictions[idx,
                                        :],
                20,
                c='b',
                marker='o',
                alpha=0.3)
    idx = np.where(learn.obs_labels == j)[0]
    plt.scatter(np.tile(learn.filtered_data_coordinates, len(idx)).reshape(len(idx), num_filtered_obs),
                learn.filtered_obs[idx, :], 50, c='r', marker='s', alpha=0.2)
    plt.title('Classifying filtered observations')
    plt.xlabel('$t$')
    plt.ylabel('$x(t)$')
    bottom, top = plt.gca().get_ylim()
    left, right = plt.gca().get_xlim()

    props = dict(boxstyle='round', facecolor='gray', alpha=0.2)
    plt.text(right - 1, top - 0.2,
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
    #  Plotting until maximum number of knots
    eig_vals = all_eig_vals[cluster].eigenvalues_[0:10]
    plt.semilogy(
        np.arange(
            np.size(eig_vals)) +
        1,
        eig_vals /
        np.sum(eig_vals) *
        100,
        marker='.',
        markersize=20,
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
    plt.ylim([1e-5, 500])
    plt.show()


plot_gap(all_eig_vals=learn.kpcas, n=1, cluster=0)
plot_gap(all_eig_vals=learn.kpcas, n=1, cluster=1)

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

# Compute marginal probabilities for each parameter and initial condition.
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
            GKDE(params[lam_ptr[j], i], weights=r_vals[j]))


# uniform distribution
def unif_dist(x, p_range):
    y = np.zeros(x.shape)
    val = 1.0 / (p_range[1] - p_range[0])
    for i, xi in enumerate(x):
        if xi < p_range[0] or xi > p_range[1]:
            y[i] = 0
        else:
            y[i] = val
    return y


# Plot predicted marginal densities for parameters
for i in range(params.shape[1]):
    fig = plt.figure(figsize=(10, 10))
    fig.clear()
    x_min = min(min(params[:, i]), min(params_obs[:, i]))
    x_max = max(max(params[:, i]), max(params_obs[:, i]))
    delt = 0.25 * (x_max - x_min)
    x = np.linspace(x_min - delt, x_max + delt, 100)
    plt.plot(x, unif_dist(x, param_range[i, :]),
             label='Initial', linewidth=4)
    mar = np.zeros(x.shape)
    for j in range(learn.num_clusters):
        mar += param_marginals[i][j](x) * cluster_weights[j]
    plt.plot(x, mar, label='Updated', linewidth=4, linestyle='dashed')
    plt.plot(x, true_param_marginals[i](x), label='Data-generating',
             linewidth=4, linestyle='dotted')
    plt.title('Densities for parameter ' + param_labels[i], fontsize=16)
    plt.legend(fontsize=20)
    if i == 0:
        plt.xticks([0, 0.05, 0.1, 0.15])
    else:
        plt.xticks([0, 0.5, 1., 1.5])
    plt.show()


def param_init_error(x):
    return np.abs(unif_dist(
        x, param_range[param_num, :]) - true_param_marginals[param_num](x))


for i in range(params.shape[1]):
    param_num = i
    TV_metric = quad(param_init_error,
                     param_range[i, 0], param_range[i, 1], maxiter=1000)
    print(TV_metric)


def param_update_kde_error(x):
    mar = np.zeros(x.shape)
    for j in range(learn.num_clusters):
        mar += param_marginals[param_num][j](x) * cluster_weights[j]
    return np.abs(mar - true_param_marginals[param_num](x))


for i in range(params.shape[1]):
    param_num = i
    TV_metric = quad(param_update_kde_error,
                     param_range[i, 0], param_range[i, 1], maxiter=1000)
    print(TV_metric)


def kde_error(x):
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
        kde_error, param_range[i, 0], param_range[i, 1], maxiter=1000)
    print(TV_metric)

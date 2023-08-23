#!/usr/bin/env python

# Copyright 2019-2020 Steven Mattis and Troy Butler

import matplotlib.pyplot as plt
from scipy.integrate import quadrature as quad
from scipy.stats import gaussian_kde as GKDE
from luq.luq import *
from scipy.stats import beta


plt.rcParams.update({'font.size': 22})
plt.rcParams.update({'axes.linewidth': 2})

np.random.seed(123456)


# The model is the 1D Burger's equation, a nonlinear PDE used to model fluid dynamics:
# $$q_t + \frac{1}{2} (q^2)_x = 0.$$
# The domain is the interval $[0, 10]$.
# We have an initial condition of the form
# \begin{equation*}
# q(x,0) = \begin{cases}
#       f_l & 0 \leq x\leq 3.25 -a  \\
#        \frac{1}{2} ((f_l + f_r) - (f_l - f_r) \frac{(x-3.25)}{a}) & 3.25 -a < x \leq 3.25 + a \\
#       f_r & 3.25 + a < x \leq 10,
#    \end{cases}
# \end{equation*}
# where $a \in [0, 3]$ is an uncertain parameter and $f_l$ and $f_r$ are positive constants with $f_l > f_r$.
# Take $f_l = 1.5$ and $f_r = 1$.
# We assume non-reflecting boundary conditions, allowing waves to pass out
# of the boundaries without reflection.

# In[ ]:


# Plot the initial condition given a, fl, and fr.
As = [0.75, 1.875, 3.0]
ss = ['-k', '--b', '-.r']

fl = 1.5
fr = 1
x = np.linspace(0, 10, 1000)

fig, ax = plt.subplots(1, 1)
q0 = np.zeros(x.shape)
for j, a in enumerate(As):
    for i in range(x.shape[0]):
        if x[i] <= (3.25 - a):
            q0[i] = fl
        elif x[i] > (3.25 + a):
            q0[i] = fr
        else:
            q0[i] = 0.5 * ((fl + fr) - (fl - fr) * (x[i] - 3.25) / a)
    ax.plot(x, q0, ss[j], linewidth=2, label="a=" + str(a))
ax.set_xlabel("x")
ax.set_ylabel("q(x,0)")
ax.legend()
ax.set_xticks((0, 6.5, 9.5))
ax.set_title('Initial Conditions')
ax.axvline(x=6.5, color='c')
ax.axvline(x=9.5, color='g')
plt.show()


# This system often can develop discontinuous solutions (shock waves),
# which complicates calculating a numerical solution.
# We use Clawpack (https://www.clawpack.org/) to calculate weak solutions to the system using a
# Godunov-type finite volume method with an appropriate limiter and Riemann solver.
# We use a uniform mesh with 500 elements.
#
# The system described above forms a shock at $t = \frac{2a}{f_l - f_r}$.
# The shock speed is $\frac{1}{2}(f_l + f_r)$.
#
# We calculte the time series solution at $x=7$, i.e. $q(7,t)$ at 500 evenly spaced time steps between 0 and 10.
#
# Two ***true*** distributions of $a$ are defined by (non-uniform)
# Beta distributions and used to generate a set of time series data.
#
# An ***initial*** uniform distribution is assumed and updated by the true
# time series data.

# Load precomputed time-series data.
times = np.loadtxt('burgers_files/unif_times.txt')
predicted_time_series = np.loadtxt('burgers_files/unif_series.txt')
params = np.loadtxt('burgers_files/unif_params.txt')
num_samples = predicted_time_series.shape[0]

# a=5, b=2
observed_time_series = np.loadtxt('burgers_files/beta_series_2_2.txt')
params_obs = np.loadtxt('burgers_files/beta_params_2_2.txt')
num_obs = observed_time_series.shape[0]

if len(params.shape) == 1:
    params = params.reshape((num_samples, 1))
    params_obs = params_obs.reshape((num_obs, 1))

# Add noise if desired
with_noise = True
noise_stdev = 0.025

if with_noise:
    predicted_time_series += noise_stdev * \
        np.random.randn(num_samples, times.shape[0])
    observed_time_series += noise_stdev * \
        np.random.randn(num_obs, times.shape[0])
param_range = np.array([[0.75, 3.0]])
param_labels = [r'$a$']

# Use LUQ to learn dynamics and QoIs
learn = LUQ(predicted_data=predicted_time_series, 
            observed_data=observed_time_series)

# time array indices over which to use
time_start_idx = 0
time_end_idx = 499

# Filter data with piecewise constant linear splines
learn.filter_data(
    filter_method='splines',
    data_coordinates=times,
    start_idx=time_start_idx,
    end_idx=time_end_idx,
    num_filtered_obs=500,
    tol=0.5 * noise_stdev,
    min_knots=3,
    max_knots=4)

# Learn and classify dynamics.
learn.dynamics(cluster_method='kmeans', kwargs={'n_clusters': 2, 'n_init': 10})

fig = plt.figure(figsize=(10, 8))
chosen_obs = [1, 3, 6]  #
colors = ['r', 'g', 'b']

for i, c in zip(chosen_obs, colors):
    plt.plot(learn.data_coordinates[time_start_idx:time_end_idx + 1],
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
plt.ylabel('$y(t)$')
plt.title('Approximating Dynamics')
plt.show()

fig = plt.figure(figsize=(10, 8))

for i, c in zip(chosen_obs, colors):
    plt.plot(learn.data_coordinates[time_start_idx:time_end_idx + 1],
             learn.observed_data[i,
                                 time_start_idx:time_end_idx + 1],
             color=c,
             linestyle='none',
             marker='.',
             markersize=10,
             alpha=0.25)

for i in chosen_obs:
    plt.plot(learn.filtered_data_coordinates,
             learn.filtered_obs[i,
                             :],
             'k',
             linestyle='none',
             marker='s',
             markersize=12)

plt.xlabel('$t$')
plt.ylabel('$y(t)$')
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
                learn.filtered_predictions[idx,
                                        :],
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

    xs = np.linspace(param_range[0, 0], param_range[0, 1], 100)
    ax2.plot(xs, GKDE(params[idx].flat[:])(xs))
    ax2.axvline(x=.65, ymin=0.0, ymax=1.0, color='r')
    ax2.set(xlabel=param_labels[0], title='Param. Distrib.')
    plt.show()

# Find best KPCA transformation for given number of QoI and transform time
# series data.
predict_map, obs_map = learn.learn_qois_and_transform(num_qoi=1)


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
    plt.text(n +
             1, eig_vals[n] /
             np.sum(eig_vals) *
             150, r'%2.3f' %
             (np.sum(eig_vals[0:n +
                              1]) /
              np.sum(eig_vals) *
              100) +
             '% of variation explained by first ' +
             '%1d' %
             (n +
                 1) +
             ' PCs.', {'color': 'k', 'fontsize': 20})
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


plot_gap(all_eig_vals=learn.kpcas, n=0, cluster=0)
plot_gap(all_eig_vals=learn.kpcas, n=0, cluster=1)

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
true_param_marginals = []
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
    plt.title(
        'Comparing pullback to actual density of parameter ' +
        param_labels[i],
        fontsize=16)
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

true_a = 2.0
true_b = 2.0


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

# Load precomputed time-series data at x=9.5.
predicted_time_series2 = np.loadtxt('burgers_files/unif_series2.txt')
observed_time_series2 = np.loadtxt('burgers_files/beta_series2_2_2.txt')
params_obs2 = np.loadtxt('burgers_files/beta_params_2_2.txt')
num_obs2 = observed_time_series.shape[0]
params_obs2 = params_obs2.reshape((num_obs2, 1))

# Add noise if desired
with_noise = True
noise_stdev = 0.025

if with_noise:
    predicted_time_series2 += noise_stdev * \
        np.random.randn(num_samples, times.shape[0])
    observed_time_series2 += noise_stdev * \
        np.random.randn(num_obs2, times.shape[0])

# Use LUQ to learn dynamics and QoIs
learn2 = LUQ(predicted_data=predicted_time_series2, 
             observed_data=observed_time_series2)

# time array indices over which to use
time_start_idx = 250
time_end_idx = 749

# Filter data with piecewise constant linear splines
learn2.filter_data(
    filter_method='splines_tol',
    data_coordinates=times,
    start_idx=time_start_idx,
    end_idx=time_end_idx,
    num_filtered_obs=500,
    tol=0.5 * noise_stdev,
    min_knots=3,
    max_knots=4)

# Learn and classify dynamics.
learn2.dynamics(
    cluster_method='kmeans',
    kwargs={
        'n_clusters': 2,
        'n_init': 10})

# # Plot clusters of predicted time series
num_filtered_obs2 = learn2.filtered_data_coordinates.shape[0]
for j in range(learn2.num_clusters):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(
        24, 8), gridspec_kw={'width_ratios': [1, 1]})
    ax1.scatter(
        np.tile(
            learn2.filtered_data_coordinates,
            num_samples).reshape(
            num_samples,
            num_filtered_obs2),
        learn2.filtered_predictions,
        50,
        c='gray',
        marker='.',
        alpha=0.2)
    idx = np.where(learn2.predict_labels == j)[0]
    ax1.scatter(np.tile(learn2.filtered_data_coordinates,
                        len(idx)).reshape(len(idx),
                                          num_filtered_obs2),
                learn2.filtered_predictions[idx,
                                         :],
                50,
                c='b',
                marker='o',
                alpha=0.2)
    idx2 = np.where(learn2.obs_labels == j)[0]
    ax1.scatter(np.tile(learn2.filtered_data_coordinates, len(idx2)).reshape(len(idx2), num_filtered_obs),
                learn2.filtered_obs[idx2, :], 50, c='r', marker='s', alpha=0.2)
    ax1.set(title='Cluster ' + str(j + 1) + ' in data')
    ax1.set_xlabel('$t$')
    ax1.set_ylabel('$x(t)$')

    xs = np.linspace(param_range[0, 0], param_range[0, 1], 100)
    ax2.plot(xs, GKDE(params[idx].flat[:])(xs))
    ax2.axvline(x=1.25, ymin=0.0, ymax=1.0, color='r')

    ax2.set(xlabel=param_labels[0], title='Param. Distrib.')
    plt.show()

# Find best KPCA transformation for given number of QoI and transform time
# series data.
predict_map2, obs_map2 = learn2.learn_qois_and_transform(num_qoi=1)

plot_gap(all_eig_vals=learn2.kpcas, n=0, cluster=0)
plot_gap(all_eig_vals=learn2.kpcas, n=0, cluster=1)

# Generate kernel density estimates on new QoI and calculate new weights
pi_predict_kdes2 = []
pi_obs_kdes2 = []
r_vals2 = []
r_means2 = []
for i in range(learn2.num_clusters):
    pi_predict_kdes2.append(GKDE(learn2.predict_maps[i].T))
    pi_obs_kdes2.append(GKDE(learn2.obs_maps[i].T))
    r_vals2.append(
        np.divide(
            pi_obs_kdes2[i](
                learn2.predict_maps[i].T), 
            pi_predict_kdes2[i](
                learn2.predict_maps[i].T)))
    r_means2.append(np.mean(r_vals2[i]))
print(f'Diagnostics: {r_means2}')

# Compute marginal probabilities for each parameter and initial condition.
param2_marginals = []
true_param_marginals = []
lam_ptr2 = []
cluster_weights2 = []
for i in range(learn2.num_clusters):
    lam_ptr2.append(np.where(learn2.predict_labels == i)[0])
    cluster_weights2.append(len(np.where(learn2.obs_labels == i)[0]) / num_obs)

for i in range(params.shape[1]):
    true_param_marginals.append(GKDE(params_obs2[:, i]))
    param2_marginals.append([])
    for j in range(learn2.num_clusters):
        param2_marginals[i].append(
            GKDE(params[lam_ptr2[j], i], weights=r_vals2[j]))

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
    for j in range(learn2.num_clusters):
        mar += param2_marginals[i][j](x) * cluster_weights2[j]
    plt.plot(x, mar, label='Updated', linewidth=4, linestyle='dashed')
    plt.plot(x, true_param_marginals[i](x), label='Data-generating',
             linewidth=4, linestyle='dotted')
    plt.title(
        'Comparing pullback to actual density of parameter ' +
        param_labels[i],
        fontsize=16)
    plt.legend(fontsize=20)
    plt.show()


def param2_update_KDE_error(x):
    mar = np.zeros(x.shape)
    for j in range(learn2.num_clusters):
        mar += param2_marginals[param_num][j](x) * cluster_weights2[j]
    return np.abs(mar - true_param_marginals[param_num](x))


for i in range(params.shape[1]):
    param_num = i
    TV_metric = quad(param2_update_KDE_error,
                     param_range[i, 0], param_range[i, 1], maxiter=1000)
    print(TV_metric)

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
                learn.filtered_predictions[idx,
                                        :],
                50,
                c='b',
                marker='o',
                alpha=0.2)
    idx2 = np.where(learn.obs_labels == j)[0]
    ax1.scatter(np.tile(learn.filtered_data_coordinates, len(idx2)).reshape(len(idx2), num_filtered_obs),
                learn.filtered_obs[idx2, :], 50, c='r', marker='s', alpha=0.2)
    ax1.set(title='Cluster ' + str(j + 1) + ' in data')
    ax1.set_xlabel('$t$')
    ax1.set_ylabel('$q(x_1, t)$')
    ax1.set_xlim(0.0, 7.5)

    ax2.scatter(
        np.tile(
            learn2.filtered_data_coordinates,
            num_samples).reshape(
            num_samples,
            num_filtered_obs),
        learn2.filtered_predictions,
        50,
        c='gray',
        marker='.',
        alpha=0.2)
    idx = np.where(learn2.predict_labels == j)[0]
    ax2.scatter(np.tile(learn2.filtered_data_coordinates,
                        len(idx)).reshape(len(idx),
                                          num_filtered_obs),
                learn2.filtered_predictions[idx,
                                         :],
                50,
                c='b',
                marker='o',
                alpha=0.2)
    idx2 = np.where(learn2.obs_labels == j)[0]
    ax2.scatter(np.tile(learn2.filtered_data_coordinates, len(idx2)).reshape(len(idx2), num_filtered_obs),
                learn2.filtered_obs[idx2, :], 50, c='r', marker='s', alpha=0.2)
    ax2.set(title='Cluster ' + str(j + 1) + ' in data')
    ax2.set_xlabel('$t$')
    ax2.set_ylabel('$q(x_2, t)$')
    ax2.set_xlim(0.0, 7.5)
    plt.show()


for j in range(learn.num_clusters):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(
        24, 8), gridspec_kw={'width_ratios': [1, 1]})
    idx = np.where(learn.predict_labels == j)[0]
    xs = np.linspace(param_range[0, 0], param_range[0, 1], 100)
    ax1.plot(xs, GKDE(params[idx].flat[:])(xs), linewidth=2)
    ax1.axvline(x=.65, ymin=0.0, ymax=1.0, color='r')
    ax1.set(xlabel=param_labels[0], title='Cluster ' + str(j + 1) + ", loc. 1")
    ax1.set_ylabel('Density')

    idx = np.where(learn2.predict_labels == j)[0]
    xs = np.linspace(param_range[0, 0], param_range[0, 1], 100)
    ax2.plot(xs, GKDE(params[idx].flat[:])(xs), linewidth=2)
    ax2.axvline(x=1.25, ymin=0.0, ymax=1.0, color='r')
    ax2.set(xlabel=param_labels[0], title='Cluster ' + str(j + 1) + ", loc. 2")
    ax2.set_ylabel('Density')
    plt.show()

for j in range(learn.num_clusters):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(
        24, 8), gridspec_kw={'width_ratios': [1, 1]})
    idx = np.where(learn.predict_labels == j)[0]
    vals = params[idx].flat[:]
    ax1.hist(vals, bins=20, range=(param_range[0, 0], param_range[0, 1]))
    ax1.axvline(x=.65, ymin=0.0, ymax=1.0, color='r')
    ax1.set(xlabel=param_labels[0], title='Cluster ' + str(j + 1) + ", loc. 1")

    idx2 = np.where(learn2.predict_labels == j)[0]
    vals2 = params[idx2].flat[:]
    ax2.set(xlabel=param_labels[0], title='Cluster ' + str(j + 1) + ", loc. 2")
    ax2.hist(vals2, bins=20, range=(param_range[0, 0], param_range[0, 1]))
    ax2.axvline(x=1.25, ymin=0.0, ymax=1.0, color='r')
    print(min(vals2), max(vals2))
    plt.show()

z = max(vals2)
z = (z - 0.75) / (3.0 - 0.75)
p2 = beta.cdf(z, 2.0, 2.0)
p1 = 1.0 - p2
print(p1, p2)
print(cluster_weights2)
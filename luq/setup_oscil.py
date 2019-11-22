# Copyright 2019 Steven Mattis and Troy Butler
#
# ## An ODE model
#
# Model is for harmonic motion
# $$y''(t) + 2cy'(t) + \omega_0^2 x = f(t)$$
# with damping constant
# $$c \in [0.1,1]$$
# and natural frequency
# $$\omega_0\in[0.5,2]$$
# and forcing term initially taken to be zero.
#
# Note that with the ranges of $c$ and $\omega_0$ above, it is possible for the system to either be under-, over-, or critically damped (and since $c\geq 0.1$ it is never undamped, which is almost always physical nonsense).
#
# The roots to the characteristic equation are given by
# $$ r_1 = -c\pm \sqrt{c^2-\omega_0^2}$$.
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
# An ***initial*** uniform distribution is assumed and updated by the true time series data.

# ### Time series data appended with differences and sums
#
# We initially assume no errors in the time series data, i.e., the observations are $y(t)$ at some finite set of times
# $\{t_i\}_{i=1}^N$, with $0\leq t_1 < t_2 < \cdots < t_N$.
#
# We also take differences and summations of the time series (to extract derivative and integral type information) and
# append to the data to determine if new/dominant features (i.e., principal components) are found.

import numpy as np

def y(t, c, omega_0, a, b):
    """
    Analytical solution to the oscillator problem.
    :param t:
    :param c:
    :param omega_0:
    :param a:
    :param b:
    :return:
    """
    z = np.zeros(len(c))
    ind_under = np.where(np.greater(omega_0, c))[0]
    ind_over = np.where(np.greater(c, omega_0))[0]
    # First solve for the under-damped case
    if ind_under.size > 0:
        omega = np.sqrt(omega_0[ind_under] ** 2 - c[ind_under] ** 2)
        C_1 = a[ind_under]
        C_2 = (b[ind_under] + c[ind_under] * a[ind_under]) / omega

        z[ind_under] = np.exp(-c[ind_under] * t) * (C_1 * np.cos(omega * t)
                                                    + C_2 * np.sin(omega * t))

    if ind_over.size > 0:
        r_1 = -c[ind_over] - np.sqrt(c[ind_over] ** 2 - omega_0[ind_over] ** 2)
        r_2 = -c[ind_over] + np.sqrt(c[ind_over] ** 2 - omega_0[ind_over] ** 2)
        C_1 = (b[ind_over] - a[ind_over] * r_2) / (r_1 - r_2)
        C_2 = (b[ind_over] - r_1 * a[ind_over]) / (r_2 - r_1)

        z[ind_over] = C_1 * np.exp(r_1 * t) + C_2 * np.exp(r_2 * t)

    return z


# Uniformly sample the parameter samples to form a "prediction" or "test" set
num_samples = int(1E3)

lam = np.random.uniform(size=(num_samples,4))

lam[:,0] = 0.1 + 0.4*lam[:,0]  #c
lam[:,1] = 0.5 + 1.5*lam[:,1] #omega_0
lam[:,2] = 1 + 1*lam[:,2] #a
lam[:,3] = -1 + 1*lam[:,3]   #b

# Construct the predicted time series data

num_time_preds = int(50)  # number of predictions (uniformly space) between [time_start,time_end]
time_start = 0.5
time_end = 3.5
times = np.linspace(time_start, time_end, num_time_preds)

predicted_time_series = np.zeros((num_samples, num_time_preds))

for i in range(num_time_preds):
    predicted_time_series[:, i] = y(times[i], lam[:, 0], lam[:, 1], lam[:, 2], lam[:, 3])

# #### Generate an observed set of data from a different distribution on parameters
#
# The idea here is to show that we can reconstruct/recover a "true" distribution on parameters from observations.
# This establishes that we are indeed ***inverting*** a distribution on outputs.
#
# However, in practice, we may only observe a ***single*** time series of data, polluted by noise, and impose a
# distribution on this time series data to invert.
#
# Below, we simulate a peaked Beta distribution on a subset of the whole parameter space.


# Simulate an observed distribution of time series data

num_obs = int(1E3)

true_a = 2
true_b = 2

lam_obs = np.random.beta(size=(num_obs, 4), a=true_a, b=true_b)

lam_obs[:, 0] = 0.1 + 0.4*lam_obs[:, 0]  # c
lam_obs[:, 1] = 0.5 + 1.5*lam_obs[:, 1]  # omega_0
lam_obs[:, 2] = 1 + 1*lam_obs[:, 2]   # a
lam_obs[:, 3] = -1 + 1*lam_obs[:, 3]    #b

observed_time_series = np.zeros((num_obs, num_time_preds))

with_noise = False
noise_stdev = 0.05

if with_noise:
    for i in range(num_time_preds):
        observed_time_series[:, i] = y(times[i], lam_obs[:, 0], lam_obs[:, 1], lam_obs[:, 2], lam_obs[:, 3 ]) + \
                                     noise_stdev * np.random.randn(num_obs)
else:
    for i in range(num_time_preds):
        observed_time_series[:, i] = y(times[i], lam_obs[:, 0], lam_obs[:, 1], lam_obs[:, 2], lam_obs[:, 3])

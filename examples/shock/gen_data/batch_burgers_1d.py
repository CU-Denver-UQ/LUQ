#!/usr/bin/env python
# encoding: utf-8

r"""
Burgers' equation
=========================

Solve the inviscid Burgers' equation:

.. math::
    q_t + \frac{1}{2} (q^2)_x = 0.

This is a nonlinear PDE often used as a very simple
model for fluid dynamics.

The initial condition is sinusoidal, but after a short time a shock forms
(due to the nonlinearity).
"""
from __future__ import absolute_import
import numpy as np
import scipy.io as sio
from clawpack import riemann
from clawpack import pyclaw
import matplotlib.pyplot as plt

# Define problem and solver
riemann_solver = riemann.burgers_1D_py.burgers_1D
solver = pyclaw.ClawSolver1D(riemann_solver)
solver.limiters = pyclaw.limiters.tvd.vanleer
solver.kernel_language = 'Python'
solver.bc_lower[0] = pyclaw.BC.extrap
solver.bc_upper[0] = pyclaw.BC.extrap

fl = 1.5
fr = 1.0

num_out = 500
time_final = 10.0


def burgers(params):
    times = np.linspace(0, time_final, num_out + 1)
    outputs = np.zeros((params.shape[0], num_out + 1))
    for k in range(params.shape[0]):
        # Define domain and mesh
        x = pyclaw.Dimension(0.0, 10.0, 500, name='x')
        domain = pyclaw.Domain(x)
        num_eqn = 1
        state = pyclaw.State(domain, num_eqn)
        xc = state.grid.x.centers
        state.problem_data['efix'] = True

        a = params[k, 0]
        for i in range(state.q.shape[1]):
            if xc[i] <= 0.25:
                state.q[0, i] = fl
            elif xc[i] > (0.25 + 2.0 * a):
                state.q[0, i] = fr
            else:
                state.q[0, i] = 0.5 * \
                    ((fl + fr) - (fl - fr) * (xc[i] - 0.25 - a) / a)

        # Set gauge
        grid = state.grid
        grid.add_gauges([[7.0]])
        state.keep_gauges = True

        # Setup and run
        claw = pyclaw.Controller()
        claw.tfinal = time_final
        claw.num_output_times = num_out
        claw.solution = pyclaw.Solution(state, domain)
        claw.solver = solver
        claw.outdir = './_output'
        claw.run()

        # Process output
        A = np.loadtxt('./_output/_gauges/gauge7.0.txt')
        idx = 0
        vals = []
        for j in range(A.shape[0]):
            if A[j, 0] == times[idx]:
                vals.append(A[j, 1])
                idx += 1
        outputs[k, :] = vals
    return (times, outputs)


#params = 3.0 * np.random.rand(500, 1)

#(times, time_series) = burgers(params)

#np.savetxt('burgers_files/unif_params.txt', params)
#np.savetxt('burgers_files/unif_times.txt', times)
#np.savetxt('burgers_files/unif_series.txt', time_series)

true_a = 2
true_b = 5

params_obs = 3.0 * np.random.beta(size=(500, 1), a=true_a, b=true_b)

(times_obs, time_series_obs) = burgers(params_obs)

np.savetxt('burgers_files/beta_params_2_5.txt', params_obs)
np.savetxt('burgers_files/beta_times_2_5.txt', times_obs)
np.savetxt('burgers_files/beta_series_2_5.txt', time_series_obs)

true_a = 5
true_b = 2

params_obs = 3.0 * np.random.beta(size=(500, 1), a=true_a, b=true_b)

(times_obs, time_series_obs) = burgers(params_obs)

np.savetxt('burgers_files/beta_params_5_2.txt', params_obs)
np.savetxt('burgers_files/beta_times_5_2.txt', times_obs)
np.savetxt('burgers_files/beta_series_5_2.txt', time_series_obs)

#fig = plt.figure()
# for i in range(params.shape[0]):
#    plt.plot(times, time_series[i, :])
# plt.show()
#import pdb
# pdb.set_trace()

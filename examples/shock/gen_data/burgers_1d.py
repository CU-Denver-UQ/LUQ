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
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from clawpack import riemann

from clawpack import pyclaw

riemann_solver = riemann.burgers_1D_py.burgers_1D

solver = pyclaw.ClawSolver1D(riemann_solver)
solver.limiters = pyclaw.limiters.tvd.vanleer

solver.kernel_language = 'Python'

fl = 1.5
fr = 1.0
solver.bc_lower[0] = pyclaw.BC.extrap  # pyclaw.BC.periodic
solver.bc_upper[0] = pyclaw.BC.extrap  # pyclaw.BC.periodic

x = pyclaw.Dimension(0.0, 10.0, 500, name='x')
domain = pyclaw.Domain(x)
num_eqn = 1
state = pyclaw.State(domain, num_eqn)

xc = state.grid.x.centers
#state.q[0, :] = 0.01 * np.cos(np.pi * 2 * xc) + 0.50
#beta = 10; gamma = 0; x0 = 0.5
#state.q[0,:] = 0.01 * np.exp(-beta * (xc-x0)**2) * np.cos(gamma * (xc - x0))
a = 3.0
for i in range(state.q.shape[1]):
    if xc[i] <= 0.25:
        state.q[0, i] = fl
    elif xc[i] > (0.25 + 2.0 * a):
        state.q[0, i] = fr
    else:
        state.q[0, i] = 0.5 * ((fl + fr) - (fl - fr) * (xc[i] - 0.25 - a) / a)

state.problem_data['efix'] = True

##
grid = state.grid
grid.add_gauges([[7.0]])
state.keep_gauges = True
##

claw = pyclaw.Controller()
claw.tfinal = 10.0
claw.num_output_times = 100
claw.solution = pyclaw.Solution(state, domain)
claw.solver = solver
claw.outdir = './_output'
#claw.setplot = setplot
#claw.keep_copy = True
claw.run()

A = np.loadtxt('./_output/_gauges/gauge7.0.txt')
times = np.linspace(0, claw.tfinal, claw.num_output_times + 1)

idx = 0
output = []
for i in range(A.shape[0]):
    if A[i, 0] == times[idx]:
        output.append(A[i, 1])
        idx += 1
output = np.array(output)
#plt.plot(A[:, 0], A[:, 1])
plt.plot(times, output)
plt.show()

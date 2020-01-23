# Copyright 2019 Steven Mattis and Troy Butler
import numpy as np


class IVPBase:
    """
    Base class for initial value problems.
    """
    def __init__(self):
        self.num_params = None
        self.num_equations = None
        pass

    def solve(self,
              ics,
              params,
              t_eval,
              kwargs={},
              idx=0):
        pass

    def check_dims(self, ics, params):
        assert(ics.shape[1] == self.num_equations)
        assert(params.shape[1] == self.num_params)


class HarmonicOscillator(IVPBase):
    """
    An ODE model for a harmonic oscillator

    Model is for harmonic motion
    $$y''(t) + 2cy'(t) + \omega_0^2 y = f(t)$$
    with damping constant
    $$c \in [0.1,1]$$
    and natural frequency
    $$\omega_0\in[0.5,2]$$
    and forcing term initially taken to be zero.

    Note that with the ranges of $c$ and $\omega_0$ above, it is possible for the system to either be under-, over-, or critically damped (and since $c\geq 0.1$ it is never undamped, which is almost always physical nonsense).

    The roots to the characteristic equation are given by
    $$ r_1 = -c\pm \sqrt{c^2-\omega_0^2}$$.

    When the system is under-damped, the solution is given by
    $$ y(t) = e^{-ct}[C_1\cos(\omega t) + C_2\sin(\omega t)], \ \omega=\sqrt{\omega_0^2-c^2}. $$


    When the system is over-damped, the solution is given by
    $$ y(t) = C_1 e^{r_1t}+C_2 e^{r_2t}. $$

    And, finally, when the system is critically damped, the solution is given by
    $$ y(t) = C_1e^{-ct} + C_2 te^{-ct}. $$

    However, we never expect the system to be critically damped in practice since this is "too fine-tuned"
    of a scenario.
    The constants $C_1$ and $C_2$ are determined by the initial conditions, which we assume to be given by
    $$ y(0)=a, y'(0) = b $$
    where
    $$ a\in[1,2] $$
    and
    $$ b\in[-1,0] $$.

    In the under-damped case,
    $$ C_1 = a, \ \text{and } \ C_2 = \frac{b+ca}{\omega}. $$

    In the over-damped case,
    $$ C_1 = \frac{b-ar_2}{r_1-r_2}, \ \text{and } \ C_2 = \frac{b-r_1a}{r_2-r_1} $$

    """

    def __init__(self):
        super().__init__()
        self.num_equations = 2
        self.num_params = 2

    def solve(self,
              ics,
              params,
              t_eval,
              kwargs={},
              idx=0):

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

        self.check_dims(ics, params)
        n = ics.shape[0]
        time_series = np.zeros((n, len(t_eval)))

        for i, t in enumerate(t_eval):
            time_series[:, i] = y(t, params[:, 0], params[:, 1], ics[:, 0], ics[:, 1])
        return time_series


class ODE(IVPBase):
    """
    Base class for Ordinary Differential Equations to solved numerically.
    """
    def __init__(self):
        super().__init__()
        self.f = None
        self.jacobian = False

    def define_f(self, param):
        def f(t, y):
            pass
        return f

    def define_jacobian(self, param):
        def jacobian(t, y):
            pass
        return jacobian

    def solve(self,
              ics,
              params,
              t_eval,
              kwargs={},
              idx=0):
        from scipy.integrate import solve_ivp

        self.check_dims(ics, params)

        t_span = [0.0, t_eval[-1]]
        n = ics.shape[0]
        time_series = np.zeros((n, len(t_eval)))
        for i in range(n):
            f = self.define_f(params[i, :])
            if self.jacobian:
                sol = solve_ivp(fun=f,
                                t_span=t_span,
                                y0=ics[i, :],
                                t_eval=t_eval,
                                jac=self.jacobian,
                                **kwargs)
            else:
                sol = solve_ivp(fun=f,
                                t_span=t_span,
                                y0=ics[i, :],
                                t_eval=t_eval,
                                **kwargs)
            if sol.success != True:
                import pdb
                pdb.set_trace()
            time_series[i, :] = sol.y[idx, :]
        return time_series


class Lienard(ODE):
    """
    The Lienard ODE system:
    $$u' = v$$
    $$v' = -u + (\mu - u^2) v,$$
    which has a Hopf bifurcation at $\mu = 0$.
    """
    def __init__(self):
        super().__init__()
        self.num_equations = 2
        self.num_params = 1

    def define_f(self, param):
        def f(t, y):
            return [y[1], -y[0] + (param[0] - y[0]*y[0]) * y[1]]
        return f

    def define_jacobian(self, param):
        self.jacobian = True

        def jacobian(t, y):
            return [[0.0, 1.0], [-1.0-2.0*y[0]*y[1], param[0] - y[0]*y[0]]]
        return jacobian


class Selkov(ODE):
    """
    Sel'kov ODE system for glycolysis. Has potential Hopf bifurcations. See:
    https://www.math.colostate.edu/~shipman/47/volume3b2011/M640_MunozAlicea.pdf
    """
    def __init__(self):
        super().__init__()
        self.num_equations = 2
        self.num_params = 2

    def define_f(self, param):
        def f(t, y):
            a = param[0]
            b = param[1]
            c = b / (a + b**2)
            f1 = -(y[0] + b) + a*(y[1] + c) + (y[0] + b)**2*(y[1] + c)
            f2 = b - a*(y[1] + c) - (y[0] + b)**2*(y[1] + c)
            return [f1, f2]
        return f

    def define_jacobian(self, param):
        self.jacobian = True

        def jacobian(t, y):
            a = param[0]
            b = param[1]
            c = b / (a + b ** 2)
            j11 = -1.0 + 2 * (y[0] + b) * (y[1] + c)
            j12 = a + (y[0] + b)**2
            j21 = -2.0 * (y[0] + b) * (y[1] + c)
            j22 = -a - (y[0] + b)**2
            return [[j11, j12], [j21, j22]]
        return jacobian

class Lorenz(ODE):
    """
    The Lorenz system. See https://en.wikipedia.org/wiki/Lorenz_system.
    """
    def __init__(self):
        super().__init__()
        self.num_equations = 3
        self.num_params = 3

    def define_f(self, param):
        def f(t, y):
            sigma = param[0]
            rho = param[1]
            beta = param[2]

            f1 = sigma * (y[1] - y[0])
            f2 = y[0] * (rho - y[2]) - y[1]
            f3 = y[0] * y[1] - beta * y[2]
            return [f1, f2, f3]

    def define_jacobian(self, param):
        self.jacobian = True

        def jacobian(t, y):
            sigma = param[0]
            rho = param[1]
            beta = param[2]

            j1 = [-sigma, rho - y[2], 1.0]
            j2 = [sigma, -1.0, y[0]]
            j3 = [0.0, -y[0], -beta]
            return [j1, j2, j3]
        return jacobian

# N = 10
# num_samples = int(N)
#
# params = np.random.uniform(size=(num_samples, 2))
# params[:, 0] = 0.1 * params[:, 0] + 0.05
# params[:, 1] = 0.95 * params[:, 1] + 0.05
# #params[:, 0] = 0.1
# ics = 2.0 * np.random.uniform(size=(num_samples, 2))
# times = np.linspace(0.1, 40.0, 1000)
# prob = Selkov()
# A = prob.solve(ics=ics, params=params, t_eval=times)
# import matplotlib.pyplot as plt
# plt.figure()
# for i in range(N):
#     plt.plot(times, A[i,:])
# plt.show()
# import pdb
# pdb.set_trace()










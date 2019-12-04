import numpy as np


class IVPBase:
    def __init__(self):
        pass

    def solve(self,
              ics,
              params,
              t_eval,
              kwargs={},
              idx=0):
        pass


class HarmonicOscillator(IVPBase):
    def solve(self,
              ics,
              params,
              t_eval,
              kwargs={},
              idx=0):
        n = ics.shape[0]
        time_series = np.zeros((n, len(t_eval)))

        for i, t in enumerate(t_eval):
            time_series[:, i] = y(t, params[:, 0], params[:, 1], ics[:, 0], ics[:, 1])
        return time_series


class ODE(IVPBase):
    def __init__(self):
        super().__init__()
        self.f = None
        self.jacobian = None

    def define_f(self, param):
        def f(t, y):
            pass
        return f

    def define_jacobian(self):
        pass

    def solve(self,
              ics,
              params,
              t_eval,
              kwargs={},
              idx=0):
        from scipy.integrate import solve_ivp

        t_span = [0.0, t_eval[-1]]
        n = ics.shape[0]
        time_series = np.zeros((n, len(t_eval)))
        for i in range(n):
            f = self.define_f(params[i, :])
            if self.jacobian is not None:
                Sol = solve_ivp(fun=f,
                                t_span=t_span,
                                y0=ics[i, :],
                                t_eval=t_eval,
                                jac=self.jacobian,
                                **kwargs)
            else:
                Sol = solve_ivp(fun=f,
                                t_span=t_span,
                                y0=ics[i, :],
                                t_eval=t_eval,
                                **kwargs)
            time_series[i, :] = Sol.y[idx, :]
        return time_series


class Lienard(ODE):
    def define_f(self, param):
        def f(t, y):
            return [y[1], -y[0] + (param[0] - y[0]*y[0]) * y[1]]
        return f



N = 10
num_samples = int(N)

params = 2.0 * np.random.uniform(size=(num_samples, 1)) - 1.0
ics = 2.0 * np.random.uniform(size=(num_samples, 2)) - 1.0
times = np.linspace(0.1, 40.0, 1000)
prob = Lienard()
A = prob.solve(ics=ics, params=params, t_eval=times)
import matplotlib.pyplot as plt
plt.figure()
for i in range(N):
    plt.plot(times, A[i,:])
plt.show()
import pdb
pdb.set_trace()










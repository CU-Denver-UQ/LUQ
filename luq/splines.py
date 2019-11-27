# Copyright 2019 Steven Mattis and Troy Butler
from scipy import optimize
import numpy as np

# Splines and other methods for cleaning data.


def linear_C0_spline(times, data, num_knots, clean_times):
    """
    Clean a time series over window with C0 linear splines.
    :param times: time window
    :param data: time series data
    :param num_knots: number of knots to use
    :param clean_times: number of clean values wanted
    :return:
    """

    def wrapper_fit_func(x, N, *args):
        Qs = list(args[0][0:N])
        knots = list(args[0][N:])
        return piecewise_linear(x, knots, Qs)

    def piecewise_linear(x, knots, Qs):
        knots = np.insert(knots, 0, times[0])
        knots = np.append(knots, times[-1])
        return np.interp(x, knots, Qs)

    knots_init = np.linspace(times[0], times[-1], num_knots)[1:-1]

    # find piecewise linear splines for predictions
    q_pl, _ = optimize.curve_fit(lambda x, *params_0: wrapper_fit_func(x, num_knots, params_0),
                                 times,
                                 data,
                                 p0=np.hstack([np.zeros(num_knots), knots_init]))

    # calculate clean data
    clean_data = piecewise_linear(clean_times,
                                  q_pl[num_knots:2 * num_knots],
                                  q_pl[0:num_knots])

    # calculate mean absolute error between spline and original data
    clean_data_at_original = piecewise_linear(times,
                                              q_pl[num_knots:2 * num_knots],
                                              q_pl[0:num_knots])
    error = np.average(np.abs(clean_data_at_original - data))
    error = error / np.average(np.abs(data))

    return clean_data, error

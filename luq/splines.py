# Copyright 2019-2020 Steven Mattis and Troy Butler
from scipy import optimize
import numpy as np

# Methods for finding optimal splines for cleaning data.


def linear_c0_spline(
        times,
        data,
        num_knots,
        clean_times,
        spline_old=None,
        verbose=False):
    """
    Clean a time series over window with C0 linear splines.
    :param times: time series over window
    :type times: :class:`numpy.ndarray`
    :param data: time series data
    :type data: :class:`numpy.ndarray`
    :param num_knots: number of knots to use
    :type num_knots: int
    :param clean_times: number of clean values wanted
    :type clean_times: :class:`numpy.ndarray`
    :param spline_old: array containing knots and values from previous spline iteration
    :type  spline_old: :class:`numpy.ndarray`
    :param verbose: Make true to print summaries of curvefitting routines.
    :type verbose: bool
    :return: clean data array, l1-type error between spline and data, and spline array
    :rtype: :class:`numpy.ndarray`, float,  :class:`numpy.ndarray`
    """

    def wrapper_fit_func(x, N, *args):
        Qs = list(args[0][0:N])
        knots = list(args[0][N:])
        return piecewise_linear(x, knots, Qs)

    def wrapper_fit_func_qs(x, knots, *args):
        Qs = list(args[0])
        return piecewise_linear(x, knots, Qs)

    def piecewise_linear(x, knots, Qs):
        knots = np.insert(knots, 0, 0)
        knots = np.append(knots, 1)
        return np.interp(x, knots, Qs)

    def piecewise_linear_clean(x, knots, Qs):
        knots = np.insert(knots, 0, times[0])
        knots = np.append(knots, times[-1])
        return np.interp(x, knots, Qs)

    if spline_old is None:
        vals_init = np.zeros((num_knots,))
    else:
        vals_init = piecewise_linear_clean(np.linspace(0, 1, num_knots),
                                           spline_old[num_knots - 1:],
                                           spline_old[0:num_knots - 1])

    param_bounds = np.zeros((2, 2 * num_knots - 2))
    for i in range(2 * num_knots - 2):
        if i < num_knots:
            param_bounds[:, i] = [-np.inf, np.inf]
        else:
            param_bounds[:, i] = [0, 1]

    knots_unif = np.linspace(0, 1, num_knots)[1:-1]
    q_pl_unif, _ = optimize.curve_fit(lambda x, *params_0: wrapper_fit_func_qs(
        x, knots_unif, params_0), (times - times[0]) / (times[-1] - times[0]), data, p0=vals_init)
    q_pl_unif = np.hstack([q_pl_unif, knots_unif])
    q_pl_unif[num_knots:] *= (times[-1] - times[0])
    q_pl_unif[num_knots:] += times[0]

    clean_data_at_original_unif = piecewise_linear_clean(
        times, q_pl_unif[num_knots:], q_pl_unif[0:num_knots])
    ssr_unif = np.sum((data - clean_data_at_original_unif)**2)

    # find piecewise linear splines for predictions
    opt_fail = False
    try:
        knots_init = np.linspace(0, 1, num_knots)[1:-1]
        q_pl, _ = optimize.curve_fit(lambda x, *params_0: wrapper_fit_func(x, num_knots, params_0),
                                     (times - times[0]) / (times[-1] - times[0]),
                                     data,
                                     p0=np.hstack([q_pl_unif[0:num_knots], knots_init]),
                                     bounds=param_bounds,
                                     verbose=verbose)
        q_pl[num_knots:] *= (times[-1] - times[0])
        q_pl[num_knots:] += times[0]

        # fix if knots get out of order
        inds_sort = np.argsort(q_pl[num_knots:])
        q_pl[1:num_knots - 1] = q_pl[inds_sort + 1]  # Qs
        q_pl[num_knots:] = q_pl[inds_sort + num_knots]  # knots

        clean_data_at_original = piecewise_linear_clean(times,
                                                        q_pl[num_knots:],
                                                        q_pl[0:num_knots])
        ssr = np.sum((data - clean_data_at_original) ** 2)

    except RuntimeError:
        # Use uniform knots if optimization fails
        opt_fail = True

    if opt_fail or ssr_unif < ssr:
        print('Optimization of knot locations failed. Using uniform knots.')
        q_pl = q_pl_unif
        clean_data_at_original = clean_data_at_original_unif

    # calculate clean data
    clean_data = piecewise_linear_clean(clean_times,
                                        q_pl[num_knots:],
                                        q_pl[0:num_knots])

    # calculate mean absolute error between spline and original data
    error = np.average(np.abs(clean_data_at_original - data))
    error = error / np.average(np.abs(data))
    return clean_data, error, q_pl

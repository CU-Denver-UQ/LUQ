import unittest
import os
import numpy.testing as nptest
import numpy as np
from luq.luq import LUQ
from luq.dynamical_systems import Selkov
from luq.dynamical_systems import HarmonicOscillator


class Test_Selkov(unittest.TestCase):
    """
    Testing LUQ with ODE Model
    """

    def setUp(self):
        np.random.seed(123456)

        num_samples = int(20)

        param_range = np.array([[0.01, 0.124],  # a
                                [0.05, 1.5]])  # b
        ic_range = np.array([[1.0, 1.0],  # y_0
                             [1.0, 1.0]])  # x_0

        params = np.random.uniform(size=(num_samples, 2))
        params = param_range[:, 0] + \
            (param_range[:, 1] - param_range[:, 0]) * params

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
        predicted_time_series = phys.solve(
            ics=ics, params=params, t_eval=times)

        # Simulate an observed Beta distribution of time series data

        num_obs = int(20)

        true_a = 2
        true_b = 2

        params_obs = np.random.beta(size=(num_obs, 2), a=true_a, b=true_b)
        params_obs = param_range[:, 0] + \
            (param_range[:, 1] - param_range[:, 0]) * params_obs

        ics_obs = np.random.beta(size=(num_obs, 2), a=true_a, b=true_b)
        ics_obs = ic_range[:, 0] + (ic_range[:, 1] - ic_range[:, 0]) * ics_obs

        # Solve system
        observed_time_series = phys.solve(
            ics=ics_obs, params=params_obs, t_eval=times)

        # Add noise if desired
        with_noise = True
        noise_stdev = 0.0125

        if with_noise:
            observed_time_series += noise_stdev * \
                np.random.randn(num_obs, times.shape[0])

        # Use LUQ to learn dynamics and QoIs
        self.learn = LUQ(predicted_time_series, observed_time_series, times)
        time_start_idx = 0
        time_end_idx = len(times) - 1
        self.learn.clean_data(
            time_start_idx=time_start_idx,
            time_end_idx=time_end_idx,
            num_clean_obs=20,
            tol=5.0e-2,
            min_knots=3,
            max_knots=12)

    def test_kmeans(self):
        """
        Test using k-means clustering.
        """
        self.learn.dynamics(
            cluster_method='kmeans', kwargs={
                'n_clusters': 3, 'n_init': 10})
        self.learn.learn_qois_and_transform(num_qoi=2)

    def test_spectral(self):
        """
        Test using spectral clustering
        """
        self.learn.dynamics(
            cluster_method='spectral', kwargs={
                'n_clusters': 3, 'n_init': 10})
        self.learn.learn_qois_and_transform(variance_rate=0.9)
        self.learn.generate_kdes()
        self.learn.compute_r()

    def test_gmm(self):
        """
        Test using Gaussian Mixture Model
        """
        self.learn.dynamics(
            cluster_method='gmm', kwargs={
                'n_components': 3, 'n_init': 10})
        self.learn.learn_qois_and_transform(variance_rate=0.9)


class Test_Harmonic(Test_Selkov):
    """
    Testing LUQ with harmonic oscillator
    """

    def setUp(self):
        np.random.seed(123456)
        # Uniformly sample the parameter samples to form a "prediction" or
        # "test" set
        num_samples = int(1E2)

        params = np.random.uniform(size=(num_samples, 2))
        ics = np.random.uniform(size=(num_samples, 2))

        param_range = np.array([[0.1, 1.0],  # c
                                [0.5, 1.0]])  # omega_0
        ic_range = np.array([[3.0, 3.0],  # a
                             [0.0, 0.0]])  # b
        params = param_range[:, 0] + \
            (param_range[:, 1] - param_range[:, 0]) * params
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
        predicted_time_series = phys.solve(
            ics=ics, params=params, t_eval=times)

        # Simulate an observed Beta distribution of time series data

        num_obs = int(1E2)

        true_a = 2
        true_b = 2

        params_obs = np.random.beta(size=(num_obs, 2), a=true_a, b=true_b)
        ics_obs = np.random.beta(size=(num_obs, 2), a=true_a, b=true_b)
        params_obs = param_range[:, 0] + \
            (param_range[:, 1] - param_range[:, 0]) * params_obs
        ics_obs = ic_range[:, 0] + (ic_range[:, 1] - ic_range[:, 0]) * ics_obs

        observed_time_series = phys.solve(
            ics=ics_obs, params=params_obs, t_eval=times)

        # Add noise if desired
        with_noise = True
        noise_stdev = 0.25

        if with_noise:
            observed_time_series += noise_stdev * \
                np.random.randn(num_obs, num_time_preds)

        # Use LUQ to learn dynamics and QoIs
        self.learn = LUQ(predicted_time_series, observed_time_series, times)

        # time array indices over which to use
        time_start_idx = 0
        time_end_idx = num_time_preds - 1

        num_clean_obs = 16

        # Clean data with piecewise linear splines
        self.learn.clean_data(
            time_start_idx=time_start_idx,
            time_end_idx=time_end_idx,
            num_clean_obs=num_clean_obs,
            tol=5.0e-2,
            min_knots=3,
            max_knots=10)

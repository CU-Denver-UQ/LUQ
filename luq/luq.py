# Copyright 2019-2020 Steven A. Mattis and Troy Butler

import numpy as np
from luq.splines import *


class LUQ(object):
    """
    Learning Uncertain Quantities:
    Takes in observed and predicted time series data, cleans the data, identifies dynamics and
    low-dimensional structures, and transforms the data.
    """

    def __init__(self,
                 predicted_time_series,
                 observed_time_series,
                 times):
        """
        Initializes objects. All time series arrays should be the same length.
        :param predicted_time_series: time series from predictions
        :type predicted_time_series: :class:`numpy.ndarray`
        :param observed_time_series: time series from observations
        :type observed_time_series: :class:`numpy.ndarray`
        :param times: points in time for time series
        :type times: :class:`numpy.ndarray`
        """

        self.predicted_time_series = predicted_time_series
        self.observed_time_series = observed_time_series
        self.times = times
        self.clean_times = None
        self.clean_predictions = None
        self.clean_obs = None
        self.num_clusters = None
        self.cluster_labels = None
        self.predict_labels = None
        self.obs_labels = None
        self.obs_empty_cluster = []
        self.kpcas = None
        self.q_predict_kpcas = None
        self.predict_maps = []
        self.obs_maps = []
        self.num_pcs = []
        self.variance_rate = []
        self.Xpcas = []
        self.pi_predict_kdes = []
        self.pi_obs_kdes = []
        self.scalers = []
        self.predict_knots = []
        self.obs_knots = []
        self.r = None

        # incorporate more into this
        self.info = {'clustering_method': None,
                     'num_clusters': None,
                     'classifier_type': None,
                     'classifier_kernel': None,
                     'misclassification_rate': None,
                     'kpca_kernel': None,
                     'num_principal_components': None}

    def clean_data(
            self,
            time_start_idx,
            time_end_idx,
            num_clean_obs,
            tol,
            min_knots=3,
            max_knots=100,
            verbose=False):
        """
        Clean observed and predicted time series data so that difference between iterations is within tolerance.
        :param time_start_idx: first time index to clean
        :type time_start_idx: int
        :param time_end_idx: last time index to clean
        :type time_end_idx: int
        :param num_clean_obs: number of clean observations to make
        :type num_clean_obs: int
        :param tol: tolerance for constructing splines
        :type tol: float
        :param min_knots: maximum number of knots allowed
        :type min_knots: int
        :param max_knots: minimum number of knots allowed
        :type max_knots: int
        :param verbose: display termination reports
        :type verbose: bool
        :return: arrays of clean predictions, clean observations, and clean times
        :rtype: :class:`numpy.ndarray`, :class:`numpy.ndarray`, :class:`numpy.ndarray`
        """

        # Use _old and _new to compare to tol and determine when to stop adding knots
        # Compute _old before looping and then i=i+1
        times = self.times[time_start_idx:time_end_idx + 1]
        clean_times = np.linspace(
            self.times[time_start_idx],
            self.times[time_end_idx],
            num_clean_obs)
        num_predictions = self.predicted_time_series.shape[0]
        num_obs = self.observed_time_series.shape[0]
        self.clean_predictions = np.zeros((num_predictions, num_clean_obs))
        self.clean_obs = np.zeros((num_obs, num_clean_obs))

        for idx in range(num_predictions):
            clean_predictions_old, error_old, q_pl = linear_c0_spline(times=times,
                                                                      data=self.predicted_time_series[idx,
                                                                                                      time_start_idx:time_end_idx + 1],
                                                                      num_knots=min_knots,
                                                                      clean_times=clean_times,
                                                                      verbose=verbose)
            i = min_knots + 1
            while i <= max_knots:
                clean_predictions_new, error_new, q_pl = linear_c0_spline(times=times,
                                                                          data=self.predicted_time_series[idx, time_start_idx:time_end_idx + 1],
                                                                          num_knots=i,
                                                                          clean_times=clean_times,
                                                                          spline_old=q_pl,
                                                                          verbose=verbose)

                # After an _old and a _new is computed (when i>min_knots)
                print(idx, i, error_new)
                diff = np.average(np.abs(clean_predictions_new - clean_predictions_old)) / \
                    np.average(np.abs(self.predicted_time_series[idx,
                                                                 time_start_idx:time_end_idx + 1]))
                if diff < tol:
                    break
                else:
                    i += 1
                    if i <= max_knots:
                        clean_predictions_old = clean_predictions_new
            if i > max_knots:
                print("Warning: maximum number of knots reached.")
            else:
                print(idx, i, "knots being used with error of", error_new)
            if i > max_knots and error_old < error_new:
                self.clean_predictions[idx, :] = clean_predictions_old
            else:
                self.clean_predictions[idx, :] = clean_predictions_new
            self.predict_knots.append(q_pl)

        for idx in range(num_obs):
            clean_obs_old, error_old, q_pl = linear_c0_spline(times=times,
                                                              data=self.observed_time_series[idx,
                                                                                             time_start_idx:time_end_idx + 1],
                                                              num_knots=min_knots,
                                                              clean_times=clean_times, verbose=verbose)
            i = min_knots + 1
            while i <= max_knots:
                clean_obs_new, error_new, q_pl = linear_c0_spline(times=times,
                                                                  data=self.observed_time_series[idx, time_start_idx:time_end_idx + 1],
                                                                  num_knots=i,
                                                                  clean_times=clean_times,
                                                                  spline_old=q_pl, verbose=verbose)
                print(idx, i, error_new)
                diff = np.average(np.abs(clean_obs_new - clean_obs_old)) / np.average(
                    np.abs(self.observed_time_series[idx, time_start_idx:time_end_idx + 1]))
                if diff < tol:
                    break
                else:
                    i += 1
                    if i <= max_knots:
                        clean_obs_old = clean_obs_new
            if i > max_knots:
                print("Warning: maximum number of knots reached.")
            else:
                print(idx, i, "knots being used with error of", error_new)
            if i > max_knots and error_old < error_new:
                self.clean_obs[idx, :] = clean_obs_old
            else:
                self.clean_obs[idx, :] = clean_obs_new
            self.obs_knots.append(q_pl)
        self.clean_times = clean_times
        return self.clean_predictions, self.clean_obs, self.clean_times

    def clean_data_tol(
            self,
            time_start_idx,
            time_end_idx,
            num_clean_obs,
            tol,
            min_knots=3,
            max_knots=100,
            verbose=False):
        """
        Clean observed and predicted time series data so that the mean l1 error is within a tolerance.
        :param time_start_idx: first time index to clean
        :type time_start_idx: int
        :param time_end_idx: last time index to clean
        :type time_end_idx: int
        :param num_clean_obs: number of clean observations to make
        :type num_clean_obs: int
        :param tol: tolerance for constructing splines
        :type tol: float
        :param min_knots: maximum number of knots allowed
        :type min_knots: int
        :param max_knots: minimum number of knots allowed
        :type max_knots: int
        :param verbose: display termination reports
        :type verbose: bool
        :return: arrays of clean predictions, clean observations, and clean times
        :rtype: :class:`numpy.ndarray`, :class:`numpy.ndarray`, :class:`numpy.ndarray`
        """

        i = min_knots
        # Use _old and _new to compare to tol and determine when to stop adding knots
        # Compute _old before looping and then i=i+1
        times = self.times[time_start_idx:time_end_idx + 1]
        clean_times = np.linspace(
            self.times[time_start_idx],
            self.times[time_end_idx],
            num_clean_obs)
        num_predictions = self.predicted_time_series.shape[0]
        num_obs = self.observed_time_series.shape[0]
        self.clean_predictions = np.zeros((num_predictions, num_clean_obs))
        self.clean_obs = np.zeros((num_obs, num_clean_obs))

        for idx in range(num_predictions):
            i = min_knots
            q_pl_old = None
            while i <= max_knots:
                clean_predictions, error, q_pl = linear_c0_spline(times=times,
                                                                  data=self.predicted_time_series[idx, time_start_idx:time_end_idx + 1],
                                                                  num_knots=i,
                                                                  clean_times=clean_times,
                                                                  spline_old=q_pl_old)

                # After an _old and a _new is computed (when i>min_knots)
                print(idx, i, error)
                if error <= tol:
                    break
                else:
                    i += 1
                    q_pl_old = q_pl
                    # and _old = _new
            if i > max_knots:
                print("Warning: maximum number of knots reached.")
            else:
                print(idx, i, "knots being used")
            self.clean_predictions[idx, :] = clean_predictions

        for idx in range(num_obs):
            i = min_knots
            q_pl_old = None
            while i <= max_knots:
                clean_obs, error, q_pl = linear_c0_spline(times,
                                                          data=self.observed_time_series[idx, time_start_idx:time_end_idx + 1],
                                                          num_knots=i,
                                                          clean_times=clean_times,
                                                          spline_old=q_pl_old)
                # After an _old and a _new is computed (when i>min_knots)
                print(idx, i, error)
                if error <= tol:
                    break
                else:
                    i += 1
                    q_pl_old = q_pl
                    # and _old = _new
            if i > max_knots:
                print("Warning: maximum number of knots reached.")
            else:
                print(idx, i, "knots being used.")
            self.clean_obs[idx, :] = clean_obs
        self.clean_times = clean_times

        return self.clean_predictions, self.clean_obs, self.clean_times

    def dynamics(self,
                 cluster_method='kmeans',
                 kwargs={'n_clusters': 3,
                         'n_init': 10},
                 proposals=({'kernel': 'linear'},
                            {'kernel': 'rbf'},
                            {'kernel': 'poly'},
                            {'kernel': 'sigmoid'}),
                 k=10):
        """
        Learn and classify dynamics, then classify observations.
        :param cluster_method: type of clustering to use ('kmeans' or 'spectral')
        :type cluster_method: str
        :param kwargs: keyword arguments for clustering method
        :type kwargs: dict
        :param proposals: proposal keyword arguments for svm classifier
        :type proposals: list
        :param k: number of cases for k-fold cross-validation
        :type k: int
        """

        self.learn_dynamics(cluster_method=cluster_method, kwargs=kwargs)
        self.classify_dynamics(proposals=proposals, k=k)
        self.classify_observations()

    def learn_dynamics(
        self,
        cluster_method='kmeans',
        kwargs={
            'n_clusters': 3,
            'n_init': 10}):
        """
        Learn dynamics.
        :param cluster_method: type of clustering to use ('kmeans' or 'spectral')
        :type cluster_method: str
        :param kwargs: keyword arguments for clustering method
        :type kwargs: dict
        :return: cluster labels and inertia (None if not kmeans)
        :rtype: :class:`numpy.ndarray`, float
        """
        if cluster_method == 'kmeans':
            self.cluster_labels, inertia = self.learn_dynamics_kmeans(kwargs)
        elif cluster_method == 'spectral':
            self.cluster_labels = self.learn_dynamics_spectral(kwargs)
            inertia = None
        elif cluster_method == 'dbscan':
            self.cluster_labels = self.learn_dynamics_dbscan(kwargs)
            inertia = None
        elif cluster_method == 'gmm':
            self.cluster_labels = self.learn_dynamics_gmm(kwargs)
            inertia = None
        self.num_clusters = int(np.max(self.cluster_labels) + 1)
        return self.cluster_labels, inertia

    def learn_dynamics_kmeans(self, kwargs):
        """
        Perform clustering with k-means.
        See https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html.
        :param kwargs: keyword arguments for sklearn
        :type kwargs: dict
        :return: cluster labels and inertias
        :rtype: :class:`numpy.ndarray`, float
        """
        from sklearn.cluster import KMeans

        k_means = KMeans(init='k-means++', **kwargs)
        k_means.fit(self.clean_predictions)
        return k_means.labels_, k_means.inertia_

    def learn_dynamics_spectral(self, kwargs):
        """
        Perform clustering with spectral clustering.
        See https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html#sklearn.cluster.SpectralClustering
        :param kwargs: keyword arguments
        :type kwargs: dict
        :return: cluster labels
        :rtype: :class:`numpy.ndarray`
        """
        from sklearn.cluster import SpectralClustering
        clustering = SpectralClustering(**kwargs).fit(self.clean_predictions)
        return clustering.labels_

    def learn_dynamics_dbscan(self, kwargs):
        """
        Perform clustering with DBSCAN clustering.
        See https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
        :param kwargs: keyword arguments
        :type kwargs: dict
        :return: cluster labels
        :rtype: :class:`numpy.ndarray`
        """
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(**kwargs).fit(self.clean_predictions)
        return clustering.labels_

    def learn_dynamics_gmm(self, kwargs):
        """
        Perform clustering with a Gaussian Mixture Model.
        See https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html.
        :param kwargs: keyword arguments
        :type kwargs: dict
        :return: cluster labels
        :rtype: :class:`numpy.ndarray`
        """
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(**kwargs)
        gmm.fit(self.clean_predictions)
        return gmm.predict(self.clean_predictions)

    def classify_dynamics(self,
                          proposals=({'kernel': 'linear'},
                                     {'kernel': 'rbf'},
                                     {'kernel': 'poly'},
                                     {'kernel': 'sigmoid'}),
                          k=10):
        """
        Classify dynamics using best SVM method based on k-fold cross validation from a list of proposal keyword
        arguments for `sklearn.svm.LinearSVC`.
        https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
        :param proposals: list of proposal SVM keyword arguments
        :type proposals: list
        :param k: k for k-fold cross validation
        :type k: int
        :return: classifier object and labels of predictions
        :rtype: :class:`sklearn.svm.SVC`, :class:`numpy.ndarray`
        """
        clfs = []
        misclass_rates = []

        mis_min = 1.0
        ind_min = None

        for i, prop in enumerate(proposals):
            try:
                clf, mis = self.classify_dynamics_svm_kfold(k=k, kwargs=prop)
            except BaseException:
                clf = None
                mis = 1.0
            clfs.append(clf)
            misclass_rates.append(mis)
            if mis <= mis_min:
                mis_min = mis
                ind_min = i
        print('Best classifier is ', proposals[ind_min])
        print('Misclassification rate is ', mis_min)
        self.classifier = clfs[ind_min]
        self.predict_labels = self.classifier.predict(self.clean_predictions)
        return self.classifier, self.predict_labels

    def classify_dynamics_svm_kfold(self, k=10, kwargs=None):
        """
        Classify dynamics with given SVM method and do k-fold cross validation.
        :param k: k for k-fold cross validation
        :type k: int
        :param kwargs: keyword arguments for SVM
        :type kwargs: dict
        :return: classifier object and misclassification rate
        :rtype: :class:`sklearn.svm.SVC`, float
        """
        import numpy.random as nrand
        num_clean = self.clean_predictions.shape[0]
        inds = nrand.choice(num_clean, num_clean, replace=False)
        binsize = int(num_clean / k)
        randomized_preds = self.clean_predictions[inds, :]
        randomized_labels = self.cluster_labels[inds]
        misclass_rates = []
        for i in range(k):
            testing_set = randomized_preds[i * binsize:(i + 1) * binsize, :]
            testing_labels = randomized_labels[i * binsize:(i + 1) * binsize]
            training_set = np.vstack(
                (randomized_preds[0:i * binsize, :], randomized_preds[(i + 1) * binsize:, :]))
            training_labels = np.hstack(
                (randomized_labels[0:i * binsize], randomized_labels[(i + 1) * binsize:]))

            clf = self.classify_dynamics_svm(
                kwargs=kwargs, data=training_set, labels=training_labels)
            new_labels = clf.predict(testing_set)
            misclass_rates.append(
                np.average(
                    np.not_equal(
                        new_labels,
                        testing_labels)))
        print(
            np.average(misclass_rates),
            'misclassification rate for ',
            kwargs)
        clf = self.classify_dynamics_svm(
            kwargs=kwargs,
            data=self.clean_predictions,
            labels=self.cluster_labels)
        return clf, np.average(misclass_rates)

    def classify_dynamics_svm(self, kwargs=None, data=None, labels=None):
        """
        Classify dynamics with SVM.
        :param kwargs: keyword arguments for SVM
        :type kwargs: dict
        :param data: data to classify
        :type data: :class:`numpy.ndarray`
        :param labels: labels for supervised learning
        :type labels: :class:`numpy.ndarray`
        :return: classifier object
        :rtype: :class:`sklearn.svm.SVC`
        """
        from sklearn import svm
        if data is None:
            data = self.clean_predictions
            labels = self.cluster_labels

        clf = svm.SVC(gamma='auto', **kwargs)
        clf.fit(data, labels)

        return clf

    def learn_qoi(self,
                  variance_rate=None,
                  proposals=({'kernel': 'linear'},
                             {'kernel': 'rbf'},
                             {'kernel': 'sigmoid'},
                             {'kernel': 'poly'},
                             {'kernel': 'cosine'}),
                  num_qoi=None):
        """
        Learn best quantities of interest from proposal kernel PCAs.
        See https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html.
        :param variance_rate: proportion of variance QoIs should capture.
        :type variance_rate: float
        :param proposals: proposal keyword arguments for kPCAs (tuple of dictionaries)
        :type proposals: tuple
        :param num_qoi: number of quantities of interest to take (optional)
        :type num_qoi: int
        :return: Kernel PCA object
        :rtype: :class:`sklearn.decomposition.KernelPCA`
        """
        from sklearn.decomposition import KernelPCA
        from sklearn.preprocessing import StandardScaler

        if variance_rate is None and num_qoi is None:
            variance_rate = 0.99

        if self.num_clusters is None:
            # Set up single cluster if no clustering has been done
            print("No clustering performed. Assuming a single cluster.")
            self.num_clusters = 1
            self.predict_labels = np.array(
                self.predicted_time_series.shape[0] * [0])
            self.obs_labels = np.array(
                self.observed_time_series.shape[0] * [0])

        self.kpcas = []
        self.q_predict_kpcas = []
        self.num_pcs = []
        self.variance_rate = []
        self.Xpcas = []
        self.scalers = []

        if variance_rate is not None:
            for i in range(self.num_clusters):
                scaler = StandardScaler()
                X_std = scaler.fit_transform(
                    self.clean_predictions[np.where(self.predict_labels == i)[0], :])
                self.scalers.append(scaler)
                kpcas_local = []
                X_kpca_local = []
                num_pcs = []
                rate = []
                eigenvalues = []
                ind_best = None
                num_pcs_best = np.inf
                rate_best = 0.0
                cum_sum_best = None
                for j, kwargs in enumerate(proposals):
                    try:
                        kpca = KernelPCA(**kwargs)
                        X_kpca = kpca.fit_transform(X_std)
                        X_kpca_local.append(X_kpca)
                        kpcas_local.append(kpca)
                        eigs = kpca.lambdas_
                        eigenvalues.append(eigs)
                        eigs = eigs / np.sum(eigs)
                        cum_sum = np.cumsum(eigs)
                        num_pcs.append(
                            int(np.sum(np.less_equal(cum_sum, variance_rate))) + 1)
                        rate.append(cum_sum[num_pcs[-1] - 1])
                    except BaseException:
                        print(kwargs, ' not supported.')
                        X_kpca_local.append(None)
                        kpcas_local.append(None)
                        eigenvalues.append(None)
                        num_pcs.append(np.inf)
                        rate.append(0.0)
                    if num_pcs[-1] < num_pcs_best:
                        num_pcs_best = num_pcs[-1]
                        ind_best = j
                        rate_best = rate[-1]
                        cum_sum_best = cum_sum
                    elif num_pcs[-1] == num_pcs_best:
                        if rate[-1] > rate_best:
                            ind_best = j
                            rate_best = rate[-1]
                    print(num_pcs[-1],
                          'PCs explain',
                          "{:.4%}".format(rate[-1]),
                          'of var. for cluster',
                          i + 1,
                          'with',
                          proposals[j])

                self.kpcas.append(kpcas_local[ind_best])
                self.q_predict_kpcas.append(X_kpca_local[ind_best])
                if num_qoi is None:
                    self.num_pcs.append(num_pcs[ind_best])
                else:
                    self.num_pcs.append(num_qoi)
                self.variance_rate.append(rate[ind_best])
                self.Xpcas.append(X_kpca_local[ind_best])
                print('---------------------------------------------')
                print(
                    'Best kPCA for cluster ',
                    i + 1,
                    ' is ',
                    proposals[ind_best])
                # print(self.num_pcs[-1], 'principal components explain', "{:.4%}".format(self.variance_rate[-1]),
                #      'of variance.')
                print(self.num_pcs[-1],
                      'PCs explain',
                      "{:.4%}".format(cum_sum_best[self.num_pcs[-1] - 1]),
                      'of variance.')
                print('---------------------------------------------')
        else:
            for i in range(self.num_clusters):
                scaler = StandardScaler()
                predict_ptr = np.where(self.predict_labels == i)[0]
                X_std = scaler.fit_transform(
                    self.clean_predictions[predict_ptr, :])
                self.scalers.append(scaler)
                kpcas_local = []
                X_kpca_local = []
                eigenvalues = []
                ind_best = None
                rate_best = 0.0
                for j, kwargs in enumerate(proposals):
                    try:
                        kpca = KernelPCA(**kwargs)
                        X_kpca = kpca.fit_transform(X_std)
                        X_kpca_local.append(X_kpca)
                        kpcas_local.append(kpca)
                        eigs = kpca.lambdas_
                        eigenvalues.append(eigs)
                        eigs = eigs / np.sum(eigs)
                        cum_sum = np.cumsum(eigs)
                        rate = cum_sum[num_qoi - 1]
                    except BaseException:
                        print(kwargs, ' not supported.')
                        X_kpca_local.append(None)
                        kpcas_local.append(None)
                        eigenvalues.append(None)
                        # num_pcs.append(np.inf)
                        rate = 0.0

                    if rate > rate_best:
                        rate_best = rate
                        ind_best = j
                    print(num_qoi, 'PCs explain', "{:.4%}".format(rate),
                          'of var. for cluster', i + 1,
                          'with', proposals[j])
                self.kpcas.append(kpcas_local[ind_best])
                self.q_predict_kpcas.append(X_kpca_local[ind_best])

                self.num_pcs.append(num_qoi)
                self.variance_rate.append(rate_best)
                self.Xpcas.append(X_kpca_local[ind_best])
                print('---------------------------------------------')
                print(
                    'Best kPCA for cluster ',
                    i + 1,
                    ' is ',
                    proposals[ind_best])
                # print(self.num_pcs[-1], 'principal components explain', "{:.4%}".format(self.variance_rate[-1]),
                #      'of variance.')
                print(self.num_pcs[-1], 'PCs explain',
                      "{:.4%}".format(rate_best),
                      'of variance.')
                print('---------------------------------------------')
        return self.kpcas

    def choose_qois(self):
        """
        Transform predicted time series to new QoIs.
        :return: transformed predictions
        :rtype: :class:`numpy.ndarray`
        """
        self.predict_maps = []
        for i in range(self.num_clusters):
            self.predict_maps.append(self.Xpcas[i][:, 0:self.num_pcs[i]])
        return self.predict_maps

    def classify_observations(self):
        """
        Classify observations into dynamics clusters.
        :return: cluster labels for observations
        :rtype: :class:`numpy.ndarray`
        """
        self.obs_labels = self.classifier.predict(self.clean_obs)
        # Mark empty observation clusters
        for i in range(self.num_clusters):
            if len(np.where(self.obs_labels == i)[0]) == 0:
                self.obs_empty_cluster.append(True)
            else:
                self.obs_empty_cluster.append(False)
        return self.obs_labels

    def transform_observations(self):
        """
        Transform observed time series to new QoIs.
        :return: transformed observations
        :rtype: :class:`numpy.ndarray`
        """
        self.obs_maps = []
        for i in range(self.num_clusters):
            if not self.obs_empty_cluster[i]:
                X_std = self.scalers[i].transform(
                    self.clean_obs[np.where(self.obs_labels == i)[0], :])
                X_kpca = self.kpcas[i].transform(X_std)
                self.obs_maps.append(X_kpca[:, 0:self.num_pcs[i]])
            else:
                self.obs_maps.append(None)
        return self.obs_maps

    def learn_qois_and_transform(self,
                                 variance_rate=None,
                                 proposals=({'kernel': 'linear'},
                                            {'kernel': 'rbf'},
                                            {'kernel': 'sigmoid'},
                                            {'kernel': 'poly'},
                                            {'kernel': 'cosine'}),
                                 num_qoi=None):
        """
        Learn Quantities of Interest and transform time series data.
        :param variance_rate: proportion of variance QoIs should capture.
        :type variance_rate: float
        :param proposals: proposal keyword arguments for kPCAs (a tuple of dictionaries)
        :type proposals: tuple
        :param num_qoi: number of quantities of interest to take (optional)
        :type num_qoi: int
        :return: transformed prediction and observation maps
        :rtype: :class:`numpy.ndarray`, :class:`numpy.ndarray`
        """
        self.learn_qoi(
            variance_rate=variance_rate,
            proposals=proposals,
            num_qoi=num_qoi)
        self.choose_qois()
        self.transform_observations()
        return self.predict_maps, self.obs_maps

    # Move the below out of LUQ
    def generate_kdes(self):
        from scipy.stats import gaussian_kde as GKDE

        self.pi_predict_kdes = []
        self.pi_obs_kdes = []

        for i in range(self.num_clusters):
            self.pi_predict_kdes.append(GKDE(self.predict_maps[i].T))
            self.pi_obs_kdes.append(GKDE(self.obs_maps[i].T))
        return self.pi_predict_kdes, self.pi_obs_kdes

    def compute_r(self):

        def rejection_sampling(r):
            # Perform accept/reject sampling on a set of proposal samples using
            # the weights r associated with the set of samples and return
            # the indices idx of the proposal sample set that are accepted.
            N = r.size  # size of proposal sample set
            # create random uniform weights to check r against
            check = np.random.uniform(low=0, high=1, size=N)
            M = np.max(r)
            new_r = r / M  # normalize weights
            idx = np.where(new_r >= check)[0]  # rejection criterion
            return idx

        r = []
        rs = []
        samples_to_keep = []

        for i in range(self.num_clusters):
            # First compute the rejection ratio
            r.append(
                np.divide(
                    self.pi_obs_kdes[i](
                        self.predict_maps[i].T), self.pi_predict_kdes[i](
                        self.predict_maps[i].T)))

            # Now perform rejection sampling and return the indices we keep
            samples_to_keep.append(rejection_sampling(r[i]))

            rs.append((r[i].mean()))
        print(
            'Diagnostic for clusters [sample average of ratios in each cluster]:',
            rs)
        self.r = r
        return self.r

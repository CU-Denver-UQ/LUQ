# Copyright 2019 Steven A. Mattis and Troy Butler

import numpy as np
from splines import *


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
        self.clean_times = None   # surrogate_times -> clean_times
        self.clean_predictions = None # surrogate -> clean
        self.clean_obs = None
        self.num_clusters = None
        self.cluster_labels = None
        self.predict_labels = None
        self.obs_labels = None
        self.kpcas = None
        self.q_predict_kpcas = None
        self.predict_maps = []
        self.obs_maps = []
        self.num_pcs = []
        self.variance_rate = []
        self.Xpcas = []

        # incorporate more into this
        self.info = {'clustering_method': None,
                     'num_clusters': None,
                     'classifier_type': None,
                     'classifier_kernel': None,
                     'misclassification_rate': None,
                     'kpca_kernel': None,
                     'num_principal_components': None}

    def clean_data(self, time_start_idx, time_end_idx, num_clean_obs, tol, min_knots=3, max_knots=100):
        """
        Clean observed and predicted time series data so that difference between iterations is within tolerance.
        :param time_start_idx: first time index to clean
        :param time_end_idx: last time index to clean
        :param num_clean_obs: number of clean observations to make
        :param tol: tolerance for constructing splines
        :param min_knots: maximum number of knots allowed
        :param max_knots: minimum number of knots allowed
        :return:
        """

        i = min_knots
        # Use _old and _new to compare to tol and determine when to stop adding knots
        # Compute _old before looping and then i=i+1
        times = self.times[time_start_idx:time_end_idx + 1]
        clean_times = np.linspace(self.times[time_start_idx], self.times[time_end_idx], num_clean_obs)
        num_predictions = self.predicted_time_series.shape[0]
        num_obs = self.observed_time_series.shape[0]
        self.clean_predictions = np.zeros((num_predictions, num_clean_obs))
        self.clean_obs = np.zeros((num_obs, num_clean_obs))

        for idx in range(num_predictions):
            clean_predictions_old, error_old = linear_C0_spline(times=times,
                                                                data=self.predicted_time_series[idx,
                                                                     time_start_idx:time_end_idx + 1],
                                                                num_knots=min_knots,
                                                                clean_times=clean_times)
            i = min_knots + 1
            while i <= max_knots:
                clean_predictions_new, error_new = linear_C0_spline(times=times,
                            data=self.predicted_time_series[idx, time_start_idx:time_end_idx + 1],
                            num_knots=i,
                            clean_times=clean_times)

                # After an _old and a _new is computed (when i>min_knots)
                print(idx, i, error_new)
                diff = np.average(np.abs(clean_predictions_new - clean_predictions_old)) / \
                    np.average(np.abs(self.predicted_time_series[idx,
                                                                     time_start_idx:time_end_idx + 1]))
                if diff < tol:
                    break
                else:
                    i += 1
                    clean_predictions_old = clean_predictions_new
            if i > max_knots:
                print("Warning: maximum number of knots reached.")
            else:
                print(idx, i, "knots being used with error of", error_new)
            self.clean_predictions[idx, :] = clean_predictions_new

        for idx in range(num_obs):
            clean_obs_old, error_old = linear_C0_spline(times=times,
                                                        data=self.observed_time_series[idx,
                            time_start_idx:time_end_idx + 1],
                                                        num_knots=min_knots,
                                                        clean_times=clean_times)
            i = min_knots + 1
            while i <= max_knots:
                clean_obs_new, error_new = linear_C0_spline(times=times,
                            data=self.observed_time_series[idx, time_start_idx:time_end_idx + 1],
                            num_knots=i,
                            clean_times=clean_times)
                print(idx, i, error_new)
                diff = np.average(np.abs(clean_obs_new - clean_obs_old)) / \
                    np.average(np.abs(self.observed_time_series[idx, time_start_idx:time_end_idx + 1]))
                if diff < tol:
                    clean_obs_old = clean_obs_new
                    break
                else:
                    i += 1
            if i > max_knots:
                print("Warning: maximum number of knots reached.")
            else:
                print(idx, i, "knots being used with error of", error_new)
            self.clean_obs[idx, :] = clean_obs_new
        return self.clean_predictions, self.clean_obs, self.clean_times

    def clean_data_tol(self, time_start_idx, time_end_idx, num_clean_obs, tol, min_knots=3, max_knots=100):
        """
        Clean observed and predicted time series data so that the mean l1 error is within a tolerance.
        :param time_start_idx: first time index to clean
        :param time_end_idx: last time index to clean
        :param num_clean_obs: number of clean observations to make
        :param tol: tolerance for constructing splines
        :param min_knots: maximum number of knots allowed
        :param max_knots: minimum number of knots allowed
        :return:
        """

        i = min_knots
        # Use _old and _new to compare to tol and determine when to stop adding knots
        # Compute _old before looping and then i=i+1
        times = self.times[time_start_idx:time_end_idx + 1]
        clean_times = np.linspace(self.times[time_start_idx], self.times[time_end_idx], num_clean_obs)
        num_predictions = self.predicted_time_series.shape[0]
        num_obs = self.observed_time_series.shape[0]
        self.clean_predictions = np.zeros((num_predictions, num_clean_obs))
        self.clean_obs = np.zeros((num_obs, num_clean_obs))

        for idx in range(num_predictions):
            i = min_knots
            while i <= max_knots:
                clean_predictions, error = linear_C0_spline(times=times,
                            data=self.predicted_time_series[idx, time_start_idx:time_end_idx + 1],
                            num_knots=i,
                            clean_times=clean_times)

                # After an _old and a _new is computed (when i>min_knots)
                print(idx, i, error)
                if error <= tol:
                    break
                else:
                    i += 1
                    # and _old = _new
            if i > max_knots:
                print("Warning: maximum number of knots reached.")
            else:
                print(idx, i, "knots being used")
            self.clean_predictions[idx, :] = clean_predictions

        for idx in range(num_obs):
            i = min_knots
            while i <= max_knots:
                clean_obs, error = linear_C0_spline(times,
                                    data=self.observed_time_series[idx, time_start_idx:time_end_idx + 1],
                                    num_knots=i,
                                    clean_times=clean_times)
                # After an _old and a _new is computed (when i>min_knots)
                print(idx, i, error)
                if error <= tol:
                    break
                else:
                    i += 1
                    # and _old = _new
            if i > max_knots:
                print("Warning: maximum number of knots reached.")
            else:
                print(idx, i, "knots being used.")
            self.clean_obs[idx, :] = clean_obs
        self.clean_times = clean_times

        return self.clean_predictions, self.clean_obs, self.clean_times

    def dynamics(self, cluster_method='kmeans',
                 kwargs={'n_clusters': 3, 'n_init': 10},
                 proposals = ({'kernel': 'linear'},
                 {'kernel': 'rbf'}, {'kernel': 'poly'}, {'kernel': 'sigmoid'}),
                 k = 10):
        """

        :param cluster_method: type of clustering to use ('kmeans' or 'spectral')
        :param kwargs: keyword arguments for clustering method
        :param proposals: proposal keyword arguments for svm classifier
        :param k: number of cases for k-fold cross-validation
        :return:
        """

        self.learn_dynamics(cluster_method=cluster_method, kwargs=kwargs)
        self.classify_dynamics(proposals=proposals, k=k)
        return
    
    def learn_dynamics(self, cluster_method='kmeans', kwargs={'n_clusters': 3, 'n_init': 10}):
        """
        Learn dynamics
        :param cluster_method: type of clustering to use ('kmeans' or 'spectral')
        :param kwargs: keyword arguments for clustering method
        :return:
        """
        if cluster_method == 'kmeans':
            self.cluster_labels, inertia = self.learn_dynamics_kmeans(kwargs)
        elif cluster_method == 'spectral':
            self.cluster_labels = self.learn_dynamics_spectral(kwargs)
            inertia = None
        self.num_clusters = int(np.max(self.cluster_labels) + 1)
        return self.cluster_labels, inertia

    def learn_dynamics_kmeans(self, kwargs):
        """
        Perform clustering with k-means.
        :param kwargs: keyword arguments
        :return:
        """
        from sklearn.cluster import KMeans

        k_means = KMeans(init='k-means++', **kwargs)
        k_means.fit(self.clean_predictions)
        return k_means.labels_, k_means.inertia_

    def learn_dynamics_spectral(self, kwargs):
        """
        Perform clustering with spectral clustering.
        :param kwargs: keyword arguments
        :return:
        """
        from sklearn.cluster import SpectralClustering
        clustering = SpectralClustering(**kwargs).fit(self.clean_predictions)
        return clustering.labels_

    def classify_dynamics(self, proposals=({'kernel': 'linear'},
                                           {'kernel': 'rbf'}, {'kernel': 'poly'}, {'kernel': 'sigmoid'}), k=10):
        """
        Classify dynamics using best SVM method based on k-fold cross validation.
        :param proposals: proposal SVM keyword arguments
        :param k: k for k-fold cross validation
        :return:
        """
        clfs = []
        misclass_rates = []

        mis_min = 1.0
        ind_min = None

        for i, prop in enumerate(proposals):
            clf, mis = self.classify_dynamics_svm_kfold(k=k, kwargs=prop)
            clfs.append(clf)
            misclass_rates.append(mis)
            if mis <= mis_min:
                mis_min = mis
                ind_min = i
        print('Best classifier is ', proposals[ind_min])
        print('Misclassification rate is ', mis_min)
        self.classifier = clfs[i]
        self.predict_labels = self.classifier.predict(self.clean_predictions)
        return self.classifier, self.predict_labels

    def classify_dynamics_svm_kfold(self, k=10,  kwargs={}):
        """
        Classify dynamics with given SVM method and do k-fold cross validation.
        :param k: k for k-fold cross validation
        :param kwargs: keyword arguments for SVM
        :return:
        """
        import numpy.random as nrand
        num_clean = self.clean_predictions.shape[0]
        inds = nrand.choice(num_clean, num_clean, replace=False)
        binsize = int(num_clean/k)
        randomized_preds = self.clean_predictions[inds, :]
        randomized_labels = self.cluster_labels[inds]
        misclass_rates = []
        for i in range(k):
            testing_set = randomized_preds[i*binsize:(i+1)*binsize, :]
            testing_labels = randomized_labels[i*binsize:(i+1)*binsize]
            training_set = np.vstack((randomized_preds[0:i*binsize, :], randomized_preds[(i+1)*binsize:, :]))
            training_labels = np.hstack((randomized_labels[0:i*binsize], randomized_labels[(i+1)*binsize:]))

            clf = self.classify_dynamics_svm(kwargs=kwargs, data=training_set, labels=training_labels)
            new_labels = clf.predict(testing_set)
            misclass_rates.append(np.average(np.not_equal(new_labels, testing_labels)))
        print(np.average(misclass_rates), 'misclassification rate for ', kwargs)
        clf = self.classify_dynamics_svm(kwargs=kwargs, data=self.clean_predictions, labels=self.cluster_labels)
        return clf, np.average(misclass_rates)

    def classify_dynamics_svm(self, kwargs={}, data=None, labels=None):
        """
        Classify dynamics with SVM.
        :param kwargs: keyword arguments for SVM
        :param data: data to classify
        :param labels: labels for supervised learning
        :return:
        """
        from sklearn import svm
        if data is None:
            data = self.clean_predictions
            labels = self.cluster_labels

        clf = svm.SVC(gamma='auto', **kwargs)
        clf.fit(data, labels)
        return clf
    
    def learn_qoi(self, variance_rate=0.95,
                  proposals=({'kernel': 'linear'}, {'kernel': 'rbf'},
                             {'kernel': 'sigmoid'}, {'kernel': 'poly'}, {'kernel': 'cosine'}),
                  num_qoi=None):
        """
        Learn best quantities of interest from proposal kernel PCAs.
        :param variance_rate: proportion of variance QoIs should capture.
        :param proposals: proposal keyword arguments for kPCAs
        :param num_qoi: number of quantities of interest to take (optional)
        :return:
        """
        from sklearn.decomposition import PCA, KernelPCA
        from sklearn.preprocessing import StandardScaler

        self.kpcas = []
        self.q_predict_kpcas = []
        self.num_pcs = []
        self.variance_rate =[]
        self.Xpcas = []
        for i in range(self.num_clusters):
            scaler = StandardScaler()
            X_std = scaler.fit_transform(self.clean_predictions[np.where(self.predict_labels==i)[0], :])
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
                kpca = KernelPCA(**kwargs)
                X_kpca = kpca.fit_transform(X_std)
                X_kpca_local.append(X_kpca)
                kpcas_local.append(kpca)
                eigs = kpca.lambdas_
                eigenvalues.append(eigs)
                eigs = eigs / np.sum(eigs)
                cum_sum = np.cumsum(eigs)
                num_pcs.append(int(np.sum(np.less_equal(cum_sum, variance_rate)))+1)
                rate.append(cum_sum[num_pcs[-1] - 1])
                if num_pcs[-1] < num_pcs_best:
                    num_pcs_best = num_pcs[-1]
                    ind_best = j
                    rate_best = rate[-1]
                    cum_sum_best = cum_sum
                elif num_pcs[-1] == num_pcs_best:
                    if rate[-1] > rate_best:
                        ind_best = j
                        rate_best = rate[-1]
                print(num_pcs[-1], 'principal components explain', "{:.4%}".format(rate[-1]), 'of variance for cluster', i,
                      'with', proposals[j])

            self.kpcas.append(kpcas_local[ind_best])
            self.q_predict_kpcas.append(X_kpca_local[ind_best])
            if num_qoi is None:
                self.num_pcs.append(num_pcs[ind_best])
            else:
                self.num_pcs.append(num_qoi)
            self.variance_rate.append(rate[ind_best])
            self.Xpcas.append(X_kpca_local[ind_best])
            print('Best kPCA for cluster ', i, ' is ', proposals[ind_best])
            # print(self.num_pcs[-1], 'principal components explain', "{:.4%}".format(self.variance_rate[-1]),
            #      'of variance.')
            print(self.num_pcs[-1], 'principal components explain', "{:.4%}".format(cum_sum_best[self.num_pcs[-1]-1]),
                  'of variance.')
        return self.kpcas

    def choose_qois(self):
        """
        Transform predicted time series to new QoIs.
        :return:
        """
        self.predict_maps = []
        for i in range(self.num_clusters):
            self.predict_maps.append(self.Xpcas[i][:, 0:self.num_pcs[i]])
        return self.predict_maps

    def classify_observations(self):
        """
        Classify observations into dynamics clusters.
        :return:
        """
        self.obs_labels = self.classifier.predict(self.clean_obs)
        return self.obs_labels

    def transform_observations(self):
        """
        Transform observed time series to new QoIs.
        :return:
        """
        from sklearn.preprocessing import StandardScaler
        self.obs_maps = []
        for i in range(self.num_clusters):
            scaler = StandardScaler()
            X_std = scaler.fit_transform(self.clean_obs[np.where(self.obs_labels==i)[0], :])
            X_kpca = self.kpcas[i].transform(X_std)
            self.obs_maps.append(X_kpca[:, 0:self.num_pcs[i]])
        return self.obs_maps

    def learn_qois_and_transform(self, variance_rate=0.95,
                                 proposals=({'kernel': 'linear'}, {'kernel': 'rbf'},
                                            {'kernel': 'sigmoid'}, {'kernel': 'poly'}, {'kernel': 'cosine'}),
                                 num_qoi=None):
        """
        Learn Quantities of Interest and transform time series data.
        :param variance_rate: proportion of variance QoIs should capture.
        :param proposals: proposal keyword arguments for kPCAs
        :param num_qoi: number of quantities of interest to take (optional)
        :return:
        """
        self.learn_qoi(variance_rate=variance_rate, proposals=proposals, num_qoi=num_qoi)
        self.choose_qois()
        self.classify_observations()
        self.transform_observations()
        return self.predict_maps, self.obs_maps

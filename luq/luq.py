# Copyright 2019-2020 Steven A. Mattis and Troy Butler

import numpy as np
from luq.splines import *
from luq.rbf_fit import RBFFit

class LUQ(object):
    """
    Learning Uncertain Quantities:
    Parent class that utilizes filtered observed and corresponding predicted data to identify dynamics and
    low-dimensional structures and transforms the data.
    """

    def __init__(self,
                 predicted_data,
                 observed_data=None):
        """
        Initializes objects. Shapes of data should each be (n_samples, n_dimensions)
        :param predicted_data: data from predictions
        :type predicted_data: :class:`numpy.ndarray`
        :param observed_data: data from observations
        :type observed_data: :class:`numpy.ndarray`
        """

        self.predicted_data = predicted_data
        self.filtered_predictions = None
        self.observed_data = observed_data
        self.filtered_obs = None
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
        self.r = None

        self.info = {'pred_filtering_params': None,
                    'obs_filtering_params': None,
                    'clustering_method': None,
                    'num_clusters': None,
                    'classifier_type': None,
                    'classifier_kernel': None,
                    'misclassification_rate': None,
                    'kpca_kernel': None,
                    'num_principal_components': None}
        
    def filter_data(self,
                    filter_method,
                    **kwargs):
        '''
        Wrapper filtering function. Uses either filter_data_splines, filter_data_splines_tol, or filter_data_rbfs based on filter_method parameter.
        :param filter_method: controls which filtering method is used. Either 'splines', 'splines_tol', or 'rbfs'
        :type filter_method: string
        :return: arrays of filtered predictions, filtered observations, and filtered data coordinates. If using splines, also returnes filtered data coordinates
        :rtype: :class:`numpy.ndarray`, :class:`numpy.ndarray`, :class:`numpy.ndarray`
        '''
        if filter_method == 'splines' or filter_method == 'spline':
            self.filter_data_splines(**kwargs)
        elif filter_method == 'splines_tol' or filter_method == 'spline_tol':
            self.filter_data_splines_tol(**kwargs)
        elif filter_method == 'rbfs' or filter_method == 'rbf':
            self.filter_data_rbfs(**kwargs)
        else:
            if len(self.data_coordinates) == 1:
                recommendation = 'splines'
            else:
                recommendation = 'rbfs'
            print(f'Filtering method {filter_method} not recognized. Use splines, splines_tol, or rbfs. Recommend using {recommendation} based on data dimension.')
            
    def filter_data_splines(
            self,
            filtered_data_coordinates,
            tol,
            min_knots=3,
            max_knots=100,
            verbose=False,
            predicted_data_coordinates=None,
            observed_data_coordinates=None,
            filter_predictions=True,
            filter_observations=True):
        """
        Filter observed and predicted data so that difference between iterations is within tolerance.
        :param filtered_data_coordinates: data coordinates at which filtered data is computed
        :type filtered_data_coordinates: :class:`numpy.ndarray`
        :param tol: tolerance for constructing splines
        :type tol: float
        :param min_knots: maximum number of knots allowed
        :type min_knots: int
        :param max_knots: minimum number of knots allowed
        :type max_knots: int
        :param verbose: display termination reports
        :type verbose: bool
        :param predicted_data_coordinates: coordinates at which predicted data is collected
        :type predicted_data_coordinates: :class:'numpy.ndarray'
        :param observed_data_coordinates: coordinates at which observed data is collected
        :type observed_data_coordinates: :class:'numpy.ndarray'
        :param filter_predictions: check if predictions should be filtered
        :type filter_predictions: bool
        :param filter_observations: check if observations should be filtered
        :type filter_observations: bool
        :return: arrays of filtered predictions and filtered observations
        :rtype: :class:`numpy.ndarray`, :class:`numpy.ndarray`
        """

        self.filtered_data_coordinates = filtered_data_coordinates
        self.predicted_data_coordinates = predicted_data_coordinates
        self.observed_data_coordinates = observed_data_coordinates
        self.predict_knots = []
        self.obs_knots = []
        
        # Checking if necessary data is provided
        if self.observed_data is None and filter_observations:
            print('No observed data to filter. Set observed_data for filtering.')
            filter_observations = False
        
        # filtering observations
        if filter_observations:
            if self.observed_data_coordinates is None:
                self.observed_data_coordinates = self.filtered_data_coordinates

            self.info['obs_filtering_params'] = {'filter_method': 'splines',
                                                 'tol': tol,
                                                 'min_knots': min_knots,
                                                 'max_knots': max_knots}
            # Use _old and _new to compare to tol and determine when to stop adding knots
            # Compute _old before looping and then i=i+1
            num_obs = self.observed_data.shape[0]
            self.filtered_obs = np.zeros((num_obs, self.filtered_data_coordinates.shape[0]))
            for idx in range(num_obs):
                filtered_obs_old, error_old, q_pl = linear_c0_spline(data_coordinates=self.observed_data_coordinates,
                                                                data=self.observed_data[idx,:],
                                                                num_knots=min_knots,
                                                                filtered_data_coordinates=self.filtered_data_coordinates, 
                                                                verbose=verbose)
                i = min_knots + 1
                while i <= max_knots:
                    filtered_obs_new, error_new, q_pl = linear_c0_spline(data_coordinates=self.observed_data_coordinates,
                                                                    data=self.observed_data[idx,:],
                                                                    num_knots=i,
                                                                    filtered_data_coordinates=self.filtered_data_coordinates,
                                                                    spline_old=q_pl, verbose=verbose)
                    print(idx, i, error_new)
                    diff = np.average(np.abs(filtered_obs_new - filtered_obs_old)) / np.average(
                        np.abs(self.observed_data))
                    if diff < tol:
                        break
                    else:
                        i += 1
                        if i <= max_knots:
                            filtered_obs_old = filtered_obs_new
                if i > max_knots:
                    print("Warning: maximum number of knots reached.")
                else:
                    print(idx, i, "knots being used with error of", error_new)
                if i > max_knots and error_old < error_new:
                    self.filtered_obs[idx, :] = filtered_obs_old
                else:
                    self.filtered_obs[idx, :] = filtered_obs_new
                self.obs_knots.append(q_pl)

        # filtering predictions
        if filter_predictions:
            if self.predicted_data_coordinates is None:
                self.predicted_data_coordinates = self.filtered_data_coordinates

            self.info['pred_filtering_params'] = {'filter_method': 'splines',
                                                  'tol': tol,
                                                  'min_knots': min_knots,
                                                  'max_knots': max_knots}
            # Use _old and _new to compare to tol and determine when to stop adding knots
            # Compute _old before looping and then i=i+1
            num_predictions = self.predicted_data.shape[0]
            self.filtered_predictions = np.zeros((num_predictions, self.filtered_data_coordinates.shape[0]))
            for idx in range(num_predictions):
                filtered_predictions_old, error_old, q_pl = linear_c0_spline(data_coordinates=self.predicted_data_coordinates, 
                                                                            data=self.predicted_data[idx,:],
                                                                            num_knots=min_knots,
                                                                            filtered_data_coordinates=self.filtered_data_coordinates,
                                                                            verbose=verbose)
                i = min_knots + 1
                while i <= max_knots:
                    filtered_predictions_new, error_new, q_pl = linear_c0_spline(data_coordinates=self.predicted_data_coordinates, 
                                                                                data=self.predicted_data[idx,:],
                                                                                num_knots=i,
                                                                                filtered_data_coordinates=self.filtered_data_coordinates,
                                                                                spline_old=q_pl,
                                                                                verbose=verbose)

                    # After an _old and a _new is computed (when i>min_knots)
                    print(idx, i, error_new)
                    diff = np.average(np.abs(filtered_predictions_new - filtered_predictions_old)) / \
                        np.average(np.abs(self.predicted_data))
                    if diff < tol:
                        break
                    else:
                        i += 1
                        if i <= max_knots:
                            filtered_predictions_old = filtered_predictions_new
                if i > max_knots:
                    print("Warning: maximum number of knots reached.")
                else:
                    print(idx, i, "knots being used with error of", error_new)
                if i > max_knots and error_old < error_new:
                    self.filtered_predictions[idx, :] = filtered_predictions_old
                else:
                    self.filtered_predictions[idx, :] = filtered_predictions_new
                self.predict_knots.append(q_pl)

        return self.filtered_predictions, self.filtered_obs
         
    def filter_data_splines_tol(
            self,
            filtered_data_coordinates,
            tol,
            min_knots=3,
            max_knots=100,
            verbose=False,
            predicted_data_coordinates=None,
            observed_data_coordinates=None,
            filter_predictions=True,
            filter_observations=True):
        """
        Filter observed and predicted data so that the mean l1 error is within a tolerance.
        :param filtered_data_coordinates: data coordinates at which filtered data is computed
        :type filtered_data_coordinates: :class:`numpy.ndarray`
        :param tol: tolerance for constructing splines
        :type tol: float
        :param min_knots: maximum number of knots allowed
        :type min_knots: int
        :param max_knots: minimum number of knots allowed
        :type max_knots: int
        :param verbose: display termination reports
        :type verbose: bool
        :param predicted_data_coordinates: coordinates at which predicted data is collected
        :type predicted_data_coordinates: :class:'numpy.ndarray'
        :param observed_data_coordinates: coordinates at which observed data is collected
        :type observed_data_coordinates: :class:'numpy.ndarray'
        :param filter_predictions: check if predictions should be filtered
        :type filter_predictions: bool
        :param filter_observations: check if observations should be filtered
        :type filter_observations: bool
        :return: arrays of filtered predictions and filtered observations
        :rtype: :class:`numpy.ndarray`, :class:`numpy.ndarray`
        """

        self.filtered_data_coordinates = filtered_data_coordinates
        self.predicted_data_coordinates = predicted_data_coordinates
        self.observed_data_coordinates = observed_data_coordinates
        self.predict_knots = []
        self.obs_knots = []

        # Checking if necessary data is provided
        if self.observed_data is None and filter_observations:
            print('No observed data to filter. Set observed_data for filtering.')
            filter_observations = False
        
        # filtering observations
        if filter_observations:
            if self.observed_data_coordinates is None:
                self.observed_data_coordinates = self.filtered_data_coordinates

            self.info['obs_filtering_params'] = {'filter_method': 'splines_tol',
                                                 'tol': tol,
                                                 'min_knots': min_knots,
                                                 'max_knots': max_knots}
            num_obs = self.observed_data.shape[0]
            self.filtered_obs = np.zeros((num_obs, self.filtered_data_coordinates.shape[0]))
            for idx in range(num_obs):
                i = min_knots
                q_pl_old = None
                while i <= max_knots:
                    filtered_obs, error, q_pl = linear_c0_spline(data_coordinates=self.observed_data_coordinates, 
                                                                data=self.observed_data[idx,:],
                                                                num_knots=i,
                                                                filtered_data_coordinates=self.filtered_data_coordinates,
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
                self.filtered_obs[idx, :] = filtered_obs

        if filter_predictions:
            if self.predicted_data_coordinates is None:
                self.predicted_data_coordinates = self.filtered_data_coordinates

            self.info['pred_filtering_params'] = {'filter_method': 'splines_tol',
                                                  'tol': tol,
                                                  'min_knots': min_knots,
                                                  'max_knots': max_knots}
            num_predictions = self.predicted_data.shape[0]
            self.filtered_predictions = np.zeros((num_predictions, self.filtered_data_coordinates.shape[0]))
            for idx in range(num_predictions):
                i = min_knots
                q_pl_old = None
                while i <= max_knots:
                    filtered_predictions, error, q_pl = linear_c0_spline(data_coordinates=self.predicted_data_coordinates, 
                                                                        data=self.predicted_data[idx,:],
                                                                        num_knots=i,
                                                                        filtered_data_coordinates=self.filtered_data_coordinates,
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
                self.filtered_predictions[idx, :] = filtered_predictions

        return self.filtered_predictions, self.filtered_obs

    def filter_data_rbfs(self,
                         filtered_data_coordinates,
                         num_rbf_list,
                         remove_trend=False,
                         add_poly=False,
                         poly_deg=None,
                         initializer='Halton',
                         max_opt_count=3,
                         tol=1e-4,
                         predicted_data_coordinates=None,
                         observed_data_coordinates=None,
                         filter_predictions=True,
                         filter_observations=True):
        '''
        filters data by fitting weighted sum of Gaussians with optional polynomial
        :param filtered_data_coordinates: coordinates at which resulting fitted function is evaluated
        :type filtered_data_coordinates: :class:'numpy.ndarray'
        :param num_rbf_list: list of number of rbfs to fit
        :type num_rbf_list: int, list, or range
        :param remove_trend: controls whether a polynomial trend should be removed prior to fitting. If False, data is shifted to have mean of 0
        :type remove_trend: bool
        :param add_poly: controls if polynomial is added to weighted sum of rbfs in model function
        :type add_poly: bool
        :param poly_deg: degree of polynomial for polynomial trend and/or polynomial part of model function
        :type poly_deg: int
        :param initializer: Gaussian location initialization method. Must be either 'Halton', 'kmeans', 'uniform_random', or 'all'
        :type initializer: string
        :param max_opt_count: maximum number of opimization attempts per sample per num_rbfs
        :type max_opt_count: int
        :param tol: relative error tolerance that control whether to keep fit or move to next num_rbfs in num_rbf_list
        :type tol: float
        :param predicted_data_coordinates: coordinates at which predicted data is collected
        :type predicted_data_coordinates: :class:'numpy.ndarray'
        :param observed_data_coordinates: coordinates at which observed data is collected
        :type observed_data_coordinates: :class:'numpy.ndarray'
        :param filter_predictions: controls whether predicted data is filtered
        :type filter_predictions: bool
        :param filter_observations: controls whether observed data is filtered
        :type filter_observations: bool
        :return: returns filtered predictions and filtered observations
        :rtype: :class:'numpy.ndarray', :class:'numpy.ndarray'
        '''
        
        self.filtered_data_coordinates = filtered_data_coordinates
        self.predicted_data_coordinates = predicted_data_coordinates
        self.observed_data_coordinates = observed_data_coordinates
        
        # Checking if necessary data is provided
        if self.observed_data is None and filter_observations:
            print('No observed data to filter. Set observed_data for filtering.')
            filter_observations = False
        
        if filter_observations:
            self.info['obs_filtering_params'] = {'filter_method': 'rbfs',
                                                 'num_rbf_list': num_rbf_list,
                                                 'remove_trend': remove_trend,
                                                 'add_poly': add_poly,
                                                 'poly_deg': poly_deg,
                                                 'initializer': initializer,
                                                 'max_opt_count': max_opt_count,
                                                 'tol': tol}
            
            if self.observed_data_coordinates is None:
                self.observed_data_coordinates = self.filtered_data_coordinates

            fit_obs = RBFFit(self.observed_data_coordinates,
                            self.filtered_data_coordinates,
                            remove_trend,
                            add_poly,
                            poly_deg)
            
            self.filtered_obs = fit_obs.filter_data(self.observed_data, 
                                            num_rbf_list,
                                            initializer,
                                            max_opt_count,
                                            tol)
        
        if filter_predictions:
            self.info['pred_filtering_params'] = {'filter_method': 'rbfs',
                                                  'num_rbf_list': num_rbf_list,
                                                 'remove_trend': remove_trend,
                                                 'add_poly': add_poly,
                                                 'poly_deg': poly_deg,
                                                 'initializer': initializer,
                                                 'max_opt_count': max_opt_count,
                                                 'tol': tol}
            
            if self.predicted_data_coordinates is None:
                self.predicted_data_coordinates = self.filtered_data_coordinates

            fit_pred = RBFFit(self.predicted_data_coordinates, 
                              self.filtered_data_coordinates,
                              remove_trend,
                              add_poly,
                              poly_deg)
            
            self.filtered_predictions = fit_pred.filter_data(self.predicted_data, 
                                                        num_rbf_list,
                                                        initializer,
                                                        max_opt_count,
                                                        tol)
            
        return self.filtered_predictions, self.filtered_obs

    def dynamics(self,
                 cluster_method='kmeans',
                 custom_labels=None,
                 relabel_predictions=True,
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
        :param custom_labels: custom labels for predictions; if not None, no clustering is performed
        :type custom_labels: NoneType or :class:`numpy.ndarray`
        :param relabel_predictions: if custom_labels is not None, relabel_predictions controls if predictions are relabeled via SVM classification
        :type relabel_predictions: bool
        :param kwargs: keyword arguments for clustering method
        :type kwargs: dict
        :param proposals: proposal keyword arguments for svm classifier
        :type proposals: list
        :param k: number of cases for k-fold cross-validation
        :type k: int
        """

        if self.filtered_predictions is None:
            print('Predicted data has not been filtered. Assuming predictions do not need filtering.')
            self.filtered_predictions = self.predicted_data

        self.info['clustering_method'] = cluster_method

        self.learn_dynamics(cluster_method=cluster_method,
                            custom_labels=custom_labels,
                            kwargs=kwargs)
        self.classify_dynamics(proposals=proposals,
                               relabel_predictions=relabel_predictions,
                               k=k)
        if self.observed_data is not None:
            if self.filtered_obs is None:
               print('Observed data has not been filtered. Assuming observations do not need filtering.') 
               self.filtered_obs = self.observed_data
            self.classify_observations()

    def learn_dynamics(
        self,
        cluster_method='kmeans',
        custom_labels=None,
        kwargs={
            'n_clusters': 3,
            'n_init': 10}):
        """
        Learn dynamics.
        :param cluster_method: type of clustering to use ('kmeans' or 'spectral')
        :type cluster_method: str
        :param custom_labels: custom labels for predictions; if not None, no clustering is performed
        :type custom_labels: NoneType or :class:`numpy.ndarray`
        :param kwargs: keyword arguments for clustering method
        :type kwargs: dict
        :return: cluster labels and inertia (None if not kmeans)
        :rtype: :class:`numpy.ndarray`, float
        """
        if custom_labels is not None:
            self.cluster_labels = custom_labels
            inertia = None
        elif cluster_method == 'kmeans':
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
        self.info['num_clusters'] = self.num_clusters
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
        k_means.fit(self.filtered_predictions)
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

        clustering = SpectralClustering(**kwargs).fit(self.filtered_predictions)
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


        clustering = DBSCAN(**kwargs).fit(self.filtered_predictions)
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
        gmm.fit(self.filtered_predictions)
        return gmm.predict(self.filtered_predictions)

    def classify_dynamics(self,
                          proposals=({'kernel': 'linear'},
                                     {'kernel': 'rbf'},
                                     {'kernel': 'poly'},
                                     {'kernel': 'sigmoid'}),
                          relabel_predictions=True,
                          k=10):
        """
        Classify dynamics using best SVM method based on k-fold cross validation from a list of proposal keyword
        arguments for `sklearn.svm.LinearSVC`.
        https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
        :param proposals: list of proposal SVM keyword arguments
        :type proposals: list
        :param relabel_predictions: if custom_labels is not None, relabel_predictions controls if predictions are relabeled via SVM classification
        :type relabel_predictions: bool
        :param k: k for k-fold cross validation
        :type k: int
        :return: classifier object and labels of predictions
        :rtype: :class:`sklearn.svm.SVC`, :class:`numpy.ndarray`
        """

        if self.filtered_predictions is None:
            print('Predicted data has not been filtered. Assuming predictions do not need filtering.')
            self.filtered_predictions = self.predicted_data

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
        self.info['classifier_type'] = self.classifier
        self.info['classifier_kernel'] = proposals[ind_min]
        self.info['misclassification_rate'] = mis_min
        if relabel_predictions:
            self.predict_labels = self.classifier.predict(self.filtered_predictions)
        else:
            self.predict_labels = self.cluster_labels
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
        num_filtered = self.filtered_predictions.shape[0]
        inds = nrand.choice(num_filtered, num_filtered, replace=False)
        binsize = int(num_filtered / k)
        randomized_preds = self.filtered_predictions[inds, :]
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
            data=self.filtered_predictions,
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
            data = self.filtered_predictions
            labels = self.cluster_labels

        clf = svm.SVC(gamma='auto', **kwargs)
        clf.fit(data, labels)

        return clf

    def learn_qois(self,
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

        if self.filtered_predictions is None:
            print('Predicted data has not been filtered. Assuming predictions do not need filtering.')
            self.filtered_predictions = self.predicted_data

        if variance_rate is None and num_qoi is None:
            variance_rate = 0.99

        if self.num_clusters is None:
            # Set up single cluster if no clustering has been done
            print("No clustering performed. Assuming a single cluster.")
            self.num_clusters = 1
            self.predict_labels = np.array(
                self.filtered_predictions.shape[0] * [0])

        self.kpcas = []
        self.q_predict_kpcas = []
        self.num_pcs = []
        self.variance_rate = []
        self.Xpcas = []
        self.scalers = []
        self.info['kpca_kernel'] = []
        self.info['num_principal_components'] = []

        if variance_rate is not None:
            for i in range(self.num_clusters):
                scaler = StandardScaler(with_std=False)
                X_std = scaler.fit_transform(
                    self.filtered_predictions[np.where(self.predict_labels == i)[0], :])
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
                        eigs = kpca.eigenvalues_
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
                self.info['kpca_kernel'].append(proposals[ind_best])
                self.info['num_principal_components'].append(self.num_pcs[i])
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
                scaler = StandardScaler(with_std=False)
                predict_ptr = np.where(self.predict_labels == i)[0]
                X_std = scaler.fit_transform(
                    self.filtered_predictions[predict_ptr, :])
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
                        eigs = kpca.eigenvalues_
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
                self.info['kpca_kernel'].append(proposals[ind_best])
                self.info['num_principal_components'].append(self.num_pcs[i])
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

    def transform_predictions(self):
        """
        Transform predicted data to new QoIs.
        :return: transformed predictions
        :rtype: :class:`numpy.ndarray`
        """
        self.predict_maps = []
        for i in range(self.num_clusters):
            self.predict_maps.append(self.Xpcas[i][:, 0:self.num_pcs[i]])
        return self.predict_maps
    
    def set_observations(self,
                         observed_data):
        '''
        Adds observed data as an attribute.
        :param observed_data: observed data
        :type observed_data: :class:'numpy.ndarray'
        '''

        self.observed_data = observed_data

    def filter_observations(self,
                            observed_data_coordinates=None,
                            **kwargs):
        '''
        filter observed data either using same parameters when predictions were filtered or given parameters with kwargs
        :param observed_data_coordinates: coordinates at which observed data is collected
        :type observed_data_coordinates: :class:'numpy.ndarray'
        :return: filtered observed data
        :rtype: :class:'numpy.ndarray'
        '''

        if observed_data_coordinates is not None:
            self.observed_data_coordinates = observed_data_coordinates
        elif self.observed_data_coordinates is None:
            self.observed_data_coordinates = self.filtered_data_coordinates
        
        if self.info['pred_filtering_params'] is None:
            self.filter_data(filtered_data_coordinates=self.filtered_data_coordinates,
                             observed_data_coordinates=self.observed_data_coordinates,
                             filter_predictions=False, 
                             **kwargs)
        else:
            self.filter_data(filtered_data_coordinates=self.filtered_data_coordinates,
                             observed_data_coordinates=self.observed_data_coordinates,
                             filter_predictions=False, 
                             **self.info['pred_filtering_params'])
        
        return self.filtered_obs

    def classify_observations(self):
        """
        Classify observations into dynamics clusters.
        :return: cluster labels for observations
        :rtype: :class:`numpy.ndarray`
        """

        if self.filtered_predictions is None:
            print('Predicted data has not been filtered. Assuming predictions do not need filtering.')
            self.filtered_predictions = self.predicted_data
        if self.observed_data is None:
            print('No observed data given. Set observed_data or filtered_obs.')
        else:
            if self.filtered_obs is None:
                print('Observed data has not been filtered. Assuming observations do not need filtering.')
                self.filtered_obs = self.observed_data

            if self.num_clusters is None:
                self.num_clusters = 1
                self.obs_labels = np.array(self.filtered_obs.shape[0]*[0])
                self.obs_empty_cluster = [False]
            else:
                self.obs_labels = self.classifier.predict(self.filtered_obs)
                # Mark empty observation clusters
                self.obs_empty_cluster = []
                for i in range(self.num_clusters):
                    if len(np.where(self.obs_labels == i)[0]) == 0:
                        self.obs_empty_cluster.append(True)
                    else:
                        self.obs_empty_cluster.append(False)
            return self.obs_labels

    def transform_observations(self):
        """
        Transform observed data to new QoIs.
        :return: transformed observations
        :rtype: :class:`numpy.ndarray`
        """
        if self.obs_labels is None:
            print('Observations have not been classified. Classifying observations now.')
            self.classify_observations()
        self.obs_maps = []
        for i in range(self.num_clusters):
            if not self.obs_empty_cluster[i]:
                X_std = self.scalers[i].transform(
                    self.filtered_obs[np.where(self.obs_labels == i)[0], :])
                X_kpca = self.kpcas[i].transform(X_std)
                self.obs_maps.append(X_kpca[:, 0:self.num_pcs[i]])
            else:
                self.obs_maps.append(None)
        return self.obs_maps
    
    def classify_and_transform_observations(self):
        '''
        wrapper function for methods classify_observations and transform_observations
        :return: transformed observations
        :rtype: :class:'numpy.ndarray'
        '''

        if self.observed_data is None:
            print('No observed data given. Set observed_data or filtered_obs.')
        else:
            self.classify_observations()
            self.transform_observations()
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
        Learn Quantities of Interest and transform data.
        :param variance_rate: proportion of variance QoIs should capture.
        :type variance_rate: float
        :param proposals: proposal keyword arguments for kPCAs (a tuple of dictionaries)
        :type proposals: tuple
        :param num_qoi: number of quantities of interest to take (optional)
        :type num_qoi: int
        :return: transformed prediction and observation maps
        :rtype: :class:`numpy.ndarray`, :class:`numpy.ndarray`
        """
        self.learn_qois(
            variance_rate=variance_rate,
            proposals=proposals,
            num_qoi=num_qoi)
        self.transform_predictions()
        if self.observed_data is not None:
            if self.filtered_obs is None:
                print('Observed data has not been filtered. Assuming observations do not need filtering.')
                self.filtered_obs = self.observed_data
            self.transform_observations()
            return self.predict_maps, self.obs_maps
        else:
            return self.predict_maps
        
    def save_instance(self,
                      file_name,
                      file_path=''):
        '''
        pickles current instance of LUQ
        :param file_name: file name
        :type file_name: string
        :param file_path: path to file save location
        :type file_path: string
        '''
        import pickle
        f = file_path + file_name
        pf = open(f, 'wb')
        pickle.dump(self, pf)
        pf.close()
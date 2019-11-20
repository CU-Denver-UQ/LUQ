import numpy as np
import numpy.linalg as nlinalg
import scipy.stats as sstats
from scipy.stats import norm, beta
from scipy import optimize
from sklearn.preprocessing import StandardScaler

class LoQ(object):

    def __init__(self,
                 predicted_time_series,
                 observed_time_series,
                 times):

        self.predicted_time_series = predicted_time_series
        self.observed_time_series = observed_time_series
        self.times = times
        self.clean_times = None   # surrogate_times -> clean_times
        self.clean_predictions = None # surrogate -> clean
        self.clean_obs = None
        self.cluster_labels = None
        self.predict_labels = None
        self.kpcas = None
        self.q_predict_kpcas = None

        self.info = {'clustering_method': None,
                     'num_clusters': None,
                     'classifier_type': None,
                     'classifier_kernel': None,
                     'misclassification_rate': None,
                     'kpca_kernel': None,
                     'num_principal_components': None}

    # tol ---> Interpretation is to control the average error between splines normalized/relative to
    # the average magnitude of the data for which the splines are approximating a signal. 
    # default rel_tol = 1E-3? Or, set avg_tol to 0.01 and use l1-average errors which are interpreted
    # as a discretization of the average errors in the approximated signals
    def clean_data(self, time_start_idx, time_end_idx, num_clean_obs, tol, min_knots=3, max_knots=100):
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
                clean_predictions, error = self.clean_data_spline(times=times,
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
                print(idx, i, "knots being used.")
            self.clean_predictions[idx, :] = clean_predictions

        for idx in range(num_obs):
            i = min_knots
            while i <= max_knots:
                clean_obs, error = self.clean_data_spline(times=times,
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

        return clean_predictions, clean_predictions, clean_times

    def clean_data_spline(self, times, data, num_knots, clean_times):

        def wrapper_fit_func(x, N, *args):
            Qs = list(args[0][0:N])
            knots = list(args[0][N:])
            return piecewise_linear(x, knots, Qs)

        def piecewise_linear(x, knots, Qs):
            knots = np.insert(knots, 0, times[0])
            knots = np.append(knots, times[-1])
            return np.interp(x, knots, Qs)

        knots_init = np.linspace(times[0], self.times[-1], num_knots)[1:-1]

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

    # def dynamics that calls learn and classify and uses scores of the svm on 10-fold CV on batches
    # of labeled (i.e., learned) data split into training/test (90/10) sets. Then, use the clustering 
    # (with whatever kernel for spectral clustering or the kmeans) along with the svm (with whatever
    # kernel) gives the best score and then train the svm on all the data.
    #
    # For now, set num_clusters = 3 but keyword it so that user can change this without changing 
    # anything else. 
    
    def learn_dynamics(self, cluster_method='kmeans', kwargs={'n_clusters': 3, 'n_init': 10}):
        if cluster_method == 'kmeans':
            self.cluster_labels, inertia = self.learn_dynamics_kmeans(kwargs)
        elif cluster_method == 'spectral':
            self.cluster_labels = self.learn_dynamics_spectral(kwargs)
            inertia = None
        return self.cluster_labels, inertia

    def learn_dynamics_kmeans(self, kwargs):
        from sklearn.cluster import KMeans

        k_means = KMeans(init='k-means++', **kwargs)
        k_means.fit(self.clean_predictions)
        return k_means.labels_, k_means.inertia_

    def learn_dynamics_spectral(self, kwargs):
        from sklearn.cluster import SpectralClustering
        clustering = SpectralClustering(**kwargs).fit(self.clean_predictions)
        return clustering.labels_

    def classify_dynamics(self, proposals=[{'kernel': 'linear'},
                                           {'kernel': 'rbf'}, {'kernel': 'poly'}, {'kernel': 'sigmoid'}], k=10):
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
            #import pdb
            #pdb.set_trace()
        print(np.average(misclass_rates), 'misclassification rate for ', kwargs)
        clf = self.classify_dynamics_svm(kwargs=kwargs, data=self.clean_predictions, labels=self.cluster_labels)
        return clf, np.average(misclass_rates)

    def classify_dynamics_svm(self, kwargs={}, data=None, labels=None):
        from sklearn import svm
        if data is None:
            data = self.clean_predictions
            labels = self.cluster_labels

        clf = svm.SVC(gamma='auto', **kwargs)
        clf.fit(data, labels)
        return clf


    #
    # def classify_dynamics_kmeans(self, kwargs=kwargs, data=self.clean_predictions, labels=self.cluster_labels):
    #     # Warning: same number of clusters should be used as learning dynamics
    #     from sklearn.cluster import KMeans
    #     if data is None:
    #         data = self.clean_predictions
    #         labels = self.cluster_labels
    #     k_means = KMeans(init='k-means++', **kwargs)
    #     k_means.fit(data)



    # def qoi that loops over learn_qoi and classify to get both the predict and the observed QoI
    # needs comparison criteria/metrics - use proportion of variance explained by user-specified number 
    # of QoI
    
    def learn_qoi(self, kernel):
        from sklearn.decomposition import PCA, KernelPCA
        from sklearn.preprocessing import StandardScaler

        self.kpcas = []
        self.q_predict_kpcas = []
        num_clusters = np.max(self.predict_labels) + 1
        for i in range(num_clusters):
            scaler = StandardScaler()
            X_std = scaler.fit_transform(self.clean_predictions[np.where(self.predict_labels==i)[0], :])
            kpca = KernelPCA(kernel=kernel, fit_inverse_transform=False)
            X_kpca = kpca.fit_transform(X_std)
            self.kpcas.append(kpca)
            self.q_predict_kpcas.append(X_kpca)
        return self.kpcas

    def classify_observations(self, num_components=1):
        QoI_list = range(num_components)
        q_predict_maps = []
        # for i in range



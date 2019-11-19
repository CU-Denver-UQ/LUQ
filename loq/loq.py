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
        self.cluster_label = None
        self.predict_labels = None
        self.kpcas = None
        self.q_predict_kpcas = None

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
    
    def learn_dynamics(self, cluster_methods, kwargs):
        self.cluster_labels = []
        inertias = []
        for i, meth in enumerate(cluster_methods):
            if meth == 'kmeans':
                labels, inertia = self.learn_dynamics_kmeans(kwargs[i])
                self.cluster_labels.append(labels)
                inertias.append(inertia)
            elif meth == 'spectral':
                self.cluster_labels.append(self.learn_dynamics_spectral(kwargs[i]))
        self.cluster_label = self.cluster_labels[0]
        return self.cluster_labels, inertias

    def learn_dynamics_kmeans(self, kwargs):
        from sklearn.cluster import KMeans

        k_means = KMeans(init='k-means++', **kwargs)
        k_means.fit(self.clean_predictions)
        return k_means.labels_, k_means.inertia_

    def learn_dynamics_spectral(self, kwargs):
        from sklearn.cluster import SpectralClustering
        clustering = SpectralClustering(**kwargs).fit(self.clean_predictions)
        return clustering.labels_

    def classify_dynamics(self, kernel, kwargs={}):
        from sklearn import svm

        clf = svm.SVC(kernel=kernel, gamma='auto', **kwargs)
        clf.fit(self.clean_predictions, self.cluster_label)
        self.predict_labels = clf.predict(self.clean_predictions)
        missclass_rate = float(np.sum(np.not_equal(self.predict_labels, self.cluster_label))) / \
                         float(len(self.predict_labels))
        return self.predict_labels, missclass_rate

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



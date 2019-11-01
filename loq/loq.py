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
                 predict_times,
                 obs_times):

        self.predicted_time_series = predicted_time_series
        self.observed_time_series = observed_time_series
        self.predict_times = predict_times
        self.obs_times = obs_times
        # self.q_predict_pl = None
        # self.q_obs_pl = None
        self.surrogate_times = None
        self.surrogate_predictions = None
        self.surrogate_obs = None
        self.cluster_labels = None
        self.cluster_label = None
        self.predict_labels = None
        self.kpcas = None
        self.q_predict_kpcas = None

    def clean_data(self, time_start, time_end, num_time_obs, rel_tol, min_knots, max_knots):
        i = min_knots
        while i <= max_knots:
            surrogate_predictions, surrogate_obs, surrogate_times, l2_errors_predict, l2_errors_obs = \
                self.clean_data_spline(i, time_start, time_end, num_time_obs)
            if np.average(l2_errors_obs) <= rel_tol and np.average(l2_errors_predict) <= rel_tol:
                break
            else:
                i +=1
        if i > max_knots:
            print("Warning: maximum number of knots reached.")
        else:
            print(i, "knots being used.")
        self.surrogate_predictions = surrogate_predictions
        self.surrogate_obs = surrogate_obs
        self.surrogate_times = surrogate_times

    def clean_data_spline(self, num_knots, time_start, time_end, num_time_obs):
        def wrapper_fit_func(x, N, *args):
            Qs = list(args[0][0:N])
            knots = list(args[0][N:])
            return piecewise_linear(x, knots, Qs)

        def piecewise_linear(x, knots, Qs):
            knots = np.insert(knots, 0, time_start)
            knots = np.append(knots, time_end)
            ##import pdb
            #pdb.set_trace()
            return np.interp(x, knots, Qs)

        num_samples = self.predicted_time_series.shape[0]
        num_obs = self.observed_time_series.shape[0]
        knots_init = np.linspace(time_start, time_end, num_knots)[1:-1]

        # find piecewise linear splines for predictions
        q_predict_pl = np.zeros((num_samples, 2 * num_knots - 2))

        for i in range(num_samples):
            q_predict_pl[i, :], _ = optimize.curve_fit(lambda x, *params_0: wrapper_fit_func(x, num_knots, params_0),
                                                       self.predict_times, self.predicted_time_series[i, :],
                                                       p0=np.hstack([np.zeros(num_knots), knots_init]))
        surrogate_times = np.linspace(time_start, time_end, num_time_obs)

        # calculate l2 error between spline and original data
        l2_errors_predict = np.zeros((num_samples,))
        for i in range(num_samples):
            sur_at_original = piecewise_linear(self.predict_times,
                                                q_predict_pl[i, num_knots:2 * num_knots],
                                                q_predict_pl[i, 0:num_knots])
            l2_errors_predict[i] = (nlinalg.norm(sur_at_original -
                                                self.predicted_time_series[i, :], ord=2)/float(len(self.predict_times)))

        # evaluate spline at new times
        surrogate_predictions = np.zeros((num_samples, num_time_obs))

        for i in range(num_samples):
            surrogate_predictions[i, :] = piecewise_linear(surrogate_times,
                                                           q_predict_pl[i, num_knots:2 * num_knots],
                                                           q_predict_pl[i, 0:num_knots])
        # find piecewise linear splines for observations
        q_obs_pl = np.zeros((num_obs, 2 * num_knots - 2))

        for i in range(num_samples):
            q_obs_pl[i, :], _ = optimize.curve_fit(lambda x, *params_0: wrapper_fit_func(x, num_knots, params_0),
                                                   self.obs_times, self.observed_time_series[i, :],
                                                   p0=np.hstack([np.zeros(num_knots), knots_init]))

        # calculate l2 error between spline and original data
        l2_errors_obs = np.zeros((num_obs,))
        for i in range(num_obs):
            sur_at_original = piecewise_linear(self.obs_times,
                                               q_obs_pl[i, num_knots:2 * num_knots],
                                               q_obs_pl[i, 0:num_knots])
            l2_errors_obs[i] = (nlinalg.norm(sur_at_original -
                 self.observed_time_series[i, :], ord=2)/float(len(self.obs_times)))

        # evaluate spline at new times
        surrogate_obs = np.zeros((num_obs, num_time_obs))
        for i in range(num_obs):
            surrogate_obs[i, :] = piecewise_linear(surrogate_times,
                                                   q_obs_pl[i, num_knots:2 * num_knots],
                                                   q_obs_pl[i, 0:num_knots])
        return surrogate_predictions, surrogate_obs, surrogate_times, l2_errors_predict, l2_errors_obs

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
        k_means.fit(self.surrogate_predictions)
        return k_means.labels_, k_means.inertia_

    def learn_dynamics_spectral(self, kwargs):
        from sklearn.cluster import SpectralClustering
        clustering = SpectralClustering(**kwargs).fit(self.surrogate_predictions)
        return clustering.labels_

    def classify_dynamics(self, kernel, kwargs={}):
        from sklearn import svm

        clf = svm.SVC(kernel=kernel, gamma='auto', **kwargs)
        clf.fit(self.surrogate_predictions, self.cluster_label)
        self.predict_labels = clf.predict(self.surrogate_predictions)
        return self.predict_labels

    def learn_qoi(self, kernel):
        from sklearn.decomposition import PCA, KernelPCA
        from sklearn.preprocessing import StandardScaler

        self.kpcas = []
        self.q_predict_kpcas = []
        num_clusters = np.max(self.predict_labels) + 1
        for i in range(num_clusters):
            scaler = StandardScaler()
            X_std = scaler.fit_transform(self.surrogate_predictions[np.where(self.predict_labels==i)[0], :])
            kpca = KernelPCA(kernel=kernel, fit_inverse_transform=False)
            X_kpca = kpca.fit_transform(X_std)
            self.kpcas.append(kpca)
            self.q_predict_kpcas.append(X_kpca)
        return self.kpcas


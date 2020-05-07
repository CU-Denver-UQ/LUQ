# LUQ: Learning Uncertain Quantities
LUQ is a Python package that provides simple implementations of the algorithms for learning uncertain quantities.
The package provides a simple end-to-end workflow going from raw time series data to low-dimensional quantites of interest which can be used for data-consistent inversion.
LUQ utilizes several publicly available Python packages that are commonly used for scientific computing ([NumPy](https://numpy.org/) and [SciPy](https://www.scipy.org/)) and machine learning ([scikit-learn](https://scikit-learn.org/)).

## Installation
The current development branch of LUQ can be installed from GitHub,  using ``pip``:

    pip install git+https://github.com/CU-Denver-UQ/LUQ
    
Another option is to clone the repository and install LUQ using
``python setup.py install``

## Dependencies
LUQ is tested on Python 3.6 (but should work on most recent Python 3 versions) and depends on scikit-learn, NumPy, SciPy, and matplotlib (see ``requirements.txt`` for version information).

## License
[GNU Lesser General Public License (LGPL)](https://github.com/CU-Denver-UQ/LUQ/blob/master/LICENSE.txt)

## End-to-End Workflow
The workflow for using LUQ is designed to be as straightforward and simple as possible.
The three initial ingredients for using LUQ are the predicted time series, observed time series, and the data collection times, `predicted_time_series`, `observed_time_series`, and `times`, respectively.
These are used to instantiate the LUQ object:

    from luq import LUQ
    learn = LUQ(predicted_time_series, observed_time_series, times)
    
### Cleaning data (approximating dynamics)
Next, the data is cleaned.
We fit piecewise linear splines with both adaptive numbers of knots and adaptive knot placement to approximate underlying dynamical responses.
It is then possible to approximate the underlying dynamical response to arbitrary pointwise accuracy if both a sufficiently high frequency for collecting data and number of knots are used.
The splines are then evaluated at (possibly) new time values, resulting in "cleaned" data.

In LUQ, this is done by:

    learn.clean_data(time_start_idx=time_start_idx, time_end_idx=time_end_idx,
                     num_clean_obs=num_clean_obs, tol=tol, min_knots=min_knots, 
                     max_knots=max_knots)
                     
where `time_start_idx` is the index of the beginning of the time window, `time_end_idx` is the index of the end of the time window, `num_clean_obs` is the number of uniformly spaced clean observations to take, `tol`, `min_knots`, and `max_knots` are the tolerance, minimum, and maximum number of knots.

### Clustering and classifying data (learning and classifying dynamics)
Next, we learn and classify the dynamics.
The first goal is to use (cleaned) predicted data to classify the dynamical behavior of (cleaned) observed data.
In other words, we use the predicted data as a training set to learn the types of dynamical responses that may appear in the system.
This requires labeling the dynamics present in the predicted data set.
Clustering algorithms are a useful type of unsupervised learning algorithm that label data vectors using a metric to gauge the distance of a vector from the proposed "center" of the cluster.
In LUQ, clustering methods from scikit-learn with specified parameters are used to define a clustering of predicted data.

A classifier is a type of supervised learning algorithm that uses labeled training data to tune the various parameters so that non-labeled observed data can be properly labeled (i.e., classified).
At a high-level, classifiers usually partition the training data into two subsets of data: one used to tune the parameters and the other to test the quality of the tuned parameters by tracking the rate of misclassification.
This is referred to as cross-validation and is also useful in avoiding over-fitting the classifier (i.e., over-tuning the classifier parameters) to the entire set of training data.
In LUQ, after the data is clustered, a classifier is chosen which results in the lowest misclassification rate based on k-fold cross validation.

In LUQ this is implemented by:

    learn.dynamics(cluster_method='kmeans',
                   kwargs={'n_clusters': 3, 'n_init': 10},
                   proposals = ({'kernel': 'linear'}, {'kernel': 'rbf'}, {'kernel': 'poly'}, {'kernel': 'sigmoid'}),
                   k = 10)
where `cluster_method` defines the type of clustering algorithm to use, `kwargs` is a dictionary of arguments for the clustering algorithm, `proposals` is an array of dictionaries of proposed arguments for [``sklearn.svm.SVC``](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html), and `k` is the k for the k-fold cross-validation.

### Feature extraction (learning quantities of interest)
Finally, the best kernel PCAs are calculated for each cluster and the transformed predictions and observations are computed.
A number of desired quantities of interest for each cluster can be specified

    predict_map, obs_map = learn.learn_qois_and_transform(num_qoi=1,
                                                          proposals=({'kernel': 'linear'}, {'kernel': 'rbf'},
                                                                     {'kernel': 'sigmoid'}, {'kernel': 'cosine'}))
                                                                                                                        
or a proportion of variance explained by the minimum number of components can be specified for each cluster

    predict_map, obs_map = learn.learn_qois_and_transform(variance_rate=0.9,
                                                          proposals=({'kernel': 'linear'}, {'kernel': 'rbf'},
                                                                     {'kernel': 'sigmoid'}, {'kernel': 'cosine'}))
                                                                     
where `proposals` contains dictionaries of proposed [kernel parameters](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html#sklearn.decomposition.KernelPCA).

## Examples
Several documented examples can be found in the examples directory, including time series data from:
* [A Damped Harmonic Oscillator](https://github.com/CU-Denver-UQ/LUQ/blob/master/examples/harmonic-oscillator/harmonic_oscillator.py)
* ODE systems with Hopf Bifurcations ([Sel'kov Model](https://github.com/CU-Denver-UQ/LUQ/blob/master/examples/selkov/selkov.py) and [Lienard Equations](https://github.com/CU-Denver-UQ/LUQ/blob/master/examples/lienard/lienard.py))
* [Burgers Equation](https://github.com/CU-Denver-UQ/LUQ/blob/master/examples/shock/burgers_shock.py) resulting in shock solutions


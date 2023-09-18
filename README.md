# LUQ: Learning Uncertain Quantities

[![Build Status](https://travis-ci.org/CU-Denver-UQ/LUQ.svg?branch=master)](https://travis-ci.org/CU-Denver-UQ/LUQ) [![DOI](https://zenodo.org/badge/218807243.svg)](https://zenodo.org/badge/latestdoi/218807243)


LUQ is a Python package that provides simple implementations of the algorithms for learning uncertain quantities.
The package provides a simple end-to-end workflow going from raw time series data to low-dimensional quantites of interest which can be used for data-consistent inversion.
LUQ utilizes several publicly available Python packages that are commonly used for scientific computing ([NumPy](https://numpy.org/) and [SciPy](https://www.scipy.org/)) and machine learning ([scikit-learn](https://scikit-learn.org/)).

## Installation
The current development branch of LUQ can be installed from GitHub,  using ``pip``:

    pip install git+https://github.com/CU-Denver-UQ/LUQ
    
<!-- Another option is to clone the repository and install LUQ using
``python setup.py install`` -->

## Dependencies
LUQ is tested on Python 3.6 (but should work on most recent Python 3 versions) and depends on scikit-learn, NumPy, SciPy, and matplotlib (see ``requirements.txt`` for version information).

## License
[GNU Lesser General Public License (LGPL)](https://github.com/CU-Denver-UQ/LUQ/blob/master/LICENSE.txt)

## End-to-End Workflow
The workflow for using LUQ is designed to be as straightforward and simple as possible.
The two initial ingredients needed for using LUQ is the predicted data (predicted_data) and observed data (observed_data). However, the observed data is optional and can be added after instantiation using the set_observations method allowing observations to be used as available.

These are used to instantiate the LUQ object:

    from luq import LUQ
    learn = LUQ(predicted_data, observed_data)
    
### Filtering data (approximating dynamics)
Next, the data is filtered. This is done using either piecewise linear splines or with a weighted sum of Gaussians. The splines approach uses both adaptive numbers of knots and adaptive knot placement to approximate underlying dynamical responses allowing approximating the underlying dynamical response to arbitrary pointwise accuracy if both a sufficiently high frequency for collecting data and number of knots are used. The Gaussian function approach uses a similar adaptive approach with the number of Gaussians used and fits the Gaussian locations, shape, and weight. The Gaussian approach also allows other capabilities of removing polynomial trends prior to fitting and adding a polynomial of degree 1, 2, or 3 to the weighted sum of Gaussians to be fitted. With either approach, the learned functional response is evaluated at (possibly) new coordinates (filtered_data_coordinates) resulting in "filtered" data.

In LUQ, this is done as follows with the first example filtering the data using splines and the second example filtering the data using Gaussians and polynomials.

    learn.filter_data(filter_method='splines',
                         filtered_data_coordinates=filtered_data_coordinates,
                         predicted_data_coordinates=predicted_data_coordinates,
                         observed_data_coordinates=observed_data_coordinates,
                         tol=tol,
                         min_knots=min_knots,
                         max_knots=max_knots)
    
    learn.filter_data(filter_method='rbf',
                         filtered_data_coordinates=filtered_data_coordinates,
                         predicted_data_coordinates=predicted_data_coordinates,
                         observed_data_coordinates=observed_data_coordinates,
                         num_rbf_list=num_rbf_list,
                         remove_trend=remove_trend,
                         add_poly=add_poly,
                         poly_deg=poly_deg,
                         initializer=initializer)
                     
where 'filtered_data_coordinates' are the coordinates at which the fitted function is evaluated, 'predicted_data_coordinates' and 'observed_data_coordinates' are the predicted and observed data coordinates respectively, and the other parameters are particular to the fitting methods for fitting splines or Gaussians. Use help(luq.LUQ.filter_data_splines) or help(luq.LUQ.filter_data_rbfs) for paramter details.

### Clustering and classifying data (learning and classifying dynamics)
Next, we learn and classify the dynamics.
The first goal is to use (filtered) predicted data to classify the dynamical behavior of (filtered) observed data.
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
                                                          proposals=({'kernel':'linear'}, 
                                                                     {'kernel': 'rbf'},
                                                                     {'kernel': 'sigmoid'}, 
                                                                     {'kernel': 'cosine'}))
                                                                                                                        
or a proportion of variance explained by the minimum number of components can be specified for each cluster

    predict_map, obs_map = learn.learn_qois_and_transform(variance_rate=0.9,
                                                          proposals=({'kernel': 'linear'}, 
                                                                     {'kernel': 'rbf'},
                                                                     {'kernel': 'sigmoid'}, 
                                                                     {'kernel': 'cosine'}))
                                                                     
where `proposals` contains dictionaries of proposed [kernel parameters](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html#sklearn.decomposition.KernelPCA).

## Examples
Several documented examples can be found in the examples directory, including time series data from:
* [A Damped Harmonic Oscillator](https://github.com/CU-Denver-UQ/LUQ/blob/master/examples/harmonic-oscillator/harmonic_oscillator.py)
* ODE systems with Hopf Bifurcations ([Sel'kov Model](https://github.com/CU-Denver-UQ/LUQ/blob/master/examples/selkov/selkov.py) and [Lienard Equations](https://github.com/CU-Denver-UQ/LUQ/blob/master/examples/lienard/lienard.py))
* [Burgers Equation](https://github.com/CU-Denver-UQ/LUQ/blob/master/examples/shock/burgers_shock.py) resulting in shock solutions


# LUQ: Learning Uncertain Quantities
LUQ is a Python package that provides simple implementations of the algorithms for learning uncertain quantities.
LUQ utilizes several publicly available Python packages that are commonly used for scientific computing ([NumPy](https://numpy.org/) and [SciPy](https://www.scipy.org/)) and machine learning ([scikit-learn](https://scikit-learn.org/)).
The package provides a simple end-to-end workflow going from raw time series data to low-dimensional quantites of interest which can be used for data-consistent inversion.

## Installation
The current development branch of LUQ can be installed from GitHub,  using ``pip``:

    pip install git+https://github.com/CU-Denver-UQ/LUQ
    
Another option is to clone the repository and install LUQ using
``python setup.py install``

## Dependencies
LUQ is tested on Python 3.6 and depends on scikit-learn, NumPy, SciPy, and matplotlib (see ``requirements.txt`` for version information).

## License
[GNU Lesser General Public License (LGPL)](https://github.com/CU-Denver-UQ/LUQ/LICENSE.txt)

## End-to-End Workflow
The workflow for using LUQ is designed to be as straightforward and simple as possible.
The three initial ingredients for using LUQ are the predicted time series, observed time series, and the data collection times, `predicted_time_series`, `observed_time_series`, and `times`, respectively.
These are used to instantiate the LUQ object:

    from luq import LUQ
    learn = LUQ(predicted_time_series, observed_time_series, times)
    
### Cleaning data (approximating dynamics)
Next, the data is cleaned

    learn.clean_data(time_start_idx=time_start_idx, time_end_idx=time_end_idx,
                     num_clean_obs=num_clean_obs, tol=tol, min_knots=min_knots, 
                     max_knots=max_knots)
                     
where `time_start_idx` is the index of the beginning of the time window, `time_end_idx` is the index of the end of the time window, `num_clean_obs` is the number of uniformly spaced clean observations to take, `tol`, `min_knots`, and `max_knots` are the tolerance, minimum, and maximum number of knots.

### Clustering and classifying data (learning and classifying dynamics)
Next, we learn and classify the dynamics.

    learn.dynamics(cluster_method='kmeans',
                   kwargs={'n_clusters': 3, 'n_init': 10},
                   proposals = ({'kernel': 'linear'}, {'kernel': 'rbf'}, {'kernel': 'poly'}, {'kernel': 'sigmoid'}),
                   k = 10)
where `cluster_method` defines the type of clustering algorithm to use, `kwargs` is a dictionary of arguments for the clustering algorithm, `proposals` is an array of dictionaries of proposed arguments for ``sklearn.svm.SVC``, and `k` is the k for the k-fold cross-validation.

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
                                                                     
where `proposals` contains dictionaries of proposed kernel parameters.



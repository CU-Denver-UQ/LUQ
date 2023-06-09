import numpy as np
from scipy import optimize
from scipy.linalg import lstsq
from scipy.stats.qmc import Halton
from sklearn.cluster import KMeans

class RBFFit(object):
    def __init__(self, 
                 input_data, 
                 filtered_input_data, 
                 remove_trend=False, 
                 add_poly=False, 
                 poly_deg=None):
        '''
        Fitting a sum of Gaussian exponentials plus 
        a polynomial of chosen degree to noizy output data of
        an unknown real-valued function to extract predicted 
        values at specified input values.

        Inputs
        input_data: numpy.ndarray of shape (number_of_points) or (number_of_points,dimension) representing
            input values of function to be fitted
        filtered_input_data: numpy.ndarray of shape (number_of_desired_points) or (number_of_desired_points,dimension)
            representing the input values at which to evaluate the fitted function
        remove_trend: Boolean type with default of False controlling whether a polynomial trend should be removed prior to fitting.
            If False, the data is centered on the output data's mean.
        add_poly: Boolean type with default of False controlling whether a polynomial should be added to the sum of RBFs.
        poly_deg: None or int with default None specifying the degree of polynomial trend to be removed if remove_trend is True
            or the degree of polynomial to be added to sum of RBFs if add_poly is True.

        Attributes
        dim: dimension
        input_data: input data
        filtered_input_data: filtered input data
        input_min: minimum of each input variable
        input_max: maximum of each input variable
        input_data_scaled: input data scaled to be withint 0 and 1
        loc_initials: dictionary for initializations
        powers: list of exponents for each term in polynomials for remove_trend or add_poly  
        A_orig: global design matrix corresponding to scaled input data and powers
        A_filtered: global design matrix corresponding to scaled filtered input data and powers
        remove_trend: whether to remove polynomial trend prior to fitting
        add_poly: whether to add polynomial to sum of RBFs
        poly_deg: degree of polynomial for remove_trend of add_poly
        opt_fail_count_and_idx: list of number of failed optimization attempts along with index where failure
            occurred. This can be removed due to masking of output arrays.
        fit_errors: array containing errors for each fitted function

        Methods
        rbf: inputs: x, weights, sigmas, rs; evaluates sum of RBFs at x given parameter values for weights, scales, and locations. The weight parameters should be
            a numpy.ndarray of shape (num_rbfs), the scale parameters should be a numpy.ndarray of shape (num_rbfs,dim), and the 
            location parameters should be a numpy.ndarray of shape (num_rbfs,dim).
        construct_global_design: inputs: input, scale=False; constructs global design matrix for removing polynomial trends or adding a polynomial to the sum
            of RBFs for fitting. If input data is not scaled, then switch scale=True. 
        poly: inputs: x, coeffs; evaluates polynomial of degree poly_deg at x given coefficients coeffs. This is the polynomial that is added to sum of RBFs
            if add_poly is True.
        unpack_parameters: inputs: params, N; unpacks flattened array of parameters params that contain weights, scales, and locations given by the number of 
            RBFs N. params should consist of (weights.flatten(),sigmas.flatten(),rs.flatten()). Returns weights, sigmas, and rs of shapes (num_rbfs), (num_rbfs,dim) 
            and (num_rbfs,dim) respectively.
        
        '''
        # reshaping data to be (inputs, dim)
        if len(input_data.shape) == 1:
            self.dim = 1
            self.input_data = np.reshape(input_data, (input_data.shape[0],1))
            self.filtered_input_data = np.reshape(filtered_input_data, (filtered_input_data.shape[0],1))
        else:
            self.dim = input_data.shape[1]
            self.input_data = input_data
            self.filtered_input_data = filtered_input_data

        # saving minimum and maximum input values by dimension to appropriately scale and unscale input values
        input_min = []
        input_max = []
        for d in range(self.dim):
            input_min.append(self.input_data[:,d].min())
            input_max.append(self.input_data[:,d].max())
        self.input_min = input_min
        self.input_max = input_max

        # scaling input data by dimension
        input_data_scaled = np.zeros(self.input_data.shape)
        for d in range(self.dim):
            input_data_scaled[:,d] = (self.input_data[:,d] - self.input_min[d]) / (self.input_max[d] - self.input_min[d])
        self.input_data_scaled = input_data_scaled

        # setting up dictionary for initializations of Gaussian locations so that initializations can be reused
        self.loc_initials = {'uniform_random': [], 
                            'Halton': [], 
                            'kmeans': []}

        # setting up self.powers attribute to be used in removing polynomial trends and/or adding a polynomial to sum of Gaussians for fitting then 
        # constructing corresponding matrices
        if remove_trend or add_poly:
            powers = []
            if poly_deg is None:
                poly_deg = 1
            for i in range(poly_deg+1):
                if self.dim == 1:
                    if i in range(1,poly_deg+1):
                        powers.append([i])
                else:
                    for j in range(poly_deg+1):
                        if self.dim == 2:
                            if i+j in range(1,poly_deg+1):
                                powers.append([i,j])
                        else:
                            for k in range(poly_deg+1):
                                if self.dim == 3:
                                    if i+j+k in range(1,poly_deg+1):
                                        powers.append([i,j,k])
                                else: 
                                    print('Dimension must be <= 3.')
                                    remove_trend = False  
                                    add_poly = False
            self.powers = powers                       
            self.A_orig = self.construct_global_design_matrix(self.input_data_scaled)
            if remove_trend:
                self.A_filtered = self.construct_global_design_matrix(self.filtered_input_data, scale=True)

        # setting attributes from input parameters
        self.remove_trend = remove_trend
        self.add_poly = add_poly
        self.deg = poly_deg 

    def rbf(self, 
            x, 
            weights, 
            sigmas, 
            rs):
        '''
        Calculates sum of Gaussians at x as weights[i]*exp(-sum((x-rs[i])^2/sigmas[i])).
        '''
        # x.shape = (data, dim)
        # weights.shape = (N,) 
        # sigmas.shape = (N, dim)
        # rs.shape = (N, dim)
        output = []
        for i in range(rs.shape[0]):
            output.append(weights[i]*np.exp(-np.dot((x-rs[i,:])**2,1/sigmas[i,:])))
        return np.sum(output, axis=0)

    def construct_global_design_matrix(self, 
                                       input, 
                                       scale=False):
        '''
        calculates matrix of columns corresponding to input variables raised to powers given by self.powers.
        '''
        # scaling input data if unscaled
        if scale:
            input_scaled = np.zeros(shape=input.shape)
            for d in range(self.dim):
                input_scaled[:,d] = (input[:,d] - self.input_min[d]) / (self.input_max[d] - self.input_min[d])
        else:
            input_scaled = input

        # constructing matrix 
        A = np.ones(shape=(input_scaled.shape[0],1+len(self.powers)))
        for p in range(len(self.powers)):
            for d in range(self.dim):
                A[:,1+p] *= input_scaled[:,d]**self.powers[p][d]
        return A

    def poly(self, 
             x, 
             coeffs):
        '''
        evaluates polynomial at x with input variables raised to powers given by self.powers
        '''
        A = self.construct_global_design_matrix(x)
        return np.matmul(A, coeffs)

    def unpack_parameters(self, 
                          params, 
                          N):
        '''
        takes flattened list of params of form (weights.flatten(),sigmas.flatten(),rs.flatten()) a returns weights, sigmas, and rs of shapes N, (N,dim) 
        and (N,dim) respectively; N represents number of RBFs used. Flattening parameters is necessary for optimization package used.
        '''
        # initializing weights, sigmas, and rs
        weights = params[0:N]
        sigmas = np.zeros((N, self.dim))
        rs = np.zeros((N, self.dim))

        # constructing and returning weights, sigmas, and rs
        j = -1
        for k in range(self.dim*N):
            i = k % N
            if i == 0:
                j += 1
            sigmas[i,j] = params[N + k]
            rs[i,j] = params[N + self.dim*N + k]
        if self.add_poly:
            coeffs = params[-(1+len(self.powers)):]
            return weights, sigmas, rs, coeffs
        else:
            return weights, sigmas, rs

    def unscale_parameters(self, 
                           sigmas, 
                           rs, 
                           coeffs=None):
        '''
        unscales sigmas and rs according taccording to how the input data was scaledo how the input data was scaled
        '''
        # unscaling sigmas and rs by dimension
        for d in range(self.dim):
            sigmas[:,d] *= (self.input_max[d] - self.input_min[d])**2
            rs[:,d] *= (self.input_max[d] - self.input_min[d])
            rs[:,d] += self.input_min[d]
        
        # unscaling polynomial coefficients of polynomial added to sum of Gaussians
        if coeffs is not None:
            for p in range(len(self.powers)):
                for d in range(self.dim):
                    coeffs[1+p] /= (self.input_max[d] - self.input_min[d])**self.powers[p][d]
            return sigmas, rs, coeffs
        else:
            return sigmas, rs

    def wrapper_fit_func(self, 
                         xdata, 
                         N, 
                         *args):
        '''
        function inputted into optimization package scipy.optimize.curve_fit
        '''
        # xdata.shape = (data, dim)
        # N = num_rbfs
        # args[0] = [weights, sigmas[:,0], sigmas[:,1], ..., rs[:,0], rs[:,1], ...]
        # len(args[0]) = N + dim*N + dim*N
        if self.add_poly:
            weights, sigmas, rs, coeffs = self.unpack_parameters(args[0], N)
            return self.rbf(xdata, weights, sigmas, rs) + self.poly(xdata, coeffs)
        else:
            weights, sigmas, rs = self.unpack_parameters(args[0], N)
            return self.rbf(xdata, weights, sigmas, rs)

    def initialize_loc(self, 
                       current_initializer, 
                       num_rbfs,
                       num_rbf_list_idx):
        '''
        calculates and returns initialization of locations of Gaussians using either uniform random sampling, Halton sampling, or k-means or returns previously calculated initialization
        '''
        if current_initializer == 'uniform_random':
            if self.loc_initials[current_initializer][num_rbf_list_idx] is None:
                output = np.random.uniform(0, 1, self.dim*num_rbfs)
                self.loc_initials[current_initializer][num_rbf_list_idx] = output
                return output
            else:
                return self.loc_initials[current_initializer][num_rbf_list_idx]
        elif current_initializer == 'Halton':
            if self.loc_initials[current_initializer][num_rbf_list_idx] is None:
                Halton_sampler = Halton(d=self.dim)
                output = Halton_sampler.random(n=num_rbfs).T.flatten()
                self.loc_initials[current_initializer][num_rbf_list_idx] = output
                return output
            else: 
                return self.loc_initials[current_initializer][num_rbf_list_idx]
        elif current_initializer == 'kmeans':
            if self.loc_initials[current_initializer][num_rbf_list_idx] is None:
                kmeans = KMeans(n_clusters=num_rbfs).fit(X=self.input_data_scaled)
                output = kmeans.cluster_centers_.T.flatten()
                self.loc_initials[current_initializer][num_rbf_list_idx] = output
                return output
            else: 
                return self.loc_initials[current_initializer][num_rbf_list_idx]
        else:
            print(f'Choose either uniform_random, Halton, or kmeans for location initialization.')
            print(f'Proceeding using k-means.')
            if self.loc_initials['kmeans'][num_rbf_list_idx] is None:
                kmeans = KMeans(n_clusters=num_rbfs).fit(X=self.input_data_scaled)
                output = kmeans.cluster_centers_.T.flatten()
                self.loc_initials['kmeans'][num_rbf_list_idx] = output
                return output
            else: 
                return self.loc_initials['kmeans'][num_rbf_list_idx]

    def initialize_params(self, 
                          current_initializer, 
                          num_rbfs, 
                          num_rbf_list_idx, 
                          output_data_abs_max, 
                          ps_poly=None):
        '''
        initializing weights, sigmas, and rs for sum of Gaussians and coefficients of added polynomial if self.add_poly=True
        '''
        # initialize weights
        p0 = num_rbfs * [output_data_abs_max]
        # initialize sigmas
        p0.extend(np.ones(self.dim*num_rbfs) / 2 / num_rbfs)
        # initialize rs
        p0.extend(self.initialize_loc(current_initializer, num_rbfs, num_rbf_list_idx))
        if self.add_poly:
            p0.extend(ps_poly)
        return p0

    def construct_param_bounds(self, 
                               num_rbfs, 
                               output_data_abs_max):
        '''
        construction parameter bounds for optimization. weights are bounded by the absolute value of data max, sigmas below by 1e-6, 
        rs below by 0 and above by 1 (input data is scaled to be between 0 and 1). If self.add_poly=True, coefficients of added polynomial
        are unbounded
        '''
        # initializing bounds of appropriate size
        if self.add_poly:
            param_bounds = np.zeros((2, num_rbfs + 2*self.dim*num_rbfs + 1 + len(self.powers)))
            # adding bounds for polynomial coefficients
            for i in range(1+len(self.powers)):
                param_bounds[:,-(i+1)] = [-np.inf, np.inf]
        else:
            param_bounds = np.zeros((2, num_rbfs + 2*self.dim*num_rbfs))

        for i in range(param_bounds.shape[1]):
            # bounds for weights
            if i < num_rbfs:
                param_bounds[:,i] = [-output_data_abs_max, output_data_abs_max]
            # bounds for sigmas
            elif i < num_rbfs + self.dim*num_rbfs:
                param_bounds[:,i] = [1e-6, np.inf]
            # bounds for rs
            elif i < num_rbfs + 2*self.dim*num_rbfs:
                param_bounds[:,i] = [0, 1]
        return param_bounds

    def remove_trend_func(self, 
                          output_data):
        '''
        function for removing polynomial trend
        '''
        ps, _, _, _ = lstsq(self.A_orig, output_data)
        res = output_data - np.matmul(self.A_orig, ps)
        return ps, res

    def fit_parameters(self, 
                       output_data, 
                       output_data_shifted, 
                       output_data_abs_max, 
                       num_rbfs, 
                       param_bounds, 
                       initializer, 
                       num_rbf_list_idx, 
                       ps_trend=None, 
                       ps_poly=None, 
                       max_nfev=None):
        '''
        fit parameters using scipy.optimize.curve_fit
        '''
        # try-except for scipy.optimize.curve_fit RuntimeError for optimization failure or for scipy.optimize.least_squares LinAlgError
        try:
            # initiliaze parameters
            p0 = self.initialize_params(initializer, 
                                        num_rbfs, 
                                        num_rbf_list_idx,
                                        output_data_abs_max, 
                                        ps_poly=ps_poly)

            # fit parameters
            fit, _ = optimize.curve_fit(lambda x, *params_0 : self.wrapper_fit_func(x.T, num_rbfs, params_0),
                                    self.input_data_scaled.T,
                                    output_data_shifted,
                                    p0=p0,
                                    bounds=param_bounds,
                                    max_nfev=max_nfev)

            # unpacking fitted parameters
            if self.add_poly:
                weights, sigmas_scaled, rs_scaled, coeffs_scaled = self.unpack_parameters(fit, num_rbfs)
                sigmas, rs, coeffs = self.unscale_parameters(sigmas_scaled, rs_scaled, coeffs_scaled)
                params = (weights, sigmas, rs, coeffs)
                filtered_data_at_original = self.rbf(self.input_data, weights, sigmas, rs) + self.poly(self.input_data, coeffs)
            else:
                weights, sigmas_scaled, rs_scaled = self.unpack_parameters(fit, num_rbfs)
                sigmas, rs = self.unscale_parameters(sigmas_scaled, rs_scaled)
                params = (weights, sigmas, rs)
                filtered_data_at_original = self.rbf(self.input_data, weights, sigmas, rs) 

            # adding polynomial trend back
            if self.remove_trend:
                filtered_data_at_original += np.matmul(self.A_orig, ps_trend)
            else:
                filtered_data_at_original += np.mean(output_data)

            # computing relative error
            error = np.average(np.abs(filtered_data_at_original - output_data))
            error = error / np.average(np.abs(output_data))

            return (error,) + params

        except RuntimeError or LinAlgError:
            return None
        
    def opt_attempt(self, 
                    num_rbfs, 
                    output_data, 
                    output_data_shifted, 
                    ps_trend, 
                    ps_poly, 
                    n,
                    max_opt_count, 
                    initializer, 
                    num_rbf_list_idx,
                    reshape_filtered_data): 
        '''
        function wrapping optimization attempts in self.clean_data for each initializer specified or all initializers if optimization with
        specified initializer fails and repeats up to max_opt_count number of times

        both n and reshape_filtered_data are only used for print statements that are currently commented out. n is used for print statements about optimization 
        loop updates if filtering multiple samples at once; reshape_filtered_data used to distinguish between print statements with n and without (only a single 
        sample being filtered at one time means n is always 0)
        '''
        # set up parameter list; if lists have length >1 after sucessful optimization, then parameters corresponding to minimum error are used
        errors = []
        weights_ = []
        sigmas_ = []
        rs_ = []
        coeffs_ = []

        # initializing controls for number of optimization attempts; opt_count is for checking if the maximum number of optimization attempts
        # have been made, opt_attempt_loop_control is to exit while loop if optimization succeeds or if maximum number of optimization attempts has been
        # made, max_nfev is input parameter for scipy.optimize.least_squares controlling maximum number of function evaluations (default value
        # is 100 * number of parameters when max_nfev=None)
        opt_count = 0
        opt_attempt_loop_control = True
        max_nfev = 100 * num_rbfs

        # set up parameter bounds
        param_bounds = self.construct_param_bounds(num_rbfs, np.abs(output_data_shifted).max())

        while opt_attempt_loop_control:
            # setting temporary initializer for redefining initializer if optimization takes more than one attempt
            if initializer == 'all':
                initializer_tmp = 'all'

            if initializer != 'all':
                # attempt optimization with specified initialization
                fit_output = self.fit_parameters(output_data, 
                                                    output_data_shifted, 
                                                    np.abs(output_data_shifted).max(), 
                                                    num_rbfs, 
                                                    param_bounds, 
                                                    initializer, 
                                                    num_rbf_list_idx,
                                                    ps_trend, 
                                                    ps_poly,
                                                    max_nfev)

                # fit_output is None only if optimization failed
                if fit_output is not None:
                    # extracting fitted parameters
                    if self.add_poly:
                        error, weights, sigmas, rs, coeffs = fit_output
                        coeffs_.append(coeffs)
                    else:
                        error, weights, sigmas, rs = fit_output
                    errors.append(error)
                    weights_.append(weights)
                    sigmas_.append(sigmas)
                    rs_.append(rs)
                else:
                    # reseting initialization to 'all' and removing any previously calculated initializations due to optimization failure; storing original initializer in
                    # initializer_tmp for next sample iteration
                    self.loc_initials[initializer][num_rbf_list_idx] = None
                    initializer_tmp = initializer
                    initializer = 'all'

            if initializer == 'all':
                # loop through different initializations
                for initializer_loop in ['uniform_random', 'Halton', 'kmeans']:
                    fit_output = self.fit_parameters(output_data, 
                                                    output_data_shifted, 
                                                    np.abs(output_data_shifted).max(), 
                                                    num_rbfs, 
                                                    param_bounds, 
                                                    initializer_loop, 
                                                    num_rbf_list_idx,
                                                    ps_trend, 
                                                    ps_poly,
                                                    max_nfev)

                    # fit_output is None only if optimization failed
                    if fit_output is not None:
                        # extracting fitted parameters
                        if self.add_poly:
                            error, weights, sigmas, rs, coeffs = fit_output
                            coeffs_.append(coeffs)
                        else:
                            error, weights, sigmas, rs = fit_output  
                        errors.append(error)
                        weights_.append(weights)
                        sigmas_.append(sigmas)
                        rs_.append(rs)
                    else:
                        # removing any previously calculated initializations due to optimization failure
                        self.loc_initials[initializer_loop][num_rbf_list_idx] = None
            
            # if statement in case optimization failed in current loop attempt
            if len(errors) == 0:
                # after first failed attempt, max_nfev set to None
                if opt_count == 0:
                    max_nfev = None

                # update optimization attempt count 
                opt_count += 1

                # if statement checking if maximum number of optimization attempts reached
                if opt_count+1 == max_opt_count:
                    # exiting optimization attempt loop 
                    opt_attempt_loop_control = False

                    # reseting initializer to original 
                    initializer = initializer_tmp

                    # print(f'Optimization failed at sample {n} using {num_rbfs} RBFs.')

                    # setting error to infinity and parameters to None
                    error = np.inf
                    weights = None
                    sigmas = None
                    rs = None
                    if self.add_poly:
                        coeffs = None
                # else:
                #     # printing appropriate statement for optimization retry
                #     if opt_count == 1:
                #         if reshape_filtered_data:
                #             print(f'Optimization failed using {num_rbfs} RBFs. Removing restrictions on function evaluation count.')
                #         else:
                #             print(f'Optimization failed at sample {n} using {num_rbfs} RBFs. Removing restrictions on function evaluation count.')
                #     else:
                #         if reshape_clean_data:
                #             print(f'Optimization failed using {num_rbfs} RBFs. Retrying ....')
                #         else:
                #             print(f'Optimization failed at sample {n} using {num_rbfs} RBFs. Retrying ....')
            else:
                # exiting optimization attempt loop due to successful optimization
                opt_attempt_loop_control = False

                # if reshape_filtered_data:
                #     print(f'Successful optimization using {num_rbfs} RBFs.')
                # else:
                #     print(f'Successful optimization at sample {n} using {num_rbfs} RBFs.')

                # choosing new fitted parameters corresponding to minumum error
                errors = np.array(errors)
                idx = np.where(errors == errors.min())[0][0]
                error = errors[idx]
                weights = weights_[idx]
                sigmas = sigmas_[idx]
                rs = rs_[idx]
                if self.add_poly:
                    coeffs = coeffs_[idx]

        # params for return
        params = (weights, sigmas, rs)
        if self.add_poly:
            params += (coeffs, )
        
        return error, params


    def filter_data(self, 
                    output_data_samples, 
                    num_rbf_list, 
                    initializer='Halton', 
                    max_opt_count=3, 
                    tol=1e-2):
        '''
        loop for fitting parameters and returning filtered fitted data at filtered_input_data
        '''
        # if num_rbf_list is not a list or range, it is assumed to be an int meaning only a specific number of rbfs is fitted
        if not (isinstance(num_rbf_list, list) or isinstance(num_rbf_list, range)):
            num_rbf_list = [num_rbf_list]
        # if num_rbf_list is a list, making sure it's sorted low to high
        elif isinstance(num_rbf_list, list):
            num_rbf_list.sort()

        # reshaping data to (num_samples,num_data_points)
        if len(output_data_samples.shape) == 1:
            output_data_samples = np.reshape(output_data_samples, newshape=(1,output_data_samples.shape[0]))
            reshape_filtered_data = True
        else: 
            reshape_filtered_data = False
            
        # tracking count and index where optimization fails
        self.opt_fail_count_and_idx = [0, []]
        
        # initializing errors list
        fit_errors = []

        # initializing output data
        filtered_data_samples = np.zeros(shape=(output_data_samples.shape[0],self.filtered_input_data.shape[0]))
        
        # output_data_samples.shape = (samples, data)
        for n in range(output_data_samples.shape[0]):
            # extracting data for current loop iteration
            output_data = output_data_samples[n,:]

            # shifting data
            if self.remove_trend:
                ps_trend, output_data_shifted = self.remove_trend_func(output_data)
            else:
                ps_trend = None
                output_data_shifted = output_data - np.mean(output_data)

            # initializing coefficients for added polynomial
            if self.add_poly:
                ps_poly, _, _, _ = lstsq(self.A_orig, output_data_shifted)
            else:
                ps_poly = None

            # fitting data for first num_rbfs in list
            num_rbfs = num_rbf_list[0]
            num_rbf_list_idx = 0
            for initializer_loop in ['uniform_random', 'Halton', 'kmeans']:
                self.loc_initials[initializer_loop].append(None)
            error, params = self.opt_attempt(num_rbfs,
                                             output_data, 
                                             output_data_shifted, 
                                             ps_trend,
                                             ps_poly,
                                             n, 
                                             max_opt_count,
                                             initializer,
                                             num_rbf_list_idx,
                                             reshape_filtered_data)
            
            weights, sigmas, rs = params[:3]
            if self.add_poly:
                coeffs = params[-1]

            # loop through number of rbfs to fit
            for num_rbfs in num_rbf_list[1:]:
                num_rbf_list_idx += 1
                for initializer_loop in ['uniform_random', 'Halton', 'kmeans']:
                    self.loc_initials[initializer_loop].append(None)
                error_new, params_new = self.opt_attempt(num_rbfs,
                                                         output_data,
                                                         output_data_shifted,
                                                         ps_trend,
                                                         ps_poly,
                                                         n,
                                                         max_opt_count,
                                                         initializer,
                                                         num_rbf_list_idx,
                                                         reshape_filtered_data)
                
                weights_new, sigmas_new, rs_new = params_new[:3]
                if self.add_poly:
                    coeffs_new = params_new[-1]

                # errors are infinite if optimization failed
                if error_new < np.inf:
                    if error < np.inf:
                        # storing error difference since error might be overwritten in next block
                        error_diff = np.abs(error_new - error)

                        # updating error and parameters
                        if error_new < error:
                                    error = error_new
                                    weights, sigmas, rs = weights_new, sigmas_new, rs_new
                                    if self.add_poly:
                                        coeffs = coeffs_new
                        
                        # breaking loop if error difference is less than tolerance and kept error is less than 0.50
                        if error_diff < tol and error < tol:
                            break

            # checking if optimization failed at every iteration through num_rbf_list
            if error < np.inf:  
                if reshape_filtered_data:
                    print(f'Data fitted using {len(weights)} RBFs with a relative error of {error}.')
                else:
                    print(f'Data at sample {n} fitted using {len(weights)} RBFs with a relative error of {error}.')

                # creating filtered data at filtered_input_data 
                if self.add_poly:
                    filtered_data = self.rbf(self.filtered_input_data, weights, sigmas, rs) + self.poly(self.filtered_input_data, coeffs)
                else:
                    filtered_data = self.rbf(self.filtered_input_data, weights, sigmas, rs)

                # unshifting filtered_data according to how output_data was shifted
                if self.remove_trend:
                    filtered_data += np.matmul(self.A_filtered, ps_trend)
                else:
                    filtered_data += np.mean(output_data)
                
                filtered_data_samples[n,:] = filtered_data
            else:
                if reshape_filtered_data:
                    print('Optimization failed.')
                else:
                    print(f'Optimization failed at sample {n}')

                # updating attribute counting number of optimization failures and index of failure
                self.opt_fail_count_and_idx[0] += 1
                self.opt_fail_count_and_idx[1].append(n)

            # updating list of errors for samples
            fit_errors.append(error)
            
        # reshaping if number of samples was 1
        if reshape_filtered_data:
            filtered_data_samples = filtered_data_samples[0,:]

        # updating fitted data error list
        self.fit_errors = fit_errors

        return filtered_data_samples
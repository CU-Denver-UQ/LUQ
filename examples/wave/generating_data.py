import numpy as np
from scipy.stats import norm, beta
from scipy.stats import gaussian_kde as GKDE
from scipy.integrate import quadrature
import matplotlib.pyplot as plt

# colorblind friendly color palette
c = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

# setup fontsizes for plots
plt_params = {'legend.fontsize': 14,
          'figure.figsize': (6.4, 4.8),
         'axes.labelsize': 16,
         'axes.titlesize': 16,
         'xtick.labelsize': 14,
         'ytick.labelsize': 14}
plt.rcParams.update(plt_params)


# finite-difference scheme

# defining model solve function

dx = 0.05
dy = 0.05
dt = 0.005 # satifies CFL condition

xn = np.linspace(0,5.0,101) # 101 = length in x / dx
ym = np.linspace(0,5.0,101)
tk = np.linspace(0,7.0,1401) # 1401 = length in t / dt

# defining model solve on 101x101 uniform mesh of [0,5]^2 for t = 0 to t = 7 with dt = 0.005
def M(a,b):
    # initializing the model solution
    # using Dirichlet boundary conditions,so initializing with zeros means boundary values are set
    u = np.zeros((101,101,1401))
    
    # iterate through times; t here is equivalent to time and time index
    for t in range(1401):
        
        # if t = 0, use initial condition modeling wave droplet
        if t == 0:
            mesh = np.meshgrid(xn[1:-1],ym[1:-1])
            u[1:-1,1:-1,t] = 0.2*np.exp(-10*((mesh[0].T-a)**2+(mesh[1].T-b)**2))
        
        # else solve model using finite-difference scheme
        else:
            u[1:-1,1:-1,t] = 2 * u[1:-1,1:-1,t-1] - u[1:-1,1:-1,max(0,t-2)] \
                + dt**2 / dx**2 * (u[2:,1:-1,t-1] - 2 * u[1:-1,1:-1,t-1] + u[:-2,1:-1,t-1]) \
                + dt**2 / dy**2 * (u[1:-1,2:,t-1] - 2 * u[1:-1,1:-1,t-1] + u[1:-1,:-2,t-1])
    return u

# indexing for extracting data on different grid sizes

# indexing function for flattening data
def idx_at(x,y):
    idx = []
    idx.append((x / dx).astype(int))
    idx.append((y / dy).astype(int))
    return idx

# using indexing function to extract data on uniformly-spaced mesh given by delta
def create_idx(delta):
    N = (5-delta)/delta 
    # note: only delta such that (5-delta)/delta is int can be used (or does not change value when cast as int) 
    # any other delta value requires extrapolation
    pts = np.linspace(delta,5-delta,int(N))
    grid_pts = np.meshgrid(pts,pts)
    idx = idx_at(grid_pts[0],grid_pts[1])
    return [idx[0].flatten(), idx[1].flatten()]



# Constructing predicted samples for different meshes

np.random.seed(123456)

# parameter samples of pi_init
num_samples = int(1E3)
params = np.random.uniform(low=0.0,high=5.0,size=(2,num_samples))
params.dump('data/params')

# solving model on full grid
full_grid = np.zeros((num_samples,101,101,14))
for i in range(num_samples):
    tmp = M(params[0,i],params[1,i])
    full_grid[i,:,:,:] = tmp[:,:,100::100]
    print(f'sample {i} done')

# extracting model solve on coarser grids
for i in range(20):
    delta = 0.05 * (i+1)
    N = (5-delta)/delta
    
    # check if grid is subset of full grid
    if N != int(N):
        print(f'delta={delta} does not coincide with mesh')
    
    # extracting data
    else:
        print(f'Extracting data on {int(N)}x{int(N)} grid')
        idx = create_idx(delta)
        pred = np.zeros((num_samples,int(N)**2,14))
        for i in range(num_samples):
            pred[i,:,:] = full_grid[i,idx[0],idx[1],:]
        fn = 'data/pred_' + str(int(N)) + 'x' + str(int(N))
        pred.dump(fn)


# calcuting TV metrics between representative dg samples KDE and exact dg distributions

# creating different sets of samples along with KDE's and exact DG densities, joints and marginals
num_obs_samples = 200
kde_dg = []
kde_param_marginals = []

np.random.seed(1234)
params_full = np.random.uniform(0.0, 5.0, (2,int(1E4))) # large number of uniform samples of parameter space 

exact_param_marginals = [lambda x : beta.pdf((x-1)/2,2,5)/2, # exact parameter marginals for a and b
                         lambda x : norm.pdf(x,2.5,0.5)]
exact_dg = lambda x, y : exact_param_marginals[0](x)*exact_param_marginals[1](y) # exact joint data-generating density
exact_dg = exact_dg(params_full[0,:],params_full[1,:]) # evaluate at samples for computing TV
for i in range(int(1E3)):
    params_obs = np.vstack([2 * np.random.beta(a=2, b=5, size=num_obs_samples) + 1, # temporary set of 200 observed samples
                             np.random.normal(loc=2.5, scale=0.5, size=num_obs_samples)])
    kde_dg.append(GKDE(params_obs)(params_full)) # KDE's of joints

    # KDEs of true marginals
    kde_param_marginals.append([])
    for j in range(params_obs.shape[0]):
            kde_param_marginals[i].append(GKDE(params_obs[j,:]))

# calculating TV metric between updated and exact joint distributions
TVs = []
for i in range(len(kde_dg)):
    # computing TV for each set of samples using MC integration
    TV = np.abs(kde_dg[i]-exact_dg)/2
    TVs.append(np.mean(TV)*25)

# plt.figure()
# plt.hist(TVs)
# plt.xlabel('TV')
# plt.ylabel('count')
# plt.title('TV values between KDE and exact DG joint densities')
# plt.show()

# analyzing distribution of TV's between KDE and exact of DG marginals
marginal_TVs = []
for j in range(1000):
    marginal_TVs.append([])
    for i in range(params_full.shape[0]):
        # computing TV between KDE and exact DG marginal densities directly
        diff = lambda x : np.abs(kde_param_marginals[j][i](x)-exact_param_marginals[i](x))
        TV, _ = quadrature(diff, 0.0, 5.0, tol=1e-2)
        marginal_TVs[j].append(TV/2)

# # plotting histogram
# marginal_TVs = np.array(marginal_TVs)
# param_labels = [r'$a$', r'$b$']
# for i in range(2):
#     plt.figure()
#     plt.hist(marginal_TVs[:,i])
#     plt.xlabel('TV')
#     plt.ylabel('count')
#     plt.show()
#     plt.title('TV values between KDE and exact DG marginal densities for '+param_labels[i])
#     plt.show()


# constructing data-generating samples used
    
# parameter samples for construction of pi_obs

np.random.seed(12345678)
params_obs = np.vstack([2 * np.random.beta(a=2, b=5, size=num_obs_samples) + 1,
                         np.random.normal(loc=2.5, scale=0.5, size=num_obs_samples)])

# pi_obs samples w/ noise

obs_clean = np.zeros((num_obs_samples,81,14))
obs = np.zeros((num_obs_samples,81,14))

# observed data corresponds to 9x9 grid with spatial steps of size 0.5
idx = create_idx(0.5)

for i in range(num_obs_samples):
    tmp = M(params_obs[0,i],params_obs[1,i])
    obs_clean[i,:,:] = tmp[idx[0],idx[1],100::100]
    obs[i,:,:] = obs_clean[i,:,:] + np.random.normal(0.0,2.5e-3,obs_clean[i,:,:].shape)
    print(f'sample {i} done')

params_obs.dump('data/params_obs')
obs_clean.dump('data/obs_clean')
obs.dump('data/obs')
    

# calculating TV metrics
    
# computing TV between exact and KDE of DG, joint and marginals

# loading data-generating parameters
params_obs = np.load('data/params_obs', allow_pickle=True)

# computing densities
kde_param_marginals = []

np.random.seed(1234)
params_full = np.random.uniform(low=0.0, high=5.0, size=(2,10000)) # large number of uniform samples of parameter space 

exact_param_marginals = [lambda x : beta.pdf((x-1)/2,2,5)/2, # exact parameter marginals for a and b
                         lambda x : norm.pdf(x,2.5,0.5)]

exact_dg = lambda x, y : exact_param_marginals[0](x)*exact_param_marginals[1](y) # exact joint data-generating density
exact_dg = exact_dg(params_full[0,:],params_full[1,:]) # evaluate at samples for computing TV

kde_dg = GKDE(params_obs)(params_full) # KDE's of joints
        
# calculating TV metric between updated and exact joint distributions
TV = np.abs(kde_dg-exact_dg)/2
TV = np.mean(TV)*25

param_labels = [r'$a$', r'$b$']
print(f'TV between exact and KDE of DG joint densities: {TV}')
print()

# KDEs of true marginals
kde_param_marginals = []
for i in range(params_obs.shape[0]):
        kde_param_marginals.append(GKDE(params_obs[i,:]))

marginal_TV = []
for i in range(params_obs.shape[0]):
    # computing TV between KDE and exact DG marginal densities directly
    diff = lambda x : np.abs(kde_param_marginals[i](x)-exact_param_marginals[i](x))
    TV_, _ = quadrature(diff, 0.0, 5.0, tol=1e-2)
    marginal_TV.append(TV_/2)

for i in range(params_obs.shape[0]):
    print(f'TV between exact and KDE of DG marginal densities for {param_labels[i]}: {marginal_TV[i]}')

# recreating histogram of TV values with chosen sample TV value shown

plt.figure()
plt.hist(TVs)
plt.vlines(TV, 0, 300, colors=c[1], label='TV between KDE used and exact DG density')
plt.xlabel('TV')
plt.ylabel('count')
plt.title('Joint Densities')
plt.legend()
plt.tight_layout()
plt.savefig('plots/joint_TV_hist.png', bbox_inches='tight')
plt.show()

# plotting histogram with representative TV line

marginal_TVs = np.array(marginal_TVs)
param_labels = [r'$a$', r'$b$']
param_str = ['a', 'b']

for i in range(2):
    plt.figure()
    plt.hist(marginal_TVs[:,i])
    plt.vlines(marginal_TV[i], 0, 300, colors=c[1], label='TV between KDE used and exact DG density')
    plt.xlabel('TV')
    plt.ylabel('count')
    plt.title('Marginals for '+param_labels[i])
    plt.legend()
    plt.tight_layout()
    fn = 'plots/marginal_TV_hist_' + param_str[i] + '.png'
    plt.savefig(fn, bbox_inches='tight')
    plt.show()
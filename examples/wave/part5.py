import numpy as np
from luq.luq import *
from scipy.stats import norm, beta
from scipy.stats import gaussian_kde as GKDE
from scipy.integrate import quadrature
from tabulate import tabulate
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import ipywidgets as wd

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

np.random.seed(123456)

# parameter samples of pi_init
num_samples = int(1E3)
params = np.random.uniform(low=0.0,high=5.0,size=(2,num_samples))


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

# solving model on full grid
full_grid = np.zeros((num_samples,101,101,14))
for i in range(num_samples):
    tmp = M(params[0,i],params[1,i])
    full_grid[i,:,:,:] = tmp[:,:,100::100]
    print(f'sample {i} done')

preds = []
pred_grids = []
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
        preds.append(pred)
        pred_grids.append(str(int(N)) + 'x' + str(int(N)))

# different grid sizes
print('Different grid sizes used:')
print()
for grid in pred_grids:
    print(grid)


# computing normalized kernel pca alpha vectors
    
# learning QoI over each grid; same setup as part 3 but with different grid sizes
LUQs = []
for i, pred in enumerate(preds):
    LUQs.append(LUQ(pred[:,:,9]))
    LUQs[i].learn_qois_and_transform(num_qoi=2)
    
# normalize eigenvectors
unit_alphas = []
for i in range(len(preds)):
    unit_alphas.append([])
    for j in range(2):
        unit_alphas[i].append(LUQs[i].kpcas[0].eigenvectors_[:,j] / np.linalg.norm(LUQs[i].kpcas[0].eigenvectors_[:,j], ord=2))


# comparing alpha vectors
        
# applying linear regression for each grid pair
from scipy.linalg import lstsq

As = []
ms = []
bs = []
R_squared = []
for i in range(len(preds)-1):
    As.append([])
    ms.append([])
    bs.append([])
    R_squared.append([])
    for j in range(2):
        As[i].append(np.ones((preds[i].shape[0],2)))
        As[i][j][:,1] = unit_alphas[-(i+1)][j]
        coeffs, res, _, _ = lstsq(As[i][j], unit_alphas[-(i+2)][j])
        ms[i].append(coeffs[1])
        bs[i].append(coeffs[0])
        SS_tot = np.sum((unit_alphas[-(i+1)][j] - np.mean(unit_alphas[-(i+2)][j]))**2)
        R_squared[i].append(1-res/SS_tot)

# printing diagnostics
spacing = '     '
for i in range(len(pred_grids)-1):
    print(f'Results between grids {pred_grids[-(i+1)]} and {pred_grids[-(i+2)]}:')
    for j in range(2):
        print(spacing+f'QoI component {j+1}:')
        print(2*spacing+f'slope: {ms[i][j]}')
        print(2*spacing+f'R^2: {R_squared[i][j]}')
    print()

# plotting kpca components per grid
m = []
M = []
Delta = []
for i in range(2):
    m.append(np.inf)
    M.append(-np.inf)
    for j in range(len(pred_grids)):
        if unit_alphas[-(j+1)][i].min() < m[i]:
            m[i] = unit_alphas[-(j+1)][i].min()
        if unit_alphas[-(j+1)][i].max() > M[i]:
            M[i] = unit_alphas[-(j+1)][i].max()
    Delta.append(0.1 * (M[i] - m[i]))

param_str = ['a', 'b']
for i in range(len(pred_grids)-1):
    for j in range(2):
        plt.figure()
        plt.scatter(unit_alphas[-(i+1)][j], unit_alphas[-(i+2)][j], c=c[0])
        plt.xlim([m[j]-Delta[j],M[j]+Delta[j]])
        plt.ylim([m[j]-Delta[j],M[j]+Delta[j]])
        if j == 0:
            component = r'$\alpha^{(1)}$'
        else:
            component = r'$\alpha^{(2)}$'
        plt.xlabel(component + f' from {pred_grids[-(i+1)]} grid')
        plt.ylabel(component + f' from {pred_grids[-(i+2)]} grid')
        x = np.linspace(unit_alphas[-(i+1)][j].min(),
                        unit_alphas[-(i+1)][j].max(),
                        10000)
        y = ms[i][j]*x+bs[i][j]
        plt.plot(x, y, c=c[1], label=f'slope = {np.round(ms[i][j],3)}; '+r'$R^2$'+f' = {np.round(R_squared[i][j],3)}')
        plt.legend()
        plt.tight_layout()
        fn = 'plots/' + pred_grids[-(i+1)] + '_' + pred_grids[-(i+2)] + '_' + str(j+1) + '.png'
        plt.savefig(fn, bbox_inches='tight')
        plt.show()
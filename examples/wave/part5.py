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


# loading data

# loading data

preds = []
pred_grids = []
for i in range(20):
    delta = 0.05 * (i+1)
    N = (5 - delta) / delta
    if N == int(N):
        fn = 'data/pred_' + str(int(N)) + 'x' + str(int(N))
        preds.append(np.load(fn, allow_pickle=True))
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
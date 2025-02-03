import sys; sys.path.insert(0, "../../")
from dualbounds.generic import DualBounds

import numpy as np
import pandas as pd
from otcompute import vip_estimate
from utilities import generate_data, treatment
from truevalue import true_vip1dim, true_vc1dim, true_vu1dim

seed = 123
np.random.seed(seed)

N = 1000

b0 = np.array([[0.6]])
b1 = np.array([[1.6]])
dy = 1
dz = 1 

SIMSIZE = 2

result_vip = []
result_ll_brd = []
result_ll_knn = []

for k in range(SIMSIZE):
    # generate raw data 
    raw_data = generate_data(N, b0, b1)
    
    # generate post-treatment data
    A0, A1, data = treatment(raw_data, dy, dz)
    
    # Vip estimate
    etalist = np.linspace(0, 100, 50)
    # etalist = [20.]
    vip_estlist = vip_estimate(A0, A1, dy, dz, etalist)
    result_vip.append(vip_estlist)
    
    # LL's estimate
    dbnd = DualBounds(
        f=lambda y0, y1, x: (y0 + y1) ** 2,
        covariates=data['X'],
        treatment=data['W'],
        outcome=data['y'],
        propensities=data['pis'],
        outcome_model='ridge',
    )
    
    results = dbnd.fit().results()
    LL_est = results.at['Estimate', 'Lower']
    result_ll_brd.append(LL_est)
    
    
    # LL's estimate3
    dbnd3 = DualBounds(
        f=lambda y0, y1, x: (y0 + y1) ** 2,
        covariates=data['X'],
        treatment=data['W'],
        outcome=data['y'],
        propensities=data['pis'],
        outcome_model='knn',
    )
    
    results3 = dbnd3.fit().results()
    LL_est3 = results3.at['Estimate', 'Lower']
    result_ll_knn.append(LL_est3)
    
    
    print(f'{k+1}/{SIMSIZE} complete.')
    
    
    
# true Vc value 
tvc = true_vc1dim(b0.item(), b1.item())

# # true Vip value
# tvip_list = []
# for eta in etalist:
#     tvip_list.append(true_vip1dim(eta, b0.item(), b1.item()))
    
# true Vu value
tvu = true_vu1dim(b0.item(), b1.item())

import matplotlib.pyplot as plt
# Plot data
plt.figure()

result_vip = np.array(result_vip)
plot_vip = np.mean(result_vip, axis=0)
plt.plot(etalist, plot_vip, label='Vip_est($\\eta$)')

result_ll_brd = np.array(result_ll_brd)
result_ll_knn = np.array(result_ll_knn)
LL_est_brd = np.mean(result_ll_brd)
LL_est_knn = np.mean(result_ll_knn)

plt.axhline(y=LL_est_brd, linestyle='--', color='r',
            linewidth=2, label='LL_est_ridge')
plt.axhline(y=LL_est_knn, linestyle='--', color='c',
            linewidth=2, label='LL_est_knn')
plt.axhline(y=tvc, linestyle='--', color='y', linewidth=2, label='true_vc')
plt.axhline(y=tvu, linestyle='--', color='b', linewidth=2, label='true_vu')
plt.title(label=f'N ={N}, dy={dy}, dz = {dz}, seed={seed}')

# # Add legend and show the plot
plt.xlabel('$\\eta$')
plt.ylim(-0.01, None)
plt.legend()
plt.show()


# Concatenate arrays horizontally
data = np.hstack((np.array([tvc] * SIMSIZE).reshape(-1, 1),
                 result_ll_brd.reshape(-1, 1), result_ll_knn.reshape(-1, 1), result_vip))
etaindex = np.concatenate((np.array([0, 0, 0]), etalist)).reshape(
    1, 3 + result_vip.shape[1])
data = np.vstack((etaindex, data))

# Create a DataFrame
header = ["vc", "bridge", "knn"] + \
    [f"vip_col{i+1}" for i in range(result_vip.shape[1])]
df = pd.DataFrame(data, columns=header)
df.index = ['eta'] + [f"SIM{i}" for i in range(SIMSIZE)]

# Specify the output file name
output_file = f"well_cov_{N}.csv"

# Save to CSV
df.to_csv(output_file, index=True)

print(f"Arrays saved to {output_file}.")

# %%picture 1: error
N = 3000
output_file = f"well_cov_{N}.csv"
loaded_df = pd.read_csv(output_file, index_col=0)

# Extract the first row as eta index
eta = loaded_df.iloc[0].values

# Extract the remaining rows as a numpy array A
A = loaded_df.iloc[1:].values

# Compute the absolute gap and average
abs_gaps = np.abs(A[:, 1:] - A[:, [0]])
mean_gaps = np.mean(abs_gaps, axis=0)


import matplotlib.pyplot as plt

# Plot data
plt.figure()

plt.plot(eta[3:], mean_gaps[2:], label='Vip_est($\eta$)')
plt.axhline(y=mean_gaps[0], linestyle='--', color='r', linewidth=2, label='LL_ridge')
plt.axhline(y=mean_gaps[1], linestyle='--', color='c', linewidth=2, label='LL_knn')

# plt.title(label=f'N ={N}, dy={dy}, dz = {dz}, seed={seed}')

# # Add legend and show the plot
plt.xlabel('$\eta$')
plt.ylabel('Error')
plt.ylim(-0.01, None)
plt.legend()

plt.savefig(f"L1_error_well{N}.pdf")
plt.show()

# %% picture 2: bias

loaded_df = pd.read_csv(output_file, index_col=0)

# Extract the first row as eta index
eta = loaded_df.iloc[0].values

# Extract the remaining rows as a numpy array A
A = loaded_df.iloc[1:].values

# Compute the absolute gap and average
mean_gaps = np.mean(A[:, 1:], axis=0)


import matplotlib.pyplot as plt

# Plot data
plt.figure()

plt.plot(eta[3:], mean_gaps[2:], label='Vip_est($\eta$)')
plt.axhline(y=mean_gaps[0], linestyle='--', color='r', linewidth=2, label='LL_ridge')
plt.axhline(y=mean_gaps[1], linestyle='--', color='c', linewidth=2, label='LL_knn')
plt.axhline(y=tvc, linestyle='--', color='y', linewidth=2, label='true_vc')

# plt.title(label=f'N ={N}, dy={dy}, dz = {dz}, seed={seed}')

# # Add legend and show the plot
plt.xlabel('$\eta$')
plt.ylabel('Value')
plt.ylim(-0.01, None)
plt.legend()
plt.savefig(f"bias_well{N}.pdf")
plt.show()

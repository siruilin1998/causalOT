# %% z square 1-dim
from truevalue import true_vc_square
from utilities import generate_data_square, treatment
from otcompute import vip_estimate
import numpy as np
from dualbounds.generic import DualBounds
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "../../")


seed = 123
np.random.seed(seed)

N = 3000

b0 = np.array([[0.2]])
b1 = np.array([[0.6]])
dy = 1
dz = 1

SIMSIZE = 800

result_vip = []
result_ll_brd = []
result_ll_knn = []

for k in range(SIMSIZE):
    # generate raw data
    raw_data = generate_data_square(N, b0, b1)

    # generate post-treatment data
    A0, A1, data = treatment(raw_data, dy, dz)

    # Vip estimate
    etalist = np.linspace(0, 100, 50)
    vip_estlist = vip_estimate(A0, A1, dy, dz, etalist)
    result_vip.append(vip_estlist)

    # LL-ridge estimate
    dbnd_brd = DualBounds(
        f=lambda y0, y1, x: (y0 + y1) ** 2,
        covariates=data['X'],
        treatment=data['W'],
        outcome=data['y'],
        propensities=data['pis'],
        outcome_model='ridge',
    )

    result_brd = dbnd_brd.fit().results()
    LL_brd = result_brd.at['Estimate', 'Lower']
    result_ll_brd.append(LL_brd)

    # LL-knn estimate
    dbnd_knn = DualBounds(
        f=lambda y0, y1, x: (y0 + y1) ** 2,
        covariates=data['X'],
        treatment=data['W'],
        outcome=data['y'],
        propensities=data['pis'],
        outcome_model='knn',
    )

    result_knn = dbnd_knn.fit().results()
    LL_knn = result_knn.at['Estimate', 'Lower']
    result_ll_knn.append(LL_knn)

    print(f'{k+1}/{SIMSIZE} complete.')


# true Vc value
tvc = true_vc_square(b0, b1)


# Plot data
plt.figure()

result_vip = np.array(result_vip)
plot_vip = np.mean(result_vip, axis=0)
plt.plot(etalist, plot_vip, label='Vip_est($\eta$)')

result_ll_brd = np.array(result_ll_brd)
result_ll_knn = np.array(result_ll_knn)
LL_est_brd = np.mean(result_ll_brd)
LL_est_knn = np.mean(result_ll_knn)

plt.axhline(y=LL_est_brd, linestyle='--', color='r',
            linewidth=2, label='LL_est_ridge')
plt.axhline(y=LL_est_knn, linestyle='--', color='c',
            linewidth=2, label='LL_est_knn')
plt.axhline(y=tvc, linestyle='--', color='y', linewidth=2, label='true_vc')

plt.title(label=f'N ={N}, dy={dy}, dz = {dz}, seed={seed}')

# # Add legend and show the plot
plt.xlabel('$\eta$')
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
output_file = f"square_cov_{N}.csv"

# Save to CSV
df.to_csv(output_file, index=True)

print(f"Arrays saved to {output_file}.")

# %%picture 1: error
N = 3000
output_file = f"square_cov_{N}.csv"
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
plt.ylim(-0.01, None)
plt.legend()

plt.savefig(f"L1_error_square{N}.pdf")
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
plt.ylim(-0.01, None)
plt.legend()
plt.savefig(f"bias_square{N}.pdf")
plt.show()



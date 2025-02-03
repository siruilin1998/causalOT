import sys; sys.path.insert(0, "../../")
from dualbounds.generic import DualBounds

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utilities import treatment_realdata
from otcompute import vip_estimate

# read data
df = pd.read_csv('education.csv')
outcome_col = 'GPA'
covariate_col = 'gpa_baseline'
treat_col = 'treatment'


#%%
A0, A1, data = treatment_realdata(df, outcome_col, covariate_col, treat_col, 
                                  pis=0.3, scale_covariate=.5)

dy = 1
dz = 1

A1[:, 0] *= -1.


# estimation

etalist = np.linspace(0, 100, 101)
vip_estlist = vip_estimate(A0, A1, dy, dz, etalist)


# LL-ridge estimate
dbnd_brd = DualBounds(
    f=lambda y0, y1, x: (y0 - y1) ** 2,
    covariates=data['X'],
    treatment=data['W'],
    outcome=data['y'],
    propensities=data['pis'],
    outcome_model='ridge',
)

result_brd = dbnd_brd.fit().results()
LL_brd = result_brd.at['Estimate', 'Lower']

# LL-knn estimate
dbnd_knn = DualBounds(
    f=lambda y0, y1, x: (y0 - y1) ** 2,
    covariates=data['X'],
    treatment=data['W'],
    outcome=data['y'],
    propensities=data['pis'],
    outcome_model='knn',
)

result_knn = dbnd_knn.fit().results()
LL_knn = result_knn.at['Estimate', 'Lower']


#%%
# Plot data
plt.figure()

plt.plot(etalist, vip_estlist, label='Vip_est($\\eta$)')
plt.axhline(y=LL_brd, linestyle='--', color='r', linewidth=2, label='LL_ridge')
plt.axhline(y=LL_knn, linestyle='--', color='c', linewidth=2, label='LL_knn')

plt.title(label=f'Outcome:{outcome_col}, Covariate:{covariate_col}')

# # Add legend and show the plot
plt.xlabel('$\\eta$')
# plt.ylim(-0.01, None)
plt.legend()

# plt.savefig('Real Data')
plt.show()

#%%
A0, A1, data = treatment_realdata(df, outcome_col, covariate_col, treat_col, 
                                  pis=0.3, scale_covariate=.3)

dy = 1
dz = 1


# estimation

etalist = np.linspace(0, 100, 101)
vip_estlist2 = vip_estimate(A0, A1, dy, dz, etalist)


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
LL_brd2 = result_brd.at['Estimate', 'Lower']

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
LL_knn2 = result_knn.at['Estimate', 'Lower']



# Plot data
plt.figure()

plt.plot(etalist, vip_estlist2, label='Vip_est($\eta$)')
plt.axhline(y=LL_brd2, linestyle='--', color='r', linewidth=2, label='LL_ridge')
plt.axhline(y=LL_knn2, linestyle='--', color='c', linewidth=2, label='LL_knn')

plt.title(label=f'Outcome:{outcome_col}, Covariate:{covariate_col}')

# # Add legend and show the plot
plt.xlabel('$\eta$')
# plt.ylim(-0.01, None)
plt.legend()

# plt.savefig('Real Data')
plt.show()


#%% rho
A0, A1, data = treatment_realdata(df, outcome_col, covariate_col, treat_col, 
                                  pis=0.3, scale_covariate=.5)

# compute the upper bound for rho
std0 = np.std(A0[:, 0])
std1 = np.std(A1[:, 0])
M20 = np.mean(A0[:, 0] ** 2)
M21 = np.mean(A1[:, 0] ** 2)
M10 = np.mean(A0[:, 0])
M11 = np.mean(A1[:, 0])

vip_lower = (0.5 * (np.array(vip_estlist2) - M20 - M21) - M10 * M11) * std0 ** (-1) * std1 ** (-1)
LL_brd_lower = (0.5 * (np.array(LL_brd2) - M20 - M21) - M10 * M11) * std0 ** (-1) * std1 ** (-1)
LL_knn_lower = (0.5 * (np.array(LL_knn2) - M20 - M21) - M10 * M11) * std0 ** (-1) * std1 ** (-1)

vip_upper = (-0.5 * (np.array(vip_estlist) - M20 - M21) - M10 * M11) * std0 ** (-1) * std1 ** (-1)

LL_brd_upper = (-0.5 * (np.array(LL_brd) - M20 - M21) - M10 * M11) * std0 ** (-1) * std1 ** (-1)


LL_knn_upper = (-0.5 * (np.array(LL_knn) - M20 - M21) - M10 * M11) * std0 ** (-1) * std1 ** (-1)




# Plot data
plt.figure()

plt.plot(etalist, vip_lower, label='Vip_est($\eta$)_l')
plt.axhline(y=LL_brd_lower, linestyle='--', color='r', linewidth=2, label='LL_ridge_l')
plt.axhline(y=LL_knn_lower, linestyle='--', color='c', linewidth=2, label='LL_knn_l')


plt.plot(etalist, vip_upper, label='Vip_est($\eta$)_u')
plt.axhline(y=LL_brd_upper, linestyle='--', color='y', linewidth=2, label='LL_ridge_u')
plt.axhline(y=LL_knn_upper, linestyle='--', color='g', linewidth=2, label='LL_knn_u')

plt.title(label=f'rho: Outcome:{outcome_col}, Covariate:{covariate_col}')

# # Add legend and show the plot
plt.xlabel('$\eta$')
# plt.ylim(-0.01, None)
plt.legend()

# plt.savefig('Real Data')
plt.show()

#%% save file
upperarray = np.concatenate((np.array([LL_brd_upper, LL_knn_upper]), vip_upper))
lowerarray = np.concatenate((np.array([LL_brd_lower, LL_knn_lower]), vip_lower))
etaarray = np.concatenate(([0,0], etalist))
df = pd.DataFrame(np.vstack((etaarray, upperarray, lowerarray)), index=['eta', 'upper', 'lower'])
df.to_csv('rho.csv', index=True)


#%% Neymanian

S02 = std0 ** 2
S02 /= A0.shape[0]

S12 = std1 ** 2
S12 /= A1.shape[0]

meandiff_sq = (np.mean(A0[:, 0]) - np.mean(A1[:, 0])) ** 2 / (A0.shape[0] + A1.shape[0])

V_vip_lower = S02 + S12 - np.array(vip_estlist) / (A0.shape[0] + A1.shape[0]) + meandiff_sq
V_LL_brd_lower = S02 + S12 - LL_brd_lower / (A0.shape[0] + A1.shape[0]) + meandiff_sq
V_LL_knn_lower = S02 + S12 - LL_knn_lower / (A0.shape[0] + A1.shape[0]) + meandiff_sq

# Plot data
plt.figure()

plt.plot(etalist, V_vip_lower, label='Vip_est($\eta$)_l')
plt.axhline(y=V_LL_brd_lower, linestyle='--', color='r', linewidth=2, label='LL_ridge_l')
plt.axhline(y=V_LL_knn_lower, linestyle='--', color='c', linewidth=2, label='LL_knn_l')

plt.title(label=f'V (lower): Outcome:{outcome_col}, Covariate:{covariate_col}')

# # Add legend and show the plot
plt.xlabel('$\eta$')
# plt.ylim(-0.01, None)
plt.legend()

# plt.savefig('Real Data')
plt.show()

#%%
Vlowerarray = np.concatenate((np.array([V_LL_brd_lower, V_LL_knn_lower]), V_vip_lower))
etaarray = np.concatenate(([0,0], etalist))
dfV = pd.DataFrame(np.vstack((etaarray, Vlowerarray)), index=['eta', 'lower'])
dfV.to_csv('V_.csv', index=True)



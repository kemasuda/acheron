#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import corner
import glob, dill, sys, os
%matplotlib inline

#%%
import seaborn as sns
sns.set(style='ticks', font_scale=1.6, font='times')
from matplotlib import rc
rc('text', usetex=True)

#%%
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
import jax.random as random
from jax.config import config
config.update('jax_enable_x64', True)
#numpyro.set_host_device_count(2)

#%%
def set_matrix(N, a, jit):
    K = -1 * np.ones((N, N))
    for i in range(N):
        K[i][i] = 2
    K[0,0] = 1
    K[N-1,N-1] = 1
    K[0,N-1] = 0
    K[N-1,0] = 0
    return K
    #return a * K + np.eye(N) * jit

def det_pred(N, a, jit):
    return N * jit * a**(N-1)

#%%
N, a = 5, 10

#%%
jit = np.logspace(-5, 3, 100)

#%%
K0 = jnp.array(set_matrix(N, a, j) )

#%%
dets = []
for j in jit:
    #K = set_matrix(N, a, j)
    K = a * K0 + jnp.eye(N) * j
    #dets.append(jnp.linalg.det(K))
    dets.append(jnp.linalg.slogdet(K)[1])
dets = np.array(dets)

#%%
plt.xscale("log")
plt.yscale("log")
plt.plot(jit, jnp.exp(dets), '.')
plt.plot(jit, det_pred(N, a, jit), '-')

#%%
N = 4
arr = np.logspace(-3, 3, 10)

#%%
jits = np.logspace(-10, 0, 6)[::-1]
jits = np.array([10, 1, 0.1, 0.01, 0.001, 1e-5])

#%%
plt.figure(figsize=(14,7))
for jit in jits:
    dets = []
    for a in arr:
        K = set_matrix(N, a, jit)
        dets.append(np.linalg.det(K))
    dets = np.array(dets)

    plt.xlabel("a")
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(arr, dets / N / jit, '-')
    plt.plot(arr, arr**(N-1), '-', color='gray', lw=5, alpha=0.2)

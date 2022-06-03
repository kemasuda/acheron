#%%
import numpy as np
import pandas as pd
import sys, os, glob
import matplotlib.pyplot as plt
from arviz import hdi
import corner
%matplotlib inline

#%%
import seaborn as sns
sns.set(style='ticks', font_scale=1.6, font='times')
#sns.set(style='whitegrid', font_scale=1.6, font='times')
from matplotlib import rc
rc('text', usetex=True)
plt.rcParams['figure.facecolor'] = 'white'

#%%
postdir, idx = '../posteriors_simulated_cks/', 1408-1408+3989
id = "%05d"%int(idx)
file = postdir + str(id) + '_samples.csv'
dp = pd.read_csv(file)

#%%
keys = ['mass', 'age', 'teff', 'feh', 'kmag', 'parallax']
truths = np.array(d.loc[idx][[k+"_true" for k in keys]])
fig = corner.corner(dp[keys], truths=truths, show_titles=True, title_fmt='.2f')
#plt.savefig(outdir+"%04d.png"%int(idx), dpi=200, bbox_inches="tight")

#%%
#d = pd.read_csv("check_recovery_merged.csv")
#outdir = 'check_recovery_corner/'

d = pd.read_csv("../simulated-cks_results.csv")
outdir = './'

#%%
d.iloc[3989]

#%%
#_mass, _age, _feh = 1.15, 6.0, 0.0
#_mass, _age, _feh = 1.1, 9.0, 0.0
_mass, _age, _feh = 1.0, 10.0, 0.1
_mass, _age, _feh = 1.05, 1.0, 0.1
_mass, _age, _feh = 0.9, 10, -0.1
_mass, _age, _feh = 0.85, 3, -0.
_mass, _age, _feh = 1.05, 6, -0.


#%%
dists = (d.mass_true - _mass)**2 + (d.age_true - _age)**2 + (d.feh_true - _feh)**2
idx = np.argmin(dists)

#%%
print ((d.teff_true - d.iso_teff)[idx])
print ((d.feh_true - d.iso_feh)[idx])

#%%
#postdir = 'simulated-cks/'
postdir = '../posteriors_simulated_cks/'
id = "%05d"%int(idx)
file = postdir + str(id) + '_samples.csv'
dp = pd.read_csv(file)
print ("%d chosen."%idx)

#%%
keys = ['mass', 'age']
fmts = ['%.2f', '%.1f']
isos = np.array(d.iloc[idx][["iso_"+k for k in keys]])
isos_upp = np.array(d.iloc[idx][["iso_"+k+"_upp" for k in keys]])
isos_low = np.array(d.iloc[idx][["iso_"+k+"_low" for k in keys]])
isos
truths = np.array(d.loc[idx][[k+"_true" for k in keys]])
fig = corner.corner(dp[keys], truths=truths, show_titles=True*0, title_fmt='.2f',
labels=['mass ($M_\odot$)', 'age (Gyr)'])
axes = np.array(fig.axes).reshape((2,2))
for i in range(len(keys)):
    ax = axes[i,i]
    ys = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 100)
    ax.fill_betweenx(ys, isos[i]-isos_low[i], isos[i]+isos_upp[i], alpha=0.2, color='gray')
    ax.set_title(("truth: %s"%fmts[i])%truths[i]+("\n 16/50/84\,th: $%s^{+%s}_{-%s}$"%(fmts[i], fmts[i], fmts[i]))%(isos[i], isos_upp[i], isos_low[i]), fontsize=18)
    ax.axvline(x=isos[i], alpha=0.8, color='gray', ls='dashed', lw=1.5)
#plt.savefig(outdir+"ma_%04d.png"%int(idx), dpi=200, bbox_inches="tight")

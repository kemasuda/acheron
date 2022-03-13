#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import corner
import glob, dill, sys, os
sys.path.append(os.path.join(os.path.dirname('__file__'), '/Users/k_masuda/Dropbox/jaxstar/mistfit/'))
from mistfit_iso import MistFit, MistGridIso
from jax.scipy.ndimage import map_coordinates as mapc
%matplotlib inline

#%%
import seaborn as sns
sns.set(style='ticks', font_scale=1.6, font='times')
from matplotlib import rc
rc('text', usetex=True)

#%% simulated3
"""
#Nsample = 200
Nsample = 20000
label = 'simulated3-large'
#masses = np.random.rand(Nsample)*0.2 + 0.85
#masses = np.random.rand(Nsample)*0.3 + 0.85
masses = np.random.rand(Nsample)*0.4 + 0.75
#ages = np.random.rand(Nsample)*13.5
ages = np.random.rand(Nsample)*13.8 + 0.1
fehs = np.random.randn(Nsample)*0.1

#%% simulated4
label = 'simulated4-large'
masses = np.random.randn(Nsample)*0.1 + 1
ages = np.random.randn(Nsample)*1 + 5
fehs = np.random.randn(Nsample)*0.1
logages = np.log10(ages*1e9)
"""

#%%
d = pd.read_csv("/Users/k_masuda/Dropbox/nnstar/isoinput_cks.csv")
d = d[d.kepid!=3957082].reset_index(drop=True)
d['teff'] = d.teff.astype(float)
d['feh'] = d.feh.astype(float)
d['logplx'] = np.log10(d.parallax)

#%%
stats = ["mean", "median", "std", "min", "max"]
print (d.agg({"logplx": stats, "parallax_over_error": stats, "kmagcorr_err": stats, "teff": stats, "feh": stats}))

#%%
mass_min, mass_max = 0.7, 1.3
kmag_err = 0.023 # median
parallax_over_err = 100 # median
logplx_mean = 0.18
logplx_std = 0.25
feh_mean = 0.03
feh_std = 0.18
eep_max = 600

#%%
teff_err, feh_err = 110, 0.1

#%% simulate_cks
np.random.seed(123)
label = 'simulated-cks'
N = 20000 * 10
masses = np.random.rand(N) * (mass_max - mass_min) + mass_min
ages = np.random.rand(N) * 13.7 + 0.1
fehs = np.random.randn(N) * feh_std + feh_mean
logplxs = np.random.randn(N) * logplx_std + logplx_mean
logages = np.log10(ages*1e9)

#%%
bins = np.linspace(-0.6, 0.4, 30)
plt.hist(d.feh, density=True, bins=bins)
plt.hist(fehs, density=True, histtype='step', bins=bins);

#%%
bins = np.linspace(-0.8, 1.4, 30)
plt.hist(d.logplx, density=True, bins=bins)
plt.hist(logplxs, density=True, histtype='step', bins=bins);

#%%
mf = MistGridIso("/Users/k_masuda/Dropbox/jaxstar/mistfit/mistgrid_iso.npz")

#%%
mf.set_keys(['teff', 'kmag', 'mass', 'radius'])

#%%
eeps = np.arange(eep_max)
ones = np.ones_like(eeps)
teffs, kmags, idx, rads, eepints = [], [], [], [], []
n = N#sample
for loga, feh, mass in zip(logages[:n], fehs[:n], masses[:n]):
    _teff, _kmag, _mass, _rad = mf.values(loga*ones, feh*ones, eeps)
    #_idx = np.nanargmin(np.abs(_mass-mass))
    _idxu = np.searchsorted(_mass, mass)
    _idxl = _idxu - 1
    if _idxu == len(eeps):
        idx.append(False)
        continue
    idx.append(True)
    _ml, _mu = _mass[_idxl], _mass[_idxu]
    wl, wu = (_mu-mass)/(_mu-_ml), (mass-_ml)/(_mu-_ml)
    teffs.append(_teff[_idxl]*wl + _teff[_idxu]*wu)
    kmags.append(_kmag[_idxl]*wl + _kmag[_idxu]*wu)
    rads.append(_rad[_idxl]*wl + _rad[_idxu]*wu)
    eepints.append(eeps[_idxl]*wl + eeps[_idxu]*wu)
    if _teff[_idxl]*wl + _teff[_idxu]*wu != _teff[_idxl]*wl + _teff[_idxu]*wu:
        print (_idxl, loga, feh, mass, _mass)
teffs = np.array(teffs)
kmags = np.array(kmags)
rads = np.array(rads)
eepints = np.array(eepints)
idx = np.array(idx)
print (np.sum(idx))

#%%
dsim = pd.DataFrame(data={
    "teff_true": teffs, "kmag_true": kmags - 5*logplxs[idx] + 10, "eep_true": eepints, "radius_true": rads,
    "mass_true": masses[idx], "age_true": ages[idx], "feh_true": fehs[idx], "parallax_true": 10**logplxs[idx],
    "teff_error": teff_err, "feh_error": feh_err, "kmag_error": kmag_err,
    "parallax_error": 10**logplxs[idx] / parallax_over_err,
})

#%%
np.random.seed(123)
for key in ["teff", "feh", "kmag", "parallax"]:
    dsim[key+"_obs"] = dsim[key+"_true"] + np.random.randn(len(dsim)) * dsim[key+"_error"]

#%%
dsim

#%%
"""
np.random.seed(123)
parallax_true = 10**(np.random.randn(np.sum(idx)) * 0.25 + 0.2)
#parallax_error = parallax_true / 10**(np.random.randn(np.sum(idx)) * 0.35 + 2)
parallax_error = parallax_true / 100
parallax_obs = parallax_true + np.random.randn(len(parallax_true)) * parallax_error

#%%
dsim = pd.DataFrame(data={"teff_true": teffs, "kmag_true": kmags+10-5*np.log10(parallax_true), "mass_true": masses[idx], "age_true": ages[idx], "feh_true": fehs[idx], "eep_true": eepints, "radius_true": rads})

#%%
#np.random.seed(1233)
kerr = 0.023
dsim['teff_obs'] = dsim.teff_true + np.random.randn(len(dsim)) * 110
dsim['feh_obs'] = dsim.feh_true + np.random.randn(len(dsim)) * 0.1
dsim['kmag_obs'] = dsim.kmag_true + np.random.randn(len(dsim)) * kerr
dsim['teff_error'] = 110
dsim['feh_error'] = 0.1
dsim['kmag_error'] = kerr
dsim['parallax_true'] = parallax_true
dsim['parallax_obs'] = parallax_obs
dsim['parallax_error'] = parallax_error
"""

#%%
dsim['kepid'] = ["%05d"%s for s in np.arange(len(dsim))]

#%%
label
if not os.path.exists(label):
    os.system("mkdir %s"%label)
dsim.to_csv(label+"-long.csv", index=False)

#%%
plt.plot(dsim.mass_true, dsim.age_true, '.')

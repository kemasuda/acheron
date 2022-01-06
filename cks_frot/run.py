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
#from jax.config import config
#config.update('jax_enable_x64', True)

#%%
from hierarchical import TwodHierarchical

#%%
name = "iso_loga_m0.75-1.25"
sample_log = True

#%%
d = pd.read_csv(name+".csv")
mmed, amed = d.iso_mass, d.iso_age
rotflag = d.acf

#%%
samples = np.load(name+".npz")['samples']
Nsys, Nsample, _ = np.shape(samples)
print ('# %s samples for %d stars.'%(Nsample, Nsys))

#%%
bin_log = True*0

#%%
if bin_log:
    ymin, ymax, dy = 8, 10.14, 0.15
    amed = np.log10(amed*1e9)
else:
    ymin, ymax, dy = 0, 14, 1

#%%
xmin, xmax, dx  = 0.7, 1.3, 0.05

#%%
hm = TwodHierarchical(samples, xmin, xmax, dx, ymin, ymax, dy, bin_log, sample_log, xvals=mmed, yvals=amed)

#%%
n_sample = 2000
hm.setup_hmc(num_warmup=n_sample, num_samples=n_sample)

#%%
eps = 2
rflag = np.array(rotflag).astype(float)
rflag = None

#%%
rng_key = random.PRNGKey(0)
hm.run_hmc(rng_key, rflag=rflag, eps=eps, extra_fields=('potential_energy',))

#%%
hm.summary_plots(rotflag=rotflag)

#%%
rotfracs = np.array(hm.samples['fracs'])
rotfracs[:,~hm.idx_valid_mass_age] = np.nan
Nxbin, Nybin = hm.Nxbin, hm.Nybin
xbins, ybins, xbins_center, ybins_center = hm.xbins, hm.ybins, hm.xbins_center, hm.ybins_center
priors_grid = hm.samples['priors'].reshape((n_sample, Nybin, Nxbin))

#%%
rotfracs_grid = rotfracs.reshape((n_sample, Nybin, Nxbin))
rxs = jnp.sum(rotfracs_grid, axis=1)
rys = jnp.sum(rotfracs_grid, axis=2)
rxmean, rxstd = jnp.mean(rxs, axis=0), jnp.std(rxs, axis=0)
rymean, rystd = jnp.mean(rys, axis=0), jnp.std(rys, axis=0)

#%%
from scipy.stats import gaussian_kde  as kde
#from sklearn.neighbors import KernelDensity as KD
f0 = np.linspace(0, 1, 100)
"""
skde = KD(kernel='epanechnikov', bandwidth=0.1).fit(fm[:,0,5].reshape(-1, 1))
plt.plot(f0, kde(fm[:,0,5], bw_method=0.05)(f0))
plt.plot(f0, np.exp(skde.score_samples(f0.reshape(-1,1))))
plt.hist(fm[:,0,5], density=True, bins=100);
"""

#%%
def rotfrac_bin(rotfracs_grid, ibin, pct=False):
    fm = rotfracs_grid[:,:,ibin]
    pm = priors_grid[:,:,ibin]

    yvals, ylows, yupps, fms = [], [], [], []
    for i in range(len(ybins_center)):
        #fmarr = fm[:,i].ravel()
        #fmarr = np.average(fm[:,i], axis=1, weights=pm[:,i])
        fmarr = np.array(fm[:,i].ravel())
        idx = (fmarr==fmarr)

        if np.sum(idx)>=n_sample:
            fmarr, pmarr = fmarr[idx], np.array(pm[:,i].ravel())[idx]
            pmarr = pmarr / np.sum(pmarr)
            fmarr = np.random.choice(fmarr, n_sample, p=pmarr)
            fms.append(fmarr)
            if pct:
                yl, yval, yu = np.percentile(fmarr, [16, 50, 84])
            else:
                yl, yu = hdi(fmarr, hdi_prob=0.68)
                #_yl, _yu = hdi(fmarr, hdi_prob=0.68)
                #yval = 0.5 * (_yl + _yu)
                yval = f0[np.argmax(kde(fmarr, bw_method=0.1)(f0))]
        else:
            yval, yu, yupp = np.nan, np.nan, np.nan
            fms.append(np.ones(n_sample)*np.nan)
        yvals.append(yval)
        yupps.append(yu-yval)
        ylows.append(yval-yl)
    return yvals, ylows, yupps, np.array(fms).T
ms = np.array([0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.3])

#%%
fig, ax = plt.subplots(3, 2, figsize=(16, 12*0.8), sharex=True, sharey=True)
#plt.xlim(0., 14)
plt.xlim(ybins[0], ybins[-1])
plt.ylim(0, 1)
for i, (ml, mu) in enumerate(zip(ms[:-1], ms[1:])):
    if i==5:
        ml, mu = 0.75, 1.25
        title = 'all mass'
        label = "peak"
    else:
        title = "mass $%.2f$-$%.2f\,M_\odot$"%(ml, mu)
        label = "peak"
    ibin = (ml<xbins_center)&(xbins_center<mu)
    #yvals, ylows, yupps = rotfrac_bin(rotfracs_grid, ibin, pct=True)
    yvals, ylows, yupps, fmi = rotfrac_bin(rotfracs_grid, ibin, pct=False)

    ax[i%3, i//3].errorbar(ybins_center, yvals, xerr=0.47, yerr=[ylows, yupps], fmt='o', label=label, mfc='steelblue', lw=0)

    """
    _fmi = []
    for j in range(len(yvals)):
        yl, yu = yvals[j]-ylows[j], yvals[j]+yupps[j]
        _fm = fmi[:,j]
        _fmi.append(_fm[(yl<_fm)&(_fm<yu)])
    fmi = np.array(_fmi).T
    """

    violins = ax[i%3, i//3].violinplot(fmi, positions=ybins_center, widths=0.6*dy, showextrema=True*0, showmeans=True, bw_method=0.15)
    for pc in violins['bodies']:
        pc.set_facecolor('C0')
        pc.set_edgecolor('black')
    violins['cmeans'].set_edgecolor("C0")
    violins['cmeans'].set_label("mean")

    ax[i%3, i//3].set_title(title)
    #ax[i%3, i//3].set_xscale("log")
    #ax[i%3, i//3].set_yscale("log")
    #ax[i%3, i//3].set_xlim(1, 13.5)
ax[0,1].legend(loc='best', handlelength=0.7)
#for i in range(3):
ax[1,0].set_ylabel("fraction of stars with rotation periods")
ax[2,0].set_xlabel("age (Gyr)")
ax[2,1].set_xlabel("age (Gyr)")
plt.tight_layout()
#plt.savefig(outname+"_rotfrac-hdi_weighted.png", dpi=200, bbox_inches="tight")
#plt.savefig(outname+"_rotfrac-violin_weighted.png", dpi=200, bbox_inches="tight")

#%%
"""
#for ibin in range(len(xbins_center)):
for i, (ml, mu) in enumerate(zip(mgrid[:-1], mgrid[1:])):
    ibin = (ml<xbins_center)&(xbins_center<mu)
    #yvals, ylows, yupps = rotfrac_bin(rotfracs_grid, ibin, pct=True)
    yvals, ylows, yupps = rotfrac_bin(rotfracs_grid, ibin, pct=False)
    plt.figure(figsize=(10,5))
    plt.xlim(0, 14)
    plt.ylim(0, 1.0)
    #plt.title("%.2f-%.2f"%(xbins[ibin], xbins[ibin+1]))
    plt.title("%.2f-%.2f"%(ml, mu))
    plt.errorbar(ybins_center, yvals, fmt='.', marker='o', xerr=0.47, yerr=[ylows, yupps], lw=1)


#%%
mgrid = [0.75, 0.85, 0.95, 1.05, 1.15, 1.25]

#%%
np.shape(rotfracs[:,Xbins_center<1.0])
ml, mu = mgrid[0], mgrid[1]
X2d = Xbins_center.reshape(Nybin, Nxbin)
#midx = (ml<X2d) & (X2d<mu)
midx = (ml<xbins_center)&(xbins_center<mu)
np.shape()
np.shape(rotfracs_grid[:,midx])
np.shape(rotfracs_grid)

#%%
plt.figure(figsize=(10,7*2))
plt.xlabel("age (Gyr)")
plt.ylabel("fraction of stars with Prot")
n = len(mgrid) - 1
for i, (ml, mu) in enumerate(zip(mgrid[:-1], mgrid[1:])):
    plt.subplot(n,1,i+1)
    midx = (ml<xbins_center)&(xbins_center<mu)
    mmarg = jnp.mean(rotfracs_grid[:,:,midx], axis=2)
    fmarg = jnp.mean(mmarg, axis=0)
    fmarg_std = jnp.std(mmarg, axis=0)
    plt.xlim(0, 14)
    plt.ylim(0, 1.05)
    plt.plot(np.repeat(ybins, 2), np.r_[[0], np.repeat(fmarg, 2), [0]], color='C%d'%i, label='prediction')
    plt.fill_between(np.repeat(ybins, 2), np.r_[[0], np.repeat(fmarg-fmarg_std, 2), [0]], np.r_[[0], np.repeat(fmarg+fmarg_std, 2), [0]], color='C%d'%i, alpha=0.2)
    plt.title("$M_\star=%.2f-%.2f\,M_\odot$"%(ml,mu))
plt.legend(loc='best', bbox_to_anchor=(1,1))
plt.tight_layout()
#plt.savefig(outname+"_r_age.png", dpi=200, bbox_inches="tight")

#%%
plt.figure(figsize=(10,7*2))
plt.xlabel("age (Gyr)")
plt.ylabel("fraction of stars with Prot")
n = int(np.sum((xbins_center>=0.75) & (xbins_center<=1.25)))
j = 1
for i,m in enumerate(xbins_center):
    #plt.subplot(n+1,1,j)
    plt.subplot(6,2,j)
    if m<0.75 or m>1.25:
        continue
    fmarg = jnp.mean(rotfracs_grid[:,:,i], axis=0)
    fmarg_std = jnp.std(rotfracs_grid[:,:,i], axis=0)
    plt.xlim(0, 14)
    plt.ylim(0, 1.05)
    plt.plot(np.repeat(ybins, 2), np.r_[[0], np.repeat(fmarg, 2), [0]], color='C%d'%i, label='prediction (%.2f)'%m)
    plt.fill_between(np.repeat(ybins, 2), np.r_[[0], np.repeat(fmarg-fmarg_std, 2), [0]], np.r_[[0], np.repeat(fmarg+fmarg_std, 2), [0]], color='C%d'%i, alpha=0.2)
    plt.title("$M_\star=%.2f\,M_\odot$"%m)
    j += 1
plt.legend(loc='best', bbox_to_anchor=(1,1))
plt.tight_layout()
plt.savefig(outname+"_r_age.png", dpi=200, bbox_inches="tight")
"""

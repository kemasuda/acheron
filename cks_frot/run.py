#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import corner
import glob, dill, sys, os
from jhbayes import TwodHierarchical

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

#%%
#name, sample_log, bin_log = "iso_lina_m0.70-1.30", False, False
#name, sample_log, bin_log = "joint_lina_m0.70-1.30", False, False
name, sample_log, bin_log = sys.argv[1], False, False

#%%
d = pd.read_csv(name+".csv")
mmed, amed, rotflag = d.iso_mass, d.iso_age, d.acf

#%%
samples = np.load(name+".npz")['samples']
Nsys, Nsample, _ = np.shape(samples)
print ('# %s samples for %d stars.'%(Nsample, Nsys))

#%%
xlabel = 'mass ($M_\odot$)'
xmin, xmax, dx  = 0.7, 1.3, 0.05
ymin, ymax, dy = 0, 14, 1
ylabel = 'age (Gyr)'

#%%
def valid_mass_age(mass, age):
    return age < -30 * (mass-1.25) + 5.

#%%
hm = TwodHierarchical(samples, xmin, xmax, dx, ymin, ymax, dy, bin_log, sample_log, xvals=mmed, yvals=amed, valid_xybin_func=valid_mass_age)

#%%
model, gpkernel = 'step', None
#model, gpkernel = 'gp', 'rbf'

#%%
n_sample = 5000
hm.setup_hmc(num_warmup=n_sample, num_samples=n_sample, model=model)

#%%
rflag = np.array(rotflag).astype(float)

#%%
outname = "results/" + name + "_" + model
if 'gp' in outname:
    outname += "-" + gpkernel
outname += "_n%d"%(n_sample)
outname

#%%
resume = True

#%%
mcmcfile = outname+"_mcmc.pkl"
if os.path.exists(mcmcfile) and resume:
    hm.load_mcmc(outname+"_mcmc.pkl")
    print ("# mcmc file loaded.")
else:
    rng_key = random.PRNGKey(0)
    hm.run_hmc(rng_key, rflag=rflag, extra_fields=('potential_energy',), save=outname, gpkernel=gpkernel)

#%%
hm.summary_plots(xlabel, ylabel, rotflag=rotflag, save=outname)

#%%
if model=='gp':
    keys = ["lnlenx", "lnleny", "lna"]
else:
    keys = ["lneps"]
labels = [k.replace("_", "") for k in keys]
hyper = pd.DataFrame(data=dict(zip(keys, [hm.samples[k] for k in keys])))
fig = corner.corner(hyper, labels=labels, show_titles=".2f")
plt.savefig(outname+"_hyper.png", dpi=200, bbox_inches="tight")

#%%
from scipy.stats import gaussian_kde as kde
from arviz import hdi
bw_method = 0.05
f0 = np.linspace(0, 1, 100)
def rotfrac_bin(fm, pm, pct=False):
    yvals, ylows, yupps, fms = [], [], [], []
    for i in range(len(ybins_center)):
        fmarr, pmarr = np.array(fm[:,i].ravel()), np.array(pm[:,i].ravel())
        idx = (fmarr==fmarr) # grids outside valid region

        if np.sum(idx)>=n_sample:
            fmarr, pmarr = fmarr[idx], pmarr[idx]
            pmarr = pmarr / np.sum(pmarr)
            fmarr = np.random.choice(fmarr, n_sample, p=pmarr)
            fms.append(fmarr)
            if pct:
                yl, yval, yu = np.percentile(fmarr, [16, 50, 84])
            else:
                yl, yu = hdi(fmarr, hdi_prob=0.68)
                yval = f0[np.argmax(kde(np.r_[fmarr, -fmarr, 2-fmarr], bw_method=bw_method)(f0))] # reflect edges
        else:
            yval, yu, yl = np.nan, np.nan, np.nan
            fms.append(np.ones(n_sample)*np.nan)
        yvals.append(yval)
        yupps.append(yu-yval)
        ylows.append(yval-yl)
    return yvals, ylows, yupps, np.array(fms).T

#%%
Nxbin, Nybin = hm.Nxbin, hm.Nybin
xbins, ybins, xbins_center, ybins_center = hm.xbins, hm.ybins, hm.xbins_center, hm.ybins_center
priors_grid = hm.samples['priors'].reshape((n_sample, Nybin, Nxbin))
rotfracs = np.array(hm.samples['fracs'])
rotfracs[:,~hm.idx_valid_xybin] = np.nan
rotfracs_grid = rotfracs.reshape((n_sample, Nybin, Nxbin))

#%%
ms = np.array([0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.3])

#%%
ddet = pd.read_csv("detection_model/"+name+"_det.csv")

#%%
fig, ax = plt.subplots(3, 2, figsize=(16*0.9, 9.6*0.9), sharex=True, sharey=True)
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
        label = "peak \& 68\% HDI"
    ibin = (ml < xbins_center) & (xbins_center < mu)
    fm = rotfracs_grid[:,:,ibin]
    pm = priors_grid[:,:,ibin]
    yvals, ylows, yupps, ysmp = rotfrac_bin(fm, pm, pct=False)

    # violin
    violins = ax[i%3, i//3].violinplot(np.r_[ysmp, -ysmp, 2-ysmp], positions=ybins_center, widths=0.6*dy, showextrema=False, showmeans=False, bw_method=bw_method)
    for pc in violins['bodies']:
        pc.set_facecolor('C0')
        pc.set_edgecolor('black')

    # peak & HDI
    ax[i%3, i//3].errorbar(ybins_center, yvals, yerr=[ylows, yupps], fmt='o', label=label, #mfc='white',
    lw=1, capsize=4, color='steelblue', markersize=7)

    if title!='all mass':
        ax[i%3, i//3].plot(ddet['a%02d'%(5*(ml+mu))], ddet['m%02d'%(5*(ml+mu))], lw=3, ls='dashed', alpha=0.6, color='tan')

    # mean
    ymean = np.mean(ysmp, axis=0)
    _idx = ymean == ymean
    eb = ax[i%3, i//3].errorbar(ybins_center[_idx], ymean[_idx], fmt='.', ms=0, xerr=0.25*dy, color='gray', lw=1.5, label='mean')
    eb[-1][0].set_linestyle('dotted')

    ax[i%3, i//3].set_title(title)
ax[1,1].legend(loc='best', handlelength=0.7)
ax[1,0].set_ylabel("fraction of stars with detected rotation periods")
ax[2,0].set_xlabel(ylabel)
ax[2,1].set_xlabel(ylabel)
plt.tight_layout()
plt.savefig(outname+"_violin.png", dpi=200, bbox_inches="tight")

#%% check KDE width
"""
reflect = True
for i in range(len(ybins)-1):
    _y = ysmp[:,i]
    plt.figure()
    plt.xlim(0, 1)
    if reflect:
        plt.hist(np.r_[_y, -_y, 2-_y], bins=100, density=True)
        plt.plot(f0, kde(np.r_[_y, -_y, 2-_y], bw_method=bw_method)(f0))
    else:
        plt.hist(_y, bins=100, density=True)
        plt.plot(f0, kde(_y, bw_method=bw_method)(f0));
"""

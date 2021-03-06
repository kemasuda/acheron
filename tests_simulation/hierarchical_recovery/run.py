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
#name, sample_log, bin_log = "func0", False, False
name, sample_log, bin_log = sys.argv[1], False, False

#%%
d = pd.read_csv(name+".csv")
mmed, amed, rotflag = d.iso_mass, d.iso_age, None
mtrue, atrue = np.load(name+".npz")['dtruths'].T

#%%
samples = np.load(name+".npz")['samples']
Nsys, Nsample, _ = np.shape(samples)
print ('# %s samples for %d stars.'%(Nsample, Nsys))

#%%
xlabel = 'mass ($M_\odot$)'
ymin, ymax, dy = 0, 14, 1
ylabel = 'age (Gyr)'

#%%
xmin, xmax, dx  = 0.7, 1.3, 0.05

#%%
sidx = (xmin<samples[:,:,0])&(samples[:,:,0]<xmax)&(ymin<samples[:,:,1])&(samples[:,:,1]<ymax)
validsample = np.sum(sidx, axis=1) > 0
print ("%s stars have no posterior samples in the valid region."%np.sum(~validsample))

#%%
def valid_mass_age(mass, age):
    return age < -30 * (mass-1.25) + 5.

#%%
hm = TwodHierarchical(samples[validsample], xmin, xmax, dx, ymin, ymax, dy, bin_log, sample_log, xvals=mmed, yvals=amed, vallabel='sample medians', xtrue=mtrue, ytrue=atrue, truelabel='truth', valid_xybin_func=valid_mass_age)

#%%
model, gpkernel = 'step', None

#%%
n_sample = 10000
hm.setup_hmc(num_warmup=n_sample, num_samples=n_sample, model=model)

#%%
rflag = None
lneps = None

#%%
outname = "results/" + name + "_" + model
if 'gp' in outname:
    outname += "-" + gpkernel
outname += "_n%d"%(n_sample)

#%%
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
    hm.run_hmc(rng_key, rflag=rflag, extra_fields=('potential_energy',), save=outname, gpkernel=gpkernel, fix_lneps=lneps)

#%%
hm.summary_plots(xlabel, ylabel, rotflag=rotflag, save=outname, show_vals=True)

#%%
if model=='gp':
    keys = ["lnlenx", "lnleny", "lna"]
    labels = ["$\ln\rho_x$", "$\ln\rho_y$", "$\ln a$"]
else:
    keys = ["lneps"]
    labels = ["$\ln \epsilon$"]
#labels = [k.replace("_", "") for k in keys]
hyper = pd.DataFrame(data=dict(zip(keys, [hm.samples[k] for k in keys])))
fig = corner.corner(hyper, labels=labels, show_titles=".2f")
plt.savefig(outname+"_hyper.png", dpi=200, bbox_inches="tight")


#%%
xbins, ybins = hm.xbins, hm.ybins
true_dens, _, _ = np.histogram2d(hm.xtrue, hm.ytrue, bins=[xbins, ybins], density=True)
ptrue = np.where(hm.idx_valid_grid, true_dens.T, np.nan)
pgrid = hm.priors_grid
diff = pgrid - ptrue
diff_mean = np.mean(diff, axis=0)
diff_sigma = np.std(diff, axis=0)
diff_frac = diff_mean / diff_sigma
print (np.nanmean(diff_frac), np.nanstd(diff_frac))

#%%
if 'func0' in outname:
    vmin, vmax = 0, 0.3
elif 'func1' in outname:
    vmin, vmax = 0, 1.7
elif 'func2' in outname:
    vmin, vmax = 0, 0.85
elif 'func3' in outname:
    vmin, vmax = 0, 1.3
else:
    vmin, vmax = None, None

#%%
plt.figure(figsize=(14,7))
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.imshow(np.mean(hm.priors_grid, axis=0), extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]], aspect='auto', origin='lower', cmap=plt.cm.binary, alpha=0.8, vmin=vmin, vmax=vmax)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.colorbar(pad=0.02, label='probability density')
plt.title("mean prediction")
plt.savefig(outname+"_2dpred.png", dpi=200, bbox_inches="tight")

#%%
plt.figure(figsize=(14,7))
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.imshow(ptrue, extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]], aspect='auto', origin='lower',  cmap=plt.cm.binary, alpha=0.8, vmin=vmin, vmax=vmax)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.colorbar(pad=0.02, label='probability density')
plt.title("truth")
plt.savefig(outname+"_2dtruth.png", dpi=200, bbox_inches="tight")

#%%
plt.figure(figsize=(14,7))
plt.xlabel('mass ($M_\odot$)')
plt.ylabel('age (Gyr)')
#plt.imshow(diff_mean, extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]], aspect='auto', origin='lower',  cmap=plt.cm.binary, alpha=0.8)
plt.imshow(diff_mean/diff_sigma, extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]], aspect='auto', origin='lower',
#cmap=plt.cm.binary,
cmap='coolwarm',
alpha=0.8,
vmin=-3, vmax=3,
)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.colorbar(pad=0.02, label='$(p_\mathrm{pred}-p_\mathrm{truth})/\sigma_\mathrm{pred}$')
plt.title("difference: $%.3f \pm %.3f$"%(np.nanmean(diff_frac), np.nanstd(diff_frac)))
plt.savefig(outname+"_difffrac.png", dpi=200, bbox_inches="tight")

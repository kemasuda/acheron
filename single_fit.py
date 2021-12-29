#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import corner
from jaxstar.mistfit import MistFit
from jax import random
import sys

#%%
#input = "input/isoinput_hall.csv"
#outdir = "posteriors_hall_linage/"
input = sys.argv[1]
outdir = sys.argv[2]
imin = int(sys.argv[3])
imax = int(sys.argv[4])
agepar = sys.argv[5]
if agepar=='lin':
    linear_age = True
else:
    linear_age = False

#%%
d = pd.read_csv(input)

#%%
print (np.sum(d.kmag_err_corrected!=d.kmag_err_corrected))

#%%
mf = MistFit()

#%%
rng_key = random.PRNGKey(0)
for i in range(imin, imax):
    _d = d.iloc[i]
    kepid = int(_d.kepid)
    print ("KIC %s"%kepid)

    #try:
    kmag_obs, kmag_err, teff_obs, feh_obs, parallax_obs, parallax_err = np.array(_d[['kmag_corrected', 'kmag_err_corrected', 'teff', 'feh', 'parallax_corrected', 'parallax_error_corrected']]).astype(float)
    #except:
    #    continue
    teff_err, feh_err = 110, 0.1

    mf.set_data(['kmag', 'teff', 'feh', 'parallax'], [kmag_obs, teff_obs, feh_obs, parallax_obs], [kmag_err, teff_err, feh_err, parallax_err])

    n_sample = 20000
    mf.setup_hmc(num_warmup=n_sample, num_samples=n_sample)
    mf.run_hmc(rng_key, linear_age=linear_age, flat_age_marginal=False, nodata=False)
    samples = mf.samples

    samples.to_csv(outdir+"%s_samples.csv"%kepid, index=False)

    fig = corner.corner(samples[mf.obskeys+['mass', 'radius', 'age', 'eep']], show_titles='.2f', truths=mf.obsvals+[None]*4)
    fig.savefig(outdir+"%s_corner.png"%kepid, dpi=200, bbox_inches='tight', facecolor='white')

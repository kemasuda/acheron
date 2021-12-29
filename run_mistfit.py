#%%
import os
import pandas as pd

#%%
input = "input/isoinput_hall.csv"
agepar, outdir = "lin", "posteriors_hall_linage/"
agepar, outdir = "log", "posteriors_hall_logage/"

#%%
N = len(pd.read_csv(input))
m = int(N / 10) + 1
for j in range(m):
    imin, imax = j*10, j*10+10
    os.system("python single_fit.py %s %s %s %s %s"%(input, outdir, imin, imax, agepar))

#%%
import os
import pandas as pd

#%%
input = "input/isoinput_hall.csv"
agepar, outdir = "lin", "posteriors_hall_linage/"

#%%
input = "input/isoinput_hall.csv"
agepar, outdir = "log", "posteriors_hall_logage/"

#%%
input = "../input/isoinput_cks_valid.csv" # skip 3957082 (entry 912)
agepar, outdir = "log", "posteriors_cks_logage/"
#d=pd.read_csv(input)
#d.iloc[919].kepid # 6063220
#os.system("python single_fit.py %s %s %s %s %s"%(input, outdir, 919, 920, agepar))

#%%
input = "../input/isoinput_cks_valid.csv" # skip 3957082 (entry 912)
agepar, outdir = "lin", "posteriors_cks_linage/"

#%%
def run_all():
    N = len(pd.read_csv(input))
    m = int(N / 10) + 1
    for j in range(m):
        imin, imax = j*10, j*10+10
        if imax > N:
            imax = N
        print (imin, imax)
        os.system("python single_fit.py %s %s %s %s %s"%(input, outdir, imin, imax, agepar))

def run_first_half():
    N = len(pd.read_csv(input)) // 2
    m = int(N / 10) + 1
    for j in range(m):
        imin, imax = j*10, j*10+10
        if imax > N:
            imax = N
        print (imin, imax)
        os.system("python single_fit.py %s %s %s %s %s"%(input, outdir, imin, imax, agepar))

def run_second_half():
    N = len(pd.read_csv(input))
    m = int(N / 10) + 1
    for j in range(m):
        imin, imax = j*10, j*10+10
        imin += N // 2
        imax += N // 2
        if imax > N:
            imax = N
        if imin >= imax:
            break
        print (imin, imax)
        os.system("python single_fit.py %s %s %s %s %s"%(input, outdir, imin, imax, agepar))

#%%
#run_first_half()
run_second_half()

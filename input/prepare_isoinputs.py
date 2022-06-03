#%%
import pandas as pd
from jaxstar.utils import correct_gedr3_parallax, correct_kmag

#%%
filename = "cks/all_edr3_binflag.csv"
d = pd.read_csv(filename)
d = correct_gedr3_parallax(d)
d = correct_kmag(d)
d.to_csv("isoinput_cks.csv", index=False)

#%%
filename = "hall/all_edr3_binflag.csv"
d = pd.read_csv(filename)
d = correct_gedr3_parallax(d)
d = correct_kmag(d)
d.to_csv("isoinput_hall.csv", index=False)

#%%
d = pd.read_csv("isoinput_cks.csv")
d = d[d.kepid!=3957082].reset_index(drop=True)
len(d)

#%% get rotation info
import numpy as np
drot = pd.read_csv("mazeh+15.tsv", delimiter="|", comment="#")
drot["kepid"] = drot.KIC
prots, vars, lphs = [], [], []
for i in range(len(drot)):
    try:
        prots.append(float(drot.Prot[i]))
    except:
        prots.append(np.nan)
    try:
        vars.append(float(drot.Rvar[i]))
    except:
        vars.append(np.nan)
    try:
        lphs.append(float(drot.LPH[i]))
    except:
        lphs.append(np.nan)
drot['Prot'] = prots
drot['Rvar'] = vars
drot['LPH'] = lphs

idxm15 = (drot.F==0)&(drot.G==0)&(drot['T']==0) # False positives, Giant, Temperature
idxm15 &= (drot.C==0) & (drot.kepid!=8043882)   # written after the ACF search but Rvar/Prot were not assigined
drotm15 = drot[idxm15].reset_index(drop=True)
drot['acf'] = (drot.M1==0)&(drot.M2==0)&(drot.R==0) # robust period
print ('ACF sample: %d'%len(drotm15))
print ('Prot assigned: %d'%np.sum(drotm15.Prot==drotm15.Prot))

#%%
dmerged = pd.merge(d, drot[['kepid', 'acf', 'Prot', 'e_Prot', 'Rvar', 'LPH', 'w', 'D', 'N', 'C', 'G', 'T', 'F', 'R', 'M1', 'M2']], how='left', on='kepid')

#%% 
_dmerged = pd.merge(d, drot[['kepid', 'acf', 'Prot', 'e_Prot', 'Rvar', 'LPH', 'w', 'D', 'N', 'C', 'G', 'T', 'F', 'R', 'M1', 'M2']], on='kepid')
print (len(_dmerged))
print (np.sum(_dmerged.acf))

#%%
dmerged.reset_index(drop=True).to_csv("isoinput_cks_valid.csv", index=False)

#%%
len(dmerged)
dmerged[dmerged.kepid==5369827].Prot

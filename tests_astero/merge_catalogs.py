#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% import seismic parameters
dast1 = pd.read_csv("sa17_table4.tsv", delimiter='|', comment='#') # LEGACY
dast1 = dast1[dast1.Pipe == 'BASTA '].reset_index(drop=True)
print (len(dast1))
dast1['planet'] = False
dast2 = pd.read_csv("sa15_table3.tsv", delimiter='|', comment='#') # Kegas
dast2 = dast2[~dast2.KIC.isin(dast1.KIC)].reset_index(drop=True) # remove those in LEGACY
dast2['planet'] = True
print (len(dast2))

#%%
dast = pd.concat([dast1, dast2]).reset_index(drop=True)
dast['kepid'] = dast.KIC

#%%
dast['logage'] = np.log10(dast.Age*1e9)
dast['logage_upp'] = np.log10((dast.Age+dast.E_Age)*1e9) - dast.logage
dast['logage_low'] = dast.logage - np.log10((dast.Age-dast.e_Age)*1e9)

#%%
idx = dast.planet
plt.figure(figsize=(14,7))
plt.plot(dast.Mass[~idx], dast.Age[~idx], 'o', mfc='none', label='no planet')
plt.plot(dast.Mass[idx], dast.Age[idx], 'o', color='C0', label='planet')
plt.legend()

#%%
dast.to_csv("dast.csv", index=False)

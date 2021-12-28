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

#%%
import pandas as pd
#%%
tiling_file = 'S2_TilingSystem2-1.txt'
tdf = pd.read_csv(tiling_file,delim_whitespace =True)
tdf.shape
tiles = tdf['TilID'].tolist()


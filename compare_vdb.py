import sys
import numpy as np
import matplotlib.pyplot as plt
import datetime
import glob
import pandas as pd
import xarray as xr
sys.path.append('/media/callum/storage/Documents/Eureka/processing/staircase_experiment')
sys.path.append('/media/callum/storage/Documents/Eureka/processing/staircase_experiment/reimplement/')
from detect_staircases import classify_staircase

argo_ncs = glob.glob('/media/callum/storage/Documents/Eureka/processing/staircase_experiment/argo*.nc')

ds_sub = []
for nc in argo_ncs:
    ds_curr = xr.open_dataset(nc)
    if 6901769 in ds_curr.FloatID:
        ds_sub.append(ds_curr)

ds = ds_sub[0]
# From vanderBoog paper, the dive we want is at index 2531 in this dataset:
# Argo float 6901769 at 8.9 ◦ E and 37.9 ◦ N on 31 October 2017.
ds.lat[2531]
ds.lon[2531]
datetime.date(1950, 1, 1) + datetime.timedelta(days=ds.juld[2531].values.item())
dive_id = 2531
ds_argo_target = ds[dict(Nobs=[2512])]

demo_argo_data = pd.DataFrame({'pressure': ds.pressure,
                               'conservative_temperature': ds.ct[dive_id, :],
                               'absolute_salinity': ds.sa[dive_id, :]})
demo_argo_data.to_csv('vanderboog_argo_demo_data.csv', index=False)
df_callum_argo, mixes_callum_argo, grads_callum_argo = classify_staircase(demo_argo_data.pressure,
                                                                          demo_argo_data.conservative_temperature,
                                                                          demo_argo_data.absolute_salinity)

df_callum_argo_mod, mixes_callum_argo_mod, grads_callum_argo_mod = classify_staircase(demo_argo_data.pressure,
                                                                          demo_argo_data.conservative_temperature,
                                                                          demo_argo_data.absolute_salinity,
                                                                          layer_height_ratio=0.9,
                                                                          ml_density_difference=0.0048)

# Extract data for the Ice tethered profiler ITP64 at 137.8 ◦ W and 75.2 ◦ N on 29 January 2013
itp_nc = xr.open_dataset('/media/callum/storage/Documents/Eureka/processing/staircase_experiment/itp_00000060_00000069.nc')


fig, ax = plt.subplots()
offset=0.2
ax.plot(ds.ct[dive_id, :], ds.pressure, color='gray', alpha=0.3)
ax.plot(ds.ct[dive_id,:] * ds.mask_ml_sf[dive_id,:] / ds.mask_ml_sf[dive_id,:], ds.pressure, color='C0')
ax.plot(ds.ct[dive_id,:] * ds.mask_gl_sf[dive_id,:] / ds.mask_gl_sf[dive_id,:], ds.pressure, color='C1')
ax.plot(df_callum_argo.ct + offset, df_callum_argo.p, color='gray', alpha=0.3)
ax.plot(df_callum_argo.ct + offset*2, df_callum_argo.p, color='gray', alpha=0.3)
ax.plot(np.ma.array(df_callum_argo.ct, mask=df_callum_argo['mixed_layer_final_mask']) + offset, df_callum_argo.p, color='C0')
ax.plot(np.ma.array(df_callum_argo.ct, mask=df_callum_argo['gradient_layer_final_mask']) + offset, df_callum_argo.p, color='C1')
ax.plot(np.ma.array(df_callum_argo_mod.ct, mask=df_callum_argo_mod['mixed_layer_final_mask']) + offset*2, df_callum_argo_mod.p, color='C0')
ax.plot(np.ma.array(df_callum_argo_mod.ct, mask=df_callum_argo_mod['gradient_layer_final_mask']) + offset*2, df_callum_argo_mod.p, color='C1')
ax.set(xlim=(13, 15), ylim=(100, 900), xlabel='Offset conservative temperature ($^{\circ}$C)', ylabel='Pressure (dbar)')
ax.invert_yaxis()
fig.savefig('vdb_argo_comparison.png')



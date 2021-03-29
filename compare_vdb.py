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

demo_argo_data = pd.DataFrame({'pressure': ds.pressure,
                               'conservative_temperature': ds.ct[dive_id, :],
                               'absolute_salinity': ds.sa[dive_id, :]})
demo_argo_data.to_csv('vanderboog_argo_demo_data.csv', index=False)
df, mixes, grads = classify_staircase(ds.pressure, ds.ct[dive_id,:], ds.sa[dive_id, :])

fig, ax = plt.subplots()
offset=0.1
ax.plot(ds.ct[dive_id, :], ds.pressure, color='gray', alpha=0.3)
ax.plot(ds.ct[dive_id,:] * ds.mask_ml_sf[dive_id,:] / ds.mask_ml_sf[dive_id,:], ds.pressure, color='C0')
ax.plot(ds.ct[dive_id,:] * ds.mask_gl_sf[dive_id,:] / ds.mask_gl_sf[dive_id,:], ds.pressure, color='C1')
ax.plot(df.ct + offset, df.p, color='gray', alpha=0.3)
ax.plot(np.ma.array(df.ct, mask=df['mixed_layer_final_mask'])+offset, df.p, color='C0')
ax.plot(np.ma.array(df.ct, mask=df['gradient_layer_final_mask'])+offset, df.p, color='C1')
ax.set(xlim=(13, 15), ylim=(400,900))
ax.invert_yaxis()
plt.show()

fig, ax = plt.subplots()
offset=0.1
ax.plot(ds.sa[dive_id, :], ds.pressure, color='gray', alpha=0.3)
ax.plot(ds.sa[dive_id,:] * ds.mask_ml_sf[dive_id,:] / ds.mask_ml_sf[dive_id,:], ds.pressure, color='C0')
ax.plot(ds.sa[dive_id,:] * ds.mask_gl_sf[dive_id,:] / ds.mask_gl_sf[dive_id,:], ds.pressure, color='C1')
ax.plot(df.sa + offset, df.p, color='gray', alpha=0.3)
ax.plot(np.ma.array(df.sa, mask=df['mixed_layer_final_mask'])+offset, df.p, color='C0')
ax.plot(np.ma.array(df.sa, mask=df['gradient_layer_final_mask'])+offset, df.p, color='C1')
ax.set(xlim=(38.6, 39.1), ylim=(400,900))
ax.invert_yaxis()
plt.show()

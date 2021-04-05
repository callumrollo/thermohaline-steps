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

df_glider = pd.read_csv('/media/callum/storage/Documents/Eureka/processing/staircase_experiment/reimplement/glider_1db.csv')

for dive_no in np.arange(1011, 1016):
    df_sel = df_glider[df_glider.dive_limb_ident == dive_no]

    for i, dive in enumerate(df_sel.dive_limb_ident.unique()):
        df_dive = df_sel.loc[df_sel.dive_limb_ident == dive, :].copy()
        new_index = pd.Index(np.arange(0, df_dive.pressure_1db.max(), 1), name="pressure_1db")
        df_1db = df_dive.set_index("pressure_1db").reindex(new_index).reset_index()
        df, df_ml_stats, df_gl_stats = classify_staircase(df_1db.pressure_1db, df_1db.cons_temp_despike, df_1db.abs_salinity_despike,
                                                        ml_grad=0.001, ml_density_difference=0.01, av_window=200,
                                                        interface_max_height=30, temp_flag_only=True, show_steps=True)




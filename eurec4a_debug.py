import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
sys.path.append('/media/callum/storage/Documents/Eureka/processing/staircase_experiment')
sys.path.append('/media/callum/storage/Documents/Eureka/processing/staircase_experiment/reimplement/')
from detect_staircases import classify_staircase

df_glider = pd.read_csv('/media/callum/storage/Documents/Eureka/processing/staircase_experiment/reimplement/glider_1db.csv')
figa, axa = plt.subplots(figsize=(20,10))
offset = 0
for dive_no in np.arange(1011, 1026):
    df_dive = df_glider[df_glider.dive_limb_ident == dive_no].copy()
    new_index = pd.Index(np.arange(0, df_dive.pressure_1db.max(), 1), name="pressure_1db")
    df_1db = df_dive.set_index("pressure_1db").reindex(new_index).reset_index()
    df, df_ml_stats, df_gl_stats = classify_staircase(df_1db.pressure_1db, df_1db.cons_temp_despike, df_1db.abs_salinity_despike,
                                                    ml_grad=0.001, ml_density_difference=0.01, av_window=100,
                                                    interface_max_height=30, temp_flag_only=True, show_steps=False)
    axa.plot(df_1db.cons_temp_despike + offset, df_1db.pressure_1db, color='gray', alpha=0.5)
    axa.plot(np.ma.array(df.ct, mask=df['mixed_layer_final_mask']) + offset,
             np.ma.array(df.p, mask=df['mixed_layer_final_mask']), color='C0')
    axa.plot(np.ma.array(df.ct, mask=df['gradient_layer_final_mask']) + offset,
             np.ma.array(df.p, mask=df['gradient_layer_final_mask']), color='C1')
    offset = offset + 1
axa.plot((-1,-1), (-1,-1), color='C0', label='Mixed layer')
axa.plot((-1,-1), (-1,-1), color='C1', label='Gradient layer')
axa.legend()
axa.set(ylabel='Pressure (dbar)', xlabel='Offset conservative temperature', ylim=(250, 600), xlim=(6, 25))
axa.invert_yaxis()
figa.show()
all_dives = df_glider.dive_limb_ident.unique()
melon_dives = all_dives[(all_dives>1000) & (all_dives<2000)]
df_out = pd.DataFrame()
offset = 0
for dive_no in melon_dives[10:-10]:
    df_dive = df_glider[df_glider.dive_limb_ident == dive_no].copy()
    new_index = pd.Index(np.arange(0, df_dive.pressure_1db.max(), 1), name="pressure_1db")
    df_1db = df_dive.set_index("pressure_1db").reindex(new_index).reset_index()
    df, df_ml_stats, df_gl_stats = classify_staircase(df_1db.pressure_1db, df_1db.cons_temp_despike, df_1db.abs_salinity_despike,
                                                    ml_grad=0.001, ml_density_difference=0.01, av_window=100,
                                                    interface_max_height=30, temp_flag_only=True, show_steps=False)
    df['dive'] = dive_no
    df_out = df_out.append(df)

fig0, ax0 = plt.subplots(figsize=(20,10))
offset = 0
for dive in df_out.dive.unique():
    df = df_out.loc[df_out.dive == dive, :]
    blank = np.ones(len(df.ct)) * dive
    ax0.plot(blank, np.ma.array(df.p, mask=df['mixed_layer_final_mask']), color='C0')
    #ax0.plot(blank, np.ma.array(df.p, mask=df['gradient_layer_final_mask']), color='C1')
    offset = offset + 1
fig0.show()



df_micro = pd.read_csv('/media/callum/storage/Documents/Eureka/processing/staircase_experiment/reimplement/micro_test.csv')
df, df_ml_stats, df_gl_stats = classify_staircase(df_micro.depth, df_micro.ct,
                                                  df_micro.sa,
                                                  ml_grad=0.001, ml_density_difference=0.01, av_window=100,
                                                  interface_max_height=30, temp_flag_only=True, show_steps=True)
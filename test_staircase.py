"""
To test:
Input
input data at varying pressure steps
reject/fix non uniform data
data with gaps
"""
import sys

sys.path.append('/media/callum/storage/Documents/Eureka/processing/staircase_experiment/reimplement/')
from detect_staircases import classify_staircase
import numpy as np
import matplotlib.pyplot as plt

p = np.arange(1000)
ct = np.linspace(20, 0, len(p))
sa = np.linspace(35, 33, len(p))
ct_orig = ct.copy()
sa_orig = sa.copy()
span = 40
for center in np.arange(5, 850, 100):
    ct[center - span:center + span] = np.mean(ct_orig[center - span:center + span])
    sa[center - span:center + span] = np.mean(sa_orig[center - span:center + span])


def test_ideal():
    # Ideal dataset should contain 8 mixed layers and 7 gradient layers
    df, mixes, grads = classify_staircase(p, ct, sa)
    assert len(mixes) == 8
    assert len(grads) == 7

def test_temp_flag_only():
    # With ideal input, temp flag only should return the same results
    df, mixes, grads = classify_staircase(p, ct, sa)
    df_t, mixes_t, grads_t = classify_staircase(p, ct, sa, temp_flag_only=True)
    assert (df.mixed_layer_final_mask == df_t.mixed_layer_final_mask).all()
    plotter(df_t, mixes_t, grads_t)


def plotter(df, mixes, grads):
    fig, ax = plt.subplots(1, 2, figsize=(18, 10), sharey='row')
    ax[0].plot(df.ct, df.p, color='gray', alpha=0.2)
    ax[0].plot(np.ma.array(df.ct, mask=df['mixed_layer_final_mask']),
               np.ma.array(df.p, mask=df['mixed_layer_final_mask']), color='C0')
    ax[0].plot(np.ma.array(df.ct, mask=df['gradient_layer_final_mask']),
               np.ma.array(df.p, mask=df['gradient_layer_final_mask']), color='C1')

    ax[1].plot(df.sa, df.p, color='gray', alpha=0.2)
    ax[1].plot(np.ma.array(df.sa, mask=df['mixed_layer_final_mask']),
               np.ma.array(df.p, mask=df['mixed_layer_final_mask']), color='C0')
    ax[1].plot(np.ma.array(df.sa, mask=df['gradient_layer_final_mask']),
               np.ma.array(df.p, mask=df['gradient_layer_final_mask']), color='C1')
    plt.show()

"""
To test:
Input
input data at varying pressure steps
reject/fix non uniform data
data with gaps
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
sys.path.append('/media/callum/storage/Documents/Eureka/processing/staircase_experiment/reimplement/')
from detect_staircases import classify_staircase


p = np.arange(1000)
ct = np.linspace(20, 0, len(p))
sa = np.linspace(35, 33, len(p))
ct_orig = ct.copy()
sa_orig = sa.copy()
span = 40
for center in np.arange(0, 850, 100):
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
    assert (mixes.p_end == mixes_t.p_end).all()
    assert (grads.p_start == grads_t.p_start).all()
    plotter(df_t, mixes_t, grads_t)


def test_spacing_long():
    p = np.arange(0, 1000, 2)
    ct = np.linspace(20, 0, len(p))
    sa = np.linspace(35, 33, len(p))
    ct_orig = ct.copy()
    sa_orig = sa.copy()
    span = 20
    for center in np.arange(0, 425, 50):
        ct[center - span:center + span] = np.mean(ct_orig[center - span:center + span])
        sa[center - span:center + span] = np.mean(sa_orig[center - span:center + span])
    df, mixes, grads = classify_staircase(p, ct, sa, av_window=50)
    plotter(df, mixes, grads)
    assert len(mixes) == 8
    assert len(grads) == 7


def test_spacing_close():
    p = np.arange(0, 1000, 0.1)
    ct = np.linspace(20, 0, len(p))
    sa = np.linspace(35, 33, len(p))
    ct_orig = ct.copy()
    sa_orig = sa.copy()
    span = 400
    for center in np.arange(0, 8500, 1000):
        ct[center - span:center + span] = np.mean(ct_orig[center - span:center + span])
        sa[center - span:center + span] = np.mean(sa_orig[center - span:center + span])
    df, mixes, grads = classify_staircase(p, ct, sa, av_window=1000)
    plotter(df, mixes, grads)
    assert len(mixes) == 8
    assert len(grads) == 7


def test_vanderboog_argo():
    # checking against the Argo data from vanderBoog paper
    vdb = pd.read_csv('vanderboog_argo_demo_data.csv')
    vdb = vdb.loc[:1000, :]
    df, mixes, grads = classify_staircase(vdb.pressure, vdb.conservative_temperature,
                                          vdb.absolute_salinity)
    mixes = mixes[~mixes.bad_mixed_layer]
    plotter(df, mixes, grads)
    assert mixes.p_start.min() > 400
    assert mixes.p_end.max() < 950
    assert len(mixes) >= 8
    assert len(mixes) <= 10


def plotter(df, mixes, grads):
    fig, ax = plt.subplots(1, 2, figsize=(18, 10), sharey='row')
    ax[0].plot(df.ct, df.p, color='gray', alpha=0.2)
    ax[0].plot(np.ma.array(df.ct, mask=df['mixed_layer_salt_finger_mask']),
               np.ma.array(df.p, mask=df['mixed_layer_salt_finger_mask']), color='C0')
    ax[0].plot(np.ma.array(df.ct, mask=df['gradient_layer_salt_finger_mask']),
               np.ma.array(df.p, mask=df['gradient_layer_salt_finger_mask']), color='C1')

    ax[1].plot(df.sa, df.p, color='gray', alpha=0.2)
    ax[1].plot(np.ma.array(df.sa, mask=df['mixed_layer_salt_finger_mask']),
               np.ma.array(df.p, mask=df['mixed_layer_salt_finger_mask']), color='C0')
    ax[1].plot(np.ma.array(df.sa, mask=df['gradient_layer_salt_finger_mask']),
               np.ma.array(df.p, mask=df['gradient_layer_salt_finger_mask']), color='C1')
    plt.show()

"""
Detection of thermohaline staircases in CTD data.
Original idea and default constants from Carine van der Boog (2020)
https://doi.org/10.5281/zenodo.4286170

Re-implementation and generalisation by Callum Rollo 2021
This re-implementation generalises to irregularly sampled data
This work is free to reuse under the terms of the GPL v3. See COPYING file or https://fsf.org
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from reimplement.staircase_functions import prep_data, identify_mixed_layers, mixed_layer_max_variability, \
    mixed_layer_stats, gradient_layer_stats, identify_staircase_sequence, filter_gradient_layers, \
    classify_salt_finger_diffusive_convective


def classify_staircase(p, ct, sa, ml_grad=0.0005, ml_density_difference=0.005, av_window=200, interface_max_height=30,
                       temp_flag_only=False, show_steps=False):
    """
    all data should be at 1 dbar resolution (for now)
    Notes:
    - Currently dropping min and max pressure values, so can have Turner angle at all points
    - Turner angle and stability-ratio from smoothed profile
    :param p: pressure (dbar)
    :param ct: conservative temperature (degrees celsius)
    :param sa: absolute salinity (g kg-1)
    :param ml_grad: density gradient for detection of mixed layer (kg m^-3 dbar^-1), default: 0.0005
    :param ml_density_difference: maximum density gradient difference of mixed layer (kg m^-3), default: 0.005
    :param av_window: averaging window to obtain background profiles (dbar), default: 200
    :param interface_max_height: Maximum allowed height of interfaces between mixed layers (dbar), default: 30
    :param temp_flag_only: bool, if True, will flag potential mixed layers only by temperature, default: False
    :return: dataframe at supplied pressure steps, dataframe of mixed layers, dataframe of gradient layers.
    """
    # TODO import test: are data evenly sampled in pressure? Are the monotonically increasing?
    """
    Step 0: Prepare data. Using pandas dataframes to keep neat
    """
    df, df_smooth, df_diff, pressure_step = prep_data(p, ct, sa, av_window)

    """
    Following the 5 steps described in https://essd.copernicus.org/articles/13/43/2021/
    """

    """
    Step 1 Detect extent of subsurface mixed layers (ml)
    """
    df, df_ml = identify_mixed_layers(df, df_smooth, df_diff, ml_grad, temp_flag_only)

    if show_steps:
        fig, ax = plt.subplots()
        offset_step = 0.5
        offset = 0
        ax = progress_plotter(ax, df.p, df.ct + offset, df.mixed_layer_temp_mask, label='0.5')
        offset += offset_step

    # If 1 mixed layer or less, bail out
    if len(df_ml) < 2:
        return df, None, None

    df_ml = mixed_layer_max_variability(df_ml, ml_density_difference, temp_flag_only)
    df, df_ml_stats = mixed_layer_stats(df, df_ml, pressure_step)

    if show_steps:
        ax = progress_plotter(ax, df.p, df.ct + offset, df.mixed_layer_step1_mask, label='Step 1')
        offset += offset_step
    """
    Step 2  Assess gradient/interface layers between mixed layers and calculate their properties
    """
    df, df_gl_stats = gradient_layer_stats(df, df_ml_stats)

    if show_steps:
        ax = progress_plotter(ax, df.p, df.ct + offset, df.grad_layer_step2_mask, grad=True, label='Step 2')
        offset += offset_step
    """
    Step 3 Exclude interfaces that are relatively thick, or do not have step shape
    """

    df, df_gl_stats = filter_gradient_layers(df, df_ml_stats, df_gl_stats, interface_max_height, temp_flag_only, pressure_step)

    if show_steps:
        ax = progress_plotter(ax, df.p, df.ct + offset, df.grad_layer_step3_mask, grad=True, label='Step 3')
        offset += offset_step
    """
    Step 4 Classify layers in the double diffusive regime as salt finger (sf) or diffusive convection (dc)
    """

    df, df_gl_stats = classify_salt_finger_diffusive_convective(df, df_ml_stats, df_gl_stats)

    if show_steps:
        ax = progress_plotter(ax, df.p, df.ct + offset, df.grad_layer_step4_mask, grad=True, label='Step 4')
        offset += offset_step
    """
    Step 5 Identify sequences of steps. A step is defined as a mixed layer bounded by two interfaces of matching double
    diffusive regime. This excludes most thermohaline intrusions
    """

    df, df_ml_stats, df_gl_stats = identify_staircase_sequence(df, df_ml_stats, df_gl_stats, pressure_step)

    if show_steps:
        ax = progress_plotter(ax, df.p, df.ct + offset, df.gradient_layer_final_mask, grad=True, label='Step 5')
        offset += offset_step
        ax = progress_plotter(ax, df.p, df.ct + offset, df.mixed_layer_final_mask, label='Step 5')
        ax.set(xlabel='Offset conservative temperature (C)', ylabel='Pressure (dbar)',
               #    ylim=(100, 1000), xlim=(12.5, 18))
               ylim=(250, 600), xlim=(6, 15))
        ax.invert_yaxis()
        plt.show()

    return df, df_ml_stats, df_gl_stats


def progress_plotter(ax, p, ct, mask, label='', grad=False):
    if grad:
        line_color = 'C1'
    else:
        line_color = 'C0'
    ax.plot(ct, p, color='gray', alpha=0.3)
    ax.plot(np.ma.array(ct, mask=mask),
            np.ma.array(p, mask=mask), color=line_color)
    ax.text(ct.mean() + 0.8, 90, label, rotation=45)
    return ax


def plotter(df):
    fig, ax = plt.subplots(1, 2, figsize=(18, 10), sharey='row')
    ax[0].plot(df.ct, df.p, color='gray', alpha=0.3)
    ax[0].plot(np.ma.array(df.ct, mask=df['mixed_layer_final_mask']),
               np.ma.array(df.p, mask=df['mixed_layer_final_mask']), color='C0')
    ax[0].plot(np.ma.array(df.ct, mask=df['gradient_layer_final_mask']),
               np.ma.array(df.p, mask=df['gradient_layer_final_mask']), color='C1')

    ax[1].plot(df.sa, df.p, color='gray', alpha=0.3)
    ax[1].plot(np.ma.array(df.sa, mask=df['mixed_layer_final_mask']),
               np.ma.array(df.p, mask=df['mixed_layer_final_mask']), color='C0')
    ax[1].plot(np.ma.array(df.sa, mask=df['gradient_layer_final_mask']),
               np.ma.array(df.p, mask=df['gradient_layer_final_mask']), color='C1')
    plt.show()


if __name__ == '__main__':
    p_in = np.arange(1000)
    ct_in = np.linspace(20, 0, len(p_in))
    sa_in = np.linspace(35, 33, len(p_in))
    ct_orig = ct_in.copy()
    sa_orig = sa_in.copy()
    span = 40
    for center in np.arange(100, 1000, 100):
        ct_in[center - span:center + span] = np.mean(ct_orig[center - span:center + span])
        sa_in[center - span:center + span] = np.mean(sa_orig[center - span:center + span])
    df_in = pd.read_csv(
        '/media/callum/storage/Documents/Eureka/processing/staircase_experiment/vanderboog_argo_demo_data.csv')
    df_out, mixes, grads = classify_staircase(df_in.pressure, df_in.conservative_temperature, df_in.absolute_salinity,
                                              temp_flag_only=True, show_steps=True)
    # df, mixes, grads = classify_staircase(p_in, ct_in, sa_in)
    # plotter(df)

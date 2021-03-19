"""
Detection of thermohaline staircases in CTD data.
Original idea and default constants from Carine van der Boog (2020)
https://doi.org/10.5281/zenodo.4286170

Re-implementation and generalisation by Callum Rollo 2021
This re-implementation generalises to irregularly sampled data
This work is free to reuse under the terms of the GPL v3. See COPYING file or https://fsf.org
"""
import gsw as gsw
import pandas as pd
import numpy as np


def center_diff_df(df):
    return (df.diff() - df.diff(periods=-1)) / 2


def add_ml_stats_row(stats_df, df):
    df_min = df.min()
    df_max = df.max()
    df_mean = df.mean()
    ml_row = {'p_start': df_min.p, 'p_end': df_max.p, 'ct': df_mean.ct, 'sa': df_mean.sa, 'sigma1': df_mean.sigma1,
              'p': df_mean.p, 'ct_range': df_max.ct - df_min.ct, 'sa_range': df_max.sa - df_min.sa,
              'sigma1_range': df_max.sigma1 - df_min.sigma1,
              'pressure_extent': df_max.p - df_min.p, 'turner_ang': df_mean.turner_ang,
              'stability_ratio': df_mean.stability_ratio}
    stats_df = stats_df.append(ml_row, ignore_index=True)
    return stats_df


def classify_staircase(p, ct, sa, ml_grad=0.005, ml_density_difference=0.05, av_window=200, interface_max_height=30):
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
    :return:
    """
    # TODO import test: are data evenly sampled in pressure? Are the monotonically increasing?
    """
    Step 0: Prepare data. Using pandas dataframes to keep neat
    """
    df_input = pd.DataFrame(data={'p': p, 'ct': ct, 'sa': sa})
    df_input['sigma1'] = gsw.sigma1(df_input.sa, df_input.ct)
    # Interpolate linearly over nans in input data
    df_input = df_input.interpolate(method='linear')

    # Take rolling average of the dataframe, using av_window
    df_smooth_input = df_input.rolling(window=av_window, center=True).mean()
    df_smooth_input['alpha'] = gsw.alpha(df_smooth_input.sa, df_smooth_input.ct, df_smooth_input.p)
    df_smooth_input['beta'] = gsw.beta(df_smooth_input.sa, df_smooth_input.ct, df_smooth_input.p)
    # df_smooth = df_smooth.reindex()
    df_smooth_midpoints = df_smooth_input.rolling(window=2, center=True).mean()[1:]
    df = df_input.iloc[1:-1].reindex()
    df_smooth = df_smooth_input.iloc[1:-1].reindex()
    df['turner_ang'], df['stability_ratio'], __ = gsw.stability.Turner_Rsubrho(df_smooth_midpoints.sa,
                                                                               df_smooth_midpoints.ct,
                                                                               df_smooth_midpoints.p, axis=0)
    # Create masks of mixed layers and gradient layers at the levels of the input data
    df['mixed_layer_final_mask'] = True
    # Take the center diff of the dataframe wrt pressure
    df_diff = center_diff_df(df)
    pressure_step = df_diff.p.mean()
    """
    Following the 5 steps described in https://essd.copernicus.org/articles/13/43/2021/
    """

    """
    Step 1 Detect extent of subsurface mixed layers (ml)
    """
    df['mixed_layer_temp_mask'] = True
    df.loc[np.abs(df_smooth.alpha * df_diff.ct * 1028) < ml_grad, 'mixed_layer_temp_mask'] = False
    df['mixed_layer_sal_mask'] = True
    df.loc[np.abs(df_smooth.beta * df_diff.sa * 1028) < ml_grad, 'mixed_layer_sal_mask'] = False
    df['mixed_layer_sigma_mask'] = True
    df.loc[np.abs(df_diff.sigma1) < ml_grad, 'mixed_layer_sigma_mask'] = False
    # Combine logical masks for a mask. Mixed layers must satisfy criteria for temperature, salinity and density
    df['mixed_layer_mask'] = df.mixed_layer_temp_mask | df.mixed_layer_sal_mask | df.mixed_layer_sigma_mask
    df['gradient_layer_final_mask'] = True

    # Create dataframe of only points within mixed layers
    df_ml = df[~df.mixed_layer_mask]

    # If 1 mixed layer or less, bail out
    if len(df_ml) < 2:
        return df, None, None
    # Create a dataframe for mixed layer stats. Each row will match a mixed layer
    df_ml_stats = pd.DataFrame(
        columns=['p_start', 'p_end', 'ct', 'sa', 'sigma1', 'p', 'ct_range', 'sa_range', 'sigma1_range',
                 'pressure_extent',
                 'turner_ang', 'stability_ratio'])
    # Loop through rows of mixed layer points, identifying individual layers
    start_index = df_ml.index[0]
    prev_index = df_ml.index[0]
    continuous_ml = False
    for i, row in df_ml.iloc[1:].iterrows():
        if i == prev_index + 1:
            if not continuous_ml:
                # Current row is adjacent to previous row: start of continuous mixed layer
                start_index = prev_index
                continuous_ml = True
        else:
            if continuous_ml:
                # End of continuous mixed layer, add to table
                df_mixed_layer = df_ml.loc[start_index:prev_index]
                df_ml_stats = add_ml_stats_row(df_ml_stats, df_mixed_layer)
            else:
                # Single sample mixed layer, add to table
                df_mixed_layer = df_ml.loc[prev_index]
                ml_row = {'p_start': df_mixed_layer.p, 'p_end': df_mixed_layer.p, 'ct': df_mixed_layer.ct,
                          'sa': df_mixed_layer.sa, 'sigma1': df_mixed_layer.sigma1, 'p': df_mixed_layer.p,
                          'ct_range': 0, 'sa_range': 0, 'sigma1_range': 0, 'pressure_extent': pressure_step,
                          'turner_ang': df_mixed_layer.turner_ang, 'stability_ratio': df_mixed_layer.stability_ratio}
                df_ml_stats = df_ml_stats.append(ml_row, ignore_index=True)
            continuous_ml = False
        prev_index = i

    # Drop mixed layers with density difference exceeding ml_density_difference TODO better value here
    df_ml_stats = df_ml_stats[df_ml_stats.sigma1_range < ml_density_difference]
    df_ml_stats = df_ml_stats.reset_index()
    """
    Step 2  Assess gradient/interface layers between mixed layers and calculate their properties
    """
    # Create empty dataframe with same columns and df_ml_stats
    df_gl_stats = pd.DataFrame(columns=df_ml_stats.columns)
    prev_row = df_ml_stats.iloc[0]
    # Loop through mixed layer stats. All layers between mixed layers initially classified as gradient layers
    for i, row in df_ml_stats.iloc[1:].iterrows():
        gl_rows = df[(df.p > prev_row.p_end) & (df.p < row.p_start)]
        df_gl_stats = add_ml_stats_row(df_gl_stats, gl_rows)
        prev_row = row

    """
    Step 3 Exclude interfaces that are relatively thick, or do not have step shape
    """
    # Exclude gradient layers that are thicker than the max height, or thicker than adjacent mixed layers
    df_gl_stats['adj_ml_height'] = np.maximum.reduce(
        [df_ml_stats.iloc[1:].pressure_extent.values, df_ml_stats.iloc[:-1].pressure_extent.values])
    df_gl_stats['height_ratio'] = df_gl_stats.adj_ml_height / df_gl_stats.pressure_extent
    df_gl_stats.loc[df_gl_stats.pressure_extent > df_gl_stats.adj_ml_height, 'p_start'] = np.nan
    df_gl_stats.loc[df_gl_stats.pressure_extent > interface_max_height, 'p_start'] = np.nan

    # TODO remove interfaces with temp or salinity inversions
    # TODO Exclude mixed layers with greater variability in temperature, salinity or density than adjacent gradient layers

    """
    Step 4 Classify layers in the double diffusive regime as salt finger (sf) or diffusive convection (dc)
    """
    df_gl_stats['salt_finger'] = False
    df_gl_stats['diffusive_convection'] = False
    prev_ml_row = df_ml_stats.iloc[0]
    for i, ml_row in df_ml_stats[1:-1].iterrows():
        if (prev_ml_row.ct - ml_row.ct) / (prev_ml_row.p - ml_row.p) < 0 \
                and (prev_ml_row.sa - ml_row.sa) / (prev_ml_row.p - ml_row.p) < 0:
            df_gl_stats.loc[i - 1, 'salt_finger'] = True
        if (prev_ml_row.ct - ml_row.ct) / (prev_ml_row.p - ml_row.p) > 0 \
                and (prev_ml_row.sa - ml_row.sa) / (prev_ml_row.p - ml_row.p) > 0:
            df_gl_stats.loc[i - 1, 'diffusive_convection'] = True

    """
    Step 5 Identify sequences of steps. A step is defined as a mixed layer bounded by two interfaces of matching double
    diffusive regime. This excludes most thermohaline intrusions
    """
    df_ml_stats['salt_finger_step'] = False
    df_ml_stats['diffusive_convection_step'] = False

    df_gl_stats['salt_finger_step'] = False
    df_gl_stats['diffusive_convection_step'] = False

    for i, ml_row in df_ml_stats[1:-1].iterrows():
        prev_gl_row = df_gl_stats.loc[i - 1]
        next_gl_row = df_gl_stats.loc[i]
        if prev_gl_row.salt_finger & next_gl_row.salt_finger:
            df_ml_stats.loc[i, 'salt_finger_step'] = True
            df_gl_stats.loc[i - 1:i, 'salt_finger_step'] = True
        if prev_gl_row.diffusive_convection & next_gl_row.diffusive_convection:
            df_ml_stats.loc[i, 'diffusive_convection_step'] = True
            df_gl_stats.loc[i - 1:i, 'diffusive_convection_step'] = True

    # Drop interfaces with turner angles that do not match their double diffusive regime
    df_gl_stats = df_gl_stats.loc[~((df_gl_stats['turner_ang'] > 90) & (df_gl_stats['salt_finger'])), :]
    df_gl_stats = df_gl_stats.loc[~((df_gl_stats['turner_ang'] < 45) & (df_gl_stats['salt_finger'])), :]
    df_gl_stats = df_gl_stats.loc[~((df_gl_stats['turner_ang'] > -45) & (df_gl_stats['diffusive_convection'])), :]
    df_gl_stats = df_gl_stats.loc[~((df_gl_stats['turner_ang'] < -90) & (df_gl_stats['diffusive_convection'])), :]
    # temporary tests of functionality

    # Populate masks of mixed and gradient layers before returning dataframe
    for i, row in df_ml_stats[df_ml_stats['salt_finger_step']].iterrows():
        df.loc[row.p_start:row.p_end, 'mixed_layer_final_mask'] = False

    for i, row in df_gl_stats[df_gl_stats['salt_finger_step']].iterrows():
        df.loc[row.p_start:row.p_end, 'gradient_layer_final_mask'] = False

    return df, df_ml_stats, df_gl_stats

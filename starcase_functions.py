import gsw as gsw
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


def center_diff_df(df, pressure_step):
    """
    Performs central difference on a data frame. First and last rows are lost in this procedure
    :param df: Dataframe of evenly spaced observations
    :param pressure_step: Pressure difference between beach row in df
    :return: Central difference of each row from its neighbours
    """
    return (df.diff() - df.diff(periods=-1)) / (2 * pressure_step)


def add_layer_stats_row(stats_df, df_layer):
    """
    Adds a row of stats from a layer to a dataframe of stats
    :param stats_df: Dataframe with one row per layer
    :param df_layer: Dataframe of observations within a single identified layer
    :return: stats_df with a new row made from the contents of df_layer
    """
    df_min = df_layer.min()
    df_max = df_layer.max()
    df_mean = df_layer.mean()
    ml_row = {'p_start': df_min.p, 'p_end': df_max.p, 'ct': df_mean.ct, 'sa': df_mean.sa, 'sigma1': df_mean.sigma1,
              'p': df_mean.p, 'ct_range': df_max.ct - df_min.ct, 'sa_range': df_max.sa - df_min.sa,
              'sigma1_range': df_max.sigma1 - df_min.sigma1,
              'layer_height': df_max.p - df_min.p, 'turner_ang': df_mean.turner_ang,
              'density_ratio': df_mean.density_ratio}
    stats_df = stats_df.append(ml_row, ignore_index=True)
    return stats_df


def prep_data(p, ct, sa, av_window):
    df_input = pd.DataFrame(data={'p': p, 'ct': ct, 'sa': sa})
    df_input['sigma1'] = gsw.sigma1(df_input.sa, df_input.ct)
    # Interpolate linearly over nans in input data
    df_input = df_input.interpolate(method='linear')
    # Take rolling average of the dataframe, using av_window
    df_smooth_input = df_input.rolling(window=av_window, center=True).mean()
    df_smooth_input['alpha'] = gsw.alpha(df_smooth_input.sa, df_smooth_input.ct, df_smooth_input.p)
    df_smooth_input['beta'] = gsw.beta(df_smooth_input.sa, df_smooth_input.ct, df_smooth_input.p)
    # Turner angle calculated from a 50 m smoothing, following van der Boog 
    df_smooth_turner = df_input.rolling(window=50, center=True).mean()
    df_smooth_midpoints = df_smooth_turner.rolling(window=2, center=True).mean()[1:]
    # Drop first and last points, as they have no corresponding turner angle
    df = df_input.iloc[1:-1].reindex()
    df_smooth = df_smooth_input.iloc[1:-1].reindex()
    df['turner_ang'], df['density_ratio'], __ = gsw.stability.Turner_Rsubrho(df_smooth_midpoints.sa,
                                                                             df_smooth_midpoints.ct,
                                                                             df_smooth_midpoints.p, axis=0)
    # Create masks of mixed layers and gradient layers at the levels of the input data
    df['mixed_layer_final_mask'] = True
    # Take the center diff of the dataframe wrt pressure
    pressure_step = df.iloc[1].p - df.iloc[0].p
    df_diff = center_diff_df(df, pressure_step)
    return df, df_smooth, df_diff, pressure_step


def identify_mixed_layers(df, df_smooth, df_diff, ml_grad, temp_flag_only):
    df['mixed_layer_temp_mask'] = True
    df.loc[np.abs(df_smooth.alpha * df_diff.ct * 1028) < ml_grad, 'mixed_layer_temp_mask'] = False
    df['mixed_layer_sal_mask'] = True
    df.loc[np.abs(df_smooth.beta * df_diff.sa * 1028) < ml_grad, 'mixed_layer_sal_mask'] = False
    df['mixed_layer_sigma_mask'] = True
    df.loc[np.abs(df_diff.sigma1) < ml_grad, 'mixed_layer_sigma_mask'] = False
    # Combine logical masks for a mask. Mixed layers must satisfy criteria for temperature, salinity and density
    if temp_flag_only:
        df['mixed_layer_mask'] = df['mixed_layer_temp_mask']
    else:
        df['mixed_layer_mask'] = df.mixed_layer_temp_mask | df.mixed_layer_sal_mask | df.mixed_layer_sigma_mask
    df['gradient_layer_final_mask'] = True
    # Create dataframe of only points within mixed layers
    df_ml = df[~df.mixed_layer_mask]
    return df, df_ml


def mixed_layer_max_variability(df_ml, ml_density_difference):
    # TODO apply temp only method here
    # Loop through mixed layers, making breaks where the mixed layer exceeds maximum allowed density variation
    # Differing from van der Boog et al., this removes only the last point of a mixed layer, not first and last
    new_layer = True
    start_pden = df_ml.iloc[0].sigma1
    prev_index = 0
    breaks = []
    for i, row in df_ml.iterrows():
        if i == prev_index + 1:
            if new_layer:
                start_pden = df_ml.loc[prev_index, 'sigma1']
                new_layer = False
            if np.abs(row.sigma1 - start_pden) > ml_density_difference:
                if i - 1 not in breaks:
                    breaks.append(i)
                new_layer = True
        else:
            new_layer = True
        prev_index = i
    df_ml = df_ml.drop(breaks)
    return df_ml


def mixed_layer_stats(df, df_ml, pressure_step):
    # Create a dataframe for mixed layer stats. Each row will match a mixed layer
    df_ml_stats = pd.DataFrame(
        columns=['p_start', 'p_end', 'ct', 'sa', 'sigma1', 'p', 'ct_range', 'sa_range', 'sigma1_range',
                 'layer_height',
                 'turner_ang', 'density_ratio'])
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
                df_ml_stats = add_layer_stats_row(df_ml_stats, df_mixed_layer)
            else:
                # Single sample mixed layer, add to table
                df_mixed_layer = df_ml.loc[prev_index]
                ml_row = {'p_start': df_mixed_layer.p, 'p_end': df_mixed_layer.p, 'ct': df_mixed_layer.ct,
                          'sa': df_mixed_layer.sa, 'sigma1': df_mixed_layer.sigma1, 'p': df_mixed_layer.p,
                          'ct_range': 0, 'sa_range': 0, 'sigma1_range': 0, 'layer_height': pressure_step,
                          'turner_ang': df_mixed_layer.turner_ang, 'density_ratio': df_mixed_layer.density_ratio}
                df_ml_stats = df_ml_stats.append(ml_row, ignore_index=True)
            continuous_ml = False
        prev_index = i
    # Add final mixed layer
    df_mixed_layer = df_ml.loc[start_index:prev_index]
    df_ml_stats = add_layer_stats_row(df_ml_stats, df_mixed_layer)

    df['mixed_layer_step1_mask'] = True
    for i, row in df_ml_stats.iterrows():
        df.loc[row.p_start:row.p_end, 'mixed_layer_step1_mask'] = False
    return df, df_ml_stats


def gradient_layer_stats(df, df_ml_stats):
    # Create empty dataframe with same columns and df_ml_stats
    df_gl_stats = pd.DataFrame(columns=df_ml_stats.columns)
    if len(df_ml_stats) < 2:
        return df, df_ml_stats, df_gl_stats
    prev_row = df_ml_stats.iloc[0]
    # Loop through mixed layer stats. All layers between mixed layers initially classified as gradient layers
    for i, row in df_ml_stats.iloc[1:].iterrows():
        gl_rows = df[(df.p > prev_row.p_end) & (df.p < row.p_start)]
        df_gl_stats = add_layer_stats_row(df_gl_stats, gl_rows)
        prev_row = row

    # Exclude gradient layers with lesser variability in temp, salinity or density than adjacent mixed layers
    df_gl_stats['bad_grad_layer'] = False
    for property_range in ['ct_range', 'sa_range', 'sigma1_range']:
        df_gl_stats.loc[
            df_ml_stats.iloc[1:][property_range].values > df_gl_stats[property_range].values, 'bad_grad_layer'] = True
        df_gl_stats.loc[
            df_ml_stats.iloc[:-1][property_range].values > df_gl_stats[property_range].values, 'bad_grad_layer'] = True
    df['grad_layer_step2_mask'] = True
    for i, row in df_gl_stats[~df_gl_stats['bad_grad_layer']].iterrows():
        df.loc[row.p_start:row.p_end, 'grad_layer_step2_mask'] = False
    return df, df_gl_stats


def identify_staircase_sequence(df, df_ml_stats, df_gl_stats):
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
    # Class first and last mixed layers under both regimes, will be classified by Turner angle
    df_ml_stats.loc[i + 1, 'salt_finger_step'] = True
    df_ml_stats.loc[0, 'salt_finger_step'] = True
    df_ml_stats.loc[i + 1, 'diffusive_convection'] = True
    df_ml_stats.loc[0, 'diffusive_convection'] = True
    # Drop interfaces with turner angles that do not match their double diffusive regime
    df_gl_stats.loc[((df_gl_stats['turner_ang'] > 90) & (df_gl_stats['salt_finger'])), 'bad_grad_layer'] = True
    df_gl_stats.loc[((df_gl_stats['turner_ang'] < 45) & (df_gl_stats['salt_finger'])), 'bad_grad_layer'] = True
    df_gl_stats.loc[
        ((df_gl_stats['turner_ang'] > -45) & (df_gl_stats['diffusive_convection'])), 'bad_grad_layer'] = True
    df_gl_stats.loc[
        ((df_gl_stats['turner_ang'] < -90) & (df_gl_stats['diffusive_convection'])), 'bad_grad_layer'] = True

    # Flag bad mixed layers following grad layers
    df_ml_stats['bad_mixed_layer'] = False
    df_ml_stats.loc[1:, 'bad_mixed_layer'] = df_gl_stats.bad_grad_layer.values
    df_ml_stats.loc[0, 'bad_mixed_layer'] = df_gl_stats.bad_grad_layer.values[0]

    # Populate masks of mixed and gradient layers before returning dataframe
    for i, row in df_ml_stats[(df_ml_stats['salt_finger_step']) & (~df_ml_stats['bad_mixed_layer'])].iterrows():
        df.loc[row.p_start:row.p_end, 'mixed_layer_final_mask'] = False

    for i, row in df_gl_stats[(df_gl_stats['salt_finger_step']) & (~df_gl_stats['bad_grad_layer'])].iterrows():
        df.loc[row.p_start:row.p_end, 'gradient_layer_final_mask'] = False
    return df, df_ml_stats, df_gl_stats


def filter_gradient_layers(df, df_ml_stats, df_gl_stats, interface_max_height, temp_flag_only, pressure_step):
    # Exclude gradient layers that are thicker than the max height, or more than twice as thick as adjacent mixed layers
    df_gl_stats['adj_ml_height'] = np.nanmax(
        np.array([df_ml_stats.iloc[1:].layer_height.values, df_ml_stats.iloc[:-1].layer_height.values]), 0)
    df_gl_stats['height_ratio'] = df_gl_stats.adj_ml_height / df_gl_stats.layer_height
    df_gl_stats.loc[df_gl_stats.height_ratio < 0.5, 'bad_grad_layer'] = True
    df_gl_stats.loc[df_gl_stats.layer_height > interface_max_height, 'bad_grad_layer'] = True

    # Remove interfaces with temp or salinity inversions
    for i, row in df_gl_stats.iterrows():
        # do not look for turning points in layers with only one point
        if row.p_end - row.p_start <= pressure_step:
            continue
        layer = df.loc[((df.p >= row.p_start) & (df.p <= row.p_end)), :]
        maxima_ct = len(argrelextrema(layer['ct'].values, np.greater)[0])
        maxima_sa = len(argrelextrema(layer['sa'].values, np.greater)[0])
        minima_ct = len(argrelextrema(layer['ct'].values, np.less)[0])
        minima_sa = len(argrelextrema(layer['sa'].values, np.less)[0])
        if temp_flag_only:
            turning_points = np.array([maxima_ct, minima_ct])
        else:
            turning_points = np.array([maxima_ct, maxima_sa, minima_ct, minima_sa])
        if (turning_points > 2).any():
            df_gl_stats.loc[i, 'bad_grad_layer'] = True

    df['grad_layer_step3_mask'] = True
    for i, row in df_gl_stats[~df_gl_stats['bad_grad_layer']].iterrows():
        df.loc[row.p_start:row.p_end, 'grad_layer_step3_mask'] = False
    return df, df_gl_stats


def classify_salt_finger_diffusive_convective(df, df_ml_stats, df_gl_stats):
    df_gl_stats['salt_finger'] = False
    df_gl_stats['diffusive_convection'] = False
    df_ml_stats_diff = df_ml_stats.diff()[1:]
    temp_grad = df_ml_stats_diff.ct / df_ml_stats_diff.p
    salt_grad = df_ml_stats_diff.sa / df_ml_stats_diff.p
    df_gl_stats.loc[(salt_grad.values < 0) & (temp_grad.values < 0), 'salt_finger'] = True
    df_gl_stats.loc[(salt_grad.values > 0) & (temp_grad.values > 0), 'diffusive_convection'] = True

    df['grad_layer_step4_mask'] = True
    for i, row in df_gl_stats[~df_gl_stats['bad_grad_layer']].iterrows():
        if row.salt_finger or row.diffusive_convection:
            df.loc[row.p_start:row.p_end, 'grad_layer_step4_mask'] = False
    return df, df_gl_stats

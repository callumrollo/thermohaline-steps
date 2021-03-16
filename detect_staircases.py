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


def main():
    df = pd.read_csv(
        '/media/callum/storage/Documents/Eureka/processing/staircase_experiment/reimplement/glider_staircase_sample.csv')
    output = classify_staircase(df.pressure_1db, df.cons_temp, df.abs_salinity)
    print('done')


def center_diff_df(df):
    return (df.diff() - df.diff(periods=-1)) / 2


def add_ml_stats_row(df_ml_stats, df):
    df_min = df.min()
    df_max = df.max()
    df_mean = df.mean()
    ml_row = {'p_start': df_min.p, 'p_end': df_max.p, 'ct': df_mean.ct, 'sa': df_mean.sa, 'sigma1': df_mean.sigma1,
              'p': df_mean.p, 'ct_range': df_max.ct - df_min.ct, 'sa_range': df_max.sa - df_min.sa,
              'sigma1_range': df_max.sigma1 - df_min.sigma1,
              'pressure_extent': df_max.p - df_min.p, 'turner_ang': df_mean.turner_ang,
              'stability_ratio': df_mean.stability_ratio}
    df_ml_stats = df_ml_stats.append(ml_row, ignore_index=True)
    return df_ml_stats


def classify_staircase(p, ct, sa, ml_grad=0.0005, ml_density_difference=0.005, av_window=200, height_ratio=30):
    """
    all data should be at 1 dbar resolution (for now)
    Notes:
    - Currently working on data at steps between input (for Turner angle)
    - Turner angle and stability-ratio from smoothed profile
    :param p: pressure (dbar)
    :param ct: conservative temperature (degrees celsius)
    :param sa: absolute salinity (g kg-1)
    :param ml_grad: density gradient for detection of mixed layer (kg m^-3 dbar^-1), default: 0.0005
    :param ml_density_difference: maximum density gradient difference of mixed layer (kg m^-3), default: 0.005
    :param av_window: averaging window to obtain background profiles (dbar), default: 200
    :param height_ratio: ratio between mixed layer height and gradient layer height, default: 30
    :return:
    """
    # TODO import test: are data evenly sampled in pressure?
    # Step 0: Prepare data. Using a pandas dataframe to keep neat
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
    # Combine logical masks for a mask that is False for mixed layer detected in temp, salt or density
    df['mixed_layer_mask'] = df.mixed_layer_temp_mask & df.mixed_layer_sal_mask & df.mixed_layer_sigma_mask

    # Create dataframe of only points within mixed layers
    df_ml = df[~df.mixed_layer_mask]
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

    # Drop mixed layers with density difference exceeding ml_density_difference
    df_ml_stats = df_ml_stats[df_ml_stats.sigma1_range < ml_density_difference]
    """
    Step 2 Assess interface layers (if) between ml
    """
    # Step 3 measure height of if and their properties
    # Step 4 Classify layers in the double diffusive regime as salt finger (sf) or diffusive convection (dc)
    # Step 5 Identify sequences of steps

    # temporary tests of functionality

    return df_diff


if __name__ == '__main__':
    main()

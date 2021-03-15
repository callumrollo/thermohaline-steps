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


def center_diff_df(df):
    return (df.diff() - df.diff(periods=-1)) / 2


def classify_staircase(p, ct, sa, ml_grad=0.0005, ml_density_difference=0.005, av_window=200, height_ratio=30):
    """
    all data should be at 1 dbar resolution (for now)
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
    df = pd.DataFrame(data={'p': p, 'ct': ct, 'sa': sa})
    df['sigma1'] = gsw.sigma1(df.sa, df.ct)
    # Interpolate linearly over nans in input data
    df = df.interpolate(method='linear')
    # Take the center diff of the dataframe wrt pressure
    df_diff = center_diff_df(df)
    pressure_step = df_diff.p.mean()
    # Take rolling average of the dataframe, using av_window
    df_smooth = df.rolling(window=av_window, center=True).mean()
    df_smooth['alpha'] = gsw.alpha(df_smooth.sa, df_smooth.ct, df_smooth.p)
    df_smooth['beta'] = gsw.beta(df_smooth.sa, df_smooth.ct, df_smooth.p)
    df_center = df.rolling(window=2, center=True).mean()[1:]
    df_center['turner_ang'], df_center['stability_ratio'], __ = gsw.stability.Turner_Rsubrho(df.sa, df.ct, df.p, axis=0)

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

    # TODO Measure extent of mixed layers and exclude those that exceed the max density difference
    # Create dataframe of only points within mixed layers
    df_ml = df[~df.mixed_layer_mask]
    # Create a dataframe for mixed layer stats. Each row will match a mixed layer
    df_ml_stats = pd.DataFrame(
        columns=['ct', 'sa', 'sigma1', 'p', 'ct_range', 'sa_range', 'sigma1_range', 'height', 'turner_ang',
                 'density_ratio'])
    # Loop through rows of mixed layer points, identifying individual layers
    start_index = df_ml.index[0]
    prev_index = df_ml.index[0]
    continuous_ml = False
    singletons = 1
    for i, row in df_ml.iloc[1:].iterrows():
        if i == prev_index + 1:
            if not continuous_ml:
                # Current row is adjacent to previous row: continuous mixed layer
                start_index = prev_index
                continuous_ml = True
                singletons = 0
        else:
            if singletons == 0:
                print('mixed layer from ' + str(start_index) + ' to ' + str(prev_index))
            elif singletons > 1:
                print('single layer at ' + str(prev_index))
            singletons += 1
            continuous_ml = False
        prev_index = i

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

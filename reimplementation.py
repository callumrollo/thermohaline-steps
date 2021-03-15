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


def main():
    df = pd.read_csv(
        '/media/callum/storage/Documents/Eureka/processing/staircase_experiment/reimplement/glider_staircase_sample.csv')
    output = classify_staircase(df.pressure_1db, df.cons_temp, df.abs_salinity)
    print(output.max())


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
    # Following the 5 steps described in https://essd.copernicus.org/articles/13/43/2021/
    # Step 0: Prepare data. Using a pandas dataframe to keep neat
    df = pd.DataFrame(index=p, data={'p': p, 'ct': ct, 'sa': sa})
    df = df.interpolate(method='linear')  # Interpolate linearly over nans in input data
    df_diff = (df.diff() + df.diff(periods=-1)) / 2  # Take the center diff of the dataframe wrt pressure
    df_av = df.rolling(window=200, center=True).mean()
    df['sigma1'] = gsw.sigma1(df.sa, df.ct)
    # Step 1 Detect data in subsurface mixed layers (ml)
    # Step 2 Assess interface layers (if) between ml
    # Step 3 measure height of if and their properties
    # Step 4 Classify layers in the double diffusive regime as salt finger (sf) or diffusive convection (dc)
    # Step 5 Identify sequences of steps

    return df_av


if __name__ == '__main__':
    main()

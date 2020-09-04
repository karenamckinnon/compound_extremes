"""
Calculate trends in given percentiles of humidity conditional on temperature.
"""

import numpy as np
import pandas as pd
from humidity_variability.utils import add_date_columns, add_GMT
from humidity_variability.models import fit_interaction_model
import os
from subprocess import check_call
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('start_month', type=int, help='Integer month to start (1-indexed)')
    parser.add_argument('end_month', type=int, help='Integer month to end (1-indexed)')
    parser.add_argument('id_start', type=int, help='Station index to start with')
    parser.add_argument('n_id', type=int, help='Number of stations to run')
    parser.add_argument('datadir', type=str, help='Full path to data')

    args = parser.parse_args()
    metadata = pd.read_csv('%s/new_metadata.csv' % (args.datadir))
    qs = np.array([0.05, 0.10, 0.5, 0.90, 0.95])
    start_year = 1979
    end_year = 2019
    start_month = args.start_month
    end_month = args.end_month
    temp_var = 'TMP'
    humidity_var = 'Q'

    # make sure dirs to save exist
    paramdir = '%s/params' % (args.datadir)
    if not os.path.isdir(paramdir):
        cmd = 'mkdir %s' % paramdir
        check_call(cmd.split())

    def return_lambda(x_data):
        """Model for lambda from JABES paper"""
        stdT = np.std(x_data)
        loglam = -0.3675503 + 0.67099232*stdT - 0.06247437*stdT**2

        return np.exp(loglam)

    # Loop through stations
    for _, row in metadata.iloc[args.id_start:(args.id_start + args.n_id), :].iterrows():
        this_id = row['station_id']
        f = '%s/%s.csv' % (args.datadir, this_id)
        df = pd.read_csv(f)
        print(this_id)
        savename = ('%s/ERA5_US_extremes_params_%i_%i_month_%i-%i_%s.npz'
                    % (paramdir, start_year, end_year, start_month, end_month, this_id))

        if os.path.isfile(savename):
            continue

        # Add a small amount of noise so no temperatures are identical
        df = df.assign(**{'%s_anom' % temp_var: df['%s_anom' % temp_var] + 1e-8*np.random.randn(len(df))})

        # Add additional date columns
        df = add_date_columns(df)

        # add GMT anoms
        df = add_GMT(df, GMT_fname='/glade/work/mckinnon/BEST/Land_and_Ocean_complete.txt')

        # Pull out season
        df = df.loc[(df['month'] >= start_month) & (df['month'] <= end_month)]

        lam_use = return_lambda(df['%s_anom' % temp_var].values)

        # remove mean of GMT
        df = df.assign(GMT=df['GMT'] - np.mean(df['GMT']))

        if not os.path.isfile(savename):
            # Sort data frame by temperature to allow us to minimize the second derivative of the T-Td relationship
            df = df.sort_values('%s_anom' % temp_var)

            # Create X, the design matrix
            # Intercept, linear in GMT, knots at all data points for temperature, same times GMT
            n = len(df)
            ncols = 2 + 2*n
            X = np.ones((n, ncols))
            X[:, 1] = df['GMT'].values
            X[:, 2:(2 + n)] = np.identity(n)
            X[:, (2 + n):] = np.identity(n)*df['GMT'].values
            # Fit the model
            BETA, _ = fit_interaction_model(qs, lam_use*np.ones(len(qs)), 'Fixed', X,
                                            df['%s_anom' % humidity_var].values, df['%s_anom' % temp_var].values)
            del X

            # Save primary fit
            np.savez(savename,
                     T=df['%s_anom' % temp_var].values,
                     H=df['%s_anom' % humidity_var].values,
                     G=df['GMT'].values,
                     BETA=BETA,
                     lambd=lam_use,
                     lat=row['lat'],
                     lon=row['lon'])

"""
Calculate trends in given percentiles of humidity conditional on temperature.
"""

import geopandas
import numpy as np
import pandas as pd
from humidity_variability.utils import add_date_columns, jitter, add_GMT
from humidity_variability.models import fit_interaction_model
from helpful_utilities.meteo import F_to_C, C_to_F
from compound_extremes.utils import fit_seasonal_cycle
import os
from subprocess import check_call
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('start_year', type=int, help='Integer year to start')
    parser.add_argument('end_year', type=int, help='Integer year to end')
    parser.add_argument('id_start', type=int, help='Station index to start with')
    parser.add_argument('n_id', type=int, help='Number of stations to run')
    parser.add_argument('datadir', type=str, help='Full path to data')
    parser.add_argument('datatype', type=str, help='GSOD or ISD')
    parser.add_argument('gjson_fname', type=str, help='Full path and filename of ROI geometry or None')
    args = parser.parse_args()

    if args.datatype == 'GSOD':
        humidity_var = 'dewp'
        temp_var = 'temp'
    elif args.datatype == 'ISD':
        humidity_var = 'Q'
        temp_var = 'TMP'

    spread = 1/10  # data rounded to 1/10 deg F
    offset = 0
    metadata = pd.read_csv('%s/new_metadata.csv' % (args.datadir))
    qs = np.array([0.05, 0.1, 0.5, 0.9, 0.95])
    start_year = args.start_year
    end_year = args.end_year

    if args.datatype == 'ISD':
        station_id = ['%06d-%05d' % (row['usaf'], row['wban']) for _, row in metadata.iterrows()]
        metadata = metadata.assign(station_id=station_id)

    # make sure dirs to save exist
    paramdir = '%s/params' % (args.datadir)
    if not os.path.isdir(paramdir):
        cmd = 'mkdir %s' % paramdir
        check_call(cmd.split())

    if args.gjson_fname != 'None':
        interior_west = geopandas.read_file(args.gjson_fname)

        # Only calculate trends in stations in west
        gdf = geopandas.GeoDataFrame(geometry=geopandas.points_from_xy(metadata.lon, metadata.lat))
        gdf.crs = 'epsg:4326'  # set projection to lat/lon
        within_ROI = gdf.within(interior_west['geometry'].loc[0]).values
        metadata = metadata.iloc[within_ROI].reset_index()

    def return_lambda(x_data):
        """Model for lambda from JABES paper"""
        stdT = np.std(x_data)
        loglam = -0.3675503 + 0.67099232*stdT - 0.06247437*stdT**2

        return np.exp(loglam)

    # Exclude AK and HI
    metadata = metadata.loc[(metadata['state'] != 'AK') & (metadata['state'] != 'HI')]
    # Loop through stations
    for _, row in metadata.iloc[args.id_start:(args.id_start + args.n_id), :].iterrows():
        this_id = row['station_id']
        f = '%s/%s.csv' % (args.datadir, this_id)
        df = pd.read_csv(f)
        print(this_id)
        savename = '%s/%s_US_extremes_params_%i_%i_%s.npz' % (paramdir, args.datatype, start_year, end_year, this_id)

        # Perform data QC

        # Drop missing data
        df = df[~np.isnan(df[humidity_var])]
        df = df[~np.isnan(df[temp_var])]

        # Drop places where less than four obs were used for average
        if args.datatype == 'GSOD':
            df = df[~((df['temp_c'] < 4) | (df['dewp_c'] < 4))]

            # Drop places where dew point exceeds temperature
            # Not strictly correct because both are daily averages, but unlikely to happen in valid data
            df = df[df[temp_var] >= df[humidity_var]]

        # Reset index, then get rid of the extra column
        df = df.reset_index()
        df = df.drop(columns=[df.columns[0]])

        # Add additional date columns
        df = add_date_columns(df)

        df_start_year = np.min(df['year'].values)
        if df_start_year > start_year:
            continue

        # add GMT anoms
        df = add_GMT(df, GMT_fname='/glade/work/mckinnon/BEST/Land_and_Ocean_complete.txt')

        # Drop Feb 29, and rework day of year counters
        leaps = np.arange(1904, 2020, 4)  # leap years
        for ll in leaps:
            old_doy = df.loc[(df['year'] == ll) & (df['month'] > 2), 'doy'].values
            df.loc[(df['year'] == ll) & (df['month'] > 2), 'doy'] = old_doy - 1
        df = df[~((df['month'] == 2) & (df['doy'] == 60))]

        if args.datatype == 'GSOD':
            # Add jitter and convert to C
            df = df.assign(**{temp_var: F_to_C(jitter(df[temp_var], offset, spread))})
            df = df.assign(**{humidity_var: F_to_C(jitter(df[humidity_var], offset, spread))})

        elif args.datatype == 'ISD':  # US data was originally 1/10 F, but converted to C
            df = df.assign(**{temp_var: F_to_C(jitter(C_to_F(df[temp_var]), 0, 1/10))})

        # Fit seasonal cycle with first three harmonics and remove
        _, residual_T, _ = fit_seasonal_cycle(df['doy'], df[temp_var].copy(), nbases=3)
        # Dew point seasonal cycle requires 10 harmonics because rapid uptick in monsoon regions
        _, residual_H, _ = fit_seasonal_cycle(df['doy'], df[humidity_var].copy(), nbases=10)

        df = df.assign(**{'%s_anom' % humidity_var: residual_H})
        df = df.assign(**{'%s_anom' % temp_var: residual_T})

        del residual_T, residual_H

        # Pull out JJAS
        df = df.loc[(df['month'] >= 6) & (df['month'] <= 9)]

        # Pull out correct year span
        df = df[(df['year'] >= start_year) & (df['year'] <= end_year)]

        # Check if sufficient data
        yrs = np.arange(start_year, end_year + 1)  # inclusive
        frac_avail = np.zeros((len(yrs)))
        for ct, yy in enumerate(yrs):
            count = len(df[(df['year'] == yy)])
            frac_avail[ct] = count/122  # 122 days in JJAS!

        frac_with_80 = np.sum(frac_avail > 0.8)/len(frac_avail)

        # Conditions to include station:
        # (1) Overall, must have at least 80% of coverage over at least 80% of years
        # (2) Must have data in first three and last three years of record
        # (3) Can't have more than one missing year in a row
        data_sufficient = ((np.mean(frac_avail[:3]) > 0) &
                           (np.mean(frac_avail[-3:]) > 0) &
                           (frac_with_80 > 0.8))

        if start_year > 1950:
            no_data = np.where(frac_avail[:-1] == 0)[0]
            for ii in no_data:
                if frac_avail[ii+1] == 0:
                    data_sufficient = 0
                    break
        if ~data_sufficient:
            continue

        lam_use = return_lambda(df['%s_anom' % temp_var].values)

        # remove mean of GMT
        df = df.assign(GMT=df['GMT'] - np.mean(df['GMT']))

        df0 = df.copy()
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

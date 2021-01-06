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
import calendar


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('start_year', type=int, help='Integer year to start')
    parser.add_argument('end_year', type=int, help='Integer year to end')
    parser.add_argument('start_month', type=int, help='Integer month to start (1-indexed)')
    parser.add_argument('end_month', type=int, help='Integer month to end (1-indexed)')
    parser.add_argument('id_start', type=int, help='Station index to start with')
    parser.add_argument('n_id', type=int, help='Number of stations to run')
    parser.add_argument('datadir', type=str, help='Full path to data')
    parser.add_argument('dataname', type=str, help='Source of T/q data: ISD, ERA5, or JRA55')
    parser.add_argument('gjson_fname', type=str, help='Full path and filename of ROI geometry or None')
    parser.add_argument('predictor', type=str, help='GMT or year')
    args = parser.parse_args()

    humidity_var = 'Q'
    temp_var = 'TMP'

    spread = 1/10  # data rounded to 1/10 deg F
    offset = 0
    metadata = pd.read_csv('%s/new_metadata.csv' % (args.datadir))
    qs = np.array([0.05, 0.1, 0.5, 0.9, 0.95])
    start_year = args.start_year
    end_year = args.end_year
    start_month = args.start_month
    end_month = args.end_month
    predictor = args.predictor
    dataname = args.dataname

    if dataname == 'ISD':  # fix formatting issue in metadata
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
        """Model for lambda from JABES paper, based on predictions with lowpass GMTA (not year)"""
        stdT = np.std(x_data)
        loglam = -0.3675503 + 0.67099232*stdT - 0.06247437*stdT**2

        return np.exp(loglam)

    # Exclude AK and HI
    if dataname == 'ISD':
        metadata = metadata.loc[(metadata['state'] != 'AK') & (metadata['state'] != 'HI')]
    # Loop through stations
    for _, row in metadata.iloc[args.id_start:(args.id_start + args.n_id), :].iterrows():
        this_id = row['station_id']
        f = '%s/%s.csv' % (args.datadir, this_id)
        df = pd.read_csv(f)
        print(this_id)
        savename = ('%s/%s_US_extremes_params_trend_%s_%i_%i_month_%i-%i_%s.npz'
                    % (paramdir, dataname, predictor, start_year, end_year, start_month, end_month, this_id))

        if os.path.isfile(savename):
            continue

        if dataname == 'ISD':
            # Perform data QC

            # Drop missing data
            df = df[~np.isnan(df[humidity_var])]
            df = df[~np.isnan(df[temp_var])]

            # Reset index, then get rid of the extra column
            df = df.reset_index()
            df = df.drop(columns=[df.columns[0]])

        # Add additional date columns
        df = add_date_columns(df)

        df_start_year = np.min(df['year'].values)
        if df_start_year > start_year:
            continue

        # add GMT anoms, lowpass filtered with frequency cutoff of 1/10yr
        df = add_GMT(df, lowpass_freq=1/10, GMT_fname='../data/Land_and_Ocean_complete.txt')

        # Drop Feb 29, and rework day of year counters
        leaps = np.arange(1904, 2020, 4)  # leap years
        for ll in leaps:
            old_doy = df.loc[(df['year'] == ll) & (df['month'] > 2), 'doy'].values
            df.loc[(df['year'] == ll) & (df['month'] > 2), 'doy'] = old_doy - 1
        df = df[~((df['month'] == 2) & (df['doy'] == 60))]

        if dataname == 'ISD':
            # US data was originally 1/10 F, but converted to C
            df = df.assign(**{temp_var: F_to_C(jitter(C_to_F(df[temp_var]), 0, 1/10))})

        # Fit seasonal cycle with first three harmonics and remove
        _, residual_T, _ = fit_seasonal_cycle(df['doy'], df[temp_var].copy(), nbases=3)
        # Humidity seasonal cycle requires 10 harmonics because rapid uptick in monsoon regions
        _, residual_H, _ = fit_seasonal_cycle(df['doy'], df[humidity_var].copy(), nbases=10)

        df = df.assign(**{'%s_anom' % humidity_var: residual_H})
        df = df.assign(**{'%s_anom' % temp_var: residual_T})

        if ((dataname == 'ERA5') | (dataname == 'JRA55')):
            # Add a small amount of noise so no temperatures are identical
            df = df.assign(**{'%s_anom' % temp_var: df['%s_anom' % temp_var] + 1e-8*np.random.randn(len(df))})

        del residual_T, residual_H

        # Pull out season
        df = df.loc[(df['month'] >= start_month) & (df['month'] <= end_month)]

        # Pull out correct year span
        df = df[(df['year'] >= start_year) & (df['year'] <= end_year)]

        if dataname == 'ISD':
            # Calculate the number of days in the season (all summer, so no leap year concerns)
            seasonal_days = 0
            for mo in range(start_month, end_month + 1):
                seasonal_days += calendar.monthrange(2020, mo)[-1]

            # Check if sufficient data
            yrs = np.arange(start_year, end_year + 1)  # inclusive
            frac_avail = np.zeros((len(yrs)))
            for ct, yy in enumerate(yrs):
                count = len(df[(df['year'] == yy)])
                frac_avail[ct] = count/seasonal_days

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

        # remove mean of predictor (typically GMTA)
        df = df.assign(**{predictor: df[predictor].astype(float) - np.mean(df[predictor])})

        if not os.path.isfile(savename):
            # Sort data frame by temperature to allow us to minimize the second derivative of the T-Td relationship
            df = df.sort_values('%s_anom' % temp_var)

            # Create X, the design matrix
            # Intercept, linear in GMT, knots at all data points for temperature, same times GMT
            n = len(df)
            ncols = 2 + 2*n
            X = np.ones((n, ncols))
            X[:, 1] = df[predictor].values
            X[:, 2:(2 + n)] = np.identity(n)
            X[:, (2 + n):] = np.identity(n)*df[predictor].values
            # Fit the model
            BETA, _ = fit_interaction_model(qs, lam_use*np.ones(len(qs)), 'Fixed', X,
                                            df['%s_anom' % humidity_var].values, df['%s_anom' % temp_var].values)
            del X

            # Save primary fit
            np.savez(savename,
                     T=df['%s_anom' % temp_var].values,
                     H=df['%s_anom' % humidity_var].values,
                     G=df[predictor].values,
                     BETA=BETA,
                     lambd=lam_use,
                     lat=row['lat'],
                     lon=row['lon'])

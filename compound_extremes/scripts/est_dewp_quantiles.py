"""
Calculate trends in given percentiles of dew point conditional on temperature.

Both temperature and dew point are daily average values from GSOD data.
"""

import geopandas
import numpy as np
import pandas as pd
from humidity_variability.utils import add_date_columns, jitter, add_GMT
from humidity_variability.models import fit_interaction_model
from helpful_utilities.meteo import F_to_C
from compound_extremes.utils import fit_seasonal_cycle
import os
from subprocess import check_call
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('id_start', type=int, help='Station index to start with')
    parser.add_argument('n_id', type=int, help='Number of stations to run')
    parser.add_argument('boot_start', type=int, help='Bootstrap index to start with')
    parser.add_argument('nboot', type=int, help='Number of samples')
    parser.add_argument('datadir', type=str, help='Full path to data')
    parser.add_argument('gjson_fname', type=str, help='Full path and filename of ROI geometry')
    args = parser.parse_args()

    spread = 1/10  # data rounded to 1/10 deg F
    offset = 0
    qs = np.array([0.05, 0.5, 0.95])
    start_year = 1973
    end_year = 2019

    # original metadata is sometimes incorrect
    # new_metadata has correct start/end times
    metadata = pd.read_csv('%s/new_metadata.csv' % (args.datadir))

    # make sure dirs to save exist
    paramdir = '%s/params' % (args.datadir)
    if not os.path.isdir(paramdir):
        cmd = 'mkdir %s' % paramdir
        check_call(cmd.split())

    if not os.path.isdir('%s/boot' % paramdir):
        cmd = 'mkdir %s/boot' % paramdir
        check_call(cmd.split())

    # Interior west domain for bootstrapping (lat, lon)
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

    # Loop through stations
<<<<<<< HEAD
    for _, row in metadata[(metadata['state']!='AK') & (metadata['state']!='HI')].iterrows():  # .iloc[args.id_start:(args.id_start + args.n_id), :].iterrows():
        this_id = row['station_id']
        f = '%s/%s.csv' % (args.datadir, this_id)
        df = pd.read_csv(f)

        print(this_id)

=======
    for _, row in metadata.iloc[args.id_start:(args.id_start + args.n_id), :].iterrows():
        this_id = row['station_id']
        f = '%s/%s.csv' % (args.datadir, this_id)
        df = pd.read_csv(f)
        print(this_id)
>>>>>>> updated to add bootstrap
        savename = '%s/US_extremes_params_%i_%i_%s.npz' % (paramdir, start_year, end_year, this_id)

        # Perform data QC

        # Drop missing data
        df = df[~np.isnan(df['dewp'])]
        df = df[~np.isnan(df['temp'])]

        # Drop places where less than four obs were used for average
        df = df[~((df['temp_c'] < 4) | (df['dewp_c'] < 4))]

        # Drop places where dew point exceeds temperature
        # Not strictly correct because both are daily averages, but unlikely to happen in valid data
        df = df[df['temp'] >= df['dewp']]

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

        # Add jitter
        df = df.assign(temp=jitter(df['temp'], offset, spread))
        df = df.assign(dewp=jitter(df['dewp'], offset, spread))

        # convert to C
        df = df.assign(dewp=F_to_C(df['dewp']))
        df = df.assign(temp=F_to_C(df['temp']))

        # Fit seasonal cycle with first three harmonics and remove
        _, residual_T, _ = fit_seasonal_cycle(df['doy'], df['temp'], nbases=3)
        # Dew point seasonal cycle requires 10 harmonics because rapid uptick in monsoon regions
        _, residual_DP, _ = fit_seasonal_cycle(df['doy'], df['dewp'], nbases=10)

        df = df.assign(dewp_anom=residual_DP)
        df = df.assign(temp_anom=residual_T)

        del residual_T, residual_DP

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

        no_data = np.where(frac_avail[:-1] == 0)[0]
        for ii in no_data:
            if frac_avail[ii+1] == 0:
                data_sufficient = 0
                break

        if ~data_sufficient:
            continue

        lam_use = return_lambda(df['temp_anom'].values)

        # remove mean of GMT
        df = df.assign(GMT=df['GMT'] - np.mean(df['GMT']))

        df0 = df.copy()
        if not os.path.isfile(savename):
            # Sort data frame by temperature to allow us to minimize the second derivative of the T-Td relationship
            df = df.sort_values('temp_anom')

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
                                            df['dewp_anom'].values, df['temp_anom'].values)
            del X

            # Save primary fit
            np.savez(savename,
                     T=df['temp_anom'].values,
                     Td=df['dewp_anom'].values,
                     G=df['GMT'].values,
                     BETA=BETA,
                     lambd=lam_use,
                     lat=row['lat'],
                     lon=row['lon'])

        # Check if we're doing the bootstrap
        if args.nboot > 0:

            for kk in range(args.boot_start, args.boot_start + args.nboot):
                # Set seed so we can reproduce each bootstrap sample if needed
                np.random.seed(kk)
                print('%s: %i' % (this_id, kk))

                savename = ('%s/boot/US_extremes_params_%i_%i_%s_boot_%04i.npz'
                            % (paramdir, start_year, end_year, this_id, kk))
                if os.path.isfile(savename):
                    continue

                # Reset back to original df
                df = df0.copy()
                # Resample years (full bootstrap)
                yrs_unique = np.unique(df['year'])
                nyrs = len(yrs_unique)
                new_years = np.random.choice(yrs_unique, nyrs)

                new_df = pd.DataFrame()
                for yy in new_years:
                    sub_df = df.loc[df['year'] == yy, :]
                    new_df = new_df.append(sub_df)

                new_df = new_df.reset_index()

                # remove mean of GMT
                new_df = new_df.assign(GMT=new_df['GMT'] - np.mean(new_df['GMT']))

                # Add jitter again, since we're resampling years
                # However, note that we're now in Celsius, so 1/10 F -> 5/90 C
                new_df['temp_anom'] = jitter(new_df['temp_anom'], 0, 5/90)
                new_df['dewp_anom'] = jitter(new_df['dewp_anom'], 0, 5/90)

                # Sort
                new_df = new_df.sort_values('temp_anom')

                # Create X, the design matrix
                # Intercept, linear in GMT, knots at all data points for temperature, same times GMT
                n = len(new_df)
                ncols = 2 + 2*n
                X = np.ones((n, ncols))
                X[:, 1] = new_df['GMT'].values
                X[:, 2:(2 + n)] = np.identity(n)
                X[:, (2 + n):] = np.identity(n)*new_df['GMT'].values

                # Fit the model
                BETA, _ = fit_interaction_model(qs, lam_use*np.ones(len(qs)), 'Fixed', X,
                                                new_df['dewp_anom'].values, new_df['temp_anom'].values)
                del X

                np.savez(savename,
                         T=new_df['temp_anom'].values,
                         Td=new_df['dewp_anom'].values,
                         G=new_df['GMT'].values,
                         BETA=BETA,
                         lambd=lam_use,
                         lat=row['lat'],
                         lon=row['lon'])
        del df

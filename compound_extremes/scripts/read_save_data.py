import numpy as np
import pandas as pd
import ctypes
from compound_extremes import utils
from helpful_utilities.general import fit_OLS


# Params and dirs
datadir = '/home/mckinnon/bucket/gsod'
figdir = '/home/mckinnon/projects/humidity_variability/humidity_variability/figs'
procdir = '/home/mckinnon/projects/humidity_variability/humidity_variability/proc'

start_year = 1973
end_year = 2018
expand_data = True  # should the start/end year be the edges of the data, or a minimum requirement?
search_query = {'begin': 'datetime(%i, 1, 1)' % start_year,
                'end': 'datetime(%i, 12, 31)' % end_year}

hashable = tuple((tuple(search_query.keys()), tuple(search_query.values()), expand_data))
query_hash = str(ctypes.c_size_t(hash(hashable)).value)  # ensures positive value

# original metadata is sometimes incorrect
# new_metadata has correct start/end times
metadata = pd.read_csv('%s/%s/new_metadata.csv' % (datadir, query_hash))

paramdir = '%s/%s/params' % (datadir, query_hash)
window_length = 90

# Initialize data frames
appended_data = []
station_stats = pd.DataFrame(columns=('station_id', 'peakT', 'peakTd', 'rho', 'rho_detrended'))

# Load in data from monsoon region to check what scatter plot looks like after removing seasonal cycle
for id_counter, this_id in enumerate(metadata['station_id'].values):
    print('%i/%i' % (id_counter, len(metadata)))

    metadata_idx = metadata['station_id'] == this_id

    lat = metadata.loc[metadata_idx, 'lat'].values[0]
    lon = metadata.loc[metadata_idx, 'lon'].values[0]
    name = metadata.loc[metadata_idx, 'station_name'].values[0]
    country = metadata.loc[metadata_idx, 'ctry'].values[0]

    f = '%s/%s/%s.csv' % (datadir, query_hash, this_id)
    df = pd.read_csv(f)

    # Do initial quality control

    # (1) Remove any places where dewpoint exceeds temperature
    remove_idx = (df['temp'] < df['dewp'])

    # (2) Remove any cases where there are less than four values going into either average
    remove_idx = remove_idx | (df['temp_c'] < 4) | (df['dewp_c'] < 4)

    # (3) Remove missing data
    remove_idx = remove_idx | np.isnan(df['temp']) | np.isnan(df['dewp'])

    # (4) Drop leap days
    remove_idx = remove_idx | ['02-29' in d for d in df['date']]

    df = df.loc[~remove_idx]
    df = df.reset_index()

    # Estimate seasonal cycle in temperature and dewpoint
    dates_pd = [pd.datetime.strptime(df.loc[kk, 'date'], '%Y-%m-%d') for kk in range(len(df))]
    doy = np.array([d.timetuple().tm_yday for d in dates_pd])

    seasonalT, residualT, T_ann = utils.fit_seasonal_cycle(doy, df['temp'].values)
    seasonalDP, residualDP, DP_ann = utils.fit_seasonal_cycle(doy, df['dewp'].values)

    # Find peaks for temperature and dewpoint
    # argmax finds index, we want day of year, so add 1
    double = np.hstack((T_ann, T_ann))
    tmp = np.convolve(np.ones(window_length,), double, mode='valid')
    peak_doy_T = (np.argmax(tmp) + 1) % 365, (np.argmax(tmp) + window_length + 1) % 365

    double = np.hstack((DP_ann, DP_ann))
    tmp = np.convolve(np.ones(window_length,), double, mode='valid')
    peak_doy_DP = (np.argmax(tmp) + 1) % 365, (np.argmax(tmp) + window_length + 1) % 365

    # Create new dataframe to store QC values at all days (fill in with NaNs when missing)
    full_time_vec = pd.date_range(start='%04d/1/1' % start_year, end='%04d/12/31' % end_year, freq='D')
    full_doy = np.array([d.timetuple().tm_yday for d in full_time_vec])
    if peak_doy_T[1] > peak_doy_T[0]:  # boreal summer
        idx_seasonal = (full_doy >= peak_doy_T[0]) & (full_doy < peak_doy_T[1])
    else:
        idx_seasonal = (full_doy <= peak_doy_T[0]) | (full_doy > peak_doy_T[1])

    this_df = pd.DataFrame(data={'date': full_time_vec[idx_seasonal],
                                 'doy': full_doy[idx_seasonal],
                                 'station_id': this_id})

    # Pull out temperature, dew point data
    tmp_vec = np.nan*np.ones((len(this_df), 2))
    for kk in range(len(tmp_vec)):
        try:
            idx_match = np.where((df['date']) == pd.datetime.strftime(this_df.loc[kk, 'date'],
                                                                      format='%Y-%m-%d'))[0][0]
            tmp_vec[kk, :] = df.loc[idx_match, ['temp', 'dewp']].values
        except IndexError:  # case where there is no data
            continue

    this_df = this_df.assign(temp=tmp_vec[:, 0], dewp=tmp_vec[:, 1])

    doy_ann = np.arange(1, 366)
    if peak_doy_T[1] > peak_doy_T[0]:  # boreal summer
        idx_seasonal = (doy_ann >= peak_doy_T[0]) & (doy_ann <= peak_doy_T[1])
    else:
        idx_seasonal = (doy_ann <= peak_doy_T[0]) | (doy_ann >= peak_doy_T[1])
    temp_clim = T_ann[idx_seasonal]
    dewp_clim = DP_ann[idx_seasonal]

    match_idx = np.array([np.where(doy_ann[idx_seasonal] == this_doy)[0][0] for this_doy in this_df['doy']])
    this_df = this_df.assign(temp_clim=temp_clim[match_idx], dewp_clim=dewp_clim[match_idx])

    # Do additional quality control

    # (1) Remove outliers
    X = np.vstack((this_df['temp'] - this_df['temp_clim'], this_df['dewp'] - this_df['dewp_clim']))

    X = X[:, ~np.isnan(X).any(axis=0)]
    outliers = ((X[0, :] > (np.mean(X[0, :]) + 6*np.std(X[0, :]))) |
                (X[0, :] < (np.mean(X[0, :]) - 6*np.std(X[0, :]))) |
                (X[1, :] > (np.mean(X[1, :]) + 6*np.std(X[1, :]))) |
                (X[1, :] < (np.mean(X[1, :]) - 6*np.std(X[1, :]))))

    this_df.loc[outliers, ['temp', 'dewp']] = np.nan

    # Demand at least 80% of data in 80% of years
    yrs = np.array([d.year for d in this_df['date']])
    frac_missing = np.array([np.sum(np.isnan(this_df.loc[yrs == yy, 'temp'])) /
                             np.sum(yrs == yy) for yy in np.unique(yrs)])
    frac_lt_80 = np.sum(frac_missing > 0.2)/len(frac_missing)
    if frac_lt_80 > 0.2:
        continue
    else:  # add to dataframe
        appended_data.append(this_df)

        # Calculate and save some stats for the station

        # Calculate raw correlation
        missing_rows = np.isnan(this_df['temp']) | np.isnan(this_df['dewp'])
        rho = np.corrcoef(this_df.loc[~missing_rows, 'temp'], this_df.loc[~missing_rows, 'dewp'])[0, 1]

        # Calculate detrended correlation
        time_since = pd.to_datetime(this_df.loc[~missing_rows, 'date']) - pd.datetime(start_year, 1, 1)
        time_index = [t.days for t in time_since]
        time_index -= np.mean(time_index).astype(int)

        beta, yhat = fit_OLS(time_index, this_df.loc[~missing_rows, 'temp'])
        detrendedT = this_df.loc[~missing_rows, 'temp'] - yhat

        beta, yhat = fit_OLS(time_index, this_df.loc[~missing_rows, 'dewp'])
        detrendedDP = this_df.loc[~missing_rows, 'dewp'] - yhat

        rho_detrended = np.corrcoef(detrendedT, detrendedDP)[0, 1]
        station_stats.loc[id_counter] = (this_id, peak_doy_T, peak_doy_DP, rho, rho_detrended)

appended_data = pd.concat(appended_data, ignore_index=True, sort=False)

# Save to csv
appended_data.to_csv('%s/%s/all_stations.csv' % (datadir, query_hash))
station_stats.to_csv('%s/%s/station_stats.csv' % (datadir, query_hash))

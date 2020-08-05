import numpy as np
import xarray as xr
import pandas as pd
from glob import glob
from compound_extremes import utils


def demedian(x):
    return (x - x.median())


# ### Look at GLEAM soil moisture data
#
# For each station:
# - find the closest GLEAM gridbox
# - load in GLEAM data, remove seasonal cycle
# - calculate raw correlation
# - calculate correlation of seasonal means
# - calculate correlation of within-season variability

sm_dir = '/glade/scratch/mckinnon/gleam'
files = sorted(glob('%s/SMsurf_*_GLEAM_v3.3a.nc' % sm_dir))
savedir = '/glade/work/mckinnon/gsod/2506838728791974695/gleam'

# Load all of our data
data_fname = '/glade/work/mckinnon/gsod/2506838728791974695/all_stations_60day-season.csv'
station_data = pd.read_csv(data_fname)
stats_fname = '/glade/work/mckinnon/gsod/2506838728791974695/station_stats_60day-season.csv'
station_stats = pd.read_csv(stats_fname)

metadata_fname = '/glade/work/mckinnon/gsod/2506838728791974695/new_metadata.csv'
metadata = pd.read_csv(metadata_fname)

station_ids = station_stats['station_id']
ds = xr.open_mfdataset(files)

sm_lats = ds.lat.values
dlat = sm_lats[1] - sm_lats[0]
sm_lat_bnds = np.vstack((sm_lats - dlat/2, sm_lats + dlat/2))

sm_lons = ds.lon.values
dlon = sm_lons[1] - sm_lons[0]
sm_lon_bnds = np.vstack((sm_lons - dlon/2, sm_lons + dlon/2))

sm_years = 1980, 2018  # soil moisture data more limited

# consider these lags between soil moisture and dew point
lags = np.array([21, 14, 7, 0])


appended_data = []
for station_count, this_id in enumerate(station_ids):
    print('%i/%i' % (station_count, len(station_ids)))

    this_data0 = station_data[station_data['station_id'] == this_id]
    this_lat = metadata.loc[metadata['station_id'] == this_id, 'lat']
    this_lon = metadata.loc[metadata['station_id'] == this_id, 'lon']
    yrs = np.array([int(this_data0['date'].values[ct].split('-')[0]) for ct in range(len(this_data0))])

    # Pull out summer season
    summer_doy = station_stats.loc[station_stats['station_id'] == this_id, 'peakT']
    doy1_dp = int(summer_doy.values[0].split(',')[0][1:])
    doy2_dp = int(summer_doy.values[0].split(',')[-1][:-1])

    # Pull out soil moisture at closest gridbox
    # Note that, for islands, this may be nans
    lat_idx = np.where((this_lat.values > sm_lat_bnds[1, :]) & (this_lat.values <= sm_lat_bnds[0, :]))[0][0]
    lon_idx = np.where((this_lon.values > sm_lon_bnds[0, :]) & (this_lon.values <= sm_lon_bnds[1, :]))[0][0]

    if np.isnan(ds.SMsurf[:, lon_idx, lat_idx].values).any():
        continue

    # Remove seasonal climatology
    seasonalSM, residualSM, SM_ann = utils.fit_seasonal_cycle(ds.time.dt.dayofyear.values,
                                                              ds.SMsurf[:, lon_idx, lat_idx].values)

    sm_df0 = pd.DataFrame(data={'date': ds.time,
                                'doy': ds.time.dt.dayofyear,
                                'year': ds.time.dt.year,
                                'SM_anoms': residualSM})

    # Remove leap days
    nonleap = sm_df0['doy'] < 366
    sm_df0 = sm_df0.loc[nonleap, :]

    for lag in lags:

        this_data = this_data0.copy()
        sm_df = sm_df0.copy()

        doy1 = doy1_dp - lag
        doy2 = doy2_dp - lag

        if doy1 <= 0:
            doy1 += 365
        if doy2 <= 0:
            doy2 += 365

        # Pull out summer season
        if doy2 > doy1:  # boreal summer
            idx_seasonal = (sm_df['doy'] >= doy1) & (sm_df['doy'] < doy2)
        else:
            idx_seasonal = (sm_df['doy'] >= doy1) | (sm_df['doy'] < doy2)

        sm_df = sm_df.loc[idx_seasonal, :]

        # For cases where summer season spans years, assign everything to the later year
        yrs = np.array([int(this_data['date'].values[ct].split('-')[0]) for ct in range(len(this_data))])
        this_data = this_data.assign(year=yrs)

        # Subselect dew point to soil moisture years
        this_data = this_data[(this_data['year'] >= sm_years[0]) & (this_data['year'] <= sm_years[1])]

        if doy2_dp < doy1_dp:
            this_data.loc[this_data['doy'] <= doy2_dp, 'year'] += 1
        if doy2 < doy1:  # austral summer
            sm_df.loc[sm_df['doy'] <= doy2, 'year'] += 1

        # Calculate various correlations

        # Remove missing values
        missing = np.isnan(this_data['dewp'].values)
        sm_df = sm_df[~missing]
        this_data = this_data[~missing]

        this_data = this_data.assign(dewp_anoms=(this_data['dewp'].values - this_data['dewp_clim'].values))

        # Raw
        rho_raw = np.corrcoef(sm_df['SM_anoms'].values, this_data['dewp_anoms'].values)[0, 1]

        # Anomalies
        sm_anomaly = sm_df.groupby(sm_df.year).transform(demedian)
        dewp_anomaly = this_data.groupby(this_data.year).transform(demedian)
        rho_anoms = np.corrcoef(sm_anomaly['SM_anoms'].values, dewp_anomaly['dewp_anoms'].values)[0, 1]

        # Between seasonal medians
        sm_df = sm_df[(sm_df['year'] >= sm_years[0]) & (sm_df['year'] <= sm_years[1])]
        sm_ann = sm_df.groupby('year').median()
        dewp_ann = this_data.groupby('year').median()

        # Make sure that the years match
        match_years = np.intersect1d(sm_ann.index, dewp_ann.index)
        sm_ann = sm_ann.loc[np.isin(sm_ann.index.values, match_years), :]
        dewp_ann = dewp_ann.loc[np.isin(dewp_ann.index.values, match_years), :]
        rho_ann = np.corrcoef(sm_ann['SM_anoms'].values, dewp_ann['dewp_anoms'].values)[0, 1]

        # save
        this_df = pd.DataFrame(data={'station_id': this_id,
                                     'lat': this_lat,
                                     'lon': this_lon,
                                     'lag': lag,
                                     'rho_raw': rho_raw,
                                     'rho_ann': rho_ann,
                                     'rho_anoms': rho_anoms})

        appended_data.append(this_df)

    if station_count % 100 == 0:
        appended_tmp = pd.concat(appended_data, ignore_index=True, sort=False)
        # Save to csv
        appended_tmp.to_csv('%s/gleam_dewp_corr_lags_TMP%04d.csv' % (savedir, station_count))
        del appended_tmp

appended_data = pd.concat(appended_data, ignore_index=True, sort=False)

appended_data.to_csv('%s/gleam_dewp_corr_lags.csv' % (savedir))

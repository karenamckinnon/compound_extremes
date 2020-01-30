import numpy as np
import xarray as xr
import pandas as pd
from glob import glob
from compound_extremes import utils
from time import time

# ### Look at GLEAM soil moisture data
#
# For each station:
# - find the closest GLEAM gridbox
# - load in GLEAM data, remove seasonal cycle
# - calculate raw correlation
# - calculate correlation of seasonal means
# - calculate correlation of within-season variability

sm_dir = '/glade/scratch/mckinnon/gleam'
files = sorted(glob('%s/SMsurf*.nc' % sm_dir))
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


def demedian(x):
    return (x - x.median())


appended_data = []
for station_count, this_id in enumerate(station_ids):
    print('%i/%i' % (station_count, len(station_ids)))
    t1 = time()
    this_data = station_data[station_data['station_id'] == this_id]
    this_lat = metadata.loc[metadata['station_id'] == this_id, 'lat']
    this_lon = metadata.loc[metadata['station_id'] == this_id, 'lon']
    yrs = np.array([int(this_data['date'].values[ct].split('-')[0]) for ct in range(len(this_data))])

    # Pull out summer season
    summer_doy = station_stats.loc[station_stats['station_id'] == this_id, 'peakT']
    doy1 = int(summer_doy.values[0].split(',')[0][1:])
    doy2 = int(summer_doy.values[0].split(',')[-1][:-1])

    # Pull out soil moisture at closest gridbox
    # Note that, for islands, this may be nans
    lat_idx = np.where((this_lat.values > sm_lat_bnds[1, :]) & (this_lat.values <= sm_lat_bnds[0, :]))[0][0]
    lon_idx = np.where((this_lon.values > sm_lon_bnds[0, :]) & (this_lon.values <= sm_lon_bnds[1, :]))[0][0]

    if np.isnan(ds.SMsurf[0, lon_idx, lat_idx].values):
        continue

    # Remove seasonal climatology
    seasonalSM, residualSM, SM_ann = utils.fit_seasonal_cycle(ds.time.dt.dayofyear.values,
                                                              ds.SMsurf[:, lon_idx, lat_idx].values)

    sm_df = pd.DataFrame(data={'date': ds.time,
                               'doy': ds.time.dt.dayofyear,
                               'year': ds.time.dt.year,
                               'SM_anoms': residualSM})

    # Remove leap days
    nonleap = sm_df['doy'] < 366
    sm_df = sm_df.loc[nonleap, :]

    # Pull out summer season
    if doy2 > doy1:  # boreal summer
        idx_seasonal = (sm_df['doy'] >= doy1) & (sm_df['doy'] < doy2)
    else:
        idx_seasonal = (sm_df['doy'] >= doy1) | (sm_df['doy'] < doy2)

    sm_df = sm_df.loc[idx_seasonal, :]

    # Subselect dew point to soil moisture years
    yrs = np.array([int(this_data['date'].values[ct].split('-')[0]) for ct in range(len(this_data))])
    this_data = this_data.assign(year=yrs)
    this_data = this_data[(this_data['year'] >= sm_years[0]) & (this_data['year'] <= sm_years[1])]

    # For cases where summer season spans years, assign everything to the later year
    if doy2 < doy1:  # austral summer
        this_data.loc[this_data['doy'] <= doy2, 'year'] += 1
        sm_df.loc[sm_df['doy'] <= doy2, 'year'] += 1

    # Calculate various correlations

    # Remove missing values
    missing = np.isnan(this_data['dewp'].values)
    sm_df = sm_df[~missing]
    this_data = this_data[~missing]

    this_data = this_data.assign(dewp_anoms=(this_data['dewp'].values - this_data['dewp_clim'].values))

    # Raw
    rho_raw = np.corrcoef(sm_df['SM_anoms'].values, this_data['dewp_anoms'].values)[0, 1]

    # Between seasonal medians
    sm_ann = sm_df.groupby('year').median()
    dewp_ann = this_data.groupby('year').median()

    rho_ann = np.corrcoef(sm_ann['SM_anoms'].values, dewp_ann['dewp_anoms'].values)[0, 1]

    sm_anomaly = sm_df.groupby(sm_df.year).transform(demedian)
    dewp_anomaly = this_data.groupby(this_data.year).transform(demedian)

    rho_anoms = np.corrcoef(sm_anomaly['SM_anoms'].values, dewp_anomaly['dewp_anoms'].values)[0, 1]

    # save
    this_df = pd.DataFrame(data={'station_id': this_id,
                                 'lat': this_lat,
                                 'lon': this_lon,
                                 'rho_raw': rho_raw,
                                 'rho_ann': rho_ann,
                                 'rho_anoms': rho_anoms})

    appended_data.append(this_df)
    dt = time() - t1
    print('%02d seconds' % dt)
appended_data = pd.concat(appended_data, ignore_index=True, sort=False)

appended_data.to_csv('%s/gleam_dewp_corr.csv' % (savedir))

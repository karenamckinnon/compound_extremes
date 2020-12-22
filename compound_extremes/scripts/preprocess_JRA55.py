import xarray as xr
import salem
import geopandas
import geojson
from shapely.geometry import shape
import numpy as np
from helpful_utilities.general import lowpass_butter
from glob import glob
import pandas as pd
import os


def subset_to_west(da, lon_range, lat_range, roi):

    """ NEED TO SORT ARRAYS BY ORDERED LAT/LON """

    try:
        da = da.sel(latitude=slice(lat_range[0], lat_range[1]),
                    longitude=slice(lon_range[0], lon_range[1]))
    except:
        da = da.sel(lat=slice(lat_range[0], lat_range[1]),
                    lon=slice(lon_range[0], lon_range[1]))

    return da.salem.roi(geometry=roi, crs='wgs84')


# projection: EPSG:3857
gjson_fname = '/glade/u/home/mckinnon/compound_extremes/compound_extremes/shapefiles/interior_west.json'
interior_west = geopandas.read_file(gjson_fname)

lon_range = (np.min(interior_west['geometry'][0].exterior.coords.xy[0]),
             np.max(interior_west['geometry'][0].exterior.coords.xy[0]))
lat_range = (np.min(interior_west['geometry'][0].exterior.coords.xy[1]),
             np.max(interior_west['geometry'][0].exterior.coords.xy[1]))

# Have to convert to shapely because weird salem issue
with open(gjson_fname) as f:
    geo = geojson.load(f)
    interior_west_shapely = shape(geo[0]['geometry'])

start_month = 7
end_month = 9
jra55_dir = '/glade/scratch/mckinnon/JRA55'
varnames_file = ['anl_surf125.011_tmp', 'anl_surf125.051_spfh']
varnames = ['TMP_GDS0_HTGL', 'SPFH_GDS0_HTGL']

for ct, this_varname in enumerate(varnames):
    savename = '/glade/work/mckinnon/JRA55/processed_%s.nc' % varnames_file[ct]
    if not os.path.isfile(savename):
        print(this_varname)
        files = sorted(glob('%s/%s*' % (jra55_dir, varnames_file[ct])))
        ds = xr.open_mfdataset(files)

        if 'fcst' in files[0]:  # use 3 hour forecast
            ds = ds.sel(forecast_time1=ds['forecast_time1'][0])

        da = ds[this_varname]
        if 'g0_lat_1' in da.coords:
            rename_dict = {'g0_lat_1': 'latitude', 'g0_lon_2': 'longitude',
                           'initial_time0_hours': 'time'}
        elif 'g0_lat_2' in da.coords:
            rename_dict = {'g0_lat_2': 'latitude', 'g0_lon_3': 'longitude',
                           'initial_time0_hours': 'time'}

        da = da.rename(rename_dict)
        # Sort by latitude
        da = da.sortby('latitude')
        # Subset to US (approx)
        da = da.sel(latitude=slice(20, 55), longitude=slice(230, 300))
        # change to -180 to 180 longitude
        da = da.assign_coords(longitude=(((da.longitude + 180) % 360) - 180))
        # subset to interior west
        da = subset_to_west(da, lon_range, lat_range, interior_west_shapely)
        # calculate daily average
        da = da.resample(time='D').mean()

        # Drop anything outside of summer
        da = da.sel(time=(da['time.month'] >= start_month) & (da['time.month'] <= end_month))

        # Remove seasonal cycle, smoothed using low pass of 1/30 days
        da_SC = da.groupby('time.dayofyear').mean('time')
        da_SC_smooth = lowpass_butter(1, 1/30, 3, da_SC.values, axis=0)
        da_SC = da_SC.copy(data=da_SC_smooth)

        da_anom = da.groupby('time.dayofyear') - da_SC
        da_anom = da_anom.rename(varnames_file[ct])

        # save
        savename = '/glade/work/mckinnon/JRA55/processed_%s.nc' % varnames_file[ct]
        da_anom.to_netcdf(savename)

    if 'TMP' in this_varname:
        da_T_anom = xr.open_dataarray(savename)
    elif 'SPFH' in this_varname:
        da_q_anom = xr.open_dataarray(savename)

da_T_anom = da_T_anom.assign_coords({'longitude': np.round(da_T_anom.longitude, 2)})
da_T_anom = da_T_anom.assign_coords({'latitude': np.round(da_T_anom.latitude, 2)})

da_q_anom = da_q_anom.assign_coords({'longitude': np.round(da_q_anom.longitude, 2)})
da_q_anom = da_q_anom.assign_coords({'latitude': np.round(da_q_anom.latitude, 2)})

# For gridboxes that have data, save in same manner as ISD
# "station_id" will be lat-lon
lons, lats = np.meshgrid(da_T_anom.longitude, da_T_anom.latitude)
has_data = ~np.isnan(da_T_anom[0, :, :])
lons = lons[has_data.values]
lats = lats[has_data.values]

station_id = ['%03.2f%03.2f' % (this_lat, this_lon) for (this_lat, this_lon) in zip(lats, lons)]

metadata = pd.DataFrame({'station_id': station_id,
                         'lat': np.round(lats, 2),
                         'lon': np.round(lons, 2)})

datadir = '/glade/work/mckinnon/JRA55/csv'
metadata.to_csv('%s/new_metadata.csv' % datadir)

# Iterate through lats, lons and make dataframes
for counter in range(len(lats)):
    this_lat = lats[counter]
    this_lon = lons[counter]

    station_id = '%03.2f%03.2f' % (this_lat, this_lon)
    print(station_id)

    this_q_ts = da_q_anom.sel(latitude=this_lat, longitude=this_lon)
    this_T_ts = da_T_anom.sel(latitude=this_lat, longitude=this_lon)

    this_df = this_q_ts.to_dataframe(name='Q_anom')
    this_df = this_df.assign(TMP_anom=this_T_ts.values)

    this_df = this_df.reset_index()

    # rename date column
    this_df = this_df.rename(columns={'time': 'date'})

    # drop columns we don't need
    this_df = this_df.drop(columns=['latitude', 'longitude', 'dayofyear'])

    # save
    this_df.to_csv('%s/%s.csv' % (datadir, station_id), index=False)

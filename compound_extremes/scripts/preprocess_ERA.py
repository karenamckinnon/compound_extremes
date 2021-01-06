"""Preproces ERA5 data to look like ISD/GSOD csv files."""

import numpy as np
import xarray as xr
import pandas as pd
from glob import glob
import geopandas
from shapely.geometry import shape
import geojson


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


def subset_to_west(da, lon_range, lat_range, roi):
    import salem  # noqa
    da = da.sel(latitude=slice(lat_range[1], lat_range[0]),
                longitude=slice(lon_range[0], lon_range[1]))

    return da.salem.roi(geometry=roi, crs='wgs84')


era_version = '5'

if era_version == '5':
    f_Td = '/glade/work/mckinnon/ERA5/day/Td/ERA5_Td_US.nc'
    f_T = '/glade/work/mckinnon/ERA5/day/2mT/ERA5_2mT_US.nc'
    f_p = '/glade/work/mckinnon/ERA5/day/p/ERA5_p_US.nc'

    da_Td = xr.open_dataarray(f_Td)  # K
    da_T = xr.open_dataarray(f_T)  # K
    da_p = xr.open_dataarray(f_p)  # Pa

elif era_version == '20C':
    f_Td = sorted(glob('/glade/work/mckinnon/ERA20C/6h/Td/*nc'))
    f_T = sorted(glob('/glade/work/mckinnon/ERA20C/6h/T2m/*nc'))
    f_p = sorted(glob('/glade/work/mckinnon/ERA20C/6h/p/*nc'))

    da_Td = xr.open_mfdataset(f_Td, concat_dim='initial_time0_hours').Td
    da_T = xr.open_mfdataset(f_T, concat_dim='initial_time0_hours').T2m
    da_p = xr.open_mfdataset(f_p, concat_dim='initial_time0_hours').p

    rename_dict = {'g4_lat_1': 'latitude', 'g4_lon_2': 'longitude', 'initial_time0_hours': 'time'}

    da_Td = da_Td.rename(rename_dict)
    da_T = da_T.rename(rename_dict)
    da_p = da_p.rename(rename_dict)

    da_Td = da_Td.assign_coords(longitude=(((da_Td.longitude + 180) % 360) - 180))
    da_T = da_T.assign_coords(longitude=(((da_T.longitude + 180) % 360) - 180))
    da_p = da_p.assign_coords(longitude=(((da_p.longitude + 180) % 360) - 180))

# Subset to smaller area
da_Td = subset_to_west(da_Td, lon_range, lat_range, interior_west_shapely)
da_T = subset_to_west(da_T, lon_range, lat_range, interior_west_shapely)
da_p = subset_to_west(da_p, lon_range, lat_range, interior_west_shapely)

# Calculate q
e = 6.112*np.exp((17.67*(da_Td - 273.15))/((da_Td - 273.15) + 243.5))
da_q = 1000*(0.622 * e)/((da_p/100) - (0.378 * e))  # g/kg

# Calculate daily averages
da_T = da_T.resample(time='D').mean()
da_q = da_q.resample(time='D').mean()

# For gridboxes that have data, save in same manner as ISD
# "station_id" will be lat-lon
lons, lats = np.meshgrid(da_T.longitude, da_T.latitude)
has_data = ~np.isnan(da_T[0, :, :])
lons = lons[has_data]
lats = lats[has_data]

station_id = ['%03.2f%03.2f' % (this_lat, this_lon) for (this_lat, this_lon) in zip(lats, lons)]

metadata = pd.DataFrame({'station_id': station_id,
                         'lat': lats,
                         'lon': lons})

datadir = '/glade/work/mckinnon/ERA5/csv'
metadata.to_csv('%s/new_metadata.csv' % datadir)

# Iterate through lats, lons and make dataframes
for counter in range(len(lats)):
    if counter % 20 == 0:
        print(counter)
    this_lat = lats[counter]
    this_lon = lons[counter]

    station_id = '%03.2f%03.2f' % (this_lat, this_lon)

    this_q_ts = da_q.sel(latitude=this_lat, longitude=this_lon)
    this_T_ts = da_T.sel(latitude=this_lat, longitude=this_lon)

    this_df = this_q_ts.to_dataframe(name='Q')
    this_df = this_df.assign(TMP=this_T_ts.values)

    this_df = this_df.reset_index()

    # rename date column
    this_df = this_df.rename(columns={'time': 'date'})

    # drop columns we don't need
    this_df = this_df.drop(columns=['latitude', 'longitude'])

    # save
    this_df.to_csv('%s/%s.csv' % (datadir, station_id), index=False)

import numpy as np
import xarray as xr
import pandas as pd
from humidity_variability.models import fit_linear_QR
import ctypes
from glob import glob
import argparse


def main(start_index, nstations, nboot):
    # Params and dirs
    datadir = '/home/mckinnon/bucket/gsod'
    cvdp_loc = '/home/mckinnon/bucket/CVDP'

    start_year = 1973
    end_year = 2018
    expand_data = True  # should the start/end year be the edges of the data, or a minimum requirement?
    search_query = {'begin': 'datetime(%i, 1, 1)' % start_year,
                    'end': 'datetime(%i, 12, 31)' % end_year}

    hashable = tuple((tuple(search_query.keys()), tuple(search_query.values()), expand_data))
    query_hash = str(ctypes.c_size_t(hash(hashable)).value)  # ensures positive value
    output_dir = '%s/%s/qr' % (datadir, query_hash)

    constraint = 'None'

    qs = 0.05, 0.5, 0.95

    # Load station data
    fname_alldata = '%s/%s/all_stations.csv' % (datadir, query_hash)

    # Annual mean is removed from temperature and dewpoint
    # Units are still F
    # The seasonal cycle has _not_ been removed but is in the data frame
    alldata = pd.read_csv(fname_alldata)
    station_ids = np.unique(alldata['station_id'])

    # Load in GM-EM
    varname = 'tas'
    fnames = sorted(glob('%s/CESM1-CAM5-BGC-LE_*.cvdp_data.1920-2017.nc' % cvdp_loc))
    cvdp_name = '%s_global_avg_mon' % varname
    ds = xr.open_mfdataset(fnames, decode_times=False, concat_dim='ensemble')
    da = ds[cvdp_name]
    del ds
    gm_em = da.mean('ensemble')
    del da
    gm_em_time = pd.date_range(start='1920-01', periods=len(gm_em), freq='M')
    gm_em = gm_em.assign_coords(time=gm_em_time)
    gm_em_ann = gm_em.groupby('time.year').mean('time')

    # estimate 2018 GM-EM as average trend over past five years
    delta = (gm_em_ann.values[-1] - gm_em_ann.values[-5])/5

    ids_to_run = station_ids[start_index:(start_index + nstations)]
    savename = '%s/qr_results_%03d.npz' % (output_dir, start_index)

    BETA = np.empty((len(ids_to_run), len(qs), 2))
    BETA_BOOT = np.empty((len(ids_to_run), len(qs), nboot, 2))

    for station_counter, this_id in enumerate(ids_to_run):
        print(this_id)
        this_data = alldata.loc[alldata['station_id'] == this_id, :]

        # Add GM-EM to dataframe
        # If boreal winter, associate with January year
        months = np.array([pd.datetime.strptime(d, '%Y-%m-%d').month for d in this_data['date']])
        years = np.array([pd.datetime.strptime(d, '%Y-%m-%d').year for d in this_data['date']])

        if np.isin(1, months):  # spanning the calendar year
            years[months == 10] += 1
            years[months == 11] += 1
            years[months == 12] += 1

        this_data = this_data.assign(year=years)

        this_data = this_data.assign(gmt=np.nan*np.ones(len(this_data)))
        for yy in np.unique(this_data.year):
            if yy <= 2017:
                this_gmt = gm_em_ann.loc[gm_em_ann['year'] == yy].values
                this_data.loc[this_data['year'] == yy, 'gmt'] = this_gmt
            else:  # only applies to 2018
                dyears = yy - 2017
                this_gmt = gm_em_ann.loc[gm_em_ann['year'] == int(yy - dyears)].values + delta*dyears
                this_data.loc[this_data['year'] == yy, 'gmt'] = this_gmt

        # Add jitter
        # Data is rounded to 0.1F, so jitter is +/- 0.05F
        jitter = 0.05*np.random.rand(len(this_data)) - 0.05/2
        this_data['dewp'] += jitter

        # Remove seasonal cycle
        this_data['dewp_anom'] = this_data['dewp'] - this_data['dewp_clim']

        # Convert to Celsius
        this_data['dewp_anom'] *= 5/9

        # cut off anything outside of our period of interest
        idx_keep = (this_data['year'] >= start_year) & (this_data['year'] <= end_year).values
        this_data = this_data.loc[idx_keep, :]

        # drop missing
        this_data.dropna(subset=['dewp_anom'], inplace=True)

        unique_years = np.unique(this_data['year'].values)
        beta = np.empty((len(qs), 2))
        beta_boot = np.empty((len(qs), nboot, 2))
        for ct, q in enumerate(qs):

            x1 = this_data['gmt'].values
            x1 -= np.mean(x1)
            X = np.vstack((np.ones(len(x1)), x1)).T
            data = this_data['dewp_anom'].values

            this_beta, yhat = fit_linear_QR(X, data, q, constraint, 'None')
            beta[ct, :] = this_beta
            residuals = data - yhat

            for kk in range(nboot):
                # Resample in yearly blocks
                new_years = np.random.choice(unique_years, len(unique_years))
                df_list = []
                for year_counter, yy in enumerate(new_years):
                    # use the original year for yhat
                    original_year = unique_years[year_counter]
                    # yhat is the same for a given year (no within year variation)
                    fitted_val = yhat[this_data['year'] == original_year][0]
                    # grab the residual from a different year
                    residual_sample = residuals[this_data['year'] == yy]
                    # but the gmt from the original year
                    this_gmt = this_data.loc[this_data['year'] == original_year, 'gmt'].values[0]
                    new_df = pd.DataFrame(data={'gmt': this_gmt*np.ones(len(residual_sample)),
                                                'dewp_anom_sample': fitted_val + residual_sample})
                    df_list.append(new_df)
                resampled_df = pd.concat(df_list)

                x1 = resampled_df['gmt'].values
                x1 -= np.mean(x1)
                X = np.vstack((np.ones(len(x1)), x1)).T
                data = resampled_df['dewp_anom_sample'].values

                this_beta, _ = fit_linear_QR(X, data, q, constraint, 'None')
                beta_boot[ct, kk, :] = this_beta

        BETA[station_counter, ...] = beta
        BETA_BOOT[station_counter, ...] = beta_boot
    np.savez(savename, ids_to_run=ids_to_run, BETA=BETA, BETA_BOOT=BETA_BOOT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('start_index', type=int, help='Which index among stations to start at')
    parser.add_argument('nstations', type=int, help='How many stations to analyze on this machine')
    parser.add_argument('nboot', type=int, help='How many bootstrap samples to do')

    args = parser.parse_args()

    main(args.start_index, args.nstations, args.nboot)

import os
from subprocess import check_call
import pandas as pd
from datetime import datetime
import numpy as np
pd.set_option('max_columns', None)


# Must start at or before start year
start_date = datetime.strptime('1973-06-01', '%Y-%m-%d')
end_date = datetime.strptime('2019-09-30', '%Y-%m-%d')
ctry = 'US'
exclude_states = ('AK', 'HI')

# Get latest metadata
tmpdir = '/glade/scratch/mckinnon/tmp'
url_base = 'https://www.ncei.noaa.gov/data/global-hourly/access'
url_history = 'ftp://ftp.ncei.noaa.gov/pub/data/noaa/isd-history.csv'
hist_name = url_history.split('/')[-1]

savedir = '/glade/work/mckinnon/ISD'

if not os.path.isfile('%s/%s' % (savedir, hist_name)):
    cmd = 'wget -q -O %s/%s %s' % (savedir, hist_name, url_history)
    check_call(cmd.split())

df_meta = pd.read_csv('%s/%s' % (savedir, hist_name))
dt_begin = np.array([datetime.strptime(str(d), '%Y%m%d') for d in df_meta['BEGIN']])
dt_end = np.array([datetime.strptime(str(d), '%Y%m%d') for d in df_meta['END']])
idx_use = ((dt_begin <= start_date) & (dt_end >= end_date) & (df_meta['CTRY'] == ctry)).values

df_meta = df_meta.assign(BEGIN=dt_begin, END=dt_end)

for state in exclude_states:
    idx_use = idx_use & (df_meta['STATE'] != state)

df_meta = df_meta[idx_use].reset_index()


def remove_bad_rows(df):
    # remove SOURCE:
    # 2: failed cross checks
    # A: failed cross checks
    # B: failed cross checks
    # O: Summary observation created by NCEI using hourly observations that
    #    may not share the same data source flag
    # 9: missing

    # remove REPORT_TYPE
    # 99999: missing

    # remove CALL_SIGN
    # 99999: missing

    # remove QUALITY_CONTROL
    # V01: no quality control

    bad_idx = ((df['SOURCE'].astype(str) == '2') |
               (df['SOURCE'].astype(str) == 'A') |
               (df['SOURCE'].astype(str) == 'B') |
               (df['SOURCE'].astype(str) == 'O') |
               (df['SOURCE'].astype(str) == '9') |
               (df['REPORT_TYPE'].astype(str) == '99999') |
               (df['CALL_SIGN'].astype(str) == '99999') |
               (df['QUALITY_CONTROL'].astype(str) == 'V010'))

    return df[~bad_idx]


def remove_bad_vals(df, varname):
    """
    Remove values from
    """
    flag = np.array([d.split(',')[-1] for d in df[varname]])
    flag = flag.astype(str)
    vals_tmp = np.array([int(d.split(',')[0]) for d in df[varname]])

    if ((varname == 'DEW') | (varname == 'TMP')):
        bad_idx = ((flag == '2') |
                   (flag == '3') |
                   (flag == '6') |
                   (flag == '7') |
                   (flag == 'A') |
                   (flag == 'C') |
                   (vals_tmp == 9999))
    elif varname == 'SLP':
        bad_idx = ((flag == '2') |
                   (flag == '3') |
                   (flag == '6') |
                   (flag == '7') |
                   (vals_tmp == 99999))

    vals = vals_tmp.astype(float)/10

    vals[bad_idx] = np.nan
    df = df.assign(**{varname: vals})

    return df


usecols = ['SOURCE', 'REPORT_TYPE', 'CALL_SIGN', 'QUALITY_CONTROL', 'DATE', 'ELEVATION', 'DEW', 'TMP', 'SLP']
keepcols = ['DATE', 'DEW', 'TMP', 'SLP']
has_data = np.ones(len(df_meta)).astype(bool)

for ct, row in df_meta.iterrows():
    savename = '%s/csv/%06d-%05d.csv' % (savedir, int(row['USAF']), int(row['WBAN']))
    if os.path.isfile(savename):
        continue
    print('%i/%i' % (ct, len(df_meta)))

    # download files for each year
    station_id = '%06d%05d' % (int(row['USAF']), int(row['WBAN']))
    yy1 = row['BEGIN'].year
    yy2 = row['END'].year

    # download all files
    all_df = []
    for yy in range(yy1, yy2 + 1):

        url = '%s/%i/%s.csv' % (url_base, yy, station_id)
        fname = '%s_%i.csv' % (station_id, yy)
        cmd = 'wget -q -O %s/%s %s' % (tmpdir, fname, url)
        try:
            check_call(cmd.split())
        except Exception as e:  # file not available
            print(str(e))
            continue

        df = pd.read_csv('%s/%s' % (tmpdir, fname), usecols=usecols, low_memory=False)
        elev = df.loc[0, 'ELEVATION']  # meters

        df = remove_bad_rows(df)
        df = df[keepcols]

        # convert data to float
        for varname in ('TMP', 'DEW', 'SLP'):
            df = remove_bad_vals(df, varname)

        cmd = 'rm -f %s/%s' % (tmpdir, fname)
        check_call(cmd.split())
        all_df.append(df)

    if len(all_df) > 0:
        all_df = pd.concat(all_df).reset_index()
    else:  # no files available
        has_data[ct] = False
        continue

    # Estimate station pressure (as per Willett et al 2014)
    stp = all_df['SLP']*((all_df['TMP'] + 273.15)/(all_df['TMP'] + 273.15 + 0.0065*elev))**5.625

    # Calculate q (g/kg)
    e = 6.112*np.exp((17.67*all_df['DEW'])/(all_df['DEW'] + 243.5))
    q = 1000*(0.622 * e)/(stp - (0.378 * e))  # g/kg

    all_df = all_df.assign(STP=stp, Q=q)
    dt = [datetime.strptime(d, '%Y-%m-%dT%H:%M:%S') for d in all_df['DATE']]
    all_df.index = dt

    resampler = all_df.resample('D')
    count = resampler.count()
    avg_df = resampler.mean()
    avg_df[count < 4] = np.nan

    # get rid of extra index
    avg_df = avg_df.iloc[:, 1:]

    # Fix title of date column
    avg_df = avg_df.reset_index()
    avg_df = avg_df.rename(columns={'index': 'date'})

    # Make sure there are a reasonable number of values
    if np.sum(~np.isnan(avg_df['Q'])) < 100:
        has_data[ct] = False
    else:
        # Save
        avg_df.to_csv(savename, index=False)

# remove stations with no data from metadata
df_meta = df_meta[has_data]

# Change to lowercase headings, add station_id, and save
df_meta.columns = [c.lower() for c in df_meta.columns]
df_meta = df_meta.assign(station_id='%s-%s' % (df_meta['usaf'], df_meta['wban']))
df_meta.to_csv('%s/csv/new_metadata.csv' % savedir)

import numpy as np
import pandas as pd
from humidity_variability.utils import add_date_columns, jitter, add_GMT
from helpful_utilities.meteo import F_to_C
from helpful_utilities.general import lowpass_butter


def fit_seasonal_cycle(doy, data, nbases=5):
    """Fit seasonal cycle of daily data with specified number of Fourier bases.

    Parameters
    ----------
    doy : numpy.ndarray
        Day of year for each data value
    data : numpy.ndarray
        Data values for seasonal fit
    nbases : int
        Number of Fourier bases to use. Default is 5.

    Returns
    -------
    rec : numpy.ndarray
        Reconstructed seasonal structure, of same length as data
    residual : numpy.ndarray
        The residual from the seasonal fit.
    rec_ann : numpy.ndarray
        The 365-day version of the seasonal cycle
    """

    mu = np.mean(data)
    data -= mu

    t_basis = (doy - 0.5)/365
    ann_basis = (np.arange(1, 366) - 0.5)/365
    nt = len(t_basis)
    bases = np.empty((nbases, nt), dtype=complex)
    bases_ann = np.empty((nbases, 365), dtype=complex)
    for counter in range(nbases):
        bases[counter, :] = np.exp(2*(counter + 1)*np.pi*1j*t_basis)
        bases_ann[counter, :] = np.exp(2*(counter + 1)*np.pi*1j*ann_basis)

    coeff = 2/nt*(np.dot(bases, data))

    rec = np.real(np.dot(np.conj(coeff), bases))
    residual = data - rec

    rec_ann = np.real(np.dot(np.conj(coeff), bases_ann)) + mu  # add mean value back into climatology

    return rec, residual, rec_ann


def fit_seasonal_cycle_lowpass(doy, data, cut_freq=1/30):
    """Estimate the seasonal cycle by using a lowpass filter on the empirical seasonal cycle.

    Parameters
    ----------
    doy : numpy.ndarray
        Day of year for each data value
    data : numpy.ndarray
        Data values for seasonal fit
    cut_freq : float
        Cutoff frequency (in 1/days) for the lowpass filter

    Returns
    -------
    rec : numpy.ndarray
        Reconstructed seasonal structure, of same length as data
    residual : numpy.ndarray
        The residual from the seasonal fit.
    rec_ann : numpy.ndarray
        The 365-day version of the seasonal cycle
    """

    tmp_df = pd.DataFrame({'doy': doy, 'data': data})
    empirical_sc = tmp_df.groupby('doy').mean()
    ann_doy = empirical_sc.index
    smooth_sc = lowpass_butter(1, cut_freq, 3, empirical_sc.values.flatten())

    residual = np.empty_like(data)
    for counter, this_doy in enumerate(ann_doy):
        match_idx = doy == this_doy
        smooth_sc_val = smooth_sc[counter]
        residual[match_idx] = data[match_idx] - smooth_sc_val

    rec = data - residual
    rec_ann = smooth_sc

    return rec, residual, rec_ann


def calculate_amplification_index2(df, meta, T0, half_width, grouping, fit_data, qs, this_q=0.05):
    """Calculate the fraction of hot days that are dry within a grouping of stations or gridboxes.

    This version accounts for the temperature dependence of q5

    Parameters
    ----------
    df : pandas.DataFrame
        Contains the temperature and humidity anomaly data for all stations/gridboxes and times of interest.
    meta : pandas.DataFrame
        Contains (at least) the weights for each station/gridbox, as well as the station_id
    T0 : float
        The middle percentile to define hot days
    half_width : float
        The half-width around T0 to consider a hot day
    grouping : string
        Can be 'month' or 'year': how are hot and hot/dry days grouped?
    fit_data : string
        Full path to npz file containing the relevant parameters from the quantile smoothing spline fit
    qs : numpy.ndarray
        The quantiles fit by the QSS model
    this_q : float
        The threshold below which to consider a day "dry"

    Returns
    -------
    amplification : numpy.ndarray
        Time series of the amplification index

    """

    # names of temperature and humidity variables
    humidity_var = 'Q'
    temp_var = 'TMP'
    # For each station or gridbox, map temperatures to percentiles

    # Load fitted QR model
    df_fit = np.load(fit_data)
    q_quantiles = df_fit['s0_H'][:, qs == this_q, :].squeeze()
    temperature_percentiles = df_fit['temperature_percentiles']
    lat_vec = np.round(df_fit['lats'], decimals=3)
    lon_vec = np.round(df_fit['lons'], decimals=3)

    # Calculate percentiles at each station of temperature
    df.loc[:, '%s_perc' % temp_var] = df.groupby('station_id')['%s_anom' % temp_var].rank(pct=True)

    # Get humidity threshold for each temperature at each station
    df_updated = []
    for this_station in np.unique(df['station_id']):
        tmp_df = df.loc[df['station_id'] == this_station].reset_index()
        this_lat = np.round(tmp_df['lat'][0], decimals=3)
        this_lon = np.round(tmp_df['lon'][0], decimals=3)
        match_idx = (lat_vec == this_lat) & (lon_vec == this_lon)
        this_quantile = q_quantiles[:, match_idx].squeeze()
        # Interpolate
        this_quantile_interp = np.interp(tmp_df['%s_perc' % temp_var],
                                         temperature_percentiles/100,
                                         this_quantile)
        tmp_df.loc[:, '%s_cut' % humidity_var] = this_quantile_interp
        df_updated.append(tmp_df)

    df = pd.concat(df_updated).reset_index()
    del df_updated
    # Drop spare index columns
    df = df.drop(df.columns[:2], axis='columns')

    # Assign each day a binary index for whether it is hot, and whether it is hot and dry
    df = df.assign(is_hot=np.nan*np.ones(len(df)))
    df = df.assign(is_hot_dry=np.nan*np.ones(len(df)))
    for station in np.unique(df['station_id']):
        tmp_df = df.loc[df['station_id'] == station]

        is_hot = ((tmp_df['%s_perc' % temp_var] > (T0 - half_width)/100) &
                  (tmp_df['%s_perc' % temp_var] < (T0 + half_width)/100))

        is_hot_dry = (is_hot &
                      (tmp_df['%s_anom' % humidity_var] < tmp_df['%s_cut' % humidity_var])).astype(float)

        is_hot = is_hot.astype(float)

        df.loc[df['station_id'] == station, 'is_hot_dry'] = is_hot_dry
        df.loc[df['station_id'] == station, 'is_hot'] = is_hot

    weights = meta.set_index('station_id')
    if grouping == 'month':
        groupby_names = ['year', 'month', 'station_id']
    elif grouping == 'year':
        groupby_names = ['year', 'station_id']

    hot_dry_weighted = df.groupby(groupby_names)['is_hot_dry'].sum()*weights['area_weights']
    hot_weighted = df.groupby(groupby_names)['is_hot'].sum()*weights['area_weights']
    # Since the amplification index is the ratio of hot, dry to hot, we don't need to normalize the weights
    # The same number of stations are present in each month/year combo for both metrics, so will cancel out
    # But note that any analysis of the hot, dry or hot time series alone has not been normalized
    # appropriately
    amplification = (hot_dry_weighted.groupby(groupby_names[:-1]).sum() /
                     hot_weighted.groupby(groupby_names[:-1]).sum())

    return amplification


def preprocess_data(this_id, datadir, start_year, end_year, start_month, end_month, offset, spread):
    """
    Preprocess GSOD data, getting rid of bad data, and removing stations with insufficient data.

    Also subsets data to desired (start_year, end_year) and (start_month, end_month)

    TODO: Move hard coded things to args.

    Parameters
    ---------
    this_id : str
        GSOD station id
    datadir : str
        Location of GSOD csv files
    start_year : int
        First year of analysis. Will subset to this range.
    end_year : int
        Last year of analysis. Will subset to this range.
    start_month : int
        First month of analysis. Will subset to this range.
    end_month : int
        Last month of analysis. Will subset to this range.
    offset : float
        For jittering, should data be offset?
    spread : float
        For jittering, what is the uncertainty in the data?

    Returns
    -------
    0 if insufficient data
    otherwise
    df : pandas.dataframe
        Dataframe containing subset data for station
    """

    f = '%s/%s.csv' % (datadir, this_id)
    df = pd.read_csv(f)

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
        return 0

    # add GMT anoms
    df = add_GMT(df)

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
    df = df.loc[(df['month'] >= start_month) & (df['month'] <= end_month)]

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
        return 0
    else:
        return df

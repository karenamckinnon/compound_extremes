import numpy as np
import pandas as pd
from humidity_variability.utils import add_date_columns, jitter, add_GMT
from helpful_utilities.meteo import F_to_C


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

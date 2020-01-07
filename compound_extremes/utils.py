import numpy as np


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

    data -= np.mean(data)

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

    rec_ann = np.real(np.dot(np.conj(coeff), bases_ann))

    return rec, residual, rec_ann

'''Subtract background
'''
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.interpolate import interp1d
from astropy.io import fits
from astropy.stats import sigma_clip
from .dqflag import is_dqflagged
from .assign_wcs import wcs_calfits


##
def subtract_1fnoises_from_detector(data, dq, move=5, axis=0):
    '''Subtract 1/f noises along spatial (y or axis=0) axis.

    1/f noises is made by moving average of median values.
    To see the background levels of the detector,
    this function specifies an area not affected by sky
    by using DO_NOT_USE in the dq array.
    '''
    data_mask = np.copy(data)

    idx_DoNotUse = dq == 1  # DO_NOT_USE
    idx_DoNotUse2 = dq == 5  # DO_NOT_USE & JUMP_DET
    idx_detector = idx_DoNotUse | idx_DoNotUse2

    data_mask[~idx_detector] = np.nan
    data_mask[900:1120, :] = np.nan  # to remove fixed slits
    data_mask[:100, :] = np.nan  # to remove lower edge
    data_mask[1950:, :] = np.nan  # to remove upper edge

    data_clipped = sigma_clip(
        data_mask, sigma=3, maxiters=None, masked=False, axis=axis
    )
    data1d_ymed = np.nanmean(data_clipped, axis=axis)
    background = moving_average(data1d_ymed, 5)
    return data - np.expand_dims(background, axis)


def subtract_global_background(input_model):
    '''Subtract global background depending on wavelength.'''
    data = input_model.data
    dq = input_model.dq
    is_ok = ~is_dqflagged(dq, 'DO_NOT_USE')

    radecw = wcs_calfits(input_model)
    wavelength = radecw[2][is_ok]

    # compute global background
    idx = np.argsort(wavelength.ravel())
    flux1d = data[is_ok][idx]
    is_sigmaclipped = sigma_clip(flux1d, sigma=5, maxiters=None, masked=True).mask
    flux1d = data[is_ok][idx][~is_sigmaclipped]
    global_background = moving_average(flux1d, 50001)

    # interpolation
    idx_effective = global_background != 0
    effective_global_background = global_background[idx_effective]
    wave1d = wavelength[idx][~is_sigmaclipped][idx_effective]
    f = interp1d(
        wave1d, effective_global_background, kind='nearest', fill_value='extrapolate'
    )
    global_background = f(wavelength[idx])

    dummy = data[is_ok]
    dummy[idx] = dummy[idx] - global_background
    data[is_ok] = dummy
    return input_model, dummy


def subtract_bacground(data, dq, move=5, axis=0):
    '''Subtract background along spatial (y or axis=0) axis.

    Background is made by moving average of median values.
    '''
    data_mask = np.copy(data)
    data_mask[is_dqflagged(dq, 'DO_NOT_USE')] = np.nan
    sigma_clip_array = sigma_clip(
        data_mask, sigma=3, maxiters=None, masked=True, axis=0
    )
    data_mask[sigma_clip_array.mask == 1] = np.nan
    data1d_ymed = np.nanmedian(data_mask, axis=axis)
    background = moving_average(data1d_ymed, 5)
    return data - np.expand_dims(background, axis)


def moving_average(spec1d, n=5):
    '''Give moving average.'''
    move_avg = np.convolve(spec1d, np.ones(n), 'valid') / n
    padding = np.zeros(n // 2)
    move_avg = np.append(move_avg, padding)
    move_avg = np.append(padding, move_avg)
    return move_avg


@dataclass
class ConfigSubtractBackground:
    skip: bool = False
    move_pixels: int = 5


@dataclass
class ConfigSubtractGlobalBackground:
    skip: bool = False


def main():
    '''Example. We didn't change error spectra.'''
    fnames = [
        'calib/calib4th/jw01840017001_02101_00001_nrs1_rate_clipped_edgemask_cal.fits',
        'calib/calib4th/jw01840017001_02101_00001_nrs2_rate_clipped_edgemask_cal.fits',
        'calib/calib4th/jw01840017001_02101_00002_nrs1_rate_clipped_edgemask_cal.fits',
        'calib/calib4th/jw01840017001_02101_00002_nrs2_rate_clipped_edgemask_cal.fits',
        'calib/calib4th/jw01840017001_02101_00003_nrs1_rate_clipped_edgemask_cal.fits',
        'calib/calib4th/jw01840017001_02101_00003_nrs2_rate_clipped_edgemask_cal.fits',
        'calib/calib4th/jw01840017001_02101_00004_nrs1_rate_clipped_edgemask_cal.fits',
        'calib/calib4th/jw01840017001_02101_00004_nrs2_rate_clipped_edgemask_cal.fits',
    ]
    for fn in fnames:
        with fits.open(fn) as hdul:
            data = hdul[1].data
            dq = hdul[3].data
            hdul[1].data = subtract_bacground(data, dq)
            hdul.writeto(fn.replace('_cal', '_sub_cal'))
    return


if __name__ == '__main__':
    main()

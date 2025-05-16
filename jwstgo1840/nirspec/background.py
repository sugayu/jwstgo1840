'''Subtract background
'''

from __future__ import annotations
import warnings
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import interp1d
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.utils.exceptions import AstropyUserWarning
from jwst.datamodels import IFUImageModel
from gwcs import wcstools
from .dqflag import is_dqflagged
from .assign_wcs import wcs_calfits, get_nrs_wcs_slit, change_nrs_wcs_slit


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

    if np.sum(idx_detector) > 1e5:  # HACK: 1e5 is arbitrary
        # This threshold checks whether the slit positions are
        # already known. If not, use almost all pixels.
        data_mask[~idx_detector] = np.nan
    data_mask[900:1120, :] = np.nan  # to remove fixed slits
    data_mask[:100, :] = np.nan  # to remove lower edge
    data_mask[1950:, :] = np.nan  # to remove upper edge

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=AstropyUserWarning)
        data_clipped = sigma_clip(
            data_mask, sigma=3, maxiters=None, masked=False, axis=axis
        )
    data1d_ymed = np.nanmean(data_clipped, axis=axis)
    background = moving_average(data1d_ymed, 5)

    return data - np.expand_dims(background, axis)


def subtract_slits_background(input_model: IFUImageModel) -> IFUImageModel:
    '''Subtract slit backgrounds depending detector and slits.

    Current codes work after global background subtraction,
    but the best timing to work is under consideration.

    When computing backgrounds, this function separates data into two and
    computes two backgrounds. This is because pixel values oscillate pixel by pixel.
    Backgrounds vary from one skip to the next.
    '''
    nslits = 30  # for NIRSpec IFU
    data = input_model.data
    dq = input_model.dq
    is_flagged = is_dqflagged(dq, 'DO_NOT_USE')
    _data = data.copy()
    _data[is_flagged] = np.nan

    for i in range(nslits):
        if i == 0:
            slice_wcs = get_nrs_wcs_slit(input_model, i)
        else:
            slice_wcs = change_nrs_wcs_slit(input_model, slice_wcs, i)

        x, y = wcstools.grid_from_bounding_box(slice_wcs.bounding_box)
        x, y = x.astype(int), y.astype(int)
        slit = _data[y, x]

        # Two backgrounds estimated because of oscillation
        background1 = np.nanmedian(slit[0::2, :])
        background2 = np.nanmedian(slit[1::2, :])

        data[y[0::2, :], x[0::2, :]] -= background1
        data[y[1::2, :], x[1::2, :]] -= background2

    return input_model


def subtract_global_background(
    input_model: IFUImageModel, move_pixels: int = 50001
) -> tuple[IFUImageModel, np.ndarray]:
    '''Subtract global background depending on wavelength.

    Global background is computed with object mask, flagged with "OUTLIER".
    The computed background is also subtracted in masked pixels.

    Args:
        input_model (IFUImageModel): intened input is _cal.fits after the SPEC2 step.

    Returns:
        tuple[IFUImageModel, np.ndarray]: background-subtracted datamodel and
            2d-background.

    Note:
        input_model will be overwritten by this function because deepcopy takes
        long time for the datamodel.

    Examples:
        >>> datamodel, bk2d = subtract_global_background(datamodel)
    '''
    data = input_model.data
    dq = input_model.dq
    is_ok = ~is_dqflagged(dq, 'DO_NOT_USE')
    is_not_masked = ~is_dqflagged(dq, 'OUTLIER')
    _data = data.copy()
    _data[~(is_ok | is_not_masked)] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=AstropyUserWarning)
        is_not_clipped = ~sigma_clip(
            _data, sigma=5, maxiters=None, masked=True, axis=0
        ).mask
    idx_sky = is_ok & is_not_masked & is_not_clipped
    radecw = wcs_calfits(input_model)

    global_background_2d = np.zeros_like(data)
    for idx_amp in get_amplifier_patterns():  # independent bkgsub for 4 amplifiers
        wavelength = radecw[2][is_ok & idx_amp]
        wavelength_sky = radecw[2][idx_sky & idx_amp]
        data_sky = data[idx_sky & idx_amp]

        # compute global background
        order_skypixels = np.argsort(wavelength_sky)
        flux1d = data_sky[order_skypixels]
        global_background_1d = moving_average(flux1d, move_pixels)

        # interpolation function
        idx_effective = global_background_1d != 0
        effective_global_background = global_background_1d[idx_effective]
        wave1d = wavelength_sky[order_skypixels][idx_effective]
        f = interp1d(
            wave1d,
            effective_global_background,
            kind='nearest',
            fill_value='extrapolate',
        )

        # background_2d including masked pixels
        background_ok = global_background_2d[is_ok & idx_amp]
        order_fullpixels = np.argsort(wavelength)
        background_ok[order_fullpixels] = f(wavelength[order_fullpixels])
        global_background_2d[is_ok & idx_amp] = background_ok

    data -= global_background_2d
    return input_model, global_background_2d


def subtract_bacground(data, dq, move=5, axis=0):
    '''Subtract background along spatial (y or axis=0) axis.

    Background is made by moving average of median values.
    '''
    data_mask = np.copy(data)
    data_mask[is_dqflagged(dq, 'DO_NOT_USE')] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=AstropyUserWarning)
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


def get_amplifier_patterns() -> np.ndarray:
    '''Get bool array indicating pixels read by a amplifier.

    The IRS2 readout pattern of NIRSpec IFU uses four amplifiers when reading
    pixel values. This function iterately returns pixel positions that are read
    by an amplifier.
    '''
    n_amp = 4
    n_pix = 2048

    idx_amplifier = np.zeros((n_pix, n_pix)).reshape(n_amp, -1, n_pix).astype(bool)
    for i in range(n_amp):
        idx_amp = idx_amplifier.copy()
        idx_amp[i] = True
        yield idx_amp.reshape(-1, n_pix)


@dataclass
class ConfigSubtractBackground:
    skip: bool = False
    move_pixels: int = 5


@dataclass
class ConfigSubtractGlobalBackground:
    skip: bool = False
    save_results: bool = False
    move_pixels: int = 50001


@dataclass
class ConfigSubtractSlitsBackground:
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

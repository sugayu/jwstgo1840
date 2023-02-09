'''Subtract background
'''
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from astropy.io import fits


##
def subtract_bacground(data, dq, move=5, axis=0):
    '''Subtract background along spatial (y or axis=0) axis.

    Background is made by moving average of median values.
    '''
    data_mask = np.copy(data)
    data_mask[dq % 2 == 1] = np.nan
    data1d_ymed = np.nanmedian(data, axis=axis)
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

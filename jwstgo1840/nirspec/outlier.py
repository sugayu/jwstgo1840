'''Remove outlier
'''
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip
from jwst import datamodels
from .dqflag import dqflagging, is_dqflagged


##
def sigmaclip(data, dq, sigma=10):
    '''Sigma clipping to flag outliers after Detector1

    Save two files with suffix of "_pixelmask" and "_rate_clipped".
    '''
    sigma_clip_array = sigma_clip(
        data,  # / np.nanmedian(datamodel.data) - 1
        sigma=sigma,
        maxiters=None,
        masked=True,
    )  # normalized to avoid errors clipping for very large values??

    mask = sigma_clip_array.mask == 1  # & (mask_lines == 0)
    dq_new = dqflagging(dq, mask, 'DO_NOT_USE')
    return dq_new, mask


def create_pixelmask(filenames, sigma=3, threshold=3):
    '''Create a mask of pixels in detecter coordinates that have outlier values.

    Open all the input files and compare the value for pixel by pixel.
    '''
    list_data, list_dq = [], []
    for fn in filenames:
        with fits.open(fn) as hdul:
            list_data.append(hdul[1].data)
            list_dq.append(hdul[3].data)

    mask_count = np.zeros_like(list_data[0])
    for data, dq in zip(list_data, list_dq):
        already_flagged = is_dqflagged(dq, 'DO_NOT_USE')
        data[already_flagged] = np.nan
        mask_sigmaclip = sigma_clip(data, sigma=sigma, maxiters=None, masked=True)
        mask_count += mask_sigmaclip.mask

    mask = mask_count >= 3
    return mask


class MaskOutliers:
    '''Class to mask outliers based on mask.'''

    def __init__(self, fnames: list[str]) -> None:
        self.fname_nrs1 = fnames[0]
        self.fname_nrs2 = fnames[1]
        self.mask_nrs1 = fits.getdata(fnames[0]).astype(bool)
        self.mask_nrs2 = fits.getdata(fnames[1]).astype(bool)

    def flag_pixels(self, dq: np.ndarray, filename: str) -> np.ndarray:
        '''Mask pixels based on mask file.'''
        if 'nrs1' in filename:
            mask = np.copy(self.mask_nrs1)
        elif 'nrs2' in filename:
            mask = np.copy(self.mask_nrs2)
        else:
            raise ValueError('Could not find the detector from the filename.')

        dq_new = dqflagging(dq, mask, 'DO_NOT_USE')
        return dq_new


@dataclass
class ConfigSigmaClip:
    skip: bool = False
    sigma: float = 10.0


@dataclass
class ConfigMaskOutliers:
    skip: bool = False
    fnames_mask: list[str] = field(default_factory=list)


def main():
    '''Example'''
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
        datamodel = datamodels.open(fn)
        datamodel.dq, bad_pixel_mask = sigmaclip(datamodel.data, datamodel.dq)

        fsave = fn.replace('_cal', '_pixelmask')
        fits.writeto(fsave, bad_pixel_mask, overwrite=True)

        fsave = fn.replace('_cal', '_clipped' + '_cal')
        datamodel.save(fsave)
        return fsave

    # ==============================
    fnames_nrs1 = [
        'calib/calib_bk/jw01840017001_02101_00001_nrs1_1_rate.fits',
        'calib/calib_bk/jw01840017001_02101_00002_nrs1_1_rate.fits',
        'calib/calib_bk/jw01840017001_02101_00003_nrs1_1_rate.fits',
        'calib/calib_bk/jw01840017001_02101_00004_nrs1_1_rate.fits',
    ]
    mask_nrs1 = create_pixelmask(fnames_nrs1)
    fnames_nrs1_cal = [
        'calib/calib6th/jw01840017001_02101_00001_nrs1_1_cal.fits',
        'calib/calib6th/jw01840017001_02101_00002_nrs1_1_cal.fits',
        'calib/calib6th/jw01840017001_02101_00003_nrs1_1_cal.fits',
        'calib/calib6th/jw01840017001_02101_00004_nrs1_1_cal.fits',
    ]
    mask_nrs1_cal = create_pixelmask(fnames_nrs1_cal)
    mask_nrs1 = mask_nrs1 | mask_nrs1_cal
    fsave = 'calib/calib_data/pixelmask_nrs1.fits'
    fits.writeto(fsave, mask_nrs1.astype(int), overwrite=True)

    fnames_nrs2 = [
        'calib/calib_bk/jw01840017001_02101_00001_nrs2_1_rate.fits',
        'calib/calib_bk/jw01840017001_02101_00002_nrs2_1_rate.fits',
        'calib/calib_bk/jw01840017001_02101_00003_nrs2_1_rate.fits',
        'calib/calib_bk/jw01840017001_02101_00004_nrs2_1_rate.fits',
    ]
    mask_nrs2 = create_pixelmask(fnames_nrs2)
    fnames_nrs2_cal = [
        'calib/calib6th/jw01840017001_02101_00001_nrs2_1_cal.fits',
        'calib/calib6th/jw01840017001_02101_00002_nrs2_1_cal.fits',
        'calib/calib6th/jw01840017001_02101_00003_nrs2_1_cal.fits',
        'calib/calib6th/jw01840017001_02101_00004_nrs2_1_cal.fits',
    ]
    mask_nrs2_cal = create_pixelmask(fnames_nrs2_cal)
    mask_nrs2 = mask_nrs2 | mask_nrs2_cal
    fsave = 'calib/calib_data/pixelmask_nrs2.fits'
    fits.writeto(fsave, mask_nrs2.astype(int), overwrite=True)
    return


if __name__ == '__main__':
    main()

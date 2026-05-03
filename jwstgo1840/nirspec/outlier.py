'''Remove outlier'''

from __future__ import annotations
from typing import Sequence
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from importlib.abc import Traversable
from logging import getLogger
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.utils.exceptions import AstropyUserWarning
from .jwst import IFUImageModel
from .jwst.dqflag import dqflagging, is_dqflagged

logger = getLogger(__name__)

__all__ = [
    'ConfigSigmaClip',
    'ConfigWiseSigmaClip',
    'ConfigMaskOutliers',
    'ConfigMaskFailedFluxCalib',
    'sigmaclip',
    'wise_sigmaclip',
    'MaskOutliers',
    'mask_failedfluxcalibpix',
    'create_pixelmask',
]


##
@dataclass
class ConfigSigmaClip:
    sigma: float = 10.0
    sigma_second: float | None = None
    sigma_hotpix: float = 3.0
    skip: bool = False
    save_results: bool = False

@dataclass
class ConfigWiseSigmaClip:
    sigma: float = 10.0
    sigma_second: float | None = None
    sigma_hotpix: float = 3.0
    skip: bool = True
    save_results: bool = False

@dataclass
class ConfigMaskOutliers:
    skip: bool = False
    fnames_mask: list[Traversable] = field(default_factory=list)
    dqflags: tuple[int, ...] | tuple[str, ...] = (
        "DEAD",
        "HOT",
        "WARM",
        "LOW_QE",
        "RC",
        "TELEGRAPH",
        "NO_SAT_CHECK",
        "UNRELIABLE_BIAS",
        "UNRELIABLE_DARK",
        "UNRELIABLE_SLOPE",
        "UNRELIABLE_FLAT",
    )


@dataclass
class ConfigMaskFailedFluxCalib:
    skip: bool = False


def sigmaclip(data, dq, sigma=10):
    '''Sigma clipping to flag outliers after Detector1 and Spec2.

    Save two files with suffix of "_pixelmask" and "_rate_clipped".
    '''
    # OUTLIER = Signal from object masked with masking_objects3D() in this code.
    dq_outlier = is_dqflagged(dq, 'OUTLIER')
    dq_notuse = is_dqflagged(dq, 'DO_NOT_USE')
    mask = dq_notuse | dq_outlier
    madata = np.ma.masked_array(data, mask)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=AstropyUserWarning)
        sigma_clip_array = sigma_clip(
            madata,  # / np.nanmedian(datamodel.data) - 1
            sigma=sigma,
            maxiters=None,
            masked=True,
            axis=0,  # Clipping along spatial direction (y-axis)
        )  # normalized to avoid errors clipping for very large values??

    n = np.count_nonzero(sigma_clip_array.mask & ~madata.mask)
    logger.info(f'# of pixels removed by the sigma clip (sigma={sigma}): {n}')

    # Update mask for sigma-clipped pixels; Don't update for originally OUTLIER (=object) pixels
    mask_new = (madata.mask == 0) & (sigma_clip_array.mask == 1)
    dq_new = dqflagging(dq, mask_new, 'DO_NOT_USE')
    return dq_new, mask_new


def wise_sigmaclip(data, dq, config: ConfigWiseSigmaClip):
    '''Wise sigma clipping to find and flag outliers in AfterSpec2.

    Several masking methodologies:
    1. 1st sigma clipping with taking into account the possibility of object
       - Flag if a pixel is negative.
       - Do not flag if a pixel is sorrounded by 2nd sigma bright pixels
       - Do not flag if a pixel is smaller than twice the sorrounded pix average.

    In the future:
    1. ...
       - Do not flag if pixel is flagged among all the dithers on WCS.
    2. hot pix sigma clipping
       - Flag if a pixel is > 3rd sigma (sigma_hotpix) among all the dithers
         on the detector coordinate.
    '''
    mask = is_dqflagged(dq, 'DO_NOT_USE')
    madata: np.ma.MaskedArray = np.ma.masked_array(data, mask)

    new_clipped = np.zeros_like(data).astype(bool)

    def clip(sigma: float) -> np.ndarray:
        return sigma_clip(
            madata,  # / np.nanmedian(datamodel.data) - 1
            sigma=sigma,
            maxiters=None,
            masked=True,
            axis=0,  # Clipping along spatial direction (y-axis)
        ).mask  # normalized to avoid errors clipping for very large values??

    # Select candidates
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=AstropyUserWarning)
        candidates = clip(config.sigma)
        sigma_bright = (
            config.sigma / 2.0 if config.sigma_second is None else config.sigma_second
        )
        pixels_bright = clip(sigma_bright)

    candidates[mask] = False
    pixels_bright[mask] = False
    n = np.count_nonzero(candidates)
    logger.info(f'# of outlier candidates by the {config.sigma} sigma clip: {n}')

    # Start clipping pixels
    is_negative = data < np.nanmedian(madata, axis=0)
    new_clipped[candidates & is_negative] = True
    candidates[is_negative] = False
    n = np.count_nonzero(new_clipped)
    logger.info(f'- # of negative outliers, which are clipped: {n}')

    n_list = [0, 0]
    for y, x in zip(*np.where(candidates)):
        s = (slice(y - 1, y + 1), slice(x - 1, x + 1))
        cutout = madata[s[0], s[1]]
        cutout_is_bright = pixels_bright[s[0], s[1]]

        brightness = (np.nansum(cutout) - cutout[1, 1]) / (cutout.count() - 1)

        if np.any(cutout < 0):
            new_clipped[y, x] = True

        elif np.all(cutout_is_bright):
            n_list[0] += 1

        # elif 2 * brightness > cutout[1, 1]:
        #     n_list[1] += 1

        else:  # TODO: This will be changed if 2nd hot pix clipping method is implemented.
            new_clipped[y, x] = True

        candidates[y, x] = False

    logger.info(f'- # of pixels sorrounded by >{sigma_bright:.1f} pixels: {n_list[0]}')
    logger.info(f'- # of pixels sorrounded by bright pixels: {n_list[1]}')

    assert not np.any(candidates)

    # Update mask for sigma-clipped pixels
    logger.info(
        f'Consequently, # of sigma clipped pixels: {np.count_nonzero(new_clipped)}'
    )
    assert not np.any(madata.mask & new_clipped)
    # mask_new = ~madata.mask & new_clipped
    dq_new = dqflagging(dq, new_clipped, 'DO_NOT_USE')
    return dq_new, new_clipped


class MaskOutliers:
    '''Class to mask outliers based on mask.

    Related ADR
        - ADR0001
    '''

    def __init__(self, config: ConfigMaskOutliers) -> None:
        fnames = config.fnames_mask
        self.fname_nrs1 = fnames[0]
        self.fname_nrs2 = fnames[1]
        self._mask_nrs1: None | np.ndarray = None
        self._mask_nrs2: None | np.ndarray = None

        self.dqflags = config.dqflags

    def __call__(self, datamodel: IFUImageModel, filename: str) -> np.ndarray:
        '''Mask outlier pixels.

        Args:
            dqmap (np.ndarray): Data quality flag.
            filename (str): File name being reduced.

        Returns:
            np.ndarray: New data quality flag where outlier pixels are flagged as
                DO_NOT_USE.

        Examples:
            >>> maskoutlier = MaskOutliers(self.maskoutlier.fnames_mask)
            >>> datamodel.dq = maskoutlier.flag_pixels(datamodel.dq, filename.name)
        '''
        dqmap = datamodel.dq
        dqmap = self.mask_with_original(dqmap, filename)
        dqmap = self.mask_with_dqflags(dqmap)
        return dqmap

    def mask_with_original(self, dq: np.ndarray, filename: str) -> np.ndarray:
        '''Mask pixels based on mask file.'''
        if 'nrs1' in filename:
            mask = self.mask_nrs1
        elif 'nrs2' in filename:
            mask = self.mask_nrs2
        else:
            raise ValueError('Could not find the detector from the filename.')

        dq_new = dqflagging(dq, mask, 'DO_NOT_USE')
        return dq_new

    def mask_with_dqflags(self, dqmap: np.ndarray) -> np.ndarray:
        '''Mask outlier pixels having meaningful dqflags.'''
        for dq in self.dqflags:
            mask = is_dqflagged(dqmap, dq)
            dqmap = dqflagging(dqmap, mask, 'DO_NOT_USE')
        return dqmap

    @property
    def mask_nrs1(self) -> np.ndarray:
        if self._mask_nrs1 is None:
            self._mask_nrs1 = fits.getdata(self.fname_nrs1).astype(bool)
        return self._mask_nrs1

    @property
    def mask_nrs2(self) -> np.ndarray:
        if self._mask_nrs2 is None:
            self._mask_nrs2 = fits.getdata(self.fname_nrs2).astype(bool)
        return self._mask_nrs2


def mask_failedfluxcalibpix(
    datamodel: IFUImageModel, filename: str | Path
) -> IFUImageModel:
    '''Mask possible pixels if the flux calibration has been failed there.

    Related ADR
        - ADR0001
    '''
    p = Path(filename)
    data_rate = fits.getdata(p.parent / p.name.replace('_cal', '_rate'), 1)
    data_cal = datamodel.data

    mask = data_cal == data_rate
    if np.count_nonzero(mask):
        datamodel.dq = dqflagging(datamodel.dq, mask, 'DO_NOT_USE')

    return datamodel


def create_pixelmask(
    filenames: Sequence[str | Path], sigma: float = 3.0, threshold: float = 0.6
):
    '''Create a mask of pixels in detecter coordinates that have outlier values.

    Open all the input files and compare the value for pixel by pixel.

    Args:
        filenames (list[str]): rate filename list
        sigma (float, optional): Threshold value for sigma clipping. Defaults to 3.0.
        threshold (float, optional): Threshold number to decide how many data have
            outlier values at the physical pixel for masking out the pixel.
            Defaults to 0.6.

    Related ADR:
        - ADR0001
    '''
    mask_count = _count_outlierpix_dithers(filenames, sigma=sigma)

    _threshold = int(threshold * len(filenames))
    mask = mask_count >= _threshold
    return mask


def _count_outlierpix_dithers(
    filenames: Sequence[str | Path], sigma: float = 3
) -> np.ndarray:
    '''Count the number of dithers which has outlier values in pixels.'''
    data0 = fits.getdata(filenames[0], 1)
    mask_count = np.zeros_like(data0).astype(int)

    for fn in filenames:
        with fits.open(fn) as hdul:
            data = hdul[1].data.copy()
            # dq = hdul[3].data.copy()

        # already_flagged = is_dqflagged(dq, 'DO_NOT_USE')
        # data[already_flagged] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=AstropyUserWarning)
            mask_sigmaclip = sigma_clip(
                data, sigma=sigma, maxiters=None, masked=True, axis=0
            )
        mask_count += mask_sigmaclip.mask

    return mask_count


def clip_raws(input_model: IFUImageModel, raws: list[int]) -> IFUImageModel:
    '''Remove raws with systematic noises.'''
    data = input_model.data

    mask = np.zeros_like(data)
    for r in raws:
        mask[r, :] = True

    already_flagged = already_flagged = is_dqflagged(input_model.dq, 'DO_NOT_USE')
    mask[already_flagged] = False
    input_model.dq = dqflagging(input_model.dq, mask, 'DO_NOT_USE')
    return input_model


def main():
    '''Example'''
    from jwst import datamodels

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


if __name__ == '__main__':
    main()

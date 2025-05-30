'''Mask pixels in NIRSpec IFU data
'''
from __future__ import annotations
from typing import Optional
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from astropy.units import Quantity
from astropy.coordinates import SkyCoord
from photutils.aperture import SkyCircularAperture
from jwst import datamodels
from jwst.datamodels import IFUImageModel
from gwcs import wcstools
from .assign_wcs import get_nrs_wcs_slit, change_nrs_wcs_slit, wcs_calfits
from .dqflag import is_dqflagged, dqflag, dqflagging
import logging

__all__ = ['NIRSpecIFUMask', 'ConfigMaskingSlitedge']

logger = logging.getLogger('debuglog')


##
def masking_slitedges(datamodel: IFUImageModel) -> tuple[IFUImageModel, np.ndarray]:
    '''Mask slit edges of NIRSPec IFU.

    The slit edges of NIRSpec IFU show large noises, which should be excluded
    before constructing the 3D cube.
    '''
    logger.info(f'Masking slitedges of {datamodel.meta.filename}...')
    nslits = 30  # for NIRSpec IFU
    detector = datamodel.meta.instrument.detector

    mask_edge = np.zeros_like(datamodel.data).astype(bool)
    for i in range(nslits):
        if i == 0:
            slice_wcs = get_nrs_wcs_slit(datamodel, i)
        else:
            slice_wcs = change_nrs_wcs_slit(datamodel, slice_wcs, i)

        y, x = where_are_edges(slice_wcs, detector, i)
        mask_edge[y, x] = True

    already_flagged = is_dqflagged(datamodel.dq, 'DO_NOT_USE')
    mask_edge[already_flagged] = False
    datamodel.dq[mask_edge] += dqflag['DO_NOT_USE']

    return datamodel, mask_edge


def where_are_edges(
    slice_wcs: WCS, detector: Optional[str] = None, i_slice: Optional[int] = None
) -> tuple[np.ndarray, np.ndarray]:
    '''Detect where are edges of slits.

    Importantly, edge widths depend on the detector (nrs1 or nrs2) and
    the slit numbers (0-29).
    '''
    x, y = wcstools.grid_from_bounding_box(slice_wcs.bounding_box)
    ra, _, _ = slice_wcs(x, y)
    # ra, dec, lambda = slice_wcs(x, y)

    y2 = np.copy(y)
    y2[np.isnan(ra)] = y.min() - 1
    pos_edge_upper = np.argmax(y2, axis=0)
    y2[np.isnan(ra)] = y.max() + 1
    pos_edge_lower = np.argmin(y2, axis=0)

    wup, wlow = get_edgewidths(detector, i_slice)

    x, y = x.astype(int), y.astype(int)
    pos_x = np.arange(x.shape[1]).astype(int)
    concate = np.concatenate
    idx_y1 = concate([y[pos_edge_lower + i, pos_x] for i in range(wlow)])
    idx_x1 = concate([x[pos_edge_lower + i, pos_x] for i in range(wlow)])
    idx_y2 = concate([y[pos_edge_upper - i, pos_x] for i in range(wup)])
    idx_x2 = concate([x[pos_edge_upper - i, pos_x] for i in range(wup)])

    idx_y = concate((idx_y1, idx_y2))
    idx_x = concate((idx_x1, idx_x2))
    return idx_y, idx_x


def get_edgewidths(
    detector: Optional[str] = None, i_slice: Optional[int] = None
) -> tuple[int, int]:
    '''Get edge widths depends on detector name and slice number.

        Args:
            detector (Optional[str], optional): Defaults to None.
            i_slice (Optional[int], optional): Defaults to None
    .

        Returns:
            tuple[int, int]: width of upper and lower edges, respectively.

        Examples:
            >>> up, low = get_edgewiths('NRS1', 0)
    '''
    if (detector is None) or (i_slice is None):
        return 1, 1

    if detector == 'NRS1':
        if i_slice == 27:  # 1 from top
            return 1, 3
        if i_slice == 19:  # 5 from top
            return 1, 3
        if i_slice == 13:  # 8 from top
            return 5, 1

    if detector == 'NRS2':
        if i_slice == 27:  # 1 from top
            return 1, 3
        if i_slice == 19:  # 5 from top
            return 1, 4
        if i_slice == 13:  # 8 from top
            return 4, 1
        # if i_slice == 5:  # 12 from top
        #     return 15, 15
        if i_slice == 16:  # 23 from top
            return 15, 1
        if i_slice == 26:  # 28 from top
            return 2, 1

    return 1, 1


def masking_msa_failed_open(datamodel: IFUImageModel) -> IFUImageModel:
    '''Mask pixels with dq of MSA_FAILED_OPEN.'''
    msa_failed_open = is_dqflagged(datamodel.dq, 'MSA_FAILED_OPEN')
    already_flagged = is_dqflagged(datamodel.dq, 'DO_NOT_USE')
    msa_failed_open[already_flagged] = False
    datamodel.dq[msa_failed_open] += dqflag['DO_NOT_USE']
    return datamodel


def masking_objects3D(
    datamodel: IFUImageModel,
    fname3d: str,
    positions: SkyCoord,
    radii: Quantity,
    wavelengths: Quantity,
) -> IFUImageModel:
    '''Masking pixels manually defined as objects.

    Args:
        datamodel (datamodels): Jwst datamodel of _cal.fits
        fname3d (str): Filename of 3d data cube.
        positions (SkyCoord): Central positions of circular masks.
            SkyCoord lists of (ra, dec) tuples.
        radii (Quantity): Radii of circular masks.
            Quantity lits of radii with units of angular distance.
        wavelengths (Quantity): start and end wavelengths of circular masks.
            Quantity lits of [wave0, wave1], where 0 and 1 are start and end.

    Returns:
        IFUImageModel: datamodel with new DQ extensions.
            The masked regions are flagged as OUTLIER.

    Examples:
        >>> new_datamodel = masking_objects3D(
        >>>     datamodel,
        >>>     'cube.fits',
        >>>     SkyCoord([(('00h14m24.9217s', '-30d22m56.160s'), ('00h14m24.9291s', '-30d22m54.956s'),)]),
        >>>     [0.4, 0.4] * u.arcsec,
        >>>     [[4.3, 4.5], [4.4445, 4.45]] * u.um
        >>> )
    '''
    logger.info("Masking the following objects...")
    logger.info("Reference 3D cube file:{fname3d}")
    ifumask = NIRSpecIFUMask(fname3d)
    logger.info(f"Sky positions:{positions}")
    logger.info(f"Circular mask radii:{radii}")
    logger.info(f"Wavelengths:{wavelengths}")

    # Masking to 3D cube
    ifumask.add_circularmasks(positions, radii, wavelengths)

    # Apply the mask to 2D slit image; DQ flag = 'OUTLIER'
    mask2d = ifumask.mask_cal2d(datamodel)
    datamodel.dq = dqflagging(datamodel.dq, mask2d, 'OUTLIER')

    return datamodel


class NIRSpecIFUMask:
    '''To create masks for a data cube and _cal.fits images.

    NOTE:
    This class assumes that the coordinates of the data cube and _cal.fits images are the same.
    Do not use the data cube after a wcs correction.
    '''

    def __init__(self, filename_cube: str) -> None:
        self.fname = filename_cube
        self.data = fits.getdata(self.fname, 'SCI')
        self.header = fits.getheader(self.fname, 'SCI')
        self.wcs = WCS(self.header)
        self.mask = np.zeros_like(self.data, dtype=bool)
        self.positions: SkyCoord
        self.radii: Quantity
        self.waves: Quantity

    def add_circularmasks(
        self, positions: SkyCoord, radii: Quantity, wavelengths: list[Quantity]
    ) -> None:
        '''Add circular masks.'''
        try:
            self.positions = np.concatenate((self.positions, SkyCoord))
            self.radii = np.concatenate((self.radii, radii))
            self.waves = np.concatenate((self.waves, wavelengths))
        except AttributeError:
            self.positions = positions
            self.radii = radii
            self.wave = wavelengths

        wave_cube = self.wcs.spectral.pixel_to_world(np.arange(self.data.shape[0]))
        for p, r, w in zip(positions, radii, wavelengths):
            aperture = SkyCircularAperture(p, r=r).to_pixel(self.wcs.celestial)
            apermask = aperture.to_mask(method='center')
            mask_aperture = apermask.to_image(self.data.shape[1:]).astype(bool)
            mask_wave = (wave_cube > w[0]) & (wave_cube < w[1])
            self.mask |= (
                mask_aperture[np.newaxis, :, :] & mask_wave[:, np.newaxis, np.newaxis]
            )

    def mask_cal2d(self, fname_or_datamodel: str | IFUImageModel):
        '''Create mask for _cal.fits.'''
        if isinstance(fname_or_datamodel, str):
            datamodel = datamodels.open(fname_or_datamodel)
        elif isinstance(fname_or_datamodel, IFUImageModel):
            datamodel = fname_or_datamodel
        else:
            raise TypeError('The input argument must be str or datamodels.')

        mask = np.zeros_like(datamodel.data).astype(bool)
        ra, dec, wavelengths = wcs_calfits(datamodel)
        ra = ra * u.deg
        dec = dec * u.deg
        wavelengths = wavelengths * u.um
        for p, r, w in zip(self.positions, self.radii, self.wave):
            distance = (ra - p.ra.to(u.deg)) ** 2 + (dec - p.dec.to(u.deg)) ** 2
            within_circle = distance < r.to(u.deg) ** 2
            within_wave = (wavelengths > w[0]) & (wavelengths < w[1])
            mask |= within_circle & within_wave
        return mask


@dataclass
class ConfigMaskingSlitedge:
    skip: bool = False


@dataclass
class ConfigMaskingFailedSlitOpen:
    skip: bool = False


@dataclass
class ConfigMaskingObj:
    fname3d: str = ''
    positions: SkyCoord | None = None
    radii: Quantity | None = None
    waves: Quantity | None = None
    skip: bool = True

    def check_welldefined(self) -> bool:
        '''Check whether properties are correctly specified.'''
        errtxt = 'In masking objects, objmask.{} is not defined.'
        if not self.fname3d:
            raise AttributeError(errtxt.format('fname3d'))
        if self.positions is None:
            raise AttributeError(errtxt.format('positions'))
        if self.radii is None:
            raise AttributeError(errtxt.format('radii'))
        if self.waves is None:
            raise AttributeError(errtxt.format('waves'))

        if not Path(self.fname3d).exists():
            raise FileNotFoundError(
                f'In masking objects, datacube "{self.fname3d}" does not exits. Is product_name correct?'
            )
        if self.radii.shape != self.positions.shape:
            raise ValueError(
                'In masking objects, shapes of radii and positions are different: '
                f'radii {self.radii.shape} but positions {self.positions.shape}.'
            )
        if not ((len(self.waves.shape) == 2) and self.waves.shape[1] == 2):
            raise ValueError('In masking objects, shapes of waves must be (X, 2).')
        if self.waves.shape[0] != self.positions.shape[0]:
            raise ValueError(
                'In masking objects, shapes of waves and positions are different: '
                f'waves[0] {(self.radii.shape[0],)} but positions {self.positions.shape}.'
            )

        return True


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
        datamodel, mask_edge = masking_slitedges(datamodel)
        fsave = fn.replace('_cal', '_edgemask')
        fits.writeto(fsave, mask_edge.astype(int), overwrite=True)
        fsave = fn.replace('_cal', '_edgemask_cal')
        datamodel.save(fsave)
    return


if __name__ == '__main__':
    main()

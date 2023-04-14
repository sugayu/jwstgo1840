'''Mask pixels in NIRSpec IFU data
'''
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from astropy.units import Quantity
from astropy.coordinates import SkyCoord
from photutils.aperture import SkyCircularAperture
from jwst import datamodels
from gwcs import wcstools
from .assign_wcs import get_nrs_wcs_slit, change_nrs_wcs_slit, wcs_calfits
import logging

logger = logging.getLogger('debuglog')
dqflag = datamodels.dqflags.pixel


##
def masking_slitedges(datamodel):
    '''Mask slit edges of NIRSPec IFU.

    The slit edges of NIRSpec IFU show large noises, which should be excluded
    before constructing the 3D cube.
    '''
    logger.info(f'Masking slitedges of {datamodel.meta.filename}...')
    nslits = 30  # for NIRSpec IFU

    mask_edge = np.zeros_like(datamodel.data).astype(bool)
    for i in range(nslits):
        if i == 0:
            slice_wcs = get_nrs_wcs_slit(datamodel, i)
        else:
            slice_wcs = change_nrs_wcs_slit(datamodel, slice_wcs, i)
        x, y = wcstools.grid_from_bounding_box(slice_wcs.bounding_box)
        ra, _, _ = slice_wcs(x, y)
        # ra, dec, lambda = slice_wcs(x, y)

        y2 = np.copy(y)
        y2[np.isnan(ra)] = y.min() - 1
        pos_edge_upper = np.argmax(y2, axis=0)
        y2[np.isnan(ra)] = y.max() + 1
        pos_edge_lower = np.argmin(y2, axis=0)

        x, y = x.astype(int), y.astype(int)
        pos_x = np.arange(x.shape[1]).astype(int)
        mask_edge[y[pos_edge_lower, pos_x], x[pos_edge_lower, pos_x]] = True
        mask_edge[y[pos_edge_upper, pos_x], x[pos_edge_upper, pos_x]] = True

    already_flagged = np.bitwise_and(datamodel.dq, dqflag['DO_NOT_USE']).astype(bool)
    mask_edge[already_flagged] = False
    datamodel.dq[mask_edge] += dqflag['DO_NOT_USE']

    return datamodel, mask_edge


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

    def mask_cal2d(self, fname_or_datamodel: str | datamodels):
        '''Create mask for _cal.fits.'''
        if isinstance(fname_or_datamodel, str):
            datamodel = datamodels.open(fname_or_datamodel)
        elif isinstance(fname_or_datamodel, datamodels):
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

'''Mask pixels in NIRSpec IFU data
'''
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from astropy.io import fits
from jwst import datamodels
from jwst.assign_wcs import nirspec
from gwcs import wcstools
from .assign_wcs import change_nrs_wcs_slit
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
            slice_wcs = nirspec.nrs_wcs_set_input(datamodel, i)
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

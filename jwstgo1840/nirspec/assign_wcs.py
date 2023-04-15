'''Wrapper of jwst/assign_wcs/nirspec

Motivation is to speed up deepcopy, which is very very slow (~2sec),
used in _nrs_wcs_set_input
'''
import copy
import numpy as np
from astropy.wcs import WCS
from astropy.modeling.models import Identity
from jwst.assign_wcs.nirspec import (
    spectral_order_wrange_from_model,
    compute_bounding_box,
)
from jwst.assign_wcs.nirspec import nrs_wcs_set_input as jwst_nrs_wcs_set_input
from jwst.lib.exposure_types import is_nrs_ifu_lamp

__all__ = ['get_nrs_wcs_slit', 'change_nrs_wcs_slit', 'wcs_calfits']


##
def get_nrs_wcs_slit(input_model, slit_name) -> WCS:
    """
    Returns a WCS object for a specific slit, slice or shutter.

    Parameters
    ----------
    input_model : jwst.datamodels.DataModel
        The data model. Must have been through the assign_wcs step.
    slit_name : int or str
        Slit.name of an open slit.
    """
    _, wrange = spectral_order_wrange_from_model(input_model)
    return jwst_nrs_wcs_set_input(input_model, slit_name, wrange)


def change_nrs_wcs_slit(input_model, slit_wcs, slit_name) -> WCS:
    """
    Change a WCSs for different slits.

    Parameters
    ----------
    input_model : jwst.datamodels.DataModel
        The data model. Must have been through the assign_wcs step.
    """
    _, wrange = spectral_order_wrange_from_model(input_model)
    return nrs_wcs_set_input(input_model, slit_name, wrange, slit_wcs=slit_wcs)


def wcs_calfits(input_model):
    '''Return (ra,dec,wave) of cal.fits.

    This function takes ~20 seconds.
    '''
    shape = input_model.data.shape
    radecw = np.zeros((3, *shape))
    grid_y, grid_x = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

    nslits = 30
    for i in range(nslits):
        if i == 0:
            wcs_slit = get_nrs_wcs_slit(input_model, i)
        else:
            wcs_slit = change_nrs_wcs_slit(input_model, wcs_slit, i)
        ra, dec, wave = wcs_slit(grid_x, grid_y)
        radecw[0, ~np.isnan(ra.T)] = ra.T[~np.isnan(ra.T)]
        radecw[1, ~np.isnan(dec.T)] = dec.T[~np.isnan(dec.T)]
        radecw[2, ~np.isnan(wave.T)] = wave.T[~np.isnan(wave.T)]
    return radecw


def nrs_wcs_set_input(
    input_model,
    slit_name,
    wavelength_range=None,
    slit_y_low=None,
    slit_y_high=None,
    slit_wcs=None,
):
    """
    Returns a WCS object for a specific slit, slice or shutter.

    Parameters
    ----------
    input_model : `~jwst.datamodels.DataModel`
        A WCS object for the all open slitlets in an observation.
    slit_name : int or str
        Slit.name of an open slit.
    wavelength_range: list
        Wavelength range for the combination of filter and grating.

    Returns
    -------
    wcsobj : `~gwcs.wcs.WCS`
        WCS object for this slit.
    """

    def _get_y_range(input_model):
        # get the open slits from the model
        # Need them to get the slit ymin,ymax
        g2s = input_model.meta.wcs.get_transform('gwa', 'slit_frame')
        open_slits = g2s.slits
        slit = [s for s in open_slits if s.name == slit_name][0]
        return slit.ymin, slit.ymax

    if wavelength_range is None:
        _, wavelength_range = spectral_order_wrange_from_model(input_model)

    slit_wcs = _nrs_wcs_set_input(input_model, slit_name, slit_wcs=slit_wcs)
    transform = slit_wcs.get_transform('detector', 'slit_frame')
    is_nirspec_ifu = (
        is_nrs_ifu_lamp(input_model)
        or input_model.meta.exposure.type.lower() == 'nrs_ifu'
    )
    if is_nirspec_ifu:
        bb = compute_bounding_box(transform, wavelength_range)
    else:
        if slit_y_low is None or slit_y_high is None:
            slit_y_low, slit_y_high = _get_y_range(input_model)
        bb = compute_bounding_box(
            transform, wavelength_range, slit_ymin=slit_y_low, slit_ymax=slit_y_high
        )

    slit_wcs.bounding_box = bb
    return slit_wcs


def _nrs_wcs_set_input(input_model, slit_name, slit_wcs=None):
    """
    Returns a WCS object for a specific slit, slice or shutter.
    Does not compute the bounding box.

    Parameters
    ----------
    input_model : `~jwst.datamodels.DataModel`
        A WCS object for the all open slitlets in an observation.
    slit_name : int or str
        Slit.name of an open slit.

    Returns
    -------
    wcsobj : `~gwcs.wcs.WCS`
        WCS object for this slit.
    """
    wcsobj = input_model.meta.wcs

    if slit_wcs is None:
        slit_wcs = copy.deepcopy(wcsobj)

    slit_wcs.set_transform('sca', 'gwa', wcsobj.pipeline[1].transform[1:])
    g2s = wcsobj.pipeline[2].transform
    slit_wcs.set_transform('gwa', 'slit_frame', g2s.get_model(slit_name))

    exp_type = input_model.meta.exposure.type
    is_nirspec_ifu = is_nrs_ifu_lamp(input_model) or (exp_type.lower() == 'nrs_ifu')
    if is_nirspec_ifu:
        slit_wcs.set_transform(
            'slit_frame',
            'slicer',
            wcsobj.pipeline[3].transform.get_model(slit_name) & Identity(1),
        )
    else:
        slit_wcs.set_transform(
            'slit_frame',
            'msa_frame',
            wcsobj.pipeline[3].transform.get_model(slit_name) & Identity(1),
        )
    return slit_wcs

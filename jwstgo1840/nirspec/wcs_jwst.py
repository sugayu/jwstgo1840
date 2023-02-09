'''Mathc wcs of NIRSpecIFU and NIRCam
'''
from __future__ import annotations
import numpy as np
from math import gcd  # greatest common divisor
from scipy.signal import correlate, correlation_lags
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata.utils import Cutout2D


##
def find_pixel_offsets(from_image1, to_image2):
    '''Find offsets in pixel between two images.

    For NIRSpec and NIRCam data.
    '''
    shape1 = from_image1.shape
    shape2 = to_image2.shape
    corr = correlate(to_image2, from_image1)
    iy, ix = np.where(corr == np.max(corr))
    lags_y = correlation_lags(shape2[0], shape1[0])
    lags_x = correlation_lags(shape2[1], shape1[1])
    return lags_y[iy][0], lags_x[ix][0]


def match_pixelsize(image1, image2, pixelscales1, pixelscales2):
    '''Match pixel sizes of two images'''
    _ans = _get_dummyshapes_basedon_pixelscales(pixelscales1, pixelscales2)
    shape1, shape2, gcd_x, gcd_y = _ans

    image1_new = np.repeat(image1, shape1[0] / gcd_y, axis=0)
    image1_new = np.repeat(image1_new, shape1[1] / gcd_x, axis=1)
    image2_new = np.repeat(image2, shape2[0] / gcd_y, axis=0)
    image2_new = np.repeat(image2_new, shape2[1] / gcd_x, axis=1)
    return image1_new, image2_new, (shape1, shape2, gcd_y, gcd_x)


def _get_dummyshapes_basedon_pixelscales(pixelscales1, pixelscales2):
    '''Dummy sahpes based on pixelscales.

    Used for match_pixelsize()
    '''
    size = 1000  # depending on accuracy
    shape1 = (
        round((pixelscales1[0] / pixelscales2[0]).value * size),
        round((pixelscales1[1] / pixelscales2[1]).value * size),
    )
    shape2 = (size, size)
    gcd_x = gcd(shape1[1], shape2[1])
    gcd_y = gcd(shape1[0], shape2[0])
    return shape1, shape2, gcd_x, gcd_y


def read_data_and_wcs(fname, ext=0):
    '''Read data and wcs'''
    data = fits.getdata(fname, ext=ext)
    header = fits.getheader(fname, ext=ext)
    wcs = WCS(header)
    if wcs.naxis == 3:
        wcs = wcs.celestial
    return data, wcs


def find_celestial_offset(from_image1, to_image2, wcs1, wcs2):
    '''Find an celestial offset from image1 to iamge2.'''
    pixelscales1 = wcs1.proj_plane_pixel_scales()
    pixelscales2 = wcs2.proj_plane_pixel_scales()

    im1, im2, gcds = match_pixelsize(from_image1, to_image2, pixelscales1, pixelscales2)
    offsets_copixel = find_pixel_offsets(from_image1=im1, to_image2=im2)
    shape1, shape2, gcd_y, gcd_x = gcds
    offsets_pixel_1to2 = (
        offsets_copixel[0] / (shape1[0] / gcd_y),
        offsets_copixel[1] / (shape1[1] / gcd_x),
    )  # (y, x)

    pixelvalue_at_lowerleftcorner = (-0.5, -0.5)
    # see https://docs.astropy.org/en/stable/api/astropy.wcs.wcsapi.BaseLowLevelWCS.html#astropy.wcs.wcsapi.BaseLowLevelWCS.pixel_to_world_values
    origin1 = wcs1.pixel_to_world(*pixelvalue_at_lowerleftcorner)
    origin2 = wcs2.pixel_to_world(*pixelvalue_at_lowerleftcorner)
    offsets_image_origins = origin1.spherical_offsets_to(origin2)  # (ra, dec)

    offsets_image_origins_ra, offsets_image_origins_dec = offsets_image_origins
    offsets_pixel_y, offsets_pixel_x = offsets_pixel_1to2
    pixelscales_x2ra, pixelscales_y2dec = (-pixelscales1[0], pixelscales1[1])
    # -pixelscales1[0] represents opposite direction of x and ra
    offsets = (
        offsets_image_origins_ra + offsets_pixel_x * pixelscales_x2ra,
        offsets_image_origins_dec + offsets_pixel_y * pixelscales_y2dec,
    )
    return offsets


def fix_offsets_in_header(header, offsets):
    '''Edit CRVAL1 and CRVAL2 in header using offsets=(ra, dec) in degrees.'''
    header['CRVAL1'] = header['CRVAL1'] + offsets[0].value
    header['CRVAL2'] = header['CRVAL2'] + offsets[1].value
    return header


def moment0(data, range_widx):
    '''Create moment0 map.'''
    return data[range_widx[0] : range_widx[1], :, :].sum(axis=0)


def main():
    '''Example for A2477-OD'''
    # read logU
    fname_nircam = 'reference/20221228_Fudamoto/logu_v0.fits'
    image_nircam, wcs_nircam = read_data_and_wcs(fname_nircam)
    image_nircam = 10.0**image_nircam  # because it's logU
    image_nircam[np.isnan(image_nircam)] = 0.0

    # read [OIII] moment0
    fname_nirspec = 'calib/calib5th/product_name_g395h-f290lp_s3d.fits'
    cube_nirspec, wcs_nirspec_celestial = read_data_and_wcs(fname_nirspec, 1)
    mom0_nirspec = moment0(cube_nirspec, (2379, 2393))
    image2d_nirspec = Cutout2D(
        mom0_nirspec, (50, 61), (48, 56), wcs=wcs_nirspec_celestial
    )  # [37:85, 19:75]; position=(x,y) and size=(ny, nx)
    image_nirspec = image2d_nirspec.data
    wcs_nirspec = image2d_nirspec.wcs

    # main function
    offsets = find_celestial_offset(
        from_image1=image_nirspec,
        to_image2=image_nircam,
        wcs1=wcs_nirspec,
        wcs2=wcs_nircam,
    )

    # save data
    with fits.open(fname_nirspec) as hdul:
        hdul[1].header = fix_offsets_in_header(hdul[1].header, offsets)
        hdul.writeto(fname_nirspec.replace('.fits', '_fixoffsets.fits'), overwrite=True)

    header = fix_offsets_in_header(wcs_nirspec_celestial.to_header(), offsets)
    fname_save = 'calib/calib5th/moment0_OIII5007.fits'
    fits.writeto(fname_save, mom0_nirspec, header=header, overwrite=True)
    return


if __name__ == '__main__':
    main()

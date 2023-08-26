'''Example of running NIRSpec IFU piplelines.
'''

import os
from pathlib import Path
import glob
import logging
from multiprocess import Pool
from astropy.coordinates import SkyCoord
import astropy.units as u

# These are needed if CRDS_PATH is not set as your environment variables
os.environ["CRDS_PATH"] = 'data/crds_cache'
os.environ["CRDS_SERVER_URL"] = 'https://jwst-crds.stsci.edu'
os.environ["CRDS_CONTEXT"] = 'jwst_1077.pmap'

from jwst.pipeline import calwebb_detector1, Spec2Pipeline, Spec3Pipeline
from jwst import datamodels
from jwst.associations import asn_from_list  # Tools for creating association files
from jwst.associations.lib.rules_level3_base import DMS_Level3_Base

from jwstgo1840.nirspec import (
    AfterDetector1Pipeline,
    AfterSpec2Pipeline,
    AfterSpec3Pipeline,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# intput of output directory
output_dir = 'calib/example/'
Path(output_dir).mkdir(exist_ok=True)


def main():
    # first input files (raw data)
    dname_data = 'data/A2744_OD/'
    fname = [
        'jw01840017001_02101_00001_nrs1/jw01840017001_02101_00001_nrs1_uncal.fits',
        'jw01840017001_02101_00001_nrs2/jw01840017001_02101_00001_nrs2_uncal.fits',
        'jw01840017001_02101_00002_nrs1/jw01840017001_02101_00002_nrs1_uncal.fits',
        'jw01840017001_02101_00002_nrs2/jw01840017001_02101_00002_nrs2_uncal.fits',
        'jw01840017001_02101_00003_nrs1/jw01840017001_02101_00003_nrs1_uncal.fits',
        'jw01840017001_02101_00003_nrs2/jw01840017001_02101_00003_nrs2_uncal.fits',
        'jw01840017001_02101_00004_nrs1/jw01840017001_02101_00004_nrs1_uncal.fits',
        'jw01840017001_02101_00004_nrs2/jw01840017001_02101_00004_nrs2_uncal.fits',
    ]

    # Detector1
    for f in fname:
        run_pipeline_detector1(dname_data + f, maximum_cores='quarter')

    # After Detector1
    fnames = sorted(glob.glob(output_dir + 'jw01840017001_02101_*nrs?_rate.fits'))
    fnames = run_pipeline_after_detector1(fnames)

    # Spec2
    with Pool(4) as p:
        p.map(run_pipeline_spec2, fnames)

    # After Spec2
    fnames = sorted(glob.glob(output_dir + 'jw01840017001_02101_*1_cal.fits'))
    fnames = run_pipeline_after_spec2(fnames)

    # Spec3
    run_pipeline_spec3(fnames)


def run_pipeline_detector1(fname_uncal, maximum_cores='None'):
    '''Run pipeline of Detector1.'''
    # Create an instance of the pipeline class
    detector1 = calwebb_detector1.Detector1Pipeline()

    # Set some parameters that pertain to the entire pipeline
    detector1.output_dir = output_dir
    detector1.save_results = True

    # # Set some parameters that pertain to some of the individual steps
    # detector1.refpix.use_side_ref_pixels = True

    # # Specify the name of the trapsfilled file, which contains the state of
    # # the charge traps at the end of the preceding exposure
    # detector1.persistence.input_trapsfilled = persist_file

    # Whether or not certain steps should be skipped
    detector1.group_scale.skip = True
    detector1.dq_init.skip = False
    detector1.saturation.skip = False
    # detector1.firstframe.skip = False  # MIRI
    # detector1.lastframe.skip = False  # MIRI
    # detector1.ipc.skip = False  # ?
    detector1.linearity.skip = False
    # detector1.rscd.skip = False  # MIRI
    detector1.dark_current.skip = False
    detector1.ramp_fit.skip = False
    detector1.gain_scale.skip = False

    # save_results
    detector1.group_scale.save_results = False
    detector1.dq_init.save_results = False
    detector1.saturation.save_results = False
    detector1.superbias.save_results = False
    detector1.refpix.save_results = False
    # detector1.firstframe.save_results = False  # MIRI
    # detector1.lastframe.save_results = False  # MIRI
    # detector1.reset.save_results = False  # MIRI
    detector1.linearity.save_results = False
    # detector1.rscd.save_results = False  # MIRI
    detector1.persistence.save_results = False
    detector1.dark_current.save_results = False
    detector1.jump.save_results = False
    detector1.ramp_fit.save_results = False
    detector1.gain_scale.save_results = False

    # Snowball corr
    detector1.jump.skip = False
    detector1.jump.rejection_threshold = 3.0
    # detector1.jump.rejection_threshold = 4
    detector1.jump.expand_large_events = True
    # detector1.jump.min_jump_area = 8
    detector1.jump.use_ellipses = False
    # detector1.jump.expand_factor = 3
    # detector1.jump.after_jump_flag_dn1 = 10
    # detector1.jump.after_jump_flag_time1 = 20
    # detector1.jump.after_jump_flag_dn2 = 1000
    # detector1.jump.after_jump_flag_time2 = 3000
    # detector1.jump.sat_required_snowball=False
    detector1.jump.min_jump_to_flag_neighbors = 2.0
    detector1.jump.maximum_cores = maximum_cores

    # ramp_fit
    detector1.ramp_fit.maximum_cores = maximum_cores

    # Call the run() method
    logger.info('Running Detector 1...')
    run_output = detector1.run(fname_uncal)
    logger.info('Detector 1 completed.')
    return run_output


def run_pipeline_spec2(fname_rate):
    '''Run pipeline of Spec2.'''
    spec2 = Spec2Pipeline()
    spec2.save_results = True
    spec2.output_dir = output_dir
    # skip the flat field correction, since the simulations do not include
    # a full treatment of the throughput spec2.flat_field.skip = True

    # Whether or not certain steps should be skipped
    spec2.assign_wcs.skip = False
    spec2.bkg_subtract.skip = True
    # spec2.imprint_subtract.skip = False
    # spec2.msaflagopen.skip=False
    spec2.flat_field.skip = False
    spec2.srctype.skip = False
    spec2.photom.skip = False
    spec2.cube_build.skip = False
    spec2.extract_1d.skip = True

    spec2.cube_build.weighting = 'drizzle'  # 'emsm' or 'drizzle'
    spec2.cube_build.coord_system = (
        'skyalign'  # 'ifualign', 'skyalign', or 'internal_cal'
    )
    spec2.srctype.source_type = 'EXTENDED'

    logger.info('Running Spec 2...')
    # run_output = spec2(asn_file)
    run_output = spec2.run(fname_rate)
    logger.info('Spec 2 completed.')
    return run_output


def run_pipeline_spec3(fname_cal, extract_1d_skip=True):
    '''Run pipeline of Spec3.'''
    fname_asn = CreateAsnFile(fname_cal).dump()
    crds_config = Spec3Pipeline.get_config_from_reference(fname_asn)
    spec3 = Spec3Pipeline.from_config_section(crds_config)

    spec3.save_results = True
    spec3.output_dir = output_dir
    # # skip this step for now, because the simulations do not include outliers
    # spec3.outlier_detection.skip = True

    # Cube building configuration
    spec3.cube_build.weighting = 'drizzle'  # 'emsm' or 'drizzle'
    spec3.cube_build.coord_system = (
        'skyalign'  # 'ifualign', 'skyalign', or 'internal_cal'
    )

    # Obtain smaller pixscale
    spec3.cube_build.scale1 = 0.05
    spec3.cube_build.scale2 = 0.05

    spec3.assign_mtwcs.skip = False  # modify the wcc considering a moving target over the FoV at each exposure
    # spec3.master_background.skip = True
    spec3.outlier_detection.skip = True
    spec3.cube_build.skip = False
    spec3.extract_1d.skip = extract_1d_skip

    logger.info('Running Spec 3...')
    # result = spec3(asn_file)
    run_output = spec3.run(fname_asn)
    logger.info('Spec 3 completed.')
    return run_output


class CreateAsnFile:
    def __init__(self, fnames):
        self.fnames = fnames
        self.fname_asn = os.path.dirname(fnames[0]) + '/Spec3.json'
        self.science = []
        self.background = []
        self.contain_science_background_files()

    def contain_science_background_files(self):
        for f in self.fnames:
            model = datamodels.open(f)
            is_background = model.meta.observation.bkgdtarg
            if is_background:
                path_x1d = Path(f.replace('cal.fits', 'x1d.fits'))
                if path_x1d.exists():
                    self.background.append(path_x1d.name)
            else:
                self.science.append(Path(f).name)

    def dump(self):
        asn = asn_from_list.asn_from_list(
            self.science, rule=DMS_Level3_Base, product_name='product_name'
        )
        for bkg in self.background:
            asn['products'][0]['members'].append(
                {'expname': bkg, 'exptype': 'background'}
            )

        _, serialized = asn.dump()
        with open(self.fname_asn, 'w') as f:
            f.write(serialized)

        return self.fname_asn


def run_pipeline_after_detector1(fnames):
    '''Original pipeline for a stage between detector1 and spec2'''
    afterdet1 = AfterDetector1Pipeline()

    afterdet1.output_dir = output_dir

    # is_skip
    afterdet1.maskoutlier.skip = False
    afterdet1.subtract_1fnoise.skip = False
    afterdet1.sigmaclip.skip = True

    # parameters
    # afterdet1.maskoutlier.fnames_mask = ['file_nrs1.fits', 'file_nrs2.fits']
    afterdet1.subtract_1fnoise.move_pixels = 5
    afterdet1.sigmaclip.sigma = 10

    logger.info('Running After_Detector1...')
    return [afterdet1.run(f) for f in fnames]


def run_pipeline_after_spec2(fnames, skip_sigmaclip=False, skip_background=False):
    '''Original pipeline for a stage between spec2 and spec3'''
    afterspec2 = AfterSpec2Pipeline()

    afterspec2.output_dir = output_dir

    # is_skip
    afterspec2.failed_slit_open.skip = False
    afterspec2.sigmaclip.skip = skip_sigmaclip
    afterspec2.slitedges.skip = False
    afterspec2.global_background.skip = skip_background
    afterspec2.slits_background.skip = True
    afterspec2.objmask.skip = False  # default = True

    # save_results
    afterspec2.sigmaclip.save_results = False
    afterspec2.global_background.save_results = False

    # parameters
    afterspec2.sigmaclip.sigma = 5

    # Set object mask if needed
    if not afterspec2.objmask.skip:
        # Reference 3D cube; use one before WCS fine tuning
        afterspec2.objmask.fname3d = (
            output_dir + 'product_name_g395h-f290lp_drrizle_s3d.fits'
        )
        # Set object positions / radii / wavelengths for mask
        afterspec2.objmask.positions = SkyCoord(
            [
                ('00h14m24.9217s', '-30d22m56.160s'),
                ('00h14m24.9291s', '-30d22m54.956s'),
                ('00h14m24.9098s', '-30d22m54.936s'),
                ('00h14m24.8742s', '-30d22m54.998s'),
                ('00h14m24.7796s', '-30d22m56.020s'),
            ]
        )
        afterspec2.objmask.radii = [
            0.4,
            0.4,
            0.3,
            0.3,
            0.4,
        ] * u.arcsec
        afterspec2.objmask.waves = [
            [4.3, 4.5],
            [4.4445, 4.45],
            [4.442, 4.448],
            [4.442, 4.447],
            [4.44, 4.446],
        ] * u.um

    logger.info('Running After_Spec2...')
    return [afterspec2.run(f) for f in fnames]


if __name__ == '__main__':
    main()

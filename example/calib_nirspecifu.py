'''Calibration script.

Assumed Directory Structure:
    parent
    |--raw
    |   |--jw01840011001_02101_00001_nrs1_uncal.fits
    |   |--jw01840011001_02101_00001_nrs2_uncal.fits
    |   |--jw01840011001_02101_00002_nrs1_uncal.fits
    |   |--jw01840011001_02101_00002_nrs2_uncal.fits
    |   |--jw01840011001_02101_00003_nrs1_uncal.fits
    |   |--jw01840011001_02101_00003_nrs2_uncal.fits
    |   |--jw01840011001_02101_00004_nrs1_uncal.fits
    |   |--jw01840011001_02101_00004_nrs2_uncal.fits
    |
    |--calib
    |
    |--script
        |--calib_nirspecifu.py
        |--calib_nirspecifu.yaml

Run:
    python ./script/calib_XXX.py
'''

import os
from pathlib import Path
import logging

os.environ["CRDS_PATH"] = '/home/sugayu/nas/data/JWST/crds_cache'
os.environ["CRDS_SERVER_URL"] = 'https://jwst-crds.stsci.edu'
os.environ["CRDS_CONTEXT"] = 'jwst_1364.pmap'

from jwstgo1840.nirspec import JWSTPipelineRunner, JWSTPipelineConfig


logger = logging.getLogger(__name__)
FILE_CONFIG = 'script/config_nirspec.yaml'


##
def main_1st():
    global FILE_CONFIG
    config = JWSTPipelineConfig.load(FILE_CONFIG)
    Path(config.output_dir).mkdir(exist_ok=True)

    jwstpipe = JWSTPipelineRunner(
        product_name=config.target_name,
        output_dir=config.output_dir,
    )
    fnames = config.filenames_uncal

    # =====Main Pipeline=====
    fnames = jwstpipe.run_detector1(fnames, maximum_cores='quarter')

    fnames = jwstpipe.run_after_detector1(fnames)

    fnames = jwstpipe.run_spec2(fnames, maximum_cores=8)

    jwstpipe.prepare_afterspec2_1strun()
    fnames = jwstpipe.run_after_spec2(fnames)

    jwstpipe.prepare_spec3(fnames, firstrun=True)
    fname = jwstpipe.run_spec3(fnames)


def main_2nd():
    global FILE_CONFIG
    config = JWSTPipelineConfig.load(FILE_CONFIG)
    jwstpipe = JWSTPipelineRunner(
        product_name=config.target_name,
        output_dir=config.output_dir,
    )
    fnames = config.filenames_cal1

    # =====Main Pipeline=====
    jwstpipe.set_mask(config.filename_3d, config.apertures)
    jwstpipe.prepare_afterspec2_2ndrun()
    fnames = jwstpipe.run_after_spec2(fnames)

    jwstpipe.prepare_spec3(fnames, firstrun=False)
    fname = jwstpipe.run_spec3(fnames)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    main_1st()
    main_2nd()

'''Test of runner.py.'''

import importlib
from ...nirspec.masking import Aperture3D
from ...nirspec.runner import JWSTPipelineConfig


##
def test_JWSTPipelineConfig() -> None:
    path = importlib.resources.files('jwstgo1840.test.nirspec').joinpath('config.yaml')
    config = JWSTPipelineConfig.load(path)
    assert config.uncal[0].name == 'jw0100000_02000_f00001_nrs1_uncal.fits'
    assert config.rate[0].name == 'jw0100000_02000_f00001_nrs1_rate.fits'
    assert isinstance(config.apertures[0], Aperture3D)

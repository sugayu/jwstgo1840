'''Scripts of runner of each pipeline.'''

from typing import Self, Sequence
from importlib.resources.abc import Traversable
import logging
from copy import deepcopy
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import yaml

from jwst.pipeline import Detector1Pipeline, Spec2Pipeline, Spec3Pipeline
from jwstgo1840.nirspec import (
    AfterDetector1Pipeline,
    AfterSpec2Pipeline,
    AfterSpec3Pipeline,
    CreateAsnFile,
)
from jwstgo1840.nirspec import masking
from jwstgo1840.nirspec.masking import Aperture3D


logger = logging.getLogger(__name__)


__all__ = ['JWSTPipelineRunner', 'JWSTPipelineConfig']


##
class JWSTPipelineRunner:
    '''Run JWST pipelines.'''

    def __init__(
        self, product_name: str, output_dir: str, without_custom: bool = False
    ) -> None:
        self.product_name = product_name
        if isinstance(output_dir, Path):
            output_dir = str(output_dir)
        self.output_dir = output_dir
        self.detector1 = Detector1Pipeline()
        self.spec2 = Spec2Pipeline()
        self.afterdet1 = AfterDetector1Pipeline()
        self.afterspec2 = AfterSpec2Pipeline()

        if not without_custom:
            self.custom_detector1(self.detector1)
            self.custom_spec2(self.spec2)
            self.custom_afterdet1(self.afterdet1)
            self.custom_afterspec2(self.afterspec2)

    def run_detector1(
        self,
        fnames: Sequence[str | Path],
        maximum_cores='None',
    ) -> list[Path]:
        '''Run pipeline of Detector1.'''
        detector1 = self.detector1
        detector1.jump.maximum_cores = maximum_cores
        detector1.ramp_fit.maximum_cores = maximum_cores

        # Call the run() method
        fnames_output = []
        logger.info('Running Detector 1...')
        for fname in fnames:
            _detector1 = deepcopy(detector1)
            run_output = _detector1.run(fname)
            fnames_output.append(Path(_detector1.make_output_path()))
        logger.info('Detector 1 completed.')

        return fnames_output

    def custom_detector1(self, detector1: Detector1Pipeline) -> None:
        '''Custom paramters of Detector1'''
        # Set some parameters that pertain to the entire pipeline
        detector1.output_dir = self.output_dir
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
        detector1.jump.rejection_threshold = 4.0  # 3.0
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
        detector1.jump.min_jump_to_flag_neighbors = 10.0  # 2.0

    def run_spec2(
        self,
        fnames: Sequence[str | Path],
        maximum_cores: int = 1,
    ) -> list[Path]:
        '''Run pipeline of Spec2.'''
        logger.info('Running Spec 2...')

        if maximum_cores == 1:
            fnames_output = [self._run_spec2(fname) for fname in fnames]

        elif maximum_cores > 1:
            with ProcessPoolExecutor(maximum_cores) as exe:
                fnames_output = list(exe.map(self._run_spec2, fnames))

        logger.info('Spec 2 completed.')
        return fnames_output

    def _run_spec2(self, fname: str | Path) -> Path:
        '''Helper to run pipeline of Spec2.'''
        # run_output = spec2(asn_file)
        _spec2 = deepcopy(self.spec2)
        run_output = _spec2.run(fname)
        fname_output = _spec2.make_output_path(suffix='cal')
        return Path(fname_output)

    def custom_spec2(self, spec2: Spec2Pipeline) -> None:
        '''Custom paramters of Detector1'''
        spec2.save_results = True
        spec2.output_dir = self.output_dir
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

    def prepare_spec3(
        self,
        fnames_cal: Sequence[str | Path],
        firstrun: bool,
    ) -> None:
        '''Run pipeline of Spec3.'''
        if firstrun:
            product_name = self.product_name + '_1strun'
        else:
            product_name = self.product_name
        self.fname_asn = CreateAsnFile(fnames_cal).dump(product_name)
        crds_config = Spec3Pipeline.get_config_from_reference(self.fname_asn)
        self.spec3 = Spec3Pipeline.from_config_section(crds_config)
        self.custom_spec3(self.spec3)

    def run_spec3(self, fnames_cal: Sequence[str | Path]) -> str:
        '''Run pipeline of Spec3.'''
        spec3 = self.spec3

        logger.info('Running Spec 3...')
        # result = spec3(asn_file)
        run_output = spec3.run(self.fname_asn)
        logger.info('Spec 3 completed.')
        fname_output = self.spec3.make_output_path()

        return fname_output

    def custom_spec3(self, spec3: Spec3Pipeline) -> None:
        '''Custom parameters of Spec3.'''
        spec3.save_results = True
        spec3.output_dir = self.output_dir
        # # skip this step for now, because the simulations do not include outliers
        # spec3.outlier_detection.skip = True

        # Cube building configuration
        spec3.cube_build.weighting = 'drizzle'  # 'emsm' or 'drizzle'
        spec3.cube_build.coord_system = (
            'skyalign'  # 'ifualign', 'skyalign', or 'internal_cal'
        )

        # Obtain smaller pixscale
        spec3.cube_build.scalexy = 0.05
        # spec3.cube_build.scale1 = 0.05
        # spec3.cube_build.scale2 = 0.05

        spec3.assign_mtwcs.skip = False  # modify the wcc considering a moving target over the FoV at each exposure
        # spec3.master_background.skip = True
        spec3.outlier_detection.skip = True
        spec3.cube_build.skip = False
        spec3.extract_1d.skip = False

    def run_after_detector1(self, fnames: Sequence[Path | str]) -> list[Path]:
        '''Original pipeline for a stage between detector1 and spec2'''
        logger.info('Running After_Detector1...')
        return [self.afterdet1.run(f) for f in fnames]

    def custom_afterdet1(self, afterdet1: AfterDetector1Pipeline) -> None:
        '''Custom parameters of AfterDetector1Pipeline.'''
        afterdet1.output_dir = self.output_dir

        # is_skip
        afterdet1.maskoutlier.skip = False
        afterdet1.subtract_1fnoise.skip = False
        afterdet1.sigmaclip.skip = True

        # parameters
        # afterdet1.maskoutlier.fnames_mask = files_maskoutlier
        afterdet1.subtract_1fnoise.move_pixels = 5
        afterdet1.sigmaclip.sigma = 10

    def run_after_spec2(self, fnames: Sequence[Path | str]) -> list[Path]:
        '''Original pipeline for a stage between spec2 and spec3'''
        logger.info('Running After_Spec2...')
        return [self.afterspec2.run(f) for f in fnames]

    def custom_afterspec2(self, afterspec2: AfterSpec2Pipeline) -> None:
        '''Custom parameters of AfterSpec2Pipeline.'''
        afterspec2.output_dir = self.output_dir

        # is_skip
        afterspec2.failed_slit_open.skip = False
        afterspec2.sigmaclip.skip = False
        afterspec2.wisesigmaclip.skip = True
        afterspec2.slitedges.skip = False
        afterspec2.global_background.skip = False
        afterspec2.slits_background.skip = True
        afterspec2.objmask.skip = False
        afterspec2.failed_fluxcalib.skip = False

        # save_results
        afterspec2.sigmaclip.save_results = False
        afterspec2.global_background.save_results = False

        # parameters
        afterspec2.sigmaclip.sigma = 5
        afterspec2.wisesigmaclip.sigma = 5

    def prepare_afterspec2_1strun(self) -> None:
        '''Custom parameters of AfterSpec2Pipeline for the 1st run.'''
        afterspec2 = self.afterspec2

        # is_skip
        afterspec2.global_background.skip = True
        afterspec2.objmask.skip = True
        afterspec2.wisesigmaclip.skip = True

    def prepare_afterspec2_2ndrun(self) -> None:
        '''Custom parameters of AfterSpec2Pipeline for the 2nd run.'''
        afterspec2 = self.afterspec2

        # is_skip
        afterspec2.global_background.skip = False
        afterspec2.objmask.skip = False
        afterspec2.wisesigmaclip.skip = True

        # save_results
        afterspec2.global_background.save_results = True

        afterspec2.set_suffix(index=3)

    def set_mask(self, fname3d: str, apertures: list[Aperture3D]) -> None:
        afterspec2 = self.afterspec2
        afterspec2.objmask.skip = False

        # Reference 3D cube; use one before WCS fine tuning
        afterspec2.objmask.fname3d = fname3d
        afterspec2.objmask.apertures = apertures


class JWSTPipelineConfig:
    '''This can treat config files.'''

    def __init__(self, config: dict) -> None:
        self._config = config
        self._filenames = self._config['common']['filenames_uncal']

        p = Path(self._config['common']['data_dir'])
        if not p.exists():
            raise FileNotFoundError(
                f'No directory {p.absolute()}. Is the current working directory correct?'
            )
        self._output_path = Path(self._config['common']['output_dir'])

    @classmethod
    def load(cls, fname_config: Traversable | str) -> Self:
        if isinstance(fname_config, str):
            fname_config = Path(fname_config)
        with fname_config.open('r') as f:
            config = yaml.safe_load(f)
        return cls(config)

    @property
    def target_name(self) -> str:
        return self._config['common']['target_name']

    @property
    def data_dir(self) -> Path:
        return Path(self._config['common']['data_dir'])

    @property
    def output_dir(self) -> str:
        return self._config['common']['output_dir']

    @property
    def filenames_uncal(self) -> list[Path]:
        return [self.data_dir / f for f in self._filenames]

    @property
    def filenames_rate(self) -> list[Path]:
        return [self._output_path / f.replace('uncal', 'rate') for f in self._filenames]

    @property
    def filenames_rate1(self) -> list[Path]:
        return [
            self._output_path / f.replace('uncal', '1_rate') for f in self._filenames
        ]

    @property
    def filenames_cal(self) -> list[Path]:
        return [self._output_path / f.replace('uncal', 'cal') for f in self._filenames]

    @property
    def filenames_cal1(self) -> list[Path]:
        return [
            self._output_path / f.replace('uncal', '1_cal') for f in self._filenames
        ]

    @property
    def filenames_cal2(self) -> list[Path]:
        return [
            self._output_path / f.replace('uncal', '2_cal') for f in self._filenames
        ]

    @property
    def filenames_cal3(self) -> list[Path]:
        return [
            self._output_path / f.replace('uncal', '3_cal') for f in self._filenames
        ]

    @property
    def filename_3d(self) -> str:
        return self.output_dir + self._config['main_2nd']['filename3d']

    @property
    def apertures(self) -> list[Aperture3D]:
        apertures: list[Aperture3D] = []
        if 'mask' in self._config['main_2nd'].keys():
            for ClassAperture3D, values in self._config['main_2nd']['mask'].items():
                SubAperture3D: Aperture3D = getattr(masking, ClassAperture3D)
                apertures += SubAperture3D.from_config(values)
        return apertures

    @property
    def can_1st(self) -> bool:
        return self._config.get('main_1st', None) is not None

    @property
    def can_2nd(self) -> bool:
        return self._config.get('main_2nd', None) is not None

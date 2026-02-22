'''Pipelines
'''

from __future__ import annotations
from typing import Sequence
from importlib import resources
from importlib.abc import Traversable
from pathlib import Path
from astropy.io import fits

from jwst.associations import asn_from_list
from jwst.associations.lib.rules_level3_base import DMS_Level3_Base
from jwst import datamodels

from .background import (
    subtract_1fnoises_from_detector,
    subtract_global_background,
    subtract_slits_background,
    ConfigSubtractBackground,
    ConfigSubtractGlobalBackground,
    ConfigSubtractSlitsBackground,
)
from .masking import (
    masking_slitedges,
    masking_msa_failed_open,
    ConfigMaskingSlitedge,
    ConfigMaskingFailedSlitOpen,
    ConfigMaskingObjects,
    masking_objects3D,
)
from .outlier import (
    sigmaclip,
    MaskOutliers,
    mask_failedfluxcalibpix,
    ConfigSigmaClip,
    ConfigMaskOutliers,
    ConfigMaskFailedFluxCalib,
)
from .filtergratingflag import can_process_nrs2, ConfigCanProcessNRS2
from .jwst import IFUImageModel


##
class AfterDetector1Pipeline:
    '''Pipeline to run after Detector1Pipeline.'''

    output_dir: Path | str | None = None

    def __init__(self) -> None:
        self.sigmaclip = ConfigSigmaClip()
        self.maskoutlier = ConfigMaskOutliers()
        self.subtract_1fnoise = ConfigSubtractBackground()
        self.check_process_nrs2 = ConfigCanProcessNRS2()

    def run(self, filename: str | Path) -> Path:
        '''Run pipeline.'''
        if isinstance(filename, str):
            filename = Path(filename)
        datamodel: IFUImageModel = datamodels.open(filename)

        if not self.check_process_nrs2.skip:
            detector = datamodel.meta.instrument.detector
            is_nrs2 = detector == 'NRS2'
            if is_nrs2 and (not can_process_nrs2(datamodel)):
                grating = datamodel.meta.instrument.grating
                filter_ = datamodel.meta.instrument.filter
                raise ValueError(
                    f'In this setup of {grating}/{filter_},'
                    'the spectra do not extend to nrs2. '
                    'Please remove nrs2 from the input data set.'
                )

        if not self.maskoutlier.skip:
            if self.maskoutlier.fnames_mask == []:
                self.maskoutlier.fnames_mask = self.find_pixelmaskfiles(datamodel)
            mask_outlier = MaskOutliers(self.maskoutlier)
            datamodel.dq = mask_outlier(datamodel, filename.name)

        if not self.subtract_1fnoise.skip:
            datamodel.data = subtract_1fnoises_from_detector(
                datamodel.data, datamodel.dq, self.subtract_1fnoise.move_pixels
            )

        if not self.sigmaclip.skip:
            datamodel.dq, _ = sigmaclip(
                datamodel.data, datamodel.dq, sigma=self.sigmaclip.sigma
            )

        path = Path(filename)
        fsave = path.name.replace('_rate', '_1_rate')
        output_dir = self.path_output_dir(path)
        datamodel.save(output_dir / fsave)

        datamodel.close()
        return output_dir / fsave

    def path_output_dir(self, fname: Path) -> Path:
        output_dir: Path | str
        if self.output_dir is None:
            output_dir = fname.parent
        else:
            output_dir = self.output_dir
        if not isinstance(output_dir, Path):
            path = Path(output_dir)
        return path

    @staticmethod
    def find_pixelmaskfiles(datamodel: datamodels.IFUImageModel) -> list[Traversable]:
        '''Find filenames of pixel masks according to fileters and gratings.

        This function is used to find masks for MaskOutliers.
        '''
        grating = datamodel.meta.instrument.grating
        filter_ = datamodel.meta.instrument.filter
        root = resources.files('jwstgo1840.nirspec')
        fnames_mask = [
            f'data/pixelmask_{filter_}{grating}_nrs1.fits',
            f'data/pixelmask_{filter_}{grating}_nrs2.fits',
        ]

        fnames_mask = [
            'data/pixelmask_nrs1.fits',
            'data/pixelmask_nrs2.fits',
        ]
        return [root.joinpath(f) for f in fnames_mask]


class AfterSpec2Pipeline:
    '''Pipeline to run after Spec2Pipeline.'''

    suffix = '_2'
    output_dir: Path | str | None = None

    def __init__(self) -> None:
        self.failed_slit_open = ConfigMaskingFailedSlitOpen()
        self.sigmaclip = ConfigSigmaClip()
        self.slitedges = ConfigMaskingSlitedge()
        self.global_background = ConfigSubtractGlobalBackground()
        self.slits_background = ConfigSubtractSlitsBackground()
        self.objmask = ConfigMaskingObjects()
        self.failed_fluxcalib = ConfigMaskFailedFluxCalib()
        self.suffix = self.suffix

    def run(self, filename: str | Path) -> Path:
        '''Run pipeline.'''
        datamodel = datamodels.open(filename)
        path = Path(filename)

        if not self.failed_slit_open.skip:
            datamodel = masking_msa_failed_open(datamodel)

        if not self.failed_fluxcalib.skip:
            datamodel = mask_failedfluxcalibpix(datamodel, filename)

        # Mask objects
        if not self.objmask.skip:
            self.objmask.check_welldefined()
            datamodel = masking_objects3D(
                datamodel,
                self.objmask.fname3d,
                self.objmask.apertures,
            )

        if not self.sigmaclip.skip:
            datamodel.dq, clmask = sigmaclip(
                datamodel.data, datamodel.dq, sigma=self.sigmaclip.sigma
            )
            if self.sigmaclip.save_results:
                fsave = path.name.replace('_1_cal', self.suffix + '_cal_clipped')
                output_dir = self.path_output_dir(path)
                fits.writeto(output_dir / fsave, clmask.astype(int), overwrite=True)

        if not self.slitedges.skip:
            datamodel, _ = masking_slitedges(datamodel)

        if not self.global_background.skip:
            datamodel, bk2d = subtract_global_background(
                datamodel, move_pixels=self.global_background.move_pixels
            )
            if self.global_background.save_results:
                fsave = path.name.replace('_1_cal', self.suffix + '_globalbkg')
                output_dir = self.path_output_dir(path)
                fits.writeto(output_dir / fsave, bk2d, overwrite=True)

        if not self.slits_background.skip:
            datamodel = subtract_slits_background(datamodel)

        fsave = path.name.replace('_1_cal', self.suffix + '_cal')
        output_dir = self.path_output_dir(path)
        datamodel.save(output_dir / fsave)

        datamodel.close()
        return output_dir / fsave

    def path_output_dir(self, fname: Path) -> Path:
        output_dir: Path | str
        if self.output_dir is None:
            output_dir = fname.parent
        else:
            output_dir = self.output_dir
        if not isinstance(output_dir, Path):
            path = Path(output_dir)
        return path

    def set_suffix(self, index: int) -> None:
        '''Set suffix for different runs.'''
        self.suffix = self.suffix.replace('2', str(index))


class AfterSpec3Pipeline:
    '''Pipeline to run after Spec3Pipeline.'''

    def __init__(self) -> None:
        pass

    def run(self, filename: str) -> None:
        '''Run pipeline.'''
        pass


class CreateAsnFile:
    def __init__(self, fnames: Sequence[str | Path]):
        self.fnames = fnames
        self.fname_asn = Path(fnames[0]).parent / 'Spec3.json'
        self.science: list[str] = []
        self.background: list[str] = []
        self.contain_science_background_files()

    def contain_science_background_files(self):
        for f in self.fnames:
            with datamodels.open(f) as model:
                is_background = model.meta.observation.bkgdtarg
                if is_background:
                    path_x1d = Path(f.replace('cal.fits', 'x1d.fits'))
                    if path_x1d.exists():
                        self.background.append(path_x1d.name)
                else:
                    self.science.append(Path(f).name)

    def dump(self, product_name: str = 'product_name'):
        asn = asn_from_list.asn_from_list(
            self.science, rule=DMS_Level3_Base, product_name=product_name
        )
        for bkg in self.background:
            asn['products'][0]['members'].append(
                {'expname': bkg, 'exptype': 'background'}
            )

        _, serialized = asn.dump()
        with self.fname_asn.open('w') as f:
            f.write(serialized)

        return self.fname_asn

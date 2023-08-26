'''Pipelines
'''
from __future__ import annotations
import os
from pathlib import Path
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
    ConfigMaskingObj,
    masking_objects3D,
)
from .outlier import sigmaclip, MaskOutliers, ConfigSigmaClip, ConfigMaskOutliers
from .filtergratingflag import can_process_nrs2, ConfigCanProcessNRS2
from astropy.io import fits
from jwst.associations import asn_from_list
from jwst.associations.lib.rules_level3_base import DMS_Level3_Base


##
class AfterDetector1Pipeline:
    '''Pipeline to run after Detector1Pipeline.'''

    output_dir: Path | str | None = None

    def __init__(self) -> None:
        self.sigmaclip = ConfigSigmaClip()
        self.maskoutlier = ConfigMaskOutliers()
        self.subtract_1fnoise = ConfigSubtractBackground()
        self.check_process_nrs2 = ConfigCanProcessNRS2()

    def run(self, filename: str) -> str:
        '''Run pipeline.'''
        datamodel = datamodels.open(filename)

        if not self.check_process_nrs2.skip:
            detector = datamodel.meta.instrument.detector
            is_nrs2 = detector == 'NRS2'
            if is_nrs2 and (not can_process_nrs2(datamodel)):
                grating = datamodel.meta.instrument.grating
                filter_ = datamodel.meta.instrument.filter
                raise ValueError(
                    f'In this setup of {grating}/{filter_}, the spectra do not extend to nrs2. '
                    f'Please remove nrs2 from the input data set.'
                )

        if not self.maskoutlier.skip:
            if self.maskoutlier.fnames_mask == []:
                fnames_data = ['pixelmask_nrs1.fits', 'pixelmask_nrs2.fits']
                path_data = Path(__file__).resolve().parent / 'data/'
                self.maskoutlier.fnames_mask = [str(path_data / f) for f in fnames_data]
            maskoutlier = MaskOutliers(self.maskoutlier.fnames_mask)
            datamodel.dq = maskoutlier.flag_pixels(datamodel.dq, filename)

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

        return str(output_dir / fsave)

    def path_output_dir(self, fname: Path) -> Path:
        output_dir: Path | str
        if self.output_dir is None:
            output_dir = fname.parent
        else:
            output_dir = self.output_dir
        if not isinstance(output_dir, Path):
            path = Path(output_dir)
        return path


class AfterSpec2Pipeline:
    '''Pipeline to run after Spec2Pipeline.'''

    output_dir: Path | str | None = None

    def __init__(self) -> None:
        self.failed_slit_open = ConfigMaskingFailedSlitOpen()
        self.sigmaclip = ConfigSigmaClip()
        self.slitedges = ConfigMaskingSlitedge()
        self.global_background = ConfigSubtractGlobalBackground()
        self.slits_background = ConfigSubtractSlitsBackground()
        self.objmask = ConfigMaskingObj()

    def run(self, filename: str) -> str:
        '''Run pipeline.'''
        datamodel = datamodels.open(filename)
        path = Path(filename)

        if not self.failed_slit_open.skip:
            datamodel = masking_msa_failed_open(datamodel)

        # Mask objects
        if not self.objmask.skip:
            self.objmask.check_welldefined()
            datamodel = masking_objects3D(
                datamodel,
                self.objmask.fname3d,
                self.objmask.positions,
                self.objmask.radii,
                self.objmask.waves,
            )

        if not self.sigmaclip.skip:
            datamodel.dq, clmask = sigmaclip(
                datamodel.data, datamodel.dq, sigma=self.sigmaclip.sigma
            )
            if self.sigmaclip.save_results:
                fsave = path.name.replace('_1_cal', '_2_cal_clipped')
                output_dir = self.path_output_dir(path)
                fits.writeto(output_dir / fsave, clmask.astype(int), overwrite=True)

        if not self.slitedges.skip:
            datamodel, _ = masking_slitedges(datamodel)

        if not self.global_background.skip:
            datamodel, bk2d = subtract_global_background(datamodel)
            if self.global_background.save_results:
                fsave = path.name.replace('_1_cal', '_2_globalbkg')
                output_dir = self.path_output_dir(path)
                fits.writeto(output_dir / fsave, bk2d, overwrite=True)

        if not self.slits_background.skip:
            datamodel = subtract_slits_background(datamodel)

        fsave = path.name.replace('_1_cal', '_2_cal')
        output_dir = self.path_output_dir(path)
        datamodel.save(output_dir / fsave)
        return str(output_dir / fsave)

    def path_output_dir(self, fname: Path) -> Path:
        output_dir: Path | str
        if self.output_dir is None:
            output_dir = fname.parent
        else:
            output_dir = self.output_dir
        if not isinstance(output_dir, Path):
            path = Path(output_dir)
        return path


class AfterSpec3Pipeline:
    '''Pipeline to run after Spec3Pipeline.'''

    def __init__(self) -> None:
        pass

    def run(self, filename: str) -> str:
        '''Run pipeline.'''
        pass


class CreateAsnFile:
    def __init__(self, fnames: list[str]):
        self.fnames = fnames
        self.fname_asn = os.path.dirname(fnames[0]) + '/Spec3.json'
        self.science: list[str] = []
        self.background: list[str] = []
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

    def dump(self, product_name: str = 'product_name'):
        asn = asn_from_list.asn_from_list(
            self.science, rule=DMS_Level3_Base, product_name=product_name
        )
        for bkg in self.background:
            asn['products'][0]['members'].append(
                {'expname': bkg, 'exptype': 'background'}
            )

        _, serialized = asn.dump()
        with open(self.fname_asn, 'w') as f:
            f.write(serialized)

        return self.fname_asn

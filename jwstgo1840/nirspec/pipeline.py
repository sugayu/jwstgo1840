'''Pipelines
'''
from __future__ import annotations
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
from astropy.io import fits


##
class AfterDetector1Pipeline:
    '''Pipeline to run after Detector1Pipeline.'''

    output_dir: Path | str | None = None

    def __init__(self) -> None:
        self.sigmaclip = ConfigSigmaClip()
        self.maskoutlier = ConfigMaskOutliers()
        self.subtract_1fnoise = ConfigSubtractBackground()

    def run(self, filename: str) -> str:
        '''Run pipeline.'''
        datamodel = datamodels.open(filename)
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

        not_skip_objmask = (
            (not self.objmask.skip)
            and (self.objmask.fname3d != '')
            and (self.objmask.positions is not None)
            and (self.objmask.radii is not None)
            and (self.objmask.waves is not None)
        )
        if not_skip_objmask:
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
            datamodel, _ = subtract_global_background(datamodel)

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

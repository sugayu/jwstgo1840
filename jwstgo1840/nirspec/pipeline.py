'''Pipelines
'''
from __future__ import annotations
from pathlib import Path
from jwst import datamodels
from .background import subtract_bacground, ConfigSubtractBackground
from .masking import masking_slitedges, ConfigMaskingSlitedge
from .outlier import sigmaclip, MaskOutliers, ConfigSigmaClip, ConfigMaskOutliers


##
class AfterDetector1Pipeline:
    '''Pipeline to run after Detector1Pipeline.'''

    output_dir: Path | str | None = None

    def __init__(self) -> None:
        self.sigmaclip = ConfigSigmaClip()
        self.maskoutlier = ConfigMaskOutliers()

    def run(self, filename: str) -> str:
        '''Run pipeline.'''
        datamodel = datamodels.open(filename)
        if not self.sigmaclip.skip:
            datamodel.dq, _ = sigmaclip(
                datamodel.data, datamodel.dq, sigma=self.sigmaclip.sigma
            )

        if not self.maskoutlier.skip:
            if self.maskoutlier.fnames_mask == []:
                fnames_data = ['pixelmask_nrs1.fits', 'pixelmask_nrs2.fits']
                path_data = Path(__file__).resolve().parent / 'data/'
                self.maskoutlier.fnames_mask = [str(path_data / f) for f in fnames_data]
            maskoutlier = MaskOutliers(self.maskoutlier.fnames_mask)
            datamodel.dq = maskoutlier.flag_pixels(datamodel.dq, filename)

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
        self.sigmaclip = ConfigSigmaClip()
        self.slitedges = ConfigMaskingSlitedge()
        self.background = ConfigSubtractBackground()

    def run(self, filename: str) -> str:
        '''Run pipeline.'''
        datamodel = datamodels.open(filename)

        if not self.sigmaclip.skip:
            datamodel.dq, _ = sigmaclip(
                datamodel.data, datamodel.dq, sigma=self.sigmaclip.sigma
            )

        if not self.slitedges.skip:
            datamodel, _ = masking_slitedges(datamodel)

        if not self.background.skip:
            datamodel.data = subtract_bacground(
                datamodel.data, datamodel.dq, self.background.move_pixels
            )

        path = Path(filename)
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

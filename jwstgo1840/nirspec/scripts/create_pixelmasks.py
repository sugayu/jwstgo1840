'''Create pixel masks.

Related ADRs:
- ADR0001
'''

from pathlib import Path
from astropy.io import fits
from jwstgo1840.nirspec.outlier import create_pixelmask


##
def main(path: Path, p_save: Path) -> None:

    pathlist = [
        path / 'JADES_1093/F290LPG395H/calib/',
        path / 'JADES_61888/F290LPG395H/calib/',
        path / 'COS-ZS7-1-COS-7.15/F290LPG395M/calib/',
        path / 'GN-108036/F290LPG395M/calib/',
        path / 'GS-9209/F170LPG235H/calib/',  # QSO
        path / 'J0217-0208/F170LPG235H/calib/',
        path / 'LACES104037/F170LPG235M/calib/',
        path / 'LACES94460/F170LPG235M/calib/',
    ]
    fnames = []
    for s in pathlist:
        fnames += list(Path(s).glob('*nrs1_1_rate.fits'))
    mask_nrs1 = create_pixelmask(fnames, sigma=3.0, threshold=0.5)

    fsave = p_save / 'pixelmask_nrs1.fits'
    fits.writeto(fsave, mask_nrs1.astype(int), overwrite=True)

    fnames = []
    for s in pathlist:
        fnames += list(Path(s).glob('*nrs2_1_rate.fits'))
    mask_nrs2 = create_pixelmask(fnames, sigma=3.0, threshold=0.5)

    fsave = p_save / 'pixelmask_nrs2.fits'
    fits.writeto(fsave, mask_nrs2.astype(int), overwrite=True)


if __name__ == '__main__':
    path = Path('your/data/path')
    p_save = Path('pipeline/data/path')
    main(path, p_save)

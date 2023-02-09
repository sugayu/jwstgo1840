'''NIRSpec pipeline
'''
from .background import subtract_bacground
from .masking import masking_slitedges
from .outlier import sigmaclip, MaskOutliers, create_pixelmask

__all__ = [
    'subtract_bacground',
    'masking_slitedges',
    'sigmaclip',
    'MaskOutliers',
    'create_pixelmask',
]

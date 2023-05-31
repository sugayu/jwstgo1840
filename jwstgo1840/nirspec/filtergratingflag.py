'''Flag for filters or gratings
'''
from __future__ import annotations
from dataclasses import dataclass
from jwst import datamodels
import logging

logger = logging.getLogger('debuglog')

__all__ = ['can_process_nrs2']

##
def can_process_nrs2(datamodel: datamodels):
    '''Return true/false if nrs2 detector can be processed.

    The cases to use nrs2 detector are
        only for G140H/F100LP, G235H/F170LP, G395H/F290LP.
    '''
    # detector = datamodel.meta.instrument.detector
    # if detector == 'NRS1':
    #     raise ValueError('The input data is for the detector NRS1.')

    grating = datamodel.meta.instrument.grating
    filter_ = datamodel.meta.instrument.filter  # "_" is to avoid
    setup = grating + '/' + filter_

    if setup in ['G140H/F100LP', 'G235H/F170LP', 'G395H/F290LP']:
        return True
    else:
        return False


@dataclass
class ConfigCanProcessNRS2:
    skip: bool = False

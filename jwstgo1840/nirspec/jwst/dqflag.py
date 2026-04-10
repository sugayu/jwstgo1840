'''Utilities using dq flag
'''

import numpy as np
from stdatamodels.jwst.datamodels import dqflags


##
dqflag = dqflags.pixel


def dqflagging(dq: np.ndarray, mask: np.ndarray, flag: int | str) -> np.ndarray:
    '''Flag dq according to mask.'''
    dq_new, mask_new = np.copy(dq), np.copy(mask)
    already_flagged = is_dqflagged(dq, flag)
    mask_new[already_flagged] = False
    dq_new[mask_new] += dqflag[flag]
    return dq_new


def is_dqflagged(dqmap: np.ndarray, flag: int | str) -> np.ndarray:
    '''Return boolean array of data quality flag.'''
    if isinstance(flag, int):
        bytevalue = 2**flag
    elif isinstance(flag, str):
        bytevalue = dqflag[flag]
    else:
        raise ValueError(
            f'DQ flag key has to be int or str, but the input is {flag} ({type(flag)}).'
        )

    return np.bitwise_and(dqmap, bytevalue).astype(bool)

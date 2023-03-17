'''Utilities using dq flag
'''
import numpy as np
from jwst import datamodels


##
dqflag = datamodels.dqflags.pixel


def dqflagging(dq, mask, flag):
    '''Flag dq according to mask.'''
    dq_new, mask_new = np.copy(dq), np.copy(mask)
    already_flagged = is_dqflagged(dq, flag)
    mask_new[already_flagged] = False
    dq_new[mask_new] += dqflag[flag]
    return dq_new


def is_dqflagged(dq, flag):
    '''Return boolean array of data quality flag.'''
    return np.bitwise_and(dq, dqflag[flag]).astype(bool)

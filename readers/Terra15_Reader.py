# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 14:26:13 2025

@author: jmjeh
"""
import numpy as np
import h5py
from glob import glob
from datetime import datetime, timezone

from .. import Data2D_XT
from .reader_utils import create_spool_common

def timestamt2datetime(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)

def get_time_range(filename):
    with h5py.File(filename,'r') as f:
        ts = f['data_product/posix_time'][0]
        start_time = timestamt2datetime(ts)
        ts = f['data_product/posix_time'][-1]
        end_time = timestamt2datetime(ts)
    return start_time, end_time

def reader(filename, bgtime, edtime):
    with h5py.File(filename,'r') as f:
        timestamps = f['data_product/posix_time'][:]
        timestamps = np.array([timestamt2datetime(ts) for ts in timestamps])

        # get depth info
        Nx = f['data_product/data'].attrs['nx']
        dx = f['data_product/data'].attrs['dx']
        daxis = np.arange(Nx)*dx

        # read data
        ind = (timestamps>=bgtime) & (timestamps<=edtime)
        ind = np.where(ind)[0]
        data = f['data_product/data'][ind,:]
        timestamps = timestamps[ind]

        DASdata = Data2D_XT.Data2D()
        DASdata.set_time_from_datetime(timestamps)
        DASdata.daxis = daxis
        DASdata.data = data.T
    
    return DASdata

def create_spool(datapath, search_pattern='*.h5'):
    return create_spool_common(datapath,get_time_range,reader,search_pattern=search_pattern)

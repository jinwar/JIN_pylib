from dateutil import parser
from datetime import timedelta, timezone

from .reader_utils import create_spool_common

from .. import Data2D_XT,Spool
import numpy as np
import pandas as pd
import h5py
from glob import glob
from tqdm import tqdm
from datetime import datetime

def timestamt2datetime(ts):
    return datetime.fromtimestamp(ts/1.0e6, tz=timezone.utc).replace(tzinfo=None)

def get_time_range(filename):
    with h5py.File(filename,'r') as f:
        # data = f['Acquisition/Raw[0]/RawData'][:,:]
        # get time info
        # timestr = f['Acquisition/Raw[0]/RawData'].attrs['PartStartTime']
        # start_time = parser.parse(timestr).replace(tzinfo=None)
        # timestr = f['Acquisition/Raw[0]/RawData'].attrs['PartEndTime']
        # end_time = parser.parse(timestr).replace(tzinfo=None)
        ts = f['Acquisition/Raw[0]/RawDataTime'][0]
        start_time = timestamt2datetime(ts)
        ts = f['Acquisition/Raw[0]/RawDataTime'][-1]
        end_time = timestamt2datetime(ts)
    return start_time, end_time

def reader(filename, bgtime, edtime):

    with h5py.File(filename,'r') as f:
        # print(filename)
        # data = f['Acquisition/Raw[0]/RawData'][:,:]
        # get time info
       
        timestamps = f['Acquisition/Raw[0]/RawDataTime'][:]
        # timestamps = np.array([timestamt2datetime(ts) for ts in timestamps])

        # get depth info
        Nx = f['Acquisition/Raw[0]/RawData'].shape[0]
        dx = f['Acquisition'].attrs['SpatialSamplingInterval']
        daxis = np.arange(Nx)*dx

        # read data
        # ind = (timestamps>=bgtime) & (timestamps<=edtime)

        ind = (timestamps>=bgtime.timestamp()*1.0e6) & (timestamps<edtime.timestamp()*1.0e6)
        ind = np.where(ind)[0]
        data = f['Acquisition/Raw[0]/RawData'][:, ind]
        timestamps = timestamps[ind]
        timestamps = np.array([timestamt2datetime(ts) for ts in timestamps])

        DASdata = Data2D_XT.Data2D()
        DASdata.set_time_from_datetime(timestamps)
        DASdata.daxis = daxis
        #DASdata.data = data.T
        DASdata.data = data
    
    return DASdata

def create_spool(datapath, search_pattern='*.hdf5'):
    return create_spool_common(datapath,get_time_range,reader,search_pattern=search_pattern)

def create_spool_from_files(files):
    bgtimes = []
    edtimes = []
    print('Indexing Files....')
    for file in tqdm(files):
        bgt, edt = get_time_range(file)
        bgtimes.append(bgt)
        edtimes.append(edt)
    
    df = pd.DataFrame({'file': files, 'start_time': bgtimes, 'end_time': edtimes})
    
    sp = Spool.spool(df, reader, support_partial_reading=True)
    return sp

    



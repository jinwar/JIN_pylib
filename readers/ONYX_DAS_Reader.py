from dateutil import parser
from datetime import timedelta, timezone

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
        # data = f['Acquisition/Raw[0]/RawData'][:,:]
        # get time info
        # dt = 1/f['Acquisition/Raw[0]'].attrs['OutputDataRate']
        # Nt = f['Acquisition/Raw[0]/RawData'].shape[0]
        # timestr = f['Acquisition/Raw[0]/RawData'].attrs['PartStartTime']
        # start_time = parser.parse(timestr).replace(tzinfo=None)
        # taxis = np.arange(Nt)*dt
        # end_time = start_time + timedelta(seconds=taxis[-1])
        # bgt = (bgtime - start_time).total_seconds()
        # edt = (edtime - start_time).total_seconds()
        timestamps = f['Acquisition/Raw[0]/RawDataTime'][:]
        timestamps = np.array([timestamt2datetime(ts) for ts in timestamps])

        # get depth info
        Nx = f['Acquisition/Raw[0]/RawData'].shape[1]
        dx = f['Acquisition'].attrs['SpatialSamplingInterval']
        daxis = np.arange(Nx)*dx

        # read data
        ind = (timestamps>=bgtime) & (timestamps<=edtime)
        data = f['Acquisition/Raw[0]/RawData'][ind,:]
        timestamps = timestamps[ind]

        DASdata = Data2D_XT.Data2D()
        DASdata.set_time_from_datetime(timestamps)
        DASdata.daxis = daxis
        DASdata.data = data.T
    
    return DASdata

def create_spool(datapath,extension = '.h5'):
    files = glob(datapath+'/*'+extension)
    bgtimes = []
    edtimes = []
    final_files = []
    print('Indexing Files....')
    for file in tqdm(files):
        try:
            bgt,edt = get_time_range(file)
            bgtimes.append(bgt)
            edtimes.append(edt)
            final_files.append(file)
        except:
            print('Error reading file:',file)
            continue
    
    df = pd.DataFrame({'file':final_files,'start_time':bgtimes,'end_time':edtimes})

    sp = Spool.spool(df,reader, support_partial_reading=True)
    return sp



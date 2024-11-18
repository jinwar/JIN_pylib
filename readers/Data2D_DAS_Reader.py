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
        start_time = datetime.strptime(f['data'].attrs['start time'],'%Y%m%d_%H%M%S.%f')
        bgtime = start_time + timedelta(seconds=f['taxis'][0])
        edtime = start_time + timedelta(seconds=f['taxis'][-1])
        
    return bgtime, edtime

def reader(filename):

    DASdata = Data2D_XT.load_h5(filename)
    
    return DASdata

def create_spool(datapath,extension = '.h5'):
    files = glob(datapath+'/*'+extension)
    bgtimes = []
    edtimes = []
    print('Indexing Files....')
    for file in tqdm(files):
        bgt,edt = get_time_range(file)
        bgtimes.append(bgt)
        edtimes.append(edt)
    
    df = pd.DataFrame({'file':files,'start_time':bgtimes,'end_time':edtimes})

    sp = Spool.spool(df,reader, support_partial_reading=False)
    return sp



from datetime import timedelta, timezone
from .. import Data2D_XT
import h5py
from datetime import datetime

from .reader_utils import create_spool_common

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

def create_spool(datapath, search_pattern='*.h5'):
    return create_spool_common(datapath,get_time_range,reader,search_pattern=search_pattern, support_partial_reading=False)

import pandas as pd
from .Data2D_XT import Data2D
import numpy as np
from datetime import datetime
import h5py



def read_NB_csv(filename):
    df = pd.read_csv(filename)

    timestrs = list(df.columns[1:])

    timestamps = [datetime.strptime(t,'%m/%d/%Y %H:%M:%S.%f') for t in timestrs]
    timestamps = np.array(timestamps)
    mds = df['Length'].values
    data = df.iloc[:,1:].values

    chans = np.arange(data.shape[0])

    Ddata = Data2D()
    Ddata.set_data(data)
    Ddata.set_mds(mds)
    Ddata.set_chans(chans)
    Ddata.set_time_from_datetime(timestamps)
    Ddata.timestamps = timestamps
    
    return Ddata

class NB_h5:
    def __init__(self,filename,transverse = False):
        with h5py.File(filename, 'r') as f:
            depth = f['depth'][:]
            timestamps = f['stamps'][:]
        timestamps = [s.decode('utf-8') for s in timestamps]
        timestamps = pd.to_datetime(timestamps)

        self.timestamps = timestamps
        self.depth = depth
        self.filename = filename
        self.transverse = transverse
    
    
    def select(self, time=(None,None), depth=(None,None)):

        def _select_range(axis,val_range):
            if val_range[0] is None:
                bgval = axis[0]
            else:
                bgval = val_range[0]
            if val_range[1] is None:
                edval = axis[-1]
            else:
                edval = val_range[1]
            val_range_ind = np.where((axis>=bgval)&(axis<=edval))[0]
            return val_range_ind[0],val_range_ind[-1]

        bg_t,ed_t = _select_range(self.timestamps,time)
        bg_d,ed_d = _select_range(self.depth,depth)

        print('time index:',bg_t,ed_t)
        print('depth index:',bg_d,ed_d)

        with h5py.File(self.filename, 'r') as f:
            if self.transverse:
                data = f['data'][bg_t:ed_t,bg_d:ed_d]
                data = data.T
            else:
                data = f['data'][bg_d:ed_d,bg_t:ed_t]
        
        DSSdata = Data2D()
        DSSdata.data = data
        DSSdata.daxis = self.depth[bg_d:ed_d]
        DSSdata.set_time_from_datetime(self.timestamps[bg_t:ed_t])
        return DSSdata



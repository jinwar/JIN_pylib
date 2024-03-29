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
            datashape = f['data'].shape
        timestamps = [s.decode('utf-8') for s in timestamps]
        timestamps = pd.to_datetime(timestamps)

        self.timestamps = timestamps
        self.depth = depth
        self.filename = filename
        self.transverse = transverse
        self.datashape = datashape
    
    def info(self):
        print(f'data shape (depth,time): {self.datashape}')
        print(f'Time Range: {self.timestamps[0]} - {self.timestamps[-1]}')
        print(f'Depth Range: {self.depth[0]} - {self.depth[-1]}')

    
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


def slippage_removal(trc,detect_thres=3,data_point_removal=4
        ,sm_N=2,abs_thres=False,local_std_N=500,is_interp=True):
    # calculate strain rate
    trc_diff = np.diff(trc)
    # detect slippage events
    if abs_thres:
        ind = np.abs(trc_diff)>detect_thres
    else:
        ind = np.abs(trc_diff)>detect_thres*np.std(trc_diff)
    ind = np.where(ind)[0]

    if len(ind)>len(trc)*0.2:
        return trc
    
    # for each slippage, check local variance again
    if not abs_thres:
        ind_to_remove = []
        for i in range(len(ind)-1):
            bgind = np.max((0,ind[i]-local_std_N//2))
            edind = np.min((ind[i]+local_std_N//2,len(trc_diff)))

            local_std = np.std(trc_diff[bgind:edind])
            if np.abs(trc_diff[ind[i]]) < detect_thres*local_std:
                ind_to_remove.append(i)
        
        ind = np.delete(ind,ind_to_remove)

    
    # remove data points after slippage events
    new_ind = []
    for i in ind:
        for j in range(0,data_point_removal):
            if i+j < len(trc_diff)-1:
                new_ind.append(i+j)
            
    if len(new_ind) == 0 :
        return trc
        
    # perform interpolation
    good_ind = np.abs(trc_diff)>-1
    good_ind[new_ind] = False
    x = np.arange(len(trc_diff))
    good_trc_diff = trc_diff[good_ind].copy()
    good_trc_diff = np.convolve(good_trc_diff,np.ones(sm_N)/sm_N,'same')
    if is_interp:
        trc_diff[~good_ind] = np.interp(x[~good_ind],x[good_ind],good_trc_diff)
    else:
        trc_diff[~good_ind] = 0
    # change back to strain change
    trc_cor = np.cumsum(trc_diff)
    trc_cor = np.concatenate(([0],trc_cor))
    return trc_cor


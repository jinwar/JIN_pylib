import numpy as np
from . import Data2D_XT
from datetime import datetime
import h5py



#Define the class to read Treble data from individual file
class read_Treble():
    
    def __init__(self,filename):
        self.filename = filename
        self.gauge_length_chanN = 3
        self.deformation = False
    
    def get_data(self,bgtime,edtime,tz):
        with h5py.File(self.filename,'r') as fp:
            timestamps = np.array([datetime.fromtimestamp(s,tz) # apply time zone
                                   for s in fp['deformation/gps_time'][:]])
            ind = (timestamps>bgtime)&(timestamps<edtime)
            data = fp['deformation/data'][ind,:]
            timestamps = timestamps[ind]
            dx = fp['deformation/data'].attrs['dx']

            # apply_gauge_length
            n = self.gauge_length_chanN//2
            strain_data = np.zeros_like(data)
            for i in range(n,data.shape[1]-n):
                strain_data[:,i] = data[:,i+n]-data[:,i-n]

            DASdata = Data2D_XT.Data2D()  # you need to replace this with your own class
            
            if self.deformation:
                DASdata.data = data.T
            else:
                DASdata.data = strain_data.T
            DASdata.start_time = timestamps[0]
            DASdata.taxis = np.array([dt.total_seconds() for dt in (timestamps-timestamps[0])])
            DASdata.chan = np.arange(data.shape[1])
            DASdata.md = DASdata.chan*dx
            DASdata.attrs['Gauge Length'] = self.gauge_length_chanN*dx
            DASdata.attrs['dx'] = dx
        return DASdata
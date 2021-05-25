import numpy as np
from . import Data2D_XT
from datetime import datetime
import h5py
import pytz
import shutil
import os
from glob import glob
import pandas as pd

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
            DASdata.chans = np.arange(data.shape[1])
            DASdata.mds = DASdata.chans*dx
            DASdata.attrs['Gauge Length'] = self.gauge_length_chanN*dx
            DASdata.attrs['dx'] = dx
            dt = np.median(np.diff(DASdata.taxis))
            DASdata.attrs['dt'] = dt
        return DASdata


class Treble_io():

    def __init__(self,datapath,timezone):
        self.datapath = datapath
        self.timezone = timezone # timezone in pytz
        self.build_database()

    def build_database(self):
        files = glob(self.datapath+'/*/*.hdf5')
        filedf = pd.DataFrame()
        filedf['filename'] = files
        timestamps = []
        for f in files:
            bgc = f.find('YMD',-50)+3
            edc = f.find('_seq',-50)
            t = datetime.strptime(f[bgc:edc],'%Y%m%d-HMS%H%M%S.%f')\
                    .replace(tzinfo=pytz.utc).astimezone(self.timezone) 
            timestamps.append(t)
        filedf['time'] = timestamps
        self.filedf = filedf.sort_values(by='time')

    # define function to file[s] that contains the data
    def get_filename(self,bgtime,edtime):
        filedf = self.filedf
        ind = filedf['time'] < bgtime
        bgfileid = np.where(ind)[0][-1]
        ind = filedf['time'] < edtime
        edfileid = np.where(ind)[0][-1]
        if bgfileid == edfileid:
            filename = [filedf['filename'].iloc[bgfileid]]
        else:
            filename = [filedf['filename'].iloc[bgfileid],filedf['filename'].iloc[edfileid]]
        return filename
    
    def get_data_bydatetime(self,bgtime,edtime):
        files = self.get_filename(bgtime,edtime)
        print('Getting data from file:', files)
        print('time range:',bgtime, edtime)
        if len(files)==1:
            rt = read_Treble(files[0])
            DASdata = rt.get_data(bgtime,edtime,self.timezone)
        if len(files)==2:
            rt = read_Treble(files[0])
            DASdata = rt.get_data(bgtime,edtime,self.timezone)
            rt = read_Treble(files[1])
            DASdata1 = rt.get_data(bgtime,edtime,self.timezone)
            DASdata.right_merge(DASdata1)
        return DASdata
    
    def get_data_bytimestr(self,bgtimestr,edtimestr):
        """
        The input string has to follow format: '%Y/%m/%d %H:%M:%S.%f'
        example: '2021/05/19 13:03:16.000'
        """
        # bgtime = datetime.strptime(bgtimestr,'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=self.timezone)
        bgtime = datetime.strptime(bgtimestr,'%Y/%m/%d %H:%M:%S.%f')
        bgtime = self.timezone.localize(bgtime)
        edtime = datetime.strptime(edtimestr,'%Y/%m/%d %H:%M:%S.%f')
        edtime = self.timezone.localize(edtime)
        return self.get_data_bydatetime(bgtime,edtime)


class Treble_io_colab(Treble_io):

    def __init__(self,datapath,timezone):
        super().__init__(datapath, timezone)
        if not os.path.isdir('./temp'):
            os.mkdir('./temp')

    def get_filename(self,bgtime,edtime):
        files = super().get_filename(bgtime,edtime)
        local_files = []
        for f in files:
            local_file = './temp/'+os.path.basename(f)
            if not os.path.isfile(local_file):
                print('copying '+os.path.basename(f)+' to local')
                shutil.copy(f,local_file)
            local_files.append(local_file)
        return local_files

from . import gjsignal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
from scipy.signal import medfilt2d
import matplotlib.dates as mdates
from dateutil.parser import parse
import h5py
import copy

class Data2D():

    def __init__(self):
        self.data = None   # data, 2D array
        self.start_time = None  # starting time using datetime
        self.taxis = []  # time axis in second from start_time
        self.chans = [] # fiber channel number
        self.mds = []  # fiber physical distance or location
        self.attrs = {'Python Class Version':'1.1'} # data attributes
        self.history = []

    def set_data(self,data):
        self.data = data
    
    def set_time_from_datetime(self,timestamps):
        self.start_time = timestamps[0]
        self.taxis = np.array([(t-timestamps[0]).total_seconds()
             for t in timestamps])
    
    def apply_timeshift(self,ts):
        self.start_time += timedelta(hours=ts)
    
    def cal_timestamp_from_taxis(self):
        timestamps = [self.start_time + timedelta(seconds=t) 
            for t in self.taxis]
        self.timestamps = timestamps
    
    def set_mds(self,mds):
        self.mds = mds
    
    def select_depth(self,mindp,maxdp):
        ind = (self.mds>mindp)&(self.mds<maxdp)
        self.mds = self.mds[ind]
        self.data = self.data[ind,:]
    
    def set_chans(self,chans):
        self.chans = chans
    
    def median_filter(self,kernel_size=(5,3)):
        self.data = medfilt2d(self.data,kernel_size=kernel_size)
    
    def window_data_time(self,bgtime, edtime,reset_startime=True):
        ind = (self.taxis>bgtime)&(self.taxis<edtime)
        self.data = self.data[:,ind]
        self.taxis = self.taxis[ind]
        t0 = self.taxis[0]
        if reset_startime:
            self.taxis = self.taxis-t0
            self.start_time += timedelta(seconds=t0)
    
    def window_data_depth(self,bgmd,edmd,ismd=True):
        if ismd:
            ind = (self.mds>bgmd)&(self.mds<edmd)
        else:
            ind = (self.chans>bgmd)&(self.chans<edmd)
        self.data = self.data[ind,:]
        try:
            self.mds = self.mds[ind]
        except:
            print('cannot find mds field')
            pass
        try:
            self.chans = self.chans[ind]
        except:
            print('cannot find chans field')
            pass
    
    def lp_filter(self,corner_freq,order=2):
        dt = np.median(np.diff(self.taxis))
        for ichan in range(self.data.shape[0]):
            self.data[ichan,:] = gjsignal.lpfilter(self.data[ichan,:]
                ,dt,corner_freq,order=order)

    def bp_filter(self,lowf,highf,order=2):
        dt = np.median(np.diff(self.taxis))
        for ichan in range(self.data.shape[0]):
            self.data[ichan,:] = gjsignal.bpfilter(self.data[ichan,:]
                ,dt,lowf,highf,order=order)
    
    def down_sample(self,ds_R):
        dt = np.median(np.diff(self.taxis))
        self.lp_filter(1/dt/2/ds_R*0.8)
        self.data = self.data[:,::ds_R]
        self.taxis = self.taxis[::ds_R]

    def take_diff(self):
        data = np.diff(self.data,axis=1)
        self.taxis = self.taxis[1:]
        self.data = data
    
    def plot_simple_waterfall(self,downsample = [1,1]):
        extent = [0,self.data.shape[1],self.data.shape[0],0]
        plt.imshow(self.data[::downsample[0],::downsample[1]]
                ,cmap=plt.get_cmap('bwr'),aspect='auto',extent=extent)
    
    def get_extent(self,ischan=False,timescale='second',use_timestamp=False):
        xlim = np.array([self.taxis[0],self.taxis[-1]])
        if timescale == 'hour':
            xlim = xlim/3600
        if timescale == 'day':
            xlim = xlim/3600/24
        if ischan:
            ylim = [self.chans[-1],self.chans[0]]
        else:
            ylim = [self.mds[-1],self.mds[0]]
        if use_timestamp:
            edtime = self.start_time + timedelta(seconds=self.taxis[-1])
            xlim = [self.start_time,edtime]
            xlim = mdates.date2num(xlim)
        extent = [xlim[0],xlim[-1],ylim[0],ylim[-1]]
        return extent

    def plot_waterfall(self,ischan = False, cmap=plt.get_cmap('bwr')
            , timescale='second',use_timestamp=False
            ,timefmt = '%m/%d %H:%M:%S', is_shorten=False,downsample=[1,1]):
        extent = self.get_extent(ischan=ischan
            ,timescale=timescale,use_timestamp=use_timestamp)
        plt.imshow(self.data[::downsample[0],::downsample[1]]
                ,cmap = cmap, aspect='auto',extent=extent)
        if use_timestamp:
            if is_shorten:
                plt.subplot2grid((5,1),(0,0),rowspan=4)
            plt.gca().xaxis_date()
            date_format = mdates.DateFormatter(timefmt)
            plt.gca().xaxis.set_major_formatter(date_format)
            plt.xticks(rotation=45)
    
    def fill_gap_zeros(self):
        dt = np.median(np.diff(self.taxis))
        N = int(np.round((np.max(self.taxis)-np.min(self.taxis))/dt))+1
        new_taxis = np.linspace(np.min(self.taxis),np.max(self.taxis)+dt,N)
        new_data = np.zeros((self.data.shape[0],N))
        for i in range(self.data.shape[1]):
            ind = int(np.round(self.taxis[i]/dt))
            new_data[:,ind] = self.data[:,i]
        self.data = new_data
        self.taxis = new_taxis

    def fill_gap_interp(self):
        dt = np.median(np.diff(self.taxis))
        N = int(np.round((np.max(self.taxis)-np.min(self.taxis))/dt))+1
        new_taxis = np.linspace(np.min(self.taxis),np.max(self.taxis)+dt,N)
        new_data = np.zeros((self.data.shape[0],N))
        for i in range(self.data.shape[0]):
            new_data[i,:] = np.interp(new_taxis,self.taxis,self.data[i,:],left=0,right=0)
        self.data = new_data
        self.taxis = new_taxis
    
    def get_value_by_depth(self,depth):
        ind = np.argmin(np.abs(self.mds-depth))
        md = self.mds[ind]
        return md,self.data[ind,:]

    def get_value_by_timestr(self,timestr,fmt=None):
        if fmt is None:
            t = parse(timestr)
        else:
            t = datetime.strptime(timestr,fmt)
        dt = (t-self.start_time).total_seconds()
        ind = np.argmin(np.abs(self.taxis-dt))
        output_time = self.start_time + timedelta(seconds=self.taxis[ind])
        return output_time,self.data[:,ind]
    
    def saveh5(self,filename):
        with h5py.File(filename,'w') as f:
            # save main dataset
            dset = f.create_dataset('data',data=self.data)
            # save all the attributes to the main dataset
            dset.attrs['start time'] = self.start_time.strftime('%Y%m%d_%H%M%S.%f')
            for k in self.attrs.keys():
                dset.attrs[k] = self.attrs[k]
            # save all other ndarray and lists in the class
            for k in self.__dict__.keys():
                if k == 'data':
                    continue
                if k == 'start_time':
                    continue
                if k == 'attrs':
                    continue
                try:
                    f.create_dataset(k,data=self.__dict__[k])
                except:
                    print('cannot save variable: ',k)


    def loadh5(self,filename):
        f = h5py.File(filename,'r')
        # read start_time
        self.start_time = datetime.strptime(f['data'].attrs['start time'],'%Y%m%d_%H%M%S.%f')
        # read attributes
        self.attrs = {}
        for k in f['data'].attrs.keys():
            if k == 'start time':
                continue
            self.attrs[k] = f['data'].attrs[k]
        # read all other ndarrays
        for k in f.keys():
            setattr(self,k,np.array(f[k]))
        f.close()
    
    def right_merge(self,data):
        taxis = data.taxis + (data.start_time - self.start_time).total_seconds()
        self.taxis = np.concatenate((self.taxis,taxis))
        self.data = np.concatenate((self.data.T,data.data.T)).T

def merge_data2D(data_list):
    data_list = np.array(data_list)
    bgtime_lst = np.array([d.start_time for d in data_list])
    ind = np.argsort(bgtime_lst)
    bgtime_lst = bgtime_lst[ind]
    data_list = data_list[ind]

    t_samples = [d.data.shape[1] for d in data_list]
    N_samples = np.sum(t_samples)

    bgtime = data_list[0].start_time
    taxis_list = [d.taxis + (d.start_time-bgtime).total_seconds() for d in data_list]

    merge_data = copy.deepcopy(data_list[0])
    merge_data.data = np.concatenate([d.data.T for d in data_list]).T
    merge_data.taxis = np.concatenate(taxis_list)
    return merge_data

def load_h5(file):
    data = Data2D()
    data.loadh5(file)
    return data
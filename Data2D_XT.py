import gjsignal
import numpy as np
import matplotlib.pyplot as plt
from .BasicClass import BasicClass
from datetime import datetime,timedelta
from scipy.signal import medfilt2d
import matplotlib.dates as mdates

class Data2D(BasicClass):

    def __init__(self):
        self.version = '1.0'
        self.data = []
        self.chans = []
        self.taxis = []

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
    
    def lp_filter(self,corner_freq,order=2):
        dt = np.median(np.diff(self.taxis))
        for ichan in range(self.data.shape[0]):
            self.data[ichan,:] = gjsignal.lpfilter(self.data[ichan,:]
                ,dt,corner_freq,order=order)

    def take_diff(self):
        data = np.diff(self.data,axis=1)
        self.taxis = self.taxis[1:]
        self.data = data
    
    def plot_simple_waterfall(self):
        plt.imshow(self.data,cmap=plt.get_cmap('bwr'),aspect='auto')
    
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
            , timescale='second',use_timestamp=False):
        extent = self.get_extent(ischan=ischan
            ,timescale=timescale,use_timestamp=use_timestamp)
        plt.imshow(self.data,cmap = cmap, aspect='auto'
            ,extent=extent)
        if use_timestamp:
            plt.gca().xaxis_date()
            date_format = mdates.DateFormatter('%m/%d %H:%M')
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
        return self.data[ind,:]

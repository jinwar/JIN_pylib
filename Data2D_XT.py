import gjsignal
import numpy as np
import matplotlib.pyplot as plt
from BasicClass import BasicClass
from datetime import datetime
from scipy.signal import medfilt2d

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
    
    def get_extent(self,ischan=False,timescale='second'):
        xlim = np.array([self.taxis[0],self.taxis[-1]])
        if timescale == 'hour':
            xlim = xlim/3600
        if timescale == 'day':
            xlim = xlim/3600/24
        if ischan:
            ylim = [self.chans[-1],self.chans[0]]
        else:
            ylim = [self.mds[-1],self.mds[0]]
        extent = [xlim[0],xlim[-1],ylim[0],ylim[-1]]
        return extent

    def plot_waterfall(self,ischan = False, cmap=plt.get_cmap('bwr')
            , timescale='second'):
        plt.imshow(self.data,cmap = cmap, aspect='auto'
            ,extent=self.get_extent(ischan=ischan,timescale=timescale))


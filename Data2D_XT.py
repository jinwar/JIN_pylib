from dataclasses import dataclass

from tqdm import tqdm
from . import gjsignal
from .VizUtil import PrecisionDateFormatter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
from scipy.signal import medfilt2d
try:
    from scipy.signal import tukey
except:
    pass
try:
    from scipy.signal.windows import tukey
except:
    pass
from scipy.interpolate import interp1d
import matplotlib.dates as mdates
from copy import copy
import h5py
import copy

class Data2D():

    def __init__(self):
        self.data = None   # data, 2D array
        self.start_time = None  # starting time using datetime
        self.taxis = []  # time axis in second from start_time
        self.chans = [] # fiber channel number
        self.daxis = []  # fiber physical distance or location
        self.attrs = {'Python Class Version':'1.1'} # data attributes
        self.history = []

    def set_data(self,data):
        self.data = data
    
    # set mds linked to daxis to be compabible to old versions
    @property
    def mds(self):
        return self.daxis

    @mds.setter
    def mds(self, mds):
        self.daxis = mds
    
    def get_datetime64(self):
        timestamps = []
        for t in self.taxis:
            timestamps.append(np.datetime64(self.start_time + timedelta(seconds=t)))
        return np.array(timestamps)
    
    def get_mdates_taxis(self):
        timestamps = [self.start_time + timedelta(seconds=t) for t in self.taxis]
        return mdates.date2num(timestamps)
    
    def get_datetime(self):
        timestamps = [self.start_time + timedelta(seconds=t) for t in self.taxis]
        return timestamps
    
    def get_stalta(self,sta,lta):
        dt = np.median(np.diff(self.taxis))
        stalta_ratio = gjsignal.sta_lta_2d(self.data,dt,sta,lta)
        ind = (self.taxis>sta/2+lta)&(self.taxis<self.taxis[-1]-sta/2)
        timestamps = self.get_datetime64()
        stalta_ratio = stalta_ratio
        return timestamps,stalta_ratio
    
    def get_chunk(self, length, overlap, is_partial=False):
        bgtime = self.start_time + pd.to_timedelta(self.taxis[0], unit='s')
        edtime = self.start_time + pd.to_timedelta(self.taxis[-1], unit='s')
        length = timedelta(seconds= length)
        overlap = timedelta(seconds= overlap)
        chunk_list = []
        while bgtime < edtime:
            if edtime - bgtime < length:
                if is_partial:
                    chunk_list.append((bgtime,edtime))
                break
            else:
                chunk_list.append((bgtime,bgtime+length))
                bgtime += length - overlap
        return chunk_list
    
    def set_time_from_datetime(self, timestamps):
        """
        Sets the start time and time axis for the data from a list of datetime objects.

        Parameters:
        timestamps (list): A list of datetime objects representing the timestamps of the data.

        This function sets the start time as the first timestamp and calculates the time axis as the 
        total seconds from the start time for each timestamp.

        """
            # Check if timestamps are in np.datetime64 format and convert them to datetime
        if isinstance(timestamps[0], np.datetime64):
            timestamps = [pd.to_datetime(t).to_pydatetime() for t in timestamps]
        self.start_time = timestamps[0]
        self.taxis = np.array([(t-timestamps[0]).total_seconds()
             for t in timestamps])
    
    def apply_timeshift(self,ts):
        """
        Applies a time shift to the start time of the data.

        Parameters:
        ts (int): The time shift in hours to be applied.

        This function adds the time shift to the start time of the data.
        """
        self.start_time += timedelta(hours=ts)
    
    def print_info(self):
        print(f'Start time: {self.start_time}')
        print(f'taxis: {self.taxis[0]} - {self.taxis[-1]} seconds')
        dt = np.diff(self.taxis)
        print(f'time interval: min: {np.min(dt)}, max: {np.max(dt)}, median: {np.median(dt)}')
        print(f'daxis: {self.daxis[0]} - {self.daxis[-1]}')
        print(f'data dimension: {self.data.shape}')
        print(f'data size: {int(self.data.size*4/1e6)} MB')
        print(f'taxis dimension: {self.taxis.shape}')
        print(f'daxis dimension: {self.daxis.shape}')
        print(f'history: {self.history}')
    
    def cal_timestamp_from_taxis(self):
        """
        Calculates the timestamps from the time axis.

        This function calculates the timestamps by adding the time axis (in seconds) to the start time. 
        The calculated timestamps are then stored in the timestamps attribute of the object.
        """
        timestamps = [self.start_time + timedelta(seconds=t) 
            for t in self.taxis]
        self.timestamps = timestamps
    
    def set_mds(self,mds):
        self.mds = mds
    
    def _check_inputtime(self,t,t0):
        out_t = t
        if t is None:
            out_t = t0
        if isinstance(t, (datetime, pd.Timestamp)):
            out_t = (t-self.start_time).total_seconds()
        if isinstance(t,str):
            out_t = (pd.to_datetime(t)-self.start_time).total_seconds()
        return out_t

    def __add__(self,other):
        out_data = self.copy()
        out_data.data += other.data
        return out_data
    
    def reset_starttime(self):
        """
        Resets the start time of the data.

        This function adjusts the start time by adding the first value of the time axis (in seconds) 
        to the current start time. It also adjusts the time axis by subtracting the first value from 
        all its elements.
        """
        self.start_time += timedelta(seconds=self.taxis[0])
        self.taxis -= self.taxis[0]

    def select_time(self, bgtime, edtime, makecopy=False, reset_starttime=True):
        """
        Selects a time range from the data.

        Parameters:
        bgtime (float, datetime, string): The beginning time of the selection in seconds.
        edtime (float, datetime, string): The ending time of the selection in seconds.
        makecopy (bool, optional): If True, a copy of the data is made before the selection. Default is False.
        reset_starttime (bool, optional): If True, the start time is reset after the selection. Default is True.

        This function selects a time range from the data based on the provided beginning and ending times. 
        It checks the input times and selects the data within the time range. If makecopy is True, a copy 
        of the data is made before the selection. If reset_starttime is True, the start time is reset after 
        the selection by adding the first value of the new time axis and subtracting it from all its elements. 
        The data is then updated or a new data object is returned depending on the makecopy parameter.
        """
        bgt = self._check_inputtime(bgtime,self.taxis[0])
        edt = self._check_inputtime(edtime,self.taxis[-1])
        
        ind = (self.taxis>=bgt)&(self.taxis<edt)
        if makecopy:
            out_data = self.copy()
            out_data.taxis = self.taxis[ind]
            if reset_starttime:
                out_data.start_time += timedelta(seconds=out_data.taxis[0])
                out_data.taxis -= out_data.taxis[0]
            out_data.data = out_data.data[:,ind]
            return out_data
        else:
            self.taxis = self.taxis[ind]
            if reset_starttime:
                self.start_time += timedelta(seconds=self.taxis[0])
                self.taxis -= self.taxis[0]
            self.data = self.data[:,ind]
            return self

    def select_depth(self,bgdp,eddp,makecopy=False,ischan=False):
        
        if ischan:
            dists = self.chans
        else:
            dists = self.daxis
        bgdp = self._check_inputtime(bgdp,dists[0])
        eddp = self._check_inputtime(eddp,dists[-1])
        
        ind = (dists>=bgdp)&(dists<=eddp)
        if makecopy:
            out_data = self.copy()
            out_data.data = out_data.data[ind,:]
            try:
                out_data.daxis =out_data.daxis[ind]
            except: 
                pass
            try:
                out_data.chans =out_data.chans[ind]
            except: 
                pass

            return out_data
        else:
            self.data = self.data[ind,:]
            try:
                self.daxis =self.daxis[ind]
            except: 
                pass
            try:
                self.chans =self.chans[ind]
            except: 
                pass
            return self
    
    def copy(self):
        return copy.deepcopy(self)
    
    def set_chans(self,chans):
        self.chans = chans
    
    def median_filter(self,kernel_size=(5,3)):
        self.data = medfilt2d(self.data,kernel_size=kernel_size)
        self.history.append('median_filter(kernel_size={})'.format(str(kernel_size)))
    
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
    
    def lp_filter(self,corner_freq,order=2,axis=1,edge_taper=0.0):
        """
        Apply a low-pass filter to the data.

        Parameters:
        corner_freq (float): The cutoff frequency of the low-pass filter.
        order (int, optional): The order of the filter. Default is 2.
        axis (int, optional): The axis along which to apply the filter. Default is 1.
        edge_taper (float, optional): The seconds of the data to taper at the edges. Default is 0.0.

        Returns:
        self
        """
        if axis == 1:
            dt = np.median(np.diff(self.taxis))
        if axis == 0:
            dt = np.median(np.diff(self.mds))
        self.edge_taper(edge_taper=edge_taper,axis=axis)
        self.data = gjsignal.lpfilter(self.data,dt,corner_freq,order=order,axis=axis)
        self.history.append('lp_filter(corner_freq={},order={},axis={})'
                .format(corner_freq,order,axis))
        return self

    def hp_filter(self,corner_freq,order=2,axis=1,edge_taper=0.0):
        """
        Apply a high-pass filter to the data.

        Parameters:
        corner_freq (float): The cutoff frequency of the high-pass filter.
        order (int, optional): The order of the filter. Default is 2.
        axis (int, optional): The axis along which to apply the filter. Default is 1.
        edge_taper (float, optional): The seconds of the data to taper at the edges. Default is 0.0.

        Returns:
        Self. The data is modified in-place.
        """
        self.edge_taper(edge_taper=edge_taper,axis=axis)
        if axis == 1:
            dt = np.median(np.diff(self.taxis))
        if axis == 0:
            dt = np.median(np.diff(self.mds))
        self.data = gjsignal.hpfilter(self.data,dt,corner_freq,order=order,axis=axis)
        self.history.append('hp_filter(corner_freq={},order={},axis={})'
                .format(corner_freq,order,axis))
        return self
    
    def edge_taper(self,edge_taper=0,axis=1):
        if axis == 1:
            dt = np.median(np.diff(self.taxis))
            edge_N = edge_taper/dt
            edge_taper_ratio = edge_N/self.data.shape[1]
            self.data *= tukey(self.data.shape[1],edge_taper_ratio*2).reshape((1,-1))
        if axis == 0:
            dt = np.median(np.diff(self.daxis))
            edge_N = edge_taper/dt
            edge_taper_ratio = edge_N/self.data.shape[0]
            self.data *= tukey(self.data.shape[0],edge_taper_ratio*2).reshape((-1,1))
        self.history.append('edge_taper(axis={})'.format(axis))

    def bp_filter(self, lowf, highf, order=2, axis=1, edge_taper=0.0):
        """
        Apply a bandpass filter to the data.

        Parameters:
        lowf (float): The lower frequency limit of the bandpass filter.
        highf (float): The upper frequency limit of the bandpass filter.
        order (int, optional): The order of the filter. Default is 2.
        axis (int, optional): The axis along which to apply the filter. Default is 1.
        edge_taper (float, optional): The seconds of the data to taper at the edges. Default is 0.0.

        Returns:
        self. The data is modified in-place.
        """
        if axis == 1:
            dt = np.median(np.diff(self.taxis))
        if axis == 0:
            dt = np.median(np.diff(self.mds))
        self.edge_taper(edge_taper=edge_taper, axis=axis)
        self.data = gjsignal.bpfilter(self.data, dt, lowf, highf, order=order, axis=axis)
        self.history.append('bp_filter(lowf={},highf={},order={},axis={})'
                .format(lowf, highf, order, axis))
        return self
    
    def take_gradient(self,axis=1):
        data = np.gradient(self.data,axis=axis)
        if axis == 1:
            data /= np.median(np.diff(self.taxis))
        if axis == 0:
            data /= np.median(np.diff(self.mds))
        self.data = data
        self.history.append('take_gradient(axis={})'.format(axis))
    
    def down_sample(self,ds_R, **kwargs):
        """
        Downsamples the data by a given reduction factor.

        Parameters:
        ds_R (int, edge_taper = 0): The reduction factor by which to downsample the data.

        edge_taper (float, optional): The seconds of the data to taper at the edges. Default is 0.0.

        This method performs the following steps:
        1. Calculates the median time difference (dt) from the time axis.
        2. Applies a low-pass filter to the data with a cutoff frequency based on the downsampling rate.
        3. Downsamples the data and the time axis by the given reduction factor.
        4. Appends the downsampling operation to the history.

        Note:
        - The low-pass filter is applied to prevent aliasing during the downsampling process.
        """
        dt = np.median(np.diff(self.taxis))
        self.lp_filter(1/dt/2/ds_R*0.8, **kwargs)
        self.data = self.data[:,::ds_R]
        self.taxis = self.taxis[::ds_R]
        self.history.append('down_sample({}, {})'.format(ds_R, kwargs))


    def take_time_diff(self):
        data = np.diff(self.data,axis=1)
        data = data/np.diff(self.taxis).reshape((1,-1))
        data = np.hstack((np.zeros((data.shape[0],1)),data))
        self.data = data
        self.history.append('take_diff()')
    
    def apply_gauge_length(self,gauge_chan_num=1):
        strain_data = self.data[gauge_chan_num:,:]-self.data[:-gauge_chan_num,:]
        strain_data /= (self.daxis[gauge_chan_num:]-self.daxis[:-gauge_chan_num]).reshape((-1,1))
        self.data = strain_data
        self.daxis = (self.daxis[gauge_chan_num:]+self.daxis[:-gauge_chan_num])/2
        self.history.append(f'apply_gauge_length(gauge_chan_num={gauge_chan_num})')

    
    def cumsum(self,axis=1,makecopy = False):
        data = np.cumsum(self.data,axis=axis)
        if axis==1:
            ds = np.diff(self.taxis)
            ds = np.concatenate(([1],ds))
            data = data*ds.reshape((1,-1))
        if axis==0:
            ds = np.diff(self.mds)
            ds = np.concatenate(([1],ds))
            data = data*ds.reshape((-1,1))

        if makecopy:
            out_data = self.copy()
            out_data.data = data
            out_data.history.append(f'cumsum(axis={axis})')
            return out_data
        else:
            self.data = data
            self.history.append(f'cumsum(axis={axis})')
    
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
            bgtime = self.start_time + timedelta(seconds=self.taxis[0])
            xlim = [bgtime,edtime]
            xlim = mdates.date2num(xlim)
        extent = [xlim[0],xlim[-1],ylim[0],ylim[-1]]
        return extent

    def plot_waterfall(self,ischan = False, cmap=plt.get_cmap('bwr')
            , timescale='second',use_timestamp=False
            ,downsample=[1,1]
            ,xaxis_rotation=0
            ,xtickN = 4
            ,timefmt = '%y/%m/%d\n%H:%M:%S.{ms}' 
            ,timefmt_ms_precision = 1
            ,scale = None
            ,islog = False
            ,interpolation='antialiased'
            ):
        '''
        timescale options: 'second','hour','day'
        '''
        extent = self.get_extent(ischan=ischan
            ,timescale=timescale,use_timestamp=use_timestamp)
        plotdata = self.data[::downsample[0],::downsample[1]]
        if islog:
            plotdata = 10*np.log10(plotdata.copy())

        plt.imshow(plotdata ,cmap = cmap, aspect='auto',extent=extent,interpolation=interpolation)

        if use_timestamp:
            plt.gca().xaxis_date()
            date_format = PrecisionDateFormatter(timefmt
                ,precision=timefmt_ms_precision)
            plt.gca().xaxis.set_major_locator(plt.MaxNLocator(xtickN))
            plt.xticks(rotation=xaxis_rotation)
            plt.gca().xaxis.set_major_formatter(date_format)
        if scale is not None:
            if isinstance(scale,(int,float)):
                minval = np.nanpercentile(plotdata,scale/2)
                maxval = np.nanpercentile(plotdata,100-scale/2)
                plt.clim(minval,maxval)
            elif isinstance(scale,(list,tuple,np.ndarray)):
                plt.clim(scale)
        

    def plot_wiggle(self,scale=1,trace_step = 1,linewidth=1):
        # Extract the data, time axis, and distance axis from the seismic_data object
        data = self.data
        taxis = self.taxis
        daxis = self.daxis

        # Get the number of time and distance points
        nt = len(taxis)
        nd = len(daxis)

        # Loop over each trace
        for i in range(0,nd,trace_step):
            # Scale and shift the data for this trace
            trace = data[i, :] * scale + daxis[i]

            # Plot the trace as a line
            plt.plot(trace, taxis, color='k',linewidth=linewidth)

            # Fill between the trace and the zero line
            plt.fill_betweenx(taxis, daxis[i], trace, where=(trace>daxis[i]), color='r', linewidth=0.5*linewidth)
            plt.fill_betweenx(taxis, daxis[i], trace, where=(trace<daxis[i]), color='b', linewidth=0.5*linewidth)
        
        plt.gca().invert_yaxis()

    
    def fill_gap_zeros(self,fill_value=0,dt=None, is_average = False):
        """
        def fill_gap_zeros(self, fill_value=0, dt=None, is_average=False):
            Fills gaps in the data with zeros or a specified fixed value.

            Parameters:
            -----------
            fill_value : int or float, optional
                The value to fill the gaps with. Default is 0.
            dt : float, optional
                The time interval to use for filling gaps. If None, the median of the differences
                in the time axis (self.taxis) is used. Default is None.
            is_average : bool, optional
                If True, the gaps are filled with the average of the surrounding data points.
                If False, the gaps are filled with the specified fill_value. Default is False.

            Returns:
            --------
            None
                The method updates the object's data and time axis in place.

            Notes:
            ------
            - The method modifies the object's `data` and `taxis` attributes.
            - The method appends a description of the operation to the object's `history` attribute.
        """
        if dt is None:
            dt = np.median(np.diff(self.taxis))
        N = int(np.round((np.max(self.taxis)-np.min(self.taxis))/dt))+1
        new_taxis = np.linspace(np.min(self.taxis),np.max(self.taxis)+dt,N)
        new_data = np.zeros((self.data.shape[0],N))
        new_data[:,:] = fill_value
        if is_average:
            for i in tqdm(range(N)):
                ind = np.abs(self.taxis-new_taxis[i])<dt/2
                if np.sum(ind)==0:
                    continue
                new_data[:,i] = np.mean(self.data[:,ind],axis=1)
        else:
            for i in tqdm(range(self.data.shape[1])):
                ind = int(np.round(self.taxis[i]/dt))
                new_data[:,ind] = self.data[:,i]
        self.data = new_data
        self.taxis = new_taxis
        self.history.append(f'fill_gap_zeros(fill_value={fill_value},dt={dt})')

    def fill_gap_interp(self,dt=None):
        """
        def fill_gap_interp(self, dt=None):
            Interpolates to fill gaps in the time axis and updates the data accordingly.

            Parameters:
            dt (float, optional): The desired time interval for interpolation. If not provided, 
                                  the median of the differences in the existing time axis is used.

            Updates:
            self.data (numpy.ndarray): The data array with gaps filled by interpolation.
            self.taxis (numpy.ndarray): The time axis with gaps filled by interpolation.
            self.history (list): Appends a string indicating that fill_gap_interp was called with the specified dt.
        """
        if dt is None:
            dt = np.median(np.diff(self.taxis))
        N = int(np.round((np.max(self.taxis)-np.min(self.taxis))/dt))+1
        new_taxis = np.linspace(np.min(self.taxis),np.max(self.taxis),N)
        new_data = np.zeros((self.data.shape[0],N))
        print('Filling data gap by interpolation...')
        for i in tqdm(range(self.data.shape[0])):
            new_data[i,:] = np.interp(new_taxis,self.taxis,self.data[i,:],left=0,right=0)
        self.data = new_data
        self.taxis = new_taxis
        self.history.append(f'fill_gap_interp(dt={dt})')
    
    def remove_duplicate_time(self, tol=1e-2, re_interpolate = False):
        """
        def remove_duplicate_time(self, re_interpolate=True):
            Remove duplicate time points from the time axis and corresponding data.

            This method identifies and removes duplicate entries in the time axis (`taxis`).
            The corresponding data points in `data` are also removed to maintain alignment.
            Optionally, it can re-interpolate the data to remove abnormal close time points.

            Args:
                re_interpolate (bool): If True, re-interpolates the data to fill gaps created by the removal of duplicates. Default is True.
                !! need to make sure that the overlap section is less than 50% 

            Returns:
                None
        """
        # sort the time axis
        ind = np.argsort(self.taxis)
        self.taxis = self.taxis[ind]
        self.data = self.data[:,ind]

        dt = np.median(np.diff(self.taxis))
        idx = np.where(np.diff(self.taxis)>dt*tol)[0]
        self.taxis = self.taxis[idx]
        self.data = self.data[:, idx]
        if re_interpolate:
            self.fill_gap_zeros()
        self.history.append('remove_duplicate_time()')

    def interp_time(self,new_taxis):
        new_data = np.zeros((self.data.shape[0],len(new_taxis)))
        for i in range(self.data.shape[0]):
            new_data[i,:] = np.interp(new_taxis,self.taxis,self.data[i,:],left=0,right=0)
        self.data = new_data
        self.taxis = new_taxis
    
    def get_value_by_depth(self,depth):
        """
        def get_value_by_depth(self, depth):
            Get the data value at the specified depth.

            Parameters:
            depth (float): The depth value to query.

            Returns:
            md (float): The closest depth value found in the data.
            data (numpy.ndarray): The data values at the specified depth.
        """
        ind = np.argmin(np.abs(self.mds-depth))
        md = self.mds[ind]
        return md,self.data[ind,:]
    
    def get_spectrum_by_depth(self,depth):
        md,trace = self.get_value_by_depth(depth)
        dt = np.median(np.diff(self.taxis))
        f,spe = gjsignal.amp_spectrum(trace,dt)
        return f, spe
    
    def get_value_by_time(self,t):
        ind = np.argmin(np.abs(self.taxis-t))
        actual_t = self.taxis[ind]
        return actual_t,self.data[:,ind]

    def get_value_by_timestr(self,timestr,fmt=None):
        if fmt is None:
            t = pd.to_datetime(timestr)
        else:
            t = datetime.strptime(timestr,fmt)
        dt = (t-self.start_time).total_seconds()
        ind = np.argmin(np.abs(self.taxis-dt))
        output_time = self.start_time + timedelta(seconds=self.taxis[ind])
        return output_time,self.data[:,ind]
    
    def get_time_average_value(self,center_time,**kargs):
        '''
        Usage: 
            center_time = '2023-01-01 05:02:12' # can also be datetime
            get_time_average_value(center_time,seconds=5)
        '''
        center_time = self._check_inputtime(center_time,None)
        time_range = timedelta(**kargs).total_seconds()
        bgtime = center_time-time_range/2
        edtime = center_time+time_range/2
        ind = (self.taxis>=bgtime)&(self.taxis<=edtime)
        return np.nanmean(self.data[:,ind],axis=1)

    def get_depth_average_value(self,center_depth,depth_range):
        bgdist = center_depth-depth_range/2
        eddist = center_depth+depth_range/2
        ind = (self.daxis>=bgdist)&(self.daxis<=eddist)
        return np.nanmean(self.data[ind,:],axis=0)

    
    def make_audio_file(self,filename,bgdp=None,eddp=None):
        from scipy.io.wavfile import write
        DASdata = self.select_depth(bgdp,eddp,makecopy=True)
        rate = int(1/np.median(np.diff(DASdata.taxis)))
        data = np.mean(DASdata.data,axis=0)
        scaled = np.int16(data / np.max(np.abs(data)) * 32767)
        write(filename, rate, scaled)
        return scaled
    
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
        self.history = list(self.history)
    
    def right_merge(self,data):
        taxis = data.taxis + (data.start_time - self.start_time).total_seconds()
        self.taxis = np.concatenate((self.taxis,taxis))
        self.data = np.concatenate((self.data.T,data.data.T)).T
    
    def quick_populate(self,data,dt,dx):
        self.data = data
        self.taxis = np.arange(data.shape[1])*dt
        self.daxis = np.arange(data.shape[0])*dx

def merge_data2D(data_list, daxis = None):
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
    if daxis is None:
        merge_data.data = np.concatenate([d.data for d in data_list], axis=1)
    elif isinstance(daxis, (np.ndarray, list)):
        tmp = []
        for d in data_list:
            f = interp1d(d.daxis,d.data,axis=0, fill_value=np.nan, bounds_error=False)
            tmp.append(f(daxis))
        merge_data.data = np.concatenate(tmp,axis=1)
        merge_data.daxis = daxis
    elif isinstance(daxis, int):
        daxis = data_list[daxis].daxis
        tmp = []
        for d in data_list:
            f = interp1d(d.daxis,d.data,axis=0, fill_value=np.nan, bounds_error=False)
            tmp.append(f(daxis))
        merge_data.data = np.concatenate(tmp,axis=1)
        merge_data.daxis = daxis
    else:
        raise ValueError('daxis should be either ndarray or list or int or None')

    merge_data.taxis = np.concatenate(taxis_list)
    return merge_data

def load_h5(file):
    data = Data2D()
    data.loadh5(file)
    return data

def Patch_to_Data2D(dascore_data):
    data = dascore_data
    DASdata = Data2D()
    axis = data.dims.index('distance')
    if axis == 1:
        DASdata.data = data.data.T
    else:
        DASdata.data = data.data
    DASdata.daxis = data.coords['distance']
    DASdata.start_time = pd.to_datetime(data.coords['time'][0])
    DASdata.taxis = (data.coords['time']-data.coords['time'][0])/np.timedelta64(1,'s')
    return DASdata

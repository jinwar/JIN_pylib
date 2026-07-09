
import copy
import numpy as np
from scipy.interpolate import interp1d
from . import Data2D_XT
from tqdm import tqdm

class Data3D:

    def __init__(self):
        self.data = None   # data, 3D array, time, channel, frequency
        self.start_time = None  # starting time using datetime
        self.taxis = []  # time axis in second from start_time
        self.chans = [] # fiber channel number
        self.daxis = []  # fiber physical distance or location
        self.faxis = []  # fiber frequency axis
        self.attrs = {'Python Class Version':'1.1'} # data attributes
        self.history = []
    

    def get_FBA(self, freq_min, freq_max):
        """
        Get the frequency band averaged data.
        :param freq_min: minimum frequency
        :param freq_max: maximum frequency
        :return: Data2D_XT.Data2D object with averaged data
        """
        ind = (self.faxis >= freq_min) & (self.faxis <= freq_max)
        data = self.data[:, :, ind].mean(axis=2)
        data = data.T
        
        faxis = self.faxis[ind]
        
        result = Data2D_XT.Data2D()
        result.taxis = self.taxis
        result.daxis = self.daxis
        result.data = data
        result.faxis = faxis
        result.attrs = self.attrs.copy()
        result.history = self.history.copy()
        result.start_time = self.start_time
        result.history.append(f'Frequency band average from {freq_min} to {freq_max} Hz')
        
        return result
    
    def get_channel_spectrum(self):
        """
        Get the channel spectrum data.
        :return: Data2D_XT.Data2D object with channel spectrum data
        """
        data = self.data.mean(axis=0)

        result = Data2D_XT.Data2D()
        result.start_time = self.start_time
        result.taxis = self.faxis
        result.faxis = self.faxis
        result.daxis = self.daxis
        result.data = data
        result.attrs = self.attrs.copy()
        result.history = self.history.copy()
        result.history.append('Channel spectrum data')
        return result


Data3D.saveh5 = Data2D_XT.Data2D.saveh5
Data3D.loadh5 = Data2D_XT.Data2D.loadh5

def load_h5(filename):
    result = Data3D()
    result.loadh5(filename)
    return result

def get_FBA_from_files(files, freq_min, freq_max):
    """
    Get the frequency band averaged data from multiple files.
    :param files: list of file names
    :param freq_min: minimum frequency
    :param freq_max: maximum frequency
    :return: Data2D_XT.Data2D object with averaged data
    """
    FBAs = []
    for file in tqdm(files):
        data3D = load_h5(file)
        FBAdata = data3D.get_FBA(freq_min, freq_max)
        FBAs.append(FBAdata)

    FBAdata = Data2D_XT.merge_data2D(FBAs)

    return FBAdata

def merge_data3D(data_list, daxis=None):
    """
    Merge a list of Data3D objects in time (data axis order is
    time, channel, frequency), analogous to Data2D_XT.merge_data2D.
    :param data_list: list of Data3D objects
    :param daxis: None to require matching channel axes, an ndarray/list to
        interpolate every patch onto a common channel axis, or an int index
        into data_list (after time-sorting) to reuse that patch's channel axis
    :return: merged Data3D object
    """
    data_list = np.array(data_list)
    bgtime_lst = np.array([d.start_time for d in data_list])
    ind = np.argsort(bgtime_lst)
    data_list = data_list[ind]

    bgtime = data_list[0].start_time
    taxis_list = [d.taxis + (d.start_time-bgtime).total_seconds() for d in data_list]

    merge_data = copy.deepcopy(data_list[0])
    if daxis is None:
        merge_data.data = np.concatenate([d.data for d in data_list], axis=0)
    elif isinstance(daxis, (np.ndarray, list)):
        tmp = []
        for d in data_list:
            f = interp1d(d.daxis,d.data,axis=1, fill_value=np.nan, bounds_error=False)
            tmp.append(f(daxis))
        merge_data.data = np.concatenate(tmp,axis=0)
        merge_data.daxis = daxis
    elif isinstance(daxis, int):
        daxis = data_list[daxis].daxis
        tmp = []
        for d in data_list:
            f = interp1d(d.daxis,d.data,axis=1, fill_value=np.nan, bounds_error=False)
            tmp.append(f(daxis))
        merge_data.data = np.concatenate(tmp,axis=0)
        merge_data.daxis = daxis
    else:
        raise ValueError('daxis should be either ndarray or list or int or None')

    merge_data.taxis = np.concatenate(taxis_list)
    return merge_data


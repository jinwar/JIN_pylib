
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

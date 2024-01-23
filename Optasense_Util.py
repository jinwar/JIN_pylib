import h5py
import datetime
import pandas as pd
from glob import glob
import numpy as np
from tqdm import tqdm
from . import Data2D_XT
from dateutil import parser 


class DataFolder:

    def __init__(self, folder_path, file_extension='h5', verbose=False):
        self.folder_path = folder_path
        self.timestamp_bgind = -34
        self.timestamp_edind = -15
        self.filename_timefmt = '%Y%m%d_%H%M%S.%f'
    
    def set_filename_timestamp_loc(self,timestamp_bgind,timestamp_edind):
        self.timestamp_bgind = timestamp_bgind
        self.timestamp_edind = timestamp_edind
    
    def get_data_folder_info(self):
        files = glob(self.folder_path+'/*.h5')
        files = np.sort(files)
        timestamps = np.array([self.get_timestamp_from_filename(f) for f in files])
        return files,timestamps
    
    def get_timestamp_from_filename(self,filename):
        return datetime.datetime.strptime(filename[self.timestamp_bgind:self.timestamp_edind],self.filename_timefmt)
    
    def select(self,start_time=None,end_time=None):
        files,timestamps = self.get_data_folder_info()

        # Check if start_time or end_time is a string and convert to datetime
        if isinstance(start_time, str):
            start_time = parser.parse(start_time)
        if isinstance(end_time, str):
            end_time = parser.parse(end_time)

        if start_time is None:
            start_time = timestamps[0]
        if end_time is None:
            end_time = timestamps[-1]
        
        ind = np.where((timestamps>=start_time) & (timestamps<=end_time))[0]
        datalist = []
        for filename in tqdm(files[ind]):
            data, timestamps = read_h5(filename)
            DASdata = Data2D_XT.Data2D()
            DASdata.data = data
            DASdata.set_time_from_datetime(timestamps)
            DASdata.chans = np.arange(data.shape[0])
            datalist.append(DASdata)
        merge_data = Data2D_XT.merge_data2D(datalist)
        return merge_data



def read_h5(filename):
    """
    Reads a 2D matrix and a 1D datetime series from an HDF5 file and converts the time to Python datetime objects.

    Parameters:
    filename (str): Path to the HDF5 file.

    Returns:
    tuple: A tuple containing the following elements:
        - 2D numpy array: The sensor data.
        - 1D list: The timestamps corresponding to the sensor data, converted to Python datetime objects.
    """
    with h5py.File(filename, 'r') as file:
        # Read the 2D matrix ('RawData') and 1D series ('RawDataTime')
        raw_data = file['Acquisition/Raw[0]/RawData'][:]
        raw_data_time = file['Acquisition/Raw[0]/RawDataTime'][:]

    # Convert 'RawDataTime' to Python datetime objects
    raw_data_time = [datetime.datetime.fromtimestamp(ts / 1e6) for ts in raw_data_time]

    return raw_data, raw_data_time
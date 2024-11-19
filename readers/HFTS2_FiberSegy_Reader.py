# Filename: file_to_DASdata.py
# Author: Samantha Burton
# Created: 2024-11-18
# Description: This is a reader from .segy files into Data2D_XT.Data2D objects


import numpy as np
from glob import glob
import pandas as pd
import datetime
import segyio
from JIN_pylib import Data2D_XT
from JIN_pylib import Spool



def get_file_starttime(filename):

    filename = filename.split('_')
    dateFormat = '%Y%m%d'
    filetime = filename[2]
    timeFormat = '%H%M%S.%f+0000.segy'
    date = datetime.datetime.strptime(filename[1] + filetime, dateFormat+timeFormat)
    return date

def reader(filename):
    with segyio.open(filename, 'r', strict=False) as segyfile:
        
        data = segyfile.trace.raw[:]                 # this is a 2D array where rows are traces (time series)
        taxis = np.arange(data.shape[1])*segyio.tools.dt(segyfile)/1e6            # this is an array of times starting at 0
        daxis = segyfile.attributes(41)[:] # 41 is ReceiverGroupElevation        
        
        start_time = get_file_starttime(filename)
        DASdata = Data2D_XT.Data2D()
        DASdata.data = data
        DASdata.taxis = taxis
        DASdata.daxis = daxis
        DASdata.start_time = start_time
        DASdata.take_gradient()

    return DASdata



def create_spool(datafolderpath, extension='.segy'):
    
    def make_data_catalog(folderpath, length=datetime.timedelta(seconds=0.9999e+01)):
        '''Makes a catalog dataframe of filenames, start times, and end times'''
        files = glob(folderpath + '/*.segy')

        fileDF = pd.DataFrame(columns=['file', 'start_time', 'end_time'])
        i = 0
        for file in files:
            fileTitle = file.split('\\')[-1]
            start = get_file_starttime(fileTitle)
            end = start+length
            fileDF.loc[i] = [file, start, end]
            i += 1
        return fileDF
    
    dataFileCatalog = make_data_catalog(datafolderpath, length=datetime.timedelta(seconds=0.9999e+01))
    
    sp = Spool.spool(df=dataFileCatalog, reader=reader)
    
    return sp
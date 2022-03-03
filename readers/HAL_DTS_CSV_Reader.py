from datetime import datetime
from glob import glob
import pandas as pd
from JIN_pylib import Data2D_XT

def read_HAL_DTS_CSV(file):
    df = pd.read_csv(file,skiprows=9,error_bad_lines=False)

    data = df.iloc[:,1:].values
    mds = df.iloc[:,0].values

    with open(file,'r') as f:
        l = f.readline()
        l = l.replace('\n','')
        l = l.split(',')


    timestamps = [datetime.strptime(s,'%Y/%m/%d @ %I:%M:%S %p') for s in l[1:]]
    
    DTSdata = Data2D_XT.Data2D()
    DTSdata.data = data
    DTSdata.mds = mds
    DTSdata.set_time_from_datetime(timestamps)
    return DTSdata


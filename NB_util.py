import pandas as pd
from .Data2D_XT import Data2D
import numpy as np
from datetime import datetime



def read_NB_csv(filename):
    df = pd.read_csv(filename)

    timestrs = list(df.columns[1:])

    timestamps = [datetime.strptime(t,'%m/%d/%Y %H:%M:%S.%f') for t in timestrs]
    timestamps = np.array(timestamps)
    mds = df['Length'].values
    data = df.iloc[:,1:].values

    chans = np.arange(data.shape[0])

    Ddata = Data2D()
    Ddata.set_data(data)
    Ddata.set_mds(mds)
    Ddata.set_chans(chans)
    Ddata.set_time_from_datetime(timestamps)
    Ddata.timestamps = timestamps
    
    return Ddata
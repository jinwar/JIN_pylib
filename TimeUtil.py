import numpy as np
from datetime import datetime


def StrArray_to_timestamps(ts_strs,fmt):
    timestamps = [datetime.strptime(s,fmt) for s in ts_strs]
    timestamps = np.array(timestamps)
    return timestamps

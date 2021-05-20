
from obspy import read, Trace, Stream, UTCDateTime
from obspy.core import AttribDict
from obspy.io.segy.segy import SEGYTraceHeader, SEGYBinaryFileHeader
from obspy.io.segy.core import _read_segy
import sys
import numpy as np
from . import gjsignal

def savesegy(DASdata,filename):
    stream = Stream()
    for i in range(DASdata.data.shape[0]):
        data = DASdata.data[i,:]
        data = np.require(data, dtype=np.float32)
        DASdt = DASdata.attrs['dt']
        if DASdt<0.001:
            dt = 0.001
            data = gjsignal.lpfilter(data,DASdt,500)
            oldtaxis = np.arange(len(data))*DASdt
            newtaxis = np.arange(0,oldtaxis[-1],dt)
            data = np.interp(newtaxis,oldtaxis,data)
        else:
            dt = DASdt

        trace = Trace(data=data)

        # Attributes in trace.stats will overwrite everything in
        # trace.stats.segy.trace_header
        trace.stats.delta = dt
        # SEGY does not support microsecond precision! Any microseconds will
        # be discarded.
        trace.stats.starttime = UTCDateTime(DASdata.start_time)

        # If you want to set some additional attributes in the trace header,
        # add one and only set the attributes you want to be set. Otherwise the
        # header will be created for you with default values.
        if not hasattr(trace.stats, 'segy.trace_header'):
            trace.stats.segy = {}
        trace.stats.segy.trace_header = SEGYTraceHeader()
        trace.stats.segy.trace_header.trace_sequence_number_within_line = i + 1
        trace.stats.segy.trace_header.receiver_group_elevation = 444

        # Add trace to stream
        stream.append(trace)

    # A SEGY file has file wide headers. This can be attached to the stream
    # object.  If these are not set, they will be autocreated with default
    # values.
    stream.stats = AttribDict()
    stream.stats.textual_file_header = 'Textual Header!'
    stream.stats.binary_file_header = SEGYBinaryFileHeader()
    stream.stats.binary_file_header.trace_sorting_code = 5

    print(stream)
    stream.write(filename, format='SEGY', data_encoding=1,
            byteorder=sys.byteorder)
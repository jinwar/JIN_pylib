# Filename: HFTS2_FiberSegy_Reader.py
# Reader for the HFTS2 fiber-microseismic OptaSense SEG-Y acquisition (see
# `Microseismic Interferometry/DATASET.md` in the HFTS2 student-folder repo that
# consumes this reader): 10-second SEG-Y segments, 2000 Hz, one file per segment.
# File naming: segy_YYYYMMDD_HHMMSS.fff+0000.segy (UTC segment start time).

import os
from datetime import datetime, timedelta

import numpy as np
import segyio

from .. import Data2D_XT
from . import reader_utils

SEGMENT_LENGTH = timedelta(seconds=10)


def get_file_starttime(filename):
    """Parse the UTC segment start time from the filename alone (no file IO)."""
    stem = os.path.basename(filename)
    _, datestr, timestr = stem.split('_')
    return datetime.strptime(datestr + timestr, '%Y%m%d%H%M%S.%f+0000.segy')


def get_time_range(filename):
    """Cheap, metadata-only time range -- derived from the filename, not the file contents."""
    start_time = get_file_starttime(filename)
    return start_time, start_time + SEGMENT_LENGTH


def reader(filename):
    """Whole-file read (each segment is only 10 s) into a Data2D_XT.Data2D.

    daxis is each channel's receiver elevation (ft), scaled by SourceGroupScalar --
    matches the convention used in `100 Microseismic Event Map View.ipynb` to put
    channel geometry in the same coordinate system as the event catalog. It is a
    physical-location proxy, not distance-along-fiber; no channel-depth table ships
    with these SEG-Ys (see DATASET.md).
    """
    with segyio.open(filename, 'r', ignore_geometry=True, endian='big') as segyfile:
        data = segyfile.trace.raw[:]
        taxis = np.arange(data.shape[1]) * segyio.tools.dt(segyfile) / 1e6
        elev = np.asarray(segyfile.attributes(segyio.TraceField.ReceiverGroupElevation)[:], dtype=float)
        scalar = np.asarray(segyfile.attributes(segyio.TraceField.SourceGroupScalar)[:], dtype=float)
        start_time = get_file_starttime(filename)

    scale = np.where(scalar < 0, 1.0 / (-scalar), np.where(scalar > 0, scalar, 1.0))

    DASdata = Data2D_XT.Data2D()
    DASdata.data = data
    DASdata.taxis = taxis
    DASdata.daxis = elev * scale
    DASdata.chans = np.arange(data.shape[0])
    DASdata.start_time = start_time
    DASdata.attrs['units'] = 'strain (radians, relative to 1550.12 nm)'
    return DASdata


def create_spool(datapath, extension='.segy', search_subdirs=False):
    return reader_utils.create_spool_common(
        datapath, get_time_range, reader,
        search_pattern='*' + extension,
        search_subdirs=search_subdirs,
        support_partial_reading=False,
    )

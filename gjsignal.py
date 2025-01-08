from scipy.signal import butter, lfilter, filtfilt, freqz
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
from scipy.stats.stats import pearsonr
import sys


def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_lppass(freqcut, fs, order=2):
    nyq = 0.5 * fs
    w = freqcut / nyq
    b, a = butter(order, w, btype='low')
    return b, a


def butter_hppass(freqcut, fs, order=2):
    nyq = 0.5 * fs
    w = freqcut / nyq
    b, a = butter(order, w, btype='high')
    return b, a


def bpfilter(data, dt, lowcut, highcut, order=2,axis=-1):
    ''' bpfilter(data, dt, lowcut, highcut, order=2)
	'''
    fs = 1 / dt
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data,axis=axis)
    return y


def lpfilter(data, dt, freqcut, order=2, plotSpectrum=False,axis=-1):
    fs = 1 / dt
    b, a = butter_lppass(freqcut, fs, order=order)
    y = filtfilt(b, a, data,axis=axis)
    return y


def hpfilter(data, dt, freqcut, order=2,axis=-1):
    fs = 1 / dt
    b, a = butter_hppass(freqcut, fs, order=order)
    y = filtfilt(b, a, data,axis=axis)
    return y


def amp_spectrum(data, dt, norm=None):
    data = data.flatten()
    N = len(data)
    freqs = np.fft.fftfreq(N, dt)
    asp = np.abs(np.fft.fft(data,norm=norm))
    idx = np.argsort(freqs)
    freqs = freqs[idx]
    asp = asp[idx]
    ind = freqs >= 0
    return freqs[ind], asp[ind]


def samediff(data):
    y = np.diff(data)
    y = np.append(y, y[-1])
    return y


def fillnan(data):
    ind = ~np.isnan(data)
    x = np.array(range(len(data)))
    y = np.interp(x, x[ind], data[ind]);
    return y


def timediff(ts1, ts2):
    tdiff = ts1 - ts2
    if tdiff.days < 0:
        return -(-tdiff).seconds
    else:
        return tdiff.seconds


def diffplot(data1, data2, crange=(-1, 1)):
    plt.figure(figsize=(10, 8))
    plt.subplot(1, 3, 1)
    plt.imshow(data1, aspect='auto')
    plt.clim(crange)
    plt.subplot(1, 3, 2)
    plt.imshow(data2, aspect='auto')
    plt.clim(crange)
    plt.subplot(1, 3, 3)
    plt.imshow(data1 - data2, aspect='auto')
    plt.clim(crange)
    plt.show()


def get_interp_mat(anchor_N, N, kind='quadratic'):
    x = np.arange(N)
    anchor_x = np.linspace(x[0], x[-1], anchor_N)
    interp_mat = np.zeros((N, anchor_N))
    for i in range(anchor_N):
        test_y = np.zeros(anchor_N, )
        test_y[i] = 1
        col = interp1d(anchor_x, test_y, kind=kind)(x)
        interp_mat[:, i] = col
    return interp_mat


def get_interp_mat_anchorx(x, anchor_x, kind='quadratic'):
    anchor_N = len(anchor_x)
    N = len(x)
    interp_mat = np.zeros((N, anchor_N))
    for i in range(anchor_N):
        test_y = np.zeros(anchor_N, )
        test_y[i] = 1
        col = interp1d(anchor_x, test_y, kind=kind)(x)
        interp_mat[:, i] = col
    return interp_mat


def get_smooth_curve(x0, anchor_x, data, kind='quadratic', errstd=3, iterN=2):
    iterdata = data.copy()
    iterx = x0.copy()
    for i in range(iterN + 1):
        interp_mat = get_interp_mat_anchorx(iterx, anchor_x, kind=kind)
        x = np.linalg.lstsq(interp_mat, iterdata)[0]
        err = np.abs(iterdata - np.dot(interp_mat, x)) ** 2
        goodind = err < np.std(err) * errstd
        iterdata = iterdata[goodind]
        iterx = iterx[goodind]
    interp_mat = get_interp_mat_anchorx(x0, anchor_x, kind=kind)
    smdata = np.dot(interp_mat, x)
    return smdata, x


def rms(a,axis=None):
    return np.sqrt(np.mean((a) ** 2,axis=axis))

def matdatenum_to_pydatetime(matlab_datenum):
    python_datetime = datetime.fromordinal(int(matlab_datenum)) + timedelta(days=matlab_datenum % 1) - timedelta(
        days=366)
    return python_datetime


def rfft_xcorr(x, y):
    M = len(x) + len(y) - 1
    N = 2 ** int(np.ceil(np.log2(M)))
    X = np.fft.rfft(x, N)
    Y = np.fft.rfft(y, N)
    cxy = np.fft.irfft(X * np.conj(Y))
    cxy = np.hstack((cxy[:len(x)], cxy[N - len(y) + 1:]))
    return cxy


def xcor_match(a, b):
    x = a.copy()
    ref = b.copy()
    x -= np.mean(x)
    ref -= np.mean(ref)
    r = pearsonr(x, ref)[0]
    if abs(r) < 0.3:
        return np.nan
    cxy = rfft_xcorr(x, ref)
    index = np.argmax(cxy)
    if index < len(x):
        return index
    else:  # negative lag
        return index - len(cxy)


def timeshift_xcor(data1, data2, winsize, step=1, lowf=1 / 10):
    N = len(data1)
    ori_x = np.arange(N)
    cx = np.arange(np.int(winsize / 2), np.int(N - winsize / 2 - 1), step)
    cy = np.zeros(len(cx))
    for i in range(len(cx)):
        winbg = cx[i] - np.int(winsize / 2)
        wined = cx[i] + np.int(winsize / 2)
        cy[i] = xcor_match(data1[winbg:wined], data2[winbg:wined])
    ind = ~np.isnan(cy)
    ts = np.interp(ori_x, cx[ind], cy[ind])
    ts = lpfilter(ts, 1, lowf)
    tar_x = ori_x + ts
    shift_data1 = np.interp(tar_x, ori_x, data1)
    return ts, shift_data1


def running_average(data, N):
    outdata = np.convolve(data, np.ones((N,)) / N, mode='same')
    halfN = int(N / 2)
    for i in range(halfN + 1):
        outdata[i] = np.mean(data[:i + halfN])
    for i in range(1, halfN + 1):
        outdata[-i] = np.mean(data[-i - halfN:])
    return outdata

def down_sample(data, ratio):
    '''
    Downsamples the input data by the given ratio after applying a low-pass filter.
    def down_sample(data, ratio):

    Parameters:
    data (array-like): The input data to be downsampled.
    ratio (int): The downsampling ratio. Must be a positive integer.

    Returns:
    array-like: The downsampled data.
    '''
    lpdata = lpfilter(data, 1, 1 / ratio / 2)
    return lpdata[::ratio]



def degC_to_degK(C):
    return C - 273.15


def degC_to_degF(C):
    return C * 9 / 5 + 32


def degF_to_degC(C):
    return (C - 32) * 5 / 9


def degK_to_degF(K):
    return K * 9 / 5 - 459.67


def degF_to_degK(F):
    return (F + 459.67) * 5 / 9


def print_progress(n):
    sys.stdout.write("\r" + str(n))
    sys.stdout.flush()


def phase_wrap(data):
    phase = np.angle(np.exp(1j * data))
    return phase

def dist_3D(x, y, z, x1, y1, z1):
    r = ((x - x1) ** 2 + (y - y1) ** 2 + (z - z1) ** 2) ** 0.5
    return r


def dummy_fun():
    print('hello world')


def robust_polyfit(x,y,order=1,errtol=2):
    para = np.polyfit(x,y,order)
    errs = np.abs(y-np.polyval(para,x))
    err_threshold = np.std(errs)*errtol
    goodind = errs<err_threshold
    para = np.polyfit(x[goodind],y[goodind],order)
    return para

from dateutil.parser import parse
from datetime import timedelta
def fetch_timestamp_fast(timestamp_strs, downsampling=100):
    x = np.arange(len(timestamp_strs))
    x_sparse = list(map(int,np.round(np.linspace(0,x[-1],len(x)//downsampling))))
    ts_sparse = np.array(list(map(parse,timestamp_strs[x_sparse])))
    t_sparse = np.array([(t-ts_sparse[0]).total_seconds() for t in ts_sparse])
    t = np.interp(x,x_sparse,t_sparse)
    ts = [(ts_sparse[0]+timedelta(seconds=dt)) for dt in t]
    return ts,t

def multi_legend(lns,loc='best'):
    labs = [l.get_label() for l in lns]
    plt.legend(lns,labs,loc=loc)


def datetime_interp(timex,timex0,y0):
    """
    def datetime_interp(timex,timex0,y0):
        Interpolate data to a new time axis.
        Parameters:
        
        timex (list): List of datetime objects for the new time axis.
        timex0 (list): List of datetime objects for the original time axis.
        y0 (list): List of data values corresponding to the original time axis.
        
        Returns:
        list: List of data values interpolated to the new time axis.
    """
    x = [(t-timex0[0]).total_seconds() for t in timex]
    x0 = [(t-timex0[0]).total_seconds() for t in timex0]
    return np.interp(x,x0,y0)
    
def running_average(data,N):
    return np.convolve(data,np.ones((N,))/N,mode='same')


def find_lpf_edge_effect(corf, dt, threshold=1e-6, isfigure=False):
    """
    def find_lpf_edge_effect(corf, dt, threshold=1e-6):
    Calculate the edge effect of a low-pass filter.
    This function generates a test signal, applies a low-pass filter to it, 
    and determines the time difference between the center of the test signal 
    and the first point where the filtered signal exceeds a given threshold.
    Parameters:
    corf (float): The cutoff frequency of the low-pass filter.
    dt (float): The time step of the signal.
    threshold (float, optional): The threshold value to determine the edge effect. Default is 1e-6.
    Returns:
    float: The time difference between the center of the test signal and the first point 
           where the filtered signal exceeds the threshold.
    """

    N = int(1/corf/dt*20)
    test_data = np.zeros(N)
    test_data[N//2] = 1
    taxis = np.arange(N)*dt
    taxis -= taxis[N//2]
    ori_testdata = test_data.copy()

    test_data = lpfilter(test_data, dt, corf)

    # find the first point with value larger than threshold
    idx = np.where(np.abs(test_data)>threshold*np.max(test_data))[0][0]

    if isfigure:
        plt.plot(taxis,ori_testdata)
        plt.plot(taxis,test_data)
        plt.ylim(np.min(test_data)*5,np.max(test_data)*2)
        plt.axvline(taxis[idx],color='r', linestyle='--')

    return taxis[N//2]-taxis[idx]


def sta_lta_1d(timeseries, dt, STA, LTA):
    """
    Implement STA/LTA method for earthquake detection using RMS and convolution.
    def sta_lta_1d(timeseries, dt, STA, LTA):

    Parameters:
    timeseries (numpy array): 1D array of time series data
    dt (float): Time sample interval in seconds
    STA (float): Length of Short-Term Average window in seconds
    LTA (float): Length of Long-Term Average window in seconds

    Returns:
    numpy array: 1D array of STA/LTA ratio
    """
    # Convert LTA and STA from seconds to number of samples
    LTA_samples = int(LTA / dt)
    STA_samples = int(STA / dt)

    # Calculate the squared values of the timeseries for RMS calculation
    squared_timeseries = timeseries ** 2

    # Define windows for STA and LTA
    STA_window = np.ones(STA_samples) / STA_samples
    LTA_window = np.ones(LTA_samples) / LTA_samples

    # Compute RMS for STA and LTA using convolution
    sta_rms = np.sqrt(np.convolve(squared_timeseries, STA_window, mode='same'))
    lta_rms = np.sqrt(np.convolve(squared_timeseries, LTA_window, mode='same'))

    # Shift STA ahead of LTA by LTA_samples to ensure STA window is ahead
    lta_rms_shifted = np.roll(lta_rms, int((LTA_samples + STA_samples) / 2))

    # Avoid division by zero and calculate STA/LTA ratio

    sta_lta_ratio = sta_rms/lta_rms_shifted
    sta_lta_ratio[:LTA_samples+STA_samples//2] = 0  # Set initial values to zero to avoid edge effects
    sta_lta_ratio[-STA_samples//2:] = 0  # Set final values to zero to avoid edge effects
    
    return sta_lta_ratio

def sta_lta_2d(timeseries, dt, STA, LTA):
    """
    Implement STA/LTA method for earthquake detection using RMS and convolution.
    def sta_lta_2d(timeseries, dt, STA, LTA):

    Parameters:
    timeseries (numpy array): 2D array of time series data (channels x samples)
    dt (float): Time sample interval in seconds
    STA (float): Length of Short-Term Average window in seconds
    LTA (float): Length of Long-Term Average window in seconds

    Returns:
    numpy array: 1D array of averaged STA/LTA ratio across all channels
    """
    # Initialize an array to store the STA/LTA ratios for each channel
    sta_lta_ratios = np.zeros(timeseries.shape)

    for i in range(timeseries.shape[0]):
        sta_lta_ratios[i] = sta_lta_1d(timeseries[i], dt, STA, LTA)

    # Average the STA/LTA ratios across all channels
    averaged_sta_lta_ratio = np.mean(sta_lta_ratios, axis=0)

    return averaged_sta_lta_ratio


def interp_to_matrix(x0,x,kind='cubic'):
    ''' Function convert interpolation from grid location x0 to data location x to an interpolation matrix
    so that interp1d(x0,y0)(x) is equivalent to A.dot(y0)
    FYI: this is a very slow algorithm for the coding easiness.
    usage: A = interp_to_matrix(x0,x)
    input:
        x0: interpolation control point location, (n,) array
        x: data point location, (m,) array
    output:
        matrix A with shape (m,n)
    written by Ge Jin, gjin@mines.edu, 09/2019
    '''
    A = np.zeros((len(x),len(x0)))
    for i in range(len(x0)):
        y0 = np.zeros(len(x0))
        y0[i] = 1
        f = interp1d(x0,y0,kind=kind,bounds_error=False,fill_value='extrapolation')
        A[:,i] = f(x)

    return A

def control_point_curvefit(xc, x0, y0, kind = 'cubic'):
    ''' Function to fit a curve to the control points
    usage: y = control_point_curvefit(xc, x0, y0, smooth=1e-4, kind = 'cubic')
    input:
        xc: location of control points
        x0: location of data points
        y0: data points
        smooth: smoothing factor for the curve fitting
        kind: kind of interpolation
    output:
        y: fitted curve
    written by Ge Jin,
    '''
    A = interp_to_matrix(xc,x0,kind=kind)
    yc = np.linalg.lstsq(A,y0)[0]
    f = interp1d(xc,yc,kind=kind,bounds_error=False,fill_value=(yc[0],yc[-1]))
    return f
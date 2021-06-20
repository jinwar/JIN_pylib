import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from . import gjsignal, Data2D_XT

def Data2D_fft2(DASdata):
    fftdata = np.fft.fft2(DASdata.data)
    fftdata = np.fft.fftshift(fftdata)

    dt = np.median(np.diff(DASdata.taxis))
    dx = np.median(np.diff(DASdata.mds))
    faxis = np.fft.fftfreq(DASdata.data.shape[1],dt)
    kaxis = np.fft.fftfreq(DASdata.data.shape[0],dx)
    faxis = np.fft.fftshift(faxis)
    kaxis = np.fft.fftshift(kaxis)
    DASdata.faxis = faxis
    DASdata.kaxis = kaxis
    DASdata.fftdata = fftdata
    DASdata.history.append('fft2 transform')
    return DASdata

def Data2D_plotfk(DASdata):
    faxis = DASdata.faxis
    kaxis = DASdata.kaxis
    plt.imshow(np.abs(DASdata.fftdata),aspect='auto'
            ,extent=[faxis[0],faxis[-1],kaxis[0],kaxis[-1]])
    plt.xlabel('Frequency')
    plt.ylabel('Wave number')

def Data2D_fkfilter_applymask(DASdata,mask):
    DASdata.fftdata *= mask
    DASdata.data = np.real(np.fft.ifft2(np.fft.ifftshift(DASdata.fftdata)))
    return DASdata

def Data2D_fkfilter_maskgen(DASdata,vmin,vmax,filter_std):
    mask = np.ones(DASdata.fftdata.shape)
    fi,ki = np.meshgrid(DASdata.faxis,DASdata.kaxis)
    v_mat = fi/ki
    ind = (v_mat>vmin)&(v_mat<vmax)
    mask[ind] = 0
    # smooth the boundary
    mask = gaussian_filter(mask,filter_std)
    return mask

def Data2D_fkfilter_velocity(DASdata,vmin,vmax,filter_std=0):
    mask = Data2D_fkfilter_maskgen(DASdata,vmin,vmax,filter_std)
    DASdata = Data2D_fkfilter_applymask(DASdata,mask)
    return DASdata

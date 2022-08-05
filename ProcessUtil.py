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
    DASdata.history.append('applied custom fk filter')
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
    DASdata.history.pop()
    DASdata.history.append(
        'applied velocity filter from {} to {}, with smoothing kernel: {}'
        .format(vmin,vmax,filter_std))
    return DASdata

def disp_analysis(DASdata,v_array,f_array,upward=True):
    dt = np.median(np.diff(DASdata.taxis))
    dx = np.median(np.diff(DASdata.mds))
    data = DASdata.data

    dists = np.arange(data.shape[0])*dx

    fftdata = np.fft.fft(data,axis=1)
    freqs = np.fft.fftfreq(fftdata.shape[1],dt)

    semb_mat = np.zeros((len(v_array),len(f_array)))

    actual_f_array = np.zeros_like(f_array)

    for iv in range(len(v_array)):
        for ifreq in range(len(f_array)):
            f_ind = np.argmin(np.abs(f_array[ifreq]-freqs))
            actual_f_array[ifreq] = freqs[f_ind]
            if upward:
                phase_shift = np.exp(-1j*2*np.pi*freqs[f_ind]/v_array[iv]*dists)
            else:
                phase_shift = np.exp(1j*2*np.pi*freqs[f_ind]/v_array[iv]*dists)
            shifted_data = fftdata[:,f_ind]*phase_shift
            semb_mat[iv,ifreq] = np.abs(np.sum(shifted_data))
    
    result = {'semb_mat':semb_mat
            ,'norm_semb_mat':semb_mat/np.max(semb_mat,axis=0)
            ,'v_array':v_array
            , 'input_f_array':f_array
            , 'f_array': actual_f_array}

    return result

def plot_disp_result(result,plot_data = 'norm_semb_mat'):
    data = result[plot_data]
    f = result['f_array']
    v = result['v_array']
    extent = [f[0],f[-1],v[-1],v[0]]
    plt.imshow(data,aspect='auto',extent=extent,cmap='seismic')



import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interpn,interp1d

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
            ,extent=[faxis[0],faxis[-1],kaxis[-1],kaxis[0]])
    plt.xlabel('Frequency')
    plt.ylabel('Wave number')

def Data2D_fkfilter_applymask(DASdata,mask):
    if not hasattr(DASdata,'fftdata'):
        DASdata = Data2D_fft2(DASdata)
    DASdata.fftdata *= mask
    DASdata.data = np.real(np.fft.ifft2(np.fft.ifftshift(DASdata.fftdata)))
    DASdata.history.append('applied custom fk filter')
    return DASdata

def Data2D_fkfilter_maskgen(DASdata,vmin,vmax,filter_std,accept=False):
    if not hasattr(DASdata,'fftdata'):
        DASdata = Data2D_fft2(DASdata)
    fi,ki = np.meshgrid(DASdata.faxis,DASdata.kaxis)
    v_mat = fi/ki
    ind = (v_mat>vmin)&(v_mat<vmax)
    if accept:
        mask = np.zeros(DASdata.fftdata.shape)
        mask[ind] = 1
    else:
        mask = np.ones(DASdata.fftdata.shape)
        mask[ind] = 0
    # smooth the boundary
    mask = gaussian_filter(mask,filter_std)
    return mask

def Data2D_fkfilter_velocity(DASdata,vmin,vmax,filter_std=0,direction=None):
    if not hasattr(DASdata,'fftdata'):
        DASdata = Data2D_fft2(DASdata)
    v_array = np.array([vmin,vmax])
    if direction=='up':
        v_array = -np.abs(v_array)
    if direction=='down':
        v_array = np.abs(v_array)
    mask = Data2D_fkfilter_maskgen(DASdata,min(v_array),max(v_array),filter_std)
    DASdata = Data2D_fkfilter_applymask(DASdata,mask)
    DASdata.history.pop()
    DASdata.history.append(
        'applied velocity filter from {} to {}, with smoothing kernel: {}'
        .format(vmin,vmax,filter_std))
    return DASdata

def fk_velocity_analysis(DASdata,vmin, vmax
            ,grid_search_num=20
            ,find_peak_dense_sample=10
            ,gaussian_smooth=[1,1]
            ,direction=None
            ,print_process=False
            ):
    if not hasattr(DASdata,'fftdata'):
        DASdata = Data2D_fft2(DASdata)
    fkamp = np.abs(DASdata.fftdata)
    fkamp = gaussian_filter(fkamp,gaussian_smooth)

    coors = (DASdata.kaxis,DASdata.faxis)
    if direction=='up':
        vmin,vmax = np.sort(np.abs([vmin,vmax]))
    if direction=='down':
        vmin,vmax = np.sort(-np.abs([vmin,vmax]))

    def fun(points):
        val = interpn(coors,fkamp,points
            ,bounds_error=False,fill_value=np.nan,method='splinef2d')
        return val
    
    v_array = np.linspace(vmin,vmax,grid_search_num)
    amp = []
    for vel in v_array:
        if print_process:
            gjsignal.print_progress(vel)
        if len(DASdata.faxis)>len(DASdata.kaxis):
            f = DASdata.faxis
            k = f/vel
        else:
            k = DASdata.kaxis
            f = k*vel
        amp.append(np.nanmean(fun(list(zip(k,f)))))
    amp=np.array(amp)

    dense_v_array = np.linspace(vmin,vmax,grid_search_num*find_peak_dense_sample)
    dense_amp = interp1d(v_array,amp,kind='cubic')(dense_v_array)

    ind = np.argmax(dense_amp)
    best_vel = dense_v_array[ind]

    results = {
        'best_vel':best_vel,
        'v_array':v_array,
        'stack_amp':amp,
        'dense_v_array':dense_v_array,
        'dense_stack_amp':dense_amp,
        'sm_fk_amp':fkamp,
        'origin_fk_amp':np.abs(DASdata.fftdata),
        'faxis':DASdata.faxis,
        'kaxis':DASdata.kaxis
    }
    return results

def fk_velocity_analysis_viz(results,kind = 'velocity',
                    clim=None, colorbar=True,
                    auto_scale=True
                    ):
    best_vel = results['best_vel']
    if kind == 'velocity':
        print(f'Best Velocity: {best_vel}')
        plt.plot(results['v_array'],results['stack_amp'],'o')
        amp = np.max(results['dense_stack_amp'])
        plt.plot(best_vel,amp,'rx')
        if auto_scale:
            max_val = np.max(results['stack_amp'])
            plt.ylim(max_val*0.7,max_val*1.1)
    if kind == 'fk_amplitude':
        faxis = results['faxis']
        kaxis = results['kaxis']
        extent = [faxis[0],faxis[-1],kaxis[-1],kaxis[0]]
        ax = plt.subplot(1,2,1)
        plt.imshow(results['origin_fk_amp'],aspect='auto',extent=extent)
        plt.plot(kaxis*best_vel,kaxis,'r')
        if clim is not None:
            plt.clim(clim)
        if colorbar:
            plt.colorbar()
        plt.subplot(1,2,2,sharex=ax,sharey=ax)
        plt.imshow(results['sm_fk_amp'],aspect='auto',extent=extent)
        plt.plot(kaxis*best_vel,kaxis,'r')
        if clim is not None:
            plt.clim(clim)
        if colorbar:
            plt.colorbar()



def dispersion_analysis(DASdata,v_array,f_array,upward=True):
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


def spectrum_analysis(DASdata,depth_range=None):
    if depth_range is None:
        ind = DASdata.daxis>-np.inf
    else:
        ind = (DASdata.daxis>=depth_range[0])&(DASdata.daxis<=depth_range[1])
    data = DASdata.data[ind,:]
    dt = np.median(np.diff(DASdata.taxis))
    f,amp = gjsignal.amp_spectrum(data[0,:],dt)
    for ichan in range(1,data.shape[0]):
        f,tmp = gjsignal.amp_spectrum(data[ichan,:],dt)
        amp += tmp
    amp /= data.shape[0]
    return f, amp

def spectrum_transform_2D(DASdata):
    dt = np.median(np.diff(DASdata.taxis))
    f,amp = gjsignal.amp_spectrum(DASdata.data[0,:],dt)
    data = np.zeros((DASdata.data.shape[0],len(amp)))
    for i in range(data.shape[0]):
        f,amp = gjsignal.amp_spectrum(DASdata.data[i,:],dt)
        data[i,:] = amp
    
    spe_data = Data2D_XT.Data2D()
    spe_data.taxis = f
    spe_data.data = data
    spe_data.daxis = DASdata.daxis
    return spe_data
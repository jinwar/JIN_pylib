import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.display import display
import matplotlib.dates as mdates
from dateutil.parser import parse
import matplotlib.ticker as ticker

class CoPlot_Simple:
    def __init__(self,fig,Ddata,Pdata,
            c_center=0,c_range=1):
        self.fig = fig
        self.Pdata = Pdata
        self.Ddata = Ddata
        self.c_center = c_center
        self.c_range = c_range

    def draw(self,waterfall_plot_args = {}, curve_plot_args={}):
        self.fig.clf()
        cx = np.array([-1,1])

        axs = [None,None]
        axs[0] = plt.subplot2grid((5,10),(0,0),rowspan=3,colspan=9)
        self.Ddata.plot_waterfall(use_timestamp=True,**waterfall_plot_args)
        plt.clim(cx*self.c_range+self.c_center)
        plt.setp(axs[0].get_xticklabels(), visible=False)
        cbaxes = self.fig.add_axes([0.85, 0.45, 0.02, 0.4])
        plt.colorbar(cax=cbaxes)
        axs[1] = plt.subplot2grid((5,10),(3,0),rowspan=1,sharex = axs[0],colspan=9)
        self.Pdata.plot_multi_cols(use_timestamp=True,**curve_plot_args)
        self.axs = axs

class CoPlot_Waterfall_Pumping:

    def __init__(self,fig,Ddata,Pdata,
            scale=10,c_center=0,c_range=1):
        self.fig = fig
        self.Pdata = Pdata
        self.Ddata = Ddata
        self.default_c_center = c_center
        self.default_c_range = c_range
        self.define_widgets()


    def define_widgets(self):

        c_center = widgets.FloatText(value=self.default_c_center,description='Color Center')
        c_range = widgets.FloatText(value=self.default_c_range,description='Color Range')
        update_button = widgets.Button(description="Update Fig")
        reset_button = widgets.Button(description="Reset Fig")
            
        update_button.on_click(self.update_fig)
        reset_button.on_click(self.reset_fig)

        self.c_center = c_center
        self.c_range = c_range
        self.update_button = update_button
        self.reset_button = reset_button

        display(widgets.HBox([self.c_center,self.c_range
            ,self.update_button]))

    
    def reset_fig(self,b):
        self.draw()

    def update_fig(self,b):
        plt.sca(self.axs[0])
        cx = np.array([-1,1])
        plt.clim(cx*self.c_range.value+self.c_center.value)

    
    def draw(self,waterfall_plot_args = {}, curve_plot_args={}):
        self.fig.clf()
        cx = np.array([-1,1])

        axs = [None,None]
        axs[0] = plt.subplot2grid((5,10),(0,0),rowspan=3,colspan=9)
        self.Ddata.plot_waterfall(use_timestamp=True,**waterfall_plot_args)
        plt.clim(cx*self.c_range.value+self.c_center.value)
        plt.setp(axs[0].get_xticklabels(), visible=False)
        cbaxes = self.fig.add_axes([0.85, 0.45, 0.02, 0.4])
        plt.colorbar(cax=cbaxes)
        axs[1] = plt.subplot2grid((5,10),(3,0),rowspan=1,sharex = axs[0],colspan=9)
        self.Pdata.plot_multi_cols(use_timestamp=True,**curve_plot_args)
        self.axs = axs


class Interactive_Waterfall:

    def __init__(self,fig,Ddata):
        self.fig = fig
        self.Ddata = Ddata
        self.define_widgets()


    def define_widgets(self):
        c_center = widgets.FloatText(value=0.0,description='Color Center')
        c_range = widgets.FloatText(value=1.0,description='Color Range')
        update_button = widgets.Button(description="Update Fig")
        reset_button = widgets.Button(description="Reset Fig")
            
        update_button.on_click(self.update_fig)
        reset_button.on_click(self.reset_fig)

        self.c_center = c_center
        self.c_range = c_range
        self.update_button = update_button
        self.reset_button = reset_button

        display(widgets.HBox([self.c_center,self.c_range
            ,self.update_button]))

    
    def reset_fig(self,b):
        self.draw()

    def update_fig(self,b):
        plt.sca(self.axs[0])
        cx = np.array([-1,1])
        plt.clim(cx*self.c_range.value+self.c_center.value)
    
    def draw(self):
        self.fig.clf()
        cx = np.array([-1,1])

        axs = [None,None]
        axs[0] = plt.subplot2grid((5,1),(0,0),rowspan=4)	
        self.Ddata.plot_waterfall(use_timestamp=True)
        plt.clim(cx*self.c_range.value+self.c_center.value)
        self.axs = axs


def get_timeaxis_plot(row,timefmt='%m/%d %H:%M:%S',figsize=(8,6)):
    fig,axs = plt.subplots(row,1,sharex=True,figsize=figsize)
    date_format = mdates.DateFormatter(timefmt)
    for ax in axs:
        date_format = mdates.DateFormatter(timefmt)
        ax.xaxis.set_major_formatter(date_format)
    for ax in axs[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)
    fig.autofmt_xdate()
    return fig,axs

def xlim_timestr(str1,str2):
    bgt = parse(str1)
    edt = parse(str2)
    plt.xlim([bgt,edt])

def multi_plot_legend(lns,loc='best'):
    labs = [l.get_label() for l in lns]
    plt.gca().legend(lns,labs,loc=loc)


class PrecisionDateFormatter(ticker.Formatter):
    """
    Extend the `matplotlib.ticker.Formatter` class to allow for millisecond
    precision when formatting a tick (in days since the epoch) with a
    `~datetime.datetime.strftime` format string.

    """

    def __init__(self, fmt, precision=3):
        """
        Parameters
        ----------
        fmt : str
            `~datetime.datetime.strftime` format string.
        Usage: ax.xaxis.set_major_formatter(PrecisionDateFormatter('%H:%M:%S.{ms}'))
        """
        from matplotlib.dates import num2date
        self.num2date = num2date
        self.fmt = fmt
        self.precision = precision

    def __call__(self, x, pos=0):
        if x == 0:
            raise ValueError("DateFormatter found a value of x=0, which is "
                             "an illegal date; this usually occurs because "
                             "you have not informed the axis that it is "
                             "plotting dates, e.g., with ax.xaxis_date()")

        dt = self.num2date(x)
        ms = dt.strftime("%f")[:self.precision]

        return dt.strftime(self.fmt).format(ms=ms)



def plot_interactive(x,y):
    import plotly.express as px
    df = pd.DataFrame()
    df['x'] = x
    df['y'] = y
    fig = px.line(data_frame=df,x='x',y='y')
    fig.show()


class TDSlice:

    def __init__(self, fig, data):
        self.data = data
        self.pick_t = np.median(data.taxis)
        self.pick_d = np.median(data.daxis)
        self.fig = fig
        self.ax1 = plt.subplot2grid((6, 6), (0, 0), rowspan=4, colspan=4)
        self.ax2 = plt.subplot2grid((6, 6), (4, 0), rowspan=2, colspan=4, sharex=self.ax1)
        self.ax3 = plt.subplot2grid((6, 6), (0, 4), rowspan=4, colspan=2, sharey=self.ax1)
        self.trc_lim = np.array([-1,1])*4
        self.hline = None
        self.vline = None
        self.xlim = None
        self.ylim = None
        self.ori_xlim = None
        self.ori_ylim = None
        self.pending_zoom_x = False
        self.pending_zoom_y = False

        # Connect 's' key press event to the update function
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    def draw(self, **kwargs):
        plt.sca(self.ax1)
        self.data.plot_waterfall(**kwargs)
        self.update_trace_plots()
        if self.xlim is None:
            self.xlim = self.ax1.axis()[:2]
            self.ylim = self.ax1.axis()[2:]
            self.ori_xlim = self.xlim
            self.ori_ylim = self.ylim

    def update_trace_plots(self):
        if self.vline:
            self.vline.remove()
        if self.hline:
            self.hline.remove()

        md, trc = self.data.get_value_by_depth(self.pick_d)
        self.ax2.clear()
        self.ax2.plot(self.data.taxis, trc)
        self.vline = self.ax1.axvline(self.pick_t, color='k', linestyle='--')
        self.ax2.set_ylim(self.trc_lim)

        t, trc = self.data.get_value_by_time(self.pick_t)
        self.ax3.clear()
        self.ax3.plot(trc, self.data.daxis)
        self.ax3.set_xlim(self.trc_lim)
        self.hline = self.ax1.axhline(self.pick_d, color='k', linestyle='--')

        self.fig.canvas.draw()
    
    def update_zoom(self):
        self.ax1.set_xlim(self.xlim)
        self.ax1.set_ylim(self.ylim)
        self.fig.canvas.draw()
    
    def _clear_pending(self):
        self.pending_zoom_x = False
        self.pending_zoom_y = False

    def on_key_press(self, event):
        if event.key == 'a':
            if event.inaxes == self.ax1:
                if event.xdata is not None and event.ydata is not None:
                    self.pick_t = event.xdata
                    self.pick_d = event.ydata
                    self.update_trace_plots()
            if event.inaxes == self.ax2:
                if event.xdata is not None and event.ydata is not None:
                    self.pick_t = event.xdata
                    self.update_trace_plots()
            if event.inaxes == self.ax3:
                if event.xdata is not None and event.ydata is not None:
                    self.pick_d = event.ydata
                    self.update_trace_plots()

        if event.key == '=':
            self.trc_lim = self.trc_lim*1.2
            self.update_trace_plots()

        if event.key == '-':
            self.trc_lim = self.trc_lim/1.2
            self.update_trace_plots()

        if event.key == 'o':
            self.xlim = self.ori_xlim
            self.ylim = self.ori_ylim
            self.update_zoom()
        
        if event.key == 'y':
            if (event.inaxes == self.ax1) or (event.inaxes == self.ax3):
                if self.pending_zoom_y:
                    y1 = self.pending_value
                    y2 = event.ydata
                    y1,y2 = np.sort([y1,y2])
                    self.ylim = [y2,y1]
                    self.update_zoom()

                else:
                    self.pending_zoom_y = True
                    self.pending_value = event.ydata
                    return
        
        self._clear_pending()
# Usage example:
# tdslice = TDSlice(plt.figure(), your_data_object)
# tdslice.draw()
# plt.show()

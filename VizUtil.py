import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.display import display
import matplotlib.dates as mdates
from dateutil.parser import parse

class CoPlot_Waterfall_Pumping:

    def __init__(self,fig,Ddata,Pdata):
        self.fig = fig
        self.Pdata = Pdata
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
        axs[0] = plt.subplot2grid((5,10),(0,0),rowspan=3,colspan=9)
        self.Ddata.plot_waterfall(use_timestamp=True)
        plt.clim(cx*self.c_range.value+self.c_center.value)
        plt.setp(axs[0].get_xticklabels(), visible=False)
        cbaxes = self.fig.add_axes([0.85, 0.45, 0.02, 0.4])
        plt.colorbar(cax=cbaxes)
        axs[1] = plt.subplot2grid((5,10),(3,0),rowspan=1,sharex = axs[0],colspan=9)
        self.Pdata.plot_multi_cols(use_timestamp=True)
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


def get_timeaxis_plot(row,timefmt='%m/%d %H:%M:%S'):
    fig,axs = plt.subplots(row,1,sharex=True)
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
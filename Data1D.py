from . import gjsignal
import numpy as np
import matplotlib.pyplot as plt
from .BasicClass import BasicClass
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from .VizUtil import PrecisionDateFormatter
from dateutil.parser import parse

class PumpCurve(BasicClass):

    def __init__(self,df=None):
        self.version = '1.0'
        self.taxis = []
        self.df = df
        self.plot_cols = []
        self.ylims = None
    
    def set_df(self,df):
        self.df = df
    
    def set_ylims(self,ylims):
        self.ylims = ylims
    
    def list_columns(self,is_print=True):
        if is_print:
            print(self.df.columns)
        return self.df.columns
    
    def set_plot_columns(self,cols):
        self.plot_cols = cols
    
    def clean_plot_columns(self):
        self.df[self.plot_cols] = self.df[self.plot_cols].applymap(replace_non_float_with_nan)
    
    def set_time_from_datetime(self,timestamps):
        self.start_time = timestamps[0]
        self.timestamps = timestamps
        self.taxis = np.array([(t-timestamps[0]).total_seconds()
             for t in timestamps])
    
    def cal_timestamp_from_taxis(self):
        timestamps = [self.start_time + timedelta(seconds=t) 
            for t in self.taxis]
        self.timestamps = timestamps

    def set_time_from_timestamp(self,timestamps):
        self.start_time = timestamps[0].to_pydatetime()
        self.taxis = np.array([(t-timestamps[0]).total_seconds()
             for t in timestamps])
        self.timestamps = timestamps
    
    def plot_multi_cols(self,timescale='second',use_timestamp=False,legend_loc='best',
                xaxis_rotation=0 ,xtickN = 4 ,timefmt = '%m/%d\n%H:%M:%S.{ms}' ,
                timefmt_ms_precision = 1):
        ts = 1
        if timescale == 'hour':
            ts = 3600
        if timescale == 'day':
            ts = 3600/24
        axs = []
        cols = self.plot_cols
        axs.append(plt.gca())
        axs.append(axs[0].twinx())
        axs.append(axs[0].twinx())
        axs[2].spines["right"].set_position(("axes", 1.06))
        axs[2].spines["right"].set_visible(True)
        lines = [None,None,None]
        for i,c in zip(range(3),['b','r','g']):
            if use_timestamp:
                lines[i], = axs[i].plot(self.timestamps,self.df[cols[i]],c,label=cols[i])
            else:
                lines[i], = axs[i].plot(self.taxis/ts,self.df[cols[i]],c,label=cols[i])
        for i in range(3):
            axs[i].tick_params(axis='y', colors=lines[i].get_color())
        for i in range(3):
            # ylim = axs[i].axis()[2:4]
            if self.ylims is None:
                values = self.df[cols[i]].values
                top = np.nanpercentile(values, 95)
                edge = top*0.2
                ylim = [0-edge,top+edge]
            else:
                if isinstance(self.ylims[i],(list,np.ndarray)):
                    ylim = self.ylims[i]
                else:
                    ylim = [0,self.ylims[i]]
            axs[i].set_ylim(ylim[0],ylim[1])
        if use_timestamp:
            plt.gca().xaxis_date()
            date_format = PrecisionDateFormatter(timefmt
                ,precision=timefmt_ms_precision)
            plt.gca().xaxis.set_major_locator(plt.MaxNLocator(xtickN))
            plt.xticks(rotation=xaxis_rotation)
            plt.gca().xaxis.set_major_formatter(date_format)
        axs[0].legend(lines, [l.get_label() for l in lines],loc=legend_loc,fontsize=5)
        
    def plot_single_col(self,col,timescale='second',use_timestamp=False,is_shrink=False):
        ts = 1
        if timescale == 'hour':
            ts = 3600
        if timescale == 'day':
            ts = 3600/24
        if is_shrink:
            plt.subplot2grid((5,1),(0,0),rowspan=4)
        if use_timestamp:
            plt.plot(self.timestamps,self.df[col],label=col)
        else:
            plt.plot(self.taxis/ts,self.df[col],label=col)
        ax = plt.gca()
        if use_timestamp:
            ax.xaxis_date()
            date_format = mdates.DateFormatter('%m/%d %H:%M')
            ax.xaxis.set_major_formatter(date_format)
            ax.tick_params(axis='x',labelrotation=45)
        
# Define a function to replace non-float values with np.nan
def replace_non_float_with_nan(value):
    if isinstance(value, (int, float)):
        return value
    else:
        return np.nan
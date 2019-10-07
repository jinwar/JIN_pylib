import gjsignal
import numpy as np
import matplotlib.pyplot as plt
from BasicClass import BasicClass

class PumpCurve(BasicClass):

    def __init__(self):
        self.version = '1.0'
        self.time = []
        self.df = []
        self.plot_cols = []
    
    def set_df(self,df):
        self.df = df
    
    def set_plot_columns(self,cols):
        self.plot_cols = cols
    
    def set_time_from_datetime(self,timestamps):
        self.start_time = timestamps[0]
        self.taxis = np.array([(t-timestamps[0]).total_seconds()
             for t in timestamps])

    def set_time_from_timestamp(self,timestamps):
        self.start_time = timestamps[0].to_pydatetime()
        self.taxis = np.array([(t-timestamps[0]).total_seconds()
             for t in timestamps])
    
    def plot_multi_cols(self):
        axs = []
        cols = self.plot_cols
        axs.append(plt.gca())
        axs.append(axs[0].twinx())
        axs.append(axs[0].twinx())
        axs[2].spines["right"].set_position(("axes", 1.06))
        axs[2].spines["right"].set_visible(True)
        lines = [None,None,None]
        for i,c in zip(range(3),['b','r','g']):
            lines[i], = axs[i].plot(self.taxis/3600,self.df[cols[i]],c,label=cols[i])
        for i in range(3):
            axs[i].tick_params(axis='y', colors=lines[i].get_color())
        for i in range(3):
            ylim = axs[i].axis()[2:4]
            axs[i].set_ylim(ylim[0],ylim[1]*1.2)
        axs[0].legend(lines, [l.get_label() for l in lines],loc='best')
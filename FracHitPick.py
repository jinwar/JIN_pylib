from .BasicClass import BasicClass
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from .VizUtil import CoPlot_Waterfall_Pumping
import matplotlib.dates as mdates


class PickFrac_nodraw(BasicClass):

    def __init__(self,pickfile,fig):
        self.pickfile = pickfile
        self.fig = fig
        try:
            self.pickdf = pd.read_csv(pickfile)
        except:
            self.pickdf = pd.DataFrame(columns=['time','md'])
        self.lines = []
    
    def draw(self):
        plt.draw()
        self.cid = self.fig.canvas.mpl_connect('key_press_event',self.event_handle)
        self.update_lines()
    
    def update_lines(self):
        try:
            for l in self.lines:
                l.pop(0).remove()
        except:
            pass
        lines = []
        edtime = self.Ddata.get_extent(use_timestamp=True)[1]
        for i,row in self.pickdf.iterrows():
            bgtime = pd.to_datetime(row['time'])
            bgtime = mdates.date2num(bgtime)
            lines.append(self.viz.axs[0].plot([bgtime,edtime],[row['md'],row['md']],'k--'))
        self.lines = lines

    
    def event_handle(self,event):
        if event.key == 'a':
            self.pickdf.loc[len(self.pickdf),'time'] = mdates.num2date(event.xdata).strftime('%Y-%m-%d %H:%M:%S')
            self.pickdf.loc[len(self.pickdf)-1,'md'] = event.ydata
            self.update_lines()

        if event.key == 'd':
            mds = self.pickdf['md'].values.flatten()
            mindist = np.min(np.abs(mds-event.ydata))
            if mindist < 20:
                ind = np.argmin(np.abs(mds-event.ydata))
                self.pickdf = self.pickdf.drop(ind)
                self.pickdf.reset_index(drop=True,inplace=True)
            self.update_lines()


class PickFrac(BasicClass):

    def __init__(self,Ddata,Pdata,pickfile):
        self.Ddata = Ddata
        self.Pdata = Pdata
        self.pickfile = pickfile
        self.fig = plt.figure()
        try:
            self.pickdf = pd.read_csv(pickfile)
        except:
            self.pickdf = pd.DataFrame(columns=['time','md'])
        self.viz = CoPlot_Waterfall_Pumping(self.fig,self.Ddata,self.Pdata)
        self.lines = []
    
    def draw(self):
        self.viz.draw()
        self.cid = self.fig.canvas.mpl_connect('key_press_event',self.event_handle)
        self.update_lines()
    
    def update_lines(self):
        try:
            for l in self.lines:
                l.pop(0).remove()
        except:
            pass
        lines = []
        edtime = self.Ddata.get_extent(use_timestamp=True)[1]
        for i,row in self.pickdf.iterrows():
            bgtime = pd.to_datetime(row['time'])
            bgtime = mdates.date2num(bgtime)
            lines.append(self.viz.axs[0].plot([bgtime,edtime],[row['md'],row['md']],'k--'))
        self.lines = lines

    
    def event_handle(self,event):
        if event.key == 'a':
            self.pickdf.loc[len(self.pickdf),'time'] = mdates.num2date(event.xdata).strftime('%Y-%m-%d %H:%M:%S')
            self.pickdf.loc[len(self.pickdf)-1,'md'] = event.ydata
            self.update_lines()

        if event.key == 'd':
            mds = self.pickdf['md'].values.flatten()
            mindist = np.min(np.abs(mds-event.ydata))
            if mindist < 20:
                ind = np.argmin(np.abs(mds-event.ydata))
                self.pickdf = self.pickdf.drop(ind)
                self.pickdf.reset_index(drop=True,inplace=True)
            self.update_lines()

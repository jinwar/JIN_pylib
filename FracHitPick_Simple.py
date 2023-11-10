from tkinter import W
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.dates as mdates

class PickFrac():

    def __init__(self,pickfile,fig):
        self.pickfile = pickfile
        self.fig = fig
        try:
            self.pickdf = pd.read_csv(pickfile)
        except:
            self.pickdf = pd.DataFrame(columns=['time','md'])
        self.lines = []
        self.cx = np.array([-1,1])
        self.plot_current_only = False
    
    def set_init_c_range(self, crange):
        self.cx = np.array([-1,1])*crange
    
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
        edtime = plt.gca().axis()[1]
        axis_range = plt.gca().axis()
        for i,row in self.pickdf.iterrows():
            bgtime = pd.to_datetime(row['time'])
            bgtime = mdates.date2num(bgtime)
            if self.plot_current_only:
                if bgtime < axis_range[0]:
                    continue
            lines.append(plt.gca().plot([bgtime,edtime],[row['md'],row['md']],'k--'))
               
        self.lines = lines
        plt.axis(axis_range)
        plt.draw()

    
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

        if event.key == '=':
            self.cx = self.cx/1.2
            plt.clim(self.cx)
            plt.draw()

        if event.key == '-':
            self.cx = self.cx*1.2
            plt.clim(self.cx)
            plt.draw()
        
        if event.key == 'w':
            self.save()
            print('Pick file saved')

        if event.key == 'c':
            self.plot_current_only = not self.plot_current_only
            self.update_lines()
            
    def save(self):
        self.pickdf.to_csv(self.pickfile,index=False)